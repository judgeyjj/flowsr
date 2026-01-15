import json
from types import SimpleNamespace
import torch
import argparse
import numpy as np
import yaml
import os
import torchaudio 
from torchinfo import summary
from data import AudioDataset

from cfm_superresolution import (
    FLowHigh,
    MelVoco,
    ConditionalFlowMatcherWrapper
)

from trainer import FLowHighTrainer

def _is_rank0() -> bool:
    # works for torchrun / accelerate
    try:
        return int(os.environ.get("RANK", "0")) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0
    except Exception:
        return True

def r0_print(*args, **kwargs):
    if _is_rank0():
        print(*args, **kwargs)

def _to_namespace(d):
    return json.loads(json.dumps(d), object_hook=lambda x: SimpleNamespace(**x))

def _normalize_yaml_to_flowhigh_dict(cfg: dict) -> dict:
    """
    将 sr 风格的 YAML 配置映射为 FlowHigh 原始代码所需的字段结构：
      - random_seed
      - data.{data_path, valid_path, ...}
      - model.{...}
      - train.{...}
    """
    if not isinstance(cfg, dict):
        raise TypeError("config must be a dict")

    # If it already looks like FlowHigh json schema, return as-is
    if 'random_seed' in cfg and 'data' in cfg and 'model' in cfg and 'train' in cfg:
        return cfg

    seed = cfg.get('seed', cfg.get('random_seed', 104))
    data = cfg.get('data', {}) or {}
    model = cfg.get('model', {}) or {}
    training = cfg.get('training', cfg.get('train', {})) or {}
    output = cfg.get('output', {}) or {}
    optimizer = cfg.get('optimizer', {}) or {}

    # Model / architecture knobs (FlowHigh + Mamba/Mamba2)
    architecture = str(model.get('architecture', 'mamba'))
    mamba_d_state = int(model.get('mamba_d_state', 16))
    mamba_d_conv = int(model.get('mamba_d_conv', 4))
    mamba_expand = int(model.get('mamba_expand', 2))
    mamba2_headdim = int(model.get('mamba2_headdim', 64))

    # Recommended defaults for Mamba2 (do not change Mamba1 defaults)
    if architecture == 'mamba2' and mamba_d_state == 16:
        mamba_d_state = 128

    # sr/config.yaml uses output.exp_dir + output.exp_name; FlowHigh expects train.save_dir
    exp_dir = output.get('exp_dir', None)
    exp_name = output.get('exp_name', None)
    save_dir = output.get('save_dir', None)
    if (not save_dir) and exp_dir and exp_name:
        save_dir = os.path.join(str(exp_dir), str(exp_name))

    # Prevent checkpoint overwrite between different architectures/configs
    if architecture == 'mamba2':
        suffix = f"_mamba2_d{int(mamba_d_state)}_h{int(mamba2_headdim)}"
        if save_dir:
            save_dir_str = str(save_dir)
            # avoid duplicating suffix if user already added it
            if ("mamba2" not in os.path.basename(save_dir_str).lower()) and (not save_dir_str.endswith(suffix)):
                save_dir = save_dir_str + suffix
        elif exp_dir and exp_name:
            save_dir = os.path.join(str(exp_dir), str(exp_name) + suffix)

    # sr/config.yaml uses data.target_sr + data.source_sr_list
    target_sr = int(data.get('target_sr', model.get('target_sr', 48000)))
    source_sr_list = data.get('source_sr_list', None)
    downsample_min = int(data.get('downsample_min', min(source_sr_list) if isinstance(source_sr_list, list) and len(source_sr_list) else 4000))
    downsample_max = int(data.get('downsample_max', max(source_sr_list) if isinstance(source_sr_list, list) and len(source_sr_list) else 32000))

    # Mel params are not present in sr/config.yaml; use sensible defaults
    n_mel_channels = int(data.get('n_mel_channels', 256))
    mel_fmin = float(data.get('mel_fmin', 20))
    mel_fmax = float(data.get('mel_fmax', target_sr / 2))

    out = {
        'random_seed': seed,
        'data': {
            'data_path': data.get('train_dir', data.get('data_path', '')),
            'valid_path': data.get('val_dir', data.get('valid_path', '')),
            'valid_prepare': bool(data.get('valid_prepare', True)),
            'samplingrate': int(data.get('samplingrate', target_sr)),
            'max_wav_value': float(data.get('max_wav_value', 32767.0)),
            'n_fft': int(model.get('n_fft', data.get('n_fft', 2048))),
            'hop_length': int(model.get('hop_length', data.get('hop_length', 480))),
            'win_length': int(data.get('win_length', model.get('n_fft', data.get('n_fft', 2048)))),
            'n_mel_channels': n_mel_channels,
            'mel_fmin': mel_fmin,
            'mel_fmax': mel_fmax,
            'downsample_min': downsample_min,
            'downsample_max': downsample_max,
            'downsampling_method': str(data.get('downsampling_method', 'scipy')),
        },
        'model': {
            # sr/config.yaml model.* 与 FlowHigh 不同；这里全部给默认值（你只维护 sr 的 config.yaml）
            'modelname': str(model.get('modelname', 'FLowHigh-DiM')),
            'architecture': architecture,
            'dim': int(model.get('dim', 1024)),
            'n_layers': int(model.get('n_layers', 2)),
            'n_heads': int(model.get('n_heads', 16)),
            'dim_head': int(model.get('dim_head', 64)),
            'cfm_path': str(model.get('cfm_path', 'independent_cfm_adaptive')),
            'sigma': float(model.get('sigma', 1e-4)),
            'vocoder': str(model.get('vocoder', 'bigvgan')),
            # 默认用仓库内 BigVGAN
            'vocoderpath': str(model.get('vocoderpath', 'vocoder/BIGVGAN/checkpoint/g_48_00850000')),
            'vocoderconfigpath': str(model.get('vocoderconfigpath', 'vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json')),
            # DiM(Mamba) optional hyperparams
            'mamba_d_state': int(mamba_d_state),
            'mamba_d_conv': int(mamba_d_conv),
            'mamba_expand': int(mamba_expand),
            # Mamba2 optional hyperparams
            'mamba2_headdim': int(mamba2_headdim),
        },
        'train': {
            'random_split_seed': int(training.get('random_split_seed', 53)),
            # sr/config.yaml: batch_size 在 data.batch_size，lr 在 optimizer.lr
            'batchsize': int(data.get('batch_size', training.get('batch_size', training.get('batchsize', 128)))),
            'lr': float(optimizer.get('lr', training.get('lr', 3e-4))),
            'initial_lr': float(training.get('initial_lr', optimizer.get('lr', 1e-5))),
            # sr/config.yaml: 用 num_epochs 驱动；FlowHigh trainer 支持 num_epochs
            'num_epochs': int(cfg.get('num_epochs', training.get('num_epochs', 0)) or 0),
            'n_train_steps': int(training.get('num_train_steps', training.get('n_train_steps', 0)) or 0),
            'n_warmup_steps': int(training.get('num_warmup_steps', training.get('n_warmup_steps', 0)) or 0),
            'log_every': int(training.get('log_every', 10000)),
            'save_results_every': int(training.get('save_results_every', 10000)),
            'save_model_every': int(training.get('save_model_every', 100000)),
            'save_dir': str(save_dir or cfg.get('checkpoint_dir', 'results')),
            'weighted_loss': bool(training.get('weighted_loss', False)),
            # logging / ddp knobs
            'use_swanlab': bool(output.get('use_swanlab', False)),
            'swanlab_project': str(output.get('swanlab_project', 'flowsr')),
            'swanlab_log_interval_steps': int(output.get('swanlab_log_interval_steps', 0) or 0),
            'ddp_find_unused_parameters': bool(training.get('ddp_find_unused_parameters', False)),
        }
    }
    return out

def load_config(config_path: str):
    if config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg = _normalize_yaml_to_flowhigh_dict(cfg)
        return _to_namespace(cfg)

    with open(config_path, 'r') as file:
        config_dict = json.load(file)
    return _to_namespace(config_dict)


if __name__ == "__main__":
    # Avoid CPU oversubscription / deadlocks when DataLoader workers use scipy/librosa
    # (torchrun already sets OMP_NUM_THREADS=1 by default; also cap BLAS-related threads)
    for _k in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_k, "1")
    
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file (.yaml/.yml/.json)')
    args = parser.parse_args()
    
    hparams = load_config(args.config)

    # Resolve relative paths against repo root (flowsr/)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if hasattr(hparams, 'model'):
        if hasattr(hparams.model, 'vocoderpath') and isinstance(hparams.model.vocoderpath, str) and hparams.model.vocoderpath and not os.path.isabs(hparams.model.vocoderpath):
            hparams.model.vocoderpath = os.path.join(repo_root, hparams.model.vocoderpath)
        if hasattr(hparams.model, 'vocoderconfigpath') and isinstance(hparams.model.vocoderconfigpath, str) and hparams.model.vocoderconfigpath and not os.path.isabs(hparams.model.vocoderconfigpath):
            hparams.model.vocoderconfigpath = os.path.join(repo_root, hparams.model.vocoderconfigpath)
    if hasattr(hparams, 'data'):
        if hasattr(hparams.data, 'data_path') and isinstance(hparams.data.data_path, str) and hparams.data.data_path and not os.path.isabs(hparams.data.data_path):
            hparams.data.data_path = os.path.join(repo_root, hparams.data.data_path)
        if hasattr(hparams.data, 'valid_path') and isinstance(hparams.data.valid_path, str) and hparams.data.valid_path and not os.path.isabs(hparams.data.valid_path):
            hparams.data.valid_path = os.path.join(repo_root, hparams.data.valid_path)

    torch.manual_seed(hparams.random_seed)
    np.random.seed(hparams.random_seed)
    
    r0_print('Num of current cuda devices:', n_gpus)
    r0_print('Initializing logger...')
    
    
    r0_print('Initializing data loaders...')
    dataset= AudioDataset(folder=hparams.data.data_path, downsampling = hparams.data.downsampling_method)
    validset = AudioDataset(folder=hparams.data.valid_path, downsampling = hparams.data.downsampling_method, mode='valid')

    sampling_rates = list(range(hparams.data.downsample_min, hparams.data.downsample_max + 1000, 1000))
    
    r0_print(f'Initializing Mel vocoder...')
    audio_enc_dec_type = MelVoco(n_mels= hparams.data.n_mel_channels, 
                                 sampling_rate= hparams.data.samplingrate, 
                                 f_max= hparams.data.mel_fmax, 
                                 n_fft= hparams.data.n_fft, 
                                 win_length= hparams.data.win_length, 
                                 hop_length= hparams.data.hop_length,
                                 vocoder= hparams.model.vocoder, 
                                 vocoder_config= hparams.model.vocoderconfigpath,
                                 vocoder_path = hparams.model.vocoderpath,
                                 # DDP startup optimization: training does not need vocoder decode
                                 load_vocoder_on_init=False
    )        
    # audio_enc_dec_type = LinearVoco()
    # audio_enc_dec_type = SpecVoco()
        
    r0_print('Initializing FLowHigh...')
    
    # Print Mamba2 import status if using mamba2 architecture
    if getattr(hparams.model, 'architecture', None) == 'mamba2':
        try:
            import modules as _modules
            r0_print(f"[mamba2] Import status: MAMBA2_AVAILABLE={getattr(_modules, 'MAMBA2_AVAILABLE', False)}")
        except Exception as e:
            r0_print(f"[mamba2] Import status check failed: {e}")

    model = FLowHigh(
                    architecture=hparams.model.architecture,
                    dim_in= hparams.data.n_mel_channels, # Same with Mel-bins 
                    audio_enc_dec= audio_enc_dec_type,
                    dim= hparams.model.dim,
                    depth= hparams.model.n_layers, 
                    dim_head= hparams.model.dim_head, 
                    heads= hparams.model.n_heads,
                    mamba_d_state=getattr(hparams.model, 'mamba_d_state', 16),
                    mamba_d_conv=getattr(hparams.model, 'mamba_d_conv', 4),
                    mamba_expand=getattr(hparams.model, 'mamba_expand', 2),
                    mamba2_headdim=getattr(hparams.model, 'mamba2_headdim', 64),
                    )
    
    r0_print('Initializing CFM Wrapper...')
    cfm_wrapper = ConditionalFlowMatcherWrapper(flowhigh= model,
                                                cfm_method= hparams.model.cfm_path,
                                                sigma= hparams.model.sigma)
    
    if _is_rank0():
        summary(cfm_wrapper)
        # Unified parameter reporting in MB
        total_bytes = sum(p.numel() * p.element_size() for p in cfm_wrapper.parameters())
        trainable_bytes = sum(p.numel() * p.element_size() for p in cfm_wrapper.parameters() if p.requires_grad)
        r0_print(f"Params size (trainable/total): {trainable_bytes / (1024 ** 2):.2f} MB / {total_bytes / (1024 ** 2):.2f} MB")

    r0_print('Initializing FLowHigh Trainer...')
    # Prefer epoch-based training if provided by sr/config.yaml; otherwise fall back to step-based
    num_epochs = getattr(hparams.train, 'num_epochs', 0) if hasattr(hparams, 'train') else 0
    num_epochs = int(num_epochs) if num_epochs else None
    num_train_steps = int(hparams.train.n_train_steps) if hasattr(hparams, 'train') and int(getattr(hparams.train, 'n_train_steps', 0) or 0) > 0 else None

    trainer = FLowHighTrainer(cfm_wrapper= cfm_wrapper,
                              batch_size= hparams.train.batchsize,
                              dataset= dataset,
                              validset= validset,
                              num_train_steps= num_train_steps,
                              num_warmup_steps= hparams.train.n_warmup_steps,
                              num_epochs=num_epochs,
                              lr= hparams.train.lr,
                              initial_lr= hparams.train.initial_lr,
                              log_every = hparams.train.log_every, 
                              save_results_every = hparams.train.save_results_every, 
                              save_model_every = hparams.train.save_model_every,
                              results_folder= hparams.train.save_dir,
                              random_split_seed = hparams.train.random_split_seed,
                              original_sampling_rate = hparams.data.samplingrate, 
                              downsampling= hparams.data.downsampling_method,
                              valid_prepare = hparams.data.valid_prepare,
                              sampling_rates = sampling_rates,
                              cfm_method = hparams.model.cfm_path,
                              weighted_loss = hparams.train.weighted_loss,
                              model_name = hparams.model.modelname,
                              use_swanlab=getattr(hparams.train, 'use_swanlab', False),
                              swanlab_project=getattr(hparams.train, 'swanlab_project', 'flowsr'),
                              swanlab_log_interval_steps=getattr(hparams.train, 'swanlab_log_interval_steps', 0),
                              ddp_find_unused_parameters=getattr(hparams.train, 'ddp_find_unused_parameters', False),
                              dataloader_num_workers=getattr(hparams.data, 'num_workers', 0),
                              dataloader_persistent_workers=False,
                              )
    
    r0_print('Start training...')
    trainer.train()
    

