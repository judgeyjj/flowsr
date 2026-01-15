import torch
import argparse
import numpy as np
from cfm_superresolution import (
    MelVoco,
    FLowHigh,
    ConditionalFlowMatcherWrapper
)
import os 
from glob import glob 
from scipy.io.wavfile import write
from postprocessing import PostProcessing
from tqdm import tqdm
import librosa
import scipy
from torchinfo import summary
from metrics import compute_all_metrics
import yaml
import time
from modules import MAMBA2_AVAILABLE


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def super_resolution(input_path, output_dir, config, cfm_wrapper, pp):
    """
    音频超分辨率推理函数（支持从高分辨率自动下采样测试）
    
    Args:
        input_path: 输入音频目录（高分辨率音频）
        output_dir: 输出音频目录
        config: 配置字典
        cfm_wrapper: CFM包装器模型
        pp: 后处理对象
    """
    # 从配置中获取参数
    inf_cfg = config['inference']
    target_sr = inf_cfg['target_sampling_rate']
    simulate_low_sr = inf_cfg.get('simulate_low_sr')
    upsampling_method = inf_cfg['up_sampling_method']
    cfm_method = inf_cfg['cfm_method']
    timestep = inf_cfg['time_step']
    gt_path = inf_cfg.get('gt_path')
    compute_metrics_flag = inf_cfg.get('compute_metrics', True)
    n_fft = inf_cfg['n_fft']
    hop_length = inf_cfg['hop_length']
    win_length = inf_cfg['win_length']
    
    extension = 'wav'
    audio_files = glob(os.path.join(input_path, f'*.{extension}'))
    
    print("="*70)
    print("推理配置:")
    print(f"  输入路径: {input_path}")
    print(f"  输出路径: {output_dir}")
    print(f"  目标采样率: {target_sr} Hz")
    print(f"  模拟低采样率: {simulate_low_sr} Hz" if simulate_low_sr else "  模拟低采样率: 未启用")
    print(f"  上采样方法: {upsampling_method}")
    print(f"  CFM方法: {cfm_method}")
    print(f"  ODE步数: {timestep}")
    print(f"  音频文件数: {len(audio_files)}")
    if gt_path and compute_metrics_flag:
        print(f"  Ground Truth路径: {gt_path}")
        print("  将计算客观指标")
    print("="*70)
    
    # 用于累积指标
    all_metrics = {'LSD': [], 'SI-SNR': [], 'PESQ': [], 'STOI': []}
    # 用于累积RTF
    all_rtf = []
    all_audio_durations = []
    all_inference_times = []
    
    with torch.no_grad():
        
        # 遍历所有音频文件进行超分辨率重建
        for id, wav_file in enumerate(tqdm(audio_files, desc="处理音频文件")):

            audio_file_name = os.path.basename(wav_file).replace('.wav','')
            save_dir = os.path.join(output_dir, f'{audio_file_name}.wav')
            
            # 记录推理开始时间
            start_time = time.time()
            
            # 加载高分辨率音频
            audio_hr, sr_original = librosa.load(wav_file, sr=None, mono=True)
            
            # 计算音频时长（秒）
            audio_duration = len(audio_hr) / sr_original
            
            # 如果启用了模拟低采样率，先下采样
            if simulate_low_sr is not None and simulate_low_sr > 0:
                # 下采样到低采样率（模拟低分辨率输入）
                if sr_original != simulate_low_sr:
                    # 使用scipy.resample_poly进行下采样（与FLowHigh论文一致）
                    audio_lr = scipy.signal.resample_poly(audio_hr, simulate_low_sr, sr_original)
                else:
                    audio_lr = audio_hr.copy()
                
                # 再上采样到目标采样率（这是模型的输入条件）
                if upsampling_method == 'scipy':
                    cond = scipy.signal.resample_poly(audio_lr, target_sr, simulate_low_sr)
                elif upsampling_method == 'librosa':
                    cond = librosa.resample(audio_lr, orig_sr=simulate_low_sr, target_sr=target_sr, res_type='soxr_hq')
                else:
                    raise ValueError(f"不支持的上采样方法: {upsampling_method}")
            else:
                # 不模拟低采样率，直接将输入重采样到目标采样率
                if sr_original != target_sr:
                    cond = librosa.resample(audio_hr, orig_sr=sr_original, target_sr=target_sr)
                else:
                    cond = audio_hr.copy()
            
            # 归一化（与FLowHigh一致，不加epsilon）
            cond /= np.max(np.abs(cond))
            
            # 转换为tensor（确保 float32 / contiguous / 正确 device）
            if isinstance(cond, np.ndarray):
                cond = torch.from_numpy(cond).float()
            if cond.ndim == 1:
                cond = cond.unsqueeze(0)
            cond = cond.contiguous()
            cond = cond.to(cfm_wrapper.device)

            # 使用CFM进行超分辨率重建（不要手工传 std_1/std_2，保持与训练/默认一致）
            HR_audio = cfm_wrapper.sample(cond=cond, time_steps=timestep, cfm_method=cfm_method)

            HR_audio = HR_audio.squeeze(1) # [1, T]

            # 后处理
            HR_audio_pp = pp.post_processing(HR_audio, cond, cond.size(-1)) # [1, T] 
            
            # 保存超分辨率音频
            HR_audio_pp_npy = (HR_audio_pp.cpu().squeeze().clamp(-1,1).numpy()*32767.0).astype(np.int16) 
            write(save_dir, target_sr, HR_audio_pp_npy)
            
            # 记录推理结束时间并计算RTF
            end_time = time.time()
            inference_time = end_time - start_time
            rtf = inference_time / audio_duration
            
            # 累积RTF统计
            all_rtf.append(rtf)
            all_audio_durations.append(audio_duration)
            all_inference_times.append(inference_time)
            
            # 打印RTF信息
            print(f"\n{audio_file_name} RTF:")
            print(f"  音频时长: {audio_duration:.2f}s")
            print(f"  推理时间: {inference_time:.2f}s")
            print(f"  RTF: {rtf:.4f}")
            
            # 计算客观指标（如果启用）
            if gt_path and compute_metrics_flag:
                # 查找对应的ground truth文件
                gt_file = os.path.join(gt_path, os.path.basename(wav_file))
                
                if os.path.exists(gt_file):
                    try:
                        # 加载ground truth音频（已经是高分辨率）
                        gt_audio, gt_sr = librosa.load(gt_file, sr=target_sr, mono=True)
                        gt_audio_tensor = torch.from_numpy(gt_audio).float()
                        
                        # 确保长度一致
                        pred_audio = HR_audio_pp.squeeze()
                        min_len = min(len(pred_audio), len(gt_audio_tensor))
                        pred_audio = pred_audio[:min_len]
                        gt_audio_tensor = gt_audio_tensor[:min_len]
                        
                        # 计算所有指标
                        metrics = compute_all_metrics(
                            pred_audio,
                            gt_audio_tensor,
                            sr=target_sr,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            win_length=win_length
                        )
                        
                        # 累积指标
                        for key in all_metrics.keys():
                            if key in metrics and metrics[key] != 0.0:
                                all_metrics[key].append(metrics[key])
                        
                        # 打印当前音频的指标
                        print(f"  客观指标:")
                        for key, value in metrics.items():
                            print(f"    {key}: {value:.4f}")
                            
                    except Exception as e:
                        print(f"警告: 无法计算 {audio_file_name} 的指标: {e}")
                else:
                    print(f"警告: 未找到 {audio_file_name} 的ground truth文件")
    
    # 打印平均RTF统计
    if len(all_rtf) > 0:
        print("\n" + "="*70)
        print("RTF统计:")
        print("="*70)
        avg_rtf = np.mean(all_rtf)
        std_rtf = np.std(all_rtf)
        min_rtf = np.min(all_rtf)
        max_rtf = np.max(all_rtf)
        total_audio_duration = np.sum(all_audio_durations)
        total_inference_time = np.sum(all_inference_times)
        
        print(f"平均RTF: {avg_rtf:.4f} ± {std_rtf:.4f}")
        print(f"最小RTF: {min_rtf:.4f}")
        print(f"最大RTF: {max_rtf:.4f}")
        print(f"总音频时长: {total_audio_duration:.2f}s")
        print(f"总推理时间: {total_inference_time:.2f}s")
        print(f"整体RTF: {total_inference_time / total_audio_duration:.4f}")
        print("="*70)
    
    # 打印平均指标
    if gt_path and compute_metrics_flag and any(len(v) > 0 for v in all_metrics.values()):
        print("\n" + "="*70)
        print("平均客观指标:")
        print("="*70)
        for key, values in all_metrics.items():
            if len(values) > 0:
                avg_value = np.mean(values)
                std_value = np.std(values)
                print(f"{key}: {avg_value:.4f} ± {std_value:.4f} (n={len(values)})")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="FLowSR音频超分辨率推理（从config.yaml读取配置）")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    
    # 可选的命令行覆盖参数
    parser.add_argument('--checkpoint', type=str, default=None, help='模型checkpoint路径（覆盖config）')
    parser.add_argument('--input_path', type=str, default=None, help='输入音频目录（覆盖config）')
    parser.add_argument('--output_path', type=str, default=None, help='输出音频目录（覆盖config）')
    parser.add_argument('--gt_path', type=str, default=None, help='Ground truth路径（覆盖config）')
    parser.add_argument('--simulate_low_sr', type=int, default=None, help='模拟低采样率（覆盖config）')
    parser.add_argument('--device', type=str, default=None, help='设备（覆盖config）')

    args = parser.parse_args()

    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)
    inf_cfg = config['inference']
    
    # 命令行参数覆盖配置文件
    if args.checkpoint is not None:
        inf_cfg['checkpoint'] = args.checkpoint
    if args.input_path is not None:
        inf_cfg['input_path'] = args.input_path
    if args.output_path is not None:
        inf_cfg['output_path'] = args.output_path
    if args.gt_path is not None:
        inf_cfg['gt_path'] = args.gt_path
    if args.simulate_low_sr is not None:
        inf_cfg['simulate_low_sr'] = args.simulate_low_sr
    if args.device is not None:
        inf_cfg['device'] = args.device
    
    # 检查必需参数
    if inf_cfg.get('checkpoint') is None:
        raise ValueError("必须在config.yaml中设置inference.checkpoint或使用--checkpoint参数")
    if inf_cfg.get('input_path') is None:
        raise ValueError("必须在config.yaml中设置inference.input_path或使用--input_path参数")
    if inf_cfg.get('output_path') is None:
        raise ValueError("必须在config.yaml中设置inference.output_path或使用--output_path参数")
    
    # 创建输出目录
    output_dir = inf_cfg['output_path'] + '/output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设备设置
    device = inf_cfg.get('device', 'cuda')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 后处理
    pp = PostProcessing(0)

    print(f'初始化FLowHigh模型...')
    if inf_cfg.get('architecture', '').lower() == 'mamba2':
        print(f"[mamba2] Import status: MAMBA2_AVAILABLE={MAMBA2_AVAILABLE}")
    
    # 初始化Mel Vocoder
    audio_enc_dec_type = MelVoco(
        n_mels=inf_cfg['n_mels'], 
        sampling_rate=inf_cfg['target_sampling_rate'], 
        f_max=inf_cfg['f_max'], 
        n_fft=inf_cfg['n_fft'], 
        win_length=inf_cfg['win_length'], 
        hop_length=inf_cfg['hop_length'],
        vocoder=inf_cfg['vocoder'], 
        vocoder_config=inf_cfg['vocoder_config_path'], 
        vocoder_path=inf_cfg['vocoder_path']
    )        
        
    # 加载模型checkpoint
    print(f"加载模型checkpoint: {inf_cfg['checkpoint']}")
    model_checkpoint = torch.load(inf_cfg['checkpoint'], map_location=device)

    # 初始化模型
    SR_generator = FLowHigh(
        dim_in=audio_enc_dec_type.n_mels, 
        audio_enc_dec=audio_enc_dec_type,
        depth=inf_cfg['n_layers'],
        dim_head=inf_cfg['dim_head'],
        heads=inf_cfg['n_heads'],
        dim=inf_cfg.get('dim', 1024),
        architecture=inf_cfg['architecture'],
    )

    # 初始化CFM包装器
    cfm_wrapper = ConditionalFlowMatcherWrapper(
        flowhigh=SR_generator, 
        cfm_method=inf_cfg['cfm_method'], 
        torchdiffeq_ode_method=inf_cfg['ode_method'], 
        sigma=inf_cfg['sigma']
    )
    
    # 加载checkpoint（strict=False因为vocoder在训练时未保存，会从单独的vocoder文件加载）
    cfm_wrapper.load_state_dict(model_checkpoint['model'], strict=False)

    print(f'设置模型为评估模式...')       
    SR_generator = SR_generator.to(device).eval()
    cfm_wrapper = cfm_wrapper.to(device).eval()

    # 计算参数量
    total_bytes = sum(p.numel() * p.element_size() for p in cfm_wrapper.parameters())
    trainable_bytes = sum(p.numel() * p.element_size() for p in cfm_wrapper.parameters() if p.requires_grad)
    print(f"模型参数占用: {trainable_bytes / (1024 ** 2):.2f} MB (trainable), {total_bytes / (1024 ** 2):.2f} MB (total)")

    summary(cfm_wrapper)

    print(f'\n开始超分辨率推理...')   
    super_resolution(
        input_path=inf_cfg['input_path'],
        output_dir=output_dir,
        config=config,
        cfm_wrapper=cfm_wrapper,
        pp=pp,
    )
    
    print(f"\n推理完成! 输出保存至: {output_dir}")


if __name__ == '__main__':
    main()

