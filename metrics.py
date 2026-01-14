import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


def compute_lsd(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    window: Optional[torch.Tensor] = None,
    center: bool = False,
    eps: float = 1e-8
) -> float:
    """
    计算对数频谱距离 (Log-Spectral Distance, LSD)
    
    Args:
        pred: 预测波形 (T,) 或 (B, T)
        target: 目标波形 (T,) 或 (B, T)
        n_fft: FFT大小
        hop_length: 跳跃长度
        win_length: 窗口长度
        window: 窗口函数
        center: 是否填充输入
        eps: 避免除零的小值
        
    Returns:
        lsd: LSD值
    """
    # 接受常见波形形状处理
    if pred.dim() == 3 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 3 and target.shape[1] == 1:
        target = target.squeeze(1)

    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    
    if window is None:
        window = torch.hann_window(win_length, device=pred.device, dtype=pred.dtype)
        
    # STFT
    pred_stft = torch.stft(pred, n_fft, hop_length, win_length, window, center=center, return_complex=True)
    target_stft = torch.stft(target, n_fft, hop_length, win_length, window, center=center, return_complex=True)
    
    pred_mag_sq = pred_stft.abs().pow(2) + eps
    target_mag_sq = target_stft.abs().pow(2) + eps
    
    # 对数频谱差异
    log_diff = torch.log10(target_mag_sq) - torch.log10(pred_mag_sq)
    
    # 平方差
    log_diff_sq = log_diff.pow(2)
    
    # 频率bins平均
    mean_freq = log_diff_sq.mean(dim=1)
    
    # 平方根 (每帧RMSE)
    rmse_per_frame = torch.sqrt(mean_freq)
    
    # 时间帧平均
    lsd = rmse_per_frame.mean(dim=1)
    
    return float(lsd.mean().item())


def compute_sisnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> float:
    """
    计算尺度不变信噪比 (Scale-Invariant Signal-to-Noise Ratio, SI-SNR)
    
    Args:
        pred: 预测波形 (T,) 或 (B, T)
        target: 目标波形 (T,) 或 (B, T)
        eps: 避免除零的小值
        
    Returns:
        sisnr: SI-SNR值 (dB)
    """
    if pred.dim() == 3 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 3 and target.shape[1] == 1:
        target = target.squeeze(1)

    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
        
    # 零均值归一化
    pred = pred - torch.mean(pred, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # 目标投影
    dot_prod = torch.sum(pred * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    scaled_target = (dot_prod / target_energy) * target
    
    # 噪声分量
    noise = pred - scaled_target
    
    # SI-SNR计算
    ratio = torch.sum(scaled_target ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    sisnr = 10 * torch.log10(ratio + eps)
    
    return float(sisnr.mean().item())


def compute_pesq(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    sr: int = 16000,
    mode: str = 'wb'
) -> float:
    """
    计算PESQ分数
    
    Args:
        pred: 预测波形
        target: 目标波形
        sr: 采样率 (必须是8000或16000)
        mode: 'wb' (宽带) 或 'nb' (窄带)
        
    Returns:
        pesq_score: PESQ分数
    """
    if not PESQ_AVAILABLE:
        print("Warning: pesq not installed, returning 0.0")
        return 0.0
        
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    if pred.ndim > 1:
        pred = pred.reshape(-1)
    if target.ndim > 1:
        target = target.reshape(-1)
    
    # 确保长度一致
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]
        
    # PESQ需要16k或8k
    if sr not in [8000, 16000]:
        # 需要重采样
        import librosa
        if sr > 16000:
            target_sr = 16000
        else:
            target_sr = 8000
        pred = librosa.resample(pred, orig_sr=sr, target_sr=target_sr)
        target = librosa.resample(target, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    try:
        score = pesq(sr, target, pred, mode)
    except Exception as e:
        print(f"Warning: PESQ computation failed: {e}")
        score = 0.0
        
    return float(score)


def compute_stoi(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    sr: int = 16000,
    extended: bool = False
) -> float:
    """
    计算短时客观可懂度 (Short-Time Objective Intelligibility, STOI)
    
    Args:
        pred: 预测波形
        target: 目标波形
        sr: 采样率
        extended: 是否使用扩展版本 (ESTOI)
        
    Returns:
        stoi_score: STOI分数 (0-1之间，或扩展版本的-1到1)
    """
    if not STOI_AVAILABLE:
        print("Warning: pystoi not installed, returning 0.0")
        return 0.0
        
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    if pred.ndim > 1:
        pred = pred.reshape(-1)
    if target.ndim > 1:
        target = target.reshape(-1)
    
    # 确保长度一致
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]
        
    try:
        score = stoi(target, pred, sr, extended=extended)
    except Exception as e:
        print(f"Warning: STOI computation failed: {e}")
        score = 0.0
        
    return float(score)


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sr: int = 48000,
    n_fft: int = 2048,
    hop_length: int = 480,
    win_length: int = 2048
) -> dict:
    """
    计算所有客观指标
    
    Args:
        pred: 预测波形
        target: 目标波形
        sr: 采样率
        n_fft: FFT大小
        hop_length: 跳跃长度
        win_length: 窗口长度
        
    Returns:
        metrics: 包含所有指标的字典
    """
    metrics = {}
    
    # LSD
    try:
        metrics['LSD'] = compute_lsd(pred, target, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    except Exception as e:
        print(f"Warning: LSD computation failed: {e}")
        metrics['LSD'] = 0.0
    
    # SI-SNR
    try:
        metrics['SI-SNR'] = compute_sisnr(pred, target)
    except Exception as e:
        print(f"Warning: SI-SNR computation failed: {e}")
        metrics['SI-SNR'] = 0.0
    
    # PESQ (需要16k或8k)
    try:
        metrics['PESQ'] = compute_pesq(pred, target, sr=sr)
    except Exception as e:
        print(f"Warning: PESQ computation failed: {e}")
        metrics['PESQ'] = 0.0
    
    # STOI
    try:
        metrics['STOI'] = compute_stoi(pred, target, sr=sr)
    except Exception as e:
        print(f"Warning: STOI computation failed: {e}")
        metrics['STOI'] = 0.0
    
    return metrics

