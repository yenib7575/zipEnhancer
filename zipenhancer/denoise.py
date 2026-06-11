import os
import time
import tempfile
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from log import get_logger
from zipenhancer.standalone import ZipEnhancerStandalone

logger = get_logger("zipenhancer.denoise")

# 模型列表
MODEL_ZIPENHANCER = "iic/speech_zipenhancer_ans_multiloss_16k_base"
MODEL_FRCRN = "iic/speech_frcrn_ans_cirm_16k"
MODEL_MOSSFORMER2 = "iic/speech_mossformer2_ans_48k"

# 模型固定使用 16kHz 处理
PROCESS_SR = 16000

# 模型缓存
_model_pipeline = None
_current_model_name: Optional[str] = None


def load_model(model_name: str):
    """加载模型（返回模型对象，不缓存）"""
    if model_name == MODEL_ZIPENHANCER:
        logger.info(f"加载剥离后的 ZipEnhancer: {model_name}")
        start = time.time()
        ans = ZipEnhancerStandalone(model_name)
        logger.info(f"模型加载完成，耗时: {time.time() - start:.1f}s")
        return ans

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    logger.info(f"加载模型: {model_name}")
    start = time.time()
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model=model_name,
        disable_update=True,
        disable_log=True,
    )
    logger.info(f"模型加载完成，耗时: {time.time() - start:.1f}s")
    return ans


def ensure_model(model_name: str):
    """确保指定模型已加载，需要时才切换"""
    global _model_pipeline, _current_model_name

    if model_name != _current_model_name:
        logger.info(f"切换模型: {_current_model_name} → {model_name}")
        _model_pipeline = load_model(model_name)
        _current_model_name = model_name

    if _model_pipeline is None:
        raise RuntimeError("模型未加载")


def normalize_audio(data: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """音量归一化到目标响度"""
    peak = np.max(np.abs(data))
    if peak > 1e-10:
        target_peak = 10 ** (target_db / 20)
        data = data * (target_peak / peak)
    if np.max(np.abs(data)) > 0.99:
        data = data * 0.95 / np.max(np.abs(data))
    return data


def denoise(
    audio: np.ndarray,
    sample_rate: int,
    model: str = MODEL_ZIPENHANCER,
    normalize: bool = True,
    target_sr: int = 0,
    strength: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """ 降噪

    Args:
        audio (np.ndarray): 输入音频
        sample_rate (int): 输入采样率
        model (str, optional): 模型名称. Defaults to MODEL_ZIPENHANCER.
        normalize (bool, optional): 是否音量归一化. Defaults to True.
        target_sr (int, optional): 目标采样率. Defaults to 0.
        strength (float, optional): 降噪强度 0.0~1.0. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, float, float]: (降噪后音频, 处理时间, 原始音频时长)
    """
    ensure_model(model)
    # 设置降噪强度
    if hasattr(_model_pipeline, 'strength'):
        _model_pipeline.strength = max(0.0, min(1.0, strength))

    orig_channels = audio.shape[0] if audio.ndim > 1 else 1
    output_sr = target_sr if target_sr > 0 else sample_rate

    # 多声道混音为单声道用于模型处理
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio_mono = librosa.to_mono(audio)
    else:
        audio_mono = audio.flatten() if audio.ndim > 1 else audio

    # 重采样到 16kHz（模型固定）
    if sample_rate != PROCESS_SR:
        audio_mono = librosa.resample(audio_mono, orig_sr=sample_rate, target_sr=PROCESS_SR)

    duration = len(audio_mono) / PROCESS_SR

    # 写临时文件 → 模型处理 → 读取（模型接口只支持文件路径）
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_raw:
        tmp_path = tmp_raw.name

    try:
        sf.write(tmp_path, audio_mono, PROCESS_SR)

        start = time.time()
        _model_pipeline(tmp_path, output_path=tmp_path)
        proc_time = time.time() - start

        denoised, _ = sf.read(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 重采样到目标采样率
    if PROCESS_SR != output_sr:
        denoised = librosa.resample(denoised, orig_sr=PROCESS_SR, target_sr=output_sr)

    if normalize:
        denoised = normalize_audio(denoised)

    # 恢复原始声道数
    if orig_channels > 1:
        denoised = np.column_stack([denoised] * orig_channels)

    return denoised, proc_time, duration
