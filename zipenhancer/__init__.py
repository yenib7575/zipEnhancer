"""
zipenhancer — 语音降噪核心包

用法:
    from zipenhancer import denoise, write

    audio, sr = librosa.load("noisy.wav", sr=16000)
    denoised, proc_time, duration = denoise(audio, sr)
    write("output.flac", denoised, sr, fmt="flac")
"""

from zipenhancer.codec import write, FORMATS, get_supported_formats, WriteResult
from zipenhancer.codec import FormatConfig, CodecError
from zipenhancer.denoise import denoise, load_model, ensure_model, normalize_audio
from zipenhancer.denoise import MODEL_ZIPENHANCER, MODEL_FRCRN, MODEL_MOSSFORMER2

__all__ = [
    "denoise",
    "write",
    "load_model",
    "ensure_model",
    "normalize_audio",
    "FORMATS",
    "get_supported_formats",
    "WriteResult",
    "FormatConfig",
    "CodecError",
    "MODEL_ZIPENHANCER",
    "MODEL_FRCRN",
    "MODEL_MOSSFORMER2",
]
