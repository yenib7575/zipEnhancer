import os
import time
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv

from log import get_logger
from zipenhancer import (
    denoise as core_denoise,
    write as codec_write,
    ensure_model,
    FORMATS,
    get_supported_formats,
    MODEL_ZIPENHANCER,
    MODEL_FRCRN,
    MODEL_MOSSFORMER2,
)

# 支持的音频格式
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

load_dotenv()

DEFAULT_MODEL = os.getenv("DENOISE_MODEL")
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

logger = get_logger("app")
app = FastAPI(title="语音降噪服务", version="1.0.0")


@app.on_event("startup")
async def startup():
    ensure_model(DEFAULT_MODEL)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "model_loaded": True,
    }


@app.get("/models")
async def list_models():
    return {
        "models": [
            {"id": MODEL_ZIPENHANCER, "name": "ZipEnhancer", "description": "轻量降噪"},
            {"id": MODEL_FRCRN, "name": "FRCRN", "description": "实时降噪"},
            {"id": MODEL_MOSSFORMER2, "name": "MossFormer2", "description": "高质量降噪"},
        ],
        "current": DEFAULT_MODEL,
        "output_formats": get_supported_formats(),
    }


def _process_file(
    in_path: str,
    out_path: str,
    model: str,
    normalize: bool,
    output_sr: int = 0,
    output_format: str = "wav",
    bitrate: str = None,
    compression_level: int = None,
) -> tuple:
    """处理单个音频文件，返回 (耗时, 时长, 采样率, 格式信息)"""
    if output_format not in FORMATS:
        from zipenhancer.codec import FormatNotSupported
        raise FormatNotSupported(output_format)

    # 获取原始音频信息
    info = sf.info(in_path)
    orig_channels = info.channels
    orig_subtype = info.subtype
    if output_sr == 0:
        output_sr = int(info.samplerate)

    # WAV 不支持的编码（如 MP3）改用 16-bit PCM
    if orig_subtype not in {'PCM_S8', 'PCM_16', 'PCM_24', 'PCM_32', 'PCM_U8',
                             'FLOAT', 'DOUBLE', 'IMA_ADPCM', 'MS_ADPCM',
                             'GSM610', 'ULAW', 'ALAW'}:
        orig_subtype = 'PCM_16'

    # 加载音频（保持原始声道数）
    audio, sr = librosa.load(in_path, sr=16000, mono=False)

    # 降噪
    denoised, proc_time, duration = core_denoise(
        audio, sr,
        model=model,
        normalize=normalize,
        target_sr=output_sr,
    )

    # 编码输出
    sub = orig_subtype if output_format in ("wav", "flac") else None
    result = codec_write(
        out_path, denoised, output_sr,
        fmt=output_format,
        subtype=sub,
        bitrate=bitrate,
        compression=compression_level,
        atomic=True,
    )

    fmt_info = {
        "output_format": result.format,
        "output_subtype": result.subtype,
        "bitrate": result.bitrate,
        "compression": result.compression,
    }
    return proc_time, duration, output_sr, fmt_info


@app.post("/denoise")
async def denoise(
    file: UploadFile = File(..., description="音频文件"),
    output_dir: str = Form(..., description="输出文件夹路径"),
    model: str = Form(DEFAULT_MODEL, description="模型名称"),
    normalize: bool = Form(True, description="是否音量归一化"),
    target_sr: int = Form(0, description="输出采样率，0=保持原始采样率"),
    output_format: str = Form("wav", description="输出格式: wav/flac/mp3/ogg"),
    bitrate: str = Form(None, description="比特率 (mp3/ogg)，如 192k"),
    compression_level: int = Form(None, ge=0, le=8, description="压缩级别 (flac 0-8)"),
):
    """上传单个音频 → 降噪 → 保存 → 返回 JSON"""
    if not file.filename:
        raise HTTPException(400, "文件名为空")

    if output_format not in FORMATS:
        raise HTTPException(400, f"不支持的输出格式: {output_format}")

    stem = Path(file.filename).stem
    ext = FORMATS[output_format].ext
    output_path = os.path.join(output_dir, f"{stem}_denoised{ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".wav") as tmp_in:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "上传文件为空")
        tmp_in.write(content)
        in_path = tmp_in.name

    try:
        logger.info(f"接收文件: {file.filename} ({len(content)} bytes) [format={output_format}]")
        proc_time, duration, output_sr, fmt_info = _process_file(
            in_path, output_path, model, normalize, target_sr,
            output_format, bitrate, compression_level,
        )
        logger.info(f"降噪完成: {proc_time:.2f}s (x{duration/proc_time:.1f} 实时比)")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"降噪失败: {e}")
        raise HTTPException(500, f"降噪失败: {e}")
    finally:
        os.unlink(in_path)

    return {
        "code": 0,
        "message": "success",
        "data": {
            "output_path": output_path,
            "sample_rate": output_sr,
            "output_format": fmt_info["output_format"],
            "output_subtype": fmt_info["output_subtype"],
            "bitrate": fmt_info.get("bitrate"),
            "compression": fmt_info.get("compression"),
            "processing_time": f"{proc_time:.2f}s",
            "real_time_factor": f"{duration/proc_time:.1f}x",
            "model": model,
        },
    }


@app.post("/denoise/batch")
async def denoise_batch(
    input_dir: str = Form(..., description="输入文件夹路径"),
    output_dir: str = Form(..., description="输出文件夹路径"),
    model: str = Form(DEFAULT_MODEL, description="模型名称"),
    normalize: bool = Form(True, description="是否音量归一化"),
    target_sr: int = Form(0, description="输出采样率，0=保持原始采样率"),
    output_format: str = Form("wav", description="输出格式: wav/flac/mp3/ogg"),
    bitrate: str = Form(None, description="比特率 (mp3/ogg)，如 192k"),
    compression_level: int = Form(None, ge=0, le=8, description="压缩级别 (flac 0-8)"),
):
    """批量降噪：扫描输入文件夹所有音频，逐个处理"""
    if not os.path.isdir(input_dir):
        raise HTTPException(400, f"输入文件夹不存在: {input_dir}")

    if output_format not in FORMATS:
        raise HTTPException(400, f"不支持的输出格式: {output_format}")

    # 扫描音频文件
    files = []
    for f in sorted(os.listdir(input_dir)):
        ext = Path(f).suffix.lower()
        if ext in AUDIO_EXTENSIONS:
            files.append(f)

    if not files:
        raise HTTPException(400, f"输入文件夹中没有找到音频文件: {input_dir}")

    ext = FORMATS[output_format].ext
    logger.info(f"批量降噪: {input_dir} → {output_dir}，共 {len(files)} 个文件 [format={output_format}]")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    success = 0
    failed = 0
    total_start = time.time()

    for filename in files:
        in_path = os.path.join(input_dir, filename)
        stem = Path(filename).stem
        out_path = os.path.join(output_dir, f"{stem}_denoised{ext}")

        logger.info(f"处理 ({success + failed + 1}/{len(files)}): {filename}")
        try:
            proc_time, duration, output_sr, fmt_info = _process_file(
                in_path, out_path, model, normalize, target_sr,
                output_format, bitrate, compression_level,
            )
            results.append({
                "filename": filename,
                "output_path": out_path,
                "sample_rate": output_sr,
                "output_format": fmt_info["output_format"],
                "output_subtype": fmt_info["output_subtype"],
                "bitrate": fmt_info.get("bitrate"),
                "compression": fmt_info.get("compression"),
                "processing_time": f"{proc_time:.2f}s",
                "real_time_factor": f"{duration/proc_time:.1f}x",
                "status": "success",
            })
            success += 1
        except Exception as e:
            logger.error(f"处理失败: {filename} - {e}")
            results.append({
                "filename": filename,
                "status": "failed",
                "error": str(e),
            })
            failed += 1

    total_time = time.time() - total_start

    return {
        "code": 0,
        "message": "success",
        "data": {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "total": len(files),
            "success": success,
            "failed": failed,
            "total_time": f"{total_time:.2f}s",
            "model": model,
            "output_format": output_format,
            "results": results,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
