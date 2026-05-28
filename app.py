import os
import time
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv

from log import get_logger

# 支持的音频格式
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

load_dotenv()

# 模型列表
MODEL_ZIPENHANCER = "iic/speech_zipenhancer_ans_multiloss_16k_base"
MODEL_FRCRN = "iic/speech_frcrn_ans_cirm_16k"
MODEL_MOSSFORMER2 = "iic/speech_mossformer2_ans_48k"

DEFAULT_MODEL = os.getenv("DENOISE_MODEL")
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

logger = get_logger("app")
app = FastAPI(title="语音降噪服务", version="1.0.0")

model_pipeline = None
current_model_name = None


def load_model(model_name: str):
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


def normalize_audio(data: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(data))
    if peak > 1e-10:
        target_peak = 10 ** (target_db / 20)
        data = data * (target_peak / peak)
    if np.max(np.abs(data)) > 0.99:
        data = data * 0.95 / np.max(np.abs(data))
    return data


@app.on_event("startup")
async def startup():
    global model_pipeline, current_model_name
    current_model_name = DEFAULT_MODEL
    model_pipeline = load_model(current_model_name)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": current_model_name,
        "model_loaded": model_pipeline is not None,
    }


@app.get("/models")
async def list_models():
    return {
        "models": [
            {"id": MODEL_ZIPENHANCER, "name": "ZipEnhancer", "description": "轻量降噪"},
            {"id": MODEL_FRCRN, "name": "FRCRN", "description": "实时降噪 (ClearerVoice)"},
            {"id": MODEL_MOSSFORMER2, "name": "MossFormer2", "description": "高质量降噪 (ClearerVoice)"},
        ],
        "current": current_model_name,
    }


def _ensure_model(model: str):
    """确保使用指定模型，需要时才切换"""
    global model_pipeline, current_model_name
    if model != current_model_name:
        logger.info(f"切换模型: {current_model_name} → {model}")
        try:
            model_pipeline = load_model(model)
            current_model_name = model
        except Exception as e:
            raise HTTPException(400, f"模型加载失败: {e}")
    if model_pipeline is None:
        raise HTTPException(500, "模型未加载")


def _process_file(in_path: str, out_path: str, model: str, normalize: bool, output_sr: int = 0) -> tuple:
    """处理单个音频文件，返回 (耗时, 时长)"""
    _ensure_model(model)

    # 模型固定使用 16kHz 处理
    process_sr = 16000

    # 获取原始音频信息（采样率、声道数、位深）
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
    audio, _ = librosa.load(in_path, sr=process_sr, mono=False)

    # 多声道则混音为单声道用于模型处理
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio_mono = librosa.to_mono(audio)
    else:
        audio_mono = audio.flatten() if audio.ndim > 1 else audio

    duration = len(audio_mono) / process_sr

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_raw:
        tmp_path = tmp_raw.name

    try:
        sf.write(tmp_path, audio_mono, process_sr)
        start = time.time()
        model_pipeline(tmp_path, output_path=tmp_path)
        proc_time = time.time() - start

        denoised, _ = sf.read(tmp_path)

        # 重采样到目标采样率
        if process_sr != output_sr:
            denoised = librosa.resample(denoised, orig_sr=process_sr, target_sr=output_sr)

        if normalize:
            denoised = normalize_audio(denoised, target_db=-3.0)

        # 恢复原始声道数（多声道时复制声道）
        if orig_channels > 1:
            denoised = np.column_stack([denoised] * orig_channels)

        os.makedirs(Path(out_path).parent, exist_ok=True)
        sf.write(out_path, denoised, output_sr, subtype=orig_subtype)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return proc_time, duration, output_sr


@app.post("/denoise")
async def denoise(
    file: UploadFile = File(..., description="音频文件"),
    output_dir: str = Form(..., description="输出文件夹路径"),
    model: str = Form(DEFAULT_MODEL, description="模型名称"),
    normalize: bool = Form(True, description="是否音量归一化"),
    target_sr: int = Form(0, description="输出采样率，0=保持原始采样率"),
):
    """上传单个音频 → 降噪 → 保存 → 返回 JSON"""
    if not file.filename:
        raise HTTPException(400, "文件名为空")

    stem = Path(file.filename).stem
    suffix = Path(file.filename).suffix or ".wav"
    output_path = os.path.join(output_dir, f"{stem}_denoised.wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "上传文件为空")
        tmp_in.write(content)
        in_path = tmp_in.name

    try:
        logger.info(f"接收文件: {file.filename} ({len(content)} bytes)")
        proc_time, duration, output_sr = _process_file(in_path, output_path, model, normalize, target_sr)
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
            "processing_time": f"{proc_time:.2f}s",
            "real_time_factor": f"{duration/proc_time:.1f}x",
            "model": current_model_name,
        },
    }


@app.post("/denoise/batch")
async def denoise_batch(
    input_dir: str = Form(..., description="输入文件夹路径"),
    output_dir: str = Form(..., description="输出文件夹路径"),
    model: str = Form(DEFAULT_MODEL, description="模型名称"),
    normalize: bool = Form(True, description="是否音量归一化"),
    target_sr: int = Form(0, description="输出采样率，0=保持原始采样率"),
):
    """批量降噪：扫描输入文件夹所有音频，逐个处理"""
    if not os.path.isdir(input_dir):
        raise HTTPException(400, f"输入文件夹不存在: {input_dir}")

    # 扫描音频文件
    files = []
    for f in sorted(os.listdir(input_dir)):
        ext = Path(f).suffix.lower()
        if ext in AUDIO_EXTENSIONS:
            files.append(f)

    if not files:
        raise HTTPException(400, f"输入文件夹中没有找到音频文件: {input_dir}")

    logger.info(f"批量降噪: {input_dir} → {output_dir}，共 {len(files)} 个文件")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    success = 0
    failed = 0
    total_start = time.time()

    for filename in files:
        in_path = os.path.join(input_dir, filename)
        stem = Path(filename).stem
        out_path = os.path.join(output_dir, f"{stem}_denoised.wav")

        logger.info(f"处理 ({success + failed + 1}/{len(files)}): {filename}")
        try:
            proc_time, duration, output_sr = _process_file(in_path, out_path, model, normalize, target_sr)
            results.append({
                "filename": filename,
                "output_path": out_path,
                "sample_rate": output_sr,
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
            "model": current_model_name,
            "results": results,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
