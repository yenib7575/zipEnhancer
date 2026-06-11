# API 接口文档

- 服务地址：`http://localhost:8765`
- 基础路径：无

---

## 健康检查

```bash
curl http://localhost:8765/health
```

**响应示例：**
```json
{
  "status": "ok",
  "model": "iic/speech_zipenhancer_ans_multiloss_16k_base",
  "model_loaded": true
}
```

---

## 获取模型和格式列表

```bash
curl http://localhost:8765/models
```

**响应示例：**
```json
{
  "models": [...],
  "current": "iic/speech_zipenhancer_ans_multiloss_16k_base",
  "output_formats": [
    {
      "format": "wav",
      "description": "WAV（无损，兼容性最佳）",
      "extension": ".wav",
      "default_subtype": "PCM_16",
      "subtypes": ["PCM_16", "PCM_24", "PCM_32", "FLOAT"]
    },
    {
      "format": "flac",
      "description": "FLAC（无损压缩，文件较小）",
      "extension": ".flac",
      "default_subtype": "PCM_16",
      "subtypes": ["PCM_16", "PCM_24"],
      "compression_range": [0, 8],
      "default_compression": 5
    },
    {
      "format": "mp3",
      "description": "MP3（有损压缩，广泛兼容）",
      "extension": ".mp3",
      "default_subtype": "libmp3lame",
      "bitrate_range": [32000, 320000],
      "default_bitrate": "192k"
    },
    {
      "format": "ogg",
      "description": "OGG Opus/Vorbis（有损压缩，开源优选）",
      "extension": ".ogg",
      "default_subtype": "opus",
      "bitrate_range": [6000, 510000],
      "default_bitrate": "192k"
    }
  ]
}
```

---

## 单文件降噪

```bash
curl -X POST http://localhost:8765/denoise \
  -F "file=@input.wav" \
  -F "output_dir=./output"
```

### 参数

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `file` | 是 | file | 音频文件（wav/mp3/m4a/flac/ogg/aac/wma） |
| `output_dir` | 是 | str | 输出文件夹路径 |
| `model` | 否 | str | 模型名称（默认 .env 中配置） |
| `normalize` | 否 | bool | 音量归一化（默认 true） |
| `target_sr` | 否 | int | 输出采样率，0=保持原始（默认 0） |
| `output_format` | 否 | str | 输出格式: wav/flac/mp3/ogg（默认 wav） |
| `bitrate` | 否 | str | 比特率，仅 mp3/ogg，如 "192k" |
| `compression_level` | 否 | int | 压缩级别，仅 flac (0-8) |
| `strength` | 否 | float | 降噪强度 0.0~1.0（默认 1.0=全力降噪） |

### 各格式测试示例

```bash
# WAV（默认，16-bit PCM）
curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "output_format=wav"

# FLAC（默认压缩）
curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "output_format=flac"

# FLAC 最大压缩
curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "output_format=flac" \
  -F "compression_level=8"

# MP3
curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "output_format=mp3" \
  -F "bitrate=192k"

# MP3 最高码率
curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "output_format=mp3" \
  -F "bitrate=320k"

# OGG
curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "output_format=ogg" \
  -F "bitrate=128k"

# 立体声文件 → FLAC
curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_stereo.wav" \
  -F "output_dir=./output" \
  -F "output_format=flac"

# 长音频 → MP3（分段处理测试）
curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_long.wav" \
  -F "output_dir=./output" \
  -F "output_format=mp3" \
  -F "bitrate=192k"

# 降噪强度测试（从无到全力，逐步对比）
curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "strength=0" \
  -F "normalize=false"

curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "strength=0.3" \
  -F "normalize=false"

curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "strength=0.7" \
  -F "normalize=false"

curl -X POST http://localhost:8765/denoise \
  -F "file=@tests/audio/test_mono.wav" \
  -F "output_dir=./output" \
  -F "strength=1.0" \
  -F "normalize=false"
```

### 响应

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "output_path": "./output/input_denoised.mp3",
    "sample_rate": 48000,
    "output_format": "mp3",
    "output_subtype": "libmp3lame",
    "bitrate": "192k",
    "compression": null,
    "processing_time": "0.62s",
    "real_time_factor": "22.0x",
    "model": "iic/speech_zipenhancer_ans_multiloss_16k_base",
    "strength": 1.0
  }
}
```

### 错误响应

```json
{
  "detail": "不支持的输出格式: aac"
}
```

```json
{
  "detail": "上传文件为空"
}
```

---

## 批量降噪

```bash
curl -X POST http://localhost:8765/denoise/batch \
  -F "input_dir=./input_folder" \
  -F "output_dir=./output_folder"
```

### 参数

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `input_dir` | 是 | str | 输入文件夹路径 |
| `output_dir` | 是 | str | 输出文件夹路径 |
| `model` | 否 | str | 模型名称 |
| `normalize` | 否 | bool | 音量归一化（默认 true） |
| `target_sr` | 否 | int | 输出采样率，0=保持原始（默认 0） |
| `output_format` | 否 | str | 输出格式: wav/flac/mp3/ogg（默认 wav） |
| `bitrate` | 否 | str | 比特率，仅 mp3/ogg |
| `compression_level` | 否 | int | 压缩级别，仅 flac (0-8) |
| `strength` | 否 | float | 降噪强度 0.0~1.0（默认 1.0） |

### 批量测试示例

```bash
# 全部转 MP3
curl -X POST http://localhost:8765/denoise/batch \
  -F "input_dir=./tests/audio" \
  -F "output_dir=./output" \
  -F "output_format=mp3" \
  -F "bitrate=192k"

# 全部转 FLAC 最高压缩
curl -X POST http://localhost:8765/denoise/batch \
  -F "input_dir=./tests/audio" \
  -F "output_dir=./output" \
  -F "output_format=flac" \
  -F "compression_level=8"

# 批量降噪 + 强度控制
curl -X POST http://localhost:8765/denoise/batch \
  -F "input_dir=./tests/audio" \
  -F "output_dir=./output" \
  -F "output_format=mp3" \
  -F "bitrate=192k" \
  -F "strength=0.5"
```

### 响应

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "input_dir": "./input_folder",
    "output_dir": "./output_folder",
    "total": 3,
    "success": 3,
    "failed": 0,
    "total_time": "1.85s",
    "model": "iic/speech_zipenhancer_ans_multiloss_16k_base",
    "strength": 1.0,
    "output_format": "mp3",
    "results": [
      {
        "filename": "test_mono.wav",
        "output_path": "./output/test_mono_denoised.mp3",
        "sample_rate": 48000,
        "output_format": "mp3",
        "output_subtype": "libmp3lame",
        "bitrate": "192k",
        "processing_time": "0.52s",
        "real_time_factor": "28.0x",
        "status": "success"
      }
    ]
  }
}
```

---

## 格式支持矩阵

| 格式 | 编码选项 | 压缩率参考 | 引擎 |
|------|---------|-----------|------|
| WAV | PCM_16 / PCM_24 / PCM_32 / FLOAT | 无损（基准） | soundfile |
| FLAC | PCM_16 / PCM_24, compression 0-8 | ~40-60% | soundfile |
| MP3 | 32-320 kbps | ~15-25% | ffmpeg |
| OGG | 6-510 kbps | ~15-25% | ffmpeg |

---

## 切换模型

```bash
curl -X POST http://localhost:8765/denoise \
  -F "file=@input.wav" \
  -F "output_dir=./output" \
  -F "model=iic/speech_frcrn_ans_cirm_16k"
```
