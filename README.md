# ZipEnhancer 语音降噪

基于阿里达摩院 ZipEnhancer 模型的语音降噪 FastAPI 服务。

## 快速开始

### 1. 创建虚拟环境

```bash
conda create -n zipenhancer python=3.10 -y
conda activate zipenhancer
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

GPU 加速（NVIDIA 显卡，**先于上一步**安装 CUDA 版 PyTorch）：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. 配置

复制 `.env.example` 为 `.env`，按需求修改：

```bash
cp .env.example .env
```

### 4. 启动

```bash
uvicorn app:app --host 0.0.0.0 --port 60000
```

## API 接口

### 健康检查

```bash
curl http://127.0.0.1:60000/health
```

### 查看可用模型

```bash
curl http://127.0.0.1:60000/models
```

### 语音降噪（单个文件）

上传音频文件，指定输出文件夹，降噪后的文件自动保存到该目录。

```bash
curl -X POST http://127.0.0.1:60000/denoise ^
  -F "file=@input.wav" ^
  -F "output_dir=./output"
```

**参数说明：**

| 参数 | 必填 | 说明 |
|------|------|------|
| `file` | 是 | 音频文件（wav/mp3/m4a/flac/ogg） |
| `output_dir` | 是 | 输出文件夹路径 |
| `model` | 否 | 模型名称（默认 .env 中配置） |
| `normalize` | 否 | 是否自动音量归一化（默认 true） |
| `target_sr` | 否 | 输出采样率，0=保持原始采样率（默认 0） |

**返回结果：**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "output_path": "./output/input_denoised.wav",
    "sample_rate": 48000,
    "processing_time": "0.62s",
    "real_time_factor": "22.0x",
    "model": "iic/speech_zipenhancer_ans_multiloss_16k_base"
  }
}
```

### 语音降噪（批量处理）

扫描输入文件夹中的所有音频文件，逐个降噪并保存到输出文件夹。

```bash
curl -X POST http://127.0.0.1:60000/denoise/batch ^
  -F "input_dir=./input_folder" ^
  -F "output_dir=./output_folder"
```

**参数说明：**

| 参数 | 必填 | 说明 |
|------|------|------|
| `input_dir` | 是 | 输入文件夹路径 |
| `output_dir` | 是 | 输出文件夹路径 |
| `model` | 否 | 模型名称（默认 .env 中配置） |
| `normalize` | 否 | 是否自动音量归一化（默认 true） |
| `target_sr` | 否 | 输出采样率，0=保持原始采样率（默认 0） |

**返回结果：**

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "input_dir": "./input_folder",
    "output_dir": "./output_folder",
    "total": 10,
    "success": 10,
    "failed": 0,
    "total_time": "5.23s",
    "model": "iic/speech_zipenhancer_ans_multiloss_16k_base",
    "results": [
      {
        "filename": "audio1.wav",
        "output_path": "./output_folder/audio1_denoised.wav",
        "sample_rate": 48000,
        "processing_time": "0.52s",
        "real_time_factor": "28.0x",
        "status": "success"
      }
    ]
  }
}
```

### 输出格式说明

输出文件会尽可能保留原始音频的参数：
- **采样率**：默认与原始文件一致（传 `target_sr` 可覆盖）
- **声道数**：立体声输入 → 立体声输出，单声道输入 → 单声道输出
- **位深**：32-bit float 输入 → 32-bit float 输出，16-bit → 16-bit
- **格式**：WAV 输入保持原始位深，MP3 等其他格式转为 16-bit PCM WAV

### 切换模型

```bash
curl -X POST http://127.0.0.1:60000/denoise ^
  -F "file=@input.wav" ^
  -F "output_dir=./output" ^
  -F "model=iic/speech_frcrn_ans_cirm_16k"
```

## 可用模型

| 模型 ID | 说明 |
|---------|------|
| `iic/speech_zipenhancer_ans_multiloss_16k_base` | ZipEnhancer（轻量） |
| `iic/speech_frcrn_ans_cirm_16k` | FRCRN（实时降噪） |
| `iic/speech_mossformer2_ans_48k` | MossFormer2（高质量） |

## 项目结构

```
├── app.py               # FastAPI 服务主程序
├── log.py               # 日志管理模块
├── requirements.txt     # 依赖列表
├── .env                 # 环境配置（不上传）
├── .env.example         # 环境配置模板
├── .gitignore           # Git 忽略规则
├── README.md            # 使用文档
└── logs/                # 日志输出目录
    ├── app/             # 全部日志
    └── error/           # 错误日志
```
