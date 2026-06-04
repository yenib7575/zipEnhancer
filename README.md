<div align="center">

<pre>
 ______       _____       _                               
|__  (_)_ __ | ____|_ __ | |__   __ _ _ __   ___ ___ _ __ 
  / /| | '_ \|  _| | '_ \| '_ \ / _` | '_ \ / __/ _ \ '__|
 / /_| | |_) | |___| | | | | | | (_| | | | | (_|  __/ |   
/____|_| .__/|_____|_| |_|_| |_|\__,_|_| |_|\___\___|_|   
       |_|                                                
</pre>

</div>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue?style=flat-square&logo=python" alt="Python 3.10">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-teal?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/ModelScope-达摩院-6240ff?style=flat-square" alt="ModelScope">
</p>



### 做了什么

- **模型剥离** — 从 ModelScope 黑盒 pipeline 中提取出 ZipEnhancer，用原生 PyTorch 加载权重推理，不再依赖 pipeline 封装
- **FP16 半精度推理** — 仅模型计算部分使用 FP16，STFT/iSTFT 保持 FP32 避免 cuFFT 精度问题，显存占用降低 ~40%
- **长音频分段** — 4s 滑动窗口 + 75% 重叠的 overlap-add 策略，支持任意时长音频，彻底解决 CUDA OOM
- **多模型切换** — 同时支持 ZipEnhancer（轻量）、FRCRN（实时）、MossFormer2（高质）三种模型
- **声道/位深保持** — 立体声输入 → 立体声输出，32-bit float / 16-bit PCM 自动适配

### 它能做什么？

- 清除录音中的**环境噪声**（空调声、风扇声、键盘声、街道噪音等）
- 支持**单文件**和**批量处理**两种模式
- 多种降噪模型**一键切换**
- GPU 加速，实时率可达 **20x 以上**（RTX 4090）

无需 ModelScope pipeline 黑盒，一行命令启动服务，适合集成到语音处理流程、会议录音后处理、音频预处理管道等场景。

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
uvicorn app:app --host 0.0.0.0 --port 8765
```

## API 接口

### 健康检查

```bash
curl http://127.0.0.1:8765/health
```

![健康检查](images/健康检查.png)

### 查看可用模型

```bash
curl http://127.0.0.1:8765/models
```

![查看可用模型](images/查看可用模型.png)

### 语音降噪（单个文件）

上传音频文件，指定输出文件夹，降噪后的文件自动保存到该目录。

```bash
curl -X POST http://127.0.0.1:8765/denoise ^
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

![音频降噪](images/音频降噪.png)

### 语音降噪（批量处理）

扫描输入文件夹中的所有音频文件，逐个降噪并保存到输出文件夹。

```bash
curl -X POST http://127.0.0.1:8765/denoise/batch ^
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
curl -X POST http://127.0.0.1:8765/denoise ^
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

## Roadmap

### 已完成
- [x] 单文件语音降噪
- [x] 批量文件语音降噪
- [x] 多模型切换（ZipEnhancer / FRCRN / MossFormer2）
- [x] 音量归一化
- [x] 自定义输出采样率
- [x] 声道/位深保持

### 计划中
- [ ] Web UI 界面（拖拽上传 + 在线试听）
- [ ] Docker 一键部署
- [ ] CLI 命令行工具
- [ ] 实时流式降噪（WebSocket）
- [ ] 模型量化（减小模型体积，提升推理速度）
- [ ] 超分（低采样率 → 高采样率）
- [ ] 去混响
- [ ] VAD 自动静音切除
- [ ] 语音识别（ASR）
- [ ] 说话人分离
- [ ] 音频格式转换


## 项目结构

```
├── app.py                 # FastAPI 服务主程序
├── log.py                 # 日志管理模块
├── zipenhancer/           # 降噪核心包
│   ├── __init__.py
│   ├── standalone.py      # 剥离版推理（纯 PyTorch）
│   ├── models/            # 模型架构
│   │   ├── zipenhancer.py
│   │   └── layers/
│   └── configs/
│       └── configuration.json
├── images/                # README 截图
├── test/                  # 测试音频
├── requirements.txt       # 依赖列表
├── LICENSE                # MIT 开源许可证
├── .env                   # 环境配置（不上传）
├── .env.example           # 环境配置模板
├── .gitignore             # Git 忽略规则
├── README.md              # 使用文档
└── logs/                  # 日志输出目录
    ├── app/               # 全部日志
    └── error/             # 错误日志
```

## Credits

- 降噪模型：[阿里达摩院 ZipEnhancer](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)（Apache 2.0）
- 模型提取参考：[boreas-l/zipEnhancer](https://github.com/boreas-l/zipEnhancer)

## License

[MIT](LICENSE) © 2024 gao yi jun
