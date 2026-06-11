import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from zipenhancer.models.zipenhancer import ZipenhancerDecorator


def load_config(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def find_model_dir(model_name="iic/speech_zipenhancer_ans_multiloss_16k_base"):
    alt_dir = os.path.join(
        "D:\\", "modelscope", "models", model_name.replace("/", os.sep))
    if os.path.exists(alt_dir):
        return alt_dir
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
    if os.path.exists(cache_dir):
        for root, dirs, files in os.walk(cache_dir):
            if "pytorch_model.bin" in files:
                return root
    return None


def download_via_modelscope(model_name):
    from modelscope.hub.snapshot_download import snapshot_download
    return snapshot_download(model_name)


class ZipEnhancerStandalone:
    """
    剥离版 ZipEnhancer
    原生 16kHz 处理（匹配模型设计采样率）
    兼容 ModelScope pipeline(input_path, output_path=output_path)
    """
    SAMPLE_RATE = 16000  # 模型原生采样率（16k_base）

    def __init__(self, model_name="iic/speech_zipenhancer_ans_multiloss_16k_base"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.strength = 1.0
        self._load_model()

    def _load_model(self):
        from log import get_logger
        logger = get_logger("zipenhancer")
        logger.info(f"加载剥离版 ZipEnhancer: {self.model_name}")

        model_dir = find_model_dir(self.model_name)
        if model_dir is None:
            logger.info("本地未找到模型，从 ModelScope 下载...")
            model_dir = download_via_modelscope(self.model_name)

        weight_path = os.path.join(model_dir, "pytorch_model.bin")
        config_path = os.path.join(model_dir, "configuration.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "configuration.json")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"模型权重未找到: {weight_path}")

        cfg_dict = load_config(config_path)
        model_configs = cfg_dict.get('model', cfg_dict)

        start = time.time()
        self.decorator = ZipenhancerDecorator(weight_path, **model_configs)
        self.model = self.decorator.model
        self.model.eval()
        self.model.to(self.device)
        logger.info(f"模型加载完成，耗时: {time.time() - start:.1f}s")

    def __call__(self, input_path, output_path=None):
        # 1. 读取音频，重采样到 16kHz
        audio_arr, sr_in = sf.read(input_path)
        if sr_in != self.SAMPLE_RATE:
            audio_arr = librosa.resample(audio_arr, orig_sr=sr_in, target_sr=self.SAMPLE_RATE)

        nsamples = audio_arr.shape[0]
        duration = nsamples / self.SAMPLE_RATE

        # [T] → [1, T]
        ndarray = audio_arr.astype(np.float32).reshape(1, -1)

        # 2. 分段参数（4s 窗口，步进 75%，25% 重叠）
        window = self.SAMPLE_RATE * 4       # 64000 samples
        stride = int(window * 0.75)         # 48000
        do_segment = nsamples > window * 2  # >8s 开始分段（防爆显存）

        # 3. Padding（确保能整段处理）
        t = nsamples
        pad_len = 0
        if t < window:
            pad_len = window - t
        elif do_segment:
            need = (t - window) % stride
            if need != 0:
                pad_len = stride - need

        if pad_len > 0:
            ndarray = np.concatenate(
                [ndarray, np.zeros((1, pad_len), dtype=np.float32)], 1)

        # 4. 推理
        start = time.time()
        input_tensor = torch.from_numpy(ndarray).to(self.device)
        b, t = input_tensor.shape

        with torch.no_grad():
            if do_segment:
                outputs = np.zeros(t, dtype=np.float64)
                overlap = window - stride  # 重叠长度
                # 前半段直接输出，后半段用 overlap-add
                first_half = window - overlap // 2
                pos = 0
                while pos + window <= t:
                    seg_out = self.decorator.forward(
                        dict(noisy=input_tensor[:, pos:pos + window]),
                        strength=self.strength,
                    )['wav_l2'][0].cpu().numpy()

                    if pos == 0:
                        outputs[pos:pos + first_half] = seg_out[:first_half]
                    else:
                        # overlap-add 中间段
                        outputs[pos + overlap // 2:pos + window - overlap // 2] += \
                            seg_out[overlap // 2:window - overlap // 2]

                    pos += stride

                # 最后一段
                if pos < t:
                    seg_out = self.decorator.forward(
                        dict(noisy=input_tensor[:, t - window:]),
                        strength=self.strength,
                    )['wav_l2'][0].cpu().numpy()
                    outputs[t - first_half:] = seg_out[-first_half:]

                denoised = outputs[:nsamples].astype(np.float32)
            else:
                denoised = self.decorator.forward(
                    dict(noisy=input_tensor),
                    strength=self.strength,
                )['wav_l2'][0].cpu().numpy()[:nsamples]

        proc_time = time.time() - start

        # 5. 保存
        if output_path:
            os.makedirs(Path(output_path).parent, exist_ok=True)
            sf.write(output_path, denoised, self.SAMPLE_RATE)

        return proc_time, duration
