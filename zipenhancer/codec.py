import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf

from log import get_logger

logger = get_logger("zipenhancer.codec")


@dataclass(frozen=True)
class FormatConfig:
    """格式能力定义"""
    name: str                     # 唯一标识符: wav / flac / mp3 / ogg
    description: str              # 人类可读描述
    ext: str                      # 文件扩展名（含点）
    subtypes: tuple               # 有效 subtype 列表，空 tuple 表示由 codec 名决定
    default_subtype: str          # 默认 subtype
    needs_ffmpeg: bool            # 是否依赖 ffmpeg
    supports_compression: bool = False
    supports_bitrate: bool = False
    compression_range: tuple = ()
    bitrate_range: tuple = ()
    default_bitrate: Optional[str] = None
    default_compression: Optional[int] = None


FORMATS = {
    "wav": FormatConfig(
        name="wav",
        description="WAV（无损，兼容性最佳）",
        ext=".wav",
        subtypes=("PCM_16", "PCM_24", "PCM_32", "FLOAT"),
        default_subtype="PCM_16",
        needs_ffmpeg=False,
    ),
    "flac": FormatConfig(
        name="flac",
        description="FLAC（无损压缩，文件较小）",
        ext=".flac",
        subtypes=("PCM_16", "PCM_24"),
        default_subtype="PCM_16",
        needs_ffmpeg=False,
        supports_compression=True,
        compression_range=(0, 8),
        default_compression=5,
    ),
    "mp3": FormatConfig(
        name="mp3",
        description="MP3（有损压缩，广泛兼容）",
        ext=".mp3",
        subtypes=(),
        default_subtype="libmp3lame",
        needs_ffmpeg=True,
        supports_bitrate=True,
        bitrate_range=(32000, 320000),
        default_bitrate="192k",
    ),
    "ogg": FormatConfig(
        name="ogg",
        description="OGG Opus/Vorbis（有损压缩，开源优选）",
        ext=".ogg",
        subtypes=(),
        default_subtype="opus",
        needs_ffmpeg=True,
        supports_bitrate=True,
        bitrate_range=(6000, 510000),
        default_bitrate="192k",
    ),
}


class CodecError(Exception):
    """编解码错误基类"""
    def __init__(self, message: str, hint: str = ""):
        self.hint = hint
        super().__init__(message if not hint else f"{message}（{hint}）")


class FormatNotSupported(CodecError):
    """不支持的格式"""
    def __init__(self, fmt: str):
        super().__init__(
            f"不支持的输出格式: {fmt}",
            f"可选: {', '.join(FORMATS)}",
        )


class SubtypeNotSupported(CodecError):
    """不支持的编码参数"""
    def __init__(self, fmt: str, subtype: str, valid: tuple):
        super().__init__(
            f"格式 {fmt} 不支持编码 {subtype}",
            f"{fmt} 支持的编码: {', '.join(valid)}",
        )


class FfmpegNotFound(CodecError):
    """ffmpeg 未安装"""
    def __init__(self):
        super().__init__(
            "系统未找到 ffmpeg",
            "请安装 ffmpeg 并将其加入 PATH（https://ffmpeg.org/download.html）",
        )


class DiskSpaceError(CodecError):
    """磁盘空间不足"""
    pass


class EncodeError(CodecError):
    """编码失败"""
    pass


@dataclass
class WriteResult:
    """编码写入结果"""
    path: str
    format: str
    subtype: str
    bitrate: Optional[str] = None
    compression: Optional[int] = None


def write(
    path: str,
    data: np.ndarray,
    sample_rate: int,
    fmt: str = "wav",
    subtype: Optional[str] = None,
    bitrate: Optional[str] = None,
    compression: Optional[int] = None,
    atomic: bool = True,
) -> WriteResult:
    """
    编码并写入音频文件。

    参数
    ----
    path : str
        输出路径（扩展名可不匹配 fmt，以 fmt 为准）
    data : np.ndarray
        音频数据，shape [samples] 或 [samples, channels]
    sample_rate : int
        采样率
    fmt : str
        输出格式，wav / flac / mp3 / ogg
    subtype : str, optional
        编码子类型，默认用格式的 default_subtype
    bitrate : str, optional
        比特率 e.g. "192k"，仅 mp3/ogg
    compression : int, optional
        压缩级别，仅 flac (0-8)
    atomic : bool
        是否原子写入（默认 True）

    返回
    ----
    WriteResult
    """
    cfg = FORMATS.get(fmt)
    if cfg is None:
        raise FormatNotSupported(fmt)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # 1. 校验 & 填充默认值
    subtype, bitrate, compression = _resolve_params(cfg, subtype, bitrate, compression)

    # 2. 估算大小 + 磁盘空间检查
    estimated = _estimate_size(data, sample_rate, cfg, subtype, bitrate, compression)
    _check_disk_space(path, estimated)

    # 3. 写入
    if atomic:
        _atomic_write(path, data, sample_rate, cfg, subtype, bitrate, compression)
    else:
        _do_write(path, data, sample_rate, cfg, subtype, bitrate, compression)

    return WriteResult(
        path=path,
        format=fmt,
        subtype=subtype,
        bitrate=bitrate,
        compression=compression,
    )


def _resolve_params(
    cfg: FormatConfig,
    subtype: Optional[str],
    bitrate: Optional[str],
    compression: Optional[int],
) -> tuple:
    """校验并填充编码参数，必要时自动降级"""
    # subtype
    if subtype is None:
        subtype = cfg.default_subtype
    else:
        if cfg.subtypes and subtype not in cfg.subtypes:
            logger.warning(
                "格式 %s 不支持编码 %s，降级为 %s",
                cfg.name, subtype, cfg.default_subtype,
            )
            subtype = cfg.default_subtype

    # bitrate
    if cfg.supports_bitrate:
        if bitrate is None:
            bitrate = cfg.default_bitrate
        else:
            try:
                bps = _parse_bitrate(bitrate)
            except (ValueError, TypeError):
                logger.warning("比特率格式无效 %s，使用默认值 %s", bitrate, cfg.default_bitrate)
                bitrate = cfg.default_bitrate
            else:
                if bps < cfg.bitrate_range[0] or bps > cfg.bitrate_range[1]:
                    logger.warning(
                        "比特率 %s 超出 %s 范围 (%d-%d bps)，使用默认值 %s",
                        bitrate, cfg.name, cfg.bitrate_range[0], cfg.bitrate_range[1],
                        cfg.default_bitrate,
                    )
                    bitrate = cfg.default_bitrate
    else:
        bitrate = None

    # compression
    if cfg.supports_compression:
        if compression is None:
            compression = cfg.default_compression
        elif compression < cfg.compression_range[0] or compression > cfg.compression_range[1]:
            logger.warning(
                "压缩级别 %d 超出 %s 范围 (%d-%d)，使用默认值 %d",
                compression, cfg.name, cfg.compression_range[0], cfg.compression_range[1],
                cfg.default_compression,
            )
            compression = cfg.default_compression
    else:
        compression = None

    return subtype, bitrate, compression


def _parse_bitrate(bitrate: str) -> int:
    """将比特率字符串转为 bps，如 '192k' → 192000"""
    s = str(bitrate).strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    elif s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    else:
        return int(s)


def _estimate_size(
    data: np.ndarray,
    sample_rate: int,
    cfg: FormatConfig,
    subtype: str,
    bitrate: Optional[str],
    compression: Optional[int],
) -> int:
    """估算输出文件大小（字节）"""
    if data.ndim == 1:
        n_channels = 1
        n_samples = data.shape[0]
    else:
        n_channels = data.shape[1] if data.shape[1] > 0 else 1
        n_samples = data.shape[0]

    duration = n_samples / sample_rate

    if cfg.name == "wav":
        bps = _subtype_bytes(subtype)
        return int(n_samples * n_channels * bps * 1.02)  # + header overhead
    elif cfg.name == "flac":
        bps = _subtype_bytes(subtype)
        # FLAC 通常压缩到原始 PCM 的 50-70%
        return int(n_samples * n_channels * bps * 0.65)
    elif cfg.name == "mp3" and bitrate:
        bps = _parse_bitrate(bitrate)
        return int(bps / 8 * duration * 1.05)
    elif cfg.name == "ogg" and bitrate:
        bps = _parse_bitrate(bitrate)
        return int(bps / 8 * duration * 1.05)
    else:
        return int(n_samples * n_channels * 4 * 1.1)


def _subtype_bytes(subtype: str) -> int:
    """subtype → 每样本字节数"""
    return {
        "PCM_S8": 1, "PCM_U8": 1,
        "PCM_16": 2,
        "PCM_24": 3,
        "PCM_32": 4, "FLOAT": 4,
        "DOUBLE": 8,
    }.get(subtype, 4)


def _do_write(
    path: str,
    data: np.ndarray,
    sample_rate: int,
    cfg: FormatConfig,
    subtype: str,
    bitrate: Optional[str],
    compression: Optional[int],
):
    """实际写入（依格式派发）"""
    if cfg.name in ("wav", "flac"):
        _write_sf(path, data, sample_rate, cfg, subtype, compression)
    elif cfg.needs_ffmpeg:
        _write_ffmpeg(path, data, sample_rate, cfg, subtype, bitrate)
    else:
        raise EncodeError(f"未知格式: {cfg.name}")


def _write_sf(
    path: str,
    data: np.ndarray,
    sample_rate: int,
    cfg: FormatConfig,
    subtype: str,
    compression: Optional[int],
):
    """通过 soundfile 写入 WAV/FLAC"""
    # data 约定: [samples] 或 [samples, channels]，已是 soundfile 原生格式
    if data.ndim == 1:
        data_for_sf = data
    else:
        data_for_sf = data

    # RF64 自动降级（>4GB WAV）
    sf_format = cfg.name.upper()
    if cfg.name == "wav" and data_for_sf.nbytes > 3.5 * 1024 ** 3:
        sf_format = "RF64"
        logger.info("文件超过 4GB，自动使用 RF64 格式")

    sf_kwargs = {"format": sf_format, "subtype": subtype}
    if compression is not None and cfg.name == "flac":
        # soundfile 0.13.x 的 compression_level: FLAC 有效范围 0.0-1.0
        sf_kwargs["compression_level"] = compression / 8.0

    try:
        sf.write(path, data_for_sf, sample_rate, **sf_kwargs)
    except Exception as e:
        raise EncodeError(
            f"soundfile 编码失败: {e}",
            hint="检查磁盘空间和文件权限",
        ) from e


# 每个格式的首选编码器列表（按优先级）
_FFMPEG_ENCODERS = {
    "mp3": ["libmp3lame", "mp3_mf"],
    "ogg": ["libvorbis", "opus", "vorbis"],
}
_encoder_cache = {}


def _detect_encoder(format_name: str) -> str:
    """自动检测系统可用的 ffmpeg 编码器（缓存结果）"""
    if format_name in _encoder_cache:
        return _encoder_cache[format_name]

    candidates = _FFMPEG_ENCODERS.get(format_name, [])
    if not candidates:
        _encoder_cache[format_name] = ""
        return ""

    try:
        proc = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        available = proc.stdout
    except Exception:
        _encoder_cache[format_name] = candidates[0]
        return candidates[0]

    for enc in candidates:
        if enc in available:
            _encoder_cache[format_name] = enc
            logger.info("ffmpeg 编码器检测: %s → %s", format_name, enc)
            return enc

    _encoder_cache[format_name] = candidates[0]
    logger.warning("ffmpeg 编码器 %s 均未检测到，尝试 %s", candidates, candidates[0])
    return candidates[0]


def _write_ffmpeg(
    path: str,
    data: np.ndarray,
    sample_rate: int,
    cfg: FormatConfig,
    subtype: str,
    bitrate: Optional[str],
):
    """通过 ffmpeg 写入 MP3/OGG（自动检测编码器）"""
    # 先写临时 WAV，再转码
    tmp_wav = path + ".tmp.wav." + uuid.uuid4().hex[:12]
    try:
        _write_sf(tmp_wav, data, sample_rate, FORMATS["wav"], "PCM_16", None)

        encoder = _detect_encoder(cfg.name)
        cmd = ["ffmpeg", "-y", "-i", tmp_wav, "-codec:a", encoder]
        if bitrate:
            cmd.extend(["-b:a", bitrate])

        # opus 编码器在部分 Windows 构建中标记为 experimental
        if encoder == "opus":
            cmd.extend(["-strict", "-2"])

        cmd.extend(["-f", cfg.name, path])

        _run_ffmpeg(cmd)
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def _run_ffmpeg(cmd: list, timeout: int = 300):
    """执行 ffmpeg 命令"""
    _check_ffmpeg()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise FfmpegNotFound()
    except subprocess.TimeoutExpired:
        raise EncodeError(
            f"ffmpeg 超时（>{timeout}s）",
            hint="可尝试降低采样率或比特率",
        )

    if proc.returncode != 0:
        err_lines = proc.stderr.strip().splitlines()
        brief = err_lines[-3:] if len(err_lines) > 3 else err_lines
        raise EncodeError(
            f"ffmpeg 编码失败 (code={proc.returncode}): {'; '.join(brief)}",
            hint="检查 ffmpeg 版本和编码器支持",
        )


def _check_ffmpeg():
    """检查 ffmpeg 是否可用"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise FfmpegNotFound() from e


def _check_disk_space(path: str, estimated_bytes: int):
    """写入前检查磁盘空间（预留 10% 余量）"""
    try:
        parent = os.path.dirname(os.path.abspath(path)) or "."
        usage = shutil.disk_usage(parent)
        needed = int(estimated_bytes * 1.1)
        if usage.free < needed:
            raise DiskSpaceError(
                f"磁盘空间不足: 需要约 {needed / 1024**3:.2f} GB，"
                f"可用 {usage.free / 1024**3:.2f} GB",
            )
    except OSError:
        pass  # 无法检查时静默跳过


def _atomic_write(
    path: str,
    data: np.ndarray,
    sample_rate: int,
    cfg: FormatConfig,
    subtype: str,
    bitrate: Optional[str],
    compression: Optional[int],
):
    """原子写入：写临时文件 → os.replace 覆盖"""
    tmp_path = path + ".tmp." + uuid.uuid4().hex[:12]
    try:
        _do_write(tmp_path, data, sample_rate, cfg, subtype, bitrate, compression)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def get_supported_formats() -> list:
    """返回格式列表（用于 API /models 端点）"""
    result = []
    for cfg in FORMATS.values():
        entry = {
            "format": cfg.name,
            "description": cfg.description,
            "extension": cfg.ext,
            "default_subtype": cfg.default_subtype,
        }
        if cfg.subtypes:
            entry["subtypes"] = list(cfg.subtypes)
        if cfg.supports_bitrate:
            entry["bitrate_range"] = list(cfg.bitrate_range)
            entry["default_bitrate"] = cfg.default_bitrate
        if cfg.supports_compression:
            entry["compression_range"] = list(cfg.compression_range)
            entry["default_compression"] = cfg.default_compression
        result.append(entry)
    return result
