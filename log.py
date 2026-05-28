import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv

# 加载 .env 配置
load_dotenv()

# 配置（优先级: .env 文件 > 环境变量 > 默认值）
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_DIR_APP = os.path.join(LOG_DIR, "app")
LOG_DIR_ERROR = os.path.join(LOG_DIR, "error")
LOG_LEVEL = os.getenv("LOG_LEVEL").upper()
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES"))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT"))


# 日志格式
CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s"
ERROR_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class LoggerManager:
    """日志管理器，统一管理所有日志文件"""

    _instances = {}

    def __init__(self):
        os.makedirs(LOG_DIR_APP, exist_ok=True)
        os.makedirs(LOG_DIR_ERROR, exist_ok=True)

    def get_logger(self, name: str = "app") -> logging.Logger:
        """获取或创建一个 logger"""
        if name in self._instances:
            return self._instances[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        # 避免重复添加 handler
        if logger.handlers:
            return logger

        today = datetime.now().strftime("%Y-%m-%d")

        # ── 控制台 Handler ──
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)

        # ── 全部日志 Handler (INFO+) ──
        app_log = os.path.join(LOG_DIR_APP, f"{today}.log")
        app_handler = RotatingFileHandler(
            app_log,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        logger.addHandler(app_handler)

        # ── 错误日志 Handler (ERROR+) ──
        err_log = os.path.join(LOG_DIR_ERROR, f"{today}.log")
        err_handler = RotatingFileHandler(
            err_log,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(logging.Formatter(ERROR_FILE_FORMAT, DATE_FORMAT))
        logger.addHandler(err_handler)

        self._instances[name] = logger
        return logger

    def get_log_path(self) -> str:
        """获取今天的全部日志文件路径"""
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(LOG_DIR_APP, f"{today}.log")

    def get_error_log_path(self) -> str:
        """获取今天的错误日志文件路径"""
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(LOG_DIR_ERROR, f"{today}.log")


# 全局单例
_manager = LoggerManager()


def get_logger(name: str = "app") -> logging.Logger:
    """快捷获取 logger"""
    return _manager.get_logger(name)


def get_log_path() -> str:
    """获取当前全部日志文件路径"""
    return _manager.get_log_path()


def get_error_log_path() -> str:
    """获取当前错误日志文件路径"""
    return _manager.get_error_log_path()


def list_logs() -> list:
    """列出所有日志文件"""
    result = []
    for sub_dir in ["app", "error"]:
        path = os.path.join(LOG_DIR, sub_dir)
        if not os.path.exists(path):
            continue
        for f in sorted(os.listdir(path), reverse=True):
            if not f.endswith(".log"):
                continue
            fpath = os.path.join(path, f)
            result.append({
                "name": f"{sub_dir}/{f}",
                "size": _format_size(os.path.getsize(fpath)),
                "path": fpath,
            })
    return result


def _format_size(bytes_size: int) -> str:
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f}KB"
    else:
        return f"{bytes_size / 1024 / 1024:.1f}MB"
