import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(component)s] [%(feature)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

STEP_LOG_MAX_BYTES = 10 * 1024 * 1024
STEP_LOG_BACKUP_COUNT = 5


class _ContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "component"):
            record.component = "-"
        if not hasattr(record, "feature"):
            record.feature = "-"
        return super().format(record)


def _make_formatter() -> _ContextFormatter:
    return _ContextFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT)


def _configure_logger(logger: logging.Logger, *handlers: logging.Handler) -> logging.Logger:
    if logger.handlers:
        return logger
    for handler in handlers:
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def bind_general_logger(
    log_path: Path,
    *,
    log_level: int = logging.INFO,
    console_level: int = logging.WARNING,
    component: str = "runner",
) -> logging.Logger:
    """绑定主日志：performance_test.log（文件）+ stdout。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("performance_test")
    if logger.handlers:
        return SessionLoggerAdapter(logger, {"component": component, "feature": "-"})

    logger.setLevel(min(log_level, console_level))
    formatter = _make_formatter()

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(formatter)

    _configure_logger(logger, file_handler, stream_handler)
    return SessionLoggerAdapter(logger, {"component": component, "feature": "-"})


def bind_step_logger(
    log_path: Path,
    monitor: Literal["network", "resource"],
    feature: str,
    *,
    log_level: int = logging.INFO,
    rotate: bool = False,
) -> logging.Logger:
    """绑定步骤日志：step.log（仅文件，不上 stdout）。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = f"{monitor}_{feature}_step"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return SessionLoggerAdapter(logger, {"component": monitor, "feature": feature})

    logger.setLevel(log_level)
    formatter = _make_formatter()

    if rotate:
        file_handler: logging.Handler = RotatingFileHandler(
            log_path,
            maxBytes=STEP_LOG_MAX_BYTES,
            backupCount=STEP_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")

    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    _configure_logger(logger, file_handler)
    return SessionLoggerAdapter(logger, {"component": monitor, "feature": feature})


class SessionLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("component", self.extra.get("component", "-"))
        extra.setdefault("feature", self.extra.get("feature", "-"))
        return msg, kwargs
