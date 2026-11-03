import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from performance.runner.jsonl_writer import JsonlWriter
from performance.runner.logging_setup import bind_general_logger, bind_step_logger

API_TIMING_FILE = "api_timing.jsonl"
POD_RESOURCES_FILE = "pod_resources.jsonl"
CDP_EVENTS_DIR = "cdp_events"
STEP_LOG_FILE = "step.log"
GENERAL_LOG_FILE = "performance_test.log"

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _parse_log_level(level: str | int, default: int = logging.INFO) -> int:
    if isinstance(level, int):
        return level
    return _LOG_LEVELS.get(level.upper(), default)


@dataclass(frozen=True)
class SessionPaths:
    """一次运行的目录布局（由 runner 初始化时创建）"""

    run_id: str
    root: Path
    general_log: Path
    network_monitor: Path
    resource_monitor: Path

    def feature_network_dir(self, feature: str) -> Path:
        return self.network_monitor / feature

    def feature_resource_dir(self, feature: str) -> Path:
        return self.resource_monitor / feature


@dataclass
class SessionContext:
    """日志与数据写的统一入口"""

    paths: SessionPaths
    log_level: int = logging.INFO
    console_level: int = logging.WARNING
    enable_cdp_log: bool = False
    _general_logger: logging.Logger | None = field(default=None, repr=False)
    _step_loggers: dict[str, logging.Logger] = field(default_factory=dict, repr=False)
    _jsonl_writers: dict[str, JsonlWriter] = field(default_factory=dict, repr=False)

    def general_logger(self) -> logging.Logger:
        if self._general_logger is None:
            self.paths.root.mkdir(parents=True, exist_ok=True)
            self._general_logger = bind_general_logger(
                self.paths.general_log,
                log_level=self.log_level,
                console_level=self.console_level,
            )
        return self._general_logger

    def step_logger(
        self,
        monitor: Literal["network", "resource"],
        feature: str,
        *,
        rotate: bool = False,
    ) -> logging.Logger:
        key = f"{monitor}:{feature}"
        if key not in self._step_loggers:
            feature_dir = (
                self.paths.feature_network_dir(feature)
                if monitor == "network"
                else self.paths.feature_resource_dir(feature)
            )
            feature_dir.mkdir(parents=True, exist_ok=True)
            self._step_loggers[key] = bind_step_logger(
                feature_dir / STEP_LOG_FILE,
                monitor,
                feature,
                log_level=self.log_level,
                rotate=rotate,
            )
        return self._step_loggers[key]

    def _get_writer(self, key: str, path: Path) -> JsonlWriter:
        if key not in self._jsonl_writers:
            self._jsonl_writers[key] = JsonlWriter(path)
        return self._jsonl_writers[key]

    def network_timing_writer(self, feature: str) -> JsonlWriter:
        path = self.paths.feature_network_dir(feature) / API_TIMING_FILE
        return self._get_writer(f"timing:{feature}", path)

    def network_cdp_writer(self, feature: str, case: str) -> JsonlWriter:
        path = self.paths.feature_network_dir(feature) / CDP_EVENTS_DIR / f"{case}.jsonl"
        return self._get_writer(f"cdp:{feature}:{case}", path)

    def resource_pod_writer(self, feature: str) -> JsonlWriter:
        path = self.paths.feature_resource_dir(feature) / POD_RESOURCES_FILE
        return self._get_writer(f"pods:{feature}", path)


def create_session(
    work_dir: Path | None = None,
    run_id: str | None = None,
    log_level: str | int = "INFO",
    console_level: str | int = "WARNING",
    enable_cdp_log: bool = False,
) -> SessionContext:
    rid = run_id or datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    base = work_dir or Path.cwd()
    root = base / "performance_log" / f"performance_log-{rid}"
    paths = SessionPaths(
        run_id=rid,
        root=root,
        general_log=root / GENERAL_LOG_FILE,
        network_monitor=root / "network_monitor",
        resource_monitor=root / "resource_monitor",
    )
    paths.network_monitor.mkdir(parents=True, exist_ok=True)
    paths.resource_monitor.mkdir(parents=True, exist_ok=True)
    return SessionContext(
        paths=paths,
        log_level=_parse_log_level(log_level),
        console_level=_parse_log_level(console_level, logging.WARNING),
        enable_cdp_log=enable_cdp_log,
    )
