import json
import threading
from pathlib import Path


class JsonlWriter:
    """线程安全的 JSONL 追加写入器（不走 logging）"""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
