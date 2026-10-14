"""Write a text log next to stdout. Used by every run script."""

from __future__ import annotations

import sys
from pathlib import Path


class Tee:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> int:
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


def start_log(path: Path) -> Tee:
    tee = Tee(path)
    sys.stdout = tee
    print(f"LOG: {path}", flush=True)
    return tee
