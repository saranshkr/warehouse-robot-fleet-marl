from __future__ import annotations

from pathlib import Path


def make_run_dir(root: str | Path = "runs") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
