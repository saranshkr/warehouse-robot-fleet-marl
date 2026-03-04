from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class MetricsLogger:
    """
    Simple, robust metrics logger:
      - CSV with a fixed schema (prevents DictWriter header/key mismatches)
      - TensorBoard (optional), logs any numeric metric
      - Ignores any extra CSV keys not in schema (extrasaction="ignore")

    NOTE:
      CSV headers cannot be safely expanded mid-file without rewriting the file.
      So we keep a fixed list of columns and rely on TensorBoard for all ad-hoc metrics.
    """

    # Add/remove fields here as you standardize what you want in metrics.csv
    DEFAULT_FIELDS = [
        "step",
        "time",
        # Common training metrics (feel free to extend)
        "train/episode_return",
        "train/episode_len",
        "train/deliveries",
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "train/approx_kl",
        "train/clip_frac",
        "train/explained_var",
        # Evaluation (if you log these from train loop)
        "eval/mean_return",
        "eval/std_return",
        "eval/mean_deliveries",
        "eval/pick_rate",
    ]

    def __init__(
        self,
        run_dir: Path,
        use_tensorboard: bool = True,
        *,
        csv_fields: Optional[list[str]] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.run_dir / "metrics.csv"
        # Append mode so resume works; header is written only if file is empty.
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")

        self._fieldnames = list(csv_fields) if csv_fields is not None else list(self.DEFAULT_FIELDS)
        if "step" not in self._fieldnames:
            self._fieldnames.insert(0, "step")
        if "time" not in self._fieldnames:
            self._fieldnames.insert(1, "time")

        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=self._fieldnames,
            extrasaction="ignore",
        )

        # Write header only for new file
        if self._csv_file.tell() == 0:
            self._writer.writeheader()
            self._csv_file.flush()

        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception:
                self.tb = None

    def write_hparams(self, cfg: Dict[str, Any]) -> None:
        (self.run_dir / "config.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        # Ensure "step" and "time" always exist in CSV
        row: Dict[str, Any] = {"step": int(step), "time": float(time.time())}
        row.update(metrics or {})

        # Write CSV row (extras ignored)
        try:
            self._writer.writerow(row)
            self._csv_file.flush()
        except Exception as e:
            raise RuntimeError(f"Failed to write metrics to CSV at {self.csv_path}: {e}") from e

        # TensorBoard: log any numeric scalars
        if self.tb is not None and metrics:
            for k, v in metrics.items():
                if isinstance(v, (bool, str, bytes, dict, list, tuple)):
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)):
                    self.tb.add_scalar(k, float(v), global_step=int(step))

    def close(self) -> None:
        try:
            if self.tb is not None:
                self.tb.close()
        finally:
            self._csv_file.close()


def make_run_dir(root: str, experiment: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(root) / experiment / ts


def save_text(run_dir: Path, name: str, text: str) -> None:
    (Path(run_dir) / name).write_text(text, encoding="utf-8")


def save_json(run_dir: Path, name: str, data: Dict[str, Any]) -> None:
    (Path(run_dir) / name).write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
