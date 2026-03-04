from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """
    Parse `key=value` pairs into nested dicts, e.g. algo.lr_actor=3e-4
    """
    out: Dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Override must be key=value, got: {p}")
        k, v = p.split("=", 1)
        # try infer numeric/bool
        val: Any = v
        if v.lower() in {"true", "false"}:
            val = v.lower() == "true"
        else:
            try:
                if "." in v or "e" in v.lower():
                    val = float(v)
                else:
                    val = int(v)
            except ValueError:
                val = v
        cur = out
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = val
    return out
