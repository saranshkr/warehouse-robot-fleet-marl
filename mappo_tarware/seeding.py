from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

def set_global_seeds(seed: int, deterministic_torch: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # Makes results more reproducible at a small perf cost.
        torch.use_deterministic_algorithms(False)  # allow some ops; true can break some kernels
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
