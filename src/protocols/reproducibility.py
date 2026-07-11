"""One reproducibility entrypoint for formal protocol seeds."""

from __future__ import annotations

import importlib
import os
import random
from typing import Dict

import numpy as np

from .experiment_protocol import FORMAL_SEEDS, ProtocolViolation


def set_protocol_seed(seed: int, *, include_frameworks: bool = True) -> Dict[str, bool]:
    normalized_seed = int(seed)
    if normalized_seed not in FORMAL_SEEDS:
        raise ProtocolViolation(f"formal seed must be one of {FORMAL_SEEDS}")
    os.environ["PYTHONHASHSEED"] = str(normalized_seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    status = {"python": True, "numpy": True, "tensorflow": False, "pytorch": False}
    if not include_frameworks:
        return status

    try:
        tensorflow = importlib.import_module("tensorflow")
        tensorflow.random.set_seed(normalized_seed)
        if hasattr(tensorflow.keras.utils, "set_random_seed"):
            tensorflow.keras.utils.set_random_seed(normalized_seed)
        status["tensorflow"] = True
    except ImportError:
        pass

    try:
        torch = importlib.import_module("torch")
        torch.manual_seed(normalized_seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalized_seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        status["pytorch"] = True
    except ImportError:
        pass
    return status
