from __future__ import annotations

import logging
import os
from typing import Literal


VerboseMode = Literal["summary", "full"]

_VALID_MODES = {"summary", "full"}
_current_mode: str = os.getenv("MSML_TL_VERBOSE_MODE", "summary").strip().lower() or "summary"
if _current_mode not in _VALID_MODES:
    _current_mode = "summary"



def set_verbose_mode(verbose_mode: str) -> str:
    global _current_mode
    mode = str(verbose_mode).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported verbose_mode: {verbose_mode}")
    _current_mode = mode
    os.environ["MSML_TL_VERBOSE_MODE"] = mode
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" if mode == "summary" else "0"
    return mode



def get_verbose_mode() -> str:
    return _current_mode



def is_summary_mode() -> bool:
    return _current_mode == "summary"



def keras_verbose() -> int:
    return 0 if is_summary_mode() else 1



def log_level_name() -> str:
    return "WARNING" if is_summary_mode() else "INFO"



def apply_logging_level() -> None:
    level = logging.WARNING if is_summary_mode() else logging.INFO
    logger = logging.getLogger("experiment")
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
