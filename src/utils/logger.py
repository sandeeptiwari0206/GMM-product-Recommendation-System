"""
src/utils/logger.py
===================
Centralised logging setup. Import get_logger() in every module.

Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger.

    Level resolution order:
    1. `level` argument
    2. GMM_LOG_LEVEL environment variable
    3. INFO (default)
    """
    log_level_str = level or os.environ.get("GMM_LOG_LEVEL", "INFO")
    log_level     = getattr(logging, log_level_str.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    logger.setLevel(log_level)
    logger.propagate = False
    return logger
