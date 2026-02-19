"""
src/utils/config_loader.py
==========================
Loads config/config.yaml and resolves environment (local vs SageMaker).
All other modules import from here instead of hardcoding paths.

Usage
-----
    from src.utils.config_loader import load_config, get_env, get_paths
    cfg  = load_config()
    env  = get_env()          # "local" | "sagemaker"
    paths = get_paths()       # dict of all resolved file paths
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

import yaml

logger = logging.getLogger("config_loader")

# ---------------------------------------------------------------------------
# Locate project root (directory containing config/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/utils/ → project root
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load and return the YAML config as a nested dict.

    Parameters
    ----------
    config_path : str, optional
        Override path to config.yaml. Defaults to config/config.yaml.
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    logger.debug("Config loaded from %s", path)
    return cfg


def get_env(cfg: Dict[str, Any] = None) -> str:
    """
    Detect runtime environment.

    Resolution order:
    1. GMM_ENV environment variable ("local" | "sagemaker")
    2. config.yaml environment field (if not "auto")
    3. Auto-detect: SageMaker sets SM_MODEL_DIR to /opt/ml/model
    """
    # 1. Explicit override
    env_var = os.environ.get("GMM_ENV", "").lower()
    if env_var in ("local", "sagemaker"):
        return env_var

    # 2. Config file
    if cfg is None:
        cfg = load_config()
    config_env = str(cfg.get("environment", "auto")).lower()
    if config_env in ("local", "sagemaker"):
        return config_env

    # 3. Auto-detect via SageMaker env vars
    if os.environ.get("SM_MODEL_DIR", "").startswith("/opt/ml"):
        return "sagemaker"

    return "local"


def get_paths(cfg: Dict[str, Any] = None) -> Dict[str, str]:
    """
    Return resolved file paths for data and artifacts based on environment.

    Returns a flat dict:
        products, users, invoices, processed_dir,
        artifacts_dir, model, preprocessor, probabilities,
        products_clustered, training_params, evaluation_output
    """
    if cfg is None:
        cfg = load_config()

    env = get_env(cfg)

    if env == "sagemaker":
        sm_model_dir  = os.environ.get("SM_MODEL_DIR",  "/opt/ml/model")
        sm_output_dir = os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output")
        sm_input_dir  = os.environ.get("SM_INPUT_DIR",  "/opt/ml/input/data")

        data_dir      = sm_input_dir
        artifacts_dir = sm_model_dir
        output_dir    = sm_output_dir
    else:
        root          = _PROJECT_ROOT
        data_dir      = str(root)
        artifacts_dir = str(root / cfg["artifacts"]["local_dir"])
        output_dir    = str(root / cfg["evaluation"]["output_dir"])

    art = cfg["artifacts"]
    data_cfg = cfg["data"]["local"]

    def _resolve(base_dir: str, relative: str) -> str:
        """If relative is already absolute, return as-is; else join with base."""
        p = Path(relative)
        if p.is_absolute():
            return str(p)
        return str(Path(base_dir) / p)

    if env == "sagemaker":
        products_path  = os.path.join(data_dir, "products", "products_raw.csv")
        users_path     = os.path.join(data_dir, "users",    "users.csv")
        invoices_path  = os.path.join(data_dir, "invoices", "invoices.csv")
    else:
        root = _PROJECT_ROOT
        products_path  = str(root / data_cfg["products"])
        users_path     = str(root / data_cfg["users"])
        invoices_path  = str(root / data_cfg["invoices"])

    return {
        # Data
        "products":           products_path,
        "users":              users_path,
        "invoices":           invoices_path,
        "processed_dir":      os.path.join(artifacts_dir, "processed"),
        # Artifacts
        "artifacts_dir":      artifacts_dir,
        "model":              os.path.join(artifacts_dir, art["model_file"]),
        "preprocessor":       os.path.join(artifacts_dir, art["preprocessor_file"]),
        "probabilities":      os.path.join(artifacts_dir, art["probabilities_file"]),
        "products_clustered": os.path.join(artifacts_dir, art["products_clustered"]),
        "training_params":    os.path.join(artifacts_dir, art["training_params"]),
        # Evaluation
        "output_dir":         output_dir,
        "evaluation_output":  os.path.join(output_dir, cfg["evaluation"]["output_file"]),
    }


def get_feature_config(cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return the features section of config."""
    if cfg is None:
        cfg = load_config()
    return cfg["features"]


def get_training_config(cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return the training section of config."""
    if cfg is None:
        cfg = load_config()
    return cfg["training"]


def get_recommendation_config(cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return the recommendation section of config."""
    if cfg is None:
        cfg = load_config()
    return cfg["recommendation"]


def get_sagemaker_config(cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return the sagemaker section of config."""
    if cfg is None:
        cfg = load_config()
    return cfg["sagemaker"]
