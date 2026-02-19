"""
src/data/preprocessing.py
==========================
Production-ready preprocessing module.

Responsibilities
----------------
- Load and validate all three datasets
- Handle missing values safely
- Engineer composite features (value_score, popularity_weighted_rating,
  price_bucket, normalized_price)
- One-hot encode categoricals, StandardScale numerics
- Save fitted preprocessor for reuse at inference time

Config-driven: All column names, bins, and paths come from config/config.yaml
Env-aware:     Works identically on local machines and SageMaker containers
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import get_logger, load_config, get_paths, get_feature_config

logger = get_logger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_datasets(
    paths: Dict[str, str],
    feat_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load products, users, and invoices CSVs and validate required columns.

    Parameters
    ----------
    paths    : dict from get_paths()
    feat_cfg : features section of config

    Returns
    -------
    (products_df, users_df, invoices_df)
    """
    logger.info("Loading datasets …")
    products_df = _load_csv(paths["products"], feat_cfg["required_product_columns"],  "products")
    users_df    = _load_csv(paths["users"],    feat_cfg["required_user_columns"],     "users")
    invoices_df = _load_csv(paths["invoices"], feat_cfg["required_invoice_columns"],  "invoices")

    logger.info(
        "Loaded  products=%d  users=%d  invoices=%d",
        len(products_df), len(users_df), len(invoices_df),
    )
    return products_df, users_df, invoices_df


def _load_csv(path: str, required_cols: List[str], name: str) -> pd.DataFrame:
    """Load one CSV and assert required columns exist."""
    if not Path(path).exists():
        raise FileNotFoundError(f"[{name}] File not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")

    logger.info("  %-12s  rows=%-6d  cols=%d", name, len(df), df.shape[1])
    return df


# =============================================================================
# Missing Value Handling
# =============================================================================

def handle_missing_values(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values in products.

    Strategy
    --------
    Numeric columns → median imputation
    Categorical     → mode imputation
    price <= 0      → dropped (cannot compute value_score safely)
    """
    logger.info("Handling missing values …")
    df = products_df.copy()

    numeric_cols     = ["price", "rating", "popularity", "discount_pct"]
    categorical_cols = ["category"]

    before = df.isnull().sum().sum()

    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            n_filled   = df[col].isnull().sum()
            df[col].fillna(median_val, inplace=True)
            logger.debug("  Imputed %d nulls in '%s' with median=%.4f", n_filled, col, median_val)

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
            n_filled = df[col].isnull().sum()
            df[col].fillna(mode_val, inplace=True)
            logger.debug("  Imputed %d nulls in '%s' with mode='%s'", n_filled, col, mode_val)

    invalid = df["price"] <= 0
    if invalid.sum() > 0:
        logger.warning("Dropping %d rows where price <= 0", invalid.sum())
        df = df[~invalid].reset_index(drop=True)

    after = df.isnull().sum().sum()
    logger.info("  Nulls before=%d  after=%d  rows_remaining=%d", before, after, len(df))
    return df


# =============================================================================
# Feature Engineering
# =============================================================================

def engineer_features(products_df: pd.DataFrame, feat_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Create composite features required for GMM training.

    New columns
    -----------
    value_score                : rating / price
    popularity_weighted_rating : rating * log1p(popularity)
    price_bucket               : categorical price tier (from config bins)
    normalized_price           : min-max scaled price in [0, 1]
    """
    logger.info("Engineering features …")
    df = products_df.copy()

    eng = feat_cfg.get("engineering", {})
    bins   = eng.get("price_bins",   [0, 50, 150, 300, 500, 99999])
    labels = eng.get("price_labels", ["budget", "economy", "mid", "premium", "luxury"])

    # value_score
    df["value_score"] = df["rating"] / (df["price"] + 1e-9)

    # popularity_weighted_rating
    df["popularity_weighted_rating"] = df["rating"] * np.log1p(df["popularity"])

    # price_bucket
    df["price_bucket"] = pd.cut(
        df["price"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype(str)

    # normalized_price
    p_min = df["price"].min()
    p_max = df["price"].max()
    df["normalized_price"] = (df["price"] - p_min) / (p_max - p_min + 1e-9)

    logger.info(
        "  Features added: value_score, popularity_weighted_rating, price_bucket, normalized_price"
    )
    return df


# =============================================================================
# Preprocessor (Encoder + Scaler)
# =============================================================================

def build_preprocessor(feat_cfg: Dict[str, Any]) -> ColumnTransformer:
    """
    Build a ColumnTransformer from feature config:
    - StandardScaler on numeric_features
    - OneHotEncoder on categorical_features
    """
    numeric_features     = feat_cfg["numeric_features"]
    categorical_features = feat_cfg["categorical_features"]

    numeric_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe     = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe,     categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_transform_save(
    products_df:  pd.DataFrame,
    feat_cfg:     Dict[str, Any],
    save_path:    str,
) -> Tuple[np.ndarray, ColumnTransformer]:
    """
    Fit preprocessor, transform products, and save to disk.

    Returns
    -------
    X            : np.ndarray  (n_products, n_features)
    preprocessor : fitted ColumnTransformer
    """
    logger.info("Fitting preprocessor …")
    preprocessor = build_preprocessor(feat_cfg)
    X = preprocessor.fit_transform(products_df)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    joblib.dump(preprocessor, save_path)

    logger.info("  Feature matrix shape: %s  |  Preprocessor saved → %s", X.shape, save_path)
    return X, preprocessor


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load a previously saved preprocessor from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Preprocessor not found: {path}")
    return joblib.load(path)


# =============================================================================
# Public Entry-Point
# =============================================================================

def run_preprocessing(
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Full preprocessing pipeline.  Config-driven and environment-aware.

    Returns
    -------
    X            : feature matrix ready for GMM training
    products_df  : cleaned + engineered product dataframe
    users_df     : users dataframe (unchanged)
    invoices_df  : invoices dataframe (unchanged)
    preprocessor : fitted ColumnTransformer
    """
    if cfg is None:
        cfg = load_config()

    paths    = get_paths(cfg)
    feat_cfg = get_feature_config(cfg)

    # Ensure output dirs exist
    os.makedirs(paths["artifacts_dir"],  exist_ok=True)
    os.makedirs(paths["processed_dir"], exist_ok=True)

    # Step 1: Load
    products_df, users_df, invoices_df = load_datasets(paths, feat_cfg)

    # Step 2: Clean
    products_df = handle_missing_values(products_df)

    # Step 3: Engineer
    products_df = engineer_features(products_df, feat_cfg)

    # Step 4: Fit + transform + save
    X, preprocessor = fit_transform_save(
        products_df=products_df,
        feat_cfg=feat_cfg,
        save_path=paths["preprocessor"],
    )

    # Persist engineered products for downstream modules
    products_df.to_csv(
        os.path.join(paths["processed_dir"], "products_engineered.csv"),
        index=False,
    )

    logger.info("Preprocessing complete.  X.shape=%s", X.shape)
    return X, products_df, users_df, invoices_df, preprocessor


# =============================================================================
# CLI Entry-Point
# =============================================================================
if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(description="Run GMM preprocessing pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    run_preprocessing(cfg)
    logger.info("Done.")
