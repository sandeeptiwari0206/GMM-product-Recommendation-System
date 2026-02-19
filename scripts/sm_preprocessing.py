"""
scripts/sm_preprocessing.py
============================
Self-contained SageMaker Processing script.
No imports from src.* — everything is inline.
SageMaker runs this directly; all dependencies are pre-installed in the sklearn container.

Container paths
---------------
Input  products : /opt/ml/processing/input/products/products_raw.csv
Input  users    : /opt/ml/processing/input/users/users.csv
Input  invoices : /opt/ml/processing/input/invoices/invoices.csv
Output          : /opt/ml/processing/output/
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("sm_preprocessing")

OUTPUT_DIR = "/opt/ml/processing/output"


def load_and_validate(path: str, required_cols: list, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{name}] Not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")
    logger.info("[%s] rows=%d  cols=%d", name, len(df), df.shape[1])
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["price", "rating", "popularity", "discount_pct"]:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    if "category" in df.columns:
        df["category"].fillna(df["category"].mode()[0] if not df["category"].mode().empty else "unknown", inplace=True)
    invalid = df["price"] <= 0
    if invalid.sum():
        logger.warning("Dropping %d rows with price <= 0", invalid.sum())
        df = df[~invalid].reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["value_score"]                = df["rating"] / (df["price"] + 1e-9)
    df["popularity_weighted_rating"] = df["rating"] * np.log1p(df["popularity"])
    bins   = [0, 50, 150, 300, 500, 99999]
    labels = ["budget", "economy", "mid", "premium", "luxury"]
    df["price_bucket"]    = pd.cut(df["price"], bins=bins, labels=labels, right=False, include_lowest=True).astype(str)
    p_min = df["price"].min(); p_max = df["price"].max()
    df["normalized_price"] = (df["price"] - p_min) / (p_max - p_min + 1e-9)
    return df


def build_and_fit_preprocessor(df: pd.DataFrame):
    numeric_features     = ["price", "rating", "popularity", "discount_pct",
                             "value_score", "popularity_weighted_rating", "normalized_price"]
    categorical_features = ["category", "price_bucket"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]),  numeric_features),
        ("cat", Pipeline([("enc",    OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
    ], remainder="drop")

    X = preprocessor.fit_transform(df)
    logger.info("Feature matrix shape: %s", X.shape)
    return X, preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", default="/opt/ml/processing/input/products/products_raw.csv")
    parser.add_argument("--users",    default="/opt/ml/processing/input/users/users.csv")
    parser.add_argument("--invoices", default="/opt/ml/processing/input/invoices/invoices.csv")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading datasets...")
    products_df = load_and_validate(args.products, ["product_id","category","price","rating","popularity","discount_pct"], "products")
    users_df    = load_and_validate(args.users,    ["user_id","segment","preferred_category"], "users")
    invoices_df = load_and_validate(args.invoices, ["invoice_id","user_id","product_id","price","purchase_date"], "invoices")

    logger.info("Cleaning and engineering features...")
    products_df = handle_missing(products_df)
    products_df = engineer_features(products_df)

    logger.info("Fitting preprocessor...")
    X, preprocessor = build_and_fit_preprocessor(products_df)

    logger.info("Saving artifacts to %s ...", OUTPUT_DIR)
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.joblib"))
    np.save(os.path.join(OUTPUT_DIR, "feature_matrix.npy"), X)
    products_df.to_csv(os.path.join(OUTPUT_DIR, "products_engineered.csv"), index=False)
    users_df.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)
    invoices_df.to_csv(os.path.join(OUTPUT_DIR, "invoices.csv"), index=False)

    logger.info("Preprocessing complete. X.shape=%s", X.shape)


if __name__ == "__main__":
    main()
