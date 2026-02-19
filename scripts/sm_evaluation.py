"""
scripts/sm_evaluation.py
=========================
Self-contained SageMaker Processing script for GMM evaluation.
No imports from src.* — everything inline.

Container paths
---------------
Input  model : /opt/ml/processing/input/model/  (extracted model.tar.gz)
Output       : /opt/ml/processing/output/evaluation.json
"""

import json
import logging
import os
import sys
import tarfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("sm_evaluation")

MODEL_DIR  = "/opt/ml/processing/input/model"
OUTPUT_DIR = "/opt/ml/processing/output"


def extract_model_tar(model_dir: str) -> str:
    """Extract model.tar.gz if present. Returns the extraction directory."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        extract_dir = os.path.join(model_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)
        logger.info("Extracted model.tar.gz → %s", extract_dir)
        return extract_dir
    # Already extracted
    return model_dir


def find_file(directory: str, filename: str) -> str:
    """Recursively find a file in directory tree."""
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found under {directory}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract model artifacts
    artifact_dir = extract_model_tar(MODEL_DIR)
    logger.info("Looking for artifacts in: %s", artifact_dir)
    for root, dirs, files in os.walk(artifact_dir):
        logger.info("  %s: %s", root, files)

    # Load artifacts
    model_path  = find_file(artifact_dir, "model.joblib")
    prep_path   = find_file(artifact_dir, "preprocessor.joblib")
    probs_path  = find_file(artifact_dir, "cluster_probabilities.npy")
    prods_path  = find_file(artifact_dir, "products_with_clusters.csv")

    model        = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    prob_matrix  = np.load(probs_path)
    products_df  = pd.read_csv(prods_path)

    logger.info("Model: n_components=%d  cov_type=%s", model.n_components, model.covariance_type)

    # Reconstruct feature matrix
    X      = preprocessor.transform(products_df)
    labels = model.predict(X)

    # --- Metrics ---
    bic    = float(model.bic(X))
    aic    = float(model.aic(X))
    log_ll = float(model.score(X))

    n_unique = len(np.unique(labels))
    if n_unique >= 2:
        sample = min(5000, len(X))
        idx    = np.random.choice(len(X), sample, replace=False)
        sil    = float(silhouette_score(X[idx], labels[idx]))
        db     = float(davies_bouldin_score(X, labels))
    else:
        sil, db = -1.0, float("inf")

    entropies = np.apply_along_axis(scipy_entropy, axis=1, arr=prob_matrix)
    entropy_stats = {
        "mean_entropy":   round(float(np.mean(entropies)),   4),
        "median_entropy": round(float(np.median(entropies)), 4),
        "max_possible":   round(float(np.log(prob_matrix.shape[1])), 4),
    }

    # Per-cluster stats
    cluster_stats = []
    for k in range(prob_matrix.shape[1]):
        mask = labels == k
        cluster_stats.append({
            "cluster_id":   k,
            "size":         int(mask.sum()),
            "pct_of_data":  round(float(mask.sum()) / len(labels) * 100, 2),
        })

    results = {
        "model": {
            "n_components":    model.n_components,
            "covariance_type": model.covariance_type,
            "converged":       bool(model.converged_),
        },
        "metrics": {
            "bic":                  round(bic,    2),
            "aic":                  round(aic,    2),
            "log_likelihood":       round(log_ll, 6),
            "silhouette_score":     round(sil,    4),
            "davies_bouldin_index": round(db,     4),
            "cluster_entropy":      entropy_stats,
        },
        "cluster_statistics": cluster_stats,
    }

    out_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("BIC=%.2f  AIC=%.2f  Silhouette=%.4f  DB=%.4f", bic, aic, sil, db)
    logger.info("Evaluation saved → %s", out_path)


if __name__ == "__main__":
    main()
