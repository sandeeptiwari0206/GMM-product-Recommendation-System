"""
src/models/evaluation.py
========================
Config-driven GMM evaluation module.

Metrics
-------
- BIC, AIC, Log-Likelihood
- Silhouette Score (sampled for performance)
- Davies-Bouldin Index
- Cluster Probability Entropy
- Per-cluster size, price, rating statistics

Outputs evaluation.json for SageMaker Pipeline Condition Step
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from src.utils import get_logger, load_config, get_paths
from src.utils.config_loader import get_feature_config

logger = get_logger(__name__)


# =============================================================================
# Individual Metrics
# =============================================================================

def compute_bic(model: GaussianMixture, X: np.ndarray) -> float:
    return float(model.bic(X))


def compute_aic(model: GaussianMixture, X: np.ndarray) -> float:
    return float(model.aic(X))


def compute_log_likelihood(model: GaussianMixture, X: np.ndarray) -> float:
    return float(model.score(X))


def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    sample_limit: int = 5000,
) -> float:
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        logger.warning("Silhouette skipped: only %d unique cluster", n_unique)
        return -1.0

    if len(X) > sample_limit:
        idx = np.random.choice(len(X), sample_limit, replace=False)
        return float(silhouette_score(X[idx], labels[idx]))
    return float(silhouette_score(X, labels))


def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return float("inf")
    return float(davies_bouldin_score(X, labels))


def compute_entropy_stats(prob_matrix: np.ndarray) -> Dict[str, float]:
    """
    Per-product probability vector entropy.
    High mean entropy → products spread softly across clusters (good for diverse recs).
    """
    entropies = np.apply_along_axis(scipy_entropy, axis=1, arr=prob_matrix)
    return {
        "mean_entropy":    round(float(np.mean(entropies)),   4),
        "median_entropy":  round(float(np.median(entropies)), 4),
        "std_entropy":     round(float(np.std(entropies)),    4),
        "max_possible":    round(float(np.log(prob_matrix.shape[1])), 4),
    }


def compute_per_cluster_stats(
    prob_matrix:  np.ndarray,
    labels:       np.ndarray,
    products_df:  Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Per-cluster size and optional product-level statistics."""
    n_clusters = prob_matrix.shape[1]
    stats      = []

    for k in range(n_clusters):
        mask = labels == k
        size = int(mask.sum())
        stat: Dict[str, Any] = {
            "cluster_id":   k,
            "size":         size,
            "pct_of_data":  round(size / max(len(labels), 1) * 100, 2),
            "mean_membership_prob": round(float(prob_matrix[:, k].mean()), 4),
        }

        if products_df is not None and size > 0:
            sub = products_df[mask]
            for col in ["price", "rating", "popularity", "discount_pct"]:
                if col in sub.columns:
                    stat[f"mean_{col}"] = round(float(sub[col].mean()), 4)

            if "category" in sub.columns:
                top_cats = sub["category"].value_counts().head(3).to_dict()
                stat["top_categories"] = {str(k2): int(v) for k2, v in top_cats.items()}

        stats.append(stat)

    return stats


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def run_evaluation(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load trained artifacts and compute all evaluation metrics.

    Saves evaluation.json and returns the metrics dict.
    """
    if cfg is None:
        cfg = load_config()

    paths    = get_paths(cfg)
    feat_cfg = get_feature_config(cfg)
    eval_cfg = cfg.get("evaluation", {})
    sample_limit = eval_cfg.get("silhouette_sample_limit", 5000)

    os.makedirs(paths["output_dir"], exist_ok=True)

    # --- Load artifacts ---
    logger.info("Loading artifacts from %s …", paths["artifacts_dir"])
    model        = joblib.load(paths["model"])
    preprocessor = joblib.load(paths["preprocessor"])
    prob_matrix  = np.load(paths["probabilities"])
    products_df  = pd.read_csv(paths["products_clustered"])

    # --- Reconstruct feature matrix ---
    logger.info("Transforming products …")
    X      = preprocessor.transform(products_df)
    labels = model.predict(X)

    logger.info(
        "Evaluating GMM  n_components=%d  cov_type=%s  n_products=%d",
        model.n_components, model.covariance_type, len(products_df),
    )

    # --- Compute metrics ---
    bic     = compute_bic(model, X)
    aic     = compute_aic(model, X)
    log_ll  = compute_log_likelihood(model, X)
    sil     = compute_silhouette(X, labels, sample_limit)
    db_idx  = compute_davies_bouldin(X, labels)
    entropy = compute_entropy_stats(prob_matrix)
    cluster_stats = compute_per_cluster_stats(prob_matrix, labels, products_df)

    results: Dict[str, Any] = {
        "model": {
            "n_components":    model.n_components,
            "covariance_type": model.covariance_type,
            "converged":       bool(model.converged_),
            "n_iter":          int(model.n_iter_),
        },
        "metrics": {
            "bic":                  round(bic,    2),
            "aic":                  round(aic,    2),
            "log_likelihood":       round(log_ll, 6),
            "silhouette_score":     round(sil,    4),
            "davies_bouldin_index": round(db_idx, 4),
            "cluster_entropy":      entropy,
        },
        "cluster_statistics": cluster_stats,
    }

    out_path = paths["evaluation_output"]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    _log_summary(results)
    logger.info("Evaluation saved → %s", out_path)
    return results


def _log_summary(r: Dict[str, Any]) -> None:
    m = r["metrics"]
    logger.info("=" * 60)
    logger.info("  BIC                  : %14.2f",  m["bic"])
    logger.info("  AIC                  : %14.2f",  m["aic"])
    logger.info("  Log-Likelihood       : %14.6f",  m["log_likelihood"])
    logger.info("  Silhouette Score     : %14.4f",  m["silhouette_score"])
    logger.info("  Davies-Bouldin Index : %14.4f",  m["davies_bouldin_index"])
    logger.info("  Mean Entropy         : %14.4f",  m["cluster_entropy"]["mean_entropy"])
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(description="Evaluate trained GMM model")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    run_evaluation(cfg)
