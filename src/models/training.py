"""
src/models/training.py
======================
Config-driven GMM training with full hyperparameter grid search.

- Trains all (n_components, covariance_type) combinations
- Selects best model by BIC (or AIC, configurable)
- Saves model, cluster probability matrix, and all artifacts
- SageMaker + local compatible
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from src.utils import get_logger, load_config, get_paths, get_training_config
from src.data.preprocessing import run_preprocessing

logger = get_logger(__name__)


# =============================================================================
# Single GMM fit
# =============================================================================

def train_single_gmm(
    X:               np.ndarray,
    n_components:    int,
    covariance_type: str,
    random_state:    int = 42,
    max_iter:        int = 300,
    n_init:          int = 10,
) -> GaussianMixture:
    """Fit a single GaussianMixture model, suppressing convergence warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            warm_start=False,
        )
        gmm.fit(X)

    if not gmm.converged_:
        logger.warning(
            "  [!] NOT converged  n_components=%d  cov_type=%s",
            n_components, covariance_type,
        )
    return gmm


# =============================================================================
# Hyperparameter search
# =============================================================================

def hyperparameter_search(
    X:                  np.ndarray,
    train_cfg:          Dict[str, Any],
) -> Tuple[GaussianMixture, Dict[str, Any], List[Dict]]:
    """
    Grid search over (n_components, covariance_type).
    Selection metric is configurable: "bic" or "aic".

    Returns
    -------
    best_model  : fitted GaussianMixture
    best_params : dict of winning hyperparameters
    all_results : list of dicts for every combination tried
    """
    n_min       = train_cfg.get("n_components_min",  3)
    n_max       = train_cfg.get("n_components_max",  7)
    cov_types   = train_cfg.get("covariance_types",  ["full", "tied", "diag", "spherical"])
    max_iter    = train_cfg.get("max_iter",           300)
    n_init      = train_cfg.get("n_init",             10)
    rand_state  = train_cfg.get("random_state",       42)
    metric      = train_cfg.get("selection_metric",   "bic").lower()

    n_range = list(range(n_min, n_max + 1))
    total   = len(n_range) * len(cov_types)

    logger.info(
        "Hyperparameter search started  n_components=%s  cov_types=%s  metric=%s  total_fits=%d",
        n_range, cov_types, metric, total,
    )

    best_score  = np.inf
    best_model  = None
    best_params : Dict[str, Any] = {}
    all_results : List[Dict]     = []
    run_no      = 0

    for cov_type in cov_types:
        for n_comp in n_range:
            run_no += 1
            try:
                gmm    = train_single_gmm(X, n_comp, cov_type, rand_state, max_iter, n_init)
                bic    = float(gmm.bic(X))
                aic    = float(gmm.aic(X))
                log_ll = float(gmm.score(X))
                score  = bic if metric == "bic" else aic

                all_results.append({
                    "run":             run_no,
                    "n_components":    n_comp,
                    "covariance_type": cov_type,
                    "bic":             bic,
                    "aic":             aic,
                    "log_likelihood":  log_ll,
                    "converged":       bool(gmm.converged_),
                    "n_iter":          int(gmm.n_iter_),
                })

                logger.info(
                    "  [%2d/%2d]  n_comp=%-2d  cov=%-10s  BIC=%12.2f  AIC=%12.2f  converged=%s",
                    run_no, total, n_comp, cov_type, bic, aic, gmm.converged_,
                )

                if score < best_score:
                    best_score  = score
                    best_model  = gmm
                    best_params = {
                        "n_components":    n_comp,
                        "covariance_type": cov_type,
                        "bic":             bic,
                        "aic":             aic,
                        "log_likelihood":  log_ll,
                        "selection_metric": metric,
                        "selection_score":  score,
                    }

            except Exception as exc:
                logger.error(
                    "  [%2d/%2d]  FAILED  n_comp=%d  cov=%s  error=%s",
                    run_no, total, n_comp, cov_type, exc,
                )

    if best_model is None:
        raise RuntimeError("No GMM model converged during hyperparameter search.")

    logger.info(
        "Best model → n_components=%d  cov_type=%s  BIC=%.2f",
        best_params["n_components"], best_params["covariance_type"], best_params["bic"],
    )
    return best_model, best_params, all_results


# =============================================================================
# Cluster probability matrix
# =============================================================================

def compute_cluster_probabilities(model: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """
    Compute soft cluster membership probabilities for all products.

    Returns
    -------
    prob_matrix : np.ndarray  shape (n_products, n_clusters)
    """
    logger.info("Computing cluster probability matrix …")
    prob_matrix = model.predict_proba(X)
    logger.info("  prob_matrix shape: %s  (min=%.4f  max=%.4f)",
                prob_matrix.shape, prob_matrix.min(), prob_matrix.max())
    return prob_matrix


# =============================================================================
# Artifact saving
# =============================================================================

def save_artifacts(
    model:          GaussianMixture,
    preprocessor:   Any,
    prob_matrix:    np.ndarray,
    products_df:    pd.DataFrame,
    best_params:    Dict[str, Any],
    all_results:    List[Dict],
    paths:          Dict[str, str],
) -> None:
    """
    Save all training artifacts to the artifacts directory.

    Files
    -----
    model.joblib                 – fitted GaussianMixture
    preprocessor.joblib          – fitted ColumnTransformer
    cluster_probabilities.npy    – (n_products, n_clusters)
    products_with_clusters.csv   – products with cluster assignments
    training_params.json         – hyperparameter search results
    """
    art_dir = paths["artifacts_dir"]
    os.makedirs(art_dir, exist_ok=True)

    # Model
    joblib.dump(model, paths["model"])
    logger.info("  Saved model        → %s", paths["model"])

    # Preprocessor
    joblib.dump(preprocessor, paths["preprocessor"])
    logger.info("  Saved preprocessor → %s", paths["preprocessor"])

    # Probability matrix
    np.save(paths["probabilities"], prob_matrix)
    logger.info("  Saved prob_matrix  → %s", paths["probabilities"])

    # Products with cluster info
    products_out                    = products_df.copy()
    products_out["primary_cluster"] = model.predict(preprocessor.transform(products_df))
    for k in range(prob_matrix.shape[1]):
        products_out[f"cluster_{k}_prob"] = prob_matrix[:, k]

    products_out.to_csv(paths["products_clustered"], index=False)
    logger.info("  Saved clustered products → %s", paths["products_clustered"])

    # Training params
    params_out = {
        "best_params":  best_params,
        "all_results":  all_results,
    }
    with open(paths["training_params"], "w") as f:
        json.dump(params_out, f, indent=2, default=str)
    logger.info("  Saved training_params → %s", paths["training_params"])


# =============================================================================
# Public Entry-Point
# =============================================================================

def run_training(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Full end-to-end training pipeline.  Config-driven and environment-aware.

    Returns
    -------
    best_params : dict with winning hyperparameters
    """
    if cfg is None:
        cfg = load_config()

    paths     = get_paths(cfg)
    train_cfg = get_training_config(cfg)

    # 1. Preprocess
    X, products_df, users_df, invoices_df, preprocessor = run_preprocessing(cfg)

    # 2. Hyperparameter search
    best_model, best_params, all_results = hyperparameter_search(X, train_cfg)

    # 3. Cluster probabilities
    prob_matrix = compute_cluster_probabilities(best_model, X)

    # 4. Save all artifacts
    save_artifacts(
        model=best_model,
        preprocessor=preprocessor,
        prob_matrix=prob_matrix,
        products_df=products_df,
        best_params=best_params,
        all_results=all_results,
        paths=paths,
    )

    logger.info("Training complete.")
    return best_params


# =============================================================================
# CLI / SageMaker entry-point
# =============================================================================
if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(description="Train GMM recommendation model")
    parser.add_argument("--config",             default=None)
    # Allow SageMaker to pass hyperparameters directly
    parser.add_argument("--n-components-min",   type=int,   default=None)
    parser.add_argument("--n-components-max",   type=int,   default=None)
    parser.add_argument("--covariance-types",   default=None, help="Comma-separated")
    parser.add_argument("--random-state",       type=int,   default=None)
    parser.add_argument("--selection-metric",   default=None, choices=["bic", "aic"])
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    # CLI overrides take priority over config file
    t = cfg["training"]
    if args.n_components_min  is not None: t["n_components_min"]  = args.n_components_min
    if args.n_components_max  is not None: t["n_components_max"]  = args.n_components_max
    if args.covariance_types  is not None: t["covariance_types"]  = [c.strip() for c in args.covariance_types.split(",")]
    if args.random_state      is not None: t["random_state"]      = args.random_state
    if args.selection_metric  is not None: t["selection_metric"]  = args.selection_metric

    best = run_training(cfg)
    logger.info("Best params: %s", best)
