"""
scripts/sm_training.py
=======================
Self-contained SageMaker Training script.
No imports from src.* — everything inline.
SageMaker SKLearn container calls this as the entry point.

SageMaker Training paths
------------------------
Input  : /opt/ml/input/data/products/   (preprocessed artifacts from step 1)
Output : /opt/ml/model/                 (saved as model.tar.gz automatically)
"""

import argparse
import json
import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("sm_training")

# SageMaker sets these automatically
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
INPUT_DIR = os.environ.get("SM_INPUT_DIR", "/opt/ml/input/data")


def find_file(directory: str, filename: str) -> str:
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found under {directory}")


def train_gmm(X, n_components, covariance_type, random_state, max_iter, n_init):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
        )
        gmm.fit(X)
    return gmm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-components-min",  type=int,   default=3)
    parser.add_argument("--n-components-max",  type=int,   default=7)
    parser.add_argument("--covariance-types",  type=str,   default="full,tied,diag,spherical")
    parser.add_argument("--random-state",      type=int,   default=42)
    parser.add_argument("--max-iter",          type=int,   default=300)
    parser.add_argument("--n-init",            type=int,   default=5)
    parser.add_argument("--selection-metric",  type=str,   default="bic")
    args = parser.parse_args()

    cov_types = [c.strip() for c in args.covariance_types.split(",")]
    n_range   = list(range(args.n_components_min, args.n_components_max + 1))
    metric    = args.selection_metric.lower()

    logger.info("Input dir contents:")
    for root, dirs, files in os.walk(INPUT_DIR):
        logger.info("  %s: %s", root, files)

    # Load preprocessed artifacts
    prep_path  = find_file(INPUT_DIR, "preprocessor.joblib")
    prods_path = find_file(INPUT_DIR, "products_engineered.csv")
    X_path     = find_file(INPUT_DIR, "feature_matrix.npy")

    preprocessor = joblib.load(prep_path)
    products_df  = pd.read_csv(prods_path)
    X            = np.load(X_path)

    logger.info("Loaded X.shape=%s  products=%d", X.shape, len(products_df))

    # Grid search
    best_score  = np.inf
    best_model  = None
    best_params = {}
    all_results = []
    total       = len(n_range) * len(cov_types)
    run_no      = 0

    logger.info("Starting grid search: n_components=%s  cov_types=%s  metric=%s", n_range, cov_types, metric)

    for cov_type in cov_types:
        for n_comp in n_range:
            run_no += 1
            try:
                gmm   = train_gmm(X, n_comp, cov_type, args.random_state, args.max_iter, args.n_init)
                bic   = float(gmm.bic(X))
                aic   = float(gmm.aic(X))
                score = bic if metric == "bic" else aic

                all_results.append({
                    "run": run_no, "n_components": n_comp,
                    "covariance_type": cov_type, "bic": bic, "aic": aic,
                    "converged": bool(gmm.converged_),
                })

                logger.info("[%2d/%2d] n_comp=%-2d  cov=%-10s  BIC=%12.2f  converged=%s",
                            run_no, total, n_comp, cov_type, bic, gmm.converged_)

                if score < best_score:
                    best_score  = score
                    best_model  = gmm
                    best_params = {
                        "n_components": n_comp, "covariance_type": cov_type,
                        "bic": bic, "aic": aic, "selection_metric": metric,
                    }
            except Exception as e:
                logger.error("[%2d/%2d] FAILED n_comp=%d cov=%s: %s", run_no, total, n_comp, cov_type, e)

    if best_model is None:
        raise RuntimeError("No model converged!")

    logger.info("Best: n_components=%d  cov_type=%s  BIC=%.2f",
                best_params["n_components"], best_params["covariance_type"], best_params["bic"])

    # Cluster probability matrix
    prob_matrix     = best_model.predict_proba(X)
    primary_cluster = best_model.predict(X)

    # Add cluster info to products
    products_out = products_df.copy()
    products_out["primary_cluster"] = primary_cluster
    for k in range(prob_matrix.shape[1]):
        products_out[f"cluster_{k}_prob"] = prob_matrix[:, k]

    # Save all artifacts to MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model,    os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(preprocessor,  os.path.join(MODEL_DIR, "preprocessor.joblib"))
    np.save(os.path.join(MODEL_DIR, "cluster_probabilities.npy"), prob_matrix)
    products_out.to_csv(os.path.join(MODEL_DIR, "products_with_clusters.csv"), index=False)

    with open(os.path.join(MODEL_DIR, "training_params.json"), "w") as f:
        json.dump({"best_params": best_params, "all_results": all_results}, f, indent=2)

    logger.info("All artifacts saved to %s", MODEL_DIR)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
