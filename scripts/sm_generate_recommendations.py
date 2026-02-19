"""
scripts/sm_generate_recommendations.py
=======================================
Self-contained SageMaker Processing script.
Step 5 of the pipeline — generates recommendations CSV for ALL customers.

Container paths
---------------
Input  model    : /opt/ml/processing/input/model/    (model artifacts)
Input  invoices : /opt/ml/processing/input/invoices/ (invoices.csv)
Input  users    : /opt/ml/processing/input/users/    (users.csv)
Output          : /opt/ml/processing/output/         (recommendations CSV)
"""

import argparse
import json
import logging
import os
import sys
import tarfile

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("sm_generate_recs")

MODEL_DIR   = "/opt/ml/processing/input/model"
OUTPUT_DIR  = "/opt/ml/processing/output"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def find_file(directory: str, filename: str) -> str:
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found under {directory}")


def extract_tar(model_dir: str) -> str:
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        extract_dir = os.path.join(model_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)
        logger.info("Extracted model.tar.gz → %s", extract_dir)
        return extract_dir
    return model_dir


def minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.ones_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


# ─────────────────────────────────────────────────────────────
# User cluster vector
# ─────────────────────────────────────────────────────────────

def compute_user_vector(user_id, invoices_df, products_df, prob_matrix, n_clusters):
    history   = invoices_df[invoices_df["user_id"] == user_id]
    if history.empty:
        return np.ones(n_clusters) / n_clusters

    known     = set(products_df["product_id"].astype(str))
    purchased = [p for p in history["product_id"].astype(str) if p in known]
    if not purchased:
        return np.ones(n_clusters) / n_clusters

    idx = products_df[products_df["product_id"].astype(str).isin(purchased)].index.tolist()
    vec = prob_matrix[idx].mean(axis=0)
    return vec / (vec.sum() + 1e-9)


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────

STRATEGY_WEIGHTS = {
    "balanced": {"cluster": 0.30, "price": 0.15, "popularity": 0.20, "rating": 0.20, "discount": 0.15},
    "popular":  {"cluster": 0.20, "price": 0.10, "popularity": 0.35, "rating": 0.25, "discount": 0.10},
    "value":    {"cluster": 0.20, "price": 0.20, "popularity": 0.10, "rating": 0.15, "discount": 0.35},
    "diverse":  {"cluster": 0.15, "price": 0.10, "popularity": 0.10, "rating": 0.15, "discount": 0.10},
}


def score_all(user_id, invoices_df, products_df, prob_matrix,
              norm_pop, norm_rating, norm_discount, weights):
    n_clusters = prob_matrix.shape[1]
    user_vec   = compute_user_vector(user_id, invoices_df, products_df, prob_matrix, n_clusters)

    cluster_s  = prob_matrix @ user_vec
    cluster_s  = cluster_s / (cluster_s.max() + 1e-9)

    prices     = products_df["price"].values.astype(float)
    history    = invoices_df[invoices_df["user_id"] == user_id]
    if not history.empty and "price" in history.columns and len(history["price"].dropna()) > 0:
        mean_p   = history["price"].dropna().mean()
        std_p    = max(history["price"].dropna().std(), 1.0)
        price_s  = np.exp(-0.5 * ((prices - mean_p) / std_p) ** 2)
    else:
        price_s  = np.full(len(prices), 0.5)

    final = (
        weights["cluster"]    * cluster_s    +
        weights["price"]      * price_s      +
        weights["popularity"] * norm_pop     +
        weights["rating"]     * norm_rating  +
        weights["discount"]   * norm_discount
    )
    return final, user_vec, cluster_s, price_s


def get_reasons(cluster_s, rating_s, pop_s, discount_s, price_s):
    reasons = []
    if rating_s   > 0.70: reasons.append("Highly rated")
    if cluster_s  > 0.60: reasons.append("Matches your preference")
    if pop_s      > 0.75: reasons.append("Top seller")
    if discount_s > 0.70: reasons.append("Great discount")
    if price_s    > 0.70: reasons.append("Good price match")
    if not reasons:       reasons.append("Recommended for you")
    return " | ".join(reasons)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--invoices",  default="/opt/ml/processing/input/invoices/invoices.csv")
    parser.add_argument("--users",     default="/opt/ml/processing/input/users/users.csv")
    parser.add_argument("--top-n",     type=int, default=10)
    parser.add_argument("--min-n",     type=int, default=3)
    parser.add_argument("--strategy",  default="balanced")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model artifacts
    artifact_dir = extract_tar(MODEL_DIR)
    logger.info("Loading artifacts from %s", artifact_dir)

    model_path  = find_file(artifact_dir, "model.joblib")
    prep_path   = find_file(artifact_dir, "preprocessor.joblib")
    probs_path  = find_file(artifact_dir, "cluster_probabilities.npy")
    prods_path  = find_file(artifact_dir, "products_with_clusters.csv")

    model        = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    prob_matrix  = np.load(probs_path)
    products_df  = pd.read_csv(prods_path).reset_index(drop=True)

    logger.info("Model: n_components=%d  cov_type=%s", model.n_components, model.covariance_type)
    logger.info("Products: %d  Clusters: %d", len(products_df), prob_matrix.shape[1])

    # Load users and invoices
    invoices_df = pd.read_csv(args.invoices)
    users_df    = pd.read_csv(args.users)
    logger.info("Users=%d  Invoices=%d", len(users_df), len(invoices_df))

    # Precompute normalised arrays
    norm_pop      = minmax(products_df["popularity"].values.astype(float))   if "popularity"   in products_df.columns else np.zeros(len(products_df))
    norm_rating   = minmax(products_df["rating"].values.astype(float))       if "rating"       in products_df.columns else np.zeros(len(products_df))
    norm_discount = minmax(products_df["discount_pct"].values.astype(float)) if "discount_pct" in products_df.columns else np.zeros(len(products_df))
    primary_cluster = prob_matrix.argmax(axis=1)

    weights  = STRATEGY_WEIGHTS.get(args.strategy, STRATEGY_WEIGHTS["balanced"])
    user_ids = users_df["user_id"].tolist()
    total    = len(user_ids)

    logger.info("Generating recommendations: users=%d  strategy=%s  top_n=%d  min_n=%d",
                total, args.strategy, args.top_n, args.min_n)

    all_rows = []

    for i, user_id in enumerate(user_ids, 1):
        if i % 200 == 0 or i == total:
            logger.info("  Progress: %d / %d", i, total)

        purchased = set(
            invoices_df[invoices_df["user_id"] == user_id]["product_id"].astype(str)
        )

        final_scores, user_vec, cluster_s, price_s = score_all(
            user_id, invoices_df, products_df, prob_matrix,
            norm_pop, norm_rating, norm_discount, weights,
        )

        preferred_cluster = int(np.argmax(user_vec))

        # Build candidate df excluding already purchased
        cands = products_df.copy()
        cands["_final_score"]   = final_scores
        cands["_cluster_score"] = cluster_s
        cands["_price_score"]   = price_s
        cands["_norm_rating"]   = norm_rating
        cands["_norm_pop"]      = norm_pop
        cands["_norm_discount"] = norm_discount

        cands = cands[~cands["product_id"].astype(str).isin(purchased)]
        cands = cands.sort_values("_final_score", ascending=False)

        # Guarantee minimum
        n_take   = max(args.top_n, args.min_n)
        top_cands = cands.head(n_take)

        for rank, (_, row) in enumerate(top_cands.iterrows(), start=1):
            reasons = get_reasons(
                row["_cluster_score"],
                row["_norm_rating"],
                row["_norm_pop"],
                row["_norm_discount"],
                row["_price_score"],
            )
            all_rows.append({
                "user_id":           user_id,
                "rank":              rank,
                "product_id":        row["product_id"],
                "category":          row.get("category", ""),
                "price":             round(float(row["price"]), 2),
                "rating":            round(float(row.get("rating", 0)), 2),
                "discount_pct":      round(float(row.get("discount_pct", 0)), 2),
                "final_score":       round(float(row["_final_score"]), 6),
                "cluster_id":        int(primary_cluster[row.name]),
                "cluster_prob":      round(float(prob_matrix[row.name, primary_cluster[row.name]]), 4),
                "preferred_cluster": preferred_cluster,
                "strategy":          args.strategy,
                "reasons":           reasons,
                "cluster_score":     round(float(row["_cluster_score"]), 4),
                "rating_score":      round(float(row["_norm_rating"]), 4),
                "popularity_score":  round(float(row["_norm_pop"]), 4),
                "discount_score":    round(float(row["_norm_discount"]), 4),
                "price_score":       round(float(row["_price_score"]), 4),
            })

    out_df   = pd.DataFrame(all_rows)
    out_path = os.path.join(OUTPUT_DIR, "all_customer_recommendations.csv")
    out_df.to_csv(out_path, index=False)

    logger.info("=" * 60)
    logger.info("Recommendations CSV saved → %s", out_path)
    logger.info("  Total rows        : %d", len(out_df))
    logger.info("  Unique users      : %d", out_df['user_id'].nunique())
    logger.info("  Avg recs per user : %.1f", len(out_df) / max(out_df['user_id'].nunique(), 1))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
