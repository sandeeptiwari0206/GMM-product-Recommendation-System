"""
scripts/generate_recommendations_csv.py
========================================
Saare customers ke liye recommendations generate karo aur CSV mein save karo.

Usage
-----
    python scripts/generate_recommendations_csv.py
    python scripts/generate_recommendations_csv.py --top-n 5 --strategy balanced
    python scripts/generate_recommendations_csv.py --output-dir output/

Output
------
    output/all_customer_recommendations.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("generate_recs")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────

def load_artifacts(artifacts_dir: str):
    model        = joblib.load(os.path.join(artifacts_dir, "model.joblib"))
    preprocessor = joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))
    prob_matrix  = np.load(os.path.join(artifacts_dir, "cluster_probabilities.npy"))
    products_df  = pd.read_csv(os.path.join(artifacts_dir, "products_with_clusters.csv"))
    logger.info("Artifacts loaded — products=%d  clusters=%d", len(products_df), model.n_components)
    return model, preprocessor, prob_matrix, products_df


# ─────────────────────────────────────────────────────────────
# Score components
# ─────────────────────────────────────────────────────────────

def minmax(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


def compute_user_vector(user_id, invoices_df, products_df, prob_matrix, n_clusters):
    """Average cluster probs of purchased products. Uniform if cold-start."""
    history = invoices_df[invoices_df["user_id"] == user_id]
    if history.empty:
        return np.ones(n_clusters) / n_clusters

    known = set(products_df["product_id"].astype(str))
    purchased = [p for p in history["product_id"].astype(str) if p in known]
    if not purchased:
        return np.ones(n_clusters) / n_clusters

    idx = products_df[products_df["product_id"].astype(str).isin(purchased)].index.tolist()
    vec = prob_matrix[idx].mean(axis=0)
    return vec / (vec.sum() + 1e-9)


def score_products(user_id, invoices_df, products_df, prob_matrix,
                   norm_pop, norm_rating, norm_discount, weights):
    n_clusters = prob_matrix.shape[1]
    user_vec   = compute_user_vector(user_id, invoices_df, products_df, prob_matrix, n_clusters)

    # Cluster probability score
    cluster_scores = prob_matrix @ user_vec
    cluster_scores = cluster_scores / (cluster_scores.max() + 1e-9)

    # Price similarity score
    history = invoices_df[invoices_df["user_id"] == user_id]
    prices  = products_df["price"].values.astype(float)
    if not history.empty and "price" in history.columns:
        user_prices = history["price"].dropna().astype(float)
        mean_p = user_prices.mean()
        std_p  = max(user_prices.std(), 1.0)
        price_scores = np.exp(-0.5 * ((prices - mean_p) / std_p) ** 2)
    else:
        price_scores = np.full(len(prices), 0.5)

    final = (
        weights["cluster"]   * cluster_scores +
        weights["price"]     * price_scores +
        weights["popularity"] * norm_pop +
        weights["rating"]    * norm_rating +
        weights["discount"]  * norm_discount
    )
    return final, user_vec


# ─────────────────────────────────────────────────────────────
# Reason generation
# ─────────────────────────────────────────────────────────────

def get_reasons(row):
    reasons = []
    if row.get("rating_score", 0)   > 0.70: reasons.append("Highly rated")
    if row.get("cluster_score", 0)  > 0.60: reasons.append("Matches your preference")
    if row.get("pop_score", 0)      > 0.75: reasons.append("Top seller")
    if row.get("discount_score", 0) > 0.70: reasons.append("Great discount")
    if row.get("price_score", 0)    > 0.70: reasons.append("Good price match")
    if not reasons: reasons.append("Recommended for you")
    return " | ".join(reasons)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate recommendations CSV for all customers")
    parser.add_argument("--artifacts-dir", default="artifacts",     help="Path to model artifacts")
    parser.add_argument("--invoices",      default="data/raw/invoices.csv")
    parser.add_argument("--users",         default="data/raw/users.csv")
    parser.add_argument("--output-dir",    default="output")
    parser.add_argument("--top-n",         type=int,   default=10,       help="Max recommendations per user")
    parser.add_argument("--min-n",         type=int,   default=3,        help="Minimum recommendations per user")
    parser.add_argument("--strategy",      default="balanced",
                        choices=["balanced", "popular", "value", "diverse"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    model, preprocessor, prob_matrix, products_df = load_artifacts(args.artifacts_dir)
    invoices_df = pd.read_csv(args.invoices)
    users_df    = pd.read_csv(args.users)

    logger.info("Users=%d  Products=%d  Invoices=%d",
                len(users_df), len(products_df), len(invoices_df))

    # Precompute normalised arrays
    norm_pop      = minmax(products_df["popularity"].values.astype(float))  if "popularity"   in products_df.columns else np.zeros(len(products_df))
    norm_rating   = minmax(products_df["rating"].values.astype(float))      if "rating"       in products_df.columns else np.zeros(len(products_df))
    norm_discount = minmax(products_df["discount_pct"].values.astype(float)) if "discount_pct" in products_df.columns else np.zeros(len(products_df))
    primary_cluster = prob_matrix.argmax(axis=1)

    # Strategy weights
    strategy_weights = {
        "balanced": {"cluster": 0.30, "price": 0.15, "popularity": 0.20, "rating": 0.20, "discount": 0.15},
        "popular":  {"cluster": 0.20, "price": 0.10, "popularity": 0.35, "rating": 0.25, "discount": 0.10},
        "value":    {"cluster": 0.20, "price": 0.20, "popularity": 0.10, "rating": 0.15, "discount": 0.35},
        "diverse":  {"cluster": 0.15, "price": 0.10, "popularity": 0.10, "rating": 0.15, "discount": 0.10},
    }
    weights = strategy_weights[args.strategy]

    all_rows = []
    user_ids = users_df["user_id"].tolist()
    total    = len(user_ids)

    logger.info("Generating recommendations for %d users  strategy=%s  top_n=%d  min_n=%d",
                total, args.strategy, args.top_n, args.min_n)

    for i, user_id in enumerate(user_ids, 1):
        if i % 200 == 0:
            logger.info("  Progress: %d / %d users done", i, total)

        # Products this user already bought
        purchased = set(
            invoices_df[invoices_df["user_id"] == user_id]["product_id"].astype(str)
        )

        # Score all products
        scores, user_vec = score_products(
            user_id, invoices_df, products_df, prob_matrix,
            norm_pop, norm_rating, norm_discount, weights,
        )

        # Build candidate dataframe
        cands = products_df.copy()
        cands["final_score"]    = scores
        cands["cluster_score"]  = (prob_matrix @ user_vec) / ((prob_matrix @ user_vec).max() + 1e-9)
        cands["pop_score"]      = norm_pop
        cands["rating_score"]   = norm_rating
        cands["discount_score"] = norm_discount

        # Compute price score for reason generation
        history = invoices_df[invoices_df["user_id"] == user_id]
        if not history.empty and "price" in history.columns:
            mean_p = history["price"].dropna().mean()
            std_p  = max(history["price"].dropna().std(), 1.0)
            cands["price_score"] = np.exp(-0.5 * ((cands["price"].values - mean_p) / std_p) ** 2)
        else:
            cands["price_score"] = 0.5

        # Remove purchased products
        cands = cands[~cands["product_id"].astype(str).isin(purchased)]

        # Sort by score
        cands = cands.sort_values("final_score", ascending=False)

        # Ensure minimum recommendations
        top_cands = cands.head(max(args.top_n, args.min_n))

        # If still fewer than min_n, take top min_n regardless
        if len(top_cands) < args.min_n:
            top_cands = cands.head(args.min_n)

        # Build output rows
        preferred_cluster = int(np.argmax(user_vec))

        for rank, (_, row) in enumerate(top_cands.iterrows(), start=1):
            reasons = get_reasons(row)
            all_rows.append({
                "user_id":                user_id,
                "rank":                   rank,
                "product_id":             row["product_id"],
                "category":               row.get("category", ""),
                "price":                  round(float(row["price"]), 2),
                "rating":                 round(float(row.get("rating", 0)), 2),
                "discount_pct":           round(float(row.get("discount_pct", 0)), 2),
                "final_score":            round(float(row["final_score"]), 6),
                "cluster_id":             int(primary_cluster[row.name]),
                "cluster_prob":           round(float(prob_matrix[row.name, primary_cluster[row.name]]), 4),
                "preferred_cluster":      preferred_cluster,
                "strategy":               args.strategy,
                "reasons":                reasons,
                "cluster_score":          round(float(row["cluster_score"]), 4),
                "rating_score":           round(float(row["rating_score"]), 4),
                "popularity_score":       round(float(row["pop_score"]), 4),
                "discount_score":         round(float(row["discount_score"]), 4),
                "price_score":            round(float(row["price_score"]), 4),
            })

    # Save CSV
    out_df   = pd.DataFrame(all_rows)
    out_path = os.path.join(args.output_dir, "all_customer_recommendations.csv")
    out_df.to_csv(out_path, index=False)

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("  Total rows        : %d", len(out_df))
    logger.info("  Unique users      : %d", out_df['user_id'].nunique())
    logger.info("  Avg recs per user : %.1f", len(out_df) / max(out_df['user_id'].nunique(), 1))
    logger.info("  Output saved to   : %s", out_path)
    logger.info("=" * 60)

    # Quick preview
    print("\nSample output (first 5 rows):")
    print(out_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
