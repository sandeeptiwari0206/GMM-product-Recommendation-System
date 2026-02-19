"""
src/models/recommender.py
=========================
Core GMM recommendation engine.  Fully config-driven.

Scoring formula (weighted sum of 6 components)
-----------------------------------------------
  final_score = w1*cluster_probability_score
              + w2*price_similarity_score
              + w3*popularity_score
              + w4*rating_score
              + w5*discount_score
              + w6*diversity_score

Strategies and weights are read from config/config.yaml →
recommendation.strategies so you can tune weights without touching code.

Customisation hooks
-------------------
- Add new strategies by adding a block to config.yaml
- Change reason thresholds via config.yaml → recommendation.reason_thresholds
- Add new score components by subclassing GMMRecommendationEngine
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.utils import get_logger, load_config, get_paths, get_recommendation_config

logger = get_logger(__name__)


# =============================================================================
# Recommendation dataclass
# =============================================================================

@dataclass
class Recommendation:
    product_id:   str
    product_name: Optional[str]
    category:     Optional[str]
    price:        float
    rating:       float
    discount_pct: float
    final_score:  float
    cluster_id:   int
    cluster_prob: float
    reasons:      List[str] = field(default_factory=list)
    scores:       Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id":        self.product_id,
            "product_name":      self.product_name,
            "category":          self.category,
            "price":             round(self.price, 2),
            "rating":            round(self.rating, 4),
            "discount_pct":      round(self.discount_pct, 2),
            "final_score":       round(self.final_score, 6),
            "cluster_id":        self.cluster_id,
            "cluster_prob":      round(self.cluster_prob, 4),
            "reasons":           self.reasons,
            "component_scores":  {k: round(v, 4) for k, v in self.scores.items()},
        }


# =============================================================================
# Recommendation Engine
# =============================================================================

class GMMRecommendationEngine:
    """
    Config-driven GMM soft-clustering recommendation engine.

    Parameters
    ----------
    model        : fitted GaussianMixture
    preprocessor : fitted ColumnTransformer
    products_df  : cleaned + engineered product DataFrame
    prob_matrix  : (n_products, n_clusters) float array
    invoices_df  : purchase history DataFrame
    rec_cfg      : recommendation section from config
    """

    def __init__(
        self,
        model:        GaussianMixture,
        preprocessor: Any,
        products_df:  pd.DataFrame,
        prob_matrix:  np.ndarray,
        invoices_df:  pd.DataFrame,
        rec_cfg:      Dict[str, Any],
    ) -> None:
        self.model        = model
        self.preprocessor = preprocessor
        self.products_df  = products_df.reset_index(drop=True)
        self.prob_matrix  = prob_matrix
        self.invoices_df  = invoices_df
        self.rec_cfg      = rec_cfg
        self.n_clusters   = model.n_components

        self._strategies        = rec_cfg.get("strategies", {})
        self._reason_thresholds = rec_cfg.get("reason_thresholds", {})
        self._cold_start        = rec_cfg.get("cold_start", "uniform")

        self._prepare_normalized_arrays()
        logger.info(
            "Engine ready  n_products=%d  n_clusters=%d  strategies=%s",
            len(self.products_df), self.n_clusters, list(self._strategies.keys()),
        )

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def _prepare_normalized_arrays(self) -> None:
        """Precompute normalized score arrays (done once at load time)."""
        df = self.products_df

        def minmax(series: pd.Series) -> np.ndarray:
            v = series.values.astype(float)
            mn, mx = v.min(), v.max()
            if mx == mn:
                return np.ones_like(v)
            return (v - mn) / (mx - mn)

        self.norm_popularity = minmax(df["popularity"]) if "popularity" in df else np.zeros(len(df))
        self.norm_rating     = minmax(df["rating"])     if "rating"     in df else np.zeros(len(df))
        self.norm_discount   = minmax(df["discount_pct"]) if "discount_pct" in df else np.zeros(len(df))

        if "created_at" in df.columns:
            ts = pd.to_datetime(df["created_at"], errors="coerce").view("int64").values.astype(float)
            ts = np.where(np.isnan(ts), np.nanmin(ts), ts)
            self.norm_newness = minmax(pd.Series(ts))
        else:
            self.norm_newness = np.zeros(len(df))

        self.primary_cluster = self.prob_matrix.argmax(axis=1)

    # -------------------------------------------------------------------------
    # User vector
    # -------------------------------------------------------------------------

    def get_purchase_history(self, user_id: str) -> pd.DataFrame:
        return self.invoices_df[self.invoices_df["user_id"] == user_id].copy()

    def compute_user_cluster_vector(self, user_id: str) -> np.ndarray:
        """
        Average cluster probability vectors of all products a user has purchased.
        Falls back to uniform distribution for cold-start users.
        """
        history = self.get_purchase_history(user_id)

        if history.empty:
            logger.debug("Cold-start user: %s → uniform cluster vector", user_id)
            return np.ones(self.n_clusters) / self.n_clusters

        known_ids = set(self.products_df["product_id"].astype(str))
        purchased = [p for p in history["product_id"].astype(str) if p in known_ids]

        if not purchased:
            return np.ones(self.n_clusters) / self.n_clusters

        idx = self.products_df[
            self.products_df["product_id"].astype(str).isin(purchased)
        ].index.tolist()

        vec = self.prob_matrix[idx].mean(axis=0)
        vec = vec / (vec.sum() + 1e-9)
        return vec

    def get_purchased_ids(self, user_id: str) -> set:
        h = self.get_purchase_history(user_id)
        return set(h["product_id"].astype(str))

    # -------------------------------------------------------------------------
    # Score components (all return np.ndarray of shape (n_products,) in [0,1])
    # -------------------------------------------------------------------------

    def _cluster_prob_score(self, user_vec: np.ndarray) -> np.ndarray:
        s = self.prob_matrix @ user_vec
        mx = s.max()
        return s / (mx + 1e-9)

    def _price_similarity_score(self, user_id: str) -> np.ndarray:
        history = self.get_purchase_history(user_id)
        prices  = self.products_df["price"].values.astype(float)

        if history.empty or "price" not in history.columns:
            return np.full(len(prices), 0.5)

        user_prices = history["price"].dropna().astype(float)
        if len(user_prices) == 0:
            return np.full(len(prices), 0.5)

        mean_p = user_prices.mean()
        std_p  = max(user_prices.std(), 1.0)
        return np.exp(-0.5 * ((prices - mean_p) / std_p) ** 2)

    def _diversity_score(self, user_vec: np.ndarray, diversity_weight: float) -> np.ndarray:
        rarity   = 1.0 - user_vec[self.primary_cluster]
        raw      = (1.0 - diversity_weight) * rarity + diversity_weight * self.norm_newness
        mx       = raw.max()
        return raw / (mx + 1e-9)

    # -------------------------------------------------------------------------
    # Final scoring
    # -------------------------------------------------------------------------

    def _compute_scores_df(
        self,
        user_id:          str,
        strategy:         str,
        diversity_weight: float,
    ) -> pd.DataFrame:
        """Return a DataFrame with all component scores and final_score."""
        if strategy not in self._strategies:
            logger.warning("Unknown strategy '%s'. Falling back to 'balanced'.", strategy)
            strategy = "balanced"

        weights  = self._strategies[strategy]
        user_vec = self.compute_user_cluster_vector(user_id)

        components = {
            "cluster_probability_score": self._cluster_prob_score(user_vec),
            "price_similarity_score":    self._price_similarity_score(user_id),
            "popularity_score":          self.norm_popularity,
            "rating_score":              self.norm_rating,
            "discount_score":            self.norm_discount,
            "diversity_score":           self._diversity_score(user_vec, diversity_weight),
        }

        final = sum(weights.get(k, 0.0) * v for k, v in components.items())

        df = self.products_df[["product_id"]].copy()
        for name, arr in components.items():
            df[name] = arr
        df["final_score"]     = final
        df["primary_cluster"] = self.primary_cluster
        df["cluster_prob"]    = self.prob_matrix[
            np.arange(len(self.products_df)), self.primary_cluster
        ]
        return df, user_vec, strategy

    # -------------------------------------------------------------------------
    # Reason generation
    # -------------------------------------------------------------------------

    def _build_reasons(
        self,
        row:      pd.Series,
        strategy: str,
    ) -> List[str]:
        t       = self._reason_thresholds
        reasons = []

        if row.get("rating_score", 0) > t.get("highly_rated", 0.70):
            reasons.append("Highly rated")

        if row.get("price_similarity_score", 0) > t.get("price_match", 0.70):
            reasons.append("Matches your price preference")

        if row.get("cluster_probability_score", 0) > t.get("cluster_match", 0.60):
            reasons.append("Popular in your preferred cluster")

        if row.get("popularity_score", 0) > t.get("top_seller", 0.75):
            reasons.append("Top seller")

        if row.get("discount_score", 0) > t.get("great_discount", 0.70):
            reasons.append("Great discount available")

        if strategy == "value" and row.get("discount_score", 0) > 0.5:
            reasons.append("Best value for money")

        if strategy == "new" and row.get("diversity_score", 0) > t.get("new_arrival", 0.60):
            reasons.append("New arrival")

        if strategy == "diverse":
            reasons.append("Broadens your product mix")

        if not reasons:
            reasons.append("Recommended for you")

        return reasons

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def recommend(
        self,
        user_id:          str,
        top_n:            int            = 10,
        strategy:         str            = "balanced",
        diversity_weight: float          = 0.3,
        category_filter:  Optional[str]  = None,
        price_min:        float          = 0.0,
        price_max:        float          = 1_000_000.0,
    ) -> Dict[str, Any]:
        """
        Generate top-N product recommendations for a user.

        Parameters
        ----------
        user_id          : target user identifier
        top_n            : number of recommendations to return
        strategy         : one of the strategy keys in config.yaml
        diversity_weight : weight on diversity/newness [0, 1]
        category_filter  : restrict to one product category (optional)
        price_min        : minimum product price filter
        price_max        : maximum product price filter

        Returns
        -------
        dict with user metadata and ordered recommendations list
        """
        max_n = self.rec_cfg.get("max_top_n", 50)
        top_n = min(top_n, max_n)

        logger.info(
            "recommend  user=%s  strategy=%s  top_n=%d  price=[%g, %g]",
            user_id, strategy, top_n, price_min, price_max,
        )

        scores_df, user_vec, strategy = self._compute_scores_df(user_id, strategy, diversity_weight)

        # Merge product metadata
        meta_cols = ["product_id", "category", "price", "rating", "discount_pct"]
        if "product_name" in self.products_df.columns:
            meta_cols.append("product_name")

        scores_df = scores_df.merge(self.products_df[meta_cols], on="product_id", how="left")

        # --- Filters ---
        purchased = self.get_purchased_ids(user_id)
        scores_df = scores_df[~scores_df["product_id"].astype(str).isin(purchased)]
        scores_df = scores_df[(scores_df["price"] >= price_min) & (scores_df["price"] <= price_max)]

        if category_filter:
            scores_df = scores_df[
                scores_df["category"].astype(str).str.lower() == category_filter.strip().lower()
            ]

        if scores_df.empty:
            logger.warning("No candidate products after filtering for user %s", user_id)
            recs = []
        else:
            scores_df = scores_df.sort_values("final_score", ascending=False).head(top_n)
            score_keys = list(self._strategies["balanced"].keys())

            recs = []
            for _, row in scores_df.iterrows():
                reasons = self._build_reasons(row, strategy)
                rec = Recommendation(
                    product_id   = str(row["product_id"]),
                    product_name = str(row.get("product_name", "")),
                    category     = str(row.get("category", "")),
                    price        = float(row["price"]),
                    rating       = float(row["rating"]),
                    discount_pct = float(row.get("discount_pct", 0.0)),
                    final_score  = float(row["final_score"]),
                    cluster_id   = int(row["primary_cluster"]),
                    cluster_prob = float(row["cluster_prob"]),
                    reasons      = reasons,
                    scores       = {k: float(row[k]) for k in score_keys if k in row},
                )
                recs.append(rec.to_dict())

        top_clusters = np.argsort(user_vec)[::-1].tolist()
        top_probs    = user_vec[top_clusters].tolist()

        return {
            "user_id":               user_id,
            "strategy":              strategy,
            "preferred_clusters":    top_clusters,
            "cluster_probabilities": [round(p, 4) for p in top_probs],
            "n_recommendations":     len(recs),
            "recommendations":       recs,
        }


# =============================================================================
# Factory loader
# =============================================================================

def load_engine(cfg: Optional[Dict[str, Any]] = None) -> GMMRecommendationEngine:
    """
    Load all artifacts from disk and return a ready-to-use engine.
    Config-driven and environment-aware.
    """
    if cfg is None:
        cfg = load_config()

    paths   = get_paths(cfg)
    rec_cfg = get_recommendation_config(cfg)

    model        = joblib.load(paths["model"])
    preprocessor = joblib.load(paths["preprocessor"])
    prob_matrix  = np.load(paths["probabilities"])
    products_df  = pd.read_csv(paths["products_clustered"])
    invoices_df  = pd.read_csv(paths["invoices"])

    logger.info(
        "Engine loaded  products=%d  clusters=%d  strategies=%s",
        len(products_df), model.n_components, list(rec_cfg.get("strategies", {}).keys()),
    )
    return GMMRecommendationEngine(model, preprocessor, products_df, prob_matrix, invoices_df, rec_cfg)


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse, json, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(description="Get GMM recommendations for a user")
    parser.add_argument("--user-id",         required=True)
    parser.add_argument("--config",          default=None)
    parser.add_argument("--top-n",           type=int,   default=10)
    parser.add_argument("--strategy",        default="balanced")
    parser.add_argument("--diversity-weight", type=float, default=0.3)
    parser.add_argument("--category-filter", default=None)
    parser.add_argument("--price-min",       type=float, default=0)
    parser.add_argument("--price-max",       type=float, default=1_000_000)
    args = parser.parse_args()

    cfg    = load_config(args.config) if args.config else load_config()
    engine = load_engine(cfg)
    result = engine.recommend(
        user_id=args.user_id,
        top_n=args.top_n,
        strategy=args.strategy,
        diversity_weight=args.diversity_weight,
        category_filter=args.category_filter,
        price_min=args.price_min,
        price_max=args.price_max,
    )
    print(json.dumps(result, indent=2))
