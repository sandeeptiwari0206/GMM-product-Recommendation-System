"""
tests/test_pipeline.py
======================
Automated tests for the GMM pipeline.

Run: pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_products():
    n = 200
    np.random.seed(0)
    return pd.DataFrame({
        "product_id":   [f"PROD_{i:04d}" for i in range(n)],
        "category":     np.random.choice(["Electronics", "Books", "Clothing"], n),
        "price":        np.abs(np.random.normal(150, 80, n)) + 5,
        "rating":       np.clip(np.random.normal(3.8, 0.6, n), 1.0, 5.0),
        "popularity":   np.random.randint(10, 1000, n),
        "discount_pct": np.abs(np.random.normal(10, 8, n)),
        "created_at":   pd.date_range("2023-01-01", periods=n, freq="1D").astype(str),
    })


@pytest.fixture(scope="module")
def sample_users():
    n = 50
    return pd.DataFrame({
        "user_id":            [f"USR_{i:05d}" for i in range(n)],
        "segment":            np.random.choice(["enterprise", "smb"], n),
        "preferred_category": np.random.choice(["Electronics", "Books"], n),
    })


@pytest.fixture(scope="module")
def sample_invoices(sample_products, sample_users):
    n = 300
    pids = sample_products["product_id"].values
    uids = sample_users["user_id"].values
    prices = {r["product_id"]: r["price"] for _, r in sample_products.iterrows()}
    rows = []
    for i in range(n):
        pid = np.random.choice(pids)
        rows.append({
            "invoice_id":    f"INV_{i:05d}",
            "user_id":       np.random.choice(uids),
            "product_id":    pid,
            "price":         prices[pid],
            "purchase_date": "2023-06-01",
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tmp_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {
            "artifacts_dir":      os.path.join(tmpdir, "artifacts"),
            "processed_dir":      os.path.join(tmpdir, "artifacts", "processed"),
            "output_dir":         os.path.join(tmpdir, "output"),
        }
        for d in paths.values():
            os.makedirs(d, exist_ok=True)
        yield tmpdir, paths


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_missing_value_imputation(self, sample_products):
        from src.data.preprocessing import handle_missing_values
        df = sample_products.copy()
        df.loc[0, "rating"]       = np.nan
        df.loc[1, "discount_pct"] = np.nan

        result = handle_missing_values(df)
        assert result["rating"].isnull().sum() == 0
        assert result["discount_pct"].isnull().sum() == 0

    def test_negative_price_dropped(self, sample_products):
        from src.data.preprocessing import handle_missing_values
        df = sample_products.copy()
        df.loc[0, "price"] = -5.0
        result = handle_missing_values(df)
        assert (result["price"] <= 0).sum() == 0

    def test_feature_engineering(self, sample_products):
        from src.data.preprocessing import handle_missing_values, engineer_features
        cfg = {
            "features": {
                "engineering": {
                    "price_bins":   [0, 50, 150, 300, 500, 99999],
                    "price_labels": ["budget", "economy", "mid", "premium", "luxury"],
                }
            }
        }
        df = handle_missing_values(sample_products)
        df = engineer_features(df, cfg["features"])

        assert "value_score"                in df.columns
        assert "popularity_weighted_rating" in df.columns
        assert "price_bucket"               in df.columns
        assert "normalized_price"           in df.columns
        assert df["normalized_price"].between(0, 1).all()

    def test_fit_transform_shape(self, sample_products, tmp_dirs):
        from src.data.preprocessing import handle_missing_values, engineer_features, fit_transform_save
        tmpdir, paths = tmp_dirs

        feat_cfg = {
            "numeric_features":     ["price", "rating", "popularity", "discount_pct",
                                     "value_score", "popularity_weighted_rating", "normalized_price"],
            "categorical_features": ["category", "price_bucket"],
            "engineering": {
                "price_bins":   [0, 50, 150, 300, 500, 99999],
                "price_labels": ["budget", "economy", "mid", "premium", "luxury"],
            }
        }

        df = handle_missing_values(sample_products)
        df = engineer_features(df, feat_cfg)
        save_path = os.path.join(paths["artifacts_dir"], "preprocessor.joblib")
        X, prep = fit_transform_save(df, feat_cfg, save_path)

        assert X.shape[0] == len(df)
        assert X.shape[1] > 7    # 7 numeric + one-hot cats
        assert os.path.exists(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Training tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTraining:
    def test_single_gmm_converges(self, sample_products):
        from src.data.preprocessing import handle_missing_values, engineer_features, fit_transform_save
        from src.models.training import train_single_gmm
        import tempfile

        feat_cfg = {
            "numeric_features":     ["price", "rating", "popularity", "discount_pct",
                                     "value_score", "popularity_weighted_rating", "normalized_price"],
            "categorical_features": ["category", "price_bucket"],
            "engineering": {"price_bins": [0, 50, 150, 300, 500, 99999], "price_labels": ["budget","economy","mid","premium","luxury"]},
        }
        df = handle_missing_values(sample_products)
        df = engineer_features(df, feat_cfg)
        with tempfile.TemporaryDirectory() as d:
            X, _ = fit_transform_save(df, feat_cfg, os.path.join(d, "p.joblib"))

        gmm = train_single_gmm(X, n_components=3, covariance_type="diag", max_iter=100, n_init=3)
        assert gmm.converged_
        assert gmm.n_components == 3

    def test_cluster_probabilities_sum_to_one(self, sample_products):
        from src.data.preprocessing import handle_missing_values, engineer_features, fit_transform_save
        from src.models.training import train_single_gmm, compute_cluster_probabilities
        import tempfile

        feat_cfg = {
            "numeric_features":     ["price", "rating", "popularity", "discount_pct",
                                     "value_score", "popularity_weighted_rating", "normalized_price"],
            "categorical_features": ["category", "price_bucket"],
            "engineering": {"price_bins": [0, 50, 150, 300, 500, 99999], "price_labels": ["budget","economy","mid","premium","luxury"]},
        }
        df = handle_missing_values(sample_products)
        df = engineer_features(df, feat_cfg)
        with tempfile.TemporaryDirectory() as d:
            X, _ = fit_transform_save(df, feat_cfg, os.path.join(d, "p.joblib"))

        gmm   = train_single_gmm(X, n_components=3, covariance_type="diag", max_iter=100, n_init=2)
        probs = compute_cluster_probabilities(gmm, X)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Recommender tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommender:

    @pytest.fixture(scope="class")
    def engine_fixture(self, sample_products, sample_users, sample_invoices):
        from sklearn.mixture import GaussianMixture
        from src.data.preprocessing import handle_missing_values, engineer_features, fit_transform_save
        from src.models.recommender import GMMRecommendationEngine
        import tempfile

        feat_cfg = {
            "numeric_features":     ["price", "rating", "popularity", "discount_pct",
                                     "value_score", "popularity_weighted_rating", "normalized_price"],
            "categorical_features": ["category", "price_bucket"],
            "engineering": {"price_bins": [0, 50, 150, 300, 500, 99999], "price_labels": ["budget","economy","mid","premium","luxury"]},
        }
        rec_cfg = {
            "default_strategy": "balanced", "default_top_n": 5, "max_top_n": 50,
            "cold_start": "uniform",
            "reason_thresholds": {"highly_rated": 0.70, "price_match": 0.70, "cluster_match": 0.60, "top_seller": 0.75, "great_discount": 0.70, "new_arrival": 0.60},
            "strategies": {
                "balanced": {"cluster_probability_score": 0.30, "price_similarity_score": 0.15, "popularity_score": 0.20, "rating_score": 0.20, "discount_score": 0.10, "diversity_score": 0.05},
            },
        }

        df = handle_missing_values(sample_products)
        df = engineer_features(df, feat_cfg)

        with tempfile.TemporaryDirectory() as d:
            X, prep = fit_transform_save(df, feat_cfg, os.path.join(d, "p.joblib"))
            gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=42, n_init=3)
            gmm.fit(X)
            prob_matrix = gmm.predict_proba(X)
            return GMMRecommendationEngine(gmm, prep, df, prob_matrix, sample_invoices, rec_cfg)

    def test_recommend_returns_correct_keys(self, engine_fixture, sample_users):
        uid    = sample_users["user_id"].iloc[0]
        result = engine_fixture.recommend(user_id=uid, top_n=5)
        for key in ("user_id", "recommendations", "preferred_clusters", "cluster_probabilities"):
            assert key in result

    def test_recommend_top_n_respected(self, engine_fixture, sample_users):
        uid  = sample_users["user_id"].iloc[1]
        result = engine_fixture.recommend(user_id=uid, top_n=3)
        assert result["n_recommendations"] <= 3

    def test_cold_start_user(self, engine_fixture):
        result = engine_fixture.recommend(user_id="UNKNOWN_USR_99999", top_n=5)
        assert result["n_recommendations"] >= 0   # should not crash

    def test_price_filter(self, engine_fixture, sample_users):
        uid    = sample_users["user_id"].iloc[2]
        result = engine_fixture.recommend(user_id=uid, top_n=10, price_max=50)
        for rec in result["recommendations"]:
            assert rec["price"] <= 50

    def test_reasons_not_empty(self, engine_fixture, sample_users):
        uid    = sample_users["user_id"].iloc[3]
        result = engine_fixture.recommend(user_id=uid, top_n=5)
        for rec in result["recommendations"]:
            assert len(rec["reasons"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Inference handler tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInference:
    def test_input_fn_json(self):
        from src.inference import input_fn
        body   = json.dumps({"user_id": "USR_00001", "top_n": 5, "strategy": "value"})
        result = input_fn(body, "application/json")
        assert len(result["requests"]) == 1
        assert result["requests"][0]["user_id"] == "USR_00001"
        assert result["requests"][0]["top_n"] == 5

    def test_input_fn_batch(self):
        from src.inference import input_fn
        body   = json.dumps({"requests": [{"user_id": "A"}, {"user_id": "B"}]})
        result = input_fn(body, "application/json")
        assert len(result["requests"]) == 2

    def test_input_fn_csv(self):
        from src.inference import input_fn
        result = input_fn("USR_00001\nUSR_00002", "text/csv")
        assert len(result["requests"]) == 2

    def test_input_fn_missing_user_id(self):
        from src.inference import input_fn
        with pytest.raises(ValueError):
            input_fn(json.dumps({"top_n": 5}), "application/json")

    def test_output_fn_json(self):
        from src.inference import output_fn
        preds  = [{"user_id": "USR_00001", "recommendations": []}]
        body, ct = output_fn(preds, "application/json")
        assert ct == "application/json"
        data = json.loads(body)
        assert data["user_id"] == "USR_00001"

    def test_output_fn_csv(self):
        from src.inference import output_fn
        preds = [{"user_id": "U1", "recommendations": [
            {"product_id": "P1", "final_score": 0.9, "category": "Books", "reasons": ["Highly rated"]}
        ]}]
        body, ct = output_fn(preds, "text/csv")
        assert ct == "text/csv"
        assert "product_id" in body
