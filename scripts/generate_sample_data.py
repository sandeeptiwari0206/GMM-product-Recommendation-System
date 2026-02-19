"""
scripts/generate_sample_data.py
================================
Generate realistic sample datasets for local testing.
Run this if you don't have real CSVs yet.

    python scripts/generate_sample_data.py
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CATEGORIES        = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Beauty", "Toys", "Automotive"]
SEGMENTS          = ["enterprise", "smb", "startup", "retail"]
RANDOM_SEED       = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_products(n: int = 5000) -> pd.DataFrame:
    """Generate products_raw.csv with realistic distributions."""
    cats   = np.random.choice(CATEGORIES, n)
    prices = np.where(
        cats == "Electronics", np.random.lognormal(5.5, 0.8, n),
        np.where(cats == "Books", np.abs(np.random.normal(25, 15, n)) + 5,
        np.abs(np.random.lognormal(4.0, 0.7, n)) + 10)
    )
    prices = np.clip(prices, 1.0, 2000.0)

    df = pd.DataFrame({
        "product_id":   [f"PROD_{i:05d}" for i in range(n)],
        "category":     cats,
        "price":        np.round(prices, 2),
        "rating":       np.clip(np.random.normal(3.8, 0.8, n), 1.0, 5.0).round(2),
        "popularity":   np.random.randint(10, 5000, n),
        "discount_pct": np.clip(np.random.exponential(10, n), 0, 60).round(1),
        "created_at":   pd.date_range("2022-01-01", periods=n, freq="1h").astype(str),
    })

    # Inject ~2% missing values
    for col in ["rating", "discount_pct"]:
        mask = np.random.rand(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


def generate_users(n: int = 2000) -> pd.DataFrame:
    """Generate users.csv."""
    return pd.DataFrame({
        "user_id":            [f"USR_{i:05d}" for i in range(n)],
        "segment":            np.random.choice(SEGMENTS, n),
        "preferred_category": np.random.choice(CATEGORIES, n),
    })


def generate_invoices(products_df: pd.DataFrame, users_df: pd.DataFrame, n: int = 15000) -> pd.DataFrame:
    """Generate invoices.csv respecting user category preferences somewhat."""
    rows = []
    product_ids = products_df["product_id"].values
    user_ids    = users_df["user_id"].values

    for i in range(n):
        uid  = np.random.choice(user_ids)
        pid  = np.random.choice(product_ids)
        prod = products_df[products_df["product_id"] == pid].iloc[0]
        rows.append({
            "invoice_id":    f"INV_{i:06d}",
            "user_id":       uid,
            "product_id":    pid,
            "price":         prod["price"],
            "purchase_date": f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
        })

    return pd.DataFrame(rows)


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating products …")
    products_df = generate_products(5000)
    products_df.to_csv(out_dir / "products_raw.csv", index=False)
    print(f"  Saved {len(products_df)} products → {out_dir}/products_raw.csv")

    print("Generating users …")
    users_df = generate_users(2000)
    users_df.to_csv(out_dir / "users.csv", index=False)
    print(f"  Saved {len(users_df)} users → {out_dir}/users.csv")

    print("Generating invoices …")
    inv_df = generate_invoices(products_df, users_df, 15000)
    inv_df.to_csv(out_dir / "invoices.csv", index=False)
    print(f"  Saved {len(inv_df)} invoices → {out_dir}/invoices.csv")

    print("\nSample data ready in data/raw/")


if __name__ == "__main__":
    main()
