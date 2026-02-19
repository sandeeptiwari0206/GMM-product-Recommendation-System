#!/usr/bin/env python3
"""
scripts/run_local.py
====================
One-command local pipeline runner.

Steps
-----
1. Preprocessing
2. Training
3. Evaluation
4. Test recommendations for a sample user

Usage
-----
    python scripts/run_local.py
    python scripts/run_local.py --user-id USR_00042 --strategy value
    python scripts/run_local.py --skip-training   # use existing artifacts
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_logger, load_config, get_paths
from src.data.preprocessing import run_preprocessing
from src.models.training    import run_training
from src.models.evaluation  import run_evaluation
from src.models.recommender import load_engine

logger = get_logger("run_local")


def main():
    parser = argparse.ArgumentParser(description="Run full GMM pipeline locally")
    parser.add_argument("--config",         default="config/config.yaml")
    parser.add_argument("--user-id",        default="USR_00001")
    parser.add_argument("--top-n",          type=int,   default=10)
    parser.add_argument("--strategy",       default="balanced",
                        choices=["balanced", "popular", "value", "new", "diverse"])
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip training and use existing artifacts")
    parser.add_argument("--skip-eval",      action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logger.info("=" * 65)
    logger.info("  GMM Product Recommendation — Local Pipeline")
    logger.info("=" * 65)

    # STEP 1 + 2: Preprocess & Train
    if not args.skip_training:
        logger.info("\n[Step 1/3] Training …")
        best = run_training(cfg)
        logger.info("  Best hyperparams: %s", best)
    else:
        logger.info("[Step 1/3] Skipped (--skip-training)")

    # STEP 3: Evaluate
    if not args.skip_eval:
        logger.info("\n[Step 2/3] Evaluation …")
        metrics = run_evaluation(cfg)
        m = metrics["metrics"]
        logger.info(
            "  BIC=%.2f  Silhouette=%.4f  DB-Index=%.4f",
            m["bic"], m["silhouette_score"], m["davies_bouldin_index"],
        )
    else:
        logger.info("[Step 2/3] Skipped (--skip-eval)")

    # STEP 4: Recommend
    logger.info("\n[Step 3/3] Recommendations …")
    engine = load_engine(cfg)
    result = engine.recommend(
        user_id=args.user_id,
        top_n=args.top_n,
        strategy=args.strategy,
    )

    logger.info("\n" + "=" * 65)
    logger.info("  Recommendations for user: %s  |  strategy: %s", args.user_id, args.strategy)
    logger.info("  Preferred clusters: %s", result["preferred_clusters"][:3])
    logger.info("=" * 65)

    for i, rec in enumerate(result["recommendations"], start=1):
        logger.info(
            "  %2d. %-30s  $%-8.2f  ⭐ %.1f  cluster=%d  score=%.4f",
            i,
            rec["product_id"],
            rec["price"],
            rec["rating"],
            rec["cluster_id"],
            rec["final_score"],
        )
        logger.info("      Reasons: %s", " · ".join(rec["reasons"]))

    # Save full result
    paths    = get_paths(cfg)
    out_path = paths["output_dir"] + "/recommendations_sample.json"
    import os; os.makedirs(paths["output_dir"], exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("\nFull result saved → %s", out_path)


if __name__ == "__main__":
    main()
