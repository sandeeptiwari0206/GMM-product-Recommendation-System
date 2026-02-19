"""
src/inference.py
================
AWS SageMaker Real-Time Endpoint Inference Handler.

SageMaker calls these 4 functions in sequence:
    model_fn   →  load artifacts once at container start
    input_fn   →  deserialise incoming HTTP request
    predict_fn →  run recommendation logic
    output_fn  →  serialise and return HTTP response

Local testing
-------------
    python src/inference.py --user-id USR_00001 --top-n 5

Performance
-----------
- Engine is cached at module level → loaded once, reused for every request
- Target: <200ms per request (single user)
- Supports batch requests (list of users in one call)
- Graceful fallback for unknown / cold-start users
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.utils import get_logger, load_config
from src.models.recommender import GMMRecommendationEngine, load_engine

logger = get_logger(__name__)

# Module-level cache — populated once by model_fn
_ENGINE: Optional[GMMRecommendationEngine] = None
_CFG:    Optional[Dict[str, Any]]          = None


# =============================================================================
# 1. model_fn  — called ONCE when the SageMaker container starts
# =============================================================================

def model_fn(model_dir: str) -> GMMRecommendationEngine:
    """
    Load model artifacts and return a ready GMMRecommendationEngine.

    SageMaker passes the extracted model artifact directory as model_dir.
    The engine is cached so subsequent calls return immediately.
    """
    global _ENGINE, _CFG

    if _ENGINE is not None:
        logger.debug("model_fn: returning cached engine")
        return _ENGINE

    t0 = time.monotonic()
    logger.info("model_fn: loading engine from %s …", model_dir)

    # Override SM_MODEL_DIR so config_loader resolves correct paths
    os.environ.setdefault("SM_MODEL_DIR", model_dir)

    cfg     = load_config()
    engine  = load_engine(cfg)

    _ENGINE = engine
    _CFG    = cfg

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info("model_fn: engine loaded in %.1f ms", elapsed_ms)
    return engine


# =============================================================================
# 2. input_fn  — deserialise incoming request
# =============================================================================

def input_fn(
    request_body: Union[str, bytes],
    content_type: str = "application/json",
) -> Dict[str, Any]:
    """
    Parse and validate incoming HTTP request.

    Supports
    --------
    application/json → single request or batch {"requests": [...]}
    text/csv         → one user_id per line (minimal format)

    Single request schema
    ---------------------
    {
        "user_id":          "USR_00001",   # required
        "top_n":            10,            # optional (default 10)
        "strategy":         "balanced",    # optional
        "diversity_weight": 0.3,           # optional
        "category_filter":  null,          # optional
        "price_min":        0,             # optional
        "price_max":        1000           # optional
    }
    """
    ct = content_type.lower().split(";")[0].strip()

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    if ct == "application/json":
        payload  = json.loads(request_body)
        requests = payload.get("requests", [payload]) if isinstance(payload, dict) else payload

    elif ct == "text/csv":
        user_ids = [l.strip() for l in request_body.strip().splitlines() if l.strip()]
        requests = [{"user_id": uid} for uid in user_ids]

    else:
        raise ValueError(f"Unsupported content_type: '{ct}'. Use application/json or text/csv.")

    return {"requests": [_validate(r) for r in requests]}


def _validate(raw: Dict) -> Dict:
    """Apply defaults and type-cast one request dict."""
    if not raw.get("user_id"):
        raise ValueError("Each request must contain a non-empty 'user_id'.")
    return {
        "user_id":          str(raw["user_id"]),
        "top_n":            int(raw.get("top_n",            10)),
        "strategy":         str(raw.get("strategy",         "balanced")),
        "diversity_weight": float(raw.get("diversity_weight", 0.3)),
        "category_filter":  raw.get("category_filter",     None),
        "price_min":        float(raw.get("price_min",      0.0)),
        "price_max":        float(raw.get("price_max",      1_000_000.0)),
    }


# =============================================================================
# 3. predict_fn  — run recommendations
# =============================================================================

def predict_fn(
    input_data: Dict[str, Any],
    model:      GMMRecommendationEngine,
) -> List[Dict[str, Any]]:
    """
    Execute recommendation for every request in the batch.

    Each result includes a latency_ms field.
    Failures are caught per-request and returned as fallback responses.
    """
    results = []
    for req in input_data["requests"]:
        t0 = time.monotonic()
        try:
            result = model.recommend(
                user_id=req["user_id"],
                top_n=req["top_n"],
                strategy=req["strategy"],
                diversity_weight=req["diversity_weight"],
                category_filter=req["category_filter"],
                price_min=req["price_min"],
                price_max=req["price_max"],
            )
            ms = (time.monotonic() - t0) * 1000
            result["latency_ms"] = round(ms, 2)

            if ms > 200:
                logger.warning(
                    "Latency %.1f ms > 200 ms target for user=%s", ms, req["user_id"]
                )

        except Exception as exc:
            logger.exception("predict_fn error for user=%s: %s", req["user_id"], exc)
            result = _fallback(req, str(exc))

        results.append(result)

    return results


def _fallback(req: Dict, error: str) -> Dict:
    """Graceful fallback response when prediction fails."""
    return {
        "user_id":               req["user_id"],
        "strategy":              req.get("strategy", "balanced"),
        "preferred_clusters":    [],
        "cluster_probabilities": [],
        "n_recommendations":     0,
        "recommendations":       [],
        "error":                 error,
        "fallback":              True,
    }


# =============================================================================
# 4. output_fn  — serialise response
# =============================================================================

def output_fn(
    predictions: List[Dict[str, Any]],
    accept:      str = "application/json",
) -> Tuple[str, str]:
    """
    Serialise prediction results to the HTTP response body.

    Supports
    --------
    application/json → JSON object (single) or JSON array (batch)
    text/csv         → user_id, rank, product_id, score per row
    """
    ac = accept.lower().split(";")[0].strip()

    if ac in ("application/json", "*/*", ""):
        body = json.dumps(predictions[0] if len(predictions) == 1 else predictions, default=str)
        return body, "application/json"

    elif ac == "text/csv":
        rows = []
        for pred in predictions:
            for rank, rec in enumerate(pred.get("recommendations", []), start=1):
                rows.append({
                    "user_id":    pred["user_id"],
                    "rank":       rank,
                    "product_id": rec["product_id"],
                    "score":      rec["final_score"],
                    "category":   rec.get("category", ""),
                    "reasons":    " | ".join(rec.get("reasons", [])),
                })
        body = pd.DataFrame(rows).to_csv(index=False)
        return body, "text/csv"

    else:
        raise ValueError(f"Unsupported accept type: '{ac}'")


# =============================================================================
# Local Testing Harness
# =============================================================================
if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = argparse.ArgumentParser(description="Test inference handler locally")
    parser.add_argument("--user-id",         default="USR_00001")
    parser.add_argument("--config",          default=None)
    parser.add_argument("--top-n",           type=int,   default=5)
    parser.add_argument("--strategy",        default="balanced")
    parser.add_argument("--price-min",       type=float, default=0)
    parser.add_argument("--price-max",       type=float, default=1_000_000)
    parser.add_argument("--accept",          default="application/json")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    from src.utils import get_paths
    paths  = get_paths(cfg)
    engine = model_fn(paths["artifacts_dir"])

    raw = json.dumps({
        "user_id":          args.user_id,
        "top_n":            args.top_n,
        "strategy":         args.strategy,
        "diversity_weight": 0.3,
        "price_min":        args.price_min,
        "price_max":        args.price_max,
    })

    parsed = input_fn(raw, "application/json")
    preds  = predict_fn(parsed, engine)
    body, ct = output_fn(preds, args.accept)

    print(f"\nContent-Type: {ct}\n")
    print(body)
