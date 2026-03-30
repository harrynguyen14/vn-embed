from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dataset_processor import build_dataloaders
from evaluate import evaluate_retrieval
from model import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def test(args: argparse.Namespace) -> None:
    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)

    _, _, test_triplets = build_dataloaders(
        input_path = Path(args.data),
        split_dir  = Path(args.splits_dir),
        seed       = args.seed,
    )

    logger.info("Evaluating on %d test triplets...", len(test_triplets))
    metrics = evaluate_retrieval(model, test_triplets, batch_size=args.eval_batch_size, ks=[1, 5, 10])

    logger.info("=== TEST RESULTS ===")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test trained GTE embedding model")
    p.add_argument("--model-path",     required=True, help="Path to saved model directory")
    p.add_argument("--data",            default="msmarco_vn_datasets.parquet")
    p.add_argument("--splits-dir",      default="splits")
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    test(parse_args())
