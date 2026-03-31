from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from dataset_processor import load_splits
from evaluate import evaluate_retrieval

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def test(args: argparse.Namespace) -> None:
    logger.info("Loading model from %s", args.model_path)
    model = SentenceTransformer(args.model_path)
    model.eval()

    _, _, test_triplets = load_splits(Path(args.splits_dir))

    logger.info("Evaluating on %d test triplets...", len(test_triplets))
    metrics = evaluate_retrieval(
        model,
        test_triplets,
        batch_size=args.eval_batch_size,
        ks=[1, 5, 10],
    )

    logger.info("=== TEST RESULTS ===")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--eval-batch-size", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    test(parse_args())