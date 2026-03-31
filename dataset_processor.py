from __future__ import annotations

import logging
import random
import argparse
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def build_triplets(df: pd.DataFrame) -> list[dict]:
    return df[["query", "positive", "negatives"]].to_dict("records")


def split_by_query(
    triplets: list[dict],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
):
    rng = random.Random(seed)

    unique_queries = list({t["query"] for t in triplets})
    rng.shuffle(unique_queries)

    n = len(unique_queries)
    n_dev = max(1, int(n * dev_ratio))
    n_test = max(1, int(n * (1.0 - train_ratio - dev_ratio)))

    dev_q = set(unique_queries[:n_dev])
    test_q = set(unique_queries[n_dev:n_dev + n_test])
    train_q = set(unique_queries[n_dev + n_test:])

    train = [t for t in triplets if t["query"] in train_q]
    dev   = [t for t in triplets if t["query"] in dev_q]
    test  = [t for t in triplets if t["query"] in test_q]

    logger.info("Split → Train:%d | Dev:%d | Test:%d", len(train), len(dev), len(test))
    return train, dev, test


def save_splits(train, dev, test, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        path = out_dir / f"{name}.parquet"
        pd.DataFrame(data).to_parquet(path, index=False)
        logger.info("Saved %s: %d samples", name, len(data))


def load_splits(split_dir: Path):
    return (
        pd.read_parquet(split_dir / "train.parquet").to_dict("records"),
        pd.read_parquet(split_dir / "dev.parquet").to_dict("records"),
        pd.read_parquet(split_dir / "test.parquet").to_dict("records"),
    )


def main(args):
    df = pd.read_parquet(args.input)
    logger.info("Loaded dataset: %d samples", len(df))

    triplets = build_triplets(df)

    train, dev, test = split_by_query(
        triplets,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    save_splits(train, dev, test, Path(args.output_dir))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="splits")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())