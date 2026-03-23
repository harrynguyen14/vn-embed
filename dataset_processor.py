from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

INPUT_PATH   = Path("filtered_bkai.parquet")
SPLIT_DIR    = Path("splits")
MAX_HARD_NEG = 3
TRAIN_RATIO  = 0.90
DEV_RATIO    = 0.05
SEED         = 42


def build_triplets(df: pd.DataFrame, max_hard_neg: int = MAX_HARD_NEG) -> list[dict]:
    def select_hard_negs(row):
        negs = row["negatives"]
        if len(negs) < max_hard_neg:
            return None
        return {"query": row["query"], "positive": row["positive"], "negatives": list(negs[:max_hard_neg])}

    triplets = [r for r in df.apply(select_hard_negs, axis=1) if r is not None]
    logger.info("Built %d triplets from %d records", len(triplets), len(df))
    return triplets


def split_by_query(
    triplets: list[dict],
    train_ratio: float = TRAIN_RATIO,
    dev_ratio: float   = DEV_RATIO,
    seed: int          = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng     = random.Random(seed)
    queries = list({t["query"] for t in triplets})
    rng.shuffle(queries)

    n            = len(queries)
    n_dev        = max(1, int(n * dev_ratio))
    n_test       = max(1, int(n * (1 - train_ratio - dev_ratio)))
    dev_queries  = set(queries[:n_dev])
    test_queries = set(queries[n_dev : n_dev + n_test])
    train_queries = set(queries[n_dev + n_test:])

    train = [t for t in triplets if t["query"] in train_queries]
    dev   = [t for t in triplets if t["query"] in dev_queries]
    test  = [t for t in triplets if t["query"] in test_queries]

    logger.info(
        "Split — train: %d | dev: %d | test: %d  (queries: %d / %d / %d)",
        len(train), len(dev), len(test),
        len(train_queries), len(dev_queries), len(test_queries),
    )
    return train, dev, test


def save_splits(
    train: list[dict],
    dev: list[dict],
    test: list[dict],
    split_dir: Path = SPLIT_DIR,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        path = split_dir / f"{name}.parquet"
        pd.DataFrame(data).to_parquet(path, index=False)
        logger.info("Saved %d records to %s", len(data), path)


def load_splits(split_dir: Path = SPLIT_DIR) -> tuple[list[dict], list[dict], list[dict]]:
    result = []
    for name in ("train", "dev", "test"):
        path = split_dir / f"{name}.parquet"
        result.append(pd.read_parquet(path).to_dict("records"))
    return result[0], result[1], result[2]


def build_dataloaders(
    input_path: Path  = INPUT_PATH,
    split_dir: Path  = SPLIT_DIR,
    max_hard_neg: int = MAX_HARD_NEG,
    seed: int         = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:

    train_path = split_dir / "train.parquet"
    if train_path.exists():
        logger.info("Loading splits from %s", split_dir)
        train_triplets, dev_triplets, test_triplets = load_splits(split_dir)
    else:
        logger.info("Building splits from %s", input_path)
        df = pd.read_parquet(input_path)
        triplets = build_triplets(df, max_hard_neg=max_hard_neg)
        train_triplets, dev_triplets, test_triplets = split_by_query(triplets, seed=seed)
        save_splits(train_triplets, dev_triplets, test_triplets, split_dir)

    return train_triplets, dev_triplets, test_triplets


