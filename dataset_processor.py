from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

INPUT_PATH   = Path("msmarco_vn_datasets.parquet")
SPLIT_DIR    = Path("splits")
TRAIN_RATIO  = 0.80
DEV_RATIO    = 0.10
SEED         = 42


def build_triplets(df: pd.DataFrame) -> list[dict]:
    raw_triplets = df[["query", "positive", "negatives"]].to_dict("records")

    seen_positives: set[str] = set()
    deduped = []
    for t in raw_triplets:
        if t["positive"] not in seen_positives:
            seen_positives.add(t["positive"])
            deduped.append(t)

    logger.info("Sau khi khử trùng: còn %d triplets (loại bỏ %d)", len(deduped), len(raw_triplets) - len(deduped))
    return deduped


def split_by_query(
    triplets: list[dict],
    train_ratio: float = TRAIN_RATIO,
    dev_ratio: float   = DEV_RATIO,
    seed: int          = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    
    rng = random.Random(seed)
 
    unique_queries = list({t["query"] for t in triplets})

    rng.shuffle(unique_queries)

    n = len(unique_queries)
    n_dev = max(1, int(n * dev_ratio))
    n_test = max(1, int(n * (1.0 - train_ratio - dev_ratio)))
 
    dev_q_set   = set(unique_queries[:n_dev])
    test_q_set  = set(unique_queries[n_dev : n_dev + n_test])
    train_q_set = set(unique_queries[n_dev + n_test:])

    train = [t for t in triplets if t["query"] in train_q_set]
    dev   = [t for t in triplets if t["query"] in dev_q_set]
    test  = [t for t in triplets if t["query"] in test_q_set]

    assert len(dev_q_set.intersection(train_q_set)) == 0
    assert len(test_q_set.intersection(train_q_set)) == 0

    logger.info(
        "Random Split thành công: Train: %d | Dev: %d | Test: %d",
        len(train), len(dev), len(test)
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
    input_path: Path = INPUT_PATH,
    split_dir: Path  = SPLIT_DIR,
    seed: int        = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:

    train_path = split_dir / "train.parquet"
    if train_path.exists():
        logger.info("Loading splits from %s", split_dir)
        train_triplets, dev_triplets, test_triplets = load_splits(split_dir)
    else:
        logger.info("Building splits from %s", input_path)
        df = pd.read_parquet(input_path)
        triplets = build_triplets(df)
        train_triplets, dev_triplets, test_triplets = split_by_query(triplets, seed=seed)
        save_splits(train_triplets, dev_triplets, test_triplets, split_dir)

    return train_triplets, dev_triplets, test_triplets


