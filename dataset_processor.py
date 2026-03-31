from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

INPUT_PATH   = Path("msmarco_vn_datasets.parquet")
SPLIT_DIR    = Path("splits")
TRAIN_RATIO  = 0.80
DEV_RATIO    = 0.10
SEED         = 42

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 64
DELTA_THRESHOLD = 0.05


def encode(model, texts, batch_size=BATCH_SIZE):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_tensor=True
            )
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def filter_triplets(triplets: list[dict]) -> list[dict]:
    model = SentenceTransformer(MODEL_NAME)
    model.eval()

    queries = ["query: " + t["query"] for t in triplets]
    positives = ["passage: " + t["positive"] for t in triplets]
    negatives = [
        "passage: " + t["negatives"][0] if len(t["negatives"]) > 0 else ""
        for t in triplets
    ]

    q_emb = encode(model, queries)
    p_emb = encode(model, positives)
    n_emb = encode(model, negatives)

    sim_qp = (q_emb * p_emb).sum(dim=1)
    sim_qn = (q_emb * n_emb).sum(dim=1)

    gap = sim_qp - sim_qn

    filtered = []
    for i, t in enumerate(triplets):
        if gap[i] > DELTA_THRESHOLD:
            filtered.append(t)

    logger.info("Filtered: %d -> %d", len(triplets), len(filtered))
    return filtered


def build_triplets(df: pd.DataFrame) -> list[dict]:
    triplets = df[["query", "positive", "negatives"]].to_dict("records")
    triplets = filter_triplets(triplets)
    return triplets


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
        train_triplets, dev_triplets, test_triplets = load_splits(split_dir)
    else:
        df = pd.read_parquet(input_path)
        triplets = build_triplets(df)
        train_triplets, dev_triplets, test_triplets = split_by_query(triplets, seed=seed)
        save_splits(train_triplets, dev_triplets, test_triplets, split_dir)

    return train_triplets, dev_triplets, test_triplets