from __future__ import annotations

import logging
import argparse
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 64
DELTA_THRESHOLD = 0.05


def encode(model, texts, batch_size=BATCH_SIZE):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        embs.append(emb)
    return torch.cat(embs, dim=0)


def filter_triplets(df: pd.DataFrame) -> pd.DataFrame:
    triplets = df[["query", "positive", "negatives"]].to_dict("records")

    model = SentenceTransformer(MODEL_NAME)
    model.eval()

    queries = ["query: " + t["query"] for t in triplets]
    positives = ["passage: " + t["positive"] for t in triplets]
    negatives = [
        "passage: " + t["negatives"][0] if len(t["negatives"]) > 0 else "passage: " + t["positive"]
        for t in triplets
    ]

    q_emb = encode(model, queries)
    p_emb = encode(model, positives)
    n_emb = encode(model, negatives)

    sim_qp = (q_emb * p_emb).sum(dim=1)
    sim_qn = (q_emb * n_emb).sum(dim=1)
    gap = sim_qp - sim_qn

    keep_mask = gap > DELTA_THRESHOLD
    kept = keep_mask.sum().item()

    logger.info("Filtered: %d → %d (%.2f%% kept)", len(df), kept, kept / len(df) * 100)

    return df[keep_mask.cpu().numpy()]


def main(args):
    df = pd.read_parquet(args.input)
    logger.info("Loaded raw dataset: %d samples", len(df))

    df_clean = filter_triplets(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(out_path, index=False)

    logger.info("Saved cleaned dataset → %s", out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="cleaned.parquet")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())