"""
Full-corpus retrieval evaluation.

Thay vì rank positive trong {pos + 3 negs}, script này:
1. Gộp toàn bộ positives + negatives từ test set thành 1 corpus
2. Mỗi query rank positive trong toàn bộ corpus (~vài chục nghìn docs)
3. Kết quả mới so sánh được với SOTA retrieval benchmarks
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from dataset_processor import build_dataloaders
from model import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate_full_corpus(
    model,
    triplets: list[dict],
    batch_size: int = 128,
    ks: list[int] = [1, 5, 10, 100],
) -> dict[str, float]:
    # Build corpus: tất cả unique docs (positives + negatives)
    corpus_set: dict[str, int] = {}
    for t in triplets:
        if t["positive"] not in corpus_set:
            corpus_set[t["positive"]] = len(corpus_set)
        for neg in t["negatives"]:
            if neg not in corpus_set:
                corpus_set[neg] = len(corpus_set)

    corpus_texts = list(corpus_set.keys())
    logger.info("Corpus size: %d docs", len(corpus_texts))
    logger.info("Queries: %d", len(triplets))

    # Encode
    logger.info("Encoding corpus...")
    corpus_embs = model.encode(
        corpus_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    queries = [t["query"] for t in triplets]
    logger.info("Encoding queries...")
    q_embs = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Positive index trong corpus
    pos_indices = np.array([corpus_set[t["positive"]] for t in triplets])

    # Tính scores: (n_queries, corpus_size)
    logger.info("Computing scores...")
    scores = q_embs @ corpus_embs.T  # (Q, C)

    # Rank của positive với mỗi query (1-indexed)
    # argsort descending → tìm vị trí của positive
    ranks = []
    for i, pos_idx in enumerate(pos_indices):
        score_row = scores[i]
        pos_score = score_row[pos_idx]
        rank = int(np.sum(score_row > pos_score)) + 1
        ranks.append(rank)
    ranks = np.array(ranks)

    results = {}
    for k in ks:
        rr  = np.where(ranks <= k, 1.0 / ranks, 0.0)
        dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
        results[f"MRR@{k}"]    = float(rr.mean())
        results[f"Recall@{k}"] = float((ranks <= k).mean())
        results[f"nDCG@{k}"]   = float(dcg.mean())

    # Thêm median rank để debug
    results["median_rank"] = float(np.median(ranks))
    results["mean_rank"]   = float(np.mean(ranks))

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",      default="output/gte-vn")
    p.add_argument("--data",            default="filtered_bkai.parquet")
    p.add_argument("--splits-dir",      default="splits")
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--split",           default="test", choices=["train", "dev", "test"])
    return p.parse_args()


def main():
    args = parse_args()

    train, dev, test = build_dataloaders(
        input_path=Path(args.data),
        split_dir=Path(args.splits_dir),
    )
    split_map = {"train": train, "dev": dev, "test": test}
    triplets = split_map[args.split]
    logger.info("Evaluating on %s split: %d triplets", args.split, len(triplets))

    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)

    results = evaluate_full_corpus(model, triplets, batch_size=args.batch_size)

    print("\n=== FULL CORPUS RETRIEVAL RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
