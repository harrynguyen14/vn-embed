from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


def evaluate_retrieval(
    model,
    triplets: list[dict],
    batch_size: int = 256,
    ks: list[int] = [1, 5, 10],
) -> dict[str, float]:

    if not triplets:
        return {}

    queries = ["query: " + t["query"] for t in triplets]
    raw_positives = [t["positive"] for t in triplets]

    corpus_set = {}
    for t in triplets:
        for p in [t["positive"]] + list(t["negatives"]):
            if p not in corpus_set:
                corpus_set[p] = len(corpus_set)

    corpus_texts = list(corpus_set.keys())
    pos_indices = np.array([corpus_set[p] for p in raw_positives])

    logger.info("Corpus size: %d | Queries: %d", len(corpus_texts), len(queries))

    q_embs = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    c_embs = model.encode(
        ["passage: " + t for t in corpus_texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    SCORE_BATCH = 512
    ranks = np.empty(len(queries), dtype=np.int32)

    for start in range(0, len(queries), SCORE_BATCH):
        end = min(start + SCORE_BATCH, len(queries))
        scores_batch = q_embs[start:end] @ c_embs.T

        for i, gi in enumerate(range(start, end)):
            pos_score = scores_batch[i, pos_indices[gi]]

            rank = np.sum(scores_batch[i] > pos_score) + 1
            ranks[gi] = int(rank)

    results = {}

    for k in ks:
        in_k = ranks <= k

        rr = np.where(in_k, 1.0 / ranks, 0.0)
        dcg = np.where(in_k, 1.0 / np.log2(ranks + 1), 0.0)

        results[f"MRR@{k}"] = float(rr.mean())
        results[f"Recall@{k}"] = float(in_k.mean())
        results[f"nDCG@{k}"] = float(dcg.mean())

    return results