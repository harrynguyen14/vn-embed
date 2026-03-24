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

    queries   = [t["query"]    for t in triplets]
    positives = [t["positive"] for t in triplets]
    all_negs  = [neg for t in triplets for neg in t["negatives"]]

    n_queries = len(triplets)
    n_neg     = len(triplets[0]["negatives"]) 

    q_embs = model.encode(
        queries, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    pos_embs = model.encode(
        positives, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    neg_embs = model.encode(
        all_negs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )

    neg_embs = neg_embs.reshape(n_queries, n_neg, -1)

    scores_pos = np.sum(q_embs * pos_embs, axis=1)

    scores_neg = np.einsum("nd,nkd->nk", q_embs, neg_embs)

    rank = 1 + np.sum(scores_neg > scores_pos[:, None], axis=1)

    results = {}
    for k in ks:
        rr  = np.where(rank <= k, 1.0 / rank, 0.0)
        dcg = np.where(rank <= k, 1.0 / np.log2(rank + 1), 0.0)
        
        results[f"MRR@{k}"]    = float(rr.mean())
        results[f"Recall@{k}"] = float((rank <= k).mean())
        results[f"nDCG@{k}"]   = float(dcg.mean())

    return results