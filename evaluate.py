from __future__ import annotations

import numpy as np


def evaluate_retrieval(
    model,
    triplets: list[dict],
    batch_size: int = 256,
    ks: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Encode tất cả query/pos/neg trong triplets rồi tính metrics.
    triplets: list of {"query", "positive", "negatives": list[str]}

    Mỗi query cạnh tranh với đúng hard neg của chính nó (không cross-query).
    Rank = số hard neg có score > pos + 1.
    """
    queries   = [t["query"]    for t in triplets]
    positives = [t["positive"] for t in triplets]
    # negatives: list[list[str]], mỗi query có n_neg neg
    all_negs  = [neg for t in triplets for neg in t["negatives"]]
    n_neg     = len(triplets[0]["negatives"])  # số hard neg mỗi query (cố định = 3)

    q_embs   = model.encode(queries,   batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    pos_embs = model.encode(positives, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    neg_embs = model.encode(all_negs,  batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

    # Reshape neg_embs: (N, n_neg, dim)
    neg_embs = neg_embs.reshape(len(triplets), n_neg, -1)

    scores_pos = np.sum(q_embs * pos_embs, axis=1)                     # (N,)
    scores_neg = np.einsum("nd,nkd->nk", q_embs, neg_embs)             # (N, n_neg)

    rank = 1 + np.sum(scores_neg > scores_pos[:, None], axis=1)        # (N,)

    results = {}
    for k in ks:
        rr  = np.where(rank <= k, 1.0 / rank, 0.0)
        dcg = np.where(rank <= k, 1.0 / np.log2(rank + 1), 0.0)
        results[f"MRR@{k}"]    = float(rr.mean())
        results[f"Recall@{k}"] = float((rank <= k).mean())
        results[f"nDCG@{k}"]   = float(dcg.mean())

    return results
