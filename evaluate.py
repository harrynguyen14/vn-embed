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
    """
    Full-corpus retrieval benchmark:
    - Corpus = tất cả positives (unique) + tất cả negatives trong tập triplets
    - Mỗi query tìm top-K trong corpus, tính rank của positive đúng
    - Metrics: MRR@K, Recall@K, nDCG@K (chuẩn BEIR/MTEB)
    """
    if not triplets:
        return {}

    queries   = [t["query"]    for t in triplets]
    positives = [t["positive"] for t in triplets]

    # Xây corpus: union của tất cả passages (pos + neg), dedup theo text
    corpus_set: dict[str, int] = {}
    for t in triplets:
        for p in [t["positive"]] + list(t["negatives"]):
            if p not in corpus_set:
                corpus_set[p] = len(corpus_set)

    corpus_texts = list(corpus_set.keys())
    pos_indices  = np.array([corpus_set[p] for p in positives])  # ground-truth idx trong corpus

    logger.info("Corpus size: %d | Queries: %d", len(corpus_texts), len(queries))

    q_embs = model.encode(
        queries, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
    ).astype("float32")
    c_embs = model.encode(
        corpus_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
    ).astype("float32")

    # scores: [n_queries, corpus_size]
    scores = q_embs @ c_embs.T  # cosine vì đã normalize

    max_k = max(ks)
    # rank của positive trong từng query (1-based)
    ranks = np.array([
        int(np.sum(scores[i] >= scores[i, pos_indices[i]]))
        for i in range(len(queries))
    ])  # số docs có score >= score(positive) = rank của positive

    results = {}
    for k in ks:
        in_k = ranks <= k
        rr   = np.where(in_k, 1.0 / ranks, 0.0)
        dcg  = np.where(in_k, 1.0 / np.log2(ranks + 1), 0.0)
        idcg = 1.0  # ideal: positive ở rank 1

        results[f"MRR@{k}"]    = float(rr.mean())
        results[f"Recall@{k}"] = float(in_k.mean())
        results[f"nDCG@{k}"]   = float((dcg / idcg).mean())

    return results