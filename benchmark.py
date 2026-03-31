from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DATASETS = {
    "zalo-legal":  "GreenNode/zalo-ai-legal-text-retrieval-vn",
    "hotpotqa":    "GreenNode/hotpotqa-vn",
    "scifact":     "GreenNode/scifact-vn",
    "nfcorpus":    "GreenNode/nfcorpus-vn",
}


def load_retrieval_dataset(dataset_id: str):
    """Load corpus, queries, qrels theo chuẩn MTEB."""
    logger.info("Loading corpus ...")
    corpus_ds  = load_dataset(dataset_id, "corpus",  split="test")
    logger.info("Loading queries ...")
    queries_ds = load_dataset(dataset_id, "queries", split="test")
    logger.info("Loading qrels ...")
    qrels_ds   = load_dataset(dataset_id, "qrels",   split="test")

    # corpus: id → text (title + text)
    corpus = {}
    for row in corpus_ds:
        title = (row.get("title") or "").strip()
        text  = (row.get("text")  or "").strip()
        corpus[str(row["id"])] = f"{title}. {text}".strip(". ") if title else text

    # queries: id → text
    queries = {str(row["id"]): row["text"] for row in queries_ds}

    # qrels: query_id → set of relevant corpus_ids
    qrels: dict[str, set[str]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        if int(row["score"]) > 0:
            qrels.setdefault(qid, set()).add(cid)

    logger.info("Corpus: %d | Queries: %d | Qrels: %d",
                len(corpus), len(queries), len(qrels))
    return corpus, queries, qrels


def evaluate(
    model: SentenceTransformer,
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    batch_size: int,
    ks: list[int] = [1, 5, 10],
    score_batch: int = 512,
) -> dict[str, float]:
    corpus_ids   = list(corpus.keys())
    corpus_texts = ["passage: " + corpus[cid] for cid in corpus_ids]
    query_ids    = list(queries.keys())
    query_texts  = ["query: "   + queries[qid] for qid in query_ids]

    logger.info("Encoding %d corpus passages ...", len(corpus_texts))
    c_embs = model.encode(
        corpus_texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype("float32")

    logger.info("Encoding %d queries ...", len(query_texts))
    q_embs = model.encode(
        query_texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype("float32")

    corpus_id_arr = np.array(corpus_ids)
    max_k = max(ks)

    mrr    = {k: [] for k in ks}
    recall = {k: [] for k in ks}
    ndcg   = {k: [] for k in ks}

    for start in range(0, len(query_ids), score_batch):
        end        = min(start + score_batch, len(query_ids))
        scores     = q_embs[start:end] @ c_embs.T           # [batch, corpus]
        top_k_idx  = np.argpartition(scores, -max_k, axis=1)[:, -max_k:]
        # sort within top_k
        for bi, qi in enumerate(range(start, end)):
            qid       = query_ids[qi]
            relevant  = qrels.get(qid, set())
            if not relevant:
                continue

            row_scores = scores[bi, top_k_idx[bi]]
            sorted_local = np.argsort(-row_scores)
            ranked_cids  = corpus_id_arr[top_k_idx[bi][sorted_local]].tolist()

            for k in ks:
                top = ranked_cids[:k]
                hits = [1 if cid in relevant else 0 for cid in top]

                # MRR@k
                rr = 0.0
                for rank, h in enumerate(hits, 1):
                    if h:
                        rr = 1.0 / rank
                        break
                mrr[k].append(rr)

                # Recall@k
                recall[k].append(len(set(top) & relevant) / len(relevant))

                # nDCG@k
                dcg  = sum(h / np.log2(r + 1) for r, h in enumerate(hits, 1))
                idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
                ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

    results = {}
    for k in ks:
        results[f"MRR@{k}"]    = float(np.mean(mrr[k]))
        results[f"Recall@{k}"] = float(np.mean(recall[k]))
        results[f"nDCG@{k}"]   = float(np.mean(ndcg[k]))
    return results


def run(args: argparse.Namespace) -> None:
    dataset_id = DATASETS.get(args.dataset, args.dataset)
    logger.info("Benchmark: %s", dataset_id)
    logger.info("Model: %s", args.model_path)

    model = SentenceTransformer(args.model_path, trust_remote_code=True)

    import torch
    if torch.cuda.is_available():
        model = model.cuda()

    corpus, queries, qrels = load_retrieval_dataset(dataset_id)
    results = evaluate(model, corpus, queries, qrels, batch_size=args.batch_size)

    logger.info("=== RESULTS: %s ===", args.dataset)
    for k, v in results.items():
        logger.info("  %s: %.4f", k, v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark retrieval trên VN-MTEB datasets")
    p.add_argument("--model-path", required=True,
                   help="Path to trained model hoặc HuggingFace model ID")
    p.add_argument("--dataset",    default="zalo-legal",
                   help=f"Tên dataset ({', '.join(DATASETS.keys())}) hoặc HF dataset ID")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
