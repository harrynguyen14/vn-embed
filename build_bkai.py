"""
Pipeline xây dựng hard negatives cho bkai-foundation-models/NewsSapo

Flow:
  1. Load dataset -> lấy (title, sapo)
  2. Embed toàn bộ sapo bằng multilingual-e5-base (float16)
  3. Build FAISS IndexIVFFlat
  4. Với mỗi title, search top-50 sapo -> rerank bằng cross-encoder -> lấy 3 hard neg
  5. Lưu ra filtered_bkai.parquet (cùng schema với filtered_pairs.parquet)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BI_ENCODER_NAME = "intfloat/multilingual-e5-base"
CE_MODEL_NAME   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DATASET_NAME    = "bkai-foundation-models/NewsSapo"
OUTPUT_PATH     = Path("filtered_bkai.parquet")
BATCH_SIZE      = 256   
TOP_K           = 50   
CE_TOP_K        = 10  
MAX_HARD_NEG    = 3     
IVF_NLIST       = 4096
IVF_NPROBE      = 128
MAX_ROWS        = 500000 
# ─────────────────────────────────────────────────────────────────────────────


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    prefix: str = "",
    desc: str = "Encoding texts",
) -> np.ndarray:

    if prefix:
        texts = [f"{prefix}: {t}" for t in texts]
    logger.info("%s (%d texts)...", desc, len(texts))
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vecs.astype(np.float16)


def build_faiss_index(vecs: np.ndarray) -> faiss.Index:
    vecs32 = vecs.astype(np.float32)
    dim = vecs32.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
    logger.info("Training IVF index on %d vectors...", len(vecs32))
    index.train(vecs32)
    index.nprobe = IVF_NPROBE
    index.add(vecs32)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


def mine_hard_negatives(
    titles: list[str],
    sapos: list[str],
    title_vecs: np.ndarray,
    index: faiss.Index,
    cross_encoder: CrossEncoder,
    chunk_size: int = 1_000,
) -> list[dict]:

    logger.info("Searching FAISS for %d titles (top_k=%d)...", len(titles), TOP_K)
    scores_all, indices_all = index.search(title_vecs.astype(np.float32), TOP_K + 1)

    candidates_per_title: list[list[int]] = []
    for i in range(len(titles)):
        cands = []
        for idx in indices_all[i]:
            if idx == i:
                continue
            cands.append(int(idx))
            if len(cands) >= CE_TOP_K:
                break
        candidates_per_title.append(cands)

    records = []
    logger.info("Running cross-encoder rerank in chunks of %d...", chunk_size)

    for chunk_start in tqdm(range(0, len(titles), chunk_size), desc="CE rerank"):
        chunk_end = min(chunk_start + chunk_size, len(titles))

        flat_pairs: list[tuple[str, str]] = []
        flat_meta:  list[tuple[int, int]] = []

        for i in range(chunk_start, chunk_end):
            for cand_idx in candidates_per_title[i]:
                flat_pairs.append((titles[i], sapos[cand_idx]))
                flat_meta.append((i, cand_idx))

        ce_scores = cross_encoder.predict(
            flat_pairs,
            batch_size=64,
            show_progress_bar=True,
        )

        from collections import defaultdict
        title_ce: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (title_idx, cand_idx), ce_score in zip(flat_meta, ce_scores):
            title_ce[title_idx].append((cand_idx, float(ce_score)))

        for i in range(chunk_start, chunk_end):
            ranked = sorted(title_ce[i], key=lambda x: x[1])[:MAX_HARD_NEG]
            if len(ranked) < MAX_HARD_NEG:
                continue  # bỏ qua nếu không đủ hard neg
            records.append({
                "query":    titles[i],
                "positive": sapos[i],
                "negatives": [sapos[cand_idx] for cand_idx, _ in ranked],
                "neg_scores": [ce_score for _, ce_score in ranked],
            })

    logger.info("Total records: %d (positives + hard negatives)", len(records))
    return records


def main() -> None:
    # ── 1. Load dataset ───────────────────────────────────────────────────────
    logger.info("Loading dataset %s...", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split="train")
    if MAX_ROWS:
        indices = np.random.choice(len(ds), size=min(MAX_ROWS, len(ds)), replace=False)
        ds = ds.select(indices.tolist())

    titles = [str(row["title"]) if row["title"] is not None else "" for row in ds]
    sapos  = [str(row["sapo"])  if row["sapo"]  is not None else "" for row in ds]
    logger.info("Loaded %d rows", len(titles))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 2. Bi-encoder encode (với cache) ──────────────────────────────────────
    cache_title = OUTPUT_PATH.with_name("cache_title_vecs.npy")
    cache_sapo  = OUTPUT_PATH.with_name("cache_sapo_vecs.npy")

    if cache_title.exists() and cache_sapo.exists():
        logger.info("Loading cached vectors from %s / %s", cache_title, cache_sapo)
        title_vecs = np.load(cache_title)
        sapo_vecs  = np.load(cache_sapo)
    else:
        logger.info("Loading bi-encoder %s on %s...", BI_ENCODER_NAME, device)
        bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device=device)

        if not cache_title.exists():
            title_vecs = encode_texts(bi_encoder, titles, prefix="query", desc="Encoding titles")
            np.save(cache_title, title_vecs)
            logger.info("Cached title vectors saved to %s", cache_title)
        else:
            title_vecs = np.load(cache_title)
            logger.info("Loaded cached title vectors from %s", cache_title)

        del bi_encoder
        torch.cuda.empty_cache()
        import gc; gc.collect()

        logger.info("Loading bi-encoder again for sapo encoding...")
        bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device=device)

        if not cache_sapo.exists():
            sapo_vecs = encode_texts(bi_encoder, sapos, prefix="passage", desc="Encoding sapos")
            np.save(cache_sapo, sapo_vecs)
            logger.info("Cached sapo vectors saved to %s", cache_sapo)
        else:
            sapo_vecs = np.load(cache_sapo)
            logger.info("Loaded cached sapo vectors from %s", cache_sapo)

        del bi_encoder
        torch.cuda.empty_cache()
        import gc; gc.collect()

    # ── 3. Build FAISS index ──────────────────────────────────────────────────
    index = build_faiss_index(sapo_vecs)

    # ── 4. Load cross-encoder ─────────────────────────────────────────────────
    logger.info("Loading cross-encoder %s on %s...", CE_MODEL_NAME, device)
    cross_encoder = CrossEncoder(CE_MODEL_NAME, device=device)

    # ── 5. Mine + rerank ──────────────────────────────────────────────────────
    records = mine_hard_negatives(titles, sapos, title_vecs, index, cross_encoder)

    # ── 6. Lưu parquet ───────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved %d records to %s", len(df), OUTPUT_PATH)
    logger.info("Schema: query | positive | negatives (list of %d) | neg_scores", MAX_HARD_NEG)


if __name__ == "__main__":
    main()
