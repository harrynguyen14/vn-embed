import argparse
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET    = "GreenNode/msmarco-vn"
BI_ENCODER = "intfloat/multilingual-e5-base"
DIM        = 768   # multilingual-e5-base output dim
NLIST      = 4096  # IVFPQ centroids
ENCODE_CHUNK  = 50_000   # encode 50k mỗi lần → ~150MB float32, an toàn với RAM
CHECKPOINT_EVERY = 10    # save index + ids/texts sau mỗi N chunks (mỗi 500k docs)


def _passage_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text  = (row.get("text")  or "").strip()
    return f"{title}. {text}".strip(". ") if title and title != text else text


def build_corpus_index(
    model: SentenceTransformer,
    batch_size: int,
    index_path: str,
    ids_path: str,
    texts_path: str,
    pool_size: int | None = None,
    pos_corpus_ids: list[str] | None = None,
) -> None:
    """
    Encode 8.8M corpus theo streaming chunks.
    - Không load toàn bộ embeddings vào RAM cùng lúc
    - Train FAISS trước trên 500k samples, sau đó add từng chunk
    - Lưu ids + texts ra file riêng để dùng lúc mining
    """
    logger.info("Loading corpus từ %s ...", DATASET)
    corpus_ds = load_dataset(DATASET, "corpus", split="dev", streaming=False)
    total = len(corpus_ds)
    logger.info("Corpus size: %d", total)

    # Giới hạn corpus pool nếu được yêu cầu (để chạy nhanh hơn trên Colab free)
    if pool_size and pool_size < total:
        import random as _random
        pos_ids_set = set(pos_corpus_ids or [])
        # Luôn giữ các positive ids, sample phần còn lại
        all_row_ids = list(range(total))
        _random.seed(42)
        sampled = _random.sample(all_row_ids, min(pool_size, total))
        selected_indices = sorted(set(sampled))
        corpus_ds = corpus_ds.select(selected_indices)
        total = len(corpus_ds)
        logger.info("Pool size giới hạn còn: %d", total)

    # --- Bước 1: train FAISS index trên 100k mẫu đầu ---
    TRAIN_SIZE = min(200_000, total)
    logger.info("Collecting %d samples để train FAISS ...", TRAIN_SIZE)
    train_texts = []
    for row in corpus_ds.select(range(TRAIN_SIZE)):
        train_texts.append(_passage_text(row))

    train_embs = model.encode(
        ["passage: " + t[:512] for t in train_texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    del train_texts

    if Path(index_path).exists() and Path(ids_path).exists():
        # Resume: load index đã build dở
        logger.info("Resume: loading existing index từ %s ...", index_path)
        index = faiss.read_index(index_path)
    else:
        quantizer = faiss.IndexFlatIP(DIM)
        index = faiss.IndexIVFPQ(quantizer, DIM, NLIST, 64, 8)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        logger.info("Training FAISS index ...")
        index.train(train_embs)
    del train_embs

    # --- Bước 2: encode + add từng chunk, lưu ids/texts ra disk ---
    all_ids   = []
    all_texts = []
    # Resume: load lại ids/texts đã save nếu có
    done_count = 0
    if Path(ids_path).exists() and Path(texts_path).exists():
        all_ids   = np.load(ids_path,   allow_pickle=True).tolist()
        all_texts = np.load(texts_path, allow_pickle=True).tolist()
        done_count = len(all_ids)
        logger.info("Resume: đã có %d / %d docs, bỏ qua ...", done_count, total)
    else:
        all_ids   = []
        all_texts = []

    chunk_ids   = []
    chunk_texts = []
    chunk_count = 0

    def flush_chunk():
        nonlocal chunk_ids, chunk_texts, chunk_count
        if not chunk_ids:
            return
        embs = model.encode(
            ["passage: " + t[:512] for t in chunk_texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        index.add(embs)
        all_ids.extend(chunk_ids)
        all_texts.extend(chunk_texts)
        chunk_ids.clear()
        chunk_texts.clear()
        chunk_count += 1

        # Checkpoint định kỳ
        if chunk_count % CHECKPOINT_EVERY == 0:
            faiss.write_index(index, index_path)
            np.save(ids_path,   np.array(all_ids))
            np.save(texts_path, np.array(all_texts))
            logger.info("Checkpoint: %d / %d docs saved", len(all_ids), total)

    for i, row in enumerate(tqdm(corpus_ds, desc="Encoding corpus", total=total, unit="doc")):
        if i < done_count:
            continue   # bỏ qua các doc đã xử lý
        chunk_ids.append(str(row["id"]))
        chunk_texts.append(_passage_text(row))
        if len(chunk_ids) >= ENCODE_CHUNK:
            flush_chunk()

    flush_chunk()  # phần còn lại

    logger.info("Saving final FAISS index → %s", index_path)
    faiss.write_index(index, index_path)
    np.save(ids_path,   np.array(all_ids))
    np.save(texts_path, np.array(all_texts))

    logger.info("Done. Index: %d vectors", index.ntotal)


def load_corpus_index(index_path: str, ids_path: str, texts_path: str):
    logger.info("Loading FAISS index từ %s ...", index_path)
    index = faiss.read_index(index_path)
    index.nprobe = 64
    corpus_ids   = np.load(ids_path,   allow_pickle=True).tolist()
    corpus_texts = np.load(texts_path, allow_pickle=True).tolist()
    logger.info("Index loaded: %d vectors", index.ntotal)
    return index, corpus_ids, corpus_texts


def mine(args: argparse.Namespace) -> None:
    input_path   = Path(args.input)
    output_path  = Path(args.output)
    index_path   = args.index_path
    ids_path     = args.ids_path
    texts_path   = args.texts_path

    logger.info("Loading clean dataset: %s ...", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Records: %d", len(df))

    logger.info("Loading bi-encoder: %s ...", BI_ENCODER)
    import torch
    model = SentenceTransformer(BI_ENCODER)
    model = model.to(torch.device("cuda"))
    logger.info("Model device: %s", next(model.parameters()).device)

    # Build hoặc load corpus index
    if Path(index_path).exists() and Path(ids_path).exists() and Path(texts_path).exists():
        index, corpus_ids, corpus_texts = load_corpus_index(index_path, ids_path, texts_path)
    else:
        build_corpus_index(
            model, args.corpus_batch, index_path, ids_path, texts_path,
            pool_size=args.pool_size,
            pos_corpus_ids=df["corpus_id"].astype(str).tolist(),
        )
        index, corpus_ids, corpus_texts = load_corpus_index(index_path, ids_path, texts_path)

    index.nprobe = 64
    id2text = {cid: txt for cid, txt in zip(corpus_ids, corpus_texts)}

    # Encode queries
    logger.info("Encoding %d queries ...", len(df))
    q_embs = model.encode(
        ("query: " + df["query"]).tolist(),
        batch_size=args.query_batch,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    pos_ids = df["corpus_id"].astype(str).tolist()

    # Search hard negatives
    logger.info("Searching hard negatives (top_k=%d) ...", args.top_k)
    SEARCH_BATCH = 1024
    hard_negs = []

    for start in tqdm(range(0, len(q_embs), SEARCH_BATCH), desc="Mining", unit="batch"):
        batch_q    = q_embs[start : start + SEARCH_BATCH]
        batch_pids = pos_ids[start : start + SEARCH_BATCH]
        _, indices = index.search(batch_q, args.top_k)

        for nbrs, pid in zip(indices, batch_pids):
            hard_neg = None
            for idx in nbrs:
                if idx < 0:
                    continue
                cid = corpus_ids[idx]
                if cid == pid:
                    continue
                hard_neg = id2text.get(cid)
                break
            hard_negs.append(hard_neg)

    df["negatives"] = [[n] if n else [] for n in hard_negs]
    valid = df["negatives"].apply(len) > 0
    logger.info("Records có hard neg: %d / %d (%.1f%%)",
                valid.sum(), len(df), valid.mean() * 100)

    df_out = df[valid][["query", "positive", "negatives"]].reset_index(drop=True)
    df_out.to_parquet(output_path, index=False)
    logger.info("Saved: %s (%d records)", output_path, len(df_out))


if __name__ == "__main__":
    DRIVE = "/content/drive/MyDrive/Colab Notebooks/vn-embed"

    parser = argparse.ArgumentParser(description="Mine hard negatives từ msmarco-vn corpus")
    parser.add_argument("--input",        default="gen_msmarco_vn_clean.parquet")
    parser.add_argument("--output",       default="train_msmarco_vn.parquet")
    parser.add_argument("--top-k",        type=int, default=30,  dest="top_k")
    parser.add_argument("--corpus-batch", type=int, default=256, dest="corpus_batch")
    parser.add_argument("--query-batch",  type=int, default=512, dest="query_batch")
    parser.add_argument("--index-path",   default=f"{DRIVE}/msmarco_corpus.index", dest="index_path")
    parser.add_argument("--ids-path",     default=f"{DRIVE}/msmarco_corpus_ids.npy", dest="ids_path")
    parser.add_argument("--texts-path",   default=f"{DRIVE}/msmarco_corpus_texts.npy", dest="texts_path")
    parser.add_argument("--pool-size",    type=int, default=None, dest="pool_size",
                        help="Giới hạn corpus pool (None = toàn bộ 8.8M, 1_500_000 = nhanh hơn cho Colab free)")
    args = parser.parse_args()
    mine(args)
