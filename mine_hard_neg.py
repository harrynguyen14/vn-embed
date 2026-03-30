import argparse
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET    = "GreenNode/msmarco-vn"
BI_ENCODER = "intfloat/multilingual-e5-base"
DIM        = 768
NLIST      = 4096
ENCODE_CHUNK  = 200_000
CHECKPOINT_EVERY = 5


def _passage_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text  = (row.get("text")  or "").strip()
    return f"{title}. {text}".strip(". ") if title and title != text else text


def load_ids(ids_path: str):
    with open(ids_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def build_corpus_index(
    model: SentenceTransformer,
    batch_size: int,
    index_path: str,
    ids_path: str,
    texts_path: str,
    pool_size: int | None = None,
    pos_corpus_ids: list[str] | None = None,
) -> None:

    logger.info("Loading corpus từ %s ...", DATASET)
    corpus_ds = load_dataset(DATASET, "corpus", split="dev", streaming=False)
    total = len(corpus_ds)
    logger.info("Corpus size: %d", total)

    if pool_size and pool_size < total:
        import random as _random
        _random.seed(42)
        indices = _random.sample(range(total), min(pool_size, total))
        corpus_ds = corpus_ds.select(sorted(indices))
        total = len(corpus_ds)
        logger.info("Pool size: %d", total)

    TRAIN_SIZE = min(170_000, total)
    train_texts = [_passage_text(row) for row in corpus_ds.select(range(TRAIN_SIZE))]

    train_embs = model.encode(
        ["passage: " + t[:512] for t in train_texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    if Path(index_path).exists():
        index = faiss.read_index(index_path)
    else:
        quantizer = faiss.IndexFlatIP(DIM)
        index = faiss.IndexIVFPQ(quantizer, DIM, NLIST, 64, 8)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.train(train_embs)

    del train_embs, train_texts

    Path(ids_path).parent.mkdir(parents=True, exist_ok=True)
    Path(texts_path).parent.mkdir(parents=True, exist_ok=True)

    if not Path(ids_path).exists():
        open(ids_path, "w").close()
    if not Path(texts_path).exists():
        open(texts_path, "w").close()

    chunk_ids = []
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
            convert_to_numpy=True,
        ).astype("float32")

        index.add(embs)

        with open(ids_path, "a", encoding="utf-8") as f_id:
            for i in chunk_ids:
                f_id.write(i + "\n")

        with open(texts_path, "a", encoding="utf-8") as f_txt:
            for t in chunk_texts:
                f_txt.write(t.replace("\n", " ") + "\n")

        chunk_ids.clear()
        chunk_texts.clear()
        chunk_count += 1

        if chunk_count % CHECKPOINT_EVERY == 0:
            faiss.write_index(index, index_path)
            logger.info("Checkpoint: %d chunks", chunk_count)

    for row in tqdm(corpus_ds, total=total):
        chunk_ids.append(str(row["id"]))
        chunk_texts.append(_passage_text(row))

        if len(chunk_ids) >= ENCODE_CHUNK:
            flush_chunk()

    flush_chunk()

    faiss.write_index(index, index_path)
    logger.info("Done. Index size: %d", index.ntotal)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device.type == "cuda":
        model = model.half()

    if not (Path(index_path).exists() and Path(ids_path).exists() and Path(texts_path).exists()):
        build_corpus_index(
            model,
            args.corpus_batch,
            index_path,
            ids_path,
            texts_path,
            pool_size=args.pool_size,
            pos_corpus_ids=df["corpus_id"].astype(str).tolist(),
        )

    index = faiss.read_index(index_path)
    index.nprobe = 32

    corpus_ids = load_ids(ids_path)

    corpus_texts = []
    with open(texts_path, "r", encoding="utf-8") as f:
        for line in f:
            corpus_texts.append(line.strip())

    logger.info("Encoding %d queries ...", len(df))
    q_embs = model.encode(
        ("query: " + df["query"]).tolist(),
        batch_size=args.query_batch,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    pos_ids = df["corpus_id"].astype(str).tolist()

    logger.info("Searching hard negatives (top_k=%d) ...", args.top_k)
    SEARCH_BATCH = 1024
    hard_negs = []

    for start in tqdm(range(0, len(q_embs), SEARCH_BATCH)):
        batch_q    = q_embs[start : start + SEARCH_BATCH]
        batch_pids = pos_ids[start : start + SEARCH_BATCH]

        _, indices = index.search(batch_q, args.top_k)

        for nbrs, pid in zip(indices, batch_pids):
            hard_neg = None
            for idx in nbrs:
                if idx < 0:
                    continue
                if corpus_ids[idx] == pid:
                    continue
                hard_neg = corpus_texts[idx]
                break
            hard_negs.append(hard_neg)

    df["negatives"] = [[n] if n else [] for n in hard_negs]
    valid = df["negatives"].apply(len) > 0

    df_out = df[valid][["query", "positive", "negatives"]].reset_index(drop=True)
    df_out.to_parquet(output_path, index=False)

    logger.info("Saved: %s (%d records)", output_path, len(df_out))


if __name__ == "__main__":
    DRIVE = "/content/drive/MyDrive/Colab Notebooks/vn-embed"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default="gen_msmarco_vn_clean.parquet")
    parser.add_argument("--output",       default="train_msmarco_vn.parquet")
    parser.add_argument("--top-k",        type=int, default=30,  dest="top_k")
    parser.add_argument("--corpus-batch", type=int, default=512, dest="corpus_batch")
    parser.add_argument("--query-batch",  type=int, default=1024, dest="query_batch")
    parser.add_argument("--index-path",   default=f"{DRIVE}/msmarco_corpus.index", dest="index_path")
    parser.add_argument("--ids-path",     default=f"{DRIVE}/msmarco_corpus_ids.txt", dest="ids_path")
    parser.add_argument("--texts-path",   default=f"{DRIVE}/msmarco_corpus_texts.txt", dest="texts_path")
    parser.add_argument("--pool-size",    type=int, default=None, dest="pool_size")

    args = parser.parse_args()
    mine(args)