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

DATASET        = "GreenNode/msmarco-vn"
BI_ENCODER     = "intfloat/multilingual-e5-base"
INDEX_PATH     = "msmarco_corpus.index"
CORPUS_NPZ     = "msmarco_corpus_meta.npz"   # lưu ids + texts


def _passage_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text  = (row.get("text")  or "").strip()
    return f"{title}. {text}".strip(". ") if title and title != text else text


def build_corpus_index(model: SentenceTransformer, batch_size: int):
    """Encode 8.8M corpus, build FAISS IVFPQ index, cache lên disk."""
    logger.info("Loading corpus từ %s ...", DATASET)
    corpus_ds = load_dataset(DATASET, "corpus", split="dev")

    corpus_ids   = []
    corpus_texts = []
    for row in tqdm(corpus_ds, desc="Loading corpus", unit="doc"):
        corpus_ids.append(str(row["id"]))
        corpus_texts.append(_passage_text(row))

    logger.info("Encoding %d passages ...", len(corpus_texts))
    prefixed = ["passage: " + t[:512] for t in corpus_texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    dim   = embeddings.shape[1]
    nlist = 4096
    logger.info("Building FAISS IVFPQ index (dim=%d, nlist=%d) ...", dim, nlist)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, 64, 8)
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    logger.info("Training index on %d samples ...", min(500_000, len(embeddings)))
    index.train(embeddings[: min(500_000, len(embeddings))])
    index.add(embeddings)
    index.nprobe = 64

    logger.info("Saving index + metadata ...")
    faiss.write_index(index, INDEX_PATH)
    np.savez_compressed(
        CORPUS_NPZ,
        ids=np.array(corpus_ids),
        texts=np.array(corpus_texts),
    )
    logger.info("Done. Index: %d vectors", index.ntotal)
    return index, corpus_ids, corpus_texts


def load_corpus_index():
    """Load cached FAISS index và corpus metadata."""
    logger.info("Loading cached index từ %s ...", INDEX_PATH)
    index = faiss.read_index(INDEX_PATH)
    index.nprobe = 64
    meta = np.load(CORPUS_NPZ, allow_pickle=True)
    corpus_ids   = meta["ids"].tolist()
    corpus_texts = meta["texts"].tolist()
    logger.info("Index loaded: %d vectors", index.ntotal)
    return index, corpus_ids, corpus_texts


def mine(args: argparse.Namespace) -> None:
    input_path  = Path(args.input)
    output_path = Path(args.output)

    logger.info("Loading clean dataset: %s ...", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Records: %d", len(df))

    logger.info("Loading bi-encoder: %s ...", BI_ENCODER)
    model = SentenceTransformer(BI_ENCODER, device="cuda")

    # Build hoặc load corpus index
    if Path(INDEX_PATH).exists() and Path(CORPUS_NPZ).exists():
        index, corpus_ids, corpus_texts = load_corpus_index()
    else:
        index, corpus_ids, corpus_texts = build_corpus_index(model, args.corpus_batch)

    # Map corpus_id → text
    id2text = {cid: txt for cid, txt in zip(corpus_ids, corpus_texts)}

    # Encode queries
    logger.info("Encoding %d queries ...", len(df))
    queries = ("query: " + df["query"]).tolist()
    q_embs  = model.encode(
        queries,
        batch_size=args.query_batch,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    pos_ids = df["corpus_id"].astype(str).tolist()

    # Search hard negatives theo batch
    logger.info("Searching hard negatives (top_k=%d) ...", args.top_k)
    SEARCH_BATCH = 1024
    hard_negs    = []

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
                    continue   # bỏ qua chính positive
                hard_neg = id2text.get(cid)
                break
            hard_negs.append(hard_neg)

    # Gắn hard neg vào dataframe
    df["negatives"] = [[n] if n else [] for n in hard_negs]

    valid = df["negatives"].apply(len) > 0
    logger.info("Records có hard neg: %d / %d (%.1f%%)",
                valid.sum(), len(df), valid.mean() * 100)

    df_out = df[valid][["query", "positive", "negatives"]].reset_index(drop=True)
    df_out.to_parquet(output_path, index=False)
    logger.info("Saved: %s (%d records)", output_path, len(df_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine hard negatives từ msmarco-vn corpus")
    parser.add_argument("--input",        default="gen_msmarco_vn_clean.parquet")
    parser.add_argument("--output",       default="train_msmarco_vn.parquet")
    parser.add_argument("--top-k",        type=int, default=30,  dest="top_k",
                        help="Số candidates FAISS search per query (default: 30)")
    parser.add_argument("--corpus-batch", type=int, default=512, dest="corpus_batch",
                        help="Batch size khi encode corpus (default: 512)")
    parser.add_argument("--query-batch",  type=int, default=512, dest="query_batch",
                        help="Batch size khi encode queries (default: 512)")
    args = parser.parse_args()
    mine(args)
