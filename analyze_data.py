import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse


def encode(model, texts, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_tensor=True
            )
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def main(args):
    print("=== LOAD DATA ===")
    df = pd.read_parquet(args.data)
    print(f"Total samples: {len(df):,}")

    # ===== SAMPLE =====
    sample_size = min(args.sample_size, len(df))
    sample = df.sample(sample_size, random_state=42).reset_index(drop=True)
    print(f"Using sample size: {sample_size:,}\n")

    # ===== PREPARE TEXT =====
    queries = ["query: " + q for q in sample["query"]]
    positives = ["passage: " + p for p in sample["positive"]]
    negatives = [
        "passage: " + n[0] if len(n) > 0 else ""
        for n in sample["negatives"]
    ]

    # ===== LOAD MODEL =====
    print("=== LOAD MODEL ===")
    model = SentenceTransformer(args.model)
    model.eval()

    # ===== ENCODE =====
    print("\n=== ENCODING ===")
    q_emb = encode(model, queries, args.batch_size)
    p_emb = encode(model, positives, args.batch_size)
    n_emb = encode(model, negatives, args.batch_size)

    # ===== SIMILARITY =====
    print("\n=== COMPUTE SIMILARITY ===")
    sim_qp = (q_emb * p_emb).sum(dim=1).cpu().numpy()
    sim_qn = (q_emb * n_emb).sum(dim=1).cpu().numpy()
    sim_pn = (p_emb * n_emb).sum(dim=1).cpu().numpy()

    gap = sim_qp - sim_qn

    # ===== BASIC STATS =====
    print("\n=== EMBEDDING SIMILARITY ===")
    print(f"sim(q,p): mean={sim_qp.mean():.3f}, median={np.median(sim_qp):.3f}, std={sim_qp.std():.3f}")
    print(f"sim(q,n): mean={sim_qn.mean():.3f}, median={np.median(sim_qn):.3f}, std={sim_qn.std():.3f}")
    print(f"sim(p,n): mean={sim_pn.mean():.3f}, median={np.median(sim_pn):.3f}, std={sim_pn.std():.3f}")

    # ===== GAP =====
    print("\n=== GAP Δ = sim(q,p) - sim(q,n) ===")
    print(f"mean={gap.mean():.3f}, median={np.median(gap):.3f}, std={gap.std():.3f}")
    print(f"min={gap.min():.3f}, max={gap.max():.3f}")

    # ===== FALSE NEGATIVE =====
    false_neg = (sim_qn > sim_qp).mean()
    print("\n=== FALSE NEGATIVE RATE ===")
    print(f"{false_neg * 100:.2f}% samples where sim(q,n) > sim(q,p)")

    # ===== HARDNESS DISTRIBUTION =====
    easy = (gap > 0.5).mean()
    medium = ((gap <= 0.5) & (gap > 0.2)).mean()
    hard = ((gap <= 0.2) & (gap > 0)).mean()
    very_hard = (gap <= 0).mean()

    print("\n=== HARD NEGATIVE DISTRIBUTION ===")
    print(f"Easy (Δ > 0.5):        {easy * 100:.1f}%")
    print(f"Medium (0.2–0.5):      {medium * 100:.1f}%")
    print(f"Hard (0–0.2):          {hard * 100:.1f}%")
    print(f"Very hard (Δ <= 0):    {very_hard * 100:.1f}%")

    # ===== LENGTH =====
    print("\n=== LENGTH STATS ===")
    q_len = sample["query"].str.len()
    p_len = sample["positive"].str.len()
    n_len = sample["negatives"].apply(lambda x: len(x[0]) if len(x) > 0 else 0)

    print(f"Query avg len:    {q_len.mean():.0f}")
    print(f"Positive avg len: {p_len.mean():.0f}")
    print(f"Negative avg len: {n_len.mean():.0f}")

    # ===== DUPLICATE =====
    print("\n=== DUPLICATION ===")
    dup_q = sample["query"].duplicated().mean()
    dup_p = sample["positive"].duplicated().mean()

    print(f"Duplicate query rate:    {dup_q * 100:.2f}%")
    print(f"Duplicate positive rate: {dup_p * 100:.2f}%")

    print("\n=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True,
                        help="Path to parquet dataset")
    parser.add_argument("--model", type=str,
                        default="intfloat/multilingual-e5-base")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    main(args)