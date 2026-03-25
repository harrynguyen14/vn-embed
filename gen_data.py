"""
gen_data.py — Sinh synthetic queries từ corpus GreenNode/quora-vn chưa có qrel.

Pipeline:
  1. Load corpus + qrels từ HuggingFace
  2. Lọc ra các corpus passages chưa có qrel nào ánh xạ vào
  3. Sample N passages ngẫu nhiên
  4. Với mỗi passage, gọi HuggingFace Inference API (text-generation) để sinh query
  5. Lưu ra Parquet format {"query", "positive", "negatives": []}
     (negatives để trống — sẽ được mine bởi beir.py / FAISS sau)

Usage:
  python gen_data.py \
      --hf-token hf_xxx \
      --sample 50000 \
      --output gen_quora_vn.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET   = "GreenNode/quora-vn"
HF_MODEL  = "VTSNLP/Llama3-ViettelSolutions-8B"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Prompt theo Llama-3 instruct format
PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Bạn là trợ lý tạo dữ liệu huấn luyện cho mô hình embedding tiếng Việt. "
    "Nhiệm vụ: đọc đoạn văn bản và sinh ra một câu hỏi tự nhiên bằng tiếng Việt "
    "mà câu trả lời nằm trong đoạn văn đó. "
    "Chỉ trả lời JSON duy nhất theo định dạng: {{\"query\": \"<câu hỏi>\"}}, không giải thích."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Đoạn văn:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Sinh 1 câu hỏi tiếng Việt tự nhiên cho đoạn văn trên. "
    "Trả lời JSON: {{\"query\": \"<câu hỏi>\"}}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def _passage_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text  = (row.get("text")  or "").strip()
    if title and title != text:
        return f"{title}. {text}".strip(". ")
    return text


def load_uncovered_corpus() -> dict[str, str]:
    """Trả về {corpus_id: passage_text} cho các passage chưa có qrel."""
    logger.info("Loading corpus từ %s ...", DATASET)
    corpus_ds = load_dataset(DATASET, "corpus", split="test")
    corpus: dict[str, str] = {str(row["id"]): _passage_text(row) for row in corpus_ds}
    logger.info("Corpus tổng: %d passages", len(corpus))

    logger.info("Loading qrels ...")
    qrels_ds = load_dataset(DATASET, "default", split="test")
    covered_ids: set[str] = {str(row["corpus-id"]) for row in qrels_ds}
    logger.info("Corpus-id đã có qrel: %d", len(covered_ids))

    uncovered = {cid: text for cid, text in corpus.items() if cid not in covered_ids}
    logger.info("Corpus chưa có qrel: %d (%.1f%%)", len(uncovered), len(uncovered) / len(corpus) * 100)
    return uncovered


def call_hf(
    text: str,
    hf_token: str,
    max_retries: int = 3,
    timeout: int = 60,
) -> str | None:
    """Gọi HuggingFace Inference API (text-generation), trả về query string hoặc None."""
    prompt = PROMPT_TEMPLATE.format(text=text[:800])
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 80,
            "temperature": 0.7,
            "return_full_text": False,
            "stop": ["<|eot_id|>", "\n\n"],
        },
    }
    headers = {"Authorization": f"Bearer {hf_token}"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=timeout)

            # Model đang load — chờ theo estimated_time
            if resp.status_code == 503:
                wait = resp.json().get("estimated_time", 20)
                logger.info("Model đang load, chờ %.0fs ...", wait)
                time.sleep(min(wait, 60))
                continue

            resp.raise_for_status()
            generated = resp.json()[0]["generated_text"].strip()

            # Parse JSON từ response
            match = re.search(r'\{.*?"query"\s*:\s*"(.+?)"\s*\}', generated, re.DOTALL)
            if match:
                return match.group(1).strip()

            # Fallback: json.loads
            data = json.loads(generated)
            return data.get("query") or data.get("queries", [None])[0]

        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            logger.debug("Attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


def generate(args: argparse.Namespace) -> None:
    uncovered = load_uncovered_corpus()

    all_ids = list(uncovered.keys())
    if args.sample is not None:
        rng = random.Random(args.seed)
        all_ids = rng.sample(all_ids, min(args.sample, len(all_ids)))
    logger.info("Sẽ xử lý: %d passages ...", len(all_ids))
    sampled_ids = all_ids

    # Resume: bỏ qua những passage đã xử lý
    output_path = Path(args.output)
    done_passages: set[str] = set()
    existing_records: list[dict] = []

    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        existing_records = existing_df.to_dict("records")
        done_passages = {r["positive"] for r in existing_records}
        logger.info("Resume: đã có %d records, bỏ qua ...", len(existing_records))

    records = list(existing_records)
    batch = [(cid, uncovered[cid]) for cid in sampled_ids if uncovered[cid] not in done_passages]
    logger.info("Cần xử lý: %d passages", len(batch))

    save_every = 500
    errors = 0

    for cid, text in tqdm(batch, desc="Generating queries"):
        query = call_hf(text, hf_token=args.hf_token, max_retries=args.max_retries)

        if query is None:
            errors += 1
            logger.debug("Bỏ qua corpus_id=%s (lỗi LLM)", cid)
            continue

        records.append({
            "query":     query,
            "positive":  text,
            "negatives": [],
            "corpus_id": cid,
            "source":    "quora-vn-synthetic",
        })

        if len(records) % save_every == 0:
            _save(records, output_path)

    _save(records, output_path)
    logger.info("Hoàn thành! %d records | %d lỗi | Saved: %s", len(records), errors, output_path)


def _save(records: list[dict], path: Path) -> None:
    pd.DataFrame(records).to_parquet(path, index=False)
    logger.info("Checkpoint: %d records -> %s", len(records), path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gen synthetic queries từ GreenNode/quora-vn corpus")
    p.add_argument("--hf-token",    required=True,  help="HuggingFace token (hf_xxx)")
    p.add_argument("--sample",      type=int, default=None,
                   help="Số passages cần sample (default: toàn bộ corpus chưa có qrel)")
    p.add_argument("--output",      default="gen_quora_vn.parquet",
                   help="File output Parquet (default: gen_quora_vn.parquet)")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
