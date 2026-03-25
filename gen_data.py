"""
gen_data.py — Sinh synthetic queries từ corpus GreenNode/quora-vn chưa có qrel.
Chạy trên Kaggle T4 15GB với 4-bit quantization + batch inference.

Pipeline:
  1. Load corpus + qrels từ HuggingFace
  2. Lọc corpus chưa có qrel
  3. Batch inference local (transformers + bitsandbytes 4-bit)
  4. Lưu Parquet {"query", "positive", "negatives": []}

Ước tính: ~3-5h cho 511k passages trên T4 (batch_size=16)

Cài đặt trên Kaggle:
  !pip install -q bitsandbytes accelerate

Usage:
  python gen_data.py --hf-token hf_xxx
  python gen_data.py --hf-token hf_xxx --batch-size 16 --sample 100000
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET   = "GreenNode/quora-vn"
HF_MODEL  = "VTSNLP/Llama3-ViettelSolutions-8B"

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
    logger.info("Loading corpus từ %s ...", DATASET)
    corpus_ds = load_dataset(DATASET, "corpus", split="test")
    corpus: dict[str, str] = {str(row["id"]): _passage_text(row) for row in corpus_ds}
    logger.info("Corpus tổng: %d passages", len(corpus))

    qrels_ds = load_dataset(DATASET, "default", split="test")
    covered_ids: set[str] = {str(row["corpus-id"]) for row in qrels_ds}

    uncovered = {cid: text for cid, text in corpus.items() if cid not in covered_ids}
    logger.info("Corpus chưa có qrel: %d / %d (%.1f%%)",
                len(uncovered), len(corpus), len(uncovered) / len(corpus) * 100)
    return uncovered


def load_model(hf_token: str):
    logger.info("Loading model %s (4-bit) ...", HF_MODEL)

    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "12GiB", "cpu": "30GiB"},
        token=hf_token,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    logger.info("Model loaded. VRAM: %.1f GB", torch.cuda.memory_allocated() / 1e9)
    return tokenizer, model


def _parse_query(text: str) -> str | None:
    match = re.search(r'\{.*?"query"\s*:\s*"(.+?)"\s*\}', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    try:
        data = json.loads(text)
        return data.get("query") or data.get("queries", [None])[0]
    except (json.JSONDecodeError, IndexError):
        return None


def run_batch(
    tokenizer,
    model,
    texts: list[str],
    max_new_tokens: int = 80,
) -> list[str | None]:
    prompts = [PROMPT_TEMPLATE.format(text=t[:800]) for t in texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Chỉ lấy phần generated (bỏ phần prompt)
    input_len = inputs["input_ids"].shape[1]
    results = []
    for out in outputs:
        generated = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        results.append(_parse_query(generated))
    return results


def generate(args: argparse.Namespace) -> None:
    uncovered = load_uncovered_corpus()

    all_ids = list(uncovered.keys())
    if args.sample is not None:
        rng = random.Random(args.seed)
        all_ids = rng.sample(all_ids, min(args.sample, len(all_ids)))

    output_path = Path(args.output)
    done_passages: set[str] = set()
    records: list[dict] = []

    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        records = existing_df.to_dict("records")
        done_passages = {r["positive"] for r in records}
        logger.info("Resume: đã có %d records", len(records))

    batch_items = [(cid, uncovered[cid]) for cid in all_ids if uncovered[cid] not in done_passages]
    logger.info("Sẽ xử lý: %d passages | batch_size=%d", len(batch_items), args.batch_size)
    logger.info("Ước tính: ~%.1f giờ", len(batch_items) / args.batch_size * 1.5 / 3600)

    tokenizer, model = load_model(args.hf_token)

    save_every = 2000
    errors = 0
    n_batches = (len(batch_items) + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(n_batches), desc="Generating queries"):
        chunk = batch_items[i * args.batch_size : (i + 1) * args.batch_size]
        cids  = [c for c, _ in chunk]
        texts = [t for _, t in chunk]

        queries = run_batch(tokenizer, model, texts)

        for cid, text, query in zip(cids, texts, queries):
            if query is None:
                errors += 1
                continue
            records.append({
                "query":     query,
                "positive":  text,
                "negatives": [],
                "corpus_id": cid,
                "source":    "quora-vn-synthetic",
            })

        if len(records) % save_every < args.batch_size:
            _save(records, output_path)

    _save(records, output_path)
    logger.info("Hoàn thành! %d records | %d lỗi | Saved: %s", len(records), errors, output_path)


def _save(records: list[dict], path: Path) -> None:
    pd.DataFrame(records).to_parquet(path, index=False)
    logger.info("Checkpoint: %d records -> %s", len(records), path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gen synthetic queries - local inference T4")
    p.add_argument("--hf-token",   required=True, help="HuggingFace token để download model")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size inference (default: 16, giảm nếu OOM)")
    p.add_argument("--sample",     type=int, default=None,
                   help="Giới hạn số passages (default: toàn bộ)")
    p.add_argument("--output",     default="gen_quora_vn.parquet")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
