import argparse
import json
import logging
import random
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET = "GreenNode/quora-vn"
HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Bạn là trợ lý tạo dữ liệu huấn luyện cho mô hình embedding tiếng Việt. "
    "Nhiệm vụ: đọc đoạn văn bản và sinh ra một câu hỏi tự nhiên bằng tiếng Việt "
    "mà câu trả lời nằm trong đoạn văn đó. "
    "Chỉ trả lời JSON duy nhất theo định dạng: {{\"query\": \"<câu hỏi>\"}}, không giải thích."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Đoạn văn:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Sinh 1 câu hỏi tiếng Việt tự nhiên cho đoạn văn trên. "
    "Trả lời JSON: {{\"query\": \"<câu hỏi>\"}}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

def _passage_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text  = (row.get("text")  or "").strip()
    return f"{title}. {text}".strip(". ") if title and title != text else text

def load_uncovered_corpus() -> dict[str, str]:
    logger.info("Loading corpus từ %s ...", DATASET)
    corpus_ds = load_dataset(DATASET, "corpus", split="test")
    corpus = {str(row["id"]): _passage_text(row) for row in corpus_ds}
    
    qrels_ds = load_dataset(DATASET, "default", split="test")
    covered_ids = {str(row["corpus-id"]) for row in qrels_ds}

    uncovered = {cid: text for cid, text in corpus.items() if cid not in covered_ids}
    logger.info("Corpus chưa có qrel: %d / %d", len(uncovered), len(corpus))
    return uncovered

def _parse_query(text: str) -> str | None:
    # Ưu tiên regex để lấy nội dung trong JSON query
    match = re.search(r'"query"\s*:\s*"(.+?)"', text, re.DOTALL)
    if match: return match.group(1).strip()
    try:
        data = json.loads(text)
        return data.get("query")
    except: return None

def generate(args: argparse.Namespace) -> None:
    uncovered = load_uncovered_corpus()
    all_ids = list(uncovered.keys())
    
    if args.sample:
        random.seed(args.seed)
        all_ids = random.sample(all_ids, min(args.sample, len(all_ids)))

    output_path = Path(args.output)
    done_ids = set()
    records = []

    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        records = existing_df.to_dict("records")
        done_ids = {str(r["corpus_id"]) for r in records}
        logger.info("Resume: đã có %d records", len(records))

    # Lọc những câu chưa làm
    to_process_ids = [cid for cid in all_ids if cid not in done_ids]
    if not to_process_ids:
        logger.info("Không còn dữ liệu mới để xử lý.")
        return

    logger.info("Sẽ xử lý: %d passages với vLLM", len(to_process_ids))

    # Khởi tạo vLLM - Tối ưu cho T4 16GB
    llm = LLM(
        model=HF_MODEL,
        trust_remote_code=True,
        gpu_memory_utilization=0.90, # Dùng 90% VRAM T4
        max_model_len=1024,          # Giới hạn context để tiết kiệm RAM
        dtype="float16"              # T4 chạy float16/half là tốt nhất
    )

    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=64, 
        stop=["<|im_end|>", "\n"]
    )

    # Chuẩn bị prompts
    prompts = [PROMPT_TEMPLATE.format(text=uncovered[cid][:400]) for cid in to_process_ids]

    # Chạy Batch Inference (vLLM tự quản lý batch cực nhanh)
    outputs = llm.generate(prompts, sampling_params)

    # Thu thập kết quả
    for cid, output in zip(to_process_ids, outputs):
        generated_text = output.outputs[0].text.strip()
        query = _parse_query(generated_text)
        
        if query:
            records.append({
                "query": query,
                "positive": uncovered[cid],
                "negatives": [],
                "corpus_id": cid,
                "source": "quora-vn-synthetic-vllm",
            })

    # Lưu file cuối cùng
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    logger.info("Hoàn thành! Tổng cộng: %d records. Saved: %s", len(records), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", help="Tùy chọn nếu model public")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output", default="gen_quora_vn.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args)
