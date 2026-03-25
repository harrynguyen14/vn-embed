import argparse
import logging
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET = "GreenNode/msmarco-vn"
HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Bạn là trợ lý tạo dữ liệu huấn luyện cho mô hình embedding tiếng Việt. "
    "Nhiệm vụ: đọc đoạn văn và sinh ra MỘT câu hỏi tiếng Việt tự nhiên "
    "mà câu trả lời nằm trong đoạn văn đó. "
    "Quy tắc: (1) Phải là câu hỏi, kết thúc bằng dấu '?'. "
    "(2) Không được copy nguyên văn từ đoạn văn. "
    "(3) Diễn đạt theo cách người dùng thực sự sẽ hỏi. "
    'Chỉ trả lời JSON: {{"query": "<câu hỏi>"}}, không giải thích.'
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Đoạn văn:\n\"\"\"\n{text}\n\"\"\"\n\n"
    'Trả lời JSON: {{"query": "<câu hỏi>"}}'
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    '{"query": "'
)


def _passage_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text  = (row.get("text")  or "").strip()
    return f"{title}. {text}".strip(". ") if title and title != text else text


def load_sample_corpus(sample: int, seed: int, output_path: Path) -> dict[str, str]:
    """Load corpus, loại bỏ các passage đã có qrel, sample ngẫu nhiên."""
    logger.info("Loading corpus từ %s ...", DATASET)
    corpus_ds = load_dataset(DATASET, "corpus", split="dev")
    corpus = {str(row["id"]): _passage_text(row) for row in corpus_ds}
    logger.info("Corpus total: %d", len(corpus))

    # Loại bỏ passage đã có qrel
    qrels_dev  = load_dataset(DATASET, "default", split="dev")
    qrels_test = load_dataset(DATASET, "default", split="test")
    covered_ids = (
        {str(r["corpus-id"]) for r in qrels_dev} |
        {str(r["corpus-id"]) for r in qrels_test}
    )
    uncovered = {cid: text for cid, text in corpus.items() if cid not in covered_ids}
    logger.info("Corpus chưa có qrel: %d / %d", len(uncovered), len(corpus))

    # Loại bỏ những id đã gen (resume)
    done_ids: set[str] = set()
    if output_path.exists():
        done_ids = {str(r["corpus_id"]) for r in pd.read_parquet(output_path).to_dict("records")}
        logger.info("Resume: đã có %d records", len(done_ids))

    remaining = {cid: text for cid, text in uncovered.items() if cid not in done_ids}

    # Sample
    random.seed(seed)
    needed = max(0, sample - len(done_ids))
    if needed == 0:
        logger.info("Đã đủ %d records, không cần gen thêm.", sample)
        return {}

    all_ids = list(remaining.keys())
    sampled_ids = random.sample(all_ids, min(needed, len(all_ids)))
    logger.info("Sẽ gen thêm: %d passages", len(sampled_ids))
    return {cid: remaining[cid] for cid in sampled_ids}


def _parse_query(text: str) -> str | None:
    # Vì prompt đã prefix {"query": " nên output là phần còn lại của câu hỏi
    # Strip trailing quote/brace nếu có
    query = text.strip().rstrip('"}').strip()
    if not query or len(query) < 5:
        return None
    # Phải là câu hỏi
    if "?" not in query:
        return None
    return query


def generate(args: argparse.Namespace) -> None:
    output_path = Path(args.output)

    # Load records cũ nếu resume
    records = []
    if output_path.exists():
        records = pd.read_parquet(output_path).to_dict("records")

    to_process = load_sample_corpus(args.sample, args.seed, output_path)
    if not to_process:
        return

    to_process_ids = list(to_process.keys())
    logger.info("Khởi tạo vLLM (%s)...", HF_MODEL)

    llm = LLM(
        model=HF_MODEL,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        dtype="float16",
        enforce_eager=True,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=80,
        stop=["<|im_end|>", "\"}"],
    )

    CHUNK_SIZE = args.save_every
    total = len(to_process_ids)

    for chunk_start in tqdm(range(0, total, CHUNK_SIZE), desc="Chunks", unit="chunk"):
        chunk_ids = to_process_ids[chunk_start : chunk_start + CHUNK_SIZE]
        prompts = [PROMPT_TEMPLATE.format(text=to_process[cid][:400]) for cid in chunk_ids]

        outputs = llm.generate(prompts, sampling_params)

        for cid, output in zip(chunk_ids, outputs):
            generated_text = output.outputs[0].text.strip()
            query = _parse_query(generated_text)
            if query:
                records.append({
                    "query":     query,
                    "positive":  to_process[cid],
                    "negatives": [],
                    "corpus_id": cid,
                    "source":    "msmarco-vn-synthetic-vllm",
                })

        pd.DataFrame(records).to_parquet(output_path, index=False)
        logger.info(
            "Checkpoint: %d / %d processed, %d records saved",
            chunk_start + len(chunk_ids), total, len(records),
        )

    logger.info("Hoàn thành! Tổng cộng: %d records. Saved: %s", len(records), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gen synthetic queries cho msmarco-vn")
    parser.add_argument("--sample",     type=int,   default=500_000, help="Số passages cần gen")
    parser.add_argument("--output",     default="gen_msmarco_vn.parquet")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save-every", type=int,   default=5000, dest="save_every")
    args = parser.parse_args()
    generate(args)
