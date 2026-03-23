"""
Patch GTE modeling.py trong HF cache để fix lỗi position_ids OOB
với SentenceTransformerTrainer.
Chạy 1 lần sau khi model được download.
"""
from pathlib import Path
import glob
import os

PATTERN = os.path.join(
    os.path.expanduser("~"),
    ".cache", "huggingface", "modules", "transformers_modules",
    "*NLP*", "*impl*", "*", "modeling.py"
)

OLD = "            rope_cos = rope_cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]\n            rope_sin = rope_sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]"
NEW = "            _max = rope_cos.size(0) - 1\n            rope_cos = rope_cos[position_ids.clamp(0, _max)].unsqueeze(2)  # [bs, seq_len, 1, dim]\n            rope_sin = rope_sin[position_ids.clamp(0, _max)].unsqueeze(2)  # [bs, seq_len, 1, dim]"

OUTPUT_PATTERN = os.path.join("output", "**", "modeling.py")

if __name__ == "__main__":
    files = glob.glob(PATTERN, recursive=True) + glob.glob(OUTPUT_PATTERN, recursive=True)
    if not files:
        print("modeling.py not found in cache. Run training once first to trigger download.")
    for path in files:
        text = Path(path).read_text(encoding="utf-8")
        if "_max = rope_cos.size(0) - 1" in text:
            print(f"Already patched: {path}")
            continue
        if OLD not in text:
            print(f"Pattern not found (GTE version may differ): {path}")
            continue
        Path(path).write_text(text.replace(OLD, NEW), encoding="utf-8")
        print(f"Patched: {path}")
