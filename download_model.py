from huggingface_hub import snapshot_download, constants
from pathlib import Path
import shutil
import json

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
NEW_IMPL   = "Alibaba-NLP/new-impl"
SAVE_DIR   = Path("models/gte-multilingual-base")

if __name__ == "__main__":
    print(f"Downloading {MODEL_NAME} ...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_NAME, local_dir=str(SAVE_DIR))

    print(f"Downloading {NEW_IMPL} ...")
    from transformers.utils import TRANSFORMERS_CACHE
    new_impl_cache = Path(TRANSFORMERS_CACHE) / "modules" / "transformers_modules" / "Alibaba-NLP" / "new-impl"
    new_impl_cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=NEW_IMPL, local_dir=str(new_impl_cache))

    print(f"Done. Model saved to {SAVE_DIR}")
    print(f"Custom code saved to {new_impl_cache}")
