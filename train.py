from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import SentenceEvaluator
from transformers import EarlyStoppingCallback
from datasets import Dataset

from dataset_processor import build_dataloaders
from evaluate import evaluate_retrieval
from model import get_loss, get_device, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def triplets_to_dataset(triplets: list[dict]) -> Dataset:
    rows = [
        {"query": t["query"], "positive": t["positive"],
         **{f"negative_{i+1}": neg for i, neg in enumerate(t["negatives"])}}
        for t in triplets
    ]
    return Dataset.from_list(rows)


class Evaluator(SentenceEvaluator):
    def __init__(self, triplets: list[dict], batch_size: int = 256, max_samples: int = 2000):
        self.triplets   = triplets[:max_samples]
        self.batch_size = batch_size
        self.primary_metric = "MRR_at_10"

    def __call__(self, model, output_path=None, epoch=-1, steps=-1) -> dict[str, float]:
        raw = evaluate_retrieval(model, self.triplets, batch_size=self.batch_size, ks=[1, 5, 10])
        metrics = {k.replace("@", "_at_"): v for k, v in raw.items()}
        logger.info("=== Epoch %d — Evaluation ===", epoch)
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)
        return metrics


def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def train(args: argparse.Namespace) -> None:
    free_vram()
    device = get_device()
    logger.info("Device: %s", device)

    train_triplets, dev_triplets, test_triplets = build_dataloaders(
        input_path   = Path(args.data),
        split_dir    = Path(args.splits_dir),
        max_hard_neg = args.max_hard_neg,
        seed         = args.seed,
    )

    train_dataset = triplets_to_dataset(train_triplets)

    model      = load_model(args.model_name)
    train_loss = get_loss(model)

    total_steps  = (len(train_dataset) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    training_args = SentenceTransformerTrainingArguments(
        output_dir          = args.output_dir,
        num_train_epochs    = args.epochs,
        per_device_train_batch_size = args.batch_size,
        learning_rate       = args.lr,
        warmup_steps        = warmup_steps,
        weight_decay                = args.weight_decay,
        max_grad_norm               = args.max_grad_norm,
        gradient_accumulation_steps = args.grad_accum,
        fp16                = args.fp16 and device == "cuda",
        logging_steps       = args.log_every,
        eval_strategy       = "epoch",
        save_strategy       = "epoch",
        save_total_limit    = 2,
        load_best_model_at_end = True,
        metric_for_best_model  = "MRR_at_10",
        greater_is_better      = True,
        dataloader_num_workers = 2,
        report_to           = "none",
    )

    evaluator = Evaluator(dev_triplets, batch_size=args.eval_batch_size)

    trainer = SentenceTransformerTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        loss          = train_loss,
        evaluator     = evaluator,
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    logger.info("Training done. Model saved to %s", args.output_dir)

    logger.info("Running final test evaluation...")
    best_model   = load_model(args.output_dir)
    test_metrics = evaluate_retrieval(best_model, test_triplets, ks=[1, 5, 10])
    logger.info("=== TEST RESULTS ===")
    for k, v in test_metrics.items():
        logger.info("  %s: %.4f", k, v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GTE multilingual embedding model")

    p.add_argument("--model-name",      default="Alibaba-NLP/gte-multilingual-base")
    p.add_argument("--data",            default="filtered_bkai.parquet")
    p.add_argument("--splits-dir",      default="splits")
    p.add_argument("--output-dir",      default="output/gte-vn")
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--max-hard-neg",    type=int,   default=3)
    p.add_argument("--lr",              type=float, default=2e-5)
    p.add_argument("--warmup-ratio",    type=float, default=0.1)
    p.add_argument("--eval-batch-size", type=int,   default=256)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--log-every",       type=int,   default=10)
    p.add_argument("--fp16",            action="store_true", default=True)
    p.add_argument("--no-fp16",         dest="fp16", action="store_false")
    p.add_argument("--patience",        type=int,   default=3)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--max-grad-norm",   type=float, default=1.0)
    p.add_argument("--grad-accum",      type=int,   default=4)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
