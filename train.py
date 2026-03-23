from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import SentenceEvaluator
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState
from datasets import Dataset

from dataset_processor import build_dataloaders
from evaluate import evaluate_retrieval
from model import get_loss, get_device, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def triplets_to_dataset(triplets: list[dict]) -> Dataset:
    data = {
        "anchor":    [t["query"]    for t in triplets],
        "positive":  [t["positive"] for t in triplets],
        "negatives": [t["negatives"] for t in triplets],  # list of hard negs
    }
    return Dataset.from_dict(data)


class NegativeSamplingDataset(torch.utils.data.Dataset):
    """Wraps a HF Dataset, randomly picking 1 negative per step."""
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        neg = row["negatives"][torch.randint(len(row["negatives"]), (1,)).item()]
        return {"anchor": row["anchor"], "positive": row["positive"], "negative": neg}


class Evaluator(SentenceEvaluator):
    def __init__(self, triplets: list[dict], batch_size: int = 256, max_samples: int = 2000):
        import random
        eval_data = triplets.copy()
        random.shuffle(eval_data)
        self.triplets = eval_data[:max_samples]
        self.batch_size = batch_size

    def __call__(self, model, output_path=None, epoch=-1, steps=-1) -> dict[str, float]:
        results = evaluate_retrieval(model, self.triplets, batch_size=self.batch_size)
        clean_results = {k.replace("@", "_at_"): v for k, v in results.items()}

        logger.info(f"Epoch {epoch} Step {steps} Evaluation: {clean_results}")
        return clean_results


def train(args: argparse.Namespace) -> None:
    device = get_device()
    logger.info("Device: %s", device)

    train_triplets, dev_triplets, test_triplets = build_dataloaders(
        input_path   = Path(args.data),
        split_dir    = Path(args.splits_dir),
        max_hard_neg = args.max_hard_neg,
        seed         = args.seed,
    )

    train_dataset = NegativeSamplingDataset(triplets_to_dataset(train_triplets))
    evaluator = Evaluator(dev_triplets, batch_size=args.eval_batch_size)

    model      = load_model(args.model_name)
    train_loss = get_loss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir          = args.output_dir,
        num_train_epochs    = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate       = args.lr,
        warmup_ratio        = args.warmup_ratio,
        weight_decay                = args.weight_decay,
        max_grad_norm               = args.max_grad_norm,
        gradient_accumulation_steps = 1,
        logging_steps       = args.log_every,
        eval_strategy       = "epoch",
        save_strategy       = "epoch",
        save_total_limit    = 2,
        load_best_model_at_end = True,
        metric_for_best_model  = "MRR_at_10",
        greater_is_better      = True,
        dataloader_num_workers = 2,
        report_to           = "none",
        max_steps           = args.max_steps,
    )

    evaluator = Evaluator(dev_triplets, batch_size=args.eval_batch_size)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience),
        ],
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    logger.info("Training done. Model saved to %s", args.output_dir)

    logger.info("Running final test evaluation...")
    test_metrics = evaluate_retrieval(model, test_triplets, ks=[1, 5, 10])
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
    p.add_argument("--patience",        type=int,   default=3)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--max-grad-norm",   type=float, default=1.0)
    p.add_argument("--grad-accum",      type=int,   default=4)
    p.add_argument("--resume-from",     default=None)
    p.add_argument("--max-steps",       type=int,   default=-1)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
