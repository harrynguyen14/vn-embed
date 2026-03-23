from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer, losses

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"


def load_model(model_name: str = MODEL_NAME, max_seq_length: int = 256) -> SentenceTransformer:
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    return model


def get_loss(model: SentenceTransformer) -> losses.MultipleNegativesRankingLoss:
    return losses.MultipleNegativesRankingLoss(model)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
