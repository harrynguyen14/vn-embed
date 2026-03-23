from __future__ import annotations

import torch
import logging
from sentence_transformers import SentenceTransformer, losses

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
logger = logging.getLogger(__name__)

def load_model(model_name: str = MODEL_NAME, max_seq_length: int = 256) -> SentenceTransformer:
    model = SentenceTransformer(model_name, trust_remote_code=True)

    model.max_seq_length = max_seq_length

    model = model.to(torch.float32) 
    
    logger.info(f"Loaded model {model_name} with max_seq_length={max_seq_length}")
    return model

def get_loss(model: SentenceTransformer) -> losses.MultipleNegativesRankingLoss:
    return losses.MultipleNegativesRankingLoss(model, scale=20.0)

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available(): 
        return "mps"
    return "cpu"