from __future__ import annotations

import torch
import logging
from sentence_transformers import SentenceTransformer, losses

MODEL_NAME = "intfloat/multilingual-e5-base"

logger = logging.getLogger(__name__)


def load_model(
    model_name: str = MODEL_NAME,
    max_seq_length: int = 256,
    freeze_layers: bool = True,
) -> SentenceTransformer:
    model = SentenceTransformer(
        model_name,
        model_kwargs={"torch_dtype": torch.float32},
    )
    model.max_seq_length = max_seq_length

    if freeze_layers:
        for name, param in model.named_parameters():
            if "encoder.layer.10" not in name and "encoder.layer.11" not in name:
                param.requires_grad = False

    logger.info(f"Loaded model {model_name} (freeze_layers={freeze_layers})")
    return model


class DistillLoss(torch.nn.Module):
    def __init__(
        self,
        student: SentenceTransformer,
        teacher_name: str = MODEL_NAME,
        alpha: float = 0.1,
        scale: float = 10.0,
    ):
        super().__init__()
        self.student = student
        self.teacher = SentenceTransformer(teacher_name)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.rank_loss = losses.MultipleNegativesRankingLoss(student, scale=scale)
        self.alpha = alpha

    def forward(self, sentence_features, labels=None):
        loss_rank = self.rank_loss(sentence_features, labels)

        student_emb = self.student(sentence_features[0])["sentence_embedding"]

        with torch.no_grad():
            teacher_emb = self.teacher(sentence_features[0])["sentence_embedding"]

        loss_distill = torch.nn.functional.mse_loss(student_emb, teacher_emb)

        return loss_rank + self.alpha * loss_distill


def get_loss(
    model: SentenceTransformer,
    use_distillation: bool = True,
    alpha: float = 0.1,
    scale: float = 10.0,
):
    if use_distillation:
        return DistillLoss(
            student=model,
            teacher_name=MODEL_NAME,
            alpha=alpha,
            scale=scale,
        )
    else:
        return losses.MultipleNegativesRankingLoss(model, scale=scale)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"