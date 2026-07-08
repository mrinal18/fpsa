"""Loss head for deep-supervision + ACT training (TRM reference port).

lm loss: stablemax cross-entropy (Prieto et al., as used by TRM) normalized
per sequence; halting loss: binary CE of the Q-halt logit against sequence
correctness. Metrics are counted only on halted sequences, matching the
reference protocol (a sequence's prediction is scored when it halts).
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = IGNORE_LABEL_ID):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = IGNORE_LABEL_ID):
    return F.cross_entropy(
        logits.to(torch.float32).flatten(0, -2), labels.to(torch.long).flatten(),
        ignore_index=ignore_index, reduction="none",
    ).view(labels.shape)


LOSS_FNS = {
    "stablemax_cross_entropy": stablemax_cross_entropy,
    "softmax_cross_entropy": softmax_cross_entropy,
}


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str = "stablemax_cross_entropy",
                 q_loss_coeff: float = 0.5):
        super().__init__()
        self.model = model
        self.loss_fn = LOSS_FNS[loss_type]
        self.q_loss_coeff = q_loss_coeff

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(self, carry, batch) -> Tuple:
        new_carry, outputs = self.model(carry=carry, batch=batch)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
            # Batch-level scalars (not count-normalized): averaged per
            # optimizer step by the training loop. Models emit only the stats
            # they actually have, so no sentinel filtering is needed.
            for k, v in outputs.items():
                if k.startswith("stat_") and k != "stat_jacobian_loss":
                    metrics[k] = v.detach()

        lm_loss = (self.loss_fn(outputs["logits"], labels) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        loss = lm_loss + self.q_loss_coeff * q_halt_loss

        # Pre-weighted by the model (jacobian_reg_lambda lives in the model
        # config only); scaled to the per-batch-sum convention of the loss.
        if "stat_jacobian_loss" in outputs:
            loss = loss + outputs["stat_jacobian_loss"] * labels.shape[0]

        metrics["lm_loss"] = lm_loss.detach()
        metrics["q_halt_loss"] = q_halt_loss.detach()

        return new_carry, loss, metrics, outputs["logits"].detach(), new_carry.halted.all()
