"""Per-example StableMax supervision with a correctness-trained halt head."""
import torch
import torch.nn.functional as F


def log_stablemax(logits):
    # Clamp the unselected reciprocal branch too. torch.where evaluates both
    # branches; a pole at a positive logit can otherwise poison its gradient.
    x = logits.double()
    s = torch.where(x < 0, (1 - x.clamp_max(0)).reciprocal(), 1 + x.clamp_min(0))
    return s.log() - s.sum(-1, keepdim=True).log()


def per_example_loss(logits, labels, *, ignore_index=-100):
    mask = labels != ignore_index
    safe = torch.where(mask, labels, 0).long()
    nll = -log_stablemax(logits).gather(-1, safe[..., None]).squeeze(-1)
    count = mask.sum(-1)
    valid = count > 0
    losses = (nll * mask).sum(-1) / count.clamp_min(1)
    return losses, valid, mask


def supervised_loss(output, labels, *, halt_weight=.5, stability_weight=0.):
    losses, valid, mask = per_example_loss(output['logits'], labels)
    denom = valid.sum().clamp_min(1)
    with torch.no_grad():
        correct = ((output['logits'].argmax(-1) == labels) | ~mask).all(-1) & valid
    halt = F.binary_cross_entropy_with_logits(output['q_halt_logits'],
                                              correct.to(output['q_halt_logits']), reduction='none')
    task = (losses * valid).sum() / denom
    q = (halt * valid).sum() / denom
    stability = output.get('stability_loss', task.new_zeros(()))
    loss = task + halt_weight * q + stability_weight * stability
    return loss, dict(task_loss=float(task.detach()), halt_loss=float(q.detach()),
                      exact_match=float(correct.sum() / denom), valid_examples=int(valid.sum()))
