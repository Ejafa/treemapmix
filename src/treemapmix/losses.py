"""Sparse area-weighted objectives with padding-aware reduction."""

import torch
from torch import distributed as dist, nn
from torch.nn import functional as F


def _prepare(logits, target):
    labels, weights = target
    if logits.ndim != 2 or labels.ndim != 2 or weights.shape != labels.shape:
        raise ValueError("Expected logits (B,C) and labels/weights (B,K)")
    if labels.shape[0] != logits.shape[0] or logits.shape[1] == 0:
        raise ValueError("Invalid batch size or class count")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("Labels must be integer tensors")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    logits = logits.to(dtype=dtype)
    labels = labels.to(device=logits.device, dtype=torch.long)
    weights = weights.to(device=logits.device, dtype=dtype)
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
        raise ValueError("Logits and weights must be finite")
    if (weights < 0).any():
        raise ValueError("Weights must be nonnegative")
    active = weights > 0
    valid = active.any(1)
    totals = weights.sum(1)
    if not torch.allclose(
        totals[valid], torch.ones_like(totals[valid]), atol=2e-3, rtol=0
    ):
        raise ValueError("Valid sample weights must sum to one")
    if ((labels[active] < 0) | (labels[active] >= logits.shape[1])).any():
        raise ValueError("Active labels outside the class range")
    return logits, labels.masked_fill(~active, 0), weights, active, valid


def _reduce(losses, valid, reduction, distributed):
    if reduction == "none":
        return losses
    total = losses.sum()
    if reduction == "sum":
        return total
    count = valid.sum().to(dtype=losses.dtype)
    if distributed and dist.is_initialized():
        statistics = torch.stack((total.detach(), count))
        dist.all_reduce(statistics)
        denominator = statistics[1].clamp_min(1)
        # DDP averages parameter gradients; keep the reported value globally averaged.
        gradient_loss = total * dist.get_world_size() / denominator
        global_mean = statistics[0] / denominator
        return gradient_loss + (global_mean - gradient_loss.detach())
    return total / count.clamp_min(1)


class SoftCrossEntropyLoss(nn.Module):
    """Weighted sparse CE; repeated labels contribute their combined mass."""

    def __init__(self, reduction="mean", distributed=False):
        super().__init__()
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("Unknown reduction")
        self.reduction, self.distributed = reduction, distributed

    def per_sample(self, logits, target):
        logits, labels, weights, _, valid = _prepare(logits, target)
        losses = -(weights * F.log_softmax(logits, 1).gather(1, labels)).sum(1)
        return losses, valid

    def forward(self, logits, target):
        return _reduce(
            *self.per_sample(logits, target), self.reduction, self.distributed
        )


class PlackettLuceLoss(SoftCrossEntropyLoss):
    """Paper's stable weighted PL surrogate for distinct active labels.

    Labels and weights must be in ascending area order. Zero-weight slots
    are ignored; tied weights keep their input ordering.
    """

    def per_sample(self, logits, target):
        logits, labels, weights, active, valid = _prepare(logits, target)
        if labels.shape[1] == 0:
            return logits.sum(1) * 0, valid
        for row in range(labels.shape[0]):
            ids, w = labels[row, active[row]], weights[row, active[row]]
            if ids.unique().numel() != ids.numel():
                raise ValueError("PL requires distinct active labels")
            if (w[1:] < w[:-1]).any():
                raise ValueError("PL weights must be in ascending area order")
        # Compact active positions first without changing their relative order.
        order = torch.argsort((~active).long(), dim=1, stable=True)
        labels, weights, active = [
            x.gather(1, order) for x in (labels, weights, active)
        ]
        computational_active = active.clone()
        computational_active[~valid, 0] = True
        ranked = logits.gather(1, labels)
        ranked = ranked.masked_fill(~computational_active, -torch.inf)
        present_count = torch.zeros_like(logits, dtype=torch.long)
        present_count.scatter_add_(1, labels, computational_active.long())
        absent = present_count == 0
        has_absent = absent.any(1)
        safe_absent = absent.clone()
        safe_absent[~has_absent, 0] = True
        absent_lse = torch.logsumexp(logits.masked_fill(~safe_absent, -torch.inf), 1)
        absent_lse = absent_lse.masked_fill(~has_absent, -torch.inf)
        denom = torch.logaddexp(absent_lse[:, None], torch.logcumsumexp(ranked, 1))
        terms = denom - logits.gather(1, labels)
        losses = (weights * terms).sum(1)
        return losses, valid


class MixedLoss(SoftCrossEntropyLoss):
    """beta * SCE + (1-beta) * PL, using the same valid-sample reduction."""

    def __init__(self, beta=0.5, reduction="mean", distributed=False):
        super().__init__(reduction, distributed)
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0,1]")
        self.beta = float(beta)
        self.sce, self.pl = SoftCrossEntropyLoss(), PlackettLuceLoss()

    def per_sample(self, logits, target):
        if self.beta == 1:
            return self.sce.per_sample(logits, target)
        if self.beta == 0:
            return self.pl.per_sample(logits, target)
        sce, valid = self.sce.per_sample(logits, target)
        pl, _ = self.pl.per_sample(logits, target)
        return self.beta * sce + (1 - self.beta) * pl, valid
