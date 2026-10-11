"""Small helpers for ordinary timm training loops."""

from ..losses import SoftCrossEntropyLoss, PlackettLuceLoss, MixedLoss


def create_loss(name="sce", beta=0.5, distributed=False):
    if name == "sce":
        return SoftCrossEntropyLoss(distributed=distributed)
    if name == "pl":
        return PlackettLuceLoss(distributed=distributed)
    if name == "mixed":
        return MixedLoss(beta, distributed=distributed)
    raise ValueError("loss must be sce, pl, or mixed")


def to_device(batch, device):
    images, (labels, weights) = batch
    return images.to(device, non_blocking=True), (
        labels.to(device, non_blocking=True),
        weights.to(device, non_blocking=True),
    )
