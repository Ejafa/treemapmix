# TreemapMix

PyTorch implementation of **TreemapMix: Dirichlet-Controlled Multi-Image Augmentation for Probability and Ordinal Supervision**.

TreemapMix samples a K-way Dirichlet area distribution, partitions the canvas into treemap regions, and produces K cyclic views. Each source visits every area slot once across a complete group. The same composition supports soft cross-entropy (SCE), an area-weighted Plackett-Luce (PL) ranking surrogate, or their convex combination.

## Install

```bash
git clone https://github.com/Ejafa/treemapmix.git
cd treemapmix
python -m pip install ".[timm]"
python examples/minimal.py
```

The core depends on PyTorch. timm, torchvision, and PyYAML are optional training-example dependencies. Tested versions and experimental provenance are recorded in [reproduction notes](docs/reproduction.md).

## Use in a training loop

```python
import torch
from treemapmix import TreemapMix, PlackettLuceLoss

augmentation = TreemapMix(k=4, alpha=0.2)
criterion = PlackettLuceLoss()

# Four preprocessed source tensors and distinct class labels.
images = torch.rand(4, 3, 224, 224)
labels = torch.tensor([0, 1, 2, 3])
mixed_images, target = augmentation(images, labels)
loss = criterion(model(mixed_images), target)
loss.backward()
```

Use `TreemapDataset` and `collate_sparse` to construct groups from a dataset. `set_epoch(epoch)` refreshes deterministic schedules and optional alpha/K curricula; do not use persistent workers with epoch changes. Keep validation images unmixed. Disable other Mixup/CutMix transforms on the same training samples.

## Train with timm

The example uses installed timm without modifying its source. Prepare `train/<class>/...` and `val/<class>/...` folders with matching class names. For ImageNet-LT, the training folders must contain the intended long-tailed subset; grouping preserves the provided counts.

```bash
python examples/train_timm.py --config configs/imagenet_pl.yaml --data /path/to/imagenet --output checkpoints/pl
python examples/train_timm.py --config configs/imagenet_pl.yaml --data /path/to/imagenet --resume checkpoints/pl/last.pt --output checkpoints/pl
python examples/train_timm.py --smoke --epochs 2 --output checkpoints/smoke
```

For distributed training on a supported PyTorch installation:

```bash
torchrun --standalone --nproc-per-node=2 examples/train_timm.py --config configs/imagenet_pl.yaml --data /path/to/imagenet
```

`batch_size` in a configuration is global. The sampler uses zero-weight padding rather than repeating real views, and the loss accounts for the global valid-sample count. Inputs move to the device after collation; the example intentionally uses ordinary loaders rather than timm prefetch wrappers that expect dense class targets. The example removes padded views before ordinary training forwards, keeps a zero-loss dummy forward on empty ranks, and synchronizes model buffers before validation. It does not configure SyncBatchNorm. Checkpoint resume is at epoch boundaries.

CUDA AMP uses bfloat16 when supported and float16 with gradient scaling otherwise. Multi-GPU training uses NCCL and requires a compatible platform and PyTorch build; the tested CPU distributed backend is Gloo.

## Objectives and targets

All losses accept logits `(B,C)` and sparse `(labels, weights)` tensors `(B,K)`. Active weights must be nonnegative, finite, and sum approximately to one. Zero-weight slots are padding; fully padded samples do not contribute to the mean. Half/bfloat16 logits use float32 loss calculations.

| Objective | API | Repeated active labels |
| --- | --- | --- |
| Probability matching | `SoftCrossEntropyLoss()` | Supported; class contributions add |
| Ordinal supervision | `PlackettLuceLoss()` | Rejected |
| Combined supervision | `MixedLoss(beta=0.5)` | Rejected when the PL component is active |

PL follows the paper's log-domain remaining-class normalization. Label-weight pairs must be in ascending area order; tied weights retain input order. This is a weighted ranking surrogate, not a normalized likelihood over rankings. Sampled weights remain targets even when integer pixel areas only approximate them.

For SCE, use `TreemapMix(require_distinct=False, duplicate_policy="allow")`. For PL/mixed on natural-frequency data, the explicitly selected `duplicate_policy="unmixed"` emits each original image once when a group has repeated labels. The training example reports this fraction. This policy changes historical ImageNet-LT behavior.

The old probability-space PL implementation is not included. Reported paper results were produced by the historical [LabelMix code](https://github.com/Ejafa/LabelMix/tree/814678aed0481ecc2c8046fae6419cd9323e3471); this release provides the stable mathematical objective with documented production handling.

## Tests and citation

```bash
python -m pip install ".[test,timm]"
python -m pytest -q
```

Please cite the paper using the metadata in `CITATION.cff`. Geometry and cyclic composition are extracted from LabelMix; timm remains an external dependency. See `LICENSE` and `NOTICE` for attribution.
