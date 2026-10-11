"""Treemap composition and complete cyclic group rendering."""

from typing import Dict, List, Optional, Tuple
import torch
from torch.nn import functional as F
from .layout import (
    squarify_core,
    _layout_to_pixel_boxes,
    _fallback_stripes_boxes,
    _boxes_are_valid_and_tile,
    _apply_box_symmetry,
    _apply_box_symmetry_rect,
)


class TreemapMix:
    """Compose K source tensors into K cyclic views with sparse area targets."""

    def __init__(
        self,
        k=4,
        alpha=0.2,
        duplicate_policy="error",
        require_distinct=True,
        sampling_enabled=False,
        max_aspect=10.0,
        min_side=1,
        max_attempts=64,
    ):
        if k < 1 or alpha <= 0 or duplicate_policy not in {"error", "unmixed", "allow"}:
            raise ValueError("Invalid TreemapMix configuration")
        if require_distinct and duplicate_policy == "allow":
            raise ValueError("allow is only available for SCE targets")
        self.mix_k, self.alpha = int(k), float(alpha)
        self.duplicate_policy, self.require_distinct = (
            duplicate_policy,
            require_distinct,
        )
        self.sampling_enabled, self.max_aspect = sampling_enabled, max_aspect
        self.min_side, self.max_attempts = min_side, max_attempts
        self.debug_sym = False
        self.groups = self.unmixed_groups = 0
        self._refresh_indices()

    def _refresh_indices(self):
        self._shift_idx = torch.arange(self.mix_k)
        self._circulant_idx = (
            self._shift_idx[:, None] + self._shift_idx[None, :]
        ) % self.mix_k

    def set_k(self, k):
        if k < 1:
            raise ValueError("k must be positive")
        self.mix_k = int(k)
        self._refresh_indices()

    def layout(self, weights, height, width):
        if self.mix_k > width:
            raise ValueError("k cannot exceed canvas width for stripe fallback")
        desc = weights.sort(descending=True).values
        boxes = _layout_to_pixel_boxes(squarify_core(desc, 1.0).flip(0), height, width)
        if not _boxes_are_valid_and_tile(boxes, height, width):
            boxes = _fallback_stripes_boxes(desc.flip(0), height, width)
        return boxes

    def sample_weights(self, height, width):
        attempts = max(1, self.max_attempts) if self.sampling_enabled else 1
        for _ in range(attempts):
            w = torch.distributions.Dirichlet(
                torch.full((self.mix_k,), self.alpha)
            ).sample()
            if not self.sampling_enabled:
                return w.sort().values
            desc = w.sort(descending=True).values
            boxes = _layout_to_pixel_boxes(
                squarify_core(desc, 1.0).flip(0), height, width
            )
            if not _boxes_are_valid_and_tile(boxes, height, width):
                continue
            sides = boxes[:, 2:] - boxes[:, :2]
            aspect = sides.max(1).values / sides.min(1).values.clamp_min(1)
            if (sides.min(1).values >= self.min_side).all() and (
                self.max_aspect <= 0 or (aspect <= self.max_aspect).all()
            ):
                return w.sort().values
        return torch.full((self.mix_k,), 1 / self.mix_k)

    def __call__(self, images, labels, weights=None, boxes=None):
        if (
            images.ndim != 4
            or len(images) != self.mix_k
            or labels.shape != (self.mix_k,)
        ):
            raise ValueError("Expected images (K,C,H,W) and labels (K,)")
        if not images.is_floating_point() or labels.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Expected floating images and integer labels")
        if (labels < 0).any():
            raise ValueError("Source labels must be nonnegative")
        self.groups += 1
        if labels.unique().numel() != self.mix_k and self.require_distinct:
            if self.duplicate_policy == "error":
                raise ValueError("PL groups require distinct labels")
            self.unmixed_groups += 1
            return images, (labels[:, None], images.new_ones((self.mix_k, 1)))
        height, width = images.shape[-2:]
        if weights is None:
            weights = self.sample_weights(height, width)
        if (
            weights.shape != (self.mix_k,)
            or not torch.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError("Expected K finite positive weights")
        if not torch.allclose(
            weights.sum(), weights.new_tensor(1.0), atol=2e-3, rtol=0
        ):
            raise ValueError("Weights must sum to one")
        desc = weights.sort(descending=True).values
        if boxes is None:
            boxes = self.layout(weights, height, width)
        syms = (
            {s: _apply_box_symmetry(boxes, s, width) for s in range(8)}
            if height == width
            else {
                s: _apply_box_symmetry_rect(boxes, s, height, width)
                for s in (0, 2, 4, 5)
            }
        )
        items = self._render_group(list(images), labels.tolist(), boxes, syms, desc)
        return torch.stack([x[0] for x in items]), (
            torch.stack([x[1][0] for x in items]),
            torch.stack([x[1][1] for x in items]),
        )

    def _render_group(
        self,
        imgs: List[torch.Tensor],
        targets: List[int],
        base_boxes: torch.Tensor,
        sym_boxes_cache: Dict[int, torch.Tensor],
        weights_desc: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Cache slot resizes and render cyclic views in ascending area order."""
        K = self.mix_k

        with torch.no_grad():
            batch = torch.stack(imgs, dim=0).contiguous()
            _, C, H, W = batch.shape
            is_square = H == W

            if is_square:
                sym_choices = torch.randint(0, 8, (K,), dtype=torch.int64)
            else:
                allowed = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
                sym_choices = allowed[torch.randint(0, allowed.numel(), (K,))]

            weights_asc = torch.flip(weights_desc, dims=[0]).float().contiguous()
            weights_asc = weights_asc.to(device=batch.device)

            need_swap_cache = False
            if is_square:
                need_swap_cache = bool(
                    torch.any(
                        (sym_choices == 1)
                        | (sym_choices == 3)
                        | (sym_choices == 6)
                        | (sym_choices == 7)
                    ).item()
                )

            resized0: List[torch.Tensor] = []
            resized1: Optional[List[torch.Tensor]] = [] if need_swap_cache else None
            for slot_i in range(K):
                x0 = int(base_boxes[slot_i, 0].item())
                y0 = int(base_boxes[slot_i, 1].item())
                x1 = int(base_boxes[slot_i, 2].item())
                y1 = int(base_boxes[slot_i, 3].item())
                hi = max(1, y1 - y0)
                wi = max(1, x1 - x0)

                resized0.append(
                    F.interpolate(
                        batch, size=(hi, wi), mode="bilinear", align_corners=False
                    )
                )
                if resized1 is not None:
                    resized1.append(
                        F.interpolate(
                            batch, size=(wi, hi), mode="bilinear", align_corners=False
                        )
                    )

            shifts = (
                self._shift_idx.to(batch.device, non_blocking=True)
                if batch.device.type != "cpu"
                else self._shift_idx
            )
            circulant = (
                self._circulant_idx.to(batch.device, non_blocking=True)
                if batch.device.type != "cpu"
                else self._circulant_idx
            )

            tgt = torch.as_tensor(targets, dtype=torch.int64, device=batch.device)
            labels_mat = tgt[circulant]

            out_batch = batch.new_zeros((K, C, H, W))

            for sym_t in torch.unique(sym_choices):
                sym = int(sym_t.item())
                mask = sym_choices == sym
                shift_subset = shifts[mask]

                boxes_sym = sym_boxes_cache[sym]

                swap = is_square and (sym in (1, 3, 6, 7))

                for slot_i in range(K):
                    x0 = int(boxes_sym[slot_i, 0].item())
                    y0 = int(boxes_sym[slot_i, 1].item())
                    x1 = int(boxes_sym[slot_i, 2].item())
                    y1 = int(boxes_sym[slot_i, 3].item())

                    cache = (
                        resized1[slot_i]
                        if (swap and resized1 is not None)
                        else resized0[slot_i]
                    )
                    src_idx = (slot_i + shift_subset) % K
                    patches = cache.index_select(0, src_idx)

                    th = max(1, y1 - y0)
                    tw = max(1, x1 - x0)
                    ph, pw = patches.shape[-2], patches.shape[-1]
                    if ph != th or pw != tw:
                        patches = F.interpolate(
                            patches, size=(th, tw), mode="bilinear", align_corners=False
                        )

                    out_batch[shift_subset, :, y0:y1, x0:x1] = patches

            out: List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = []
            for s in range(K):
                if self.debug_sym:
                    out.append(
                        (
                            out_batch[s],
                            (labels_mat[s], weights_asc, int(sym_choices[s].item())),
                        )
                    )
                else:
                    out.append((out_batch[s], (labels_mat[s], weights_asc)))
            return out
