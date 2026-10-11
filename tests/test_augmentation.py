import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader
from treemapmix import (
    TreemapMix,
    TreemapDataset,
    DistributedViewSampler,
    collate_sparse,
)
from treemapmix.data import build_rows
from treemapmix.layout import _apply_box_symmetry, _apply_box_symmetry_rect
from treemapmix.schedules import AlphaScheduler, KScheduler


@pytest.mark.parametrize("height,width", [(24, 24), (17, 31)])
def test_canvas_and_symmetry_coverage(height, width):
    torch.manual_seed(3)
    mix = TreemapMix(k=4)
    for weights in [torch.tensor([0.01, 0.04, 0.15, 0.8]), torch.full((4,), 0.25)]:
        boxes = mix.layout(weights, height, width)
        symmetries = range(8) if height == width else (0, 2, 4, 5)
        for sym in symmetries:
            transformed = (
                _apply_box_symmetry(boxes, sym, width)
                if height == width
                else _apply_box_symmetry_rect(boxes, sym, height, width)
            )
            canvas = torch.zeros(height, width, dtype=torch.long)
            for x0, y0, x1, y1 in transformed.tolist():
                assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
                canvas[y0:y1, x0:x1] += 1
            assert (canvas == 1).all()


def test_cyclic_exposure_and_unmixed_policy():
    images = torch.stack([torch.full((3, 16, 16), float(i)) for i in range(4)])
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    output, (labels, w) = TreemapMix(4)(images, torch.arange(4), weights)
    assert output.shape == images.shape
    for c in range(4):
        torch.testing.assert_close(w[labels == c].sort().values, weights)
    with pytest.raises(ValueError):
        TreemapMix(4)(images, torch.tensor([0, 0, 1, 2]))
    mix = TreemapMix(4, duplicate_policy="unmixed")
    output, (labels, w) = mix(images, torch.tensor([0, 0, 1, 2]))
    torch.testing.assert_close(output, images)
    assert w.shape == (4, 1) and (w == 1).all() and mix.unmixed_groups == 1


def test_sample_accounting_and_worker_seeds():
    targets = torch.tensor([0] * 8 + [1] * 3 + [2] * 2 + [3])
    base = TensorDataset(torch.randn(len(targets), 3, 12, 12), targets)
    for mode in ["balanced", "natural"]:
        mix = TreemapMix(3, require_distinct=False, duplicate_policy="allow")
        dataset = TreemapDataset(base, targets, mix, mode=mode)
        rows = build_rows(targets, mode, k=3)
        assert len(dataset) == sum(len(r) for r in rows)
        if mode == "natural":
            assert sorted(torch.cat(rows).tolist()) == list(range(len(base)))
        else:
            for row in rows:
                assert len(set(targets[row].tolist())) == 4
        first = dataset[0]
        dataset.set_epoch(0)
        torch.testing.assert_close(first[0], dataset[0][0])
        a = list(
            DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_sparse)
        )
        b = list(
            DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=collate_sparse)
        )
        for x, y in zip(a, b):
            torch.testing.assert_close(x[0], y[0])
            torch.testing.assert_close(x[1][0], y[1][0])
            torch.testing.assert_close(x[1][1], y[1][1])
        indices = [list(DistributedViewSampler(dataset, r, 3)) for r in range(3)]
        valid = [i for row in indices for i in row if i >= 0]
        assert sorted(valid) == list(range(len(dataset)))
        assert len(set(map(len, indices))) == 1


def test_schedule_endpoints():
    scheduler = AlphaScheduler(
        0.1, 0.5, "cosine", step_mode="total", total_epochs=3, cycles_per_epoch=2
    )
    assert scheduler.get_alpha(0, 0) == 0.1
    assert scheduler.get_alpha(2, 1) == 0.5
    scheduler = KScheduler(3, 6, "linear", total_epochs=4)
    assert scheduler.get_k(0) == 3 and scheduler.get_k(3) == 6


def test_area_rounding_and_filter_fallback():
    mix = TreemapMix(k=4)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    torch.manual_seed(4)
    boxes = mix.layout(weights, 256, 256)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (256 * 256)
    assert (areas - weights).abs().max() < 2 / 256
    filtered = TreemapMix(k=4, sampling_enabled=True, min_side=999, max_attempts=2)
    torch.testing.assert_close(filtered.sample_weights(16, 16), torch.full((4,), 0.25))
