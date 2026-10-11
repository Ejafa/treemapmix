"""Deterministic row schedules and sparse-target dataset integration."""

import math
import random
import sys
from contextlib import contextmanager
from collections import defaultdict, OrderedDict
import torch
from torch.utils.data import Dataset, Sampler
from .augmentation import TreemapMix


@contextmanager
def _seeded(seed):
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        random.seed(seed)
        if numpy is not None:
            numpy.random.seed(seed % 2**32)
        try:
            yield
        finally:
            random.setstate(python_state)
            if numpy is not None:
                numpy.random.set_state(numpy_state)


def build_rows(
    targets, mode="balanced", cycles=None, seed=42, epoch=0, window=1024, k=4
):
    buckets = defaultdict(list)
    for i, label in enumerate(targets):
        buckets[int(label)].append(i)
    if not buckets:
        raise ValueError("Dataset is empty")
    arrays = [torch.tensor(buckets[c]) for c in sorted(buckets)]
    epoch_seed = (seed + 1000003 * epoch) % (2**63 - 1)
    rng = torch.Generator().manual_seed(epoch_seed)
    if mode == "balanced":
        cycles = cycles or math.ceil(len(targets) / len(arrays))
        schedule = torch.stack(
            [a[torch.randint(len(a), (cycles,), generator=rng)] for a in arrays], 1
        )
        return list(
            schedule.gather(1, torch.rand(schedule.shape, generator=rng).argsort(1))
        )
    if mode != "natural":
        raise ValueError("mode must be balanced or natural")
    total, records = len(targets), []
    for ci, a in enumerate(arrays):
        gen = torch.Generator().manual_seed(
            (epoch_seed + 104729 * (ci + 1)) % (2**63 - 1)
        )
        a = a[torch.randperm(len(a), generator=gen)]
        n = len(a)
        positions = (torch.arange(n, dtype=torch.float64) + 0.5) * (total / n)
        jitter = (torch.rand(n, generator=gen, dtype=torch.float64) - 0.5) * min(
            total / n, 1.0
        )
        positions = (positions + jitter).clamp(0, total - 1)
        records.extend(zip(positions.tolist(), a.tolist()))
    records.sort(key=lambda x: x[0])
    order = torch.tensor([i for _, i in records])
    shuffled = [
        chunk[torch.randperm(len(chunk), generator=rng)]
        for chunk in order.split(window)
    ]
    return list(torch.cat(shuffled).split(k))


class TreemapDataset(Dataset):
    """Map-style cyclic views; epoch seeds make worker execution order irrelevant.

    The base dataset must return transformed floating CHW tensors and labels.
    Balanced mode preserves constructed rows; natural mode preserves every
    original source once. Negative indices represent distributed padding.
    """

    def __init__(
        self,
        dataset,
        targets,
        augmentation=None,
        mode="balanced",
        cycles=None,
        seed=42,
        alpha_schedule=None,
        k_schedule=None,
        window=1024,
        cache_groups=64,
    ):
        if len(dataset) != len(targets):
            raise ValueError("Dataset/targets size mismatch")
        self.dataset, self.targets = dataset, list(targets)
        self.augmentation = augmentation or TreemapMix()
        if window < 1 or cache_groups < 1:
            raise ValueError("Window and cache size must be positive")
        self.cache_groups = int(cache_groups)
        self.mode, self.cycles, self.seed = mode, cycles, int(seed)
        self.alpha_schedule, self.k_schedule, self.window = (
            alpha_schedule,
            k_schedule,
            window,
        )
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.k_schedule:
            self.augmentation.set_k(self.k_schedule.get_k(epoch))
        k = self.augmentation.mix_k
        self.rows = build_rows(
            self.targets, self.mode, self.cycles, self.seed, self.epoch, self.window, k
        )
        self.entries, self.groups = [], []
        for row_id, row in enumerate(self.rows):
            rng = torch.Generator().manual_seed(
                self.seed + 1000003 * self.epoch + row_id
            )
            for start in range(0, len(row), k):
                primary = row[start : start + k].tolist()
                indices = list(primary)
                if len(primary) < k and len(row) >= k:
                    earlier = row[:start]
                    indices += earlier[
                        torch.randperm(len(earlier), generator=rng)[: k - len(primary)]
                    ].tolist()
                group_id = len(self.groups)
                self.groups.append((row_id, indices))
                self.entries.extend((group_id, s) for s in range(len(primary)))
        pending, shuffled = [], []
        randomizer = random.Random(self.seed + 1000003 * self.epoch)
        for entry in self.entries:
            pending.append(entry)
            if len(pending) >= self.window:
                choice = randomizer.randrange(len(pending))
                pending[choice], pending[-1] = pending[-1], pending[choice]
                shuffled.append(pending.pop())
        randomizer.shuffle(pending)
        self.entries = shuffled + pending
        self._cache = OrderedDict()
        self.duplicate_groups = sum(
            len({int(self.targets[i]) for i in ids}) < len(ids)
            for _, ids in self.groups
            if len(ids) == k
        )
        self.unmixed_fraction = (
            self.duplicate_groups / max(1, len(self.groups))
            if (
                self.augmentation.require_distinct
                and self.augmentation.duplicate_policy == "unmixed"
            )
            else 0.0
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if index < 0:
            image, _ = self.dataset[0]
            return torch.zeros_like(image), (torch.tensor([0]), torch.tensor([0.0]))
        group_id, shift = self.entries[index]
        if group_id not in self._cache:
            row_id, indices = self.groups[group_id]
            with _seeded(self.seed + 1000003 * self.epoch + 15485863 * (group_id + 1)):
                items = [self.dataset[i] for i in indices]
                images = torch.stack([x[0] for x in items])
                labels = torch.tensor([int(x[1]) for x in items])
                if len(indices) < self.augmentation.mix_k:
                    views = images, (labels[:, None], torch.ones(len(labels), 1))
                else:
                    # Every group in one balanced row shares the sampled weights/layout seed.
                    with _seeded(
                        self.seed + 1000003 * self.epoch + 104729 * (row_id + 1)
                    ):
                        if self.alpha_schedule:
                            self.augmentation.alpha = self.alpha_schedule.get_alpha(
                                self.epoch, row_id
                            )
                        weights = self.augmentation.sample_weights(*images.shape[-2:])
                        boxes = self.augmentation.layout(weights, *images.shape[-2:])
                    views = self.augmentation(images, labels, weights, boxes)
            self._cache[group_id] = views
            if len(self._cache) > self.cache_groups:
                self._cache.popitem(last=False)
        self._cache.move_to_end(group_id)
        images, (labels, weights) = self._cache[group_id]
        return images[shift], (labels[shift], weights[shift])


class DistributedViewSampler(Sampler):
    """Equal rank lengths without repeating real views: use zero-weight padding."""

    def __init__(self, dataset, rank=0, world_size=1):
        if not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank")
        self.dataset, self.rank, self.world_size = dataset, rank, world_size

    def __len__(self):
        return math.ceil(len(self.dataset) / self.world_size)

    def __iter__(self):
        for i in range(len(self)):
            index = i * self.world_size + self.rank
            yield index if index < len(self.dataset) else -1


def collate_sparse(batch):
    images = torch.stack([item[0] for item in batch])
    k = max(len(item[1][0]) for item in batch)
    labels = torch.zeros(len(batch), k, dtype=torch.long)
    weights = torch.zeros(len(batch), k)
    for row, (_, target) in enumerate(batch):
        ids, w = target
        labels[row, : len(ids)] = ids
        weights[row, : len(w)] = w
    return images, (labels, weights)
