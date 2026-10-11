from pathlib import Path
import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from treemapmix import PlackettLuceLoss, SoftCrossEntropyLoss, MixedLoss


def _worker(rank, store, output, name):
    dist.init_process_group(
        "gloo", init_method=Path(store).as_uri(), rank=rank, world_size=2
    )
    torch.manual_seed(5)
    model = DistributedDataParallel(torch.nn.Linear(3, 3, bias=False).double())
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    ids = torch.tensor([[0, 1], [1, 2], [0, 2], [0, 0]])
    weights = torch.tensor(
        [[0.2, 0.8], [0.4, 0.6], [0.3, 0.7], [0.0, 0.0]], dtype=torch.float64
    )
    sl = slice(rank * 2, (rank + 1) * 2)
    criterion = {
        "pl": PlackettLuceLoss,
        "sce": SoftCrossEntropyLoss,
        "mixed": MixedLoss,
    }[name]
    loss = criterion(distributed=True)(model(x[sl]), (ids[sl], weights[sl]))
    loss.backward()
    torch.save(
        {"gradient": model.module.weight.grad, "loss": loss.detach()},
        Path(output) / f"rank{rank}.pt",
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("name", ["pl", "sce", "mixed"])
def test_global_valid_sample_gradient(tmp_path, name):
    mp.spawn(
        _worker,
        args=(str(tmp_path / "store"), str(tmp_path), name),
        nprocs=2,
        join=True,
    )
    torch.manual_seed(5)
    model = torch.nn.Linear(3, 3, bias=False).double()
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=torch.float64
    )
    ids = torch.tensor([[0, 1], [1, 2], [0, 2]])
    weights = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]], dtype=torch.float64)
    criterion = {
        "pl": PlackettLuceLoss,
        "sce": SoftCrossEntropyLoss,
        "mixed": MixedLoss,
    }[name]
    expected_loss = criterion()(model(x), (ids, weights))
    expected_loss.backward()
    for rank in range(2):
        torch.testing.assert_close(
            torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)["gradient"],
            model.weight.grad,
        )

        torch.testing.assert_close(
            torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)["loss"],
            expected_loss,
        )
