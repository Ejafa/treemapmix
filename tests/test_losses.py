import pytest
import torch
from torch.nn import functional as F
from treemapmix import MixedLoss, PlackettLuceLoss, SoftCrossEntropyLoss


def reference_pl(z, labels, w):
    losses = []
    for row in range(len(z)):
        terms = []
        for i, c in enumerate(labels[row]):
            remaining = [
                j for j in range(z.shape[1]) if j not in labels[row, i + 1 :].tolist()
            ]
            terms.append(
                w[row, i] * (torch.logsumexp(z[row, remaining], 0) - z[row, c])
            )
        losses.append(sum(terms))
    return torch.stack(losses).mean()


@pytest.mark.parametrize("classes,k", [(7, 3), (3, 3), (5, 1)])
def test_pl_value_and_gradient(classes, k):
    torch.manual_seed(4)
    z = torch.randn(4, classes, dtype=torch.float64, requires_grad=True)
    labels = torch.stack([torch.randperm(classes)[:k] for _ in range(4)])
    w = torch.rand(4, k, dtype=z.dtype).sort(1).values
    w /= w.sum(1, keepdim=True)
    actual = PlackettLuceLoss()(z, (labels, w))
    expected = reference_pl(z, labels, w)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, z, retain_graph=True)[0],
        torch.autograd.grad(expected, z)[0],
    )
    assert torch.autograd.gradcheck(lambda x: PlackettLuceLoss()(x, (labels, w)), (z,))


@pytest.mark.parametrize(
    "criterion", [SoftCrossEntropyLoss(), PlackettLuceLoss(), MixedLoss()]
)
def test_padding_extreme_logits(criterion):
    z = torch.tensor([[1000.0, -1000.0, 0.0], [-400.0, 600.0, 0.0]], requires_grad=True)
    labels = torch.tensor([[0, 2], [0, 1]])
    w = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    expected = criterion(z, (labels, w))
    zp = torch.cat([z, torch.zeros(1, 3)])
    lp = torch.cat([labels, torch.tensor([[-999, -999]])])
    wp = torch.cat([w, torch.zeros(1, 2)])
    actual = criterion(zp, (lp, wp))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(z.grad).all()
    for dtype in [torch.float16, torch.bfloat16]:
        x = z.detach().to(dtype).requires_grad_()
        loss = criterion(x, (labels, w))
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(x.grad).all()


def test_duplicate_sce_and_pl_rejection():
    z = torch.randn(1, 3)
    labels, w = torch.tensor([[1, 1]]), torch.tensor([[0.2, 0.8]])
    torch.testing.assert_close(
        SoftCrossEntropyLoss()(z, (labels, w)), F.cross_entropy(z, torch.tensor([1]))
    )
    with pytest.raises(ValueError, match="distinct"):
        PlackettLuceLoss()(z, (labels, w))


@pytest.mark.parametrize(
    "criterion", [SoftCrossEntropyLoss(), PlackettLuceLoss(), MixedLoss()]
)
def test_all_padded_and_empty_slots(criterion):
    for k in [0, 3]:
        z = torch.randn(2, 3, requires_grad=True)
        loss = criterion(z, (torch.full((2, k), -999), torch.zeros(2, k)))
        assert loss == 0 and torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(z.grad).all() and not z.grad.any()


def test_mixed_endpoints_and_invalid_targets():
    z = torch.randn(2, 4)
    labels = torch.tensor([[0, 1], [2, 3]])
    w = torch.tensor([[0.3, 0.7], [0.3, 0.7]])
    for beta, criterion in [(0, PlackettLuceLoss()), (1, SoftCrossEntropyLoss())]:
        torch.testing.assert_close(
            MixedLoss(beta)(z, (labels, w)), criterion(z, (labels, w))
        )
    for invalid in [-w, w * 2, torch.full_like(w, float("nan"))]:
        with pytest.raises(ValueError):
            SoftCrossEntropyLoss()(z, (labels, invalid))


@pytest.mark.parametrize(
    "criterion", [SoftCrossEntropyLoss(), PlackettLuceLoss(), MixedLoss()]
)
def test_interspersed_padding_and_ties(criterion):
    z = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0, 1, 2], [1, 2, 3]])
    w = torch.full((2, 3), 1 / 3, dtype=torch.float64)
    padded_labels = torch.stack(
        [labels[:, 0], torch.full((2,), -999), labels[:, 1], labels[:, 2]], 1
    )
    padded_weights = torch.stack([w[:, 0], torch.zeros(2), w[:, 1], w[:, 2]], 1)
    expected = criterion(z, (labels, w))
    actual = criterion(z, (padded_labels, padded_weights))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(z.grad).all()


def test_single_label_is_cross_entropy():
    z = torch.randn(4, 5, dtype=torch.float64)
    ids = torch.tensor([[0], [1], [2], [3]])
    torch.testing.assert_close(
        PlackettLuceLoss()(z, (ids, torch.ones(4, 1))), F.cross_entropy(z, ids[:, 0])
    )
