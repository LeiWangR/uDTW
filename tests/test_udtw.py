import math

import pytest
import torch

from udtw import pairwise_matrices, softmin, uDTW


DTYPE = torch.float64


def _temporal_paths(N, M):
    paths = []

    def visit(i, j, path):
        if i == N - 1 and j == M - 1:
            paths.append(tuple(path))
            return

        for di, dj in ((1, 0), (0, 1), (1, 1)):
            ni, nj = i + di, j + dj
            if ni < N and nj < M:
                visit(ni, nj, path + [(ni, nj)])

    visit(0, 0, [(0, 0)])
    return paths


def _soft_path_values(cost, penalty, gamma):
    values = []
    penalties = []

    for path in _temporal_paths(cost.shape[-2], cost.shape[-1]):
        values.append(
            sum(cost[i, j] for i, j in path)
        )
        penalties.append(
            sum(penalty[i, j] for i, j in path)
        )

    values = torch.stack(values)
    penalties = torch.stack(penalties)

    weights = torch.softmax(-values / gamma, dim=0)
    distance = -gamma * torch.logsumexp(-values / gamma, dim=0)
    expected_penalty = (weights * penalties).sum()

    return distance, expected_penalty


def test_softmin_matches_definition():
    x = torch.tensor([1.0, 2.0, 4.0], dtype=DTYPE)
    gamma = 0.3

    actual = softmin(x, gamma)
    expected = -gamma * torch.logsumexp(-x / gamma, dim=0)

    assert torch.allclose(actual, expected)


def test_pairwise_variance_matches_eq8():
    X = torch.tensor(
        [[[1.0, 2.0], [2.0, 3.0]]],
        dtype=DTYPE,
    )
    Y = torch.tensor(
        [[[0.0, 1.0], [3.0, 4.0]]],
        dtype=DTYPE,
    )
    sx = torch.tensor([[[2.0], [4.0]]], dtype=DTYPE)
    sy = torch.tensor([[[6.0], [8.0]]], dtype=DTYPE)

    _, _, variance = pairwise_matrices(X, Y, sx, sy, beta=1.0)

    expected = torch.tensor(
        [[[20.0, 34.0], [26.0, 40.0]]],
        dtype=DTYPE,
    )
    assert torch.allclose(variance, expected)


@pytest.mark.parametrize(
    "N,M,gamma",
    [
        (1, 1, 0.3),
        (2, 2, 0.2),
        (2, 3, 0.4),
        (3, 3, 0.15),
    ],
)
def test_udtw_matches_exhaustive_paths(N, M, gamma):
    torch.manual_seed(N * 10 + M)

    X = torch.rand(1, N, 3, dtype=DTYPE)
    Y = torch.rand(1, M, 3, dtype=DTYPE)
    sx = 0.5 + torch.rand(1, N, 1, dtype=DTYPE)
    sy = 0.5 + torch.rand(1, M, 1, dtype=DTYPE)

    cost, penalty, _ = pairwise_matrices(
        X, Y, sx, sy, beta=1.0
    )

    actual_d, actual_p = uDTW(
        gamma=gamma,
    ).forward(X, Y, sx, sy, beta=1.0)

    expected_d, expected_p = _soft_path_values(
        cost[0], penalty[0], gamma
    )

    assert torch.allclose(
        actual_d[0], expected_d, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        actual_p[0], expected_p, atol=1e-10, rtol=1e-10
    )


def test_normalized_udtw_matches_definition():
    torch.manual_seed(20)

    X = torch.rand(1, 3, 2, dtype=DTYPE)
    Y = torch.rand(1, 4, 2, dtype=DTYPE)
    sx = 0.5 + torch.rand(1, 3, 1, dtype=DTYPE)
    sy = 0.5 + torch.rand(1, 4, 1, dtype=DTYPE)

    raw = uDTW(gamma=0.3, normalize=False)
    normalized = uDTW(gamma=0.3, normalize=True)

    d_xy, p_xy = raw(X, Y, sx, sy, beta=1.0)
    d_xx, p_xx = raw(X, X, sx, sx, beta=1.0)
    d_yy, p_yy = raw(Y, Y, sy, sy, beta=1.0)

    expected_d = d_xy - 0.5 * (d_xx + d_yy)
    expected_p = p_xy - 0.5 * (p_xx + p_yy)

    actual_d, actual_p = normalized(X, Y, sx, sy, beta=1.0)

    assert torch.allclose(actual_d, expected_d)
    assert torch.allclose(actual_p, expected_p)


def test_full_bandwidth_matches_unpruned():
    # For N=M=4, bandwidth=3 admits every DP cell, so it is
    # equivalent to disabling pruning.
    torch.manual_seed(21)

    X = torch.rand(1, 4, 2, dtype=DTYPE)
    Y = torch.rand(1, 4, 2, dtype=DTYPE)
    sx = 0.5 + torch.rand(1, 4, 1, dtype=DTYPE)
    sy = 0.5 + torch.rand(1, 4, 1, dtype=DTYPE)

    a = uDTW(gamma=0.2, bandwidth=None)
    b = uDTW(gamma=0.2, bandwidth=3)

    da, pa = a(X, Y, sx, sy, beta=1.0)
    db, pb = b(X, Y, sx, sy, beta=1.0)

    assert torch.allclose(da, db, atol=1e-10, rtol=1e-10)
    assert torch.allclose(pa, pb, atol=1e-10, rtol=1e-10)


def test_gradcheck():
    torch.manual_seed(22)

    X = torch.rand(1, 2, 2, dtype=DTYPE, requires_grad=True)
    Y = torch.rand(1, 2, 2, dtype=DTYPE, requires_grad=True)
    sx = (0.8 + torch.rand(1, 2, 1, dtype=DTYPE)).requires_grad_()
    sy = (0.8 + torch.rand(1, 2, 1, dtype=DTYPE)).requires_grad_()

    criterion = uDTW(gamma=0.3, normalize=False)

    def distance_fn(x, y, s_x, s_y):
        d, p = criterion(x, y, s_x, s_y, beta=0.7)
        return d.sum() + p.sum()

    assert torch.autograd.gradcheck(
        distance_fn,
        (X, Y, sx, sy),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-5,
    )


def test_backward_is_finite():
    torch.manual_seed(23)

    X = torch.rand(2, 3, 4, dtype=DTYPE, requires_grad=True)
    Y = torch.rand(2, 4, 4, dtype=DTYPE, requires_grad=True)
    sx = (0.8 + torch.rand(2, 3, 1, dtype=DTYPE)).requires_grad_()
    sy = (0.8 + torch.rand(2, 4, 1, dtype=DTYPE)).requires_grad_()

    criterion = uDTW(gamma=0.2, normalize=True)

    d, p = criterion(X, Y, sx, sy, beta=1.0)
    loss = (d + p).sum()
    loss.backward()

    for tensor in (X, Y, sx, sy):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_singleton_is_exact():
    X = torch.tensor([[[1.0, 2.0]]], dtype=DTYPE)
    Y = torch.tensor([[[1.5, 2.5]]], dtype=DTYPE)
    sx = torch.tensor([[[2.0]]], dtype=DTYPE)
    sy = torch.tensor([[[4.0]]], dtype=DTYPE)

    gamma = 0.1
    d, p = uDTW(gamma=gamma)(X, Y, sx, sy, beta=1.0)

    variance = 0.5 * (2.0 ** 2 + 4.0 ** 2)
    squared_distance = 0.5
    expected_d = squared_distance / variance
    expected_p = math.log(variance)

    assert torch.allclose(
        d,
        torch.tensor([expected_d], dtype=DTYPE),
    )
    assert torch.allclose(
        p,
        torch.tensor([expected_p], dtype=DTYPE),
    )


def test_cuda_matches_cpu_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(24)

    X_cpu = torch.rand(2, 3, 4, dtype=DTYPE)
    Y_cpu = torch.rand(2, 4, 4, dtype=DTYPE)
    sx_cpu = 0.8 + torch.rand(2, 3, 1, dtype=DTYPE)
    sy_cpu = 0.8 + torch.rand(2, 4, 1, dtype=DTYPE)

    criterion = uDTW(gamma=0.2, normalize=True)

    d_cpu, p_cpu = criterion(
        X_cpu, Y_cpu, sx_cpu, sy_cpu, beta=1.0
    )

    d_gpu, p_gpu = criterion(
        X_cpu.cuda(),
        Y_cpu.cuda(),
        sx_cpu.cuda(),
        sy_cpu.cuda(),
        beta=1.0,
    )

    assert torch.allclose(
        d_cpu,
        d_gpu.cpu(),
        atol=1e-9,
        rtol=1e-8,
    )
    assert torch.allclose(
        p_cpu,
        p_gpu.cpu(),
        atol=1e-9,
        rtol=1e-8,
    )


def test_penalty_uses_paper_eq3_eq16_convention():
    # Eq. (3)/(16): Omega aggregates log(Sigma), where Sigma is variance.
    X = torch.tensor([[[0.0]]], dtype=DTYPE)
    Y = torch.tensor([[[1.0]]], dtype=DTYPE)
    sx = torch.tensor([[[2.0]]], dtype=DTYPE)
    sy = torch.tensor([[[4.0]]], dtype=DTYPE)

    distance, penalty = uDTW(gamma=0.1)(X, Y, sx, sy, beta=3.0)

    variance = 0.5 * (2.0 ** 2 + 4.0 ** 2)
    expected_d = 1.0 / variance
    expected_p = 3.0 * math.log(variance)

    assert torch.allclose(
        distance,
        torch.tensor([expected_d], dtype=DTYPE),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        penalty,
        torch.tensor([expected_p], dtype=DTYPE),
        atol=1e-12,
        rtol=1e-12,
    )
