"""
Core uncertainty-DTW implementation.

The implementation follows the paper's definitions:
  * uncertainty-weighted path cost (Eq. 2 / Eq. 16)
  * uncertainty penalty aggregated with the same soft path weights
    (Eq. 3 / Eq. 16)
  * additive pairwise variance construction (Eq. 8)
  * squared-Euclidean base distance (Eq. 14-16)

The implementation is written in plain PyTorch so the
forward computation is readable and fully differentiable. No Numba or
custom CUDA kernel is required.

Compatibility:
  The public `uDTW` module keeps my original API:
      uDTW(use_cuda=False, gamma=..., normalize=..., bandwidth=...)
  and:
      udtw(X, Y, Sigma_X, Sigma_Y, beta)
returns:
      distance, beta_weighted_uncertainty_penalty
"""

import torch
import torch.nn as nn


def _check_sequence_inputs(X, Y, Sigma_X, Sigma_Y):
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError("X and Y must have shape [batch, time, feature]")
    if Sigma_X.ndim != 3 or Sigma_Y.ndim != 3:
        raise ValueError("Sigma_X and Sigma_Y must have shape [batch, time, 1]")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same batch size")
    if X.shape[2] != Y.shape[2]:
        raise ValueError("X and Y must have the same feature dimension")
    if Sigma_X.shape[:2] != X.shape[:2] or Sigma_Y.shape[:2] != Y.shape[:2]:
        raise ValueError("uncertainty tensors must match batch/time dimensions")
    if Sigma_X.shape[2] != 1 or Sigma_Y.shape[2] != 1:
        raise ValueError("Sigma_X and Sigma_Y must have shape [batch,time,1]")
    if not X.is_floating_point() or not Y.is_floating_point():
        raise TypeError("X and Y must be floating-point tensors")
    if not Sigma_X.is_floating_point() or not Sigma_Y.is_floating_point():
        raise TypeError("Sigma_X and Sigma_Y must be floating-point tensors")


def _check_hyperparameters(gamma, beta, bandwidth):
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    if bandwidth is not None and bandwidth < 0:
        raise ValueError("bandwidth must be None or >= 0")


def softmin(values, gamma, dim=None):
    """Stable differentiable SoftMin."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if dim is None:
        values = values.reshape(-1)
        return -gamma * torch.logsumexp(-values / gamma, dim=0)
    return -gamma * torch.logsumexp(-values / gamma, dim=dim)


def softmin_weights(values, gamma, dim=0):
    """SoftMin path weights, stable for large/small costs."""
    return torch.softmax(-values / gamma, dim=dim)


def pairwise_matrices(X, Y, Sigma_X, Sigma_Y, beta=1.0, eps=1e-8):
    """
    Construct the pairwise squared-Euclidean costs and uncertainty penalty.

    X:       [B, N, D]
    Y:       [B, M, D]
    Sigma_X: [B, N, 1], positive standard deviations from SigmaNet
    Sigma_Y: [B, M, 1], positive standard deviations from SigmaNet

    Eq. (8) gives the pairwise variance:
        Sigma_ij = 0.5 * (sigma_i^2 + sigma'_j^2).

    Eq. (16) then uses:
        d_ij = ||x_i-y_j||^2 / Sigma_ij.

    Returns:
        weighted_cost [B,N,M]
        weighted_penalty [B,N,M]
        variance [B,N,M]
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if beta < 0:
        raise ValueError("beta must be >= 0")

    diff = X.unsqueeze(2) - Y.unsqueeze(1)
    squared_distance = diff.pow(2).sum(dim=3)

    sigma_x2 = Sigma_X.pow(2)
    sigma_y2 = Sigma_Y.pow(2)

    variance = 0.5 * (sigma_x2 + sigma_y2.transpose(1, 2))
    variance = variance.clamp_min(eps)

    weighted_cost = squared_distance / variance
    weighted_penalty = beta * torch.log(variance)

    return weighted_cost, weighted_penalty, variance


def _softdtw_uncertainty(cost, penalty, gamma, bandwidth=None):
    """
    Compute:
      1) SoftMin over uncertainty-weighted path costs.
      2) The same soft path-weighted aggregate of penalty.

    cost, penalty: [B,N,M]

    For each DP cell:
      R_ij = cost_ij + SoftMin(R predecessors)

      P_ij = penalty_ij
             + sum_k q_k P_pred_k

    where:
      q_k = softmin probability induced by predecessor accumulated
            costs R_pred_k.

    This P recurrence is exactly the dynamic-programming form of
    SoftMinSel in Eq. (3): the local penalty is accumulated under the
    same soft distribution over complete paths that defines the
    uncertainty-weighted distance.
    """
    if cost.ndim != 3 or penalty.ndim != 3:
        raise ValueError("cost and penalty must have shape [B,N,M]")
    if cost.shape != penalty.shape:
        raise ValueError("cost and penalty must have identical shapes")

    B, N, M = cost.shape
    if N == 0 or M == 0:
        raise ValueError("sequence lengths must be non-zero")

    # Python lists hold differentiable tensor nodes. No DP tensor is
    # modified in-place, so PyTorch autograd remains valid.
    R = [[None for _ in range(M)] for _ in range(N)]
    P = [[None for _ in range(M)] for _ in range(N)]

    def allowed(i, j):
        if bandwidth is None or bandwidth == 0:
            return True
        return abs(i - j) <= bandwidth

    for i in range(N):
        for j in range(M):
            if not allowed(i, j):
                continue

            predecessors_r = []
            predecessors_p = []

            if i > 0 and j > 0 and R[i - 1][j - 1] is not None:
                predecessors_r.append(R[i - 1][j - 1])
                predecessors_p.append(P[i - 1][j - 1])

            if i > 0 and R[i - 1][j] is not None:
                predecessors_r.append(R[i - 1][j])
                predecessors_p.append(P[i - 1][j])

            if j > 0 and R[i][j - 1] is not None:
                predecessors_r.append(R[i][j - 1])
                predecessors_p.append(P[i][j - 1])

            if i == 0 and j == 0:
                R[i][j] = cost[:, i, j]
                P[i][j] = penalty[:, i, j]
                continue

            if not predecessors_r:
                continue

            pred_r = torch.stack(predecessors_r, dim=0)  # [K,B]
            pred_p = torch.stack(predecessors_p, dim=0)  # [K,B]

            weights = softmin_weights(pred_r, gamma, dim=0)

            R[i][j] = cost[:, i, j] + softmin(pred_r, gamma, dim=0)
            P[i][j] = penalty[:, i, j] + (weights * pred_p).sum(dim=0)

    if R[N - 1][M - 1] is None:
        raise ValueError(
            "No valid DTW path reaches the final cell under the given bandwidth"
        )

    return R[N - 1][M - 1], P[N - 1][M - 1]


class uDTW(nn.Module):
    """
    Uncertainty-DTW module.

    Args:
        use_cuda: Kept for backward compatibility. Computation follows
            the device of the input tensors, so the module works on CPU
            and CUDA without a separate implementation.
        gamma: Soft-DTW relaxation temperature.
        normalize: If True, use the usual sDTW-style divergence:
            d(X,Y) - 0.5[d(X,X)+d(Y,Y)].
            This normalization is a utility option inherited from the
            original code; it is not an additional term in Eq. (2).
        bandwidth: Optional Sakoe-Chiba bandwidth. None disables pruning.

    Forward:
        X, Y: [B,T,D], [B,U,D]
        Sigma_X, Sigma_Y: [B,T,1], [B,U,1]
        beta: penalty coefficient from Eq. (15).

    Returns:
        (distance, beta_weighted_penalty)
    """

    def __init__(self, use_cuda=False, gamma=1.0, normalize=False,
                 bandwidth=None):
        super(uDTW, self).__init__()
        self.use_cuda = bool(use_cuda)
        self.gamma = float(gamma)
        self.normalize = bool(normalize)
        self.bandwidth = None if bandwidth is None else float(bandwidth)

        _check_hyperparameters(self.gamma, 0.0, self.bandwidth)

    def forward(self, X, Y, Sigma_X, Sigma_Y, beta=1.0):
        _check_sequence_inputs(X, Y, Sigma_X, Sigma_Y)
        _check_hyperparameters(self.gamma, beta, self.bandwidth)

        D_xy, S_xy, _ = pairwise_matrices(
            X, Y, Sigma_X, Sigma_Y, beta
        )
        out_xy, pen_xy = _softdtw_uncertainty(
            D_xy, S_xy, self.gamma, self.bandwidth
        )

        if not self.normalize:
            return out_xy, pen_xy

        D_xx, S_xx, _ = pairwise_matrices(
            X, X, Sigma_X, Sigma_X, beta
        )
        D_yy, S_yy, _ = pairwise_matrices(
            Y, Y, Sigma_Y, Sigma_Y, beta
        )

        out_xx, pen_xx = _softdtw_uncertainty(
            D_xx, S_xx, self.gamma, self.bandwidth
        )
        out_yy, pen_yy = _softdtw_uncertainty(
            D_yy, S_yy, self.gamma, self.bandwidth
        )

        distance = out_xy - 0.5 * (out_xx + out_yy)
        penalty = pen_xy - 0.5 * (pen_xx + pen_yy)

        return distance, penalty
