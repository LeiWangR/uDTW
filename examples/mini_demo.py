"""Minimal runnable uDTW example.

Run from the repository root:

    python examples/mini_demo.py

The example uses a tiny uncertainty network and two short sequences.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Make direct execution work from a fresh source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uDTW import uDTW


class SimpleSigmaNet(nn.Module):
    """Tiny SigmaNet-style example returning positive std values."""

    def __init__(self, dims):
        super(SimpleSigmaNet, self).__init__()
        self.fc1 = nn.Linear(dims, 2 * dims)
        self.fc2 = nn.Linear(2 * dims, dims)

    def forward(self, x, a=1.5, b=0.5):
        B, T, D = x.shape
        h = x.reshape(B * T, D)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        h = h.reshape(B, T, D).mean(dim=2, keepdim=True)
        return a * torch.sigmoid(h) + b


def main():
    torch.manual_seed(0)

    B, N, M, D = 2, 5, 6, 4

    x = torch.rand(B, N, D)
    y = torch.rand(B, M, D)

    sigmanet = SimpleSigmaNet(D)

    criterion = uDTW(
        use_cuda=torch.cuda.is_available(),
        gamma=0.1,
        normalize=True,
    )

    optimizer = optim.SGD(sigmanet.parameters(), lr=0.1)

    for step in range(3):
        optimizer.zero_grad()

        sigma_x = sigmanet(x)
        sigma_y = sigmanet(y)

        distance, penalty = criterion(
            x, y, sigma_x, sigma_y, beta=1.0
        )
        loss = (distance.mean() + penalty.mean()) / (N * M)

        loss.backward()
        optimizer.step()

        print(
            "step {:d} | loss {:.8f}".format(
                step, loss.item()
            )
        )


if __name__ == "__main__":
    main()
