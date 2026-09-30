"""Original-style uDTW training example

Run:
    python examples/original_style_example.py

The code structure follows the example supplied with the original repository.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uDTW import uDTW


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.constant_(module.bias, 0.0)


class SimpleSigmaNet(nn.Module):
    def __init__(self):
        super(SimpleSigmaNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x, a=1.5, b=0.5):
        batch_size = x.shape[0]
        length = x.shape[1]

        x = x.reshape(batch_size * length, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(batch_size, length, -1).mean(2, keepdim=True)

        return a * torch.sigmoid(x) + b


def main():
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size, len_x, len_y, dims = 4, 6, 9, 10

    x = torch.rand(
        batch_size, len_x, dims,
        device=device,
    )
    y = torch.rand(
        batch_size, len_y, dims,
        device=device,
    )

    sigmanet = SimpleSigmaNet().to(device)
    sigmanet.apply(weight_init)

    udtw = uDTW(
        use_cuda=torch.cuda.is_available(),
        gamma=0.01,
        normalize=True,
    )

    optimizer = optim.SGD(
        sigmanet.parameters(),
        lr=0.5,
        momentum=0.9,
    )

    for epoch in range(10):
        optimizer.zero_grad()

        sigma_x = sigmanet(x)
        sigma_y = sigmanet(y)

        loss_d, loss_s = udtw(
            x, y,
            sigma_x, sigma_y,
            beta=1.0,
        )

        loss = (loss_d.mean() + loss_s.mean()) / (len_x * len_y)

        loss.backward()
        optimizer.step()

        print(
            "epoch {:d} | loss {:.10f}".format(
                epoch,
                loss.item(),
            )
        )


if __name__ == "__main__":
    main()
