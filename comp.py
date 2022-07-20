import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# import uDTW
from uDTW import uDTW
# import sDTW
from sDTW import sDTW

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def sigmoid_ab(a, b, input):
    return a * torch.sigmoid(input) + b

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, input):
        return sigmoid_ab(a, b, input)

class SimpleSigmaNet(nn.Module):
    def __init__(self):
        super(SimpleSigmaNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.sigmoid = Sigmoid()

    def forward(self, x, a, b):
        batch_size = x.shape[0]
        length = x.shape[1]

        x = x.view(batch_size*length, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(batch_size, length, -1).mean(2, keepdim = True)
        sigma = self.sigmoid(a, b, x)

        return sigma

torch.manual_seed(0)
# create the sequences
batch_size, len_x, len_y, dims = 4, 6, 9, 10
# sequence x & y
if torch.cuda.is_available():
    x = torch.rand((batch_size, len_x, dims), requires_grad=True).cuda()
    y = torch.rand((batch_size, len_y, dims)).cuda()
else:
    x = torch.rand((batch_size, len_x, dims), requires_grad=True)
    y = torch.rand((batch_size, len_y, dims))

xcopy = Variable(x.clone().data, requires_grad=True)
ycopy = y.clone().data

# define parameters for scaled sigmoid function
a = 1.5
b = 0.5

# a very simple network
sigmanet_udtw = SimpleSigmaNet()
sigmanet_udtw.apply(weight_init)

if torch.cuda.is_available():
    sigmanet_udtw.cuda()

# create the criterion object
if torch.cuda.is_available():
    udtw = uDTW(use_cuda=True, gamma=0.01, normalize=True)
    sdtw = sDTW(use_cuda=True, gamma=0.01, normalize=True, bandwidth = 1)
else:
    udtw = uDTW(use_cuda=False, gamma=0.01, normalize=True)
    sdtw = sDTW(use_cuda=False, gamma=0.01, normalize=True, bandwidth = 1)

# set optimizer
optimizer_sdtw = optim.SGD([xcopy], lr=0.5, momentum=0.9)
optimizer_udtw = optim.SGD(sigmanet_udtw.parameters(), lr=0.5, momentum=0.9)

for epoch in range(10):
    optimizer_sdtw.zero_grad()
    loss_sdtw = sdtw(xcopy, ycopy)
    loss_sdtw = loss_sdtw.mean() / (len_x * len_y)
    print('epoch ', epoch, ' | ------ sDTW loss: ', '{:.10f}'.format(loss_sdtw.item()))
    loss_sdtw.backward()
    optimizer_sdtw.step()

    # ------
    optimizer_udtw.zero_grad()

    sigma_x_udtw = sigmanet_udtw(x, a, b)
    sigma_y_udtw= sigmanet_udtw(y, a, b)

    # Compute the loss value
    loss_d_udtw, loss_s_udtw = udtw(x, y, sigma_x_udtw, sigma_y_udtw, beta = 1)
    loss_udtw = (loss_d_udtw.mean() + loss_s_udtw.mean()) / (len_x * len_y)

    print('epoch ', epoch, ' | uDTW loss: ', '{:.10f}'.format(loss_udtw.item()))

    # aggregate and call backward()
    loss_udtw.backward()
    optimizer_udtw.step()

