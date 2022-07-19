# uDTW

## 1. Overview

This repo shows a very simple use case of the uncertainty-DTW (uDTW).

The implementation is based on pytorch implementation of soft-DTW (sDTW).

Note that we have solved the issues of using the bandwitdh argument for specifying the Sakoe-Chiba bandwidth for pruning.

## 2. Setup

The environment file has been provided in this repo namely as `myenv.yml`. Follow the steps below for the environment setup:

- create the environment from the provided yml file: `conda env create -f myenv.yml`

- activate the new environment: `conda activate myenv`

## 3. Usage

A sample code has been provided in `example.py`. Simply type `python3 example.py` in command to run the example codes. Here's a quick example:

```python
torch.manual_seed(0)
# create the sequences
batch_size, len_x, len_y, dims = 4, 6, 9, 10
# sequence x & y
if torch.cuda.is_available():
    x = torch.rand((batch_size, len_x, dims)).cuda()
    y = torch.rand((batch_size, len_y, dims)).cuda()
else:
    x = torch.rand((batch_size, len_x, dims))
    y = torch.rand((batch_size, len_y, dims))

# define parameters for scaled sigmoid function
a = 1.5
b = 0.5

# a very simple network
sigmanet = SimpleSigmaNet()
sigmanet.apply(weight_init)

if torch.cuda.is_available():
    sigmanet.cuda()

# create the criterion object
if torch.cuda.is_available():
    udtw = SoftDTW(use_cuda=True, gamma=0.01, normalize=True)
else:
    udtw = SoftDTW(use_cuda=False, gamma=0.01, normalize=True)

# set optimizer
optimizer = optim.SGD(sigmanet.parameters(), lr=0.5, momentum=0.9)

for epoch in range(10):
    optimizer.zero_grad()

    sigma_x = sigmanet(x, a, b)
    sigma_y = sigmanet(y, a, b)

    # Compute the loss value
    loss_d, loss_s = udtw(x, y, sigma_x, sigma_y, beta = 1)
    loss = (loss_d.mean() + loss_s.mean()) / (len_x * len_y)

    print('epoch ', epoch, ' | loss: ', '{:.10f}'.format(loss.item()))

    # aggregate and call backward()
    loss.backward()
    optimizer.step()
```


## 4. Result

Random seed has been setup for you to reproducing (results shown below). Please ignore the NumbaPerformanceWarning when running on GPU with CUDA.

#### On GPU:

```
epoch  0  | loss:  0.0022070494
epoch  1  | loss:  0.0021751155
epoch  2  | loss:  0.0021154413
epoch  3  | loss:  0.0020303086
epoch  4  | loss:  0.0019259036
epoch  5  | loss:  0.0018087073
epoch  6  | loss:  0.0016828900
epoch  7  | loss:  0.0015527449
epoch  8  | loss:  0.0014224321
epoch  9  | loss:  0.0013002274
```

#### On CPU:

```
epoch  0  | loss:  0.0022070487
epoch  1  | loss:  0.0021751146
epoch  2  | loss:  0.0021154408
epoch  3  | loss:  0.0020303084
epoch  4  | loss:  0.0019259042
epoch  5  | loss:  0.0018087067
epoch  6  | loss:  0.0016828905
epoch  7  | loss:  0.0015527471
epoch  8  | loss:  0.0014224334
epoch  9  | loss:  0.0013002268
```

Note that you should get very close/similar results on both GPU and CPU (< 1e-8 differences).

## 5. Citation
<a name="citation"></a>

You can cite the following paper for the use of our uDTW:

```
@article{udtw_eccv22,
author={Lei Wang and Piotr Koniusz},
journal={European Conference on Computer Vision},
title={Uncertainty-DTW for Time Series and Sequences},
year={2022},
pages={},
month={},}
```
