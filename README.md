# uDTW

*The code has been optimized with GPT for clarity, readability,
maintainability, and ease of reuse. An extension of this work is coming soon!* 

## Scope

This repository implements the mathematical core of uDTW from the paper:

- uncertainty-weighted path distance, Eq. (2) and Eq. (16);
- uncertainty penalty selected by the same soft path distribution, Eq. (3);
- additive pairwise variance construction, Eq. (8);
- squared-Euclidean base distance used in the Normal-model derivation,
  Eqs. (14)-(16).

My paper also contains application-specific encoding networks,
SigmaNet designs, and downstream pipelines. Those are outside
this small alignment module. The paper describes SigmaNet as the mechanism
used to generate uncertainty parameters end-to-end. 

## Installation

```bash
pip install torch pytest
```

Optional editable installation:

```bash
pip install -e .
```

The tests also work directly from a source checkout.

## Minimal example

From the repository root:

```bash
python3 examples/mini_demo.py
```

A typical output is:

```text
step 0 | loss 0.08003496
step 1 | loss 0.08002817
step 2 | loss 0.08002138
```

For a longer example that follows the `SimpleSigmaNet` training script, run:

```bash
python examples/original_style_example.py
```

A typical output is:

```
epoch 0 | loss 0.1544207633
epoch 1 | loss 0.1532181948
epoch 2 | loss 0.1509874314
epoch 3 | loss 0.1479346603
epoch 4 | loss 0.1442679465
epoch 5 | loss 0.1402121633
epoch 6 | loss 0.1359462440
epoch 7 | loss 0.1316862702
epoch 8 | loss 0.1275756955
epoch 9 | loss 0.1237251833
```

## Core API

Backward-compatible import:

```python
from uDTW import uDTW
```

or:

```python
from udtw import uDTW
```

Given:

```text
X        [B, N, D]
Y        [B, M, D]
Sigma_X  [B, N, 1]
Sigma_Y  [B, M, 1]
```

call:

```python
criterion = uDTW(
    use_cuda=torch.cuda.is_available(),
    gamma=0.1,
    normalize=True,
)

distance, penalty = criterion(
    X, Y,
    Sigma_X, Sigma_Y,
    beta=1.0,
)
```

`Sigma_X` and `Sigma_Y` are positive standard deviations produced by a
SigmaNet-style network.

## Normalization

`normalize=True` uses the standard soft-DTW-style divergence:

```text
uDTW(X,Y) - 0.5[uDTW(X,X) + uDTW(Y,Y)]
```

for both returned quantities.

This normalization is retained for compatibility with my original API and example. It is a utility option rather than an additional term in
the paper's core Eq. (2)-(3) definition.

## Bandwidth

`bandwidth=None` disables pruning.

A positive integer uses a standard Sakoe-Chiba constraint:

```text
abs(i-j) <= bandwidth
```

for valid DP cells.

The core paper formulation does not require a bandwidth, so the default is
`None`.

## CPU / GPU

This implementation follows the device of its input tensors, so the
same code works on CPU or CUDA.


## Citation

If this implementation is useful in your work, please cite the uDTW paper:

```bibtex
@inproceedings{wang2022uncertainty,
  title={Uncertainty-dtw for time series and sequences},
  author={Wang, Lei and Koniusz, Piotr},
  booktitle={European Conference on Computer Vision},
  pages={176--195},
  year={2022},
  organization={Springer}
}
```
