# uDTW

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

*The code has been optimized with GPT for clarity, readability,
maintainability, and ease of reuse. An extension of this work is coming soon!* 

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

## Pairwise uncertainty

For the additive model in Eq. (8), the pairwise variance is

```text
Sigma_ij = 0.5 * (sigma_i^2 + sigma'_j^2)
```

The uncertainty-weighted local cost is

```text
||x_i - y_j||^2 / Sigma_ij
```

as specified by Eq. (16). Follows the formal Eq. (3)/(16) definition, the returned beta-weighted penalty uses:

```text
beta * log(Sigma_ij)
```

## Core dynamic program

Let

```text
C_ij = ||x_i-y_j||^2 / Sigma_ij.
```

The uDTW distance is the SoftMin over complete DTW path costs. The DP is:

```text
R_ij = C_ij
       + SoftMin_gamma(
           R_{i-1,j-1},
           R_{i-1,j},
           R_{i,j-1}
         )
```

The uncertainty penalty uses the same soft path weights. If

```text
q_k = SoftMax(-R_pred_k / gamma)
```

then:

```text
P_ij = penalty_ij + sum_k q_k P_pred_k
```

This is the dynamic-programming form of the SoftMinSel definition in Eq. (3):
the penalty is an expectation under the same soft distribution over paths
that produces the uDTW distance.

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

## Differentiability

Updated implementation uses only PyTorch operations and avoids in-place modification of autograd-tracked DP tensors.

Therefore:

```python
loss = distance.mean() + penalty.mean()
loss.backward()
```

works through both the sequence features and SigmaNet outputs.

## Verification

Run:

```bash
pytest -q
```

The test suite independently checks:

1. the pairwise variance against Eq. (8);
2. uDTW distance against exhaustive enumeration of all DTW paths;
3. the uncertainty penalty against the corresponding exact soft path
   expectation;
4. normalized uDTW against its explicit definition;
5. bandwidth consistency in a case where the full DTW path set is unchanged;
6. PyTorch `gradcheck`;
7. finite gradients under `.backward()`;
8. the singleton boundary case.

## CPU / GPU

This implementation follows the device of its input tensors, so the
same code works on CPU or CUDA.


## Citation

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
