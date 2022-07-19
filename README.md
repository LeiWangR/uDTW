# uDTW

## 1. Overview

This repo shows a very simple use case of the uncertainty-DTW (uDTW).

The implementation is based on pytorch implementation of soft-DTW (sDTW).

## 2. Setup

The environment file has been provided in this repo namely as `myenv.yml`. Follow the steps below for the environment setup:

- create the environment from the provided yml file: `conda env create -f myenv.yml`

- activate the new environment: `conda activate myenv`

## 3. Usage

A sample code has been provided in `example.py`. Simply type `python3 example.py` in command to run the example codes. Here's a quick example:

```python
s = "Python syntax highlighting"
print s
```


## 4. Result

Random seed has been setup for you to reproducing (results shown below). Please ignore the NumbaPerformanceWarning when running on GPU with CUDA.

### On GPU:

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

### On CPU:

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
