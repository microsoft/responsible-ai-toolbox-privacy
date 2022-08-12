# Introduction

This repository provides utilities for estimating DP-$\varepsilon$ from the confusion matrix of a membership attack.

## Installation

Simply run the following command in the root of the repository:

``` bash
pip install .[dev] 
```
This should install all the relevant dependencies

## Example

For example, we can post-process a MIA on a CNN trained on CIFAR10 with $(\varepsilon = 10, \delta = 10^{-5})$ by running

``` bash
python bin/estimate-epsilon.py --alpha 0.1 --delta 1e-5 --TP 487 --TN 1 --FP 512 --FN 0 
```

This should take approximately 5 minutes and produce the following output

``` bash
Method             Interval                Significance level  eps_lo  eps_hi
Joint beta (ours)  two-sided equal-tailed  0.100               0.145   6.399
Joint beta (ours)  one-sided               0.050               0.145   inf
Clopper Pearson    two-sided equal-tailed  0.100               0.000   inf
Clopper Pearson    one-sided               0.050               0.000   inf
Jeffreys           two-sided equal-tailed  0.100               0.000   inf
Jeffreys           one-sided               0.050               0.000   inf
```


## Tests

We provide a few test cases which can be run by

``` bash
pytest .
```

