# Empirical Estimation of Differential Privacy

This repository provides utilities for estimating DP-$\varepsilon$ from the confusion matrix of a membership inference attack based on the paper <a href="/https://arxiv.org/abs/2206.05199">Bayesian Estimation of Differential Privacy</a>.

## Installation

Simply run the following command to install the privacy-estimates python package. It should install all the relevant dependencies as well.

``` bash
pip install privacy-estimates
```


## Example

The following command takes the output of a membership inference attack on a target model or multiples models in the form of true positives (TP), true negatives (TN), false positives (FP) and false negatives (FN). It also requires the value for  $\alpha$ which states the significance level of the estimate for two sided intervals of the estimated $\varepsilon$ value.

For example, we can post-proces the attack outputs of a CNN trained on CIFAR10 with $(\varepsilon = 10, \delta = 10^{-5})$ by running

``` bash
python scripts/estimate-epsilon.py --alpha 0.1 --delta 1e-5 --TP 487 --TN 1 --FP 512 --FN 0 
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

# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.