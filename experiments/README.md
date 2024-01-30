# Experiments

This folder contains code to run experiments end-to-end in Azure Machine Learning.

We differentiate three types of threat models:
1. Black-box membership inference (coming soon)
2. White-box membership inference (coming soon)
3. Differential privacy distinguisher

## Differential privacy distinguisher

We follow the attack by Nasr et al. (2023) to match the differential privacy threat model.
Currently this does not take privacy amplification via subsampling into account and only audits a single step of DP-SGD.

### Collecting membership inference data

The threat model of Differentially Private Stochastic Gradient Descent (DP-SGD) assumes that the adversary has access to the training data and each gradient in the training process.
In order to instantiate a matching adversary, we need to collect membership information during the training process.
We provide a wrapper (`privacy_estimates.experiments.attack.dpd.CanaryTrackingOptimizer`) for a PyTorch optimizer that can be used with Opacus.

`privacy_estimates.experiments.games.differential_privacy_distinguisher` contains code to run the differentially private distinguisher game.``

## Installation and setup

### Setup the local environment

The local environment is packaged within the `privacy-estimates` package.

```bash
pip install privacy-estimates[pipelines]
```

### Setup Azure ML

Setup an Azure ML workspace and download the `config.json` file ([details](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) and add it to `configs/workspace`.

Add `gpu_compute` and `cpu_compute` to the `config.json` indicating where to run the experiments.
The values should match with compute clusters in your workspace.


## Example 2: Differential Privacy Distinguisher for image classification

We can increase the threat model to match the theoretical bound by using a different attack.
We follow the gradient canary attack by Nasr et al. (2023) to match the differential privacy threat model.

```bash
python estimate_differential_privacy_image_classifier.py --config-name dpd_image_classifier +submit=True
```

## References

Nasr, M., Hayes, J., Steinke, T., Balle, B., Tramèr, F., Jagielski, M., Carlini, N. and Terzis, A., 2023. Tight Auditing of Differentially Private Machine Learning. arXiv preprint arXiv:2302.07956.

Zanella-Béguelin, S., Wutschitz, L., Tople, S., Salem, A., Rühle, V., Paverd, A., Naseri, M., Köpf, B. and Jones, D., 2023, July. Bayesian estimation of differential privacy. In International Conference on Machine Learning (pp. 40624-40636). PMLR.

