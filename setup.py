from setuptools import setup, find_packages


with open("VERSION", "r") as f:
    VERSION = f.read().strip()

with open('README.md') as f:
    long_description = f.read()


setup(
    name='privacy-estimates',
    version=VERSION,
    description='Empirical Privacy Estimates',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/privacy-estimates',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': [
        'VERSION',
        "privacy_estimates/experiments/simple_components/*/environment.aml.yaml",
        "privacy_estimates/experiments/simple_components/*/environment.local.yaml",
    ]},
    install_requires=[
        "statsmodels",
        "numpy",
        "scipy",
        "multimethod",
        "pydantic_cli",
        "scikit-learn",
        "shapely",
        "parmap",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sympy",
            "opacus",
            "tensorflow-privacy",
            "mldesigner",
            "datasets",
            "azure-ai-ml",
        ],
        "pipelines": [
            "azure-ai-ml",
            "azureml-core",
            "datasets",
            "hydra-core",
            "mldesigner",
            "mlflow-skinny",
            "mltable",
            "tqdm-loggable",
        ],
    },
    scripts=[
        "scripts/estimate-epsilon.py"
    ]
)
