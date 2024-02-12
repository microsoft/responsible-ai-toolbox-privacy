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
    url='https://aka.ms/privacy-estimates',
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    package_data={'': [
        'VERSION',
        "privacy_estimates/experiments/simple_components/*/environment.aml.yaml",
        "privacy_estimates/experiments/simple_components/*/environment.aml.yml",
        "privacy_estimates/experiments/simple_components/*/environment.conda.yaml",
        "privacy_estimates/experiments/simple_components/*/environment.conda.yml",
        "privacy_estimates/experiments/simple_components/*/component_spec.yaml",
        "privacy_estimates/experiments/simple_components/*/component_spec.yml",
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
