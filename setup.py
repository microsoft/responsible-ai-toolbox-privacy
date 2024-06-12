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
    package_data={
        '': ['VERSION'],
        "privacy_estimates.experiments.components": [
            "*/environment.aml.yaml",
            "*/environment.aml.yml",
            "*/environment.conda.yaml",
            "*/environment.conda.yml",
            "*/component_spec.yaml",
            "*/component_spec.yml",
        ],
        "privacy_estimates.experiments.attacks": [
            "*/environment.yaml",
            "*/environment.yml",
            "*/component_spec.yml",
            "*/component_spec.yaml",
        ]
    },
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
        "scripts/estimate-epsilon.py",
        "experiments/scripts/debug-component.py",
    ]
)
