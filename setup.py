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
        ],
        "privacy_estimates.experiments.scorers": [
            "*/environment.yaml",
            "*/environment.yml",
            "*/component_spec.yml",
            "*/component_spec.yaml",
        ],
    },
    install_requires=[
        "argparse_dataclass",
        "statsmodels",
        "numpy",
        "scipy",
        "multimethod",
        "scikit-learn",
        "shapely",
        "parmap",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sympy",
            "nltk",
            "opacus",
            "tensorflow-privacy",
            "torch",
        ],
        "pipelines": [
            "argparse_dataclass",
            "azure-ai-ml",
            "azureml-core",
            "azureml-fsspec",
            "datasets",
            "guidance",
            "hydra-core",
            "mldesigner",
            "mlflow-skinny",
            "mltable",
            "numpy<2",
            "tqdm-loggable",
            "pyyaml",
        ],
    },
    scripts=[
        "experiments/scripts/debug-component.py",
    ],
    entry_points={
        'console_scripts': [
            'compile-privacy-estimates-components=privacy_estimates.experiments.scripts.compile_components:run_main',
        ],
    },
)
