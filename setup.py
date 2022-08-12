from setuptools import setup

with open("VERSION") as f:
    VERSION=f.read().strip()

setup(
    name='privacy-estimates',
    version=VERSION,
    description='Privacy estimates',
    packages=['privacy-estimates'],
    install_requires=[
        "statsmodels",
        "numpy",
        "scipy",
        "multimethod",
        "pydantic_cli"
    ],
    extras_require={
        "dev": [
            "pytest",
            "sympy"
        ]
    },
    scripts=[
        "scripts/estimate-epsilon.py"
    ]
)
