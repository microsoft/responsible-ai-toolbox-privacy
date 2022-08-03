from setuptools import setup

with open("VERSION") as f:
    VERSION=f.read().strip()

setup(
    name='privacy_games',
    version=VERSION,
    description='Privacy Games',
    packages=['privacy_games', 'privacy_games.estimates'],
    install_requires=[
        "statsmodels",
        "numpy",
        "scipy",
        "multimethod",
        "datasets<2.0.0"
    ],
    extras_require={
        "dev": [
            "pytest",
            "sympy"
        ]
    }
)