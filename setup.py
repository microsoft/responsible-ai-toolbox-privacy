from setuptools import setup


VERSION='0.1.0post1'
    
with open('README.md') as f:
    long_description = f.read()

setup(
    name='privacy-estimates',
    version=VERSION,
    description='Empirical Privacy Estimates',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/privacy-estimates',
    packages=['privacy_estimates'],
    include_package_data=True,
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
