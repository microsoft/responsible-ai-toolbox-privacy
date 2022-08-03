# Introduction 

The goal of this project is to provide a uniform approach for reasoning about confidentiality of ML models trained on enterprise language data.
Our approach relies on three building blocks:

1. Cryptographic games for expressing data preprocessing, training, adversary capabilities, and confidentiality goals in a uniform way, and for formalizing their connections; 
2. Dataset specifications to explicitly model assumptions about enterprise data;
3. Empirical lower bounds for differentially private training algorithms on language models, based on dataset specifications.


## Environment Setup

### AML Environment

This environment is needed for submitting experiments to Azure Machine Learning (AML).
It is recommended to create a new virtual environment for this (conda, venv, etc.)

```
conda create -n PrivacyGames-AML python=3.8
conda activate PrivacyGames-AML
bash setup-aml-environment.sh
```


# Experiments

Experiments are submitted using Azure Machine Learning (AML) workspaces.


## AML Pipelines

We use Azure ML Pipelines to define the flow of data.
AML pipelines automatically cache expensive experiments like training a large number of language models.

Pipelines are defined as python functions in a short python script.
See `/pipelines` for examples.

Typically these scripts do the following three steps: (i) Load components, (ii) instantiate components with hyper-parameters (iii) submit the pipeline to a compute cluster.


### AML Components

Azure ML Components are reproducible units of computation.
For more details see [Components Documentation](https://componentsdk.azurewebsites.net/concepts/component.html).

We can get a component factory using a helper function:

``` python
get_component_factory("path/to/componet_spec.yaml", version="local")
```

The `version` parameter determines whether the local copy of the component will be used or a versioned from the AML workspace.


#### Re-using component output

AML will try to automatically re-use component output.
However, often it is difficult to clearly define dependencies of a component on various files in the repository.
We can therefore upload a whole snapshot of repository and create a versioned instance of the component.


```
python bin/register-components.py --workspace_config config-hai7.json
```

This registers all components.
Make sure the version number in the `./VERSION` file is updated.
In general, semantic versioning is recommended.

We can then load the pinned version of the component when defining the pipeline

``` python
get_component_factory("path_to_component_spec.yaml", version="0.1.0dev7")
```

## Submitting to HAI7

In order to submit an experiment we need to specify pipeline, set of hyper-parameters and workspace.
We can do that by running:

```
python pipelines/text_classification.py --workspace_config config-hai7.json --json-config runs/text_classification/sst2+extra-amazon+eps-8.json 
```

This will submit the text classification pipeline to HAI7.


## Directory layout

### Repository

```
|-- bin                           Executable scripts
|-- components                    Definition of AML components (see [https://componentsdk.azurewebsites.net/concepts/component.html](https://componentsdk.azurewebsites.net/concepts/component.html))
|   |-- ...
|   |-- estimate-epsilon
|   |   |-- component_spec.yaml   Definition of entry script, inputs, outputs, compute targets, etc. of this AML component.
|   |   |-- environment.yaml      Definition of environment this component will be run in.
|   |-- ...
|-- pipelines                     Definition of pipelines. These are end-to-end experiments, starting from datasets and ending in a privacy estimate (see [https://componentsdk.azurewebsites.net/concepts/pipeline.html](https://componentsdk.azurewebsites.net/concepts/pipeline.html))
|-- privacy_games                 Python package containing common routines for computing privacy estimates.
|-- runs                          Hyper-parameter configurations for experiments.
|-- tests                         Unit tests for the python package.
```


### Naming conventions

- `m` = For the heuristics, the number of samples added to the base dataset for training the model
- `N` = The total number of challenge samples across all models in an experiment

