#!/bin/bash

pip install -r requirements-aml.txt
az upgrade --yes
az extension remove -n ml
az extension remove -n azure-cli-ml
az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/componentsdk/azure_cli_ml-0.9.8-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/componentsdk/0.9.8 --yes --verbose
