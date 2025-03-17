#!/bin/bash
WORKSPACE_CONFIG="config-hai7.json"

# stop after first error
set -e

# SST2
python -m pipelines.text_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/text_classification_lira/sst2+eps-8.json
python -m pipelines.text_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/text_classification_lira/sst2+eps-inf.json
python -m pipelines.text_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/text_classification_lira/sst2+eps-8+m-8.json
python -m pipelines.text_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/text_classification_lira/sst2+eps-inf+m-8.json
python -m pipelines.text_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/text_classification_lira/sst2+eps-8+m-128.json
python -m pipelines.text_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/text_classification_lira/sst2+eps-inf+m-128.json

# CIFAR10
#python -m pipelines.image_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/image_classification_lira/cifar10+eps-inf.json
#python -m pipelines.image_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/image_classification_lira/cifar10+eps-8.json
#python -m pipelines.image_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/image_classification_lira/cifar10+eps-inf+m-8.json
#python -m pipelines.image_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/image_classification_lira/cifar10+eps-8+m-8.json
#python -m pipelines.image_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/image_classification_lira/cifar10+eps-inf+m-128.json
#python -m pipelines.image_classification_lira --workspace-config $WORKSPACE_CONFIG --json-config runs/image_classification_lira/cifar10+eps-8+m-128.json
