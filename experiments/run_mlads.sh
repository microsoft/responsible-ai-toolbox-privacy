#!/bin/bash

SUBMIT="true"
EXTRA_ARGS=""

EXTRA_ARGS_TEXT=""
#EXTRA_ARGS_TEXT="$EXTRA_ARGS_TEXT ++workspace.gpu_compute=ND96asrv4"

EXTRA_ARGS_IMAGE=""
# EXTRA_ARGS_IMAGE="$EXTRA_ARGS_IMAGE ++shared_training_parameters.num_train_epochs=2.0"

# stop after first error
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


python $SCRIPT_DIR/estimate_black_box_privacy_image_classifier.py --config-name black_box_image_classifier +submit=$SUBMIT $EXTRA_ARGS $EXTRA_ARGS_IMAGE
python $SCRIPT_DIR/estimate_black_box_privacy_text_classifier.py --config-name black_box_text_classifier +submit=$SUBMIT $EXTRA_ARGS $EXTRA_ARGS_TEXT

python $SCRIPT_DIR/estimate_differential_privacy_image_classifier.py --config-name dpd_image_classifier +submit=$SUBMIT $EXTRA_ARGS $EXTRA_ARGS_IMAGE
