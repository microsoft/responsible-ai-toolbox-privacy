#/bin/bash

EXPERIMENT_DIR="$(dirname "$(readlink -f "$0")")"

set -e

python $EXPERIMENT_DIR/estimate_differential_privacy_image_classifier.py --config-name dpd_image_classifier +submit=true

