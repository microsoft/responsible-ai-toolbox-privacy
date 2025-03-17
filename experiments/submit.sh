#/bin/bash

EXPERIMENT_DIR="$(dirname "$(readlink -f "$0")")"

set -e

python $EXPERIMENT_DIR/estimate_differential_privacy_image_classifier.py --config-name dpd_image_classifier +submit=true

python $EXPERIMENT_DIR/estimate_black_box_privacy_image_classifier.py --config-name black_box_image_classifier +submit=true
python $EXPERIMENT_DIR/estimate_black_box_privacy_text_classifier.py --config-name black_box_text_classifier +submit=true

