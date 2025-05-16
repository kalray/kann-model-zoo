#!/usr/bin/env bash

###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

SCRIPT_DIR=$(dirname $(realpath $0))
MODEL_ONNX=$1  # ONNX model
CALIB_IMGS=$2  # Calibration image DIR

python3 $SCRIPT_DIR/run_quantization.py  $MODEL_ONNX \
    --calibration_dataset $CALIB_IMGS
