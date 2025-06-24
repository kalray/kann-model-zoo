#!/usr/bin/env python3

###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import time
import numpy

from .utils import process_detections
from .utils import get_classes

classes = None

def post_process(cfg, frame, nn_outputs, device='mppa', dbg=False, **kwargs):
    # nn_outputs is a dict which contains all cnn outputs as value and their name as key
    global classes
    if classes is None:
        classes = get_classes(cfg["classes"])
    t0 = time.perf_counter()
    if 'mppa' in device:
        for name, shape, df in zip(cfg['output_nodes_name'], cfg['output_nodes_shape'], cfg['output_nodes_dformat']):
            nn_outputs[name] = nn_outputs[name].reshape(shape)
            if len(shape) > 3 and df != [0, 1, 2, 3]:
                nn_outputs[name] = nn_outputs[name].transpose(df)
            nn_outputs[name] = nn_outputs[name].astype(numpy.float32)
    t1 = time.perf_counter()
    process_detections(nn_outputs, cfg, frame, classes, dbg=dbg, **kwargs)
    t2 = time.perf_counter()
    if dbg:
        print('Post-processing Reshape  elapsed time: %.3fms' % (1e3 * (t1 - t0)))
        print('Post-processing NMS      elapsed time: %.3fms' % (1e3 * (t2 - t1)))
        print('Post-processing TOTAL    elapsed time: %.3fms' % (1e3 * (t2 - t0)))
    return frame


def post_process_eval(cfg, frame, nn_outputs, device='mppa', dbg=False, **kwargs):
    # nn_outputs is a dict which contains all cnn outputs as value and their name as key
    global classes
    if classes is None:
        classes = get_classes(cfg["classes"])
    if 'mppa' in device:
        for name, shape, df in zip(cfg['output_nodes_name'], cfg['output_nodes_shape'], cfg['output_nodes_dformat']):
            nn_outputs[name] = nn_outputs[name].reshape(shape)
            if len(shape) > 3 and df != [0, 1, 2, 3]:
                nn_outputs[name] = nn_outputs[name].transpose(df)
            nn_outputs[name] = nn_outputs[name].astype(numpy.float32)
    frame, detections = process_detections(nn_outputs, cfg, frame, classes, dbg=dbg, **kwargs)
    return frame, detections

classes = None
