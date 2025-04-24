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


cls_dict = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic_light',
    10: 'fire_hydrant',
    11: 'stop_sign',
    12: 'parking_meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
}

colors_dict = {
    'person':       [0,   0,   255],
    'bicycle':      [0,   255, 0  ],
    'car':          [255, 0,   0  ],
    'motorcycle':   [128, 128, 0  ],
    'airplane':     [0,   128, 255],
    'bus':          [128, 255, 0  ],
    'train':        [0,   255, 128],
    'truck':        [255, 128, 0  ],
    'boat':         [128, 0,   255],
    'traffic_light':[255, 0,   128],
    'fire_hydrant': [0,   0,   128],
    'stop_sign':    [0,   128, 0  ],
    'parking_meter':[128, 0,   0  ],
    'bench':        [0,   69,  255],
    'bird':         [69,  255, 0  ],
    'cat':          [255, 0,   69 ],
    'dog':          [0,   0,   69 ],
}


def post_process(cfg, frame, nn_outputs, device='mppa', dbg=True, **kwargs):
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
