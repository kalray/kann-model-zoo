###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###
import os

WORKSPACE_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__ + "/../")))
)

COCO_IDX2NAME = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic_light',
    10: 'fire_hydrant', 11: 'stop_sign', 12: 'parking_meter',
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
    18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
    23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
    28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports_ball', 33: 'kite', 34: 'baseball_bat', 35: 'baseball_glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis_racket', 39: 'bottle',
    40: 'wine_glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot_dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted_plant', 59: 'bed', 60: 'dining_table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell_phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy_bear', 78: 'hair_drier', 79: 'toothbrush'
}

COCO_NAME2IDX = {
    'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4,
    'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic_light': 9,
    'fire_hydrant': 10, 'stop_sign': 11, 'parking_meter': 12, 'bench': 13,
    'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19,
    'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24,
    'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29,
    'skis': 30, 'snowboard': 31, 'sports_ball': 32, 'kite': 33, 'baseball_bat': 34,
    'baseball_glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis_racket': 38,
    'bottle': 39, 'wine_glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44,
    'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50,
    'carrot': 51, 'hot_dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56,
    'couch': 57, 'potted_plant': 58, 'bed': 59, 'dining_table': 60, 'toilet': 61,
    'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell_phone': 67,
    'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72,
    'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy_bear': 77,
    'hair_drier': 78,'toothbrush': 79
}