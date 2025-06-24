#!/usr/bin/env python3

###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import cv2
import numpy
from . import palette


def get_classes(names:str) -> dict:

    """ From list of label from classes.txt, return a dict of labelId to name """

    result = dict()
    if len(names[0].split(" ")) == 2:
        result = {int(c.split(" ")[0]): c.split(" ")[-1].rstrip("\n") for c in names}
    elif len(names[0].split(" ")) == 1:
        result = {id: name.rstrip("\n") for id, name in enumerate(names)}
    else:
        raise ValueError(f"Format is not as expected <0 label_id> or <label_id> per row-line")
    return result


def plot_box(x:list, img:numpy.ndarray, color:list=None, label:str=None, line_thickness:int=None) -> None:

    """ Plots one bounding box on image img """

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [numpy.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        text_color = [225, 255, 255] if sum(color) / 3 < 127 else [0, 0, 0]
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, text_color,
                    thickness=tf, lineType=cv2.LINE_AA)


def process_detections(
        preds:dict,
        cfg:dict,
        frame:numpy.ndarray,
        classes:dict,
        conf_thres:float=0.2,
        iou_thres=0.4,
        dbg:bool=False
    ) -> numpy.ndarray:

    """ Determinate object detections from YOLO neural networks """

    head = "\x1b[0;30;42m"
    reset = "\x1b[0;0m"

    # For other input of post processing graph
    for k, v in preds.items():
        if k in postproc_inputs:
            postproc_inputs[k] = numpy.transpose(v, (0, 2, 3, 1))
    t = numpy.asarray(frame.shape, dtype=numpy.int32).reshape((1, 3))
    postproc_inputs['Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3:0'] = t

    outs = sess.run(None, postproc_inputs)
    preds = dict(zip(postproc_outputs, outs))

    for j in range(int(preds["num_detections:0"])):
        # draw the roi
        class_j = int(preds["detection_classes:0"][0, j]) - 1
        cls_name = classes[class_j]
        score_j = numpy.float32(preds["detection_scores:0"][0, j])
        if score_j < 0.2:
             continue
        box_j = numpy.array(preds["detection_boxes:0"][0, j, :])
        assert box_j.shape == (4,), "box_j.shape = {}".format(box_j.shape)
        x_min = float(box_j[0] * frame.shape[0])
        x_max = float(box_j[2] * frame.shape[0])
        y_min = float(box_j[1] * frame.shape[1])
        y_max = float(box_j[3] * frame.shape[1])
        xyxy = [y_min, x_min, y_max, x_max]
        label = "{} {:0.4f}".format(cls_name, score_j)
        color = palette[cls_name]
        if dbg:
            print(f"{head}  >> [Post-proc] prediction: {score_j} - {cls_name} - {[round(i, 3) for i in xyxy]}{reset}")
        plot_box(xyxy, frame, label=label, color=color, line_thickness=2)
    return frame


import onnxruntime as ort

postproc_model = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "ssd_mobilenet_v1.postprocessing.onnx")
sess = ort.InferenceSession(postproc_model)
postproc_inputs = dict()
postproc_outputs = dict()
for k in sess.get_inputs():
    dt = numpy.float32
    dt = numpy.int64 if 'int64' in k.type else dt
    dt = numpy.int32 if 'int32' in k.type else dt
    try:
        postproc_inputs[k.name] = numpy.zeros(k.shape, dtype=dt)
    except:
        postproc_inputs[k.name] = None
for k in sess.get_outputs():
    dt = numpy.float32
    dt = numpy.int64 if 'int64' in k.type else dt
    dt = numpy.int32 if 'int32' in k.type else dt
    try:
        postproc_outputs[k.name] = numpy.zeros(k.shape, dtype=dt)
    except:
        postproc_outputs[k.name] = None

postproc_inputs["BoxPredictor_0/stack:0"] = numpy.array([1, 1083, 1, 4], dtype=numpy.int32)
postproc_inputs["BoxPredictor_0/stack_1:0"] = numpy.array([1, 1083, 91], dtype=numpy.int32)
postproc_inputs["BoxPredictor_1/stack:0"] = numpy.array([1, 600, 1, 4], dtype=numpy.int32)
postproc_inputs["BoxPredictor_1/stack_1:0"] = numpy.array([1, 600, 91], dtype=numpy.int32)
postproc_inputs["BoxPredictor_2/stack:0"] = numpy.array([1, 150, 1, 4], dtype=numpy.int32)
postproc_inputs["BoxPredictor_2/stack_1:0"] = numpy.array([1, 150, 91], dtype=numpy.int32)
postproc_inputs["BoxPredictor_3/stack:0"] = numpy.array([1, 54, 1, 4], dtype=numpy.int32)
postproc_inputs["BoxPredictor_3/stack_1:0"] = numpy.array([1, 54, 91], dtype=numpy.int32)
postproc_inputs["BoxPredictor_4/stack:0"] = numpy.array([1, 24, 1, 4], dtype=numpy.int32)
postproc_inputs["BoxPredictor_4/stack_1:0"] = numpy.array([1, 24, 91], dtype=numpy.int32)
postproc_inputs["BoxPredictor_5/stack:0"] = numpy.array([1, 6, 1, 4], dtype=numpy.int32)
postproc_inputs["BoxPredictor_5/stack_1:0"] = numpy.array([1, 6, 91], dtype=numpy.int32)
