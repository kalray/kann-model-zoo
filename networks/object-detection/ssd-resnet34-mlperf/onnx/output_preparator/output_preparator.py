#!/usr/bin/env python3

import os
import cv2
import numpy
import onnxruntime as ort

head = "\x1b[0;30;42m"
reset = "\x1b[0;0m"


def plot_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [numpy.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


def post_process(cfg, frame, nn_outputs, device='mppa', **kwargs):
    global classes, colors

    if classes is None:
        classes = [str(' '.join(c.strip("\n").split(" ")[1:])) for c in cfg['classes']]
        classes.insert(0, "background")
        classes = {k: v for k, v in enumerate(classes)}
        colors = [[numpy.random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    if device == 'mppa':
        for name, shape in zip(nn_outputs.keys(), cfg['output_nodes_shape']):
           nn_outputs[name] = nn_outputs[name].reshape(shape)

    for k, v in nn_outputs.items():
        if k in postproc_inputs:
            postproc_inputs[k] = v.astype(numpy.float32)
    outs = sess.run(None, postproc_inputs)
    preds = dict(zip(postproc_outputs, outs))

    conf_thres= 0.25
    # Rescale boxes from img_size to im0 size
    boxes = numpy.array(preds["bboxes"][0])
    for j in range(int(len(preds["labels"][0]))):
        score_j = numpy.float32(preds["scores"][0, j])
        if score_j < conf_thres:
             continue
        # draw the roi
        class_num = int(preds["labels"][0, j])
        class_j = classes[class_num]
        box_j = boxes[j]
        y_min = float(box_j[0] * frame.shape[1])
        y_max = float(box_j[2] * frame.shape[1])
        x_min = float(box_j[1] * frame.shape[0])
        x_max = float(box_j[3] * frame.shape[0])
        xyxy = [y_min, x_min, y_max, x_max]
        label = "{} {:0.4f}".format(class_j, score_j)
        if kwargs["dbg"]:
            print(f"{head}  >> [Post-proc] prediction: {label} {[round(i, 3) for i in xyxy]}{reset}")
        plot_box(xyxy, frame, label=label, color=colors[int(0)], line_thickness=2)

    return frame


postproc_model = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "ssd-resnet34.postproc.onnx")
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

classes = None
colors = None