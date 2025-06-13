#!/usr/bin/env python3

###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

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


def scale_coords(coords:numpy.ndarray, nn_shape:list, img0_shape:list, ratio_pad:float=None) -> numpy.ndarray:

    """ Rescale coordinates computed from NN (img) to original image shape (img0) """

    def clip_coords(boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0] = numpy.clip(boxes[:, 0], 0, img_shape[1])
        boxes[:, 1] = numpy.clip(boxes[:, 1], 0, img_shape[0])
        boxes[:, 2] = numpy.clip(boxes[:, 2], 0, img_shape[1])
        boxes[:, 3] = numpy.clip(boxes[:, 3], 0, img_shape[0])

    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        h_gain = nn_shape[0] / img0_shape[0]
        w_gain = nn_shape[1] / img0_shape[1]
        gain = (w_gain, h_gain)
        w_pad = (nn_shape[1] - img0_shape[1] * min(gain)) / 2
        h_pad = (nn_shape[0] - img0_shape[0] * min(gain)) / 2
        pad = (w_pad, h_pad)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]   # x padding
    coords[:, [1, 3]] -= pad[1]   # y padding
    coords[:, :4] /= min(gain)
    clip_coords(coords, img0_shape)
    return coords


def filter_bboxes(
        predictions:numpy.ndarray,
        conf_thres:float=0.25,
        iou_thres:float=0.4,
        max_det:int=300,
        max_wh:int=7680,
    ) -> list:

    """
    Process filtering of a set of boxes using non-maximum suppression (NMS) with multiple labels per box.

    Args:
        predictions (dict): predictions computed by neural networks with shape (BATCH, N, COORDS+SCORE+NC), where
                            - N: SUM((GRID_SIZE_X * GRID_SIZE_Y * NB_CHANNELS)),
                                i.e. YOLOv3(416^2) : N = 13*13*3+ 26*26*3 + 52*52*3 = 768 + 2028 + 8112 = 10647
                            - COORDS: XYWH (4) + SCORE (1) + NB_CLS (80 for COCO2027) = 85
        conf_thres (float): confidence threshold, all confidence less than value would not be count
        iou_thres (float):  intersection over union threshold, all IoU less than value would not be count
        max_det (int):      maximum bounding box detection, the first <max_det> would be returned
        max_wh (int):       scale applied to predictions for batched NMS 

    Returns:
        (List[numpy.array]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    def xywh2xyxy(x:numpy.ndarray) -> numpy.ndarray:
        y = numpy.zeros_like(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    # predictions shape is (1, N, 85)
    batch = predictions.shape[0]  # batch size
    candidates = numpy.abs(predictions[..., 4:]).max(2) > conf_thres  # candidates
    outs = [numpy.zeros((0, 6))] * batch
    for pi, preds in enumerate(predictions):  # batch index, preds
        # Apply constraints
        filt = candidates[pi]  # get confidence
        preds = preds[filt]
        # Compute conf
        preds[:, 5:] *= preds[:, 4:5]  # conf = obj_conf * cls_conf
        # Detections matrix (xyxy, conf, cls)
        box_xywh = preds[:, :4]
        cls = preds[:, 5:]
        i, j = numpy.where(cls >= conf_thres)
        box_xyxy = xywh2xyxy(box_xywh[i])
        preds = numpy.concatenate((box_xyxy, preds[i, j + 5, None], j[:, None]), axis=1)
        # Batched NMS
        c = preds[:, 5:6] * max_wh if batch > 1 else 0
        # NMS
        scores = preds[:, 4]  # scores
        boxes = preds[:, :4] + c  # boxes (offset by class if batch > 1)
        i = nms(boxes, scores, iou_thres)  # NMS (coords must be xyxy)
        i = i[:max_det]  # limit detections
        outs[pi] = preds[i]
    return outs


def nms(boxes:numpy.ndarray, scores:numpy.ndarray, iou_thrs:float) -> numpy.ndarray:

    """ NMS Python implementation from https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/nms.cpp """

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    supp = numpy.zeros(len(boxes), dtype=numpy.int64)
    keep = list()
    for ki, i in enumerate(order):
        if supp[i]:
            continue
        keep.append(i)
        ix1, iy1, ix2, iy2 = x1[i], y1[i], x2[i], y2[i]
        iarea = area[i]
        for j in order[ki+1:]:
            if ~supp[j]:
                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                width = max(0.0, xx2 - xx1)
                height = max(0.0, yy2 - yy1)
                inter = width * height
                union = iarea + area[j] - inter
                iou = inter / union
                if iou >= iou_thrs:
                    supp[j] = True
    keep = numpy.array(keep, dtype=numpy.int64)
    return keep


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

    # Process detections
    predictions = preds[cfg['output_nodes_name'][0]]

    # Apply non max suppression algorithm on predictions
    out = filter_bboxes(
        predictions, # shape is (batch, xywh + nc, idx), e.g. (1, 85, N)
        conf_thres,
        iou_thres,
        max_det=100,
    )

    # Rescale and plot detections
    detect = []
    for i, det in enumerate(out):  # detections per image
        # Rescale boxes from img_size to im0 size
        net_h = cfg['input_nodes_shape'][0][cfg['input_nodes_dformat'][0][2]]
        net_w = cfg['input_nodes_shape'][0][cfg['input_nodes_dformat'][0][3]]
        # det format is [x1, y1, x2, y2, conf, class_id]
        det[:, :4] = scale_coords(det[:, :4], (net_h, net_w), frame.shape)
        # Write results
        for *xyxy, conf, cls in det:
            cls_name = classes[int(cls)]
            label = f'{cls_name} {conf:.2f}'
            color = palette[cls_name]
            detect.append((xyxy, conf, cls_name))
            plot_box(xyxy, frame, label=label, color=color, line_thickness=2)
            print(f"{head}  >> [Post-proc] prediction: {conf:.4f} - "
                  f"{cls_name} - {[round(x, 5) for x in xyxy]}{reset}") if dbg else None

    # return annotated frame
    return frame, detect
