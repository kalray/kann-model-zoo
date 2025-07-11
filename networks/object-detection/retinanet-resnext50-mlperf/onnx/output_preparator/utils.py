###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import cv2
import numpy
import torch

from . import palette
from torchvision.ops import nms

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


def generate_anchors(stride, ratio_vals, scales_vals, angles_vals=None):
    """ Generate anchors coordinates from scales/ratios """
    # https://github.com/NVIDIA/retinanet-examples/blob/main/odtk/box.py

    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
    dwh = torch.stack([ws, ws * ratios], dim=1)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales)
    return torch.cat([xy1, xy2], dim=1)


def delta2box(deltas, anchors, size, stride):
    """ Convert deltas from anchors to boxes """
    # https://github.com/NVIDIA/retinanet-examples/blob/main/odtk/box.py

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = (torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1)
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat([
        clamp(pred_ctr - 0.5 * pred_wh),
        clamp(pred_ctr + 0.5 * pred_wh - 1)
    ], 1)


def decode(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=100, anchors=None, rotated=False):
    """ Box Decoding and Filtering """
    # https://github.com/NVIDIA/retinanet-examples/blob/main/odtk/box.py

    if rotated:
        anchors = anchors[0]
    num_boxes = 4 if not rotated else 6

    device = "cpu"
    all_cls_head = torch.Tensor(all_cls_head)
    all_box_head = torch.Tensor(all_box_head)

    anchors = anchors.to(device).type(all_cls_head.dtype)
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    height, width = all_cls_head.size()[-2:]

    batch_size = all_cls_head.size()[0]
    out_scores = torch.zeros((batch_size, top_n), device=device)
    out_boxes = torch.zeros((batch_size, top_n, num_boxes), device=device)
    out_classes = torch.zeros((batch_size, top_n), device=device)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
        box_head = all_box_head[batch, :, :, :].contiguous().view(-1, num_boxes)

        # Keep scores over threshold
        keep = (cls_head >= threshold).nonzero().view(-1)
        if keep.nelement() == 0:
            continue

        # Gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        classes = (indices / width / height) % num_classes
        classes = classes.type(all_cls_head.type())

        # Infer kept bboxes
        x = indices % width
        y = (indices / width) % height
        a = indices / num_classes / height / width
        box_head = box_head.view(num_anchors, num_boxes, height, width)
        boxes = box_head[a.to(int), :, y.to(int), x.to(int)]

        if anchors is not None:
            grid = torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride + anchors[a.to(int), :]
            boxes = delta2box(boxes, grid, [width, height], stride)

        out_scores[batch, :scores.size()[0]] = scores
        out_boxes[batch, :boxes.size()[0], :] = boxes
        out_classes[batch, :classes.size()[0]] = classes

    return out_scores, out_boxes, out_classes


def _nms(detections, conf_thrs, iou_thrs):

    scores = torch.Tensor(detections[0])
    boxes = torch.Tensor(detections[1])
    labels = torch.Tensor(detections[2])

    idx = nms(boxes, scores, iou_threshold=iou_thrs)
    bboxes = numpy.array([list((b[0], b[1], b[2], b[3])) for i, (b, s) in enumerate(zip(boxes, scores)) if i in idx and s >= conf_thrs])
    llabels = numpy.array([int(l) for i, (l, s) in enumerate(zip(labels, scores)) if i in idx and s >= conf_thrs])
    sscores = numpy.array([s for i, s in enumerate(scores) if i in idx and s >= conf_thrs])
    assert len(bboxes) == len(sscores) == len(llabels)
    return bboxes, sscores, llabels


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
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def filter_bboxes(shape, cls_heads, box_heads, conf_thres, iou_thres, max_det=100):
    # Inference post-processing
    anchors = {}
    decoded = []
    for cls_head, box_head in zip(cls_heads, box_heads):
        # Generate level's anchors
        stride = shape[1] // cls_head.shape[-1]
        if stride not in anchors:
            anchors[stride] = generate_anchors(
                stride, ratio_vals=[1.0, 2.0, 0.5], scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
        # Decode and filter boxes
        det = decode(cls_head, box_head, stride, threshold=conf_thres, top_n=max_det, anchors=anchors[stride])
        decoded.append(det)
    # Perform non-maximum suppression
    decoded = [numpy.concatenate(tensors, 1)[0] for tensors in zip(*decoded)]
    out_boxes, out_scores, out_classes = _nms(decoded, conf_thrs=conf_thres, iou_thrs=iou_thres)
    predictions = [[*x, y, z] for x, y, z, in zip(out_boxes, out_scores, out_classes)]
    return predictions


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

     # sort keys like "o1", "o2", ..., "o10"
    nn_outputs = dict(sorted(preds.items(), key=lambda x: int(x[0].replace('output', ''))))
    output_tensors = list(nn_outputs.values())
    cls_head = output_tensors[:5]
    box_head = output_tensors[5:]

    # outs = detection_postprocess(cls_head, box_head)

    # Apply non max suppression algorithm
    outs = filter_bboxes(
        frame.shape, cls_head, box_head,
        conf_thres,
        iou_thres,
        max_det=100,
    )

    # Process detections
    detects = []
    for *coord, conf, cls in outs:  # detections per image
        ratio_wh = frame.shape[0] / frame.shape[1]
        xyxy = [
            coord[0], coord[1] * ratio_wh,
            coord[2], coord[3] * ratio_wh
        ]
        # Write results
        cls_name = classes[int(cls)]
        label = '%s %.2f' % (cls_name, conf)
        color = palette[cls_name]
        detects.append((xyxy, conf, cls_name))
        plot_box(xyxy, frame, label=label, color=color, line_thickness=2)
        if dbg:
            print(f"{head}  >> [Post-proc] prediction: {float(conf)} - {cls_name} - {xyxy}{reset}")

    # return annotated frame and detection object results
    return frame, detects