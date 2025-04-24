###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import cv2
import torch
import numpy
from .cmap import palette


class distFocalLoss(torch.nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = torch.nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = torch.nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def detect(t):
    """
    Concatenates and returns predicted bounding boxes and class probabilities.
    """

    def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype = feats[0].dtype
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    t = [torch.Tensor(i) for i in t]
    no = t[0].shape[1]
    nc = no - dfl.c1 * 4
    reg_max = dfl.c1

    # Ascending sort t 
    x = []
    for p in t:
        if p.shape[-1] >= t[0].shape[-1]:
            x.append(p)
        else:
            x.insert(0, p)
    stride = [32., 16., 8.] # 80x80, 40x40, 20x20
    # Inference path
    shape = x[0].shape  # BCHW
    x_cat = torch.cat([xi.view(shape[0], no, -1) for xi in x], 2)
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(x, stride, 0.5))
    box, cls = x_cat.split((reg_max * 4, nc), 1)
    odfl = dfl(box)
    dbox = dist2bbox(odfl, anchors.unsqueeze(0), dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y.numpy()


def get_classes(names):
    """
    From list of label from classes.txt, return a dict of labelId to name
    """
    result = dict()
    if len(names[0].split(" ")) == 2:
        result = {int(c.split(" ")[0]): c.split(" ")[-1].rstrip("\n") for c in names}
    elif len(names[0].split(" ")) == 1:
        result = {id: name.rstrip("\n") for id, name in enumerate(names)}
    else:
        raise ValueError(f"Format is not as expected <0 label_id> or <label_id> per row-line")
    return result


def plot_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [numpy.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)


def scale_coords(coords, nn_shape, img0_shape, ratio_pad=None):
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


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=100,
        max_nms=1200,
        max_wh=7680,
    ):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:


    Returns:
        (List[numpy.array]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    def xywh2xyxy(x):
        y = numpy.zeros_like(x)
        xy = x[..., :2]  # centers
        wh = x[..., 2:] / 2  # half width-height
        y[..., :2] = xy - wh  # top left xy
        y[..., 2:] = xy + wh  # bottom right xy
        return y

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - 4  # number of classes
    nm = prediction.shape[1] - nc - 4 # number of masks
    mi = 4 + nc  # mask start index
    xc = numpy.abs(prediction[:, 4:mi]).max(1) > conf_thres  # candidates
    xinds = numpy.array((numpy.arange(xc.shape[-1]),))[..., None]  # to track idxs

    prediction = numpy.transpose(prediction, (0, 2, 1))
    pred_xywh2xyxy = xywh2xyxy(prediction[..., :4])
    prediction = numpy.concatenate((pred_xywh2xyxy, prediction[..., 4:]), axis=-1)  # xywh to xyxy
    output = [numpy.zeros((0, 6 + nm))] * bs
    keepi = [numpy.zeros((0, 1))] * bs  # to store the kept idxs

    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)

        # Apply constraints
        filt = xc[xi]  # get confidence
        x, xk = x[filt], xk[filt]
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Detections matrix (xyxy, conf, cls)
        box = x[:, :4]
        cls = x[:, 4:4+nc]
        mask = x[:, 4+nc:4+nc+nm]
        i, j = numpy.where(cls > conf_thres)
        x = numpy.concatenate(
            (box[i],
             x[i, 4 + j,
             None],
             j[:, None].astype(numpy.float32),
             mask[i]), axis=1)
        xk = xk[i]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > prediction.shape[1]:  # excess boxes
            filt = x[:, 4].argsort()[::-1][:max_nms]  # sort by confidence and remove excess boxes
            x, xk = x[filt], xk[filt]
        # Batched NMS
        c = x[:, 5:6] * max_wh if prediction.shape[0] > 1 else 0
        scores = x[:, 4]  # scores
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi], keepi[xi] = x[i], xk[i].reshape(-1)

    return output


def nms(boxes, scores, iou_thrs):
    """ NMS Python implementation of https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/nms.cpp """

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    supp = numpy.zeros(len(boxes), dtype=numpy.uint8)
    keep = []
    for _i in range(len(order)):
        i = order[_i]
        if supp[i]:
            continue
        keep.append(i)
        ix1, iy1, ix2, iy2 = x1[i], y1[i], x2[i], y2[i]
        iarea = area[i]
        for _j in range(_i + 1, len(order)):
            j = order[_j]
            if supp[j]:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            union = iarea + area[j] - inter
            iou = inter / union
            if iou > iou_thrs:
                supp[j] = 1
    keep = numpy.array(keep, dtype=numpy.int64)
    return keep


dfl = distFocalLoss()

def process_detections(preds, cfg, frame, classes, conf_thres=0.25, iou_thres=0.4, dbg=False):
    """ Determinate object detections from YOLO neural networks """

    head = "\x1b[0;30;42m"
    reset = "\x1b[0;0m"

    # Process detections
    if isinstance(preds, dict):
        preds = list(preds.values())
    prediction = detect(preds)

    # Apply non max suppression algorithm
    out = non_max_suppression(
        prediction,
        conf_thres,
        iou_thres,
        max_det=100,
    )

    # Rescale and plot detections
    for i, det in enumerate(out):  # detections per image
        # Rescale boxes from img_size to im0 size
        net_h = cfg['input_nodes_shape'][0][cfg['input_nodes_dformat'][0][2]]
        net_w = cfg['input_nodes_shape'][0][cfg['input_nodes_dformat'][0][3]]
        # det format is [x1, y1, x2, y2, conf, class_id]
        det[:, :4] = scale_coords(det[:, :4], (net_h, net_w), frame.shape)
        # Write results
        for *xyxy, conf, cls in det:
            label = '%s %.2f' % (classes[int(cls)], conf)
            color = palette[classes[int(cls)]]
            plot_box(xyxy, frame, label=label, color=color, line_thickness=2)
            print(f"{head}  >> [Post-proc] prediction: {conf:.4f} - "
                  f"{classes[int(cls)]} - {xyxy}{reset}") if dbg else None

    # return annotated frame
    return frame
