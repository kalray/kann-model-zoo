import os
import cv2
import time
import numpy
import torch
import onnxruntime as rt
from .util import filter_out_boxes
from .util import scale_coords
from .util import plot_box
from .util import process_mask
from .util import decode_bboxes
from .util import make_anchors
from .util import DFL
from .cmap import palette


def decode_segmask(mask, cmap, nb_classes):
    r = numpy.zeros_like(mask).astype(numpy.uint8)
    g = numpy.zeros_like(mask).astype(numpy.uint8)
    b = numpy.zeros_like(mask).astype(numpy.uint8)
    for l in range(0, nb_classes):
        idx = mask == l
        r[idx] = cmap[l, 0]
        g[idx] = cmap[l, 1]
        b[idx] = cmap[l, 2]
    rgb = numpy.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image_uint8, colored_mask):
    overlay = cv2.addWeighted(image_uint8, 1., colored_mask, 0.85, 0)
    return overlay


def get_classes_from_list(classes_list):
    classes = dict()
    classes[0] = 'background'
    for lines in classes_list:
        idx, obj_class = lines.replace('\n', '').split(' ')[:2]
        classes[int(idx)] = str(obj_class)
    return classes


def get_contours(map):
    """ From Semantic segmentation map, get contour """
    b, g, r = cv2.split(map)
    cb, hb = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cg, hg = cv2.findContours(g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cr, hr = cv2.findContours(r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cb + cg + cr


def get_contour_center(contour):
    """ From Image moments, get barycenter from contours """
    cx, cy = None, None
    m = cv2.moments(contour)
    if int(m['m00']) != 0:
        cx, cy = int(m["m10"] / m['m00']), int(m["m01"] / m['m00'])
    return cx, cy


def draw_labels(image, contours, prediction_colormap, classes):
    """ Draw labels from contours """
    for c in contours:
        x, y = get_contour_center(c)
        if x is not None and y is not None and cv2.contourArea(c) > 300:
            rbg = prediction_colormap[c[0][0, 1], c[0][0, 0]]
            class_id = list(palette.values()).index((rbg.tolist())) - 1
            if class_id != 0:
                text_size, _ = cv2.getTextSize(
                    classes[class_id], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_w, text_h = text_size
                cv2.rectangle(image,
                              (x - 5, y + 5),
                              (x + text_w + 5, y - text_h - 5),
                              (200, 200, 200), -1)
                cv2.putText(image, classes[class_id], (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (100, 100, 100), 1)


def draw_contours(image, prediction_cmp):
    """ Draw contours with predictions """
    contours = get_contours(prediction_cmp)
    cv2.drawContours(image=image, contours=contours,
                     contourIdx=-1, color=(255, 255, 255),
                     thickness=1, lineType=cv2.LINE_AA)
    return contours


def annotate_overlay(img, masks, classes):
    cmap = numpy.array(list(palette.values()), dtype=numpy.uint8)
    # masks = numpy.argmax(masks, axis=-1)
    # add background to masks
    masks_ = numpy.concatenate([numpy.expand_dims(numpy.zeros((masks.shape[:2])), axis=2), masks], axis=2)
    masks = numpy.argmax(masks_, axis=-1)
    # decode semantic segmentation mask
    prediction_colormap = decode_segmask(
        masks, cmap, len(classes))
    # draw overlay mask on origin image
    img_overlay = get_overlay(img, prediction_colormap)
    # draw labels and contours
    # cs = draw_contours(img_overlay, prediction_colormap)
    # draw_labels(img_overlay, cs, prediction_colormap, classes)
    return img_overlay


def detect(x):

    """Concatenates and returns predicted bounding boxes and class probabilities."""

    det1 = [torch.Tensor(i) for i in x.values() if i.shape[1] > 64]
    det2 = [torch.Tensor(i) for i in x.values() if 80 > i.shape[1] > 32]
    seg = [torch.Tensor(i) for i in x.values() if i.shape[1] <= 32][:-1]

    for i, t in enumerate(det1):
        if t.shape[-1] > det1[0].shape[-1]:
            det1 = [det1.pop(i)] + det1

    det = [torch.cat([det1[i], det2[i]], 1) for i in range(len(det1))]

    nc = 80
    no = det[0].shape[1]
    reg_max = 16
    stride = [16., 8., 32.]

    # Inference path
    shape = det[0].shape  # BCHW
    det_cat = torch.cat([xi.view(shape[0], no, -1) for xi in det], 2)
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(det, stride, 0.5))
    box, cls = det_cat.split((reg_max * 4, nc), 1)
    odfl = distFocalLoss(box)
    dbox = decode_bboxes(odfl, anchors.unsqueeze(0)) * strides

    seg_reshaped = torch.cat([torch.reshape(x, (x.shape[0], x.shape[1], -1)) for x in seg], 2)
    y = torch.cat((dbox, cls.sigmoid(), seg_reshaped), 1)
    return y.numpy()


def post_process(cfg, frame, nn_outputs, device='mppa', **kwargs):
    verbose = kwargs.get('dbg', False)
    # nn_outputs is a dict which contains all cnn outputs as value and their name as key
    global classes, colors
    classes = dict((int(x), str(y)) for x, y in
                       [(c.strip("\n").split(" ")[0], ' '.join(c.strip("\n").split(" ")[1:]))
                        for c in cfg['classes']])
    if verbose:
        t0 = time.perf_counter()
    if device == 'mppa':
        for name, shape in zip(cfg['output_nodes_name'], cfg['output_nodes_shape']):
            nn_outputs[name] = nn_outputs[name].reshape(shape)
            if len(shape) == 4:
                H, B, W, C = range(4)
                nn_outputs[name] = nn_outputs[name].transpose((B, C, H, W))
                nn_outputs[name] = nn_outputs[name].astype(numpy.float32)
    if verbose:
        t1 = time.perf_counter()
        print('Post-processing preCNN elapsed time: %.3fms' % (1e3 * (t1 - t0)))

    # post process with the bottom neural networks
    # --
    feed_inputs = {k: nn_outputs[k] for k in postproc_inputs}
    preds = sess.run(None, feed_inputs)[0]
    # preds1 = detect(nn_outputs)
    # print(preds1 - preds)
    if verbose:
        t2 = time.perf_counter()
        print('Post-processing CNN    elapsed time: %.3fms' % (1e3 * (t2 - t1)))

    # define post process methodology
    # --
    reduce_input = True
    letterbox = True
    annotate_masks_overlay = False
    annotate_masks_fusion = ~annotate_masks_overlay
    show_boxes = True

    # Filter output as bounding boxes
    # --
    conf_thres = 0.25
    iou_thres = 0.5
    p = filter_out_boxes(preds, conf_thres, iou_thres, nc=len(classes))
    if verbose:
        t2 = time.perf_counter()
        print('Post-processing NMS    elapsed time: %.3fms' % (1e3 * (t2 - t1)))

    # Compute masks
    # --
    input_h, _, input_w, _ = cfg['input_nodes_shape'][0]
    proto = nn_outputs[cfg['output_nodes_name'][0]]
    pred = p[0]
    if len(pred) == 0:  # quit if empty
        return frame
    pred = torch.Tensor(pred)
    masks = process_mask(proto[0], pred[:, 6:], pred[:, :4], (input_h, input_w), upsample=True)  # Detect HW
    # prediction_masks = numpy.argmax(masks.numpy(), axis=0)  # dissociate object (i.e. panoptic)
    # prediction_masks = numpy.sum(masks.numpy(), axis=0)   # flat detection (semantic)
    if verbose:
        t3 = time.perf_counter()
        print('Post-processing Comp.Masks elapsed time: %.3fms' % (1e3 * (t3 - t2)))

    prediction_masks = numpy.transpose(masks.numpy(), (1, 2, 0))
    # Reduce input to work on a smaller image (i.e. 496x640=20ms instead 640x960=55ms)
    if reduce_input and letterbox:
        ratio = frame.shape[0] / frame.shape[1]
        size = max(input_h, input_w)
        if ratio < 1.:
            new_frame = cv2.resize(frame, (size, int(ratio*size)), interpolation=cv2.INTER_NEAREST)
            if new_frame.shape[0] != prediction_masks.shape[0]:
                prediction_masks = prediction_masks[
                   (prediction_masks.shape[0] - new_frame.shape[0]) // 2:
                   (new_frame.shape[0] - prediction_masks.shape[0]) // 2, ...]
            if new_frame.shape[1] != prediction_masks.shape[1]:
                prediction_masks = prediction_masks[:,
                   (prediction_masks.shape[1] - new_frame.shape[1]) // 2:
                   (new_frame.shape[1] - prediction_masks.shape[1]) // 2, ...]

        elif ratio > 1.:
            new_frame = cv2.resize(frame, (int(ratio*size), size), interpolation=cv2.INTER_NEAREST)
            if new_frame.shape[0] != prediction_masks.shape[0]:
                prediction_masks = prediction_masks[
                   (prediction_masks.shape[0] - new_frame.shape[0]) // 2:
                   (new_frame.shape[0] - prediction_masks.shape[0]) // 2, ...]
            if new_frame.shape[1] != prediction_masks.shape[1]:
                prediction_masks = prediction_masks[:,
                   (prediction_masks.shape[1] - new_frame.shape[1]) // 2:
                   (new_frame.shape[1] - prediction_masks.shape[1]) // 2, ...]
        else:
            new_frame = frame.copy()
            prediction_masks = cv2.resize(
                prediction_masks, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        if len(prediction_masks.shape) == 2:
            prediction_masks = numpy.expand_dims(prediction_masks, axis=-1)
    elif reduce_input and not letterbox:
        new_frame = cv2.resize(frame, (input_h, input_w), interpolation=cv2.INTER_NEAREST)
    # Or increase the mask size to avoid degrading input image quality
    elif not reduce_input and letterbox:
        max_size = max(frame.shape[1], frame.shape[0])
        new_frame = frame.copy()
        prediction_masks = cv2.resize(
            prediction_masks, (max_size, max_size), interpolation=cv2.INTER_NEAREST)
        if new_frame.shape[0] != prediction_masks.shape[0]:
            prediction_masks = prediction_masks[
                (prediction_masks.shape[0] - new_frame.shape[0]) // 2:
                (new_frame.shape[0] - prediction_masks.shape[0]) // 2, ...]
        if new_frame.shape[1] != prediction_masks.shape[1]:
            prediction_masks = prediction_masks[:,
                (prediction_masks.shape[1] - new_frame.shape[1]) // 2:
                (new_frame.shape[1] - prediction_masks.shape[1]) // 2, ...]
    else:
        new_frame = frame.copy()
        prediction_masks = cv2.resize(
            prediction_masks, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    if verbose:
        t4 = time.perf_counter()
        print('Post-processing Resize.Masks elapsed time: %.3fms' % (1e3 * (t4 - t3)))

    # Annotate masks on frame
    # --
    cmap = [list(palette.values())[int(i) + 1] for i in pred[:, 5]]
    if annotate_masks_overlay:
        new_frame = annotate_overlay(new_frame, prediction_masks, cmap)
    if annotate_masks_fusion:
        new_frame = annotate_fusion(new_frame, prediction_masks, cmap)
    if reduce_input:
        new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    if verbose:
        t5 = time.perf_counter()
        print('Post-processing Annotate.Masks elapsed time: %.3fms' % (1e3 * (t5 - t4)))

    # Annotate boxes on frame
    # --
    if show_boxes:
        pred[:, :4] = scale_coords((input_h, input_w), pred[:, :4], new_frame.shape, letterbox).round()
        for i, det in enumerate(pred.numpy()):
            conf, cls = det[4:6]
            xyxy = [int(i.round()) for i in det[:4]]
            label = '%s %.2f' % (classes[int(cls)], conf)
            color = list(palette.values())[int(cls) + 1]
            plot_box(xyxy, new_frame, label=label, color=color, line_thickness=2)
            if verbose:
                print('detect: %s, %.2f, %s' % (label, conf, xyxy))
        if verbose:
            t6 = time.perf_counter()
            print('Post-processing Annotate.Boxes elapsed time: %.3fms' % (1e3 * (t6 - t5)))
    return new_frame


def annotate_fusion(img, masks, cmap, use_torch=True, alpha=0.5):

    if use_torch:
        img = torch.Tensor(img)
        masks = torch.Tensor(numpy.transpose(masks, (2, 0, 1)))  # shape (n, h,w)
        colors = torch.Tensor(alpha * numpy.array(cmap))  # shape(n, 3)
        colors = colors[:, None, None]  # shape(n,1,1,3)
        prediction_mask = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = prediction_mask * colors  # shape(n,h,w,3)
        inv_alpha_masks = (1 - prediction_mask * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = masks_color.max(dim=0).values  # shape(h,w,3)
        im_gpu = img * inv_alpha_masks[-1] + mcs
        im_mask = im_gpu
        im_mask_np = im_mask.byte().numpy()
    else:  # numpy
        img = img
        masks = numpy.transpose(masks, (2, 0, 1))  # shape (n, h,w)
        colors = alpha * numpy.array(cmap)  # shape(n, 3)
        colors = numpy.expand_dims(colors, axis=(1, 2)) # shape(n,1,1,3)
        prediction_mask = numpy.expand_dims(masks, axis=3)  # shape(n,h,w,1)
        masks_color = prediction_mask * colors  # shape(n,h,w,3)

        inv_alpha_masks = numpy.cumprod((1 - prediction_mask * alpha), axis=0)  # shape(n,h,w,1)
        mcs = masks_color.max(axis=0)  # shape(h,w,3)
        im_gpu = img * inv_alpha_masks[-1] + mcs
        im_mask = im_gpu
        im_mask_np = im_mask.astype(numpy.uint8)
    return im_mask_np


sess = rt.InferenceSession(
    os.path.dirname(os.path.realpath(__file__)) + "/yolov8m-seg.postproc.onnx")
postproc_inputs = []
for postproc_input in sess.get_inputs():
    postproc_inputs.append(postproc_input.name)

classes = None
colors = None
distFocalLoss = DFL()
