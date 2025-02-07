import cv2
import time
import numpy

from .decode_util import generate_anchors, decode, torch_nms


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


def detection_postprocess(image, cls_heads, box_heads):
    # Inference post-processing
    anchors = {}
    decoded = []
    for cls_head, box_head in zip(cls_heads, box_heads):
        # Generate level's anchors
        stride = image.shape[1] // cls_head.shape[-1]
        if stride not in anchors:
            anchors[stride] = generate_anchors(
                stride, ratio_vals=[1.0, 2.0, 0.5], scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
        # Decode and filter boxes
        decoded.append(decode(
            cls_head, box_head, stride, threshold=0.2, top_n=100, anchors=anchors[stride]))
    # Perform non-maximum suppression
    decoded = [numpy.concatenate(tensors, 1)[0] for tensors in zip(*decoded)]
    out_boxes, out_scores, out_classes = torch_nms(decoded, iou_thrs=0.2)
    predictions = [[*x, y, z] for x, y, z, in zip(out_boxes, out_scores, out_classes)]
    return predictions


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = (img1_shape[1] / img0_shape[0], img1_shape[0] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain[0]) / 2, (img1_shape[0] - img0_shape[0] * gain[1]) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, [0, 2]] /= gain[0]
    coords[:, [1, 3]] /= gain[1]
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = numpy.clip(boxes[:, 0], 0, img_shape[1])
    boxes[:, 1] = numpy.clip(boxes[:, 1], 0, img_shape[0])
    boxes[:, 2] = numpy.clip(boxes[:, 2], 0, img_shape[1])
    boxes[:, 3] = numpy.clip(boxes[:, 3], 0, img_shape[0])


def post_process(cfg, frame, nn_outputs, device='mppa', dbg=True, **kwargs):
    # nn_outputs is a dict which contains all cnn outputs as value and their name as key
    global classes, colors
    if classes is None:
        classes = dict((int(x), str(y)) for x, y in
                       [(c.strip("\n").split(" ")[0], ' '.join(c.strip("\n").split(" ")[1:]))
                        for c in cfg['classes']])
        colors = [[numpy.random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    if dbg:
        t0 = time.perf_counter()
    if device == 'mppa':
        outs = dict()
        for name, shape in zip(cfg['output_nodes_name'], cfg['output_nodes_shape']):
            outs[name] = nn_outputs[name].reshape(shape)
            if len(shape) == 4:
                H, B, W, C = range(4)
                outs[name] = outs[name].transpose((B, C, H, W))
                outs[name] = outs[name].astype(numpy.float32)
        nn_outputs = outs
    if dbg:
        t1 = time.perf_counter()
        print('Post-processing preCNN elapsed time: %.3fms' % (1e3 * (t1 - t0)))

    output_tensors = list(nn_outputs.values())
    cls_head = output_tensors[:5]
    box_head = output_tensors[5:]

    # TODO: investigate PostProcessing from source repository
    outs = detection_postprocess(frame, cls_head, box_head)
    if dbg:
        t2 = time.perf_counter()
        print('Post-processing NMS    elapsed time: %.3fms' % (1e3 * (t2 - t1)))
    # Process detections
    for *coord, conf, cls in outs:  # detections per image
        input_h, _, input_w, _ = cfg['input_nodes_shape'][0]
        coord = numpy.expand_dims(numpy.array(coord), 0).round()
        ratio_wh = frame.shape[0] / frame.shape[1]
        # coord = scale_coords((input_h, input_w), coord, frame.shape).round()
        xyxy = [
            int(coord[0][0]),
            int(coord[0][1] * ratio_wh),
            int(coord[0][2]),
            int(coord[0][3] * ratio_wh)
        ]
        # Write results
        label = '%s %.1f%%' % (classes[int(cls)], 100 * float(conf))
        color = colors[int(cls)]
        plot_box(xyxy, frame, label=label, color=color, line_thickness=2)
        if dbg:
            print('detect: %s, %s' % (label, xyxy))
        plot_box(
            [10, 60, 100, 50],
            frame, label="Post processing under development",
            color=[0, 0, 255],
            line_thickness=2
        )
    if dbg:
        t3 = time.perf_counter()
        print('Post-processing PLOT   elapsed time: %.3fms' % (1e3 * (t3 - t2)))
        print('Post-processing TOTAL  elapsed time: %.3fms' % (1e3 * (t3 - t0)))
    return frame


classes = None
colors = None