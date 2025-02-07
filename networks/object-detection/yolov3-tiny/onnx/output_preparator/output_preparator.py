import os
import time
import numpy
import onnxruntime as ort

from utils import scale_coords, cls_dict, colors_dict, plot_box, filter_out_boxes


def process_detections(output_box_filtered, cfg, frame, det_classes, det_colors, dbg=False):

    # Process detections
    for i, det in enumerate(output_box_filtered):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            input_h, _, input_w, _ = cfg['input_nodes_shape'][0]
            det[:, :4] = scale_coords((input_h, input_w), det[:, :4], frame.shape).round()
            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (det_classes[int(cls)], conf)
                if int(cls) >= len(cls_dict):
                    color = det_colors[int(cls)]
                else:
                    color = colors_dict[cls_dict[int(cls)]]
                plot_box(xyxy, frame, label=label, color=color, line_thickness=2)
                if dbg:
                    print('    > detect: %s, %.2f %s' % (det_classes[int(cls)], conf, xyxy))
    return frame


def post_process(cfg, frame, nn_outputs, conf_thres=0.35, iou_thres=0.4, device='mppa', dbg=True, **kwargs):
    # nn_outputs is a dict which contains all cnn outputs as value and their name as key
    global classes, colors
    if classes is None:
        classes = dict((int(x), str(y)) for x, y in
                       [(c.strip("\n").split(" ")[0], ' '.join(c.strip("\n").split(" ")[1:]))
                        for c in cfg['classes']])
        colors = [[numpy.random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    t0 = time.perf_counter()
    if device == 'mppa':
        for name, shape in zip(cfg['output_nodes_name'], cfg['output_nodes_shape']):
            nn_outputs[name] = nn_outputs[name].reshape(shape)
            if len(shape) == 4:
                H, B, W, C = range(4)
                nn_outputs[name] = nn_outputs[name].transpose((B, C, H, W))
                nn_outputs[name] = nn_outputs[name].astype(numpy.float32)
    if dbg:
        t1 = time.perf_counter()
        print('Post-processing preCNN elapsed time: %.3fms' % (1e3 * (t1 - t0)))
    preds = sess.run(None, nn_outputs)
    if dbg:
        t2 = time.perf_counter()
        print('Post-processing CNN    elapsed time: %.3fms' % (1e3 * (t2 - t1)))
    out = filter_out_boxes(preds[0], conf_thres, iou_thres)
    if dbg:
        t3 = time.perf_counter()
        print('Post-processing NMS    elapsed time: %.3fms' % (1e3 * (t3 - t2)))
    process_detections(out, cfg, frame, classes, colors, dbg=dbg)
    if dbg:
        t4 = time.perf_counter()
        print('Post-processing PLOT   elapsed time: %.3fms' % (1e3 * (t4 - t3)))
        print('Post-processing TOTAL  elapsed time: %.3fms' % (1e3 * (t4 - t0)))
    return frame


model_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "yolov3-tiny.postprocessing.onnx"
)
sess = ort.InferenceSession(model_path)
classes = None
colors = None
