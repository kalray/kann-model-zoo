import os
import time
import numpy
import onnxruntime as rt
from .utils import get_classes, get_colors, process_detections


def post_process(cfg, frame, nn_outputs, device='mppa', dbg=True, **kwargs):
    # nn_outputs is a dict which contains all cnn outputs as value and their name as key
    global classes, colors
    if classes is None:
        classes = get_classes(cfg["classes"])
        colors = get_colors(classes)
    t0 = time.perf_counter()
    if 'mppa' in device:
        for name, shape, df in zip(cfg['output_nodes_name'], cfg['output_nodes_shape'], cfg['output_nodes_dformat']):
            nn_outputs[name] = nn_outputs[name].reshape(shape)
            nn_outputs[name] = nn_outputs[name].transpose(df)
            nn_outputs[name] = nn_outputs[name].astype(numpy.float32)
    out = sess.run(None, nn_outputs)
    t1 = time.perf_counter()
    process_detections(out[0], cfg, frame, classes, colors, dbg=dbg)
    t2 = time.perf_counter()
    if dbg:
        print('Post-processing post-CNN elapsed time: %.3fms' % (1e3 * (t1 - t0)))
        print('Post-processing PLOT     elapsed time: %.3fms' % (1e3 * (t2 - t1)))
        print('Post-processing TOTAL    elapsed time: %.3fms' % (1e3 * (t2 - t0)))
    return frame

onnx_file_path = os.path.dirname(os.path.realpath(__file__)) 
onnx_file_path = os.path.join(onnx_file_path, "efficientdet-d0.postproc.onnx")
sess = rt.InferenceSession(onnx_file_path)
classes = None
colors = None
