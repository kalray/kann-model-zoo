#!/usr/bin/env python3
import sys
import numpy
import cv2
import os
import itertools as it
import math
from collections import OrderedDict
from PIL import Image


def prepare_img(mat, out_dtype=numpy.float32):
    # https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov4
    def image_preprocess(image, target_size, gt_boxes=None):
        ih, iw = target_size
        h, w, _ = image.shape
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        image_padded = numpy.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_padded = image_padded / 255.
        if gt_boxes is None:
            return image_padded
        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_padded, gt_boxes

    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    original_image_size = mat.shape[:2]
    image_data = image_preprocess(numpy.copy(mat), [416, 416])
<<<<<<< HEAD
    image_data = image_data[numpy.newaxis, ...].astype(numpy.float32)
    return image_data
=======
    return image_data.astype(out_dtype)
>>>>>>> 474b47e ([NETWORKS] YOLOv4-tiny: replace Transpose+Split+Transpose by Slice onnx node)


def image_stream(filename):
    """ Read and prepare the sequence of images of <filename>.
        If <filename> is an int, use it as a webcam ID.
        Otherwise <filename> should be the name of an image, video
        file, or image sequence of the form name%02d.jpg
    """
    try:
        src = int(filename)
    except ValueError:
        src = filename
    stream = cv2.VideoCapture(src)
    if not stream.isOpened():
        raise ValueError('could not open stream {!r}'.format(src))
    while True:
        ok, frame = stream.read()
        if not ok:
            break
        yield prepare_img(frame)


def batches_extraction(stream):
    """ extract batches of images from a python generator of prepared images """
    batch = 1
    while True:
        imgs = list(it.islice(stream, batch))
        if imgs == []:
            break
        while len(imgs) != batch:  # last batch might not be full
            imgs.append(numpy.zeros(imgs[0].shape, dtype=imgs[0].dtype))
        # interleave the batch as required by kann (HBWC axes order)
        # note: could use np.stack(axis=1) here, but it's not available in np 1.7.0
        for i in range(len(imgs)):
            imgs[i] = numpy.reshape(imgs[i], imgs[i].shape[:1] + (1,) + imgs[i].shape[1:])
        imgs = numpy.concatenate(imgs, axis=1)
        yield imgs


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: ' + sys.argv[0] + ' <destination_file> <sources>...')
        print('called with {} args: {}'.format(len(sys.argv), ', '.join(sys.argv)))
        exit(1)
    stream = it.chain(*map(image_stream, sys.argv[2:]))
    with open(sys.argv[1], 'w') as dest:
        for imgs in batches_extraction(stream):
            imgs.tofile(dest, '')
