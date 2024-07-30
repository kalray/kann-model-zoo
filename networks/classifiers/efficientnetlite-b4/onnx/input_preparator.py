#!/usr/bin/env python3
import sys
import numpy
import cv2
import os
import itertools as it
import math
from collections import OrderedDict
import click

_R_MEAN = 0.485
_G_MEAN = 0.456
_B_MEAN = 0.406
_R_STDDEV = 0.229
_G_STDDEV = 0.224
_B_STDDEV = 0.225

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


# Inspired from https://github.com/onnx/models/blob/bb0d4cf3d4e2a5f7376c13a08d337e86296edbe8/vision/classification/imagenet_preprocess.py#L7
# and : https://github.com/onnx/models/tree/master/vision/classification/resnet#preprocessing
def prepare_img(mat):  # mat is BGR
    mat = numpy.asarray(mat, dtype=numpy.uint8, order='C')
    mat_in = numpy.flip(mat, axis=-1)  # mat is RGB

    img = resize_with_aspectratio(mat_in, 224, 224, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, 224, 224)
    img = numpy.asarray(img, dtype = 'float32')
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

#     mat = AspectPreservingResize(mat_in, smallest_side=256)
#     # CenterCrop(mat, out_h, out_w)
#     mat = CenterCrop(mat, 224, 224)
#     mat = numpy.asarray(mat, dtype=numpy.float32, order='C')
#     # Remove the mean values
#     mat /= numpy.float32(255)
#     mean = [_R_MEAN, _G_MEAN, _B_MEAN]
#     assert len(mean) == mat.shape[-1]
#     mat -= numpy.float32(mean)
#     stddev = [_R_STDDEV, _G_STDDEV, _B_STDDEV]
#     assert len(stddev) == mat.shape[-1]
#     mat /= numpy.float32(stddev)
#     return mat


def AspectPreservingResize(mat, smallest_side=256):
    h, w, d = mat.shape
    h = numpy.float32(h)
    w = numpy.float32(w)
    if h > w:
        scale = numpy.float32(smallest_side) / w
    else:
        scale = numpy.float32(smallest_side) / h
    new_h = numpy.int32(numpy.rint(h * scale))
    new_w = numpy.int32(numpy.rint(w * scale))
    # resize dimenension order is (height, width) in numpy but (width,height) in opencv
    mat = cv2.resize(mat, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # To verify the height and width
    return mat


def CenterCrop(mat, out_h, out_w):
    matHeight, matWidth, matDepth = mat.shape
    h_start = (matHeight - out_h) // 2
    w_start = (matWidth - out_w) // 2
    h_size = out_h
    w_size = out_w

    mat = mat[h_start:h_start + h_size, w_start:w_start + w_size, :]
    return mat


def image_stream(filename):
    ''' Read and prepare the sequence of images of <filename>.
    If <filename> is an int, use it as a webcam ID.
    Otherwise <filename> should be the name of an image, video
    file, or image sequence of the form name%02d.jpg '''
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


def batches_extraction(stream, batch):
    ''' extract batches of images from a python generator of prepared images '''
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


def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img


# resize the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


# crop the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


@click.command()
@click.option('--batch-size', 'batch_size', default=1, help='Images per batch.')
@click.argument('destination', type=click.File('wb'))
@click.argument('inputs', nargs=-1, type=click.Path(exists=True))
def main(batch_size, destination, inputs):
    stream = it.chain(*map(image_stream, inputs))
    for imgs in batches_extraction(stream, batch_size):
        imgs.tofile(destination, '')


if __name__ == '__main__':
    main()
