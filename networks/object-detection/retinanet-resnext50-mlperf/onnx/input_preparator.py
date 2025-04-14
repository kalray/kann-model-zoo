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

IMG_RES = 2 ** 8
IMG_SIZE = (800, 800)
PADDED_COLOR = (114, 114, 114)


def letterbox(
        img,
        new_shape=IMG_SIZE,
        color=PADDED_COLOR,
        auto=False,
        scaleFill=False,
        scaleup=True,
        auto_size=32
):

    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = numpy.mod(dw, auto_size), numpy.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def prepare_img(mat, letter_box=False):  # mat is BGR

    new_h, new_w = IMG_SIZE
    if letter_box:
        img = letterbox(mat, new_shape=(new_w, new_h), auto_size=64)[0]
    else:
        img = numpy.asarray(mat, dtype=numpy.uint8, order='C')
        img = numpy.flip(img, axis=-1)
        # resize dimension order is (height,width) in numpy but (width, height) in opencv
        if mat.shape[0:2] != (new_h, new_w):
            img = cv2.resize(mat, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    mat = numpy.asarray(img, dtype=numpy.float32, order='C')
    # Remove the mean values
    mat /= numpy.float32(255)
    mean = [_R_MEAN, _G_MEAN, _B_MEAN]
    assert len(mean) == mat.shape[-1]
    mat -= numpy.float32(mean)
    stddev = [_R_STDDEV, _G_STDDEV, _B_STDDEV]
    assert len(stddev) == mat.shape[-1]
    mat /= numpy.float32(stddev)

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
