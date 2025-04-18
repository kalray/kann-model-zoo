#!/usr/bin/env python3
import cv2
import sys
import numpy
import itertools as it

IMG_RES = 2 ** 8
IMG_WIDTH, IMG_HEIGHT = 512, 512


def prepare_img(img, out_dtype=numpy.float32):
    mat = numpy.asarray(img, dtype=numpy.uint8, order='C')
    mat = numpy.flip(mat, axis=2)
    # resize dimension order is (height,width) in numpy but (width, height) in opencv
    if img.shape[0:2] != (IMG_HEIGHT, IMG_WIDTH):
        mat = cv2.resize(mat, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
    return mat.astype(out_dtype)


def image_stream(filename):
    """ Read and prepare the sequence of images of <filename>.
      If <filename> is an int, use it as a webcam ID.
      Otherwise <filename> should be the name of an image, video
      file, or image sequence of the form name%02d.jpg """
    try: src = int(filename)
    except ValueError: src = filename
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
        while len(imgs) != batch: # last batch might not be full
            imgs.append(numpy.zeros(imgs[0].shape, dtype=imgs[0].dtype))
        # interleave the batch as required by kann (HBWC axes order)
        # note: could use np.stack(axis=1) here, but it's not available in np 1.7.0
        for i in range(len(imgs)):
            imgs[i] = numpy.reshape(imgs[i], imgs[i].shape[:1]+(1,)+imgs[i].shape[1:])
        imgs = numpy.concatenate(imgs, axis=1)
        yield imgs


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: '+sys.argv[0]+' <destination_file> <sources>...')
        print('called with {} args: {}'.format(len(sys.argv), ', '.join(sys.argv)))
        exit(1)
    stream = it.chain(*map(image_stream, sys.argv[2:]))
    with open(sys.argv[1], 'w') as dest:
        for imgs in batches_extraction(stream):
            imgs.tofile(dest, '')
