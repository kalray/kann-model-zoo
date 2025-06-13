#!/usr/bin/env python3
import cv2
import click
import numpy
import itertools as it


def prepare_img(mat, out_dtype=numpy.float32):  # mat is BGR
    mat = numpy.asarray(mat, dtype=numpy.uint8, order='C')
    mat = cv2.resize(mat, (256, 256), interpolation=cv2.INTER_LINEAR)
    mat = numpy.asarray(mat, dtype=numpy.float32, order='C')
    mat /= numpy.float32(255.)
    return mat.astype(out_dtype)


def image_stream(filename):
    """ Read and prepare the sequence of images of <filename>.
    If <filename> is an int, use it as a webcam ID.
    Otherwise <filename> should be the name of an image, video
    file, or image sequence of the form name%02d.jpg """
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
    """ extract batches of images from a python generator of prepared images """
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
