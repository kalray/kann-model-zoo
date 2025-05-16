import cv2
import sys
import numpy
import itertools as it

IMG_RES = 2 ** 8
IMG_SIZE = (640, 640)
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

    # Resize image to a 32-pixel-multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
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


def prepare_img(img0, letter_box=True, out_dtype=numpy.float32):
    new_h, new_w = IMG_SIZE
    if letter_box:
        img = letterbox(img0, new_shape=(new_w, new_h))[0]
    else:
        img = numpy.asarray(img0, dtype=numpy.uint8, order='C')
        # resize dimension order is (height,width) in numpy but (width, height) in opencv
        if img0.shape[0:2] != (new_h, new_w):
            img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Get the Values between 0 and 1 - BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / out_dtype(255.)
    return img