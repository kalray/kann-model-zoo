###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import numpy
import onnxruntime
import input_preparator
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image


def _preprocess_img(img):
    """
    Preprocess function provided by input_preparator
    """
    img = numpy.array(img)
    nhwc_data = numpy.expand_dims(input_preparator.prepare_img(img), axis=0)
    nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
    return nchw_data


def _preprocess_random_images(height: int, width: int, size_limit=1000):
    """
    Generate a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    unconcatenated_batch_data = []
    for i in range(size_limit):
        random_image = numpy.random.uniform(0, 1, size= (height, width, 3))
        nchw_data = _preprocess_img(random_image)
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    
    return batch_data


def _preprocess_images_from_folder(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        nchw_data = _preprocess_img(pillow_img)
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class CustomDataReader(CalibrationDataReader):
    """
    Custom Calibration Dataset used to quantize a network.
    parameter calibration_image_folder: path to folder storing images
    parameter model_path: path to onnx model

    If the calibration_image_folder is empty, we generate 1000 random images.
    """
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        if calibration_image_folder != '':
            self.nhwc_data_list = _preprocess_images_from_folder(
                calibration_image_folder, height, width, size_limit=0
            )
        else:
            self.nhwc_data_list = _preprocess_random_images(height, width, size_limit=1000)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
