###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

#! /usr/bin/env python

import os
import onnx
import onnx_tool
import argparse


def main(model_path, in_tensors, out_tensors):

    print("model path:  {}".format(model_path))
    print("inputs:      {}".format(in_tensors))
    print("outputs:     {}".format(out_tensors))


    # Checks
    model_file_path = os.path.realpath(model_path)
    onnx_model = onnx.load(model_file_path)  # load onnx model
    onnx.checker.check_model(onnx_model)
    print('complete ONNX model has been checked')

    input_names = in_tensors
    tensor_inter_outputs = out_tensors

    new_nn_path = os.path.join(
        os.path.dirname(model_file_path),
        os.path.basename(model_file_path).split('.onnx')[0] + str('.extract.onnx'))
    if os.path.isfile(new_nn_path):
        a = input(f"File {new_nn_path} already exists, overwrite it (y/N)? ")
        if not a == "y":
            exit(0)

    onnx.utils.extract_model(model_file_path, str(new_nn_path), input_names, tensor_inter_outputs)
    print(f'ONNX extracted network saved as {new_nn_path}')
    onnx_model = onnx.load(new_nn_path)  # load onnx model
    print('complete ONNX extracted model')
    # print the extracted model
    onnx_tool.model_profile(onnx_model)

    # Finish
    print(f'\nExport complete to {new_nn_path}. Visualize with http://netron.app')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("-i", default=None,
                        help="Inputs to split form networks, ie. -i in1,in2")
    parser.add_argument("-o", default=None,
                        help="Outputs to split from networks, ie. -o name1,name2")
    args = parser.parse_args()

    model = args.model_path
    in_t = args.i.split(',')
    out_t = args.o.split(',')
    main(model, in_t, out_t)
