###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os 
import time
import onnx
import argparse
import data_reader
import numpy as np
import onnxruntime

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


def benchmark(model_path):

    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    total = 0.0
    runs = 10
    input_data = np.zeros(input_shape, np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f} ms")
    total /= runs
    print(f"Avg: {total:.2f} ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="input model")
    parser.add_argument(
        "--output_model", default=None, help="output model")
    parser.add_argument(
        "--calibration_dataset", default="", help="calibration data set"
    )
    parser.add_argument("--per_channel", default=True, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model

    model = onnx.load(input_model_path)
    model = onnx.version_converter.convert_version(model, 20)
    onnx.save(model, input_model_path)

    os.system(f"python -m onnxruntime.quantization.preprocess --input {input_model_path} --output {input_model_path}")

    output_model_path = args.output_model
    if output_model_path is None:
        output_model_path = input_model_path.replace(".onnx", ".q-int8.onnx")
    calibration_dataset_path = args.calibration_dataset
    dr = data_reader.CustomDataReader(
        calibration_dataset_path, input_model_path
    )

    # Calibrate and quantize model
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")
    print(f"Model is saved to {output_model_path}")

    print("Benchmarking ONNX fp32 model...")
    benchmark(input_model_path)

    print("Benchmarking ONNX int8 model...")
    benchmark(output_model_path)


if __name__ == "__main__":
    main()
