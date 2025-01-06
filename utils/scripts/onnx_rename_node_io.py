###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import onnx
import argparse
import onnx_graphsurgeon as gs


def pprint(m:gs.Graph):
    print("---")
    print("Model inputs")
    print("---")
    for i in m.inputs:
        print(f"{i.name:30s}: {i.shape}")
    print("---")
    print("Model nodes")
    print("---")
    for n in m.nodes:
        input_shapes = f"{[i.shape for i in n.inputs]}"
        output_shapes = f"{[i.shape for i in n.outputs]}"
        print(f"{n.name[-30:]:30s}: {n.op:10s} in: {input_shapes:50s} out: {output_shapes:20s}")
    print("Model outputs")
    print("---")
    for o in m.outputs:
        print(f"{o.name:30s}: {o.shape}")
    print("---")


def main(model_path, old, new):
    onnx_model = onnx.load(model_path)
    gs_model = gs.import_onnx(onnx_model)

    for i in gs_model.inputs:
        if i.name == old:
            i.name = new
    for o in gs_model.outputs:
        if o.name == old:
            o.name = new

    gs_model.cleanup()
    gs_model.toposort()
    pprint(gs_model)

    new_onnx_model = gs.export_onnx(gs_model)
    onnx.checker.check_model(new_onnx_model)
    print(f'Model with new input name {new} has been checked')
    new_model_path = model_path.replace(".onnx", "-new.onnx")
    onnx.save(new_onnx_model, new_model_path)
    print(f'Model with new input name {new} has been saved to {new_model_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--old_name", '-o', default='', help="Node old name")
    parser.add_argument("--new_name", '-n', default='', help="Node new name")
    opt = parser.parse_args()
    main(opt.model_path, opt.old_name, opt.new_name)
