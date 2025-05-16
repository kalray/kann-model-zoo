###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import sys
import onnx
import onnxruntime as ort

# set models to latest version
models_path = list()
for m in sys.argv[1:]:
    model = onnx.load(m)
    model = onnx.version_converter.convert_version(model, 20)
    models_path.append(m.replace(".onnx", "_20.onnx"))
    onnx.save(model, models_path[-1])

model_one = onnx.load(models_path[0])
model_two = onnx.load(models_path[1])
sess_one = ort.InferenceSession(models_path[0])
sess_two = ort.InferenceSession(models_path[1])

m1 = { 
    "inputs_name": [i.name for i in sess_one.get_inputs()],
    "outputs_name": [o.name for o in sess_one.get_outputs()],
    "inputs_shape": [i.shape for i in sess_one.get_inputs()],
    "outputs_shape": [o.shape for o in sess_one.get_outputs()],
}

m2 ={ 
    "inputs_name": [i.name for i in sess_two.get_inputs()],
    "outputs_name": [o.name for o in sess_two.get_outputs()],
    "inputs_shape": [i.shape for i in sess_two.get_inputs()],
    "outputs_shape": [o.shape for o in sess_two.get_outputs()],
}

print("MERGING THE TWO FOLLOWING GRAPHs")
[print(f"{k}: {v}") for k, v in m1.items()]
[print(f"{k}: {v}") for k, v in m2.items()]

model_one = onnx.compose.add_prefix(model_one, "m1_", rename_inputs=False, rename_edges=False, rename_outputs=True)
model_two = onnx.compose.add_prefix(model_two, "m2_", rename_inputs=True, rename_edges=False, rename_outputs=False)

io_map = []
for o, i in zip(m1["outputs_name"], m2["inputs_name"]):
    io_map.append(("m1_" + o, "m2_" + i))

if m1["outputs_shape"] == m2["inputs_shape"]:
    new = onnx.compose.merge_models(
        model_one, model_two,
        io_map=io_map,
        inputs=m1["inputs_name"], 
        outputs=m2["outputs_name"],
        doc_string=""
    )

onnx.save(new, "merged_model.onnx")
print("Model saved to ./merged_model.onnx, to visualize it : $ netron -b merged_model.onnx")
a = input("Do you want to open it ? [y/N] ")
if a == "y" or a == "yes":
    os.system("netron -b merged_model.onnx &")
