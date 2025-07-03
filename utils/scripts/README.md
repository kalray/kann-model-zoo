# KaNN™ Model Zoo - Associated scripts

## Contents

In this sub directory, you should find different scripts to manipulate ONNX or TF neural networks. A short description is available below:

* `onnx_merge_models.py`: merge 2 ONNX models. IO mapping needs to match with the both neural networks.
* `onnx_rename_node_io.py`: rename the inputs and/or outputs of an ONNX model
* `tensorflow_get_flops_params.py`: provides model informations on FLOPS and NB params from TF graph/model
* `tensorflow_get_tensor_shape.py`: delivers tensor shape value for a TF graph/model
* `tensorflow_optimize_graph.py`: optimize a TF graph using GraphTransform API (see more at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md)
* `onnx_quantization/onnx_quantization.sh` : allow ONNX model quantization to INT8.
* `onnx_extract_model.py`: would extract a part of the ONNX model. Example:

  ```
  python3 onnx_extract_model.py <model.onnx> -i <input1>,<inputN> -o <out1>,<outN>
  ```
* `onnx_get_model_info.py`: extracts info (FLOPS, PARAMS, I/O name & shapes, Nodes) from an ONNX model. Example:

  ```
  python3 onnx_get_model_info.py <model.onnx>
    ---
    Model description:
    ------------------
    opset: domain: "" - version: 20 - 
    IR: 9
    producer:pytorch (2.6.0)
    Layers: 233
    FLOPs : 8.793 G
    Params: 3.194 M

    ---
    Model inputs
    ------------
    + images              : [1, 3, 640, 640]
    ------------
    Model outputs
    -------------
    + predictions         : [1, 84, 8400]
    -------------

    ---
    Model nodes
    -----------
    Node OP              |   Quantity |         M FLOPS 
    -------------------- | ---------- | ---------------
    Conv                 |         64 |        8,743.99  (99.4 %)
    Sigmoid              |         58 |           14.56  (0.2 %)
    Mul                  |         58 |           27.84  (0.3 %)
    Split                |          9 |            0.00  (0.0 %)
    Add                  |          8 |            2.22  (0.0 %)
    Concat               |         19 |            0.00  (0.0 %)
    MaxPool              |          3 |            3.84  (0.0 %)
    Resize               |          2 |            0.00  (0.0 %)
    Reshape              |          5 |            0.00  (0.0 %)
    Transpose            |          1 |            0.00  (0.0 %)
    Softmax              |          1 |            0.54  (0.0 %)
    Slice                |          2 |            0.00  (0.0 %)
    Sub                  |          2 |            0.07  (0.0 %)
    Div                  |          1 |            0.03  (0.0 %)
  ```
