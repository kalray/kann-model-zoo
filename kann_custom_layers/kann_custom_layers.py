###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import kann

from layers.silu import SiLU

def onnx_silu_parser_callback(neural_network, prev_imgs, onnx_nodes, model_info):
    # This callback will only be called for 'SiLU' ONNX layers, because of the
    # key associated to it in the onnx_parser_callbacks dict.
    mul_node = onnx_nodes
    assert mul_node.op_type == 'Mul'
    assert len(prev_imgs) == 1
    dstimg = kann.images.image_smem.ImageSMEM(
        neural_network, mul_node.outputs[0], prev_imgs[0].shape)
    srcview = kann.subview.Subview.fromImage(prev_imgs[0])
    dstview = kann.subview.Subview.fromImage(dstimg)
    assert srcview.count == dstview.count
    layer = SiLU(
        neural_network,
        mul_node.outputs[0],
        srcview, dstview, mul_node.name,
        simd=True
    )
    return layer, dstimg


onnx_parser_callbacks = {
    'Silu': onnx_silu_parser_callback,
}

tensorflow_parser_callbacks = {
}

tflite_parser_callbacks = {
}
