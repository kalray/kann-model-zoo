###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import kann
import onnx
import numpy

from layers.silu import SiLU


def onnx_gather_callback(neural_network, prev_imgs, onnx_node, model_info):

    """
        Support Gather ONNX layer similar to a Slice/Squeeze layers
        where "indices" MUST be a constant in this case

        from : https://onnx.ai/onnx/operators/onnx__Gather.html

        | data shape | indices shape  | axis | output shape  | output equation                          | Supported |
        |------------|----------------|------|---------------|------------------------------------------| --------- |
        | (P, Q)     | () (a scalar)  | 0    | (Q)           | output[q] = data[indices, q]             |    [x]    |
        | (P, Q, R)  | () (a scalar)  | 1    | (P, R)        | output[p, r] = data[p, indices, r]       |    [x]    |
        | (P, Q)     | (R, S)         | 0    | (R, S, Q)     | output[r, s, q] = data[indices[r, s], q] |    [ ]    |
        | (P, Q)     | (R, S)         | 1    | (P, R, S)     | output[p, r, s] = data[p, indices[r, s]] |    [ ]    |

    """

    assert onnx_node.op_type == 'Gather'

    srcimg = prev_imgs[0]
    indices = prev_imgs[1]
    axis = onnx_node.attrs.get("axis", 0)
    if not isinstance(axis, int):
        return NotImplemented
    if not isinstance(indices, kann.constants.Constants):
        return NotImplemented
    if indices.data.size > 1:
        return NotImplemented

    # if indices is a constants,
    # Gather is alike Slice layer and Dims is Squeezed
    shape = srcimg.shape
    indice = indices.data

    # Check if indice is negative
    indice = list(range(shape[axis]))[indice] if indice < 0 else indice

    # Determine attributes for Slice Node
    nb_dims = srcimg.nb_dims
    start = [indice]
    end = [indice + 1]
    axes = [axis]
    step = [1]
    for a in range(nb_dims):
        if a not in axes and (a - nb_dims) not in axes:
            start = numpy.append(start, 0)
            end = numpy.append(end, shape[a])
            axes = numpy.append(axes, a)
            step = numpy.append(step, 1)

    # Create Slice node with attributes
    slice_node = kann.parsers.onnx_to_kann.OnnxNode(
        onnx.helper.make_node(
            "Slice",
            inputs=[srcimg.name],
            outputs=[onnx_node.outputs[0] + "_slice"]
        )
    )
    prev_imgs = [srcimg, start, end, axes, step]
    node, temp_img = kann.layers.slice.onnx_parser_callback_slice(
        neural_network, prev_imgs, slice_node, model_info)

    # Create Squeeze node with axis value
    squeeze_node = kann.parsers.onnx_to_kann.OnnxNode(
        onnx.helper.make_node(
            "Squeeze",
            inputs=[temp_img.name],
            outputs=onnx_node.outputs
        )
    )
    prev_imgs = [temp_img, numpy.array([axis])]  # Squeeze 'axes' dims
    node, dstimg = kann.layers.reshape.Reshape.onnx_parser_callback_squeeze(
        neural_network, prev_imgs, squeeze_node, model_info)

    return node, dstimg


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
    'Gather': onnx_gather_callback
}
