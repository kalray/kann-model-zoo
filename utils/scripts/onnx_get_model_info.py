###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
#
# author: Quentin Muller <qmuller@kalrayinc.com>
###

import os
import onnx
import numpy
import onnx_simplify
import matplotlib.pyplot as plt
import onnx_graphsurgeon as gs
from screeninfo import get_monitors


def get_flops(m: gs.Graph):

    """ Compute the number of floating point operations in a Graph (onnx_graphsurgeon)
        Inputs:
            m: onnx_graphsurgeon.Graph, input Graph imported from an ONNX ModelProto
        Outputs:
            result: dict, the list of graph's node name with FLOPs (M) as floating point value
                    e.g. result['name'] => float
                    NB: 'total' is added to retrieve the sum of all nodes
    """

    total_mflops = 0.
    result = {}
    for n in m.nodes:
        in_shape = [i.shape for i in n.inputs if i.shape is not None]
        out_shape = [o.shape for o in n.outputs if o.shape is not None]
        # Convolutions
        if n.op in ["Conv", "ConvTranspose"]:
            kernel_ops = numpy.prod(n.attrs.get('kernel_shape'))  # Kw x Kh
            bias_ops = len(n.inputs) == 3
            group = n.attrs.get('group', 1)
            in_channels = in_shape[0][1]
            macs = numpy.prod(out_shape[0]) * (in_channels // group * kernel_ops)  # + bias_ops
            mflops = 2 * macs / 1e6
        # Matmul
        elif n.op in ['MatMul']:
            macs = in_shape[0][-2] * numpy.prod(in_shape[1])
            mflops = 2 * macs / 1e6
        elif n.op in ['Gemm']:
            macs = numpy.prod(in_shape[0]) * out_shape[0][-1]
            mflops = 2 * macs / 1e6
        # Pooling
        elif n.op in ["GlobalAveragePool", "AveragePool", "MaxPool", "GlobalMaxPool", "MinPool",
                      "ReduceMean", "ReduceMax"]:
            kernel_ops = numpy.prod(n.attrs.get('kernel_shape', 1))  # Kw x Kh
            macs = numpy.prod(out_shape[0]) * kernel_ops
            mflops = macs / 1e6
        elif n.op in ["LRN"]:
            size = numpy.prod(n.attrs.get('size', 1))  # Kw x Kh
            macs = numpy.prod(out_shape[0]) * size
            mflops = macs / 1e6
        # Maths operations with FMA
        elif n.op in ["Abs", "Add", "Sub", "Mul", "Div", "Reciprocal", "BatchNormalization", "LayerNormalization"]:
            macs = numpy.prod(out_shape[0])
            mflops = 2 * macs / 1e6
        # Activation and single math function
        elif n.op in ["Elu", "Relu", "ReLU6", "PRelu", "Gelu", "Selu", "LeakyRelu", "HardSigmoid",
                      "HardSwish", "Sigmoid", "Pow", "Softmax", "Softplus", "Tanh", "Exp", "Erf", "Log", "Sqrt"]:
            macs = numpy.prod(out_shape[0])
            mflops = macs / 1e6
        elif n.op in ["QuantizeLinear", "DequantizeLinear"]:
            macs = numpy.prod(out_shape[0])
            mflops = 2 * macs / 1e6
        else:
            mflops = 0
        result[n.name] = mflops
        total_mflops += mflops
    result['total'] = total_mflops
    return result


def get_weights(g: gs.Graph):

    """ Compute the constants (weights) in a Graph (onnx_graphsurgeon)
        Inputs:
            m: onnx_graphsurgeon.Graph, input Graph imported from an ONNX ModelProto
        Outputs:
            w: dict, the list of graph's node name with the size of constant (list(int))
                    e.g. result['name'] => [int, int]
                    NB: 'total' is added to retrieve the sum of all nodes
    """

    w = {}
    sum_w = 0
    for n in g.nodes:
        w[n.name] = []
        for o in n.inputs:
            if isinstance(o, gs.Constant):
                w[n.name].append(list(o.shape))
                sum_w += numpy.prod(o.shape)
    w['total'] = sum_w
    return w


def check_model(g: gs.Graph):
    # if graph's node has empty names, add node names to
    # count flops and weights by
    for n in g.nodes:
        if not n.name:
            n.name = "Node_" + n.outputs[0].name
    g.toposort()
    g.cleanup()
    return g


def pprint_short(m: onnx.ModelProto):

    """ Print a short descriptioin of an ONNX ModelProto """

    print("\n---")
    print("Model description:")
    print("------------------")
    opset = str(m.opset_import[0]).replace("\n", ' - ')
    print(f"  opset: {opset}")
    print(f"  IR: {m.ir_version}")
    print(f"  producer:{m.producer_name} ({m.producer_version})")
    print(f"  Layers: {len(m.graph.node)}")

    if isinstance(m, onnx.ModelProto):
        g = gs.import_onnx(m)
        check_model(g)

    total_flops = get_flops(g)['total'] # in MFLOPS
    if total_flops > 100.:
        print(f"  FLOPs : {total_flops / 1e3:,.3f} G")
    elif total_flops > 10.:
        print(f"  FLOPs : {total_flops:,.3f} M")
    else:
        print(f"  FLOPs : {total_flops * 1e3:,.3f} K")
    total_params = get_weights(g)['total'] # Params
    if total_params > 1e8:
        print(f"  Params: {total_params / 1e9:.3f} G")
    elif total_params > 1e5:
        print(f"  Params: {total_params / 1e6:.3f} M")
    else:
        print(f"  Params: {total_params / 1e3:.3f} K")


def pprint_io(m: gs.Graph):

    """ Print inputs/outputs of a Graph (onnx_graphsurgeon) """

    print("\n---")
    print("Model inputs")
    print("------------")
    for i in m.inputs:
        print(f"  + {i.name:20s}: {i.shape}")
    print("------------")
    print("Model outputs")
    print("-------------")
    for o in m.outputs:
        print(f"  + {o.name:20s}: {o.shape}")
    print("-------------")


def pprint_nodes(m: gs.Graph, nodeByNode=True):

    """ Print a profile layer by layer of a Graph (onnx_graphsurgeon)
        with the following data: NAME, OP, FLOPS, WEIGHTS, INPUTS, OUTPUTS
        if nodeByNode,
        otherwise a summary of the Graph by node operations
    """

    flops = get_flops(m)
    weights = get_weights(m)

    print("\n---")
    print("Model nodes")
    print("-----------")
    if nodeByNode:
        print(f"{'Node name':32s} : {'Node OP':20s} | {'M FLOPS':>15s} | {'W SIZE (MB)':>15s} | {'Input shapes':40s} | {'Output shapes':20s}")
        print(f"{'-'*32:32s} : {'-'*20:20s} | {'-'*15:15s} | {'-'*15:15s} | {'-'*40:40s} | {'-'*20:20s}")
        for n in m.nodes:
            input_shapes = str(",".join(["x".join([str(x) for x in i.shape]) for i in n.inputs if i.shape is not None]))
            output_shapes = str(",".join(["x".join([str(x) for x in o.shape]) for o in n.outputs if o.shape is not None]))
            mflops = flops[n.name]
            if mflops > 0:
                mflops_s = f"{mflops:,.1f} ({100*mflops/flops['total']:.1f} %)"
            else:
                mflops_s = "-"
            wsize = sum([numpy.prod(w) for w in weights[n.name]])
            if wsize > 0:
                wsize_s = f"{wsize*4/1e6:,.1f} ({100*wsize/weights['total']:.1f} %)"
            else:
                wsize_s = "-"
            print(f"  {n.name[-30:]:30s} : {n.op:20s}", end=" | ")
            print(f"{mflops_s:>15s}", end=" | ")
            print(f"{wsize_s:>15s}", end=" | ")
            print(f"{input_shapes:40s}", end=" | ")
            print(f"{output_shapes:20s}")
    else:
        nodes = {}
        for n in m.nodes:
            if n.op not in nodes:
                nodes[n.op] = [flops[n.name]]
            else:
                nodes[n.op] += [flops[n.name]]
        print(f"{'Node OP':20s} | {'Quantity':>10s} | {'M FLOPS':>15s} ")
        print(f"{'-'*20:20s} | {'-'*10:10s} | {'-'*15:15s}")
        for op, f in nodes.items():
            print(f"{op:20s} | {len(f):10d} | {sum(f):15,.2f}  ({100*sum(f)/flops['total']:.1f} %)")
    print(f"\nTotal compute : {flops['total'] / 1e3:.3f} GFLOPs, {weights['total'] / 1e6:.2f} MParams")


def print_graph(m: gs.Graph, filename, savefig=True):

    """ Print a view layer by layer of a Graph (onnx_graphsurgeon) with maptlotplib """

    flops = get_flops(m)
    weights = get_weights(m)

    # normalize values
    flops_total = flops.pop("total")
    w_total = weights.pop("total")
    model_weights = {}
    for n in m.nodes:
        model_weights[n.name] = sum([numpy.prod(w) for w in weights[n.name]])
    for n in m.nodes:
        flops[n.name] /= flops_total
        model_weights[n.name] /= w_total

    h = 15
    plt.figure(figsize=(10, h))
    yflops = numpy.array([100 * v for v in reversed(flops.values())])
    yweights = numpy.array([-100 * v for v in reversed(model_weights.values())])
    plt.barh([k for k in reversed(flops.keys())], yflops, height=1.5, color='blue', label="flops")
    plt.barh([k for k in reversed(model_weights.keys())], yweights, height=1.5, color='cyan', label="weights")
    plt.grid(linestyle='dashed', linewidth=0.5, color="lightgray")
    plt.title(f"{filename}\nLayers: {len(m.nodes)}, {flops_total/1e3:.2f} GFLOPS, {w_total / 1e6:.1f} MParams", fontsize=14, loc="left")
    plt.xlabel('<- WEIGHTS (%) | FLOPS (%) ->', fontsize=12)
    plt.ylabel('Layer name (OUT <- IN)', fontsize=12)
    plt.xlim(1.1*min((-yflops).min(), yweights.min()), 1.1*max((yflops).max(), (-yweights).max()))
    plt.yticks(fontsize=max(5, min(h*100 // len(flops), 11)))
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{filename}-profile.png", dpi=220)
        print(f"Figure has been saved here: {filename}-profile.png")
    else:
        plt.show()


def main(model_path, profile=False):

    """ Print a short description of the model targeted by model_path
        and a profiling layer by layer or a summary of all nodes (w/ FLOPs)
    """

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    opt_model, _ = onnx_simplify.simplify(onnx_model, check_n=2)  # to enable the shape inference ++
    print(f"\nAnalyzing {model_path} ({os.path.getsize(model_path) / 1e6:.2f} MB)")
    pprint_short(opt_model)

    gs_model = gs.import_onnx(opt_model)
    check_model(gs_model)

    pprint_io(gs_model)
    if profile:
        pprint_nodes(gs_model)
        try:
            get_monitors()[0]
            print_graph(gs_model, os.path.basename(model_path), savefig=False)
        except:
            print("[WARNING] ** Screen or Display has not been found **\n")
            print_graph(gs_model, os.path.basename(model_path), savefig=True)
    else:
        pprint_nodes(gs_model, False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--profile", "-p",
        action='store_true',
        default=False,
        help="Profile layer per layer")
    opt = parser.parse_args()
    main(opt.model_path, opt.profile)
