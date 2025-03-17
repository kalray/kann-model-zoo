###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
#
# Authors:
#  Quentin Muller, <qmuller@kalrayinc.com>
#  Daniel Angulo, <dangulo@kalrayinc.com
###

import os
import sys
import yaml
import onnx
import numpy
import shutil
import onnxsim
import logging
import argparse
import subprocess


class cFormatter(logging.Formatter):

    grey = "\x1b[37;2m"
    white = "\x1b[38;0m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;1m"
    bold_red = "\x1b[35;1m"
    reset = "\x1b[0;0m"
    format = "%(levelname)s: %(message)s"
    format_err = "%(asctime)s | [%(levelname)s]: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: format,
        logging.WARNING: yellow + format_err + reset,
        logging.ERROR: red + format_err + reset,
        logging.CRITICAL: bold_red + format_err + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class kHelpFormat(argparse.HelpFormatter):
    def __init__(self, prog=None, indent_increment=4, width=70):
        if prog is None:
            prog = os.path.basename(sys.argv[0])
        super().__init__(prog, indent_increment=indent_increment, width=width)


class kHelp(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print("\nThis is a wrapper to download model from 🤗 and generate model with kann\n")
        print("*" * 80)
        print("")
        parser.print_help()
        print("")
        print("*" * 80)
        print("\n $ kann generate --help for more informations (detailed below)\n")
        cmd_args = ["kann", "generate", "--help"]
        subprocess.run(cmd_args, check=True)
        sys.exit(0)


# create logger
logger = logging.getLogger("KaNN-Model-Zoo")
logger.setLevel(logging.INFO)

stdout_console_handler = logging.StreamHandler(sys.stdout)
stdout_console_handler.setFormatter(cFormatter())
logger.addHandler(stdout_console_handler)


def get_mppa_frequency():
    """ Return the frequencies set for the MPPA's clusters """
    mppa_freq_hz = []
    with open('/mppa/board0/mppa0/freq', 'r') as f_hw:
        for l in f_hw.readlines():
            txt = str(l).rstrip()
            mppa_freq_hz.append(float(txt.split(' ')[2]))
    return mppa_freq_hz


def eval_env(bin_file, pocl_file, arch):

    """ Evaluate HW and SW environment """

    logger.info('')
    logger.info('---------------------------')
    logger.info(' Evaluating HW environment:')
    logger.info('---------------------------')
    status = "\U00002705"
    try:
        with open('/mppa/board0/archver', 'r') as f_hw:
            hw_arch_version = str(f_hw.readlines()[0]).rstrip()
        with open('/mppa/board0/type', 'r') as f_hw:
            hw_board_type = str(f_hw.readlines()[0]).rstrip()
        with open('/mppa/board0/revision', 'r') as f_hw:
            hw_board_rev = str(f_hw.readlines()[0]).rstrip()
        with open('/mppa/board0/serial', 'r') as f_hw:
            hw_board_serial = str(f_hw.readlines()[0]).rstrip()
        with open('/mppa/version', 'r') as f_hw:
            hw_driver = str(f_hw.readlines()[0]).rstrip()
        clusters_freq = numpy.array(get_mppa_frequency()) / 1e6

        if hw_arch_version != arch:
            status = "\U0000274C"
            logger.warning(
                f"\U0000274C Mismatch HW ARCH revision ({hw_arch_version})"
                f" with ARCH to benchmark ({arch}), RUN is disabled")
        logger.info(f"[{status}] HW revision found:  {hw_arch_version}")
        logger.info(f"[{status}] HW board :          {hw_board_type}-rev{hw_board_rev} | {hw_board_serial}")
        logger.info(f"[{status}] Driver version:     {hw_driver}")
        logger.info(f"[{status}] Cluster freq (MHz): {clusters_freq}")

    except Exception as err:
        logger.error(
            '>> HW issue, please check that MPPA is '
            'plugged on Host machine, get {}'.format(str(err)))
        raise RuntimeError(str(err))

    logger.info('')
    logger.info('---------------------------')
    logger.info(' Evaluating SW environment:')
    logger.info('---------------------------')

    toolchain_dir = os.environ.get("KALRAY_TOOLCHAIN_DIR")
    if toolchain_dir is None:
        msg = 'KALRAY_TOOLCHAIN_DIR env is undefined, please source Kalray(R) toolchain dir'
        logger.error(msg)
        raise RuntimeError(msg)
    if not os.path.isfile(bin_file):
        status = "\U0000274C"
        logger.warning("[{}] Binary file:   {}".format(status, bin_file))
        raise RuntimeError('>> {} is not found'.format(bin_file))
    else:
        logger.info("[{}] Binary file:   {}".format(status, bin_file))

    if not os.path.isfile(pocl_file):
        status = "\U0000274C"
        logger.warning("[{}] Kernel file:  {}".format(status, pocl_file))
        raise RuntimeError('>> {} is not found'.format(pocl_file))
    else:
        logger.info("[{}] Kernel file:   {}".format(status, pocl_file))

    try:
        import kann
    except ImportError as err:
        logger.error('>> \U0000274C KaNN(tm) must be referenced into '
                     'your python environment, please source your python env.\n'
                     'and use pip to install KaNN wheel as described in the doc')
        raise RuntimeError(err)

    ace_p = subprocess.run(["pkg-config", "--modversion", "kalray-ace"],  capture_output=True)
    ace_version = f"{ace_p.stdout.decode('utf-8')}".rstrip("\n")
    logger.info("[!!] ACE release :  {} - {}".format(toolchain_dir, ace_version))
    logger.info("[!!] KaNN(TM) :     {} - {}".format(
        os.path.dirname(kann.__file__), kann.__version__))
    logger.info('---------------------------')


def set_winsize(onnxPath, hw=(224, 224), oPath=None):

    import onnx_graphsurgeon as gs

    onnxRelativePath = os.path.realpath(onnxPath)
    onnxModel = onnx.load(onnxRelativePath)
    g = gs.import_onnx(onnxModel)
    # save input name and new shape
    nn_inputs = {}
    for _in in g.inputs:
        nn_inputs[_in.name] = _in.shape
        nn_inputs[_in.name][2:] = hw
    # set dynamic output shape
    for _out in g.outputs:
        _out.shape = ['x'] * len(_out.shape)
    # clean all shape tensor to None value
    for i, n in enumerate(g.nodes):
        if i > 0:
            for node_in in n.inputs:
                if isinstance(node_in, gs.Variable):
                    node_in.shape = None
    # export graph to onnx
    newModel = gs.export_onnx(g)
    # optimize and infer shape to reset the tensor and
    # compute the output shape
    optModel, c = onnxsim.simplify(
        newModel,
        # check_n=2,
        overwrite_input_shapes=nn_inputs)
    onnx.checker.check_model(optModel)
    if oPath is None:
        oPath = f"model.optimized.{hw[0]}x{hw[1]}.onnx"
    os.makedirs(os.path.dirname(oPath), exist_ok=True)
    onnx.save(optModel, oPath)
    return oPath


def build_new_model(nPath, size, cfgDir):

    if os.path.isdir(nPath):
        cPath = os.path.join(nPath, "network.dump.yaml")
    else:
        cPath = nPath
    with open(cPath, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.Loader)

    modelPath = os.path.abspath(os.path.join(os.path.dirname(cPath), cfg.get('onnx_model')))
    if not os.path.isfile(modelPath):
        modelPath = os.path.join(nPath, "optimized-model.onnx")
        if not modelPath.startswith("/"):
            modelPath = os.path.realpath(os.path.join(os.path.dirname(cPath), modelPath))  # to get an absolute path
    if not os.path.isfile(modelPath):
        raise FileNotFoundError(f"{modelPath} is not found")

    h, w = size
    os.makedirs(cfgDir, exist_ok=True)
    modelDir = os.path.join(cfgDir, "models")
    outModelName = os.path.basename(modelPath).replace(".onnx", f".{h}x{w}.onnx")
    outModelPath = os.path.join(modelDir, outModelName)
    set_winsize(modelPath, size, outModelPath)

    cfg['onnx_model'] = outModelPath.replace(cfgDir, ".")
    for input_node in cfg['input_nodes_shape']:
        input_node[0] = h
        input_node[2] = w
    del cfg['output_nodes_shape']  # it assumes that KaNN compute the output shape

    cfg_file_path = os.path.join(cfgDir, f"network.{h}x{w}.yaml")
    with open(cfg_file_path, 'w+') as yfile:
        yaml.dump(cfg, yfile, indent=4, sort_keys=False)
    return cfg_file_path


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(prog="kann-apps-utils")
    main_parser.add_argument(
        "--yaml", default=None, type=str,
        help="Force input size on HW dims")
    main_parser.add_argument(
        "--framework", default="onnx", type=str,
        help="Define the framework to use")
    main_parser.add_argument(
        "--force-input-hw", "-hw", default=None, type=str,
        help="Force input size on HW dims")
    main_parser.add_argument(
        "--generate", default=False, action='store_true',
        help="Generate with KaNN(TM)")
    args = main_parser.parse_args()
    if args.force_input_hw is not None:
        if not os.path.isfile(args.yaml):
            raise FileNotFoundError(args.yaml)
        if args.framework == "onnx":
            # get window size to apply
            h, w = [int(d) for d in args.force_input_hw.split(',')]
            # build new networks with input size
            network = os.path.realpath(os.path.join(args.yaml, "..", ".."))
            mDir = network + f"_{h}x{w}"
            srcDir = os.path.join(network, args.framework)
            dstDir = os.path.join(mDir, args.framework)
            genDir = "generated_" + os.path.basename(mDir)
            os.makedirs(mDir, exist_ok=True)
            os.makedirs(dstDir, exist_ok=True)
            shutil.copytree(srcDir, dstDir, dirs_exist_ok=True)
            cYamlPath = build_new_model(args.yaml, (h, w), dstDir)
            logger.info(f'** New model at: {cYamlPath} ** \n')
            # Then, generate with KaNN
            if args.generate:
                cmd_args = ["kann", "generate", cYamlPath, "-d", genDir, "--force"]
                logger.info("Running: {}".format(" ".join(cmd_args)))
                subprocess.run(cmd_args, check=True)
                args.network = genDir
            logger.info('** Done ** \n')
        else:
            raise NotImplementedError
