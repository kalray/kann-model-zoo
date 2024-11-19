#! /usr/bin/env python3

###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import yaml
import numpy
import shutil
import argparse
import tempfile
import subprocess

from kann_utils import (
    build_new_model,
    generate_raw_input,
    logger,
    eval_env,
    get_mppa_frequency
)


def infer(all_options):

    options = all_options[0]
    pocl_file = os.path.join(args.pocl_dir, "mppa_kann_opencl.cl.pocl")

    # Generate IO dir for inference (custom or randoms)
    if options.data_path is not None and os.path.isdir(options.data_path):
        iodir = f"{os.path.basename(args.generated_dir)}_custom_io"
        if os.path.exists(iodir):
            shutil.rmtree(iodir)
        shutil.copytree(options.data_path, iodir)
    else:
        iodir = tempfile.mkdtemp()
        logger.info("Generating IO dir into {}".format(iodir))
        generate_raw_input(
            os.path.join(options.generated_dir, "network.dump.yaml"),
            iodir, nb_frames=options.nb_frames, data=options.data)
        if options.data_path is not None:
            logger.warning(f"[!] {options.data_path} is a wrong path, random data has been generated instead")
        options.data_path = None
    logger.info(f"[+] Inputs/Outputs directory is located to {iodir}")

    # Get serialized binary file
    if os.path.isdir(options.generated_dir):
        serialized_bin = [f for f in os.listdir(options.generated_dir) if f.split('.')[-1] == "kann"]
        if len(serialized_bin) != 1:
            raise RuntimeError(
                "Something went wrong when parsing serialized binary file in {}\n"
                "    continuing ... ".format(options.generated_dir))
    else:
        raise RuntimeError("Network path {} is not a valid directory".format(options.generated_dir))
    serialized_bin_file = os.path.join(options.generated_dir, serialized_bin[0])

    # Generate inference log file path
    i = 0
    inference_log_path = f"inference_{os.path.basename(options.generated_dir)}.log"
    _inference_log_path = inference_log_path
    while os.path.exists(_inference_log_path):
        _inference_log_path = inference_log_path.replace(".log", "") + f"_{i}.log"
        i += 1
    inference_log_path = _inference_log_path

    # Generate command to run
    cmd = [args.bin_file, serialized_bin_file, iodir]
    try:
        print("Running: {}".format(" ".join(cmd)))
        with open(inference_log_path, "w+") as inference_log:
            subprocess.run(
                cmd,
                stdout=inference_log,
                stderr=inference_log,
                env=dict(os.environ, KANN_POCL_FILE=pocl_file),
                check=True, timeout=30.)
        logger.info("Done")
    finally:
        with open(inference_log_path, "r") as inference_log:
            print(" ".join(inference_log.readlines()))
        logger.info("Log is available at {}".format(inference_log_path))

    with open(inference_log_path, "r") as inference_log:
        ilog = inference_log.readlines()

    with open(os.path.join(args.generated_dir, "network.dump.yaml"), 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.Loader)
        fbs = cfg.get('forced_batch_size', 1)
        if fbs is None:
            fbs = 1

    perf_host_qps = []
    perf_host_ms = []
    mean_perf_mppa_cycles = None

    MPPA_FREQ_KHZ = get_mppa_frequency()[0] / 1e3

    logger.info(f"***********************************")
    logger.info(f"DATA EXTRACTED FROM INFERENCE LOG:")
    logger.info(f"***********************************")
    for line in ilog:
        if line.startswith("Total: "):
            mean_perf_mppa_cycles = int(line.split(" ")[1])
        if line.startswith("[app][host] Performance of frame "):
            perf_host_qps.append(float(line.split(" ")[-2]))
            perf_host_ms.append(float(line.split(" ")[5]))
    if len(perf_host_qps) > 0:
        logger.info(f"Batch size / query: {str(fbs):>7s}")
        logger.info(f"Number of queries:  {str(len(perf_host_ms)):>7s}")
        logger.info(f"PERF HOST (ms):     {str(round(numpy.mean(perf_host_ms), 2)):>7s} ms")
        logger.info(f"PERF HOST (qps):    {str(round(numpy.mean(perf_host_qps), 2)):>7s} qps")
        logger.info(f"PERF HOST (fps):    {str(round(fbs * numpy.mean(perf_host_qps), 2)):>7s} fps")
    else:
        logger.info("No perf metrics from host")
    if mean_perf_mppa_cycles:
        logger.info(f"PERF DEVICE (ms):   {str(round(mean_perf_mppa_cycles / MPPA_FREQ_KHZ, 2)):>7s} ms")
        logger.info(f"PERF DEVICE (qps):  {str(round(1e3 * MPPA_FREQ_KHZ / mean_perf_mppa_cycles, 2)):>7s} qps")
        logger.info(f"PERF DEVICE (fps):  {str(round(fbs * (1e3 * MPPA_FREQ_KHZ) / mean_perf_mppa_cycles, 2)):>7s} fps")
    else:
        logger.info("No perf metrics from MPPA")
    logger.info(f"***********************************")

    if options.data_path is None:
        logger.info("Removing IO dir {}".format(iodir))
        shutil.rmtree(iodir)
    else:
        logger.info(f"[!] IO are available here: {iodir}")
    logger.info("Done")


def demo(all_options):
    options = all_options[0]

    pocl_file = os.path.join(args.pocl_dir, "mppa_kann_opencl.cl.pocl")
    if options.device == "mppa":
        python_script = "kann_video_demo.py"
        python_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), python_script))
        cmd_args = ["python3", python_script]
        cmd_args += [f"--binaries-dir={os.path.dirname(options.bin_file)}"]
        cmd_args += [f"--kernel-binaries-dir={os.path.dirname(pocl_file)}"]
        cmd_args += [options.generated_dir]
        cmd_args += [options.src]
        if options.save_img:
            cmd_args += [f"--save-img"]
        if options.no_replay:
            cmd_args += [f"--no-replay"]
        if options.no_display:
            cmd_args += [f"--no-display"]
        cmd_args += all_options[1]
        print(f"Running: {' '.join(cmd_args)}\n")
        subprocess.run(cmd_args, check=True)

    if options.device == "cpu":
        python_script = "cpu_video_demo.py"
        python_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), python_script))
        yaml_file_path = os.path.join(options.generated_dir, "network.dump.yaml")
        cmd_args = ["python3", python_script]
        cmd_args += [yaml_file_path]
        cmd_args += [options.src]
        if options.save_img:
            cmd_args += [f"--save-img"]
        if options.no_replay:
            cmd_args += [f"--no-replay"]
        if options.no_display:
            cmd_args += [f"--no-display"]
        cmd_args += all_options[1]
        print(f"Running: {' '.join(cmd_args)}\n")
        subprocess.run(cmd_args, check=True)


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(prog="run")
    subparsers = main_parser.add_subparsers(dest="cmd", required=True)

    infer_subparser = subparsers.add_parser(
        name="infer",
        help="Run inference on raw or custom data from kann_opencl_cnn or custom binary")
    infer_subparser.add_argument(
        "generated_dir", default=str(),
        help="Provide the path of generated DIR by KaNN")
    infer_subparser.add_argument(
        "--data", default="zeros", choices=["zeros", "random", "ones"],
        help="Provide data type to generate")
    infer_subparser.add_argument(
        "--data-path", type=str,
        help="Provide data path if custom data to analyze")
    infer_subparser.add_argument(
        "--force-hw", type=str, default=None,
        help="Force the input window size (H, W) of the initial generated DIR, save "
             "it in a DIRECTORY then do inference")
    infer_subparser.add_argument(
        "--nb-frames", "-n", default=10, type=int,
        help="Provide number of frame to generate")
    infer_subparser.set_defaults(func=infer)

    demo_subparser = subparsers.add_parser(
        name="demo",
        help="Run demo application including Pre and Post processing")
    demo_subparser.add_argument(
        "generated_dir", default=str(),
        help="Provide the path of generated DIR by KaNN")
    demo_subparser.add_argument(
        "src", default=str(),
        help="Provide source path or video stream source (i.e. 0)")
    demo_subparser.add_argument(
        "--device", "-d", default="mppa", choices=["mppa", "cpu"],
        help="Select the device to run the demo on")
    demo_subparser.add_argument(
        '--no-display', action='store_true',
        help="Disable graphical display")
    demo_subparser.add_argument(
        '--no-replay', action='store_true',
        help="Disable video loop if source is a video file.")
    demo_subparser.add_argument(
        '--save-video', action='store_true',
        help="Save input video with output predictions as video file.")
    demo_subparser.add_argument(
        '--save-img', action='store_true',
        help="Save last frame with output predictions as video file.")
    demo_subparser.set_defaults(func=demo)

    main_parser.add_argument(
        "--bin-file", default=None, type=str,
        help="Provide binary host file to use for inference, default=kann_opencl_cnn")
    main_parser.add_argument(
        "--pocl-dir", default=None, type=str,
        help="Provide POCL dir to use for inference on MPPA")
    main_parser.add_argument(
        "--disable-l2-cache", "--l2-off", default=False, action='store_true',
        help="Disable L2 cache to extend data buffer size")
    main_parser.add_argument(
        "--enable-l2-cache", "--l2-on", default=True, action='store_true',
        help="Enable L2 cache (default mode)")

    opt = main_parser.parse_known_args()
    args = opt[0]

    args.generated_dir = os.path.abspath(args.generated_dir)
    if not os.path.isdir(args.generated_dir):
        logger.warning("Targeted generated_dir to run must be a DIR")
        raise RuntimeError
    if not os.path.isfile(os.path.join(args.generated_dir, "network.dump.yaml")):
        logger.warning("Targeted generated_dir DIR to run must content the configuration dump file (network.dump.yaml)")
        raise RuntimeError

    if hasattr(args, "force_hw") and args.force_hw is not None:
        args.force_hw = args.force_hw.split(",")
        assert len(args.force_hw) == 2
        assert isinstance(args.force_hw, (tuple, list))
        if os.path.isdir(args.generated_dir):

            # get window size to apply
            h, w = [int(d) for d in args.force_hw]
            # build new networks with input size
            mDir = args.generated_dir + f"_{h}x{w}"
            genDir = mDir + "_gen"
            os.makedirs(mDir, exist_ok=True)
            shutil.copytree(args.generated_dir, mDir, dirs_exist_ok=True)
            cYamlPath = build_new_model(args.generated_dir, (h, w), mDir)
            # Then, generate with KaNN
            cmd_args = ["kann", "generate", cYamlPath, "-d", genDir, "--force"]
            logger.info("Running: {}".format(" ".join(cmd_args)))
            subprocess.run(cmd_args, check=True)
            args.generated_dir = genDir
            logger.info('** Done ** \n')

    if args.disable_l2_cache:
        args.enable_l2_cache = False
    if args.disable_l2_cache:
        os.environ["POCL_MPPA_FIRMWARE_NAME"] = "ocl_fw_l1.elf"
    elif args.enable_l2_cache:
        os.environ["POCL_MPPA_FIRMWARE_NAME"] = "ocl_fw_l2_d_1m.elf"

    with open(os.path.join(args.generated_dir, "network.dump.yaml"), 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.Loader)
        generate_options = cfg.get('generate_options')
        if generate_options is not None:
            data_buffer_size = generate_options.get('data_buffer_size', 6240000)
            if 7600000 > data_buffer_size > 6240000 and os.environ["POCL_MPPA_FIRMWARE_NAME"] != "ocl_fw_l1.elf":
                msg = f"\n" + "*" * 50
                msg += f"\n! Data_buffer_size is set to {data_buffer_size:,} B "
                msg += f"\n  which is not working with actual firmware"
                msg += f"\n  Please select firmware ocl_fw_l1.elf to disable L2 cache."
                msg += f"\n  Setting POCL_MPPA_FIRMWARE_NAME=ocl_fw_l1.elf"
                msg += f"\n  to run inference on MPPA hardware"
                msg += f"\n" + "*" * 50
                logger.warning(msg)
                os.environ["POCL_MPPA_FIRMWARE_NAME"] = "ocl_fw_l1.elf"
            elif data_buffer_size > 7600000:
                logger.warning(f"Data_buffer_size is set to {data_buffer_size} which")
                logger.warning(f"is not optimal for Coolidge2 (SMEM:8MB / kvx-clusters)")

    if args.bin_file is None:
        if os.environ.get('KALRAY_TOOLCHAIN_DIR', None) is not None:
            args.bin_file = os.path.join(os.environ.get('KALRAY_TOOLCHAIN_DIR'), "bin", "kann_opencl_cnn")
        else:
            logger.warning("KALRAY_TOOLCHAIN_DIR is not set, please source Kalray toolchain first")
            raise RuntimeError
    if args.pocl_dir is None:
        if os.environ.get('KALRAY_TOOLCHAIN_DIR', None) is not None:
            toolchain_dir = os.environ.get('KALRAY_TOOLCHAIN_DIR')
            args.pocl_dir = os.path.join(toolchain_dir, f"kvx-cos/lib/kv3-2/KAF/services/")
        else:
            logger.warning("KALRAY_TOOLCHAIN_DIR is not set, please source Kalray toolchain first")
            raise RuntimeError
    else:
        args.pocl_dir = os.path.abspath(args.pocl_dir)
    pocl_file = os.path.join(args.pocl_dir, "mppa_kann_opencl.cl.pocl")

    if "device" in args and args.device == "mppa":
        eval_env(args.bin_file, pocl_file, "kv3-2")

    print(opt)
    args.func(opt)
