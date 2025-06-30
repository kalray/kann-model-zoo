#! /usr/bin/env python3

###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import glob
import shutil
import argparse

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from kann_utils import thread_generate
from unit_tests_report import check_generate
from unit_tests_report import check_inference
from unit_tests_report import check_demo
from unit_tests_report import get_perf_from_log
from unit_tests_report import get_prediction_demo
from unit_tests_report import report
from mppa_utils import get_sw_kenv

from test_utils import logger
from test_utils import WORKSPACE_PATH
from test_utils import SOURCES


# run_tests(types, build_dir, model_to_include, dtypes)
def run_tests(list_of_files, build_dir, build=True, run=True):

    if build and os.path.exists(build_dir):
        # shutil.rmtree(build_dir)
        os.makedirs(build_dir, exist_ok=True)

    gen_dir = os.path.join(build_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    images_dir = os.path.join(build_dir, "saved_images")
    os.makedirs(images_dir, exist_ok=True)
    logs_dir = os.path.join(build_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    results = OrderedDict()

    print("\nList of tests to execute:")
    [print(f"  - {i:03d}: {f}") for i, f in enumerate(list_of_files)]
    print("--")

    # Limit cpu usage using ThreadPoolExecutor
    CPU_COUNT = os.cpu_count()
    CPU_WORKER = max(CPU_COUNT // 2, 1)
    # --

    try:
        # Generate models
        if build:
            with ThreadPoolExecutor(max_workers=CPU_WORKER) as executor:
                for idx, cfgFile in enumerate(list_of_files):
                    # Generate test ID
                    nn = cfgFile.split("/")[-4]
                    dtype = cfgFile.split("_")[-1].replace(".yaml", "")
                    test_id = f"{nn[:5]}-{idx:04d}{dtype}"
                    # Generate
                    executor.submit(thread_generate, cfgFile, test_id, gen_dir, sep="+")

        # copy gen logs to logs DIR
        os.system(f"cp {gen_dir}/*/*.log {logs_dir}")

        # Then check and run
        if run:
            for idx, cfgFile in enumerate(list_of_files):
                # Generate test ID
                nn = cfgFile.split("/")[-4]
                dtype = cfgFile.split("_")[-1].replace(".yaml", "")
                test_id = f"{nn[:5]}-{idx:04d}{dtype}"
                # Check generation from build dir
                gen_path = check_generate(cfgFile, test_id, gen_dir, results, sep="+")
                # Check that generated model runs properly
                infer_path = check_inference(gen_path, test_id, logs_dir, results)
                # Get perf from inference log
                perf = get_perf_from_log(infer_path)
                logger.info(f"host: {perf['host']:6.1f} FPS\tmppa: {perf['mppa']:6.1f} FPS ({perf['cycles']:,} cycles)")

                # Get predictions from MPPA inference
                src_files = SOURCES[nn[:5]]
                for src in src_files:
                    log_path = check_demo(gen_path, src, test_id, images_dir, results, device="mppa")
                    pred_kvx = get_prediction_demo(log_path)
                    score, pred, bbox = pred_kvx[0]
                    logger.info(f"mppa_score:{score:4.3f}\tmppa_pred: {pred} - {bbox}")
                    # Get predictions from CPU inference
                    log_path = check_demo(gen_path, src, test_id, images_dir, results, device="cpu")
                    pred_cpu = get_prediction_demo(log_path)
                    score, pred, bbox = pred_cpu[0]
                    logger.info(f"mppa_score:{score:4.3f}\tmppa_pred: {pred} - {bbox}")

    finally:
        report(logs_dir, f'{logs_dir}/report', perffile=False)


def main(opt):
    wpath = os.path.join(WORKSPACE_PATH)
    net_path = os.path.join(wpath, "networks")
    if args.build_dir is None:
        kvx_ver, knn_ver = get_sw_kenv()
        build_dir = os.path.join(wpath, "valid", "jenkins", f"single_tests_kvx_{kvx_ver[0:12]}_knn_{knn_ver}")
    else:
        build_dir = os.path.realpath(args.build_dir)
    _build = opt.build_only or not opt.run_only
    _run = not opt.build_only
    if args.clean:
        list_to_rem = sorted(glob.iglob(f"{build_dir}", recursive=True))
        for f in list_to_rem:
            if os.path.isdir(f):
                shutil.rmtree(f, ignore_errors=True)
            else:
                os.remove(f)
        return
    types = os.listdir(f"{net_path}") if opt.category == "all" \
        else [opt.category]
    dtypes = opt.datatype
    models_to_include = opt.include
    models_to_exclude = opt.exclude

    models_to_run = []
    networks_path = os.path.join(WORKSPACE_PATH, "networks")

    # Define all models available in current NETWORK directory
    all_available_models = list()
    for nn in types:
        all_available_models += list(sorted(glob.iglob(f"{networks_path}/{nn}/*/*/*.yaml", recursive=True)))

    # Get the INCLUDED only
    if len(models_to_include) > 0:
        models_to_run = [
            m for m in all_available_models
                if m.split(os.path.sep)[-3] in models_to_include
        ]
    else:
        models_to_run = all_available_models

    # Remove the EXCLUDED models
    models_to_run = [m for m in models_to_run if m.split(os.path.sep)[-3] not in models_to_exclude]

    # Keep the DATAYPES only
    if not isinstance(dtypes, list):
        dtypes = [dtypes]
    for dtype in dtypes:
        models_to_run = [m for m in models_to_run if dtype in os.path.basename(m)]

    run_tests(models_to_run, build_dir, _build, _run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="single-tests")
    parser.add_argument(
        "--category", "-c", type=str,
        default="all",
        help='Type of Neural networks to test ("all", "classifiers", "object-detection", or "segmentation")')
    parser.add_argument(
        "--datatype", "-d", type=str, nargs='+',
        default=["f16"],
        help="Specify the computation inference datatype (f16 or i8)")
    parser.add_argument(
        "--include", "-i", type=str, nargs='+',
        default=list(),
        help="Specify the model you want to include to test, if 'None' all YAML in the repository will be generated")
    parser.add_argument(
        "--exclude", "-e", type=str, nargs='+',
        default=list(),
        help="Exclude networks to test. e.g. --exclude alexnet rcnn")
    parser.add_argument(
        "--build-dir", "-b", type=str,
        default=None,
        help="Specify the path to workspace path directory")
    parser.add_argument(
        "--build-only", default=False, action='store_true',
        help="use --build-only to build only the models with kann")
    parser.add_argument(
        "--run-only", default=False, action='store_true',
        help="use --run-only to run only the models with kann (models must be generated)")
    parser.add_argument(
        "--clean", default=False, action='store_true',
        help="remove generated files by this script")
    args = parser.parse_args()
    print(args)
    main(args)
