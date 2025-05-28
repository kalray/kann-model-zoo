import os
import glob
import shutil
import argparse
import threading

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from kann_utils import thread_generate
from unit_tests_report import check_generate
from unit_tests_report import check_inference
from unit_tests_report import check_demo
from unit_tests_report import get_perf_from_log
from unit_tests_report import get_prediction_demo
from mppa_utils import get_sw_kenv

from test_utils import logger
from test_utils import WORKSPACE_PATH


def write_csv_header(f):
    f.write(f"TestID;Family;NN_name;Framework;CFG file;")
    f.write(f"GENER;INFER;PERF_HOST;PERF_MPPA;")
    f.write(f"DEMO_KVX;SCORE_KVX;PRED_KVX;")
    f.write(f"DEMO_CPU;SCORE_CPU;PRED_CPU;")
    f.write(f"\n")


def run_tests(nn_types, build_dir, datatypes='f16', build=True, run=True):

    networks_path = os.path.join(WORKSPACE_PATH, "networks")
    write_mode = "w+"

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

    if nn_types == ["all"]:
        nn_types = os.listdir(networks_path)
    if not isinstance(datatypes, list):
        datatypes = [datatypes]

    # Limit cpu usage using Semaphore and ThreadPoolExecutor
    CPU_COUNT = os.cpu_count()
    CPU_WORKER = max(CPU_COUNT // 2, 1)
    # --

    for datatype in datatypes:
        for nn_type in nn_types:
            try:
                fresults = open(f"{build_dir}/{nn_type}_{datatype}_report.csv", write_mode)
                write_csv_header(fresults)
                list_of_files = sorted(glob.iglob(
                    f"{networks_path}/{nn_type}/*/*/*_{datatype}.yaml",
                    recursive=True))
                # Generate models
                if build:
                    with ThreadPoolExecutor(max_workers=CPU_WORKER) as executor:
                        for idx, cfgFile in enumerate(list_of_files):
                            # Generate test ID
                            test_id = f"{nn_type[:5]}-{idx:04d}{datatype}"
                            # Generate
                            executor.submit(thread_generate, cfgFile, test_id, gen_dir)
                # Then check and run
                if run:
                    for idx, cfgFile in enumerate(list_of_files):
                        if nn_type in cfgFile:
                            # Generate test ID
                            test_id = f"{nn_type[:5]}-{idx:04d}{datatype}"
                            # Check generation from build dir
                            gen_path = check_generate(cfgFile, test_id, gen_dir, results, fresults)
                            # Check that generated model runs properly
                            infer_path = check_inference(gen_path, test_id, logs_dir, results, fresults)
                            # Get perf from inference log
                            perf = get_perf_from_log(infer_path)
                            fresults.write(f"{perf['host']:.1f};{perf['mppa']:.1f};")
                            logger.info(f"host: {perf['host']:6.1f} FPS\tmppa: {perf['mppa']:6.1f} FPS")
                            # Get predictions from MPPA inference
                            log_path = check_demo(gen_path, test_id, images_dir, results, fresults, "mppa")
                            score_kvx, pred_kvx = get_prediction_demo(log_path)
                            logger.info(f"mppa_score:{score_kvx:4.3f}\tmppa_pred: {pred_kvx}")
                            fresults.write(f"{score_kvx:.3f};{pred_kvx};")
                            # Get predictions from CPU inference
                            log_path = check_demo(gen_path, test_id, images_dir, results, fresults, "cpu")
                            score_cpu, pred_cpu = get_prediction_demo(log_path)
                            logger.info(f"cpu_score: {score_cpu:4.3f}\tcpu_pred:  {pred_cpu}")
                            fresults.write(f"{score_cpu:4.3f};{pred_cpu};\n")
            finally:
                fresults.close()


def main(opt):

    wpath = os.path.join(WORKSPACE_PATH)
    if args.build_dir is None:
        kvx_ver, knn_ver = get_sw_kenv()
        build_dir = os.path.join(wpath, "valid", "jenkins", f"single_tests_kvx_{kvx_ver[0:12]}_knn_{knn_ver}")
    else:
        build_dir = os.path.realpath(args.build_dir)

    categories = opt.type.split(",")
    dtype = opt.datatype.split(",")
    _build = opt.build_only or not opt.run_only
    _run = not opt.build_only

    if args.clean:
        list_to_rem = sorted(glob.iglob(f"{build_dir}", recursive=True))
        for f in list_to_rem:
            if os.path.isdir(f):
                shutil.rmtree(f, ignore_errors=True)
            else:
                os.remove(f)
        exit(0)

    run_tests(categories, build_dir, dtype, _build, _run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="single tests")
    parser.add_argument(
        "--type", type=str,
        default="all",
        help='Type of Neural networks to test ("all", "classifiers", "object-detection", or "segmentation")')
    parser.add_argument(
        "--datatype", type=str,
        default="f16",
        help="Specify the computation inference datatype")
    parser.add_argument(
        "--build-dir", type=str,
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
