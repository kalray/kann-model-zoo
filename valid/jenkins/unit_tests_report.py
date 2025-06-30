#! /usr/bin/env python3

###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import re
import numpy
import shutil
import subprocess

from test_utils import logger
from test_utils import WORKSPACE_PATH
from mppa_utils import get_mppa_frequency
from collections import OrderedDict


KPI = [
    "test_id", "family", "name", "framework", "dtype", "cfg_file",
    "gen_status", "infer_status", "host_fps", "mppa_fps", "cycles",
    "demo_mppa_status", "score_mppa", "pred_mppa", "bbox_mppa",
    "demo_cpu_status",  "score_cpu",  "pred_cpu",  "bbox_cpu",
    "abs_err", "rel_err", "accuracy"
]


def write_csv_header(f, sep=","):
    for k in KPI:
        f.write(f"{k}{sep}")
    f.write(f"\n")


def check_generate(cfg, num, odir, res, fres=None, sep="+"):

    """ Check that GENERATION step is PASS or FAIL from a generated DIR by kaNN

        Pass/Fail criteria:
           + [PASS] if *.kann extension file is existing in DIR
           + [FAIL] otherwise

        Nota Bene:
        - the script write in pointer <fres> to test result file (CSV)
        - the script update the test result in <res> dict structure, such as
          res[testID] = {"gen": False, "run": False, "demo": False}
          where False <=> FAIL, True <=> PASS

        Return Str (the path of the generated DIR) if PASS
            or None if FAIL
    """
    nn_type = cfg.split(os.sep)[-4]
    nn_name = cfg.split(os.sep)[-3]
    nn_fwk = cfg.split(os.sep)[-2]
    yaml_file = os.path.basename(cfg).replace(".yaml", "")
    generated_path = os.path.join(odir, nn_type, f"{num}{sep}{nn_name}{sep}{nn_fwk}{sep}{yaml_file}")
    generated_path = os.path.abspath(generated_path)
    if num not in res:  # create data struc for result test
        res[num] = {"gen": False, "run": False, "demo": False}
    try:
        if fres is not None:
            fres.write(f"{num};{nn_type};{nn_name};{nn_fwk};{cfg};")
        kann_file = [f for f in os.listdir(generated_path) if f.split(".")[-1] == "kann"]
        kann_file_exists = len(kann_file) == 1
        if res[num]["gen"] or kann_file_exists:
            if fres is not None:
                fres.write(f"PASS;")
            logger.info(f"** Check kann-gener: {cfg[-50:]:50s} ** > PASS")
        else:
            if fres is not None:
                fres.write(f"FAIL;")
            err = str(res[num]['gen'])
            logger.info(f"** Check kann-gener: {cfg[-50:]:50s} ** > FAIL")
            logger.warning(f">> {err}\n")
            generated_path = None
    except Exception as stderr:
        logger.info(stderr)
        if fres is not None:
            fres.write(f"FAIL;")
        logger.info(f"** Check kann-gener: {cfg[-50:]:50s} ** > FAIL")
        logger.warning(f">> {stderr}\n")
        generated_path = None
        pass
    finally:
        if fres is not None:
            fres.flush()
    return generated_path


def check_generate_from_log(log_file):
    with open(log_file) as f:
        lines = f.readlines()
    for l in reversed(lines):
        if "KaNN compilation completed into:" in l:
            return True
    return False


def check_inference(generated_path, tid, odir, res, fres=None):

    """ Check that INFERENCE step is PASS or FAIL from a generated DIR by kaNN

            Pass/Fail criteria:
               + [PASS] if "run" script runs properly in a subprocess with "infer" sub-command
               + [FAIL] otherwise

            Nota Bene:
            - the script write in pointer <fres> to test result file (CSV)
            - the script update the test result in <res> dict structure, such as
              res[testID] = {"gen": False, "run": False, "demo": False}
              where False <=> FAIL, True <=> PASS

            Return Str (the path of the inference log) if PASS
                or None if FAIL
    """
    if generated_path is None:
        return None
    try:
        cmd = [f"{WORKSPACE_PATH}/./run"]
        args = []
        if "kann_custom_layers" in os.environ.get('PYTHONPATH', ''):
            args = [f"--pocl-dir={WORKSPACE_PATH}/output/opencl_kernels"]
        args += ["infer", generated_path, "-n", "25"]  # run on 25 frames
        log_file_path = f"{odir}/infer_{os.path.basename(generated_path)}.log"
        with open(log_file_path, "w+") as flog:
            subprocess.run(cmd + args, stdout=flog, stderr=flog, check=True, timeout=60.)
        res[tid]['run'] = True
        if fres is not None:
            fres.write(f"PASS;")
        logger.info(f"** Check kann-infer: {generated_path[-50:]:50s} ** > PASS")
    except Exception as err:
        res[tid]['run'] = False
        if fres is not None:
            fres.write(f"FAIL;")
        err = str(err)
        logger.warning(f">> {err}\n")
        logger.info(f"** Check kann-infer: {generated_path[-50:]:50s} ** > FAIL")
    finally:
        flogs = [f for f in os.listdir() if '.log' == f[-4:]]
        [shutil.copy(f, odir) for f in flogs]
        [os.remove(f) for f in flogs]
        shutil.rmtree(f"inputs_outputs_kann_{os.path.basename(generated_path)}", ignore_errors=True)
        if fres is not None:
            fres.flush()
    if len(flogs) == 0:
        return
    return log_file_path


def check_demo(generated_path, src, tid, odir, res, fres=None, device="mppa"):

    """ Check that DEMO is PASS or FAIL from a generated DIR by kaNN and pre/post-process scripts
        provided in kann-models-zoo, adding the options:
            --no-replay:     to avoid an infinite loop from *-video-demo.py
            --no-display:    to avoid cv2 display issues
            --verbose (-v):  to output predictions vectors
            --device (-d):   to infer on the CPU (cpu-video-demo.py) or MPPA (kann-video-demo)

            Pass/Fail criteria:
               + [PASS] if "run" script runs properly in a subprocess with "demo" sub-command
               + [FAIL] otherwise

            Nota Bene:
            - the script write in pointer <fres> to test result file (CSV)
            - the script update the test result in <res> dict structure, such as
              res[testID] = {"gen": False, "run": False, "demo": False}
              where False <=> FAIL, True <=> PASS

            Return Str (the path of the generated DIR) if PASS
                or None if FAIL
    """
    if generated_path is None:
        return None
    try:
        # Check if input pre-post process exist in nn package
        if not os.path.isfile(os.path.join(generated_path, "input_preparator.py")):
            if fres is not None:
                fres.write(f"nc;")
            logger.info(f"> Scripts not found for pre-post proc [!]")
            return
        cmd = [f"{WORKSPACE_PATH}/./run"]
        args = []
        if "kann_custom_layers" in os.environ.get('PYTHONPATH', ''):
            args = [f"--pocl-dir={WORKSPACE_PATH}/output/opencl_kernels"]
        args += ["demo", f"--device={device}", generated_path, src]
        args += ["--no-replay", "--no-display", "--verbose"]
        if device == "mppa":
            args += ["--save-img"]
        log_file_path = f"{odir}/../logs/demo_{device}_{os.path.basename(generated_path)}.log"
        with open(log_file_path, "w+") as flog:
            subprocess.run(
                cmd + args,
                stdout=flog,
                stderr=flog,
                check=True,
                timeout=60.)
        src_file = os.path.basename(src)
        if os.path.exists(src_file):
            shutil.move(src_file, f"{odir}/{src_file}_{os.path.basename(generated_path)}.jpg")
        res[tid]["demo"] = True
        if fres is not None:
            fres.write(f"PASS;")
        logger.info(f"** Check kann-demo : {generated_path[-50:]:50s} ** > PASS")
    except Exception as err:
        res[tid]["demo"] = True
        if fres is not None:
            fres.write(f"FAIL;")
        logger.info(f"** Check kann-demo : {generated_path[-50:]:50s} ** > FAIL")
        err = str(err)
        logger.warning(f">> {err}")
        log_file_path = None
    finally:
        if fres is not None:
            fres.flush()
    return log_file_path


def get_perf_from_log(log_path):

    """
        Provide the HOST and MPPA Performance from a log given by
        kann_opencl_cnn and kann environment. Example:

        from the lines:
            [app][host] Performance of frame 1: 1.85006 ms - 540.522 fps
            [app][host] Performance of frame 2: 1.82093 ms - 549.17 fps
            [app][host] Performance of frame 3: 1.82547 ms - 547.805 fps
            [app][host] Performance of frame 4: 1.8076 ms - 553.219 fps
            [app][host] Performance of frame 5: 1.83054 ms - 546.288 fps
        -> get the list : perf_host = [1.85006, 1.82093, 1.82547, 1.8076, 1.83054]
        -> compute the average of the list: perf["host"] = 1e3 * len(perf_host) / sum(perf_host) in QPS

        > would give you the mean performance host in Queries Per Seconds

        from the lines:
            Total: 1839776 cycles are required for a single process_frame (result averaged over 10 frames).
        -> perf["mppa"] = get_mppa_frequency()[0] / 1839776

        > would give you the mean performance MPPA in Queries Per Seconds,
          considering: the MPPA frequency does not change and ALL clusters have the same CLK frequency

        Return perf dict
    """
    results = {}
    if log_path is None:
        results['host'] = 0.
        results['mppa'] = 0.
        results['cycles'] = 0.
        return results
    with open(os.path.join(log_path), 'r') as ilog:
        inference_log = ilog.readlines()
    perf_host = []
    mean_cycles = 1
    for line in inference_log:
        if '[host] Performance of frame' in line:
            perf = re.sub("[^0-9.]", "", line.split(':')[-1].split('-')[0])
            perf_host += [float(perf)]
        if 'are required for a single process_frame' in line:
            mean_cycles = int(re.sub("[^0-9]", "", line.split('cycles')[0]))
            break
    if len(perf_host) > 0:
        try:
            freq = get_mppa_frequency()[0]
        except:
            freq = 1e9
        qps_host = 1e3 * len(perf_host) / sum(perf_host)
        qps_mppa = freq / mean_cycles
    else:
        qps_host = 0.0
        qps_mppa = 0.0
    results['host'] = qps_host
    results['mppa'] = qps_mppa
    results['cycles'] = mean_cycles
    results['nb_frames'] = len(perf_host)
    return results


def get_acc_from_log(log_path):

    """
        Provide the REL_ERR and ABS_ERR from a log given by
        kann run <model> --check comands. Example:

        from the lines:
            errors:
                relative: 1.9963
                absolute: 0.0157
                std: 0.1529
                mean: 0.0021
                score f1: 0.0004
                (max_rel, abs): (1.9963,0.0157)
                (rel, max_abs): (0.0384,19.9502)

        -> get the error "relative" and "absolute"
        Return acc dict
    """
    results = {}
    results['abs_err'] = None
    results['rel_err'] = None

    if log_path is None:
        return results
    with open(os.path.join(log_path), 'r') as ilog:
        inference_log = ilog.readlines()
    for line in reversed(inference_log):
        if "errors: " in line:
            return results
        if 'relative:' in line:
            err_rel = re.sub("[^0-9.]", "", line.split(': ')[-1])
            results['rel_err'] = float(err_rel)
        if 'absolute:' in line:
            err_abs = re.sub("[^0-9.]", "", line.split(': ')[-1])
            results['abs_err'] = float(err_abs)
    return results


def get_prediction_demo(log_path):
    """
        Provide the prediction from a *-video-demo script from LOG.
        Example:
             ...
             [KaNN Demo] Opening input fifo for CNN's input : 'images'
             [KaNN Demo] Opening output fifo for CNN's output : 'output'
             >> [Post-proc] prediction: 0.417 - n02123045 tabby, tabby cat
             [KaNN Demo] frame:2/0  read: 0.03ms    pre: 1.65ms     send: 0.20ms    kann: 1.94ms  ....
             ...
        Would stop on "prediction" word and return the s = score and p = predictions
        Return list((float), ((str, [x1, y1, x2, y2]))
    """
    score, pred_str, bbox = 0., "nc", [0.] * 4
    results = []
    tag = ">> [Post-proc] prediction:"
    if log_path is not None:
        try:
            with open(os.path.join(log_path), 'r') as ilog:
                demo_log = ilog.readlines()
            for l in demo_log:
                if tag in l:
                    r = l.removesuffix('\x1b[0;0m\n')
                    r = r.split(tag)[-1]
                    score = float(r.split(" - ")[0])
                    pred_str = r.split(" - ")[1]
                    if len(r.split(" - ")) == 3:
                        a = r.split(" - ")[-1]
                        if len(a.split(',')) == 4:
                            bbox = [float(re.sub("[^0-9.]", "", x)) for x in a.split(',')]
                    results.append((score, pred_str, bbox))
        except:
            print(f"[!] Warning: format prediction is issuing, expect {tag}: score - cls name (- bbox[x1, y1, x2, y2])")
    if len(results) == 0:
        results.append((score, pred_str, bbox))  # return at list empty data struct
    return results


def get_iou(bbox_a, bbox_b):

    """
        Return the intersection over Union of 2 bounding boxes:
        y = f(a[x1, y1, x2, y2], b[x1, y1, x2, y2])
    """

    x_left   = max(bbox_a[0], bbox_b[0])
    y_top    = max(bbox_a[1], bbox_b[1])
    x_right  = min(bbox_a[2], bbox_b[2])
    y_bottom = min(bbox_a[3], bbox_b[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_A = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_B = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union_area = area_A + area_B - intersection_area
    if union_area > 0.:
        iou = intersection_area / union_area
    else:
        iou = 0
    return iou

def get_miou(bboxes_a, bboxes_b):

    """
        Return the mean intersection over Union of 2 list of bounding boxes:
        y = f(a[x1, y1, x2, y2], b[x1, y1, x2, y2])
    """

    ious = []
    for a in bboxes_a:
        # get the best ious from bboxes_b with a
        _ious = [get_iou(a, b) for b in bboxes_b]
        ious.append(max(_ious))
    miou = sum(ious) / len(ious)
    return miou


def report(logs_dir, report_dir, report_file_name="report.csv", sep='+', perffile=True):

    # init CSV file
    csv_sep = ";"
    os.makedirs(report_dir, exist_ok=True)
    report_file_path = os.path.join(report_dir, report_file_name)
    fresults = open(f"{report_file_path}", "w+")
    write_csv_header(fresults, csv_sep)

    # parse logs and reports
    results = OrderedDict()

    # Determine the log files by test id - 1st loop
    for filename in os.listdir(logs_dir):
        if os.path.isfile(os.path.join(logs_dir, filename)) and filename.split(".")[-1] == "log":
            flog = os.path.join(logs_dir, filename)

            # Check if filename can be investigate
            if len(filename.split(sep)) <= 1:
                continue

            # Get test name, type and backend device, FORMAT is SENSITIVE
            test_id = filename.split(sep)[0].split("_")[-1]
            test_nn_name = filename.split(sep)[-1].replace(".log", "")
            if "network_" in test_nn_name:
                test_nn_name = filename.split(sep)[1]
            test_cat = test_id.split("-")[0]

            # Report by test_id (must be unique)
            if test_id not in results:
                results[test_id] = dict()
            results[test_id]["family"] = test_cat
            results[test_id]["name"] = test_nn_name

            # Get test_id's dtype
            if "f16" in test_id:
                dtype = "fp16"
            elif "i8" in test_id:
                dtype = "int8"
            else:
                dtype = "nc"
            results[test_id]["dtype"] = dtype

            if filename.startswith(test_id):  # == generation log file
                with open(flog) as f:
                    first_line = f.readlines()[0]
                    cfg_file = first_line.split(" ")[2]
                    results[test_id]["cfg_file"] = cfg_file
                    results[test_id]["gen_log"] = flog
            elif filename.startswith("demo"):
                test_device = filename.split(sep)[0].split("_")[1]
                results[test_id][f"demo_{test_device}_log"] = flog
            elif filename.startswith("inference"):
                results[test_id][f"infer_log"] = flog

    # Parse log generated files by "test id" - 2nd loop
    for test_id in results:

        # Get compilation status
        results[test_id]["gen_status"] = check_generate_from_log(results[test_id]['gen_log'])
        if not results[test_id]["gen_status"] or 'infer_log' not in results[test_id]:
            continue
        # Get performance status
        perf = get_perf_from_log(results[test_id]['infer_log'])
        results[test_id]["host_fps"] = perf['host']
        results[test_id]["mppa_fps"] = perf['mppa']
        results[test_id]["cycles"] = perf['cycles']
        if perf['host'] == 0. or perf['mppa'] == 0.:
            results[test_id]["infer_status"] = False
            continue
        else:
            results[test_id]["infer_status"] = True

        # Get kann utils diff results from inference log
        acc = get_acc_from_log(results[test_id]['infer_log'])
        results[test_id]["abs_err"] = acc["abs_err"]
        results[test_id]["rel_err"] = acc["rel_err"]

        # Get score and prediction status
        if "demo_mppa_log" in results[test_id]:
            pred_kvx = get_prediction_demo(results[test_id]['demo_mppa_log'])
        else:
            pred_kvx = [(0., "nc", [0.] * 4)]
        if "demo_cpu_log" in results[test_id]:
            pred_cpu = get_prediction_demo(results[test_id]['demo_cpu_log'])
        else:
            pred_cpu = [(0., "nc", [0.] * 4)]
        # only the first vectors (best predictions) is reported
        p = pred_kvx[0]
        results[test_id]["score_mppa"] = p[0]
        results[test_id]["pred_mppa"] = p[1]
        results[test_id]["bbox_mppa"] = None if p[2] == [0.] * 4 else p[2]
        results[test_id]["demo_mppa_status"] = not(p[0] == 1. or p[0] == 0.)
        p = pred_cpu[0]
        results[test_id]["score_cpu"] = p[0]
        results[test_id]["pred_cpu"] = p[1]
        results[test_id]["bbox_cpu"] = None if p[2] == [0.] * 4 else p[2]
        results[test_id]["demo_cpu_status"] = not(p[0] == 1. or p[0] == 0.)

        # Compute accuracy with cpu computation
        if test_id.startswith("class"):
            scores_kvx = numpy.array([p[0] for p in pred_kvx])
            scores_cpu = numpy.array([p[0] for p in pred_cpu])
            # mean score relative diffs
            results[test_id]["accuracy"] = 1 - numpy.mean(numpy.abs(scores_kvx - scores_cpu) / scores_cpu)
        elif test_id.startswith("objec"):
            bbx_kvx = numpy.array([p[2] for p in pred_kvx])
            bbx_cpu = numpy.array([p[2] for p in pred_cpu])
            results[test_id]["accuracy"] = get_miou(bbx_kvx, bbx_cpu)  # mean intersection over union
        else:
            results[test_id]["accuracy"] = numpy.nan

    # report to CSV file
    for test_id in sorted(results):
        fresults.write(f"{test_id}{csv_sep}")
        for k in KPI[1:]:
            if k in results[test_id]:
                fresults.write(f"{results[test_id][k]}{csv_sep}")
            else:
                fresults.write(f"{csv_sep}")
        fresults.write("\n")
    results["path"] = os.path.realpath(report_file_path)
    fresults.close()
    print(f"CSV Report has been saved to {results['path']}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="unit-tests-report")
    parser.add_argument(
        "log_dir", type=str,
        help="Specify the DIR path of logs to parse")
    parser.add_argument(
        "report_dir", type=str,
        help="Specify the DIR path to report files")
    args = parser.parse_args()
    report(args.log_dir, args.report_dir)
