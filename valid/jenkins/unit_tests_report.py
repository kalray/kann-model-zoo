import os
import re
import shutil
import subprocess

from test_utils import logger
from test_utils import WORKSPACE_PATH
from mppa_utils import get_mppa_frequency

SOURCES = ["dog.jpg"]
SRC_PATH = f"./utils/sources"


def write_csv_header(f):
    f.write(f"TestID;Family;NN_name;Framework;CFG file;")
    f.write(f"GENER;INFER;PERF_HOST;PERF_MPPA;")
    f.write(f"DEMO_KVX;SCORE_KVX;PRED_KVX;")
    f.write(f"DEMO_CPU;SCORE_CPU;PRED_CPU;")
    f.write(f"\n")


def check_generate(cfg, num, odir, res, fres):

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
    generated_path = os.path.join(odir, nn_type, f"{num}.{nn_name}.{nn_fwk}.{yaml_file}")
    generated_path = os.path.abspath(generated_path)
    if num not in res:  # create data struc for result test
        res[num] = {"gen": False, "run": False, "demo": False}
    try:
        fres.write(f"{num};{nn_type};{nn_name};{nn_fwk};{cfg};")
        kann_file = [f for f in os.listdir(generated_path) if f.split(".")[-1] == "kann"]
        kann_file_exists = len(kann_file) == 1
        if res[num]["gen"] or kann_file_exists:
            fres.write(f"PASS;")
            logger.info(f"** Check kann-gener: {cfg[-50:]:50s} ** > PASS")
        else:
            fres.write(f"FAIL;")
            err = str(res[num]['gen'])
            logger.info(f"** Check kann-gener: {cfg[-50:]:50s} ** > FAIL")
            logger.warning(f">> {err}\n")
            generated_path = None
    except Exception as stderr:
        logger.info(stderr)
        fres.write(f"FAIL;")
        logger.info(f"** Check kann-gener: {cfg[-50:]:50s} ** > FAIL")
        logger.warning(f">> {stderr}\n")
        generated_path = None
        pass
    finally:
        fres.flush()
    return generated_path


def check_inference(generated_path, tid, odir, res, fres):

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
        fres.write(f"PASS;")
        logger.info(f"** Check kann-infer: {generated_path[-50:]:50s} ** > PASS")
    except Exception as err:
        res[tid]['run'] = False
        fres.write(f"FAIL;")
        err = str(err)
        logger.warning(f">> {err}\n")
        logger.info(f"** Check kann-infer: {generated_path[-50:]:50s} ** > FAIL")
    finally:
        flogs = [f for f in os.listdir() if '.log' == f[-4:]]
        [shutil.copy(f, odir) for f in flogs]
        [os.remove(f) for f in flogs]
        shutil.rmtree(f"inputs_outputs_kann_{os.path.basename(generated_path)}", ignore_errors=True)
        fres.flush()
    if len(flogs) == 0:
        return
    return log_file_path


def check_demo(generated_path, tid, odir, res, fres, device="mppa"):

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
            fres.write(f"nc;")
            logger.info(f"> Scripts not found for pre-post proc [!]")
            return
        cmd = [f"{WORKSPACE_PATH}/./run"]
        args = []
        if "kann_custom_layers" in os.environ.get('PYTHONPATH', ''):
            args = [f"--pocl-dir={WORKSPACE_PATH}/output/opencl_kernels"]
        args += ["demo", f"--device={device}", generated_path, f"{WORKSPACE_PATH}/utils/sources/{SOURCES[0]}"]
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
        if os.path.exists(SOURCES[0]):
            shutil.move(SOURCES[0], f"{odir}/{SOURCES[0]}_{os.path.basename(generated_path)}.jpg")
        res[tid]["demo"] = True
        fres.write(f"PASS;")
        logger.info(f"** Check kann-demo : {generated_path[-50:]:50s} ** > PASS")
    except Exception as err:
        res[tid]["demo"] = True
        fres.write(f"FAIL;")
        logger.info(f"** Check kann-demo : {generated_path[-50:]:50s} ** > FAIL")
        err = str(err)
        logger.warning(f">> {err}")
        log_file_path = None
    finally:
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
        return results
    with open(os.path.join(log_path), 'r') as ilog:
        inference_log = ilog.readlines()
    perf_host = []
    total_cycles = 1
    for line in inference_log:
        if '[host] Performance of frame' in line:
            perf = re.sub("[^0-9.]", "", line.split(':')[-1].split('-')[0])
            perf_host += [float(perf)]
        if 'are required for a single process_frame' in line:
            total_cycles = int(re.sub("[^0-9]", "", line.split('cycles')[0]))
            break
    if len(perf_host) > 0:
        qps_host = 1e3 * len(perf_host) / sum(perf_host)
        qps_mppa = get_mppa_frequency()[0] / total_cycles
    else:
        qps_host = 0.0
        qps_mppa = 0.0
    results['host'] = qps_host
    results['mppa'] = qps_mppa
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
        Would stop on "prediction" word and return the s = score and p = prediction
        Return s (float) and p (str)
    """
    score, pred, bbox = 0., "nc", [0.]*4
    tag = ">> [Post-proc] prediction:"
    if log_path is not None:
        try:
            with open(os.path.join(log_path), 'r') as ilog:
                demo_log = ilog.readlines()
            r = [l.removesuffix('\n') for l in demo_log if tag in l][-1]
            r = r.removesuffix("\x1b[0;0m")
            r = r.split(tag)[-1]
            score = float(r.split(" - ")[0])
            pred = r.split(" - ")[1]
            if len(r.split(" - ")) == 3:
                a = r.split(" - ")[-1]
                if len(a.split(',')) == 4:
                    bbox = [float(re.sub("[^0-9.]", "", x)) for x in a.split(',')]
        except:
            print(f"[!] Warning: format prediction is issuing, expect {tag}: score - cls name (- bbox[x1, y1, x2, y2])")
    return score, [pred, bbox]
