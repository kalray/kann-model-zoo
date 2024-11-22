import os
import re
import glob
import shutil
import subprocess
import threading
from collections import OrderedDict


PASS = "\U00002705"
FAIL = "\U0000274C"
DEMO_SOURCE = "cat.jpg"
WORKSPACE_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__ + "/../.."))))


def get_mppa_frequency():
    """ Return the frequencies set for the MPPA's clusters """
    mppa_freq_hz = []
    with open('/mppa/board0/mppa0/freq', 'r') as f_hw:
        for l in f_hw.readlines():
            txt = str(l).rstrip()
            mppa_freq_hz.append(float(txt.split(' ')[2]))
    return mppa_freq_hz


def get_env():
    toolchain_dir = os.environ.get("KALRAY_TOOLCHAIN_DIR")
    cmd = [os.path.join(toolchain_dir, "bin", "kvx-mppa"), "--version"]
    with open(".kvx-mppa.version", "w+") as f:
        subprocess.run(cmd, stdout=f, stderr=f)
    with open(".kvx-mppa.version", "r") as f:
        log = f.readlines()
    kvx_version = None
    for l in log:
        if "version" in l.lower():
            kvx_version = l.split("\t")[-1].rstrip()
    try:
        import kann
    except ImportError as err:
        pass
    kann_version = kann.__version__
    return kvx_version, kann_version


def write_csv_header(f):
    f.write(f"TestID;Family;NN_name;Framework;CFG file;")
    f.write(f"GENERATION;{PASS}/{FAIL};INFERENCE;{PASS}/{FAIL};PERF_HOST;PERF_MPPA;")
    f.write(f"DEMO_KVX;{PASS}/{FAIL};SCORE_KVX;PRED_KVX;")
    f.write(f"DEMO_CPU;{PASS}/{FAIL};SCORE_CPU;PRED_CPU;")
    f.write(f"\n")


def thread_generate(cfg, num, odir, res):
    test_result = {"gen": False, "run": False, "demo": False }
    nn_type = cfg.split(os.sep)[-4]
    nn_name = cfg.split(os.sep)[-3]
    nn_fwk = cfg.split(os.sep)[-2]
    yaml_file = os.path.basename(cfg).replace(".yaml", "")
    generated_path = os.path.join(odir, nn_type, f"{num}_{nn_name}_{nn_fwk}_{yaml_file}")
    if not os.path.exists(generated_path):
        try:
            cmd = [f"{WORKSPACE_PATH}/./generate"]
            args = [cfg, "-d", generated_path, "-f"]
            print(f"Generating ({num}) : {cfg}")
            os.makedirs(os.path.dirname(generated_path), exist_ok=True)
            with open(f"{generated_path}.log", 'w+') as flog:
                subprocess.run(cmd + args, stdout=flog, stderr=flog, check=True, timeout=15 * 60.)
            test_result["gen"] = True
        except Exception as err:
            with open(f"{generated_path}.log", "r") as flog:
                log_fail_text = flog.readlines()[-10:]
                test_result["gen"] = " ".join(log_fail_text)
    else:
        binfiles = [f for f in os.listdir(generated_path) if f.split('.')[-1] == 'kann']
        test_result["gen"] = len(binfiles) == 1
    res[num] = test_result


def check_generate(cfg, num, odir, res, fres):
    print(f"** Check kann-gener: {cfg[-50:]:50s} **", end=" ")
    nn_type = cfg.split(os.sep)[-4]
    nn_name = cfg.split(os.sep)[-3]
    nn_fwk = cfg.split(os.sep)[-2]
    yaml_file = os.path.basename(cfg).replace(".yaml", "")
    generated_path = os.path.join(odir, nn_type, f"{num}_{nn_name}_{nn_fwk}_{yaml_file}")
    generated_path = os.path.abspath(generated_path)
    try:
        fres.write(f"{num};{nn_type};{nn_name};{nn_fwk};{cfg};")
        if res[num]["gen"] is True:
            fres.write(f"PASS;{PASS};")
            print(f" > PASS {PASS}")
        else:
            fres.write(f"FAIL;{FAIL};")
            err = str(res[num]['gen']).replace('\n', ';')
            fres.write(f">> {err}\n")
            print(f" > FAIL {FAIL}")
            generated_path = None
    except:
        print(f"")
        pass
    finally:
        fres.flush()
    return generated_path


def check_inference(generated_path, odir, fres):
    try:

        print(f"** Check kann-infer: {generated_path[-50:]:50s} **", end=" ")
        cmd = [f"{WORKSPACE_PATH}/./run"]
        args = []
        if "kann_custom_layers" in os.environ.get('PYTHONPATH', ''):
            args = [f"--pocl-dir={WORKSPACE_PATH}/output/opencl_kernels"]
        args += ["infer", generated_path, "-n", "25"]
        log_file_path = f"{odir}/infer_{os.path.basename(generated_path)}.log"
        with open(log_file_path, "w+") as flog:
            subprocess.run(cmd + args, stdout=flog, stderr=flog, check=True, timeout=60.)
        fres.write(f"PASS;{PASS};")
        print(f" > PASS {PASS}", end="\t")
    except Exception as err:
        fres.write(f"FAIL;{FAIL}")
        err = str(err).replace('\n', ';')
        fres.write(f";;;;>> {err}\n")
        print(f" > FAIL {FAIL}")
    finally:
        flogs = [f for f in os.listdir() if '.log' == f[-4:]]
        [shutil.copy(f, odir) for f in flogs]
        [os.remove(f) for f in flogs]
        fres.flush()
    if len(flogs) == 0:
        return
    return os.path.join(odir, flogs[0])


def check_demo(generated_path, odir, fres, device="mppa"):
    try:
        print(f"** Check kann-demo : {generated_path[-50:]:50s} **", end=" ")
        # Check if input pre-post process exist in nn package
        if not os.path.isfile(os.path.join(generated_path, "input_preparator.py")):
            fres.write(f"nc;;")
            print(f"\tScripts not found for pre-post proc [!] - ", end="")
            return
        cmd = [f"{WORKSPACE_PATH}/./run"]
        args = []
        if "kann_custom_layers" in os.environ.get('PYTHONPATH', ''):
            args = [f"--pocl-dir={WORKSPACE_PATH}/output/opencl_kernels"]
        args += ["demo", f"--device={device}", generated_path, f"{WORKSPACE_PATH}/utils/sources/{DEMO_SOURCE}"]
        args += ["--no-replay", "--no-display", "--verbose"]
        if device == "mppa":
            args += ["--save-img"]
        log_file_path = f"{odir}/demo_{device}_{os.path.basename(generated_path)}.log"
        with open(log_file_path, "w+") as flog:
            subprocess.run(cmd + args, stdout=flog, stderr=flog, check=True, timeout=60.)
        if os.path.exists(DEMO_SOURCE):
            shutil.move(DEMO_SOURCE, f"{odir}/{DEMO_SOURCE}_{os.path.basename(generated_path)}.jpg")
        fres.write(f"PASS;{PASS};")
        print(f" > PASS {PASS}", end="\t")
    except Exception as err:
        fres.write(f"FAIL;{FAIL};")
        print(f" > FAIL {FAIL}", end="\t")
        err = str(err).replace('\n', ';')
        fres.write(f";>> {err};")
        log_file_path = None
    finally:
        fres.flush()
    return log_file_path


def get_perf_from_log(log_path):
    MPPA_KV3_2_FREQ_HZ = get_mppa_frequency()[0]
    results = {}
    with open(os.path.join(log_path), 'r') as ilog:
        inference_log = ilog.readlines()
    qps_host = []
    qps_mppa = 0.
    total_cycles = 1
    for line in inference_log:
        if '[host] Performance of frame' in line:
            perf_host = re.sub("[^0-9.]", "", line.split(':')[-1].split('-')[0])
            qps_host.append(float(perf_host))
        if 'are required for a single process_frame' in line:
            total_cycles = int(re.sub("[^0-9]", "", line.split('cycles')[0]))
            break
    if len(qps_host) > 0:
        qps_host = 1e3 * len(qps_host) / sum(qps_host)
        qps_mppa = MPPA_KV3_2_FREQ_HZ / total_cycles
    else:
        qps_host = 0.0
        qps_mppa = 0.0
    results['host'] = qps_host
    results['mppa'] = qps_mppa
    return results


def get_prediction_demo(log_path):
    s, p = 0., "nc"
    try:
        if log_path is not None:
            with open(os.path.join(log_path), 'r') as ilog:
                demo_log = ilog.readlines()
            r = [l.removesuffix('\n') for l in demo_log if "prediction" in l.lower()]
            r = r[-1].split(':')[-1]
            s, p = r.split(" - ")
            p = p.removesuffix("\x1b[0;0m")
    except:
        pass
    return float(s), str(p)


def run_tests(nn_types, wspace, datatypes='f16'):

    kvx_ver, knn_ver = get_env()
    networks_path = os.path.join(WORKSPACE_PATH, "networks")
    wspace += f"_kvx_{kvx_ver[0:12]}_knn_{knn_ver}"
    write_mode = "w+"

    if os.path.exists(wspace):
        shutil.rmtree(wspace)
        os.makedirs(wspace, exist_ok=True)

    gen_dir = os.path.join(wspace, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    images_dir = os.path.join(wspace, "saved_images")
    os.makedirs(images_dir, exist_ok=True)
    logs_dir = os.path.join(wspace, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    fresults = open(f"{wspace}_report.csv", write_mode)
    write_csv_header(fresults)
    results = OrderedDict()

    if nn_types == ["all"]:
        nn_types = os.listdir(networks_path)
    if not isinstance(datatypes, list):
        datatypes = [datatypes]

    try:
        for datatype in datatypes:
            for nn_type in nn_types:
                # Generate models
                list_of_files = sorted(glob.iglob(f"{networks_path}/{nn_type}/*/*/*_{datatype}.yaml", recursive=True))
                threads = list()
                for idx, cfgFile in enumerate(list_of_files):
                    test_id = f"{nn_type[:5]}-{idx:04d}{datatype}"
                    thr = threading.Thread(
                        target=thread_generate,
                        args=(cfgFile, test_id, gen_dir, results))
                    threads.append(thr)
                    thr.start()
                [t.join() for t in threads]  # barrier
                # Then check and run
                for idx, cfgFile in enumerate(list_of_files):
                    if nn_type in cfgFile:
                        test_id = f"{nn_type[:5]}-{idx:04d}{datatype}"
                        gen_path = check_generate(
                            cfgFile, test_id, gen_dir, results, fresults)
                        if gen_path:
                            infer_path = check_inference(gen_path, logs_dir, fresults)
                            if infer_path:
                                # Get perf from log
                                perf = get_perf_from_log(infer_path)
                                fresults.write(f"{perf['host']:.1f};{perf['mppa']:.1f};")
                                print(f"host: {perf['host']:6.1f} FPS\tmppa: {perf['mppa']:6.1f} FPS")
                                # Get predictions from MPPA inference
                                log_path = check_demo(gen_path, images_dir, fresults, "mppa")
                                score_kvx, pred_kvx = get_prediction_demo(log_path)
                                print(f"mppa_score:{score_kvx:4.3f}\tmppa_pred: {pred_kvx}")
                                fresults.write(f"{score_kvx:.3f};{pred_kvx};")
                                # Get predictions from CPU inference
                                log_path = check_demo(gen_path, images_dir, fresults, "cpu")
                                score_cpu, pred_cpu = get_prediction_demo(log_path)
                                print(f"cpu_score: {score_cpu:4.3f}\tcpu_pred:  {pred_cpu}")
                                fresults.write(f"{score_cpu:4.3f};{pred_cpu};\n")
    finally:
        fresults.close()


def main():

    # nn_list = ["all", "classifiers", "object-detection", "segmentation"]
    wpath = os.path.join(WORKSPACE_PATH, "valid", "jenkins", "single_tests")
    run_tests(["all"], wpath, datatypes=['i8', 'f16'])


if __name__ == "__main__":
    main()

