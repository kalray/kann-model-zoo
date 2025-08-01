###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
#
# Script inspired by
#   Inspired from : https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/validator.py
#
# Authors:
#   Quentin Muller, <qmuller@kalrayinc.com>
#   Daniel Angulo, <dangulo@kalrayinc.com>
###

import os
import re
import sys
import cv2
import time
import yaml
import json
import glob
import numpy
import psutil
import shutil
import argparse
import threading
import contextlib
import subprocess
import onnxruntime as rt

from tqdm import tqdm
from io import StringIO
from collections import OrderedDict

from kann_utils import logger
from kann_datasets import check_dataset
from kann_datasets import load_coco_references
from kann_datasets import load_imagenet_references
from kann_datasets import get_classes_id
from kann_metrics import EvalKaNN_Classify
from kann_metrics import EvalKaNN_ObjDetect


def load_preparators(gen_dir, config):
    """
    Load input and output preparator modules for neural network processing.
    Args:
        gen_dir (str): Path to the generated neural network folder
        config (dict): Configuration dictionary containing preparator paths
    Returns:
        tuple: (input_preparator, output_preparator) modules
    """
    # Prepare file paths
    input_prep_path = os.path.join(gen_dir, config['input_preparator'])
    output_prep_dir = os.path.join(gen_dir, config['output_preparator'])
    output_prep_path = os.path.join(output_prep_dir, "output_preparator.py")
    if not os.path.exists(input_prep_path):
        raise FileNotFoundError(f"Input preparator not found at {input_prep_path}")
    if not os.path.exists(output_prep_path):
        raise FileNotFoundError(f"Output preparator not found at {output_prep_path}")
    # Clean sys.path to avoid conflicts and ensure correct order
    for path in [gen_dir, output_prep_dir, os.path.dirname(input_prep_path)]:
        if path in sys.path:
            sys.path.remove(path)
    # Add paths in correct order (most specific first)
    sys.path.insert(0, os.path.dirname(input_prep_path))
    sys.path.insert(0, output_prep_dir)  # Output prep dir needs to be first for relative imports
    sys.path.insert(0, gen_dir)
    # Import preparators
    input_prep_name = os.path.splitext(os.path.basename(input_prep_path))[0]
    prepare = __import__(input_prep_name)
    output_preparator = __import__("output_preparator.output_preparator", fromlist=["output_preparator"])
    return prepare, output_preparator


def parse_results(flog):
    """
    Take a log file and check for the lines containing the word "predicion",
    which will contain the results in the form "conf - label - [x1, y1, x2, y2]".
    Finally, parse those lines and store results in a dictionary to be returned.

    Args:
        flog (list): a list of str, each str being a line from the postprocess log.
    """
    detections = []
    # Extract all prediction lines
    prediction_lines = [
        re.sub(r"\x1b\[[0-9;]*m", "", l.removesuffix("\n"))  # Clean ANSI codes
        for l in flog
        if "prediction" in l.lower()
    ]
    for line in prediction_lines:
        # Extract score, label, and bbox from line
        try:
            # Split into components (e.g., "0.43 - laptop - [x1, y1, x2, y2]")
            parts = line.split(":")[-1].strip().split(" - ")
            if len(parts) < 2 or len(parts) > 3:
                logger.warning("There was a malformed line while parsing results from the output preparator.")  # Skip malformed lines
            score = float(parts[0].strip())
            label = parts[1].strip()

            if len(parts) == 3:
                bbox = [float(coord.strip()) for coord in parts[2].strip("[]").split(",")]
                # Ensure bbox has 4 coordinates (x1, y1, x2, y2)
                if len(bbox) != 4:
                    logger.warning("There was a malformed bbox while parsing results from the output preparator.")  # Skip invalid bboxes
                detections.append((bbox, score, label))
            else:
                label = label.split(" ")[0]
                detections.append((score, label))
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse line: {line}\nError: {e}")
            continue
    return detections


def get_chunks_sizes(io_sizes, num_images, proportion_ram=0.33):
    """
    This method is called when host wants to prevent overloading
    their RAM's capacity. Depending on the number of images and
    host's RAM capacity, decide if to process all of them on one
    piece, or rather to divide in chunks. Return a list with the chunk sizes.

    Args:
        io_sizes (list): a list with one element for each output node, this element
            being the product of its dimensions. To be interpreted in bytes.
        num_images (int): the number of images in the dataset.
        proportion_ram (float): the maximum cap on the % of RAM to be used in chunking.
            This does not mean that x% of the RAM will be necessarily used, but that the
            chunking will be done such that the usage will not exceed x% of the RAM.
            By default, 0.33 works best.
    Returns:
        chunk_sizes (list): a list of ints containing the sizes of the chunks of images
            that can be loaded to memory without exceeding the proportion of RAM.
    """
    # Compute the total size for one image
    io_size_per_img = numpy.sum(io_sizes)
    io_size_per_img_mb = io_size_per_img / (1024 ** 2)  # From bytes to MB

    # Available memory for processing
    available_ram_mb = psutil.virtual_memory().total // (1024 ** 2)  # In Megabytes
    available_for_processing = proportion_ram * available_ram_mb

    # Compute maximum images that can fit in memory
    max_images_per_chunk = int(available_for_processing / io_size_per_img_mb)
    max_images_per_chunk = max(1, max_images_per_chunk)  # Ensure max_images is at least 1

    # Handle case where max_images is larger than total images (i.e. no chunk division)
    if max_images_per_chunk >= num_images:
        return [num_images], io_size_per_img_mb * num_images

    # Compute optimal number of chunks based on the memory constraint
    optimal_num_chunks = num_images // max_images_per_chunk + 1
    base_chunk_size = num_images // optimal_num_chunks
    tail = num_images - (base_chunk_size * (optimal_num_chunks - 1))
    chunks = [base_chunk_size] * (optimal_num_chunks - 1) + [tail]

    return chunks, io_size_per_img_mb * num_images


def post_proc_data(input_postproc, fn_postproc, cfg, img_shape, img_id, res, dev, conf_th, iou_th, locker=None, dbg=False):

        # OUTPUT PROCESSING STAGE
        dummy_frame = numpy.zeros(img_shape, dtype=numpy.int32)
        # TODO: once all networks have this method, remoce the conditional block
        #  to replace by "detect = output_preparator.post_process_eval"
        if hasattr(fn_postproc, "post_process_eval"):
            _, detections = fn_postproc.post_process_eval(
                cfg=cfg,
                frame=dummy_frame,
                nn_outputs=input_postproc,
                device=dev,
                conf_thres=conf_th,
                iou_thres=iou_th,
                dbg=False,
            )
        elif hasattr(fn_postproc, "post_process"):  # because not all networks have post_process_eval
            if locker is not None:
                locker.acquire()
            output_buffer = StringIO()
            with contextlib.redirect_stdout(output_buffer):
                fn_postproc.post_process(
                    cfg=cfg,
                    frame=dummy_frame,
                    nn_outputs=input_postproc,
                    device=dev,
                    conf_thres=conf_th,
                    iou_thres=iou_th,
                    dbg=True,
                )
            captured_output = output_buffer.getvalue()
            if locker is not None:
                locker.release()
            detections = parse_results(
                captured_output.splitlines()
            )  # Entries of the form (conf, label, (possibly bbox))
        else:
            raise RuntimeError("The call to 'output_process_eval' method from output_preparator.py does not exists")

        # RESULT STORING STAGE
        res[img_id] = {}
        for detection in detections:
            if len(detection) == 3:
                # Object detection: (bbox, conf, label)
                bbox, conf, label = detection
                detection_data = (conf, bbox)
                if dbg:
                    logger.info(f"id: {img_id} - label:{label:15s}, conf:{conf:1.4f}, xyxy:{bbox}")
            elif len(detection) == 2:
                # Classification: (conf, label)
                conf, label = detection
                detection_data = conf
                if dbg:
                    logger.info(f"id: {img_id} - label:{label:15s}, conf:{conf:1.4f}")
            else:
                logger.warning(f"Unexpected detection format: {detection}")
                continue

            if locker is not None:
                locker.acquire()
            if label in res[img_id]:
                res[img_id][label].append(detection_data)
            else:
                res[img_id][label] = [detection_data]
            if locker is not None:
                locker.release()


def cpu_video_pipeline(sess, fn_preproc, fn_postproc, img_path, cfg, res, dev, t, conf_th, iou_th, lock, dbg):

        # INPUT PROCESSING STAGE
        t_start = time.perf_counter()
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        content = cv2.imread(img_path)
        image_shape = content.shape
        prepared = fn_preproc.prepare_img(content)
        prepared = prepared.reshape(cfg["input_nodes_shape"][0])
        prepared = prepared.transpose(cfg["input_nodes_dformat"][0])
        inputs = {cfg["input_nodes_name"][0]: prepared}
        t['t_prep'].append(1e3 * (time.perf_counter() - t_start))

        # INFERENCE STAGE
        t_start = time.perf_counter()
        preds = sess.run(None, inputs)
        t['t_infe'].append(1e3 * (time.perf_counter() - t_start))

        # OUTPUT PROCESSING STAGE
        t_start = time.perf_counter()
        outputs_name = [o.name for o in sess.get_outputs()]
        input_postproc = {k: o for o, k in zip(preds, outputs_name)}
        #input_postproc, fn_postproc, cfg, img_shape, img_id, res, dev, conf_th, iou_th, locker=None, dbg=False
        post_proc_data(input_postproc, fn_postproc, cfg, image_shape, image_id, res, dev, conf_th, iou_th, lock, dbg)
        t['t_post'].append(1e3 * (time.perf_counter() - t_start))


def run(gen_dir, dataset_img_path, device="mppa", ratio_ram=0.33, debug=False):
    """
    Based on mppa or cpu, execute the network runtime on all of the images
    of the dataset, then capture the printed outputs by the postprocess()
    function from output_preparator.py and store them on a results dict.

    Args:
        gen_dir (str): a path to the generated neural network folder.
        dataset_img_path (str): the path of the dataset image to be used.
        device (str): the device that will execute the inference stage.
        ratio_ram (float): the maximum cap of RAM to be used in chunking.
          This parameter is not effective if device is "cpu".
        debug: (bool) print label, conf and bbox parsed from output proc

    Returns:
        results (OrderedDict): nested dict, for every image, contains its inference
            results. The value is also a dictionary containing, for every label, the
            detections in the form of a list of tuples (conf, [bounding_box]).
    """

    # Load images and prepare the environment
    image_paths = []
    image_ext = ["jpg", "jpeg", "JPEG"]
    for extension in image_ext:
        image_paths.extend(glob.iglob(f"{dataset_img_path}/**/*.{extension}", recursive=True))
    image_paths = sorted(image_paths)
    if len(image_paths) == 0:
        raise FileNotFoundError(f'File not found at this path {dataset_img_path}')

    # Parse data from .yaml file
    yaml_config = [f for f in os.listdir(gen_dir) if f.endswith(".yaml")][0]
    if not yaml_config:
        raise FileNotFoundError("yaml file does not exist in gen_dir.")
    with open(os.path.join(gen_dir, yaml_config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Parse data from .json file
    json_config = [f for f in os.listdir(gen_dir) if f.endswith(".json")][0]
    if not json_config:
        raise FileNotFoundError("json file does not exist in gen_dir.")
    with open(os.path.join(gen_dir, json_config), "r") as j:
        io = json.load(j)
        for key in io:
            if not config[key]:
                config[key] = io[key]

    extra_data = config["extra_data"]
    config["classes_file"] = os.path.join(gen_dir, extra_data["classes"])

    with open(config["classes_file"], "r") as f:  # Import classes dict, necessary for post_process
        config["classes"] = f.readlines()
    config["input_preparator"] = extra_data["input_preparators"][0]  # Support model with 1 input only
    config["output_preparator"] = extra_data["output_preparator"]  # Support model with 1 output only

    if not "input_nodes_dtype" in config:
        config["input_nodes_dtype"] = ["float32"] * len(config["input_nodes_name"])
    if not "output_nodes_dtype" in config:
        config["output_nodes_dtype"] = ["float32"] * len(config["output_nodes_name"])

    # Delete leading '/' for output_node (does not allow for path joining)
    output_nodes = config["output_nodes_name"]

    # Load preparators
    prepare, output_preparator = load_preparators(gen_dir, config)

    results = OrderedDict()  # Dict to be returned
    conf_th = 0.01
    iou_th = 0.7
    ncols = 90

    NB_WORKER = max(2 * os.cpu_count() // 3, 1)
    results = dict()

    if device == "mppa":
        input_sizes = []
        output_sizes = []

        for shape, dt in zip(config["input_nodes_shape"], config["input_nodes_dtype"]):
            element_size = 2 if "16" in dt else 4
            element_size = 1 if "8" in dt else 4
            input_sizes.append(numpy.prod(shape) * element_size)
        for shape, dt in zip(config["output_nodes_shape"], config["output_nodes_dtype"]):
            element_size = 2 if "16" in dt else 4
            element_size = 1 if "8" in dt else 4
            output_sizes.append(numpy.prod(shape) * element_size)
        parts, data = get_chunks_sizes(input_sizes + output_sizes, len(image_paths), ratio_ram)

        if len(parts) > 1:
            logger.info(f"Memory space used will exceed {round(data / 1024, 1)} GB in RAM, "
                        f"splitting output processing is realized by steps of {parts} images.")

        tmp_io = f"./.tmp_io_{config['name']}"
        if os.path.isdir(tmp_io):
            count = len([d for d in os.listdir(".") if f"tmp_io_{config['name']}" in d])
            tmp_io = f"./{tmp_io}_{count}"
        logger.info(f"Processing images in [{len(parts)}] steps of nb-images : {parts[0]}")
        logger.debug(f'  available at {tmp_io}')
        for c_idx, c_size in enumerate(parts):

            logger.info(f"STEP {c_idx+1} / {len(parts)}")
            start_idx = sum(parts[:c_idx])
            end_idx = start_idx + c_size

            # INPUT PROCESSING STAGE
            shutil.rmtree(tmp_io, ignore_errors=True)  # Remove just in case, from previous interrupted execution
            os.makedirs(tmp_io, exist_ok=True)
            if "/" in config['input_nodes_name'][0]:
                os.makedirs(os.path.join(tmp_io, os.path.dirname(config['input_nodes_name'][0])), exist_ok=True)
            shapes_by_img = OrderedDict()

            progress_bar_pre = tqdm(
                total=c_size,
                desc=f"Preparing images for inference ",
                ncols=ncols,
                colour="cyan")
            t_start = time.perf_counter()
            with open(f"{tmp_io}/{config['input_nodes_name'][0]}", "w+") as f:
                for image_path in image_paths[start_idx:end_idx]:
                    image_id = os.path.splitext(os.path.basename(image_path))[0]
                    content = cv2.imread(image_path)
                    prepared = prepare.prepare_img(content)
                    shapes_by_img[image_id] = content.shape
                    prepared.tofile(f)
                    progress_bar_pre.update(1)
                    progress_bar_pre.refresh()
            t_pre_ms = 1e3 * (time.perf_counter() - t_start)
            progress_bar_pre.close()

            # INFERENCE STAGE
            logger.info('Running inference on MPPA ...')
            inference_log_file = f"eval_inference_{os.path.basename(gen_dir)}.log"
            serialized_param_file = [f for f in os.listdir(gen_dir) if f.split(".")[-1] == "kann"][0]
            serialized_param_file = os.path.join(gen_dir, serialized_param_file)
            flog = open(inference_log_file, "w+")
            t_start = time.perf_counter()
            kann_proc = subprocess.Popen(
                ["kann_opencl_cnn", serialized_param_file, tmp_io],
                bufsize=-1, stdout=flog)
            kann_proc.wait(timeout=5. * c_size)
            t_mppa_ms = 1e3 * (time.perf_counter() - t_start)
            kann_proc.terminate()
            flog.close()

            # OUTPUT PROCESSING AND RESULT STORING STAGE
            t_start = time.perf_counter()
            chunk_image_ids = list(shapes_by_img.keys())
            preds = {}
            for out_idx, output_node in enumerate(output_nodes):
                node_dtype = config["output_nodes_dtype"][out_idx]
                node_dtype = numpy.dtype(node_dtype)

                # Compute offset in bytes to the start of this chunk's data
                offset_bytes = start_idx * output_sizes[out_idx]
                offset_bytes = 0

                # Compute total bytes to read for this chunk
                count = c_size * output_sizes[out_idx]
                has_slash = False
                if output_node.startswith("/"):
                    has_slash = True
                    output_node = output_node[1:]
                preds_data = numpy.fromfile(
                    os.path.join(tmp_io, output_node),
                    dtype=node_dtype,
                    count=count,
                    offset=offset_bytes,
                )
                if has_slash:
                    output_node = "/" + output_node

                # Reshape JUST to separate single image data
                # (the final reshape will be done inside postproc function)
                if len(preds_data) == 0:
                    raise RuntimeError(f"  >>> issue on chunk data, get {preds_data.shape}")
                preds[output_node] = preds_data.reshape(c_size, -1)

            progress_bar_post = tqdm(
                total=c_size,
                desc=f"Post-processing predictions {c_idx+1}/{len(parts)}",
                ncols=ncols,
                colour="red")
            threads = []
            mutex = threading.Lock()

            for i, image_id in enumerate(chunk_image_ids):
                # Extract this image's data from the data
                input_postproc = {  # To be passed to postproc function
                    output: preds[output][i] for output in output_nodes
                }
                thr = threading.Thread(
                    target=post_proc_data,
                    args=(
                        input_postproc, output_preparator, config,
                        shapes_by_img[image_id], image_id, results, device,
                        conf_th, iou_th, mutex, debug
                    )
                )
                threads.append(thr)
                thr.start()
                if len(threads) >= NB_WORKER:
                    [t.join() for t in threads]
                progress_bar_post.update(1)
                progress_bar_post.refresh()
            [t.join() for t in threads]
            t_post_ms = 1e3 * (time.perf_counter() - t_start)
            progress_bar_post.refresh()
            progress_bar_post.close()

            logger.info("STATISTICS:")
            logger.info(f"  Pre-processing on CPU  : {t_pre_ms  / 1e3:.3f} sec - ({t_pre_ms / c_size:.3f} ms / img)")
            logger.info(f"  Processing on MPPA :     {t_mppa_ms / 1e3:.3f} sec - ({t_mppa_ms / c_size:.3f} ms / query)")
            logger.info(f"  Post-processing on CPU : {t_post_ms / 1e3:.3f} sec - ({t_post_ms / c_size:.3f} ms / img)")

            # Free local memory immediately
            del preds
            shutil.rmtree(tmp_io, ignore_errors=True)

    elif device == "cpu":

        t = {
            "t_prep" : [],
            "t_infe" : [],
            "t_post" : [],
        }

        progress_bar = tqdm(
            total=len(image_paths),
            desc=f"Executing pipeline (pre, inference, post)",
            ncols=ncols,
            colour="blue")
        threads = []

        sess = rt.InferenceSession(config["onnx_model"])
        mutex = threading.Lock()
        for image_path in image_paths:
            thr = threading.Thread(
                target=cpu_video_pipeline,
                args=(
                    sess, prepare, output_preparator,
                    image_path, config, results, device, t, conf_th, iou_th, mutex, debug
                )
            )
            threads.append(thr)
            thr.start()
            progress_bar.update(1)
            progress_bar.refresh()
            if len(threads) >= NB_WORKER:
                [t.join() for t in threads]
        [t.join() for t in threads]
        progress_bar.close()

        logger.info("STATISTICS:")
        t_pre_ms = sum(t["t_prep"])
        t_cpu_ms = sum(t["t_infe"])
        t_post_ms = sum(t["t_post"])
        logger.info(f"  Pre-processing  : {t_pre_ms  / 1e3:.3f} sec - ({t_pre_ms / len(image_paths):.3f} ms / img)")
        logger.info(f"  Processing      : {t_cpu_ms / 1e3:.3f} sec - ({t_cpu_ms / len(image_paths):.3f} ms / query)")
        logger.info(f"  Post-processing : {t_post_ms / 1e3:.3f} sec - ({t_post_ms / len(image_paths):.3f} ms / img)")

    else:
        raise RuntimeError(f"Device unknown, get {device}")

    return results


def print_results(data):
    """
    An utility function to print the dictionaries containing either
    the results of the inference, or the references.
    Useful method for debugging.

    Args:
        data (dict): containing either inferences results or references
            for every image as key.
    """
    for img_id, detections in data.items():
        print(f"\nImage: {img_id}")
        for label_idx, (label, preds) in enumerate(detections.items()):
            prefix = "└──" if label_idx == len(detections) - 1 else "├──"
            print(
                f"{prefix} {label} ({len(preds)} detection{'s' if len(preds) > 1 else ''})"
            )
            for pred_idx, pred in enumerate(preds):
                # handle both (score, bbox) and raw bbox formats
                if isinstance(pred, tuple):
                    score, bbox = pred
                    score_str = f"Score: {score:.2f}, "
                else:
                    score_str = ""
                    bbox = pred
                sub_prefix = "    └──" if pred_idx == len(preds) - 1 else "    ├──"
                print(f"{sub_prefix} {score_str}Bbox: {bbox}")
    print("\n")


def main(opt):
    """
    Checks the dataset, then gets results from running the inference,
    loads the references as a dict and sends both to an instance
    of EvalKaNN_ObjDetect to compute the metrics.

    Args:
        opt (Namespace): the parsed arguments given at the command line.
    """
    gen_dir = opt.gen_dir
    dataset = opt.dataset
    device = opt.device
    print_all_classes = opt.all
    debug = opt.debug
    if debug:
        print_all_classes = True
    # Check dataset file system
    # Download dataset if it does not exists
    dataset_image_path = check_dataset(dataset)
    if opt.metrics == "mAP":
        references = load_coco_references(dataset_image_path)
    elif opt.metrics == "topk":
        references = load_imagenet_references(dataset_image_path)
    else:
        return NotImplementedError(f"Other metrics than topk and mAP are not implemented yet, get {opt.metrics}")
    # Start evaluation
    t_start = time.perf_counter()
    gen_dir = os.path.realpath(gen_dir)
    results = run(gen_dir, dataset_image_path, device, debug=debug)
    if debug:
        print_results(results)
    # Computing metrics
    class_names = get_classes_id(gen_dir)
    if opt.metrics == "mAP":
        c = EvalKaNN_ObjDetect
    elif opt.metrics == "topk":
        c = EvalKaNN_Classify
    else:
        return NotImplementedError(f"Other metrics than topk and mAP are not implemented yet, get {opt.metrics}")
    kEval = c(class_names, print_all_classes)
    kEval(results, references)
    t_end = time.perf_counter()

    if not print_all_classes:
        logger.info("For details, please use --all (or -a) to print the metric value for all classes")
    logger.info(f"Evaluation time on {dataset.upper()} takes {t_end - t_start:.3f} secs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute metrics of a KaNN neural network model."
    )
    parser.add_argument(
        "gen_dir",
        type=str,
        help="path to the model's generated dir."
    )
    parser.add_argument(
        "--metrics", "-m",
        type=str,
        required=True,
        help="Type of metrics evaluation (top-k, mAP, mIoU)",
        choices=["mAP", "topk"],  # TODO: future: mIoU
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="specify the dataset to compute metrics on",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        help="specify the device to execute inference on. default: mppa",
        default="mppa",
        choices=["mppa", "cpu"],
    )
    parser.add_argument(
        "--all", "-a",
        action='store_true',
        help="print all categories of dataset's image",
        default=False,
    )
    parser.add_argument(
        "--debug", "-dbg",
        action='store_true',
        help="print results for comparison",
        default=False,
    )
    args = parser.parse_args()
    main(args)
