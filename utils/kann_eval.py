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
#   Daniel Angulo, <dangulo@kalrayinc.com
###

import os
import re
import time

import cv2
import yaml
import glob
import numpy
import psutil
import shutil
import argparse
import threading
import importlib
import contextlib
import subprocess
import onnxruntime as rt

from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from io import StringIO

from metrics import ObjDetectMetrics
from metrics import box_iou
from kann_utils import logger
from utils import WORKSPACE_PATH
from utils import COCO_IDX2NAME


class EvalKaNN_Base(object):
    """
    A base class to evaluate metrics that work with pre-computed results.
    This class implements the methods __init__, and match_predictions().
    It also declares other methods for metrics computation, to be implemented by its subclasses.

    Attributes:
        names (dict): a dict containing the mapping of class indices to class names.
        seen (Any): records the number of images seen so far during validation.
        stats (dict): placeholder for statistics during validation.
        nc (int): number of classes.
    """

    def __init__(self, id_to_names=None):

        self.id_to_names = id_to_names
        self.names_to_id = {v: k for k, v in id_to_names.items()}
        self.nc = len(id_to_names)
        self.seen = None
        self.stats = None

    def __call__(self, results, references):
        """
        Executes validation process using pre-computed results.

        Args:
            tensors_by_img (dict):  a dict of tuples of the form (detections, gt_bboxes, gt_cls),
                                    each element being a numpy.ndarray.
        """
        self.init_metrics()
        arrays_by_img = self.fuse_and_arrayize(results, references)
        for k in sorted(arrays_by_img):
            self.update_metrics(arrays_by_img[k])
        self.get_stats()
        self.print_results()

    def init_metrics(self):
        """
        Initialize metrics with class information.
        """
        self.metrics.names = self.id_to_names
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def update_metrics(self, pred_gt_arrays):
        """
        Update the stats dictionary by processing the detections and ground truth arrays.

        Args:
            pred_gt_arrays (tuple): a tuple of three numpy.ndarrays containing information
                about detections and ground truth.
        """
        return NotImplementedError

    def get_stats(self):
        """
        Returns metrics statistics and results dictionary.
        """
        # Convert lists of arrays to single concatenated arrays
        stats = {}
        for k, v in self.stats.items():
            if v:  # Check if list is not empty
                stats[k] = numpy.concatenate(v)
            else:
                stats[k] = numpy.array([])
        self.nt_per_class = numpy.bincount(
            stats["target_cls"].astype(int), minlength=self.nc
        )
        self.nt_per_image = numpy.bincount(
            stats["target_img"].astype(int), minlength=self.nc
        )
        stats.pop("target_img", None)
        if len(stats) and "tp" in stats and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def fuse_and_arrayize(self, results, references):
        """
        Fuses dictionaries by entry (since both dicts' keys are the same: images' id),
        then converts them to evaluation-ready arrays.
        """
        return NotImplementedError

    def match_predictions(self, pred_classes, true_classes):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes)
        """
        return NotImplementedError

    def print_results(self):
        """
        Print validation metrics per class.
        """
        return NotImplementedError


class EvalKaNN_ObjDetect(EvalKaNN_Base):
    """
    A class extending the EvalKaNN_Base class for validating results from an detection task.

    Attributes:
        nt_per_class (dict): a dict containing the no. of instances per class over all images.
        nt_per_image (dict): a dict with no. of images where a class appears at least once.
        iouv (numpy.ndarray): implementation of the iouv attr, from 0.5 to 0.95 in steps of 0.05.
        niou (int): number of elements of the iouv array.
        metrics (Metrics): an object that does the actual computation of the metrics.
        seen (int): implementation of the seen attr, now a proper int.
    """

    def __init__(self, id_to_names, print_all=False):
        EvalKaNN_Base.__init__(self, id_to_names)
        self.nt_per_class = None
        self.nt_per_image = None
        self.iouv = numpy.linspace(0.5, 0.95, 10)
        self.niou = len(self.iouv)
        self.metrics = ObjDetectMetrics()
        self.print_all = print_all
        self.seen = 0

    def get_precision(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (numpy.ndarray): Array of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (numpy.ndarray): Array of shape (M, 4) representing ground-truth bounding box coordinates.
            gt_cls (numpy.ndarray): Array of shape (M,) representing target class indices.

        Returns:
            numpy.ndarray: Correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])  # Assuming box_iou is implemented with numpy
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def fuse_and_arrayize(self, results, references):
        """
        Fuses dictionaries by entry (since both dicts' keys are the same: images' id),
        then converts them to evaluation-ready arrays.

        Args:
            results: Dict mapping image_id to predictions {img_id: {class: [(score, bbox), ...], ...}}
            references: Dict mapping image_id to ground truth {img_id: {class: [bbox, ...], ...}}

        Returns:
            dict: {image_id: (detections, gt_bboxes, gt_cls)} where:
                - detections: numpy.ndarray (N,6) [x1, y1, x2, y2, conf, cls_idx]
                - gt_bboxes: numpy.ndarray (M,4) [x1, y1, x2, y2]
                - gt_cls: numpy.ndarray (M,) class indices
        """
        arrays_by_img = {}
        all_image_ids = set(results.keys()) | set(references.keys())
        for image_id in all_image_ids:
            # Process detections
            detections = []
            if image_id in results:
                for class_name, preds in results[image_id].items():
                    class_idx = self.names_to_id[class_name]
                    if class_idx is None:
                        continue  # Skip unknown classes
                    for score, bbox in preds:
                        detections.append([*bbox, score, class_idx])
            # Convert to array (empty array if no detections)
            detections_array = (
                numpy.array(detections, dtype=numpy.float32)
                  if detections else numpy.zeros((0, 6), dtype=numpy.float32)
            )
            # Process ground truth
            gt_bboxes = []
            gt_cls = []
            if image_id in references:
                for class_name, bboxes in references[image_id].items():
                    class_idx = self.names_to_id[class_name]
                    if class_idx is None:
                        continue  # Skip unknown classes
                    for bbox in bboxes:
                        gt_bboxes.append(bbox)
                        gt_cls.append(class_idx)
            # Convert to arrays
            gt_bboxes_array = (
                numpy.array(gt_bboxes, dtype=numpy.float32)
                if gt_bboxes
                else numpy.zeros((0, 4), dtype=numpy.float32)
            )
            gt_cls_array = (
                numpy.array(gt_cls, dtype=numpy.int64)
                if gt_cls
                else numpy.zeros(0, dtype=numpy.int64)
            )
            arrays_by_img[image_id] = (detections_array, gt_bboxes_array, gt_cls_array)
        return arrays_by_img

    def update_metrics(self, pred_gt_arrays):
        """
        Update the stats dictionary by processing the detections and ground truth arrays.

        Args:
            pred_gt_arrays (tuple): a tuple of three numpy.ndarrays containing information
                about detections and ground truth.
        """
        detections, gt_bboxes, gt_cls = pred_gt_arrays
        self.seen += 1
        npr = len(detections)  # Number of predictions
        # Initialize stat dictionary for this image
        stat = dict(
            conf=numpy.zeros(0),
            pred_cls=numpy.zeros(0),
            tp=numpy.zeros((npr, self.niou), dtype=bool),
        )
        nl = len(gt_cls)  # Number of ground truth labels
        # Store target class information
        stat["target_cls"] = gt_cls
        stat["target_img"] = numpy.unique(gt_cls) if nl > 0 else numpy.zeros(0)
        # If no predictions but we have ground truth, record missed detections
        if npr == 0:
            if nl:
                for k in self.stats.keys():
                    self.stats[k].append(stat[k])
            return
        # Store confidence and predicted class
        stat["conf"] = detections[:, 4]
        stat["pred_cls"] = detections[:, 5]
        # Evaluate predictions against ground truth
        if nl:
            stat["tp"] = self.get_precision(detections, gt_bboxes, gt_cls)
        # Update stats dictionary
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def match_predictions(self, pred_classes, true_classes, iou):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (numpy.ndarray): Predicted class indices of shape(N,).
            true_classes (numpy.ndarray): Target class indices of shape(M,).
            iou (numpy.ndarray): An NxM array containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (numpy.ndarray): Correct array of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = numpy.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        for i, threshold in enumerate(self.iouv.tolist()):
            matches = numpy.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = numpy.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        tp = correct.astype(bool)
        return tp

    def print_results(self):
        """
        Print validation metrics per class.
        """

        if not hasattr(self.metrics, "keys") or not self.metrics.keys:
            logger.error("Metrics not properly initialized or processed")
            return

        # Print column headers
        logger.info("")
        logger.info(("%22s" + "%11s" * 7) % ("Class", "Images", "Instances", "Prec", "Recall", "F1-score", "mAP50", "mAP50-95"))
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format

        # Print overall results
        mean_results = self.metrics.mean_results()
        if all(r is not None for r in mean_results):
            logger.info(pf % ("all", self.seen, int(self.nt_per_class.sum()), *mean_results))
        else:
            logger.warning("Some metrics returned None values")
        if self.nt_per_class.sum() == 0:
            logger.warning("No labels found, cannot compute metrics without labels")
            return

        # Print per-class results
        if self.print_all and hasattr(self.metrics, "ap_class_index") and self.metrics.ap_class_index is not None:
            for i, c in enumerate(self.metrics.ap_class_index):
                if c < len(self.id_to_names):  # Check valid index
                    class_name = self.id_to_names[c]
                    class_results = self.metrics.class_result(i)
                    if all(r is not None for r in class_results):
                        logger.info(
                            pf % (class_name, (int(self.nt_per_image[c]) if c < len(self.nt_per_image) else 0),
                                (int(self.nt_per_class[c]) if c < len(self.nt_per_class) else 0),
                                *class_results)
                        )
        logger.info("")


def check_coco_dataset(dataset):
    """
    Check if the COCO dataset is already present, else download it
    and place it in "./datasets" folder.

    Args:
        dataset (str): the name of the dataset ("coco8", "coco128" or "coco").
    """
    dataset_path = os.path.join(WORKSPACE_PATH, "utils", "datasets", dataset)
    if not os.path.exists(dataset_path):
        logger.warning(f"The dataset {dataset} does not exist. Downloading...")
        if dataset == "coco":
            images_url = "http://images.cocodataset.org/zips/val2017.zip"
            labels_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
            os.system(f"wget {images_url} -P {WORKSPACE_PATH}/utils/datasets/")
            os.system(f"wget {labels_url} -P {WORKSPACE_PATH}/utils/datasets/")
            os.system(f"unzip {WORKSPACE_PATH}/utils/datasets/val2017.zip -d {WORKSPACE_PATH}/utils/datasets/")
            os.system(f"unzip {WORKSPACE_PATH}/utils/datasets/coco2017labels.zip -d {WORKSPACE_PATH}/utils/datasets/")
            os.system(f"mv {WORKSPACE_PATH}/utils/datasets/val2017 {WORKSPACE_PATH}/utils/datasets/coco/images/")
            os.system(f"rm -f {WORKSPACE_PATH}/utils/datasets/coco/coco2017labels.zip")
            os.system(f"rm -f {WORKSPACE_PATH}/utils/datasets/coco/val2017.zip")
            shutil.rmtree(f"{WORKSPACE_PATH}/utils/datasets/coco/images/train2017", ignore_errors=True)
            shutil.rmtree(f"{WORKSPACE_PATH}/utils/datasets/coco/labels/train2017", ignore_errors=True)
        else:
            dataset_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip"
            os.system(f"wget {dataset_url} -P {WORKSPACE_PATH}/utils/datasets/")
            os.system(f"unzip {WORKSPACE_PATH}/utils/datasets/{dataset}.zip -d {WORKSPACE_PATH}/utils/datasets/")
            logger.info(f"Dataset {dataset} downloaded and extracted to {os.path.join(dataset_path, dataset)}")

    if dataset == "coco":
        data_img_path = os.path.join(dataset_path, "images", "val2017")
        if os.path.isdir(data_img_path):
            return data_img_path
    elif dataset == "coco128":
        data_img_path = os.path.join(dataset_path, "images", "train2017")
        if os.path.isdir(data_img_path):
            return data_img_path
    elif dataset == "coco8":
        data_img_path = os.path.join(dataset_path, "images", "val")
        if os.path.isdir(data_img_path):
            return data_img_path
    else:
        data_img_path = os.path.join(dataset_path, "images", "*")
        return data_img_path


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
            if len(parts) != 3:
                logger.warning("There was a malformed line while parsing results from the output preparator.")  # Skip malformed lines
            score = float(parts[0].strip())
            label = parts[1].strip()
            bbox = [float(coord.strip()) for coord in parts[2].strip("[]").split(",")]
            # Ensure bbox has 4 coordinates (x1, y1, x2, y2)
            if len(bbox) != 4:
                logger.warning("There was a malformed bbox while parsing results from the output preparator.")  # Skip invalid bboxes
            detections.append((bbox, score, label))
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse line: {line}\nError: {e}")
            continue
    return detections


def get_ram():
    """
    Get the RAM total capacity of the host's computer using psutil.

    Returns:
        float: total virtual memory in megabytes.
    """
    # Get virtual memory statistics
    vm = psutil.virtual_memory()
    return vm.total // (1024 ** 2)  # Convert bytes to MB


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
    available_ram_mb = get_ram()  # In megabytes
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
    image_paths = sorted(
        glob.iglob(f"{dataset_img_path}/*.jpg", recursive=True)
    )
    if len(image_paths) == 0:
        raise FileNotFoundError(f'File not found at this path {dataset_img_path}')

    # Parse data from .yaml file
    network_config = [f for f in os.listdir(gen_dir) if f.endswith(".yaml")][0]
    with open(os.path.join(gen_dir, network_config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
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

    # Load input and output preparators
    module_dir = os.path.relpath(gen_dir, WORKSPACE_PATH).replace("/", ".")
    inproc_module_name = f"{module_dir}.{config['input_preparator'].replace('.py', '')}"
    prepare = importlib.import_module(inproc_module_name)
    outproc_module_name = f"{module_dir}.{config['output_preparator']}.output_preparator"
    output_preparator = importlib.import_module(outproc_module_name)  # For output processing, later

    results = OrderedDict()  # Dict to be returned
    conf_th = 0.01
    iou_th = 0.7
    ncols = 90

    NB_WORKER = max(2 * os.cpu_count() // 3, 1)
    results = dict()

    def _post_proc_data(input_postproc, img_shape, img_id, res, locker=None):

        # OUTPUT PROCESSING STAGE
        dummy_frame = numpy.zeros(img_shape, dtype=numpy.int32)
        # TODO: once all networks have this method, remoce the conditional block
        #  to replace by "detect = output_preparator.post_process_eval"
        try:
            if hasattr(output_preparator, "post_process_eval"):
                _, detections = output_preparator.post_process_eval(
                    cfg=config,
                    frame=dummy_frame,
                    nn_outputs=input_postproc,
                    device=device,
                    conf_thres=conf_th,
                    iou_thres=iou_th,
                    dbg=False,
                )
            elif hasattr(output_preparator, "post_process"):  # because not all networks have post_process_eval
                if locker is not None:
                    locker.acquire()
                output_buffer = StringIO()
                with contextlib.redirect_stdout(output_buffer):
                    output_preparator.post_process(
                        cfg=config,
                        frame=dummy_frame,
                        nn_outputs=input_postproc,
                        device=device,
                        conf_thres=conf_th,
                        iou_thres=iou_th,
                        dbg=True,
                    )
                captured_output = output_buffer.getvalue()
                if locker is not None:
                    locker.release()
                detections = parse_results(
                    captured_output.splitlines()
                )  # Entries of the form (conf, label, bbox)
            else:
                raise RuntimeError("The call to 'output_process_eval' method from output_preparator.py failed")
        except Exception as err:
            raise RuntimeError(f" Issue on {img_id} : {err}")

        # RESULT STORING STAGE
        res[img_id] = {}
        for bbox, conf, label in detections:
            if debug:
                logger.info(f"id: {img_id} - label:{label:15s}, conf:{conf:1.4f}, xyxy:{bbox}")
            if locker is not None:
                locker.acquire()
            if label in res[img_id]:
                res[img_id][label].append((conf, bbox))
            else:
                res[img_id][label] = [(conf, bbox)]
            if locker is not None:
                locker.release()

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

        logger.info(f"Processing images in [{len(parts)}] steps of nb-images : {parts[0]}")
        for c_idx, c_size in enumerate(parts):

            logger.info(f"STEP {c_idx+1} / {len(parts)}")
            start_idx = sum(parts[:c_idx])
            end_idx = start_idx + c_size

            # INPUT PROCESSING STAGE
            shutil.rmtree(".tmp_io", ignore_errors=True)  # Remove just in case, from previous interrupted execution
            os.makedirs(".tmp_io", exist_ok=True)
            shapes_by_img = OrderedDict()

            progress_bar_pre = tqdm(
                total=c_size,
                desc=f"Preparing images for inference ",
                ncols=ncols,
                colour="cyan")
            t_start = time.perf_counter()
            with open(f".tmp_io/{config['input_nodes_name'][0]}", "w+") as f:
                for image_path in image_paths[start_idx:end_idx]:
                    image_id = os.path.basename(image_path).removesuffix(".jpg")
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
                ["kann_opencl_cnn", serialized_param_file, ".tmp_io"],
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
                    os.path.join(".tmp_io", output_node),
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
                    target=_post_proc_data,
                    args=(input_postproc, shapes_by_img[image_id], image_id, results, mutex)
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
            shutil.rmtree(".tmp_io", ignore_errors=True)

    elif device == "cpu":

        t_prep = []
        t_infe = []
        t_post = []

        def _video_pipeline(img_path, cfg, res, lock):

            sess = rt.InferenceSession(config["onnx_model"])

            # INPUT PROCESSING STAGE
            t_start = time.perf_counter()
            image_id = os.path.basename(img_path).removesuffix(".jpg")
            content = cv2.imread(img_path)
            image_shape = content.shape
            prepared = prepare.prepare_img(content)
            prepared = prepared.reshape(cfg["input_nodes_shape"][0])
            prepared = prepared.transpose(cfg["input_nodes_dformat"][0])
            inputs = {config["input_nodes_name"][0]: prepared}
            t_prep.append(1e3 * (time.perf_counter() - t_start))

            # INFERENCE STAGE
            t_start = time.perf_counter()
            preds = sess.run(None, inputs)
            t_infe.append(1e3 * (time.perf_counter() - t_start))

            # OUTPUT PROCESSING STAGE
            t_start = time.perf_counter()
            outputs_name = [o.name for o in sess.get_outputs()]
            input_postproc = {k: o for o, k in zip(preds, outputs_name)}
            _post_proc_data(input_postproc, image_shape, image_id, res, lock)
            t_post.append(1e3 * (time.perf_counter() - t_start))

        progress_bar = tqdm(
            total=len(image_paths),
            desc=f"Executing pipeline (pre, inference, post)",
            ncols=ncols,
            colour="blue")
        threads = []
        mutex = threading.Lock()
        for image_path in image_paths:
            thr = threading.Thread(
                target=_video_pipeline,
                args=(image_path, config, results, mutex)
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
        t_pre_ms = sum(t_prep)
        t_cpu_ms = sum(t_infe)
        t_post_ms = sum(t_post)
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


def load_references(dataset_img_path):
    """
    Load references from a dataset path, and format them correctly
    to return a dictionary. The coordinates labels are of the
    form (center_x, center_y, width, height)

    Args:
        dataset_path (str): the path of the dataset.
    Returns:
        references (dict): containing the ground truth for every image
            as key.
    """
    references = dict()
    dataset_label_path = dataset_img_path.replace("/images/", "/labels/")
    label_files = sorted(
        glob.iglob(f"{dataset_label_path}/*.txt", recursive=True)
    )
    if len(label_files) == 0:
        raise FileNotFoundError(f"No files have been found for {dataset_label_path}")

    for label_file in label_files:
        # Get image ID from label filename (e.g., "000000000009.txt" → "000000000009")
        image_id = os.path.splitext(os.path.basename(label_file))[0]
        # Get corresponding image path
        image_path = label_file.replace("/labels/", "/images/").replace(".txt", ".jpg")
        if not os.path.exists(image_path):
            logger.warning(f"Image {image_path} not found. Skipping.")
            continue
        # Get image size
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            # This error can only happen if the structure of the dataset folder
            # has been altered in some way that should not happen.
            logger.warning(f"Could not open {image_path}: {e}")
            continue
        # Read label file
        with open(label_file, "r") as f:
            lines = f.readlines()
        # Initialize entry for this image
        references[image_id] = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip invalid lines

            # Parse class ID and normalized coordinates
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert to absolute coordinates (x1, y1, x2, y2)
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height
            x1 = float(x_center_abs - (width_abs / 2))
            y1 = float(y_center_abs - (height_abs / 2))
            x2 = float(x_center_abs + (width_abs / 2))
            y2 = float(y_center_abs + (height_abs / 2))

            # Clamp coordinates to image boundaries
            x1 = max(0., min(x1, float(img_width)))
            y1 = max(0., min(y1, float(img_height)))
            x2 = max(0., min(x2, float(img_width)))
            y2 = max(0., min(y2, float(img_height)))

            # Map class ID to COCO name
            if class_id in COCO_IDX2NAME:
                class_name = COCO_IDX2NAME[class_id]
                if class_name not in references[image_id]:
                    references[image_id][class_name] = []
                references[image_id][class_name].append([x1, y1, x2, y2])
            else:
                logger.warning(f"class id: {class_id} not found in COCO lut name")
    return references


def get_classes_id(generated_dir):
    """
    generated dir by kann must contain a file name called classes.txt to identify the label_id and class name 
    Args:
        generated_dir (str): the name of model DIR generated by KaNN.
    Returns:
        result (dict): containing the class_id {id(int): label_id(str)}.
    """

    if not os.path.isdir(generated_dir):
        raise NotADirectoryError(f"Generated DIR {generated_dir} not found")
    yaml_file_path = os.path.join(generated_dir, "network.dump.yaml")
    if not os.path.isfile(yaml_file_path):
        raise FileNotFoundError("Generated DIR must contain a dmp of the configuration file")
    with open(yaml_file_path, "r") as fyaml:
        config = yaml.load(fyaml, Loader=yaml.FullLoader)
    class_path_file = os.path.join(generated_dir, config["extra_data"]["classes"])
    with open(class_path_file, "r") as fclasses:
        classes = [l.rstrip("\n") for l in fclasses.readlines()]

    result = dict()
    if len(classes[0].split(" ")) == 2:
        result = {int(id.split(" ")[0]): id.split(" ")[-1] for id in classes}
    elif len(classes[0].split(" ")) == 1:
        result = {id: name for id, name in enumerate(classes)}
    else:
        raise ValueError(f"{class_path_file} format is not as expected <0 label_id> or <label_id> per row-line")
    return result


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
    if opt.metrics == "mAP":
        dataset_image_path = check_coco_dataset(dataset)
    else:
        return NotImplementedError(f"Other metrics than mAP are not implemented yet, get {opt.metrics}")
    references = load_references(dataset_image_path)

    # Start evaluation
    t_start = time.perf_counter()
    gen_dir = os.path.realpath(gen_dir)
    results = run(gen_dir, dataset_image_path, device, debug=debug)
    if debug:
        print_results(results)

    # Computing metrics
    class_names = get_classes_id(gen_dir)
    if opt.metrics == "mAP":
        EvalKaNN_ObjDetect(class_names, print_all_classes)(results, references)
    else:
        return NotImplementedError(f"Other metrics than mAP are not implemented yet, get {opt.metrics}")
    t_end = time.perf_counter()
    if not print_all_classes:
        logger.info("For details, please use --all (or -a) to print mAP for all classes")
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
        choices=["mAP"],  # TODO: future: topk, mIoU
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
        help="print bounding box for comparison",
        default=False,
    )
    args = parser.parse_args()
    main(args)
