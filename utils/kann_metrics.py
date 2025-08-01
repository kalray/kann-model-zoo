###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
#
# Script inspired by
#   https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py
#
# Authors:
#   Quentin Muller, <qmuller@kalrayinc.com>
#   Daniel Angulo, <dangulo@kalrayinc.com>
###

import numpy

from typing import Dict, Optional, Tuple

from utils import IMAGENET
from kann_utils import logger


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
        return NotImplementedError

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
        return NotImplementedError

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


class EvalKaNN_Classify(EvalKaNN_Base):
    """
    Class for evaluating classification results.
    Attributes:
        targets (List[numpy.ndarray]): Ground truth class labels for each image.
        pred (List[numpy.ndarray]): Predicted top-k class indices for each image.
        metrics (ClassifyMetrics): Metrics computation object.
        topk (int): Number of top predictions to consider.
        print_all (bool): Whether to print metrics for all classes.

    Note:
        topk can be less but not more than 5, this limit being
        hardcoded on every classifiers' output preparator.
    """

    # Arbitrary value for empty/unknown class
    # Unknown if a real class from a bigger Imagenet dataset
    # or if its really not a class from Imagenet
    UNKNOWN_CLASS_ID = -1

    def __init__(self, id_to_names: Dict[int, str], print_all: bool = False) -> None:
        super().__init__(id_to_names)
        self.targets = None
        self.preds = None
        self.metrics = ClassifyMetrics()
        self.topk = 5
        self.print_all = print_all
        self.seen = 0
        self.images_per_class = None

    def init_metrics(self) -> None:
        """ Initialize metrics and prediction containers. """
        self.metrics.names = self.id_to_names
        self.preds = []
        self.targets = []

    def fuse_and_arrayize(self, results: Dict, references: Dict) -> Dict[str, Tuple]:
        """
        Process raw results and references into evaluation-ready arrays.
        Args:
            results: Dictionary mapping image_id to class predictions with scores.
            references: Dictionary mapping image_id to ground truth class labels.
        Returns:
            Dictionary mapping image_id to tuples of (predictions_array, gt_class_id_array).
        """
        arrays_by_img = {}
        all_image_ids = set(results.keys()) | set(references.keys())
        for image_id in all_image_ids:
            pred = self._predictions_to_array(results.get(image_id, {}))
            gt_class_id = self._ground_truth_to_array(references.get(image_id))
            arrays_by_img[image_id] = (pred, gt_class_id)
        return arrays_by_img

    def update_metrics(self, pred_gt_arrays: Tuple) -> None:
        """
        Update metrics with predictions and ground truth.
        Args:
            pred_gt_arrays: Tuple of (preds, gt_class_id).
        """
        predictions_array, gt_class_array = pred_gt_arrays
        if predictions_array.shape[0] == 0:
            # Handle empty predictions
            self.preds.append(numpy.zeros((1, 0), dtype=numpy.int32))
            self.targets.append(gt_class_array.astype(numpy.int32))
            return

        # Extract topk predictions in desceding order
        sorted_indices = numpy.argsort(-predictions_array[:, 1])
        sorted_predictions = predictions_array[sorted_indices]
        class_indices = sorted_predictions[:, 0].astype(numpy.int32)
        k = min(self.topk, len(class_indices))
        top_indices = class_indices[:k]

        # Store for evaluation
        self.preds.append(numpy.array([top_indices], dtype=numpy.int32))
        self.targets.append(gt_class_array.astype(numpy.int32))

    def get_stats(self) -> Dict:
        """
        Calculate and return classification metrics.
        Returns:
            Dictionary of metrics.
        """
        self.metrics.process(self.targets, self.preds)
        self.seen = len(self.targets)
        # Count images per class, skipping value for empty
        if hasattr(self.metrics, "ap_class_index") and self.metrics.ap_class_index is not None:
            self.images_per_class = numpy.zeros(len(self.metrics.ap_class_index), dtype=int)
            for target in self.targets:
                if len(target) > 0:
                    for i, c in enumerate(self.metrics.ap_class_index):
                        if c in target and not self._is_unknown(c):
                            self.images_per_class[i] += 1
        return self.metrics.results_dict

    def print_results(self) -> None:
        """ Print formatted classification results. """
        if not hasattr(self.metrics, "keys") or not self.metrics.keys:
            # No results to print
            return
        logger.info("")
        logger.info(("%22s" + "%11s" * 3) % ("Class", "Images", "top1_acc", f"top{self.topk}_acc"))
        pf = "%22s" + "%11i" + "%11.3g" * 2  # print format
        logger.info(pf % ("all", self.seen, self.metrics.top1, self.metrics.topk))
        # Print per-class metrics
        if self.print_all and hasattr(self.metrics, "ap_class_index") and self.metrics.ap_class_index is not None:
            for i, class_id in enumerate(self.metrics.ap_class_index):
                if not self._is_unknown(class_id):
                    class_name = IMAGENET[class_id]
                    class_name = class_name[0:19] # truncate to fit in column
                    img_count = self.images_per_class[i] if self.images_per_class is not None else 0
                    logger.info(pf % (
                        class_name,
                        img_count,
                        self.metrics.class_top1[class_id],
                        self.metrics.class_topk[class_id]
                    ))

    def _predictions_to_array(self, image_results: Dict) -> numpy.ndarray:
        """
        Convert image predictions dict to a structured numpy array.
        Args:
            image_results: Dictionary of class predictions for an image.
        Returns:
            Numpy array of shape (N,2) containing [class_idx, score] rows.
        """
        if not image_results:
            return numpy.zeros((0, 2), dtype=numpy.float32)
        all_preds = []
        for class_name, scores in image_results.items():
            score = scores[0]  # take the first score
            class_idx = self._parse_class_id(class_name)
            all_preds.append([class_idx, score])
        return numpy.array(all_preds, dtype=numpy.float32)

    def _ground_truth_to_array(self, gt_class: Optional[str]) -> numpy.ndarray:
        """
        Convert ground truth class string to a numpy array.
        Args:
            gt_class: Ground truth class string or None.
        Returns:
            Numpy array containing the ground truth class index.
        """
        class_idx = self._parse_class_id(gt_class)
        return numpy.array([class_idx], dtype=numpy.int64)

    def _parse_class_id(self, class_name_or_id) -> int:
        """
        Parse class ID from various formats (n00000000, string ID, or numeric ID)
        Args:
            class_name_or_id: Class identifier in string or numeric format
        Returns:
            int: Numeric class ID
        """
        if class_name_or_id is None:
            return self.UNKNOWN_CLASS_ID
        if isinstance(class_name_or_id, int):
            return class_name_or_id
        if isinstance(class_name_or_id, str):
            if class_name_or_id.startswith('n'):
                return int(class_name_or_id[1:])
            return int(class_name_or_id)
        raise ValueError(f"Unsupported class identifier format: {class_name_or_id}")

    def _is_unknown(self, class_id: int) -> bool:
        """
        Check if a class ID represents an unknown class

        Note:
            Classes are considered "unknown" if they are either:
            1. Valid ImageNet classes that don't appear in the ImageNet-A or ImageNet-O subsets
            2. Not legitimate ImageNet classes at all
        Args:
            class_id: Numeric class ID
        Returns:
            bool: True if this is the sentinel value for empty class
        """
        return class_id == self.UNKNOWN_CLASS_ID or class_id not in IMAGENET.keys()


class EvalKaNN_ObjDetect(EvalKaNN_Base):
    """
    A class extending the EvalKaNN_Base class for validating results from a detection task.

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

    def init_metrics(self):
        """
        Initialize metrics with class information.
        """
        self.metrics.names = self.id_to_names
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

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
                if gt_bboxes else numpy.zeros((0, 4), dtype=numpy.float32)
            )
            gt_cls_array = (
                numpy.array(gt_cls, dtype=numpy.int64)
                if gt_cls else numpy.zeros(0, dtype=numpy.int64)
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


class Metrics(object):
    """
    Class for computing evaluation metrics.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, mf1, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], f1[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        """Initialize a Metric instance for computing evaluation metrics for object-detection model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """
        Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (numpy.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (numpy.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Return the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Return the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def mf1(self):
        """
        Return the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.f1.mean() if len(self.f1) else 0.0

    @property
    def map50(self):
        """
        Return the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Return the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Return mean of results, mp, mr, map50, map."""
        return [self.mp, self.mr, self.mf1, self.map50, self.map]

    def class_result(self, i):
        """Return class-aware result, p[i], r[i], f1[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.f1[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """Return mAP of each class."""
        maps = numpy.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Return model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.0, 0.1, 0.9]  # weights for [P, R, F1, mAP@0.5, mAP@0.5:0.95]
        return (numpy.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Update the evaluation metrics with a new set of results.

        Args:
            results (tuple): A tuple containing evaluation metrics:
                - p (list): Precision for each class.
                - r (list): Recall for each class.
                - f1 (list): F1 score for each class.
                - all_ap (list): AP scores for all classes and all IoU thresholds.
                - ap_class_index (list): Index of class for each AP score.
                - p_curve (list): Precision curve for each class.
                - r_curve (list): Recall curve for each class.
                - f1_curve (list): F1 curve for each class.
                - px (list): X values for the curves.
                - prec_values (list): Precision values for each class.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self):
        """Return a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Return a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class ClassifyMetrics(object):
    """
    Class for computing classification metrics including top-1 and top-k accuracy.

    Attributes:
        top1 (float): The top-1 accuracy.
        topk (float): The top-k accuracy (by default 5).
        class_top1 (dict): Per-class top-1 accuracy.
        class_topk (dict): Per-class top-k accuracy.
        class_counts (dict): Number of instances per class.
        ap_class_index (list): List of class indices with data.
        speed (Dict[str, float]): A dictionary containing the time taken for each step in the pipeline.
        fitness (float): The fitness of the model, which is equal to top-k accuracy.
        results_dict (Dict[str, Union[float, str]]): A dictionary containing the classification metrics and fitness.
        keys (List[str]): A list of keys for the results_dict.

    Methods:
        process(targets, pred): Processes the targets and predictions to compute classification metrics.
        class_result(i): Returns metrics for a specific class index.
    """

    def __init__(self) -> None:
        """Initialize a ClassifyMetrics instance."""
        self.top1 = 0
        self.topk = 0
        self.class_top1 = {}
        self.class_topk = {}
        self.class_counts = {}
        self.ap_class_index = []
        self.names = {}
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "classify"

    def process(self, targets, pred):
        """Target classes and predicted classes."""
        pred, targets = numpy.concatenate(pred), numpy.concatenate(targets)
        correct = (targets[:, None] == pred).astype(numpy.float32)
        acc = numpy.column_stack((correct[:, 0], correct.max(1)))  # (top1, topk) accuracy
        self.top1, self.topk = acc.mean(0).tolist()
        unique_classes = numpy.unique(targets)
        self.ap_class_index = unique_classes.tolist()
        # Compute per-class accuracy
        for cls in unique_classes:
            cls_indices = numpy.where(targets == cls)[0]
            cls_count = len(cls_indices)
            self.class_counts[int(cls)] = cls_count
            if cls_count > 0:
                cls_correct = correct[cls_indices]
                self.class_top1[int(cls)] = float(cls_correct[:, 0].mean())
                self.class_topk[int(cls)] = float(cls_correct.max(1).mean())

    def class_result(self, i):
        """Return accuracy metrics for a specific class index."""
        cls = self.ap_class_index[i]
        return self.class_top1.get(cls, 0), self.class_topk.get(cls, 0)

    @property
    def fitness(self):
        """Returns mean of top-1 and top-k accuracies as fitness score."""
        return (self.top1 + self.topk) / 2

    @property
    def results_dict(self):
        """Returns a dictionary with model's performance metrics and fitness score."""
        return dict(zip(self.keys + ["fitness"], [self.top1, self.topk, self.fitness]))

    @property
    def keys(self):
        """Returns a list of keys for the results_dict property."""
        return ["metrics/accuracy_top1", "metrics/accuracy_topk"]

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []


class ObjDetectMetrics(object):
    """
    Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class.
        names (dict): A dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'detect'.
    """

    def __init__(self, names={}) -> None:
        """
        Initialize a DetMetrics instance with class names.

        Args:
            names (dict, optional): Dictionary mapping class indices to names.
        """
        self.names = names
        self.box = Metrics()
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }
        self.task = "detect"

    def process(self, tp, conf, pred_cls, target_cls):
        """
        Process predicted results for object detection and update metrics.

        Args:
            tp (numpy.ndarray): True positive array.
            conf (numpy.ndarray): Confidence array.
            pred_cls (numpy.ndarray): Predicted class indices array.
            target_cls (numpy.ndarray): Target class indices array.
        """
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            names=self.names,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """Return a list of keys for accessing specific metrics."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/f1-score(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Return mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Return the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Return the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Return dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Return a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
        ]

    @property
    def curves_results(self):
        """Return dictionary of computed performance metrics and statistics."""
        return self.box.curves_results


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        box1 (numpy.ndarray): An array of shape (N, 4) representing N bounding boxes.
        box2 (numpy.ndarray): An array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        numpy.ndarray: An NxM array containing the pairwise IoU values for every element in box1 and box2.
    """
    # Convert to float for accurate calculations
    box1 = box1.astype(numpy.float32)
    box2 = box2.astype(numpy.float32)
    # Expand dimensions for broadcasting
    # box1 becomes (N, 1, 4), box2 becomes (1, M, 4)
    a = numpy.expand_dims(box1, 1)  # (N, 1, 4)
    b = numpy.expand_dims(box2, 0)  # (1, M, 4)
    # Split coordinates
    a_x1, a_y1, a_x2, a_y2 = numpy.split(a, 4, axis=-1)  # Each becomes (N, 1, 1)
    b_x1, b_y1, b_x2, b_y2 = numpy.split(b, 4, axis=-1)  # Each becomes (1, M, 1)
    # Calculate intersection area
    x_min = numpy.maximum(a_x1, b_x1)
    y_min = numpy.maximum(a_y1, b_y1)
    x_max = numpy.minimum(a_x2, b_x2)
    y_max = numpy.minimum(a_y2, b_y2)
    intersection = numpy.maximum(x_max - x_min, 0) * numpy.maximum(y_max - y_min, 0)
    # Calculate areas
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    # Calculate union
    union = area_a + area_b - intersection
    # Calculate IoU
    iou = intersection / (union + eps)
    return iou.squeeze(-1)  # Remove the last dimension to get (N, M)


def ap_per_class(
    tp,
    conf,
    pred_cls,
    target_cls,
    names={},
    eps=1e-16,
):
    """
    Compute the average precision per class for object detection evaluation.

    Args:
        tp (numpy.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (numpy.ndarray): Array of confidence scores of the detections.
        pred_cls (numpy.ndarray): Array of predicted classes of the detections.
        target_cls (numpy.ndarray): Array of true classes of the detections.
        names (dict, optional): Dict of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        tp (numpy.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (numpy.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (numpy.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (numpy.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (numpy.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (numpy.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (numpy.ndarray): An array of unique classes that have data.
        p_curve (numpy.ndarray): Precision curves for each class.
        r_curve (numpy.ndarray): Recall curves for each class.
        f1_curve (numpy.ndarray): F1-score curves for each class.
        x (numpy.ndarray): X-axis values for the curves.
        prec_values (numpy.ndarray): Precision values at mAP@0.5 for each class.
    """
    # Sort by objectness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections
    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = numpy.linspace(0, 1, 1000), []
    # Average precision, precision and recall curves
    ap, p_curve, r_curve = (
        numpy.zeros((nc, tp.shape[1])),
        numpy.zeros((nc, 1000)),
        numpy.zeros((nc, 1000)),
    )
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = numpy.interp(
            -x, -conf[i], recall[:, 0], left=0
        )  # negative x, xp because xp decreases
        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = numpy.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score
        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(numpy.interp(x, mrec, mpre))  # precision at mAP@0.5
    prec_values = (
        numpy.array(prec_values) if prec_values else numpy.zeros((1, 1000))
    )  # (nc, 1000)
    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [
        v for k, v in names.items() if k in unique_classes
    ]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = (
        p_curve[:, i],
        r_curve[:, i],
        f1_curve[:, i],
    )  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return (
        tp,
        fp,
        p,
        r,
        f1,
        ap,
        unique_classes.astype(int),
        p_curve,
        r_curve,
        f1_curve,
        x,
        prec_values,
    )


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (numpy.ndarray): Precision envelope curve.
        (numpy.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = numpy.concatenate(([0.0], recall, [1.0]))
    mpre = numpy.concatenate(([1.0], precision, [0.0]))
    # Compute the precision envelope
    mpre = numpy.flip(numpy.maximum.accumulate(numpy.flip(mpre)))
    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = numpy.trapz(numpy.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = numpy.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode="valid")  # y-smoothed
