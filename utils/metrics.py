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
#   Daniel Angulo, <dangulo@kalrayinc.com
###

import numpy


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
