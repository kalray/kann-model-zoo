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
###
import os
import glob
import yaml
import shutil

from PIL import Image
from kann_utils import logger
from utils import COCO
from utils import IMAGENET
from utils import WORKSPACE_PATH


def check_dataset(dataset):
    """
    Check if the dataset is already present, else download it
    and place it in "./datasets" folder.

    Args:
        dataset (str): the name of the dataset ("coco8", "coco128", "coco",
            "imagenet-a" or "imagenet-o").
    """
    datasets_dir_path = dataset_path = os.path.join(WORKSPACE_PATH, "utils", "datasets")
    dataset_path = os.path.join(datasets_dir_path, dataset)
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
        elif dataset == "coco8" or dataset == "coco128":
            dataset_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip"
            os.system(f"wget {dataset_url} -P {WORKSPACE_PATH}/utils/datasets/")
            os.system(f"unzip {WORKSPACE_PATH}/utils/datasets/{dataset}.zip -d {WORKSPACE_PATH}/utils/datasets/")
            logger.info(f"Dataset {dataset} downloaded and extracted to {os.path.join(dataset_path, dataset)}")
        elif dataset == "imagenet-a" or "imagenet-o":
            print(f"WORKSPACE_PATH = {WORKSPACE_PATH}")
            dataset_url = f"https://people.eecs.berkeley.edu/~hendrycks/{dataset}.tar"
            os.system(f"wget {dataset_url} -P {WORKSPACE_PATH}/utils/datasets/")
            os.system(f"tar -xf {WORKSPACE_PATH}/utils/datasets/{dataset}.tar -C {WORKSPACE_PATH}/utils/datasets/")
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
    elif dataset == "imagenet-a" or dataset == "imagenet-o":
        data_img_path = os.path.join(dataset_path)
        if os.path.isdir(data_img_path):
            return data_img_path
    else:
        data_img_path = os.path.join(dataset_path, "images", "*")
        return data_img_path


def load_imagenet_references(dataset_img_path):
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
    for folder in os.listdir(dataset_img_path):
        if folder == "README.txt":
            continue  # Skip files, only process directories
        class_imgs_path = os.path.join(dataset_img_path, folder)
        for img in os.listdir(class_imgs_path):
            img = os.path.splitext(os.path.basename(img))[0]
            references[img] = folder
    return references


def load_coco_references(dataset_img_path):
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
            if class_id in COCO:
                class_name = COCO[class_id]
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
    if len(classes[0].split(" ")) == 1:         # line struct: <label_id>
        result = {id: name for id, name in enumerate(classes)}
    elif classes[0].split(" ")[0][0] == 'n':    # line struct: <n00000000 label_id>
        result = {int(id.split(" ")[0][1:]): id.split(" ")[-1] for id in classes}
    elif len(classes[0].split(" ")) == 2:       # line struct: <0 label_id>
        result = {int(id.split(" ")[0]): id.split(" ")[-1] for id in classes}
    else:
        raise ValueError(f"{class_path_file} format is not as expected <n00000000 label_id>, <0 label_id>, <label_id> per row-line")
    return result
