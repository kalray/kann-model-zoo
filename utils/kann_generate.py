#! /usr/bin/env python3

###
# Copyright (C) 2024 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import sys
import yaml
import shutil
import argparse
import requests
import subprocess

from tqdm import tqdm
from kann_utils import logger


URL_HF_PATH = "https://huggingface.co/Kalray/"

class KannHelp(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print("*" * 80)
        print("This is a wrapper script to download model from 🤗 and generate model with kann")
        print("> Usage: kann_generate.py <network_yaml_path> [kann_generate_args]")
        print("*" * 80)
        print("\n $ kann generate --help for more informations (is detailed below)\n")
        cmd_args = ["kann", "generate", "--help"]
        subprocess.run(cmd_args, check=True)
        sys.exit(0)


def get_model_from(url, dest_dir):
    model_dir = os.path.dirname(model_path)
    logger.info("Model requested directory: {}".format(model_dir))
    logger.info("Model requested path:      {}".format(model_path))

    if not os.path.exists(model_path):
        logger.warning('Model does not exists, trying to download from 🤗')
        model_name = network_dir.split("/")[-2]
        model_filename = os.path.basename(model_path)
        model_url = os.path.join(
            URL_HF_PATH, model_name, "resolve", "main", model_filename)
        model_url += "?download=true"
        os.makedirs(model_dir, exist_ok=True)
        with requests.get(model_url, stream=True) as response:
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024
                with tqdm(total=total_size,
                          unit="B", unit_scale=True,
                          desc="Download file from 🤗 {}".format(URL_HF_PATH)) \
                        as progress_bar:
                    with open(model_path, "wb+") as handle:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            handle.write(data)
                status = progress_bar.n == total_size
            else:
                status = False
        if not status:
            logger.warning('Model does not exists on our 🤗 platform or download failed ... 😢')
            logger.warning(
                'It may happen that not all the models have been '
                'migrated to Kalray HF platform. Please contact support@kalrayinc.com '
                'to download the model.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("network_yaml_path", help="Model YAML path")
    parser.add_argument("--help", action=KannHelp, nargs=0,)
    parser.add_argument(
        "--use-nfs", action="store_true",
        help="use interal url path, located on nfs (ACI and/or internal use only)")
    parser.add_argument("--debug", action="store_true", help="Run generation with kaNN python API")

    args, other_args = parser.parse_known_args()
    print(args)
    yaml_file_path = args.network_yaml_path
    network_dir = os.path.dirname(yaml_file_path)
    with open(yaml_file_path, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.Loader)

    # check framework used
    framework = cfg.get('framework')
    if framework.lower() == "onnx":
        model_path = os.path.abspath(
            os.path.join(network_dir, cfg.get('onnx_model')))
    else:
        print(f"Unknown framework, {framework} not supported !")
        sys.exit(1)

    # Check if model exists
    if not os.path.isfile(model_path):
        get_model_from(URL_HF_PATH, model_path)

    # Finally generate
    if args.debug:
        import kann
        kann.commons.log_utils.initialize("debug")
        kann.generate(yaml_file_path, dest_dir="test", log_smem_alloc=True, generate_txt_cmds=True, force=True)
    else:
        cmd_args = ["kann", "generate", yaml_file_path] + other_args
        subprocess.run(cmd_args, check=True)
