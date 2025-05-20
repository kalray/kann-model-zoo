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
from kann_utils import logger, kHelpFormat, kHelp

URL_HF_PATH = "https://huggingface.co/Kalray/"


def get_model_from(url, dest_model_path):
    model_dir = os.path.dirname(dest_model_path)
    logger.info("Model requested directory: {}".format(model_dir))
    logger.info("Model requested path:      {}".format(dest_model_path))

    if not os.path.exists(dest_model_path):
        logger.warning('Model does not exists, trying to download from 🤗')
        model_name = model_dir.split("/")[-3]
        model_filename = os.path.basename(dest_model_path)
        model_url = os.path.join(
            url, model_name, "resolve", "main", model_filename)
        model_url += "?download=true"
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"request to {model_url}")
        with requests.get(model_url, stream=True) as response:
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024
                with tqdm(total=total_size,
                          unit="B", unit_scale=True,
                          desc="Download file from 🤗 {}".format(URL_HF_PATH)) \
                        as progress_bar:
                    with open(dest_model_path, "wb+") as handle:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            handle.write(data)
                status = progress_bar.n == total_size
            else:
                status = False
        if not status:
            logger.error('Model does not exists on our 🤗 platform ... 😢')
            logger.error('Please contact us to support@kalrayinc.com or report the issue to https://github.com/kalray/kann-model-zoo/issues')
            sys.exit(1)


def main(args, other_args):
     # List the neural networks available in the owner file system
    # located at <repo>/networks/ DIR path
    models = dict()
    networks_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "networks"))
    for d in sorted(os.listdir(networks_dir)):
        if os.path.isdir(os.path.join(networks_dir, d)):
            models[d] = [nn for nn in sorted(os.listdir(os.path.join(networks_dir, d)))
                            if os.path.isdir(os.path.join(networks_dir, d, nn))]
    models_list = [v for nn in models.values() for v in nn]

    # Print the neural networks available
    # from models dict()
    if args.list:
        print("\nList of available neural networks:\n")
        for t in models:
            print(f"** {t.upper()} **")
            for i, nn in enumerate(models[t]):
                print(f" {i:02d}. {nn.lower()}")
            print("")
        sys.exit(0)

    # Define the YAML path network DIR from args
    # 1. from the list determined above
    # 2. directly from the relative yaml path
    else:
        if args.config_yaml_path is not None:
            yaml_file_path = args.config_yaml_path
            network_dir = os.path.dirname(yaml_file_path)
            if not os.path.isfile(yaml_file_path):
                raise FileNotFoundError(f"{yaml_file_path} is not a regular YAML file or does not exist ... ")

        elif args.network and args.network.lower() in models_list:
            model_name = args.network
            model_family = [t for t, nn in models.items() for v in nn if v == model_name]
            assert len(model_family) == 1
            model_family = model_family[0]

            # add an interaction with the user if multiple configurations are found
            network_dir = os.path.join(networks_dir, model_family, model_name, args.framework)
            list_of_yaml = sorted([f for f in os.listdir(network_dir) if f.split('.')[-1] == "yaml" and args.dtype in f])
            if len(list_of_yaml) > 1:
                print("\nList of available configurations:\n")
                [print(f"  {i} - {f}") for i, f in enumerate(sorted(list_of_yaml))]
                try:
                    a = int(input("\nEnter configuration file number ? "))
                    file_name = sorted(list_of_yaml)[int(a)]
                except:
                    sys.exit(1)
            elif len(list_of_yaml) == 1:
                file_name = sorted(list_of_yaml)[0]
            else:
                raise RuntimeError(f"There are no YAML file in {network_dir}")
            yaml_file_path = os.path.join(network_dir, file_name)

        else:
            logger.error(f"Network required not found or not available, get {args.network}\n")
            # parser.print_help()
            sys.exit(1)

    # Get the configuration from YAML path
    with open(yaml_file_path, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.Loader)

    # Check framework used, ONNX is actually the only one supported
    # since ACE >= 6.0.0
    framework = cfg.get('framework')
    if framework.lower() == "onnx":
        model_path = os.path.abspath(
            os.path.join(network_dir, cfg.get('onnx_model')))
    else:
        print(f"Unknown framework, {framework} not supported !")
        sys.exit(1)

    # Check if model exists locally
    # Otherwise it download it from HuggingFace
    if not os.path.isfile(model_path):
        get_model_from(URL_HF_PATH, model_path)

    # Finally generate the model with KaNN(tm)
    if args.debug:
        import kann
        kann.commons.log_utils.initialize("debug")
        kann.generate(yaml_file_path, dest_dir="test", log_smem_alloc=True, generate_txt_cmds=True, force=True)
    else:
        cmd_args = ["kann", "generate", yaml_file_path] + other_args
        subprocess.run(cmd_args, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, formatter_class=kHelpFormat)
    parser.add_argument(
        "--config_yaml_path", "-c",
        required=False,
        help="Model YAML path",
    )
    parser.add_argument(
        "--network", "-n",
        required=False,
        help="Select model, please use ./generate --list to print all networks available",
    )
    parser.add_argument(
        "--framework", default="onnx",
        required=False,
        help="Select framework if different from ONNX",
    )
    parser.add_argument(
        "--dtype", default="f16",
        required=False, choices=["f16", "i8"],
        help="Select neural network datatype computation",
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List all networks available"
    )
    parser.add_argument(
        "--use-nfs", action="store_true",
        help="use interal url path, located on nfs (ACI and/or internal use only)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run generation with kaNN python API"
    )
    parser.add_argument(
        "--help", "-h", action=kHelp, nargs=0,
        help="Display this message"
    )
    opt, other_opt = parser.parse_known_args()
    print(opt, other_opt)
    main(opt, other_opt)
