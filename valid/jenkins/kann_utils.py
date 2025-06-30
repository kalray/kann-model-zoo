#! /usr/bin/env python3

###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import time
import subprocess

from test_utils import logger
from test_utils import PASS, FAIL, WORKSPACE_PATH


def thread_generate(cfg, num, odir, sep="."):

    """ Try to generate a neural networks using "generate" script
        If the generated DIR & .kann extension exists, generation is skipped
        Otherwise generation runs in a subprocess.
        @param cfg:     path str, path of the YAML file to generate
        @param num:     str, unique test ID
        @param odir:    str path, output directory for generation
    """

    def try_generate():

        cmd = [f"{WORKSPACE_PATH}/./generate"]
        args = ["-c", cfg, "-d", generated_path, "-f"]
        logger.info(f"Generating ({num}) : {cfg}")
        os.makedirs(os.path.dirname(generated_path), exist_ok=True)
        t_start = time.perf_counter()
        try:
            with open(f"{generated_path}.log", 'w+') as flog:
                subprocess.run(
                    cmd + args,
                    stdout=flog,
                    stderr=flog,
                    check=True,
                    env=os.environ)
            t_gen_sec = time.perf_counter() - t_start
            logger.info(f'GEN {nn_name:15s}({os.path.basename(cfg)}) - #ID {num} : > {PASS} PASS ({t_gen_sec:.2f} sec)')
        except Exception as err:
            t_gen_sec = time.perf_counter() - t_start
            logger.error(f"GEN {nn_name:15s} {os.path.basename(cfg)} - #ID {num} : > {err}")
            with open(f"{generated_path}.log", "r") as flog:
                log_fail_text = flog.readlines()[-10:]
                log_fail_text = " ".join(log_fail_text)
            logger.error(f"GEN {nn_name} - #ID {num} : > {FAIL} FAIL ({t_gen_sec:.2f} sec) "
                  f">> \n***\n{log_fail_text}***\n")

    if os.environ.get("KANN_CACHE_DIR") is None:
       os.environ["KANN_CACHE_DIR"] = os.path.join(WORKSPACE_PATH, ".kann_cache")
    nn_type, nn_name, nn_fwk = cfg.split(os.sep)[-4:-1]
    yaml_file = os.path.basename(cfg).replace(".yaml", "")
    generated_path = os.path.join(odir, nn_type, f"{num}{sep}{nn_name}{sep}{nn_fwk}{sep}{yaml_file}")
    if not os.path.exists(generated_path):
        try_generate()
    else:
        logger.info(f"GEN :{nn_name} ({os.path.basename(cfg)}) - #ID {num} has been already generated at {generated_path} ")
        binfiles = [f for f in os.listdir(generated_path) if f.split('.')[-1] == 'kann']
        if len(binfiles) == 0:
            try_generate()
