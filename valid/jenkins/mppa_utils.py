#! /usr/bin/env python3

###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import subprocess


def get_mppa_frequency():
    """ Return the frequencies set for the MPPA's clusters """
    mppa_freq_hz = []
    with open('/mppa/board0/mppa0/freq', 'r') as f_hw:
        for l in f_hw.readlines():
            txt = str(l).rstrip()
            mppa_freq_hz.append(float(txt.split(' ')[2]))
    return mppa_freq_hz


def get_sw_kenv():
    """ Provide software environment information from toolchain and kann """
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
        print(err)
    kann_version = kann.__version__
    return kvx_version, kann_version
