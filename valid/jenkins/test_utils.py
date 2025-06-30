#! /usr/bin/env python3

###
# Copyright (C) 2025 Kalray SA. All rights reserved.
# This code is Kalray proprietary and confidential.
# Any use of the code for whatever purpose is subject
# to specific written permission of Kalray SA.
###

import os
import sys
import logging


PASS = "\U00002705"
FAIL = "\U0000274C"
DEMO_SOURCE = "cat.jpg"
WORKSPACE_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__ + "/../.."))))

SOURCES = {
    "class" : [f"{WORKSPACE_PATH}/utils/sources/cat.jpg"],
    "objec" : [f"{WORKSPACE_PATH}/utils/sources/birds.jpg"],
    "segme" : [f"{WORKSPACE_PATH}/utils/sources/dog.jpg"],
    "visio" : [],
    "priva" : [],
}

class cFormatter(logging.Formatter):

    grey = "\x1b[37;2m"
    white = "\x1b[38;0m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;1m"
    bold_red = "\x1b[35;1m"
    reset = "\x1b[0;0m"
    format = "[%(levelname)s]: %(message)s"
    format_err = "%(asctime)s | [%(levelname)s]: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format_err + reset,
        logging.ERROR: red + format_err + reset,
        logging.CRITICAL: bold_red + format_err + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# create logger
logger = logging.getLogger("SINGLE-TESTS")
logger.setLevel(logging.INFO)

stdout_console_handler = logging.StreamHandler(sys.stdout)
stdout_console_handler.setFormatter(cFormatter())
logger.addHandler(stdout_console_handler)

logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
logger.error("error message")
logger.critical("critical message")
