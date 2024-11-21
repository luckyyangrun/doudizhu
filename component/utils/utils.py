#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File Name:     utils.py
# Author:        Run yang
# Created Time:  2024-11-21  09:18
# Last Modified: <none>-<none>


import logging
import functools
import os
import sys


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(name, save_dir, distributed_rank, filename="log.txt", abbrev_name=None):
    """logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "ugait" if name == "ugait" else name
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger