#!/usr/bin/python
# -*- coding: UTF-8 -*-

# USED WHEN GPT-OSS AS SUPERVISED MODEL

import uuid
import colorlog  # pip install colorlog
import logging as logging_cx

_limit_level = {
    'DEBUG': logging_cx.DEBUG,
    'INFO': logging_cx.INFO,
    'WARNING':  logging_cx.WARNING,
    'ERROR': logging_cx.ERROR,
    'CRITICAL': logging_cx.CRITICAL}
_log_colors_config_common = {
    'DEBUG': 'cyan',
    'INFO': 'purple',
    'WARNING':  'green',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white'}

def Logging(limit_level='INFO', colorful=False):
    if colorful:
        fmt = colorlog.ColoredFormatter(
            "[%(asctime)s] *%(levelname)s* [%(module)s:%(funcName)s-%(lineno)d] %(log_color)s%(message)s", log_colors=_log_colors_config_common)
    else:
        fmt = logging_cx.Formatter(
            "[%(asctime)s] *%(levelname)s* [%(module)s:%(funcName)s-%(lineno)d] %(message)s")
    logger = logging_cx.getLogger(str(uuid.uuid1()))
    limit_level = _limit_level[limit_level.upper()]
    logger.setLevel(limit_level)
    stream_handle = logging_cx.StreamHandler()
    stream_handle.setFormatter(fmt)
    logger.addHandler(stream_handle)
    stream_handle.close()
    
    return logger

logging = Logging(limit_level='info', colorful=True)
