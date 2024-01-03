"""Utility methods to provide unified logging utilities."""
import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Optional

from mltemplate import Config
from mltemplate.utils import ifnone


def default_formatter(fmt: Optional[str] = None, **kwargs) -> logging.Formatter:
    fmt = ifnone(fmt, default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.Formatter(fmt, **kwargs)


def default_logger(
        name: str,
        logger_level: Optional[int] = logging.DEBUG,
        stream_level: Optional[int] = logging.WARN,
        stream_formatter: Optional[str] = None,
        file_level: Optional[int] = None,
        file_formatter: Optional[str] = None,
        file_name: Optional[str] = None,
        file_mode: str = 'w'
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logger_level)

    if stream_level is not None:
        fmt = ifnone(stream_formatter, default=default_formatter())
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_level)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    if file_level is not None:
        fmt = ifnone(file_formatter, default=default_formatter())
        file_name = ifnone(file_name, default=os.path.join(Config()['DIR_PATHS']['LOGS'], 'logs.txt'))
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(file_name, mode=file_mode)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
