"""Utility methods to provide unified logging utilities."""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def default_formatter(fmt: Optional[str] = None, **kwargs) -> logging.Formatter:
    fmt = fmt if fmt is not None else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    return logging.Formatter(fmt, **kwargs)


def default_logger(
    name: str,
    logger_level: Optional[int] = logging.DEBUG,
    stream_level: Optional[int] = logging.WARN,
    stream_formatter: Optional[str] = None,
    file_level: Optional[int] = None,
    file_formatter: Optional[str] = None,
    file_name: Optional[str] = None,
    file_mode: str = "w",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logger_level)

    if stream_level is not None:
        fmt = stream_formatter if stream_formatter is not None else default_formatter()
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_level)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    if file_level is not None:
        fmt = file_formatter if file_formatter is not None else default_formatter()
        if file_name is None:
            raise ValueError("file_name must be specified if file_level is not None.")
        #file_name = file_name if file_name is not None else os.path.join(Config()["DIR_PATHS"]["LOGS"], "logs.txt")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        file_handler = RotatingFileHandler(file_name, mode=file_mode)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
