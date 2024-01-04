"""Unit test methods for mltemplate.utils.logging utility module."""
import logging
import os

from mltemplate import Config
from mltemplate.utils import default_logger


def test_logger():
    logs_filename = Config()["FILE_PATHS"]["UNITTEST_LOGS"]
    logger = default_logger(
        name="mltemplate", stream_level=logging.DEBUG, file_level=logging.DEBUG, file_name=logs_filename
    )

    logger.debug("debug log")
    logger.info("info log")
    logger.warning("warning log")
    logger.error("error log")
    logger.critical("critical log")

    assert os.path.exists(logs_filename)
    with open(logs_filename, mode="r", encoding="utf-8") as logs:
        assert len(logs.readlines()) == 5
