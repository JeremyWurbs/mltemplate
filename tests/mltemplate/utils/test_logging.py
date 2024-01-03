import logging
import os

from lunit import Config
from lunit.utils import default_logger


def test_logger():
    logs_filename = Config()['FILE_PATHS']['UNITTEST_LOGS']
    logger = default_logger(
        name='Lunit',
        stream_level=logging.DEBUG,
        file_level=logging.DEBUG,
        file_name=logs_filename)

    logger.debug('debug log')
    logger.info('info log')
    logger.warning('warning log')
    logger.error('error log')
    logger.critical('critical log')

    assert os.path.exists(logs_filename)
    with open(logs_filename, mode='r') as logs:
        assert len(logs.readlines()) == 5
