# zyx._logging.py tests

import pytest
import logging
from zyx import _logging
from rich import print


def test_logging():

    # ensure logger is initialized
    assert _logging.logger is not None
    assert _logging.logger.name == "zyx"
    assert isinstance(_logging.logger, logging.Logger)
    print("Logger initialized: ", _logging.logger)

    # check default logging & library flag level
    assert _logging.logger.level == 0
    print("Logger level on init: ", _logging.logger.level)
    # check default verbosity level is 0 (silent)
    assert _logging.get_verbosity_level() == 0
    print("Verbosity level on init: ", _logging.get_verbosity_level())

    # enable verbose mode
    _logging.set_verbose(True)
    # check the verbosity level
    assert _logging.get_verbosity_level() == 1
    print("Verbosity level after set_verbose(True): ", _logging.get_verbosity_level())

    # test `print_verbose` method
    

    # disable verbose mode
    _logging.set_verbose(False)
    # check the verbosity level
    assert _logging.get_verbosity_level() == 0
    print("Verbosity level after set_verbose(False): ", _logging.get_verbosity_level())

    # check debug mode
    _logging.set_debug(True)
    # check logging level
    assert _logging.logger.level == 10
    print("Logger level after set_debug(True): ", _logging.logger.level)

    # disable debug mode
    _logging.set_debug(False)
    # check logging level
    assert _logging.logger.level == 30
    print("Logger level after set_debug(False): ", _logging.logger.level)


if __name__ == "__main__":
    test_logging()
