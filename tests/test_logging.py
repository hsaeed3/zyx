"""
Logging Tests
"""

import pytest
from zyx import logging


def test_logging():
    # [Ensure logger & console singleton]
    assert logging.logger is not None
    assert logging.console is not None

    # [Test Flags]
    assert logging.zyx_verbose is False
    logging.logger.info("This is a test message, that should NOT be printed")
    logging.set_zyx_verbose(True)
    logging.logger.info("This is a test message, that should be printed")


if __name__ == "__main__":
    test_logging()
