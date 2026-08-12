"""Library logging factory."""

import logging


def get_custom_logger(name: str) -> logging.Logger:
    """Return a standard module logger without configuring the host process."""
    return logging.getLogger(name)
