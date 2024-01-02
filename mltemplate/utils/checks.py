"""Utility methods relating to general object checks."""
from typing import Any


def ifnone(val: Any, default: Any):
    """Return the given value if it is not None, else return the default."""
    return val if val is not None else default
