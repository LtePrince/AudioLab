"""Validated product-only resource options; training defaults are unchanged."""
import argparse
import math

DEFAULT_MIN_AVAILABLE_RAM_GIB = 8.0


def validate_ram_reserve(value):
    """Keep at least 1 GiB protected even when the user selects a smaller reserve."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float('nan')
    if isinstance(value, bool) or not math.isfinite(number) or number < 1:
        raise ValueError('min-available-ram-gib must be finite and at least 1 GiB')
    return number


def ram_reserve_argument(value):
    try:
        return validate_ram_reserve(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
