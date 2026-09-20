"""Shared fixtures for dhbv2 BMI tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from dhbv2.bmi import DeltaModelBmi
from dhbv2.mts_bmi import MtsDeltaModelBmi


def pytest_addoption(parser):
    """Register the validation suite's --run-dir option."""
    parser.addoption(
        '--run-dir',
        action='store',
        default=None,
        help=(
            "Directory holding your own ngen run of the shipped cat-2453 "
            "example: a cat-2453.csv, plus troute_output_*.nc if you routed. "
            "Without it the ngen and t-route validation legs skip."
        ),
    )


@pytest.fixture
def daily_bmi():
    """Fresh DeltaModelBmi instance (not initialized).

    Provides access to constructor defaults, BMI info methods,
    variable dicts, and helper methods without requiring config
    files or model weights.
    """
    return DeltaModelBmi(verbose=False)


@pytest.fixture
def mts_bmi():
    """Fresh MtsDeltaModelBmi instance (not initialized).

    Provides access to constructor defaults, BMI info methods,
    variable dicts, and helper methods without requiring config
    files or model weights.
    """
    return MtsDeltaModelBmi(verbose=False)


@pytest.fixture
def daily_bmi_with_accumulator(daily_bmi):
    """DeltaModelBmi with a pre-filled day accumulator for aggregation tests.

    Sets up a 3-variable (P, T, PET) accumulator with 24 hours of
    synthetic data and Penman-Monteith PET method.
    """
    n_vars = 3
    daily_bmi._day_accumulator = np.zeros((24, 1, n_vars), dtype=np.float64)

    # Fill with known hourly values
    for h in range(24):
        daily_bmi._day_accumulator[h, 0, 0] = 0.5  # P: 0.5 mm/hr each hour
        daily_bmi._day_accumulator[h, 0, 1] = 15.0 + 5.0 * np.sin(
            2 * np.pi * h / 24,
        )  # T: diurnal cycle around 15C
        daily_bmi._day_accumulator[h, 0, 2] = max(
            0.0,
            0.3 * np.sin(2 * np.pi * (h - 6) / 24),
        )  # PET: daytime only

    daily_bmi._pet_method = 'penman_monteith'
    return daily_bmi
