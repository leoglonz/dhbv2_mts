import numpy as np
from numpy.typing import NDArray


def bmi_array(arr: list[float]) -> NDArray:
    """Wrapper to ensure standardized numpy arrays in BMI."""
    return np.array(arr, dtype='float32')
