import numpy as np
from numpy.typing import NDArray


def bmi_array(arr: list[float]) -> NDArray:
    """Wrapper ensure the expected numpy array datatype is used."""
    return np.array(arr, dtype='float32')
