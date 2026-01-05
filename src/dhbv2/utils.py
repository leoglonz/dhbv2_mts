import numpy as np
from numpy.typing import NDArray
import torch


def bmi_array(arr: list[float]) -> NDArray:
    """Wrapper to ensure standardized numpy arrays in BMI."""
    return np.array(arr, dtype='float32')


class RingBuffer:
    """
    Fixed-size circular buffer for pytorch tensors.

    Handles rolling windows without memory reallocation and fragmentation.

    Parameters
    ----------
    shape
        Tuple defining the shape of the buffer, e.g., (timesteps, space, vars).
    dtype
        Data type of the buffer elements.
    device
        Device where the buffer is stored (e.g., 'cpu' or 'cuda').
    """

    def __init__(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float64,
        device: str = 'cpu',
    ) -> None:
        self.dtype = dtype
        self.device = device

        self.buffer = torch.zeros(shape, dtype=dtype, device=device)
        self.capacity = shape[0]
        self.ptr = 0
        self.is_full = False

    def append(self, item: torch.Tensor) -> None:
        """Overwrite the oldest item with new data.

        Parameters
        ----------
        item
            New data to append to the buffer. Expected shape: (space, vars).
        """
        self.buffer[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.is_full = True

    def get_ordered(self) -> torch.Tensor:
        """Return buffer.

        Returns
        -------
        torch.Tensor
            Buffer contents ordered from oldest to newest.
        """
        if not self.is_full:
            # Return data up to current ptr
            return self.buffer[: self.ptr]

        # Roll so the oldest data (currently at ptr) moves to index 0
        return torch.roll(self.buffer, shifts=-self.ptr, dims=0)

    def get_last(self) -> torch.Tensor:
        """Get the most recently added item.

        Returns
        -------
        torch.Tensor
            The last item added to the buffer. Shape (1, space, vars).
        """
        if self.ptr == 0 and not self.is_full:
            # Return empty if no items have been added
            return np.zeros((1, *self.buffer.shape[1:]), dtype=self.buffer.dtype)

        # The last item added is at ptr - 1
        idx = (self.ptr - 1) % self.capacity
        return self.buffer[idx : idx + 1]

    def __len__(self):
        return self.capacity if self.is_full else self.ptr
