"""Colab-specific patches for functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import PIL


def cv2_imshow(a: np.ndarray):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.

    Args:
        a (np.ndarray): Array of shape (N, M) or (N, M, 1) for an NxM grayscale image, or shape (N, M, 3) for an NxM BGR
                        color image, or shape (N, M, 4) for an NxM BGRA color image.
    """
    a = a.clip(0, 255).astype('uint8')

    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    PIL.Image.fromarray(a).show()
