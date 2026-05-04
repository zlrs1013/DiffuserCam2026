"""Image-quality metrics implemented with NumPy only."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def mse(reference: FloatArray, estimate: FloatArray) -> float:
    """Mean squared error."""

    diff = np.asarray(reference, dtype=np.float64) - np.asarray(estimate, dtype=np.float64)
    return float(np.mean(diff * diff))


def psnr(reference: FloatArray, estimate: FloatArray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB."""

    error = mse(reference, estimate)
    if error == 0:
        return math.inf
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(error))


def residual_relative_l2(measurement: FloatArray, predicted: FloatArray) -> float:
    """Relative L2 residual, ||predicted - measurement|| / ||measurement||."""

    numerator = np.linalg.norm(np.ravel(predicted - measurement))
    denominator = np.linalg.norm(np.ravel(measurement))
    return float(numerator / max(denominator, 1e-12))


def _mean_filter_reflect(image: FloatArray, window_size: int) -> FloatArray:
    if window_size % 2 != 1:
        raise ValueError("window_size must be odd")

    radius = window_size // 2
    padded = np.pad(np.asarray(image, dtype=np.float64), radius, mode="reflect")
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)))
    total = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    return total / float(window_size * window_size)


def ssim(
    reference: FloatArray,
    estimate: FloatArray,
    data_range: float = 1.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural similarity index using a local square window.

    This is a compact NumPy implementation of the standard SSIM formula. It
    uses a uniform window instead of the common Gaussian window so the first
    simulation has no SciPy/scikit-image dependency.
    """

    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(estimate, dtype=np.float64)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ux = _mean_filter_reflect(x, window_size)
    uy = _mean_filter_reflect(y, window_size)
    ux2 = ux * ux
    uy2 = uy * uy
    uxuy = ux * uy

    vx = _mean_filter_reflect(x * x, window_size) - ux2
    vy = _mean_filter_reflect(y * y, window_size) - uy2
    vxy = _mean_filter_reflect(x * y, window_size) - uxuy

    numerator = (2.0 * uxuy + c1) * (2.0 * vxy + c2)
    denominator = (ux2 + uy2 + c1) * (vx + vy + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))
