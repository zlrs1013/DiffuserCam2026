"""FFT conventions for lensless-imaging simulations.

Real-space images are stored with the visual origin at the center. Fourier
transforms move that origin to the array corner with ``ifftshift`` first.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]


def centered_fft2(image: FloatArray | ComplexArray) -> ComplexArray:
    """Return the 2D FFT of a center-origin image."""

    return np.fft.fft2(np.fft.ifftshift(image))


def centered_ifft2(spectrum: ComplexArray) -> ComplexArray:
    """Return the inverse 2D FFT as a center-origin image."""

    return np.fft.fftshift(np.fft.ifft2(spectrum))


def circular_convolve2d(image: FloatArray, psf: FloatArray) -> FloatArray:
    """Convolve two same-shaped 2D arrays using circular boundaries."""

    if image.shape != psf.shape:
        raise ValueError(f"image and psf must have the same shape, got {image.shape} and {psf.shape}")

    result = centered_ifft2(centered_fft2(image) * centered_fft2(psf))
    return np.real(result).astype(np.float64, copy=False)
