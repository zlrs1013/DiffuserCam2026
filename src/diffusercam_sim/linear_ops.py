"""Padded linear-convolution operators for lensless imaging.

The Waller Lab DiffuserCam tutorial models the sensor measurement as a cropped
linear convolution. FFTs naturally compute circular convolution, so we embed the
scene and PSF in a larger padded array, compute the convolution there, and crop
back to the sensor size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .fft_ops import centered_fft2, centered_ifft2


FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""

    if value < 1:
        raise ValueError("value must be positive")
    return int(2 ** np.ceil(np.log2(value)))


def padded_linear_convolution_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    """Return a power-of-two padded shape large enough for linear convolution."""

    rows, cols = image_shape
    return next_power_of_two(2 * rows - 1), next_power_of_two(2 * cols - 1)


@dataclass(frozen=True)
class CenterCropGeometry:
    """Centered crop/pad geometry shared by the forward and adjoint operators."""

    image_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    row_start: int
    col_start: int

    @classmethod
    def from_shapes(cls, image_shape: tuple[int, int], padded_shape: tuple[int, int]) -> "CenterCropGeometry":
        rows, cols = image_shape
        padded_rows, padded_cols = padded_shape
        if padded_rows < rows or padded_cols < cols:
            raise ValueError(f"padded_shape {padded_shape} must contain image_shape {image_shape}")

        return cls(
            image_shape=image_shape,
            padded_shape=padded_shape,
            row_start=(padded_rows - rows) // 2,
            col_start=(padded_cols - cols) // 2,
        )

    @property
    def row_end(self) -> int:
        return self.row_start + self.image_shape[0]

    @property
    def col_end(self) -> int:
        return self.col_start + self.image_shape[1]

    def pad(self, image: FloatArray | ComplexArray) -> FloatArray | ComplexArray:
        """Place a sensor-sized image at the center of a padded array."""

        if image.shape != self.image_shape:
            raise ValueError(f"Expected image shape {self.image_shape}, got {image.shape}")

        padded = np.zeros(self.padded_shape, dtype=image.dtype)
        padded[self.row_start : self.row_end, self.col_start : self.col_end] = image
        return padded

    def crop(self, padded: FloatArray | ComplexArray) -> FloatArray | ComplexArray:
        """Extract the centered sensor crop from a padded array."""

        if padded.shape != self.padded_shape:
            raise ValueError(f"Expected padded shape {self.padded_shape}, got {padded.shape}")

        return padded[self.row_start : self.row_end, self.col_start : self.col_end]


class PaddedLinearConvolution:
    """Linear convolution with a finite sensor crop and an FFT implementation.

    The optimization variable is stored on the padded grid. The sensor sees only
    the centered crop of the convolution result. 
    """

    def __init__(self, psf: FloatArray, padded_shape: tuple[int, int] | None = None) -> None:
        if psf.ndim != 2:
            raise ValueError(f"Only 2D grayscale PSFs are supported, got shape {psf.shape}")

        self.image_shape = tuple(int(v) for v in psf.shape)
        self.padded_shape = padded_shape or padded_linear_convolution_shape(self.image_shape)
        self.geometry = CenterCropGeometry.from_shapes(self.image_shape, self.padded_shape)

        self.psf = np.asarray(psf, dtype=np.float64)
        self.padded_psf = self.geometry.pad(self.psf)
        self.transfer_function = centered_fft2(self.padded_psf)
        self.adjoint_transfer_function = np.conj(self.transfer_function)
        self.lipschitz_bound = float(np.max(np.abs(self.transfer_function) ** 2))

    def pad(self, image: FloatArray | ComplexArray) -> FloatArray | ComplexArray:
        return self.geometry.pad(image)

    def crop(self, padded: FloatArray | ComplexArray) -> FloatArray | ComplexArray:
        return self.geometry.crop(padded)

    def forward_padded(self, padded_scene: FloatArray | ComplexArray) -> FloatArray:
        """Apply ``A`` to a padded scene estimate and return the sensor crop."""

        if padded_scene.shape != self.padded_shape:
            raise ValueError(f"Expected padded scene shape {self.padded_shape}, got {padded_scene.shape}")

        spectrum = centered_fft2(padded_scene)
        convolved = centered_ifft2(self.transfer_function * spectrum)
        return np.real(self.crop(convolved)).astype(np.float64, copy=False)

    def forward(self, scene: FloatArray) -> FloatArray:
        """Apply ``A`` to a sensor-sized scene by padding it first."""

        return self.forward_padded(self.pad(np.asarray(scene, dtype=np.float64)))

    def adjoint_padded(self, sensor_residual: FloatArray) -> FloatArray:
        """Apply ``A^H`` to a sensor-sized residual and return a padded array."""

        if sensor_residual.shape != self.image_shape:
            raise ValueError(f"Expected residual shape {self.image_shape}, got {sensor_residual.shape}")

        padded_residual = self.pad(np.asarray(sensor_residual, dtype=np.float64))
        spectrum = centered_fft2(padded_residual)
        adjoint = centered_ifft2(self.adjoint_transfer_function * spectrum)
        return np.real(adjoint).astype(np.float64, copy=False)

    def gradient_padded(self, padded_scene: FloatArray, measurement: FloatArray) -> FloatArray:
        """Return the data-fidelity gradient ``A^H(Ax - b)``."""

        residual = self.forward_padded(padded_scene) - measurement
        return self.adjoint_padded(residual)

    def objective(self, padded_scene: FloatArray, measurement: FloatArray) -> float:
        """Return ``0.5 * ||A x - b||_2^2``."""

        residual = self.forward_padded(padded_scene) - measurement
        return float(0.5 * np.sum(residual * residual))

    def adjoint_inner_product_error(self, seed: int = 0) -> float:
        """Numerically check ``<Ax, y> = <x, A^H y>`` for this operator."""

        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.padded_shape)
        y = rng.standard_normal(self.image_shape)
        lhs = float(np.vdot(self.forward_padded(x), y).real)
        rhs = float(np.vdot(x, self.adjoint_padded(y)).real)
        return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
