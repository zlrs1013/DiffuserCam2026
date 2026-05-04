"""ADMM reconstruction with anisotropic total variation regularization.

This module follows the variable splitting used in the Waller Lab
DiffuserCam ADMM tutorial. The optimization variable lives on the padded FFT
grid, the measured sensor image is represented by a center crop, and total
variation is implemented with circular finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .fft_ops import centered_fft2, centered_ifft2
from .linear_ops import PaddedLinearConvolution
from .metrics import residual_relative_l2


FloatArray = NDArray[np.floating]


@dataclass
class ADMMRecord:
    """One row of ADMM diagnostics."""

    iteration: int
    objective: float
    data_fit: float
    tv_value: float
    relative_residual_l2: float
    nonnegativity_violation_l2: float


@dataclass
class ADMMResult:
    """Final ADMM reconstruction and diagnostic history."""

    reconstruction: FloatArray
    padded_estimate: FloatArray
    history: list[ADMMRecord]
    parameters: dict[str, float]
    snapshots: dict[int, FloatArray]
    method: str = "admm_total_variation"


def finite_difference(image: FloatArray) -> FloatArray:
    """Circular forward differences used by the Waller tutorial.

    Returns an array with shape ``(rows, cols, 2)``. The first channel is the
    row difference and the second channel is the column difference.
    """

    return np.stack(
        (
            np.roll(image, 1, axis=0) - image,
            np.roll(image, 1, axis=1) - image,
        ),
        axis=2,
    )


def finite_difference_adjoint(gradient_stack: FloatArray) -> FloatArray:
    """Adjoint of :func:`finite_difference`."""

    if gradient_stack.ndim != 3 or gradient_stack.shape[2] != 2:
        raise ValueError(f"Expected gradient stack shape (rows, cols, 2), got {gradient_stack.shape}")

    row_adjoint = np.roll(gradient_stack[..., 0], -1, axis=0) - gradient_stack[..., 0]
    col_adjoint = np.roll(gradient_stack[..., 1], -1, axis=1) - gradient_stack[..., 1]
    return row_adjoint + col_adjoint


def soft_threshold(values: FloatArray, threshold: float) -> FloatArray:
    """Elementwise soft-thresholding for anisotropic TV."""

    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def total_variation_anisotropic(image: FloatArray) -> float:
    """Return ``||Psi image||_1`` using anisotropic finite differences."""

    return float(np.sum(np.abs(finite_difference(image))))


def convolve_full(operator: PaddedLinearConvolution, padded_scene: FloatArray) -> FloatArray:
    """Full padded convolution without the sensor crop."""

    spectrum = centered_fft2(padded_scene)
    convolved = centered_ifft2(operator.transfer_function * spectrum)
    return np.real(convolved).astype(np.float64, copy=False)


def convolve_full_adjoint(operator: PaddedLinearConvolution, padded_image: FloatArray) -> FloatArray:
    """Adjoint of full padded convolution."""

    spectrum = centered_fft2(padded_image)
    adjoint = centered_ifft2(operator.adjoint_transfer_function * spectrum)
    return np.real(adjoint).astype(np.float64, copy=False)


def _precompute_x_divisor(operator: PaddedLinearConvolution, mu1: float) -> FloatArray:
    crop_mask = operator.pad(np.ones(operator.image_shape, dtype=np.float64))
    return 1.0 / (crop_mask + mu1)


def _precompute_psi_t_psi_spectrum(padded_shape: tuple[int, int]) -> FloatArray:
    kernel = np.zeros(padded_shape, dtype=np.float64)
    kernel[0, 0] = 4.0
    kernel[0, 1] = -1.0
    kernel[1, 0] = -1.0
    kernel[0, -1] = -1.0
    kernel[-1, 0] = -1.0
    return np.real(np.fft.fft2(kernel)).astype(np.float64, copy=False)


def _precompute_v_divisor(operator: PaddedLinearConvolution, mu1: float, mu2: float, mu3: float) -> FloatArray:
    mtm_component = mu1 * np.abs(operator.transfer_function) ** 2
    psi_component = mu2 * np.abs(_precompute_psi_t_psi_spectrum(operator.padded_shape))
    return 1.0 / (mtm_component + psi_component + mu3)


def _diagnostics(
    *,
    operator: PaddedLinearConvolution,
    padded_estimate: FloatArray,
    measurement: FloatArray,
    tau: float,
    iteration: int,
) -> ADMMRecord:
    full_prediction = convolve_full(operator, padded_estimate)
    sensor_prediction = operator.crop(full_prediction)
    residual = sensor_prediction - measurement
    data_fit = float(0.5 * np.sum(residual * residual))
    tv_value = total_variation_anisotropic(padded_estimate)
    objective = data_fit + tau * tv_value
    negative_part = np.minimum(padded_estimate, 0.0)
    return ADMMRecord(
        iteration=iteration,
        objective=objective,
        data_fit=data_fit,
        tv_value=tv_value,
        relative_residual_l2=residual_relative_l2(measurement, sensor_prediction),
        nonnegativity_violation_l2=float(np.linalg.norm(np.ravel(negative_part))),
    )


def admm_total_variation(
    *,
    operator: PaddedLinearConvolution,
    measurement: FloatArray,
    iterations: int,
    tau: float = 1.0e-4,
    mu1: float = 1.0e-6,
    mu2: float = 1.0e-5,
    mu3: float = 4.0e-5,
    record_every: int = 1,
    snapshot_iterations: set[int] | None = None,
) -> ADMMResult:
    """Run Waller-style ADMM with TV and nonnegativity.

    Solves the split problem used by the tutorial:

    ``0.5 ||b - Cx||_2^2 + tau ||u||_1``

    subject to ``x = Mv``, ``u = Psi v``, and ``w = v, w >= 0``.
    """

    measurement = np.asarray(measurement, dtype=np.float64)
    if measurement.shape != operator.image_shape:
        raise ValueError(f"Expected measurement shape {operator.image_shape}, got {measurement.shape}")

    x = np.zeros(operator.padded_shape, dtype=np.float64)
    u = np.zeros((*operator.padded_shape, 2), dtype=np.float64)
    v = np.zeros(operator.padded_shape, dtype=np.float64)
    w = np.zeros(operator.padded_shape, dtype=np.float64)
    xi = np.zeros(operator.padded_shape, dtype=np.float64)
    eta = np.zeros((*operator.padded_shape, 2), dtype=np.float64)
    rho = np.zeros(operator.padded_shape, dtype=np.float64)

    ct_measurement = operator.pad(measurement)
    x_divisor = _precompute_x_divisor(operator, mu1)
    v_divisor = _precompute_v_divisor(operator, mu1, mu2, mu3)
    history: list[ADMMRecord] = []
    requested_snapshots = set(snapshot_iterations or set())
    snapshots: dict[int, FloatArray] = {}

    for iteration in range(iterations + 1):
        feasible_v = np.maximum(v, 0.0)
        if iteration in requested_snapshots:
            snapshots[iteration] = operator.crop(feasible_v).copy()

        if iteration % record_every == 0 or iteration == iterations:
            history.append(
                _diagnostics(
                    operator=operator,
                    padded_estimate=feasible_v,
                    measurement=measurement,
                    tau=tau,
                    iteration=iteration,
                )
            )

        if iteration == iterations:
            break

        u = soft_threshold(finite_difference(v) + eta / mu2, tau / mu2)
        x = x_divisor * (xi + mu1 * convolve_full(operator, v) + ct_measurement)
        right_hand_side = (
            (mu3 * w - rho)
            + finite_difference_adjoint(mu2 * u - eta)
            + convolve_full_adjoint(operator, mu1 * x - xi)
        )
        v_spectrum = v_divisor * centered_fft2(right_hand_side)
        v = np.real(centered_ifft2(v_spectrum)).astype(np.float64, copy=False)

        w = np.maximum(rho / mu3 + v, 0.0)
        xi = xi + mu1 * (convolve_full(operator, v) - x)
        eta = eta + mu2 * (finite_difference(v) - u)
        rho = rho + mu3 * (v - w)

    feasible_v = np.maximum(v, 0.0)
    return ADMMResult(
        reconstruction=operator.crop(feasible_v),
        padded_estimate=feasible_v,
        history=history,
        parameters={"tau": tau, "mu1": mu1, "mu2": mu2, "mu3": mu3},
        snapshots=snapshots,
    )
