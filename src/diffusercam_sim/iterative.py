"""Projected gradient and FISTA solvers for lensless deconvolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .linear_ops import PaddedLinearConvolution
from .metrics import psnr, residual_relative_l2, ssim


FloatArray = NDArray[np.floating]


@dataclass
class IterationRecord:
    """One row of reconstruction diagnostics."""

    iteration: int
    objective: float
    relative_residual_l2: float
    psnr_db: float | None = None
    ssim: float | None = None


@dataclass
class ReconstructionResult:
    """Final reconstruction and its diagnostic history."""

    reconstruction: FloatArray
    padded_estimate: FloatArray
    history: list[IterationRecord]
    step_size: float
    method: str


def center_support_mask(operator: PaddedLinearConvolution) -> FloatArray:
    """Return a padded mask that permits nonzero values only in the center crop."""

    return operator.pad(np.ones(operator.image_shape, dtype=np.float64))


def validate_support_mask(operator: PaddedLinearConvolution, support_mask: FloatArray | None) -> FloatArray | None:
    """Validate and convert an optional support mask."""

    if support_mask is None:
        return None

    mask = np.asarray(support_mask, dtype=np.float64)
    if mask.shape != operator.padded_shape:
        raise ValueError(f"support_mask must have shape {operator.padded_shape}, got {mask.shape}")
    return np.where(mask > 0, 1.0, 0.0)


def project_nonnegative(values: FloatArray) -> FloatArray:
    """Project intensities onto the nonnegative orthant."""

    return np.maximum(values, 0.0)


def project_feasible(values: FloatArray, support_mask: FloatArray | None = None) -> FloatArray:
    """Project onto nonnegative intensities and an optional support mask."""

    projected = project_nonnegative(values)
    if support_mask is not None:
        projected = projected * support_mask
    return projected


def default_step_size(
    operator: PaddedLinearConvolution,
    l2_regularization: float = 0.0,
    safety: float = 1.8,
) -> float:
    """Return a conservative gradient step size from the FFT Lipschitz bound."""

    lipschitz = operator.lipschitz_bound + l2_regularization
    if lipschitz <= 0:
        raise ValueError("Operator has zero Lipschitz bound")
    return float(safety / lipschitz)


def constant_initialization(operator: PaddedLinearConvolution, measurement: FloatArray) -> FloatArray:
    """Initialize with a small center-crop constant, then pad with zeros.

    This matches the Waller tutorial convention: the initial image estimate has
    sensor size, and the padded optimization variable is created by padding that
    estimate into the larger FFT grid.
    """

    pixel_start = 0.5 * (float(np.max(measurement)) + float(np.min(measurement)))
    center_estimate = np.full(operator.image_shape, max(pixel_start, 0.0), dtype=np.float64)
    return operator.pad(center_estimate)


def diagnostics(
    *,
    operator: PaddedLinearConvolution,
    padded_estimate: FloatArray,
    measurement: FloatArray,
    truth: FloatArray | None,
    iteration: int,
    support_mask: FloatArray | None = None,
    l2_regularization: float = 0.0,
) -> IterationRecord:
    """Compute objective, residual, and optional image-quality metrics."""

    feasible_estimate = project_feasible(padded_estimate, support_mask)
    reconstruction = operator.crop(feasible_estimate)
    predicted = operator.forward_padded(feasible_estimate)
    objective = operator.objective(feasible_estimate, measurement)
    if l2_regularization > 0:
        objective += float(0.5 * l2_regularization * np.sum(feasible_estimate * feasible_estimate))

    record = IterationRecord(
        iteration=iteration,
        objective=objective,
        relative_residual_l2=residual_relative_l2(measurement, predicted),
    )

    if truth is not None:
        record.psnr_db = psnr(truth, np.clip(reconstruction, 0.0, 1.0), data_range=1.0)
        record.ssim = ssim(truth, np.clip(reconstruction, 0.0, 1.0), data_range=1.0)

    return record


def projected_gradient_descent(
    *,
    operator: PaddedLinearConvolution,
    measurement: FloatArray,
    iterations: int,
    step_size: float | None = None,
    truth: FloatArray | None = None,
    record_every: int = 1,
    support_mask: FloatArray | None = None,
    l2_regularization: float = 0.0,
) -> ReconstructionResult:
    """Solve nonnegative least squares with optional support and L2 penalties."""

    mask = validate_support_mask(operator, support_mask)
    alpha = step_size or default_step_size(operator, l2_regularization=l2_regularization)
    estimate = project_feasible(constant_initialization(operator, measurement), mask)
    history: list[IterationRecord] = []

    for iteration in range(iterations + 1):
        if iteration % record_every == 0 or iteration == iterations:
            history.append(
                diagnostics(
                    operator=operator,
                    padded_estimate=estimate,
                    measurement=measurement,
                    truth=truth,
                    iteration=iteration,
                    support_mask=mask,
                    l2_regularization=l2_regularization,
                )
            )

        if iteration == iterations:
            break

        gradient = operator.gradient_padded(estimate, measurement)
        if l2_regularization > 0:
            gradient = gradient + l2_regularization * estimate
        estimate = project_feasible(estimate - alpha * gradient, mask)

    return ReconstructionResult(
        reconstruction=operator.crop(project_feasible(estimate, mask)),
        padded_estimate=estimate,
        history=history,
        step_size=alpha,
        method="projected_gradient_descent",
    )


def fista(
    *,
    operator: PaddedLinearConvolution,
    measurement: FloatArray,
    iterations: int,
    step_size: float | None = None,
    truth: FloatArray | None = None,
    record_every: int = 1,
    support_mask: FloatArray | None = None,
    l2_regularization: float = 0.0,
) -> ReconstructionResult:
    """Solve with FISTA using nonnegativity, optional support, and optional L2."""

    mask = validate_support_mask(operator, support_mask)
    alpha = step_size or default_step_size(operator, l2_regularization=l2_regularization)
    estimate = project_feasible(constant_initialization(operator, measurement), mask)
    momentum_point = estimate.copy()
    t_value = 1.0
    history: list[IterationRecord] = []

    for iteration in range(iterations + 1):
        if iteration % record_every == 0 or iteration == iterations:
            history.append(
                diagnostics(
                    operator=operator,
                    padded_estimate=estimate,
                    measurement=measurement,
                    truth=truth,
                    iteration=iteration,
                    support_mask=mask,
                    l2_regularization=l2_regularization,
                )
            )

        if iteration == iterations:
            break

        previous_estimate = estimate
        gradient = operator.gradient_padded(momentum_point, measurement)
        if l2_regularization > 0:
            gradient = gradient + l2_regularization * momentum_point
        estimate = project_feasible(momentum_point - alpha * gradient, mask)
        next_t_value = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_value * t_value))
        momentum = (t_value - 1.0) / next_t_value
        momentum_point = estimate + momentum * (estimate - previous_estimate)
        t_value = next_t_value

    return ReconstructionResult(
        reconstruction=operator.crop(project_feasible(estimate, mask)),
        padded_estimate=estimate,
        history=history,
        step_size=alpha,
        method="fista",
    )
