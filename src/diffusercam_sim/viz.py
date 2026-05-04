"""Matplotlib visualization helpers for saved simulation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def normalize_for_display(image: FloatArray, percentile_clip: tuple[float, float] = (1.0, 99.0)) -> NDArray[np.float64]:
    """Map an array to the display range [0, 1] using robust clipping."""

    lo, hi = np.percentile(image, percentile_clip)
    if hi <= lo:
        hi = lo + 1e-9
    scaled = (image - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float64, copy=False)


def _record_value(record: object, field_name: str) -> float:
    return float(getattr(record, field_name))


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_montage(images: list[tuple[str, FloatArray]], path: Path, columns: int = 3) -> None:
    """Save a labeled image montage using Matplotlib."""

    if not images:
        raise ValueError("images must contain at least one item")

    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.2 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")

    for axis, (title, array) in zip(axes.ravel(), images):
        display_array = np.asarray(array)
        if title.lower().startswith("psf"):
            display_array = np.log1p(display_array / max(float(np.max(display_array)), 1e-12))
        axis.imshow(normalize_for_display(display_array), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title, fontsize=10)

    fig.tight_layout()
    _save_figure(fig, path)


def save_convergence_plot(histories: list[tuple[str, list[object]]], path: Path) -> None:
    """Save objective and PSNR convergence plots."""

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for label, records in histories:
        iterations = [_record_value(record, "iteration") for record in records]
        objective = [_record_value(record, "objective") for record in records]
        psnr_values = [_record_value(record, "psnr_db") for record in records]
        axes[0].plot(iterations, objective, marker="o", markersize=3, linewidth=1.8, label=label)
        axes[1].plot(iterations, psnr_values, marker="o", markersize=3, linewidth=1.8, label=label)

    axes[0].set_yscale("log")
    axes[0].set_title("Objective")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("0.5 ||Ax - b||^2")
    axes[1].set_title("Reconstruction Quality")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("PSNR (dB)")

    for axis in axes:
        axis.grid(True, alpha=0.28)
        axis.legend(frameon=False)

    fig.tight_layout()
    _save_figure(fig, path)


def save_objective_residual_plot(histories: list[tuple[str, list[object]]], path: Path) -> None:
    """Save objective and residual convergence for data without ground truth."""

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for label, records in histories:
        iterations = [_record_value(record, "iteration") for record in records]
        objective = [_record_value(record, "objective") for record in records]
        residual = [_record_value(record, "relative_residual_l2") for record in records]
        axes[0].plot(iterations, objective, marker="o", markersize=3, linewidth=1.8, label=label)
        axes[1].plot(iterations, residual, marker="o", markersize=3, linewidth=1.8, label=label)

    axes[0].set_yscale("log")
    axes[0].set_title("Objective")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Data fit + regularization")
    axes[1].set_title("Forward-Model Residual")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Relative L2 residual")

    for axis in axes:
        axis.grid(True, alpha=0.28)
        axis.legend(frameon=False)

    fig.tight_layout()
    _save_figure(fig, path)


def save_two_panel_metric_plot(
    *,
    x_values: list[float],
    left_series: list[tuple[str, list[float]]],
    right_series: list[tuple[str, list[float]]],
    path: Path,
    left_title: str,
    right_title: str,
    x_label: str,
    left_y_label: str,
    right_y_label: str,
    log_x: bool = False,
    log_left_y: bool = False,
    log_right_y: bool = False,
) -> None:
    """Save a two-panel Matplotlib line plot for metric sweeps."""

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    def plot_panel(
        axis: plt.Axes,
        *,
        title: str,
        y_label: str,
        series: list[tuple[str, list[float]]],
        log_y: bool,
    ) -> None:
        for label, values in series:
            axis.plot(x_values, values, marker="o", markersize=4, linewidth=1.8, label=label)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        if log_x:
            axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.grid(True, alpha=0.28)
        axis.legend(frameon=False)

    plot_panel(axes[0], title=left_title, y_label=left_y_label, series=left_series, log_y=log_left_y)
    plot_panel(axes[1], title=right_title, y_label=right_y_label, series=right_series, log_y=log_right_y)
    fig.tight_layout()
    _save_figure(fig, path)


def save_table_image(
    *,
    rows: list[dict[str, object]],
    columns: list[tuple[str, str]],
    path: Path,
    title: str,
) -> None:
    """Render a small table as a Matplotlib figure."""

    if not columns:
        raise ValueError("columns must contain at least one item")

    labels = [label for _, label in columns]
    cell_text = [[str(row.get(key, "")) for key, _ in columns] for row in rows]
    fig_height = max(1.8, 0.36 * (len(rows) + 2))
    fig_width = max(6.0, 1.35 * len(columns))
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))
    axis.axis("off")
    axis.set_title(title, fontsize=12, pad=10)

    table = axis.table(cellText=cell_text, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)

    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e9edf2")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f7f7f7")

    fig.tight_layout()
    _save_figure(fig, path)
