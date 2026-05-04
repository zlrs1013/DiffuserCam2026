"""Load and preprocess Waller-Lab DiffuserCam tutorial sample data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class WallerSample:
    """Preprocessed Waller tutorial PSF and raw sensor measurement."""

    psf: FloatArray
    measurement: FloatArray
    psf_raw: FloatArray
    measurement_raw: FloatArray
    background: float
    downsample_factor: float
    psf_path: Path
    measurement_path: Path


def load_grayscale_float(path: Path) -> FloatArray:
    """Load a grayscale image as ``float64`` without display normalization."""

    image = Image.open(path)
    return np.asarray(image, dtype=np.float64)


def box_downsample_power_of_two(image: FloatArray, factor: float) -> FloatArray:
    """Downsample by repeated 2 by 2 averaging.

    This matches the Waller tutorial's simple demosaicing/downsampling step.
    ``factor`` must be one of ``1, 1/2, 1/4, 1/8, ...``.
    """

    if factor <= 0 or factor > 1:
        raise ValueError("factor must be in the interval (0, 1]")

    levels_float = -np.log2(factor)
    levels = int(round(levels_float))
    if not np.isclose(levels, levels_float):
        raise ValueError("factor must be a power-of-two fraction such as 1/4 or 1/8")

    downsampled = np.asarray(image, dtype=np.float64)
    for _ in range(levels):
        rows = downsampled.shape[0] - downsampled.shape[0] % 2
        cols = downsampled.shape[1] - downsampled.shape[1] % 2
        downsampled = downsampled[:rows, :cols]
        downsampled = 0.25 * (
            downsampled[::2, ::2]
            + downsampled[1::2, ::2]
            + downsampled[::2, 1::2]
            + downsampled[1::2, 1::2]
        )

    return downsampled


def l2_normalize(image: FloatArray) -> FloatArray:
    """Normalize an image by its Euclidean norm."""

    norm = float(np.linalg.norm(np.ravel(image)))
    if norm <= 0:
        raise ValueError("Cannot normalize an all-zero image")
    return np.asarray(image, dtype=np.float64) / norm


def preprocess_waller_tutorial_sample(
    psf_path: Path,
    measurement_path: Path,
    downsample_factor: float = 1 / 8,
) -> WallerSample:
    """Load and preprocess the Waller tutorial PSF/raw-data pair.

    The preprocessing mirrors the tutorial notebooks:

    1. Load 16-bit TIFF images.
    2. Estimate camera background from ``psf[5:15, 5:15]``.
    3. Subtract that same background from PSF and measurement.
    4. Downsample by repeated 2 by 2 averaging.
    5. Normalize PSF and measurement to unit L2 norm.
    """

    psf_raw = load_grayscale_float(psf_path)
    measurement_raw = load_grayscale_float(measurement_path)
    if psf_raw.shape != measurement_raw.shape:
        raise ValueError(f"PSF and measurement must match, got {psf_raw.shape} and {measurement_raw.shape}")

    background = float(np.mean(psf_raw[5:15, 5:15]))
    psf_background_subtracted = psf_raw - background
    measurement_background_subtracted = measurement_raw - background

    psf_downsampled = box_downsample_power_of_two(psf_background_subtracted, downsample_factor)
    measurement_downsampled = box_downsample_power_of_two(measurement_background_subtracted, downsample_factor)

    return WallerSample(
        psf=l2_normalize(psf_downsampled),
        measurement=l2_normalize(measurement_downsampled),
        psf_raw=psf_raw,
        measurement_raw=measurement_raw,
        background=background,
        downsample_factor=downsample_factor,
        psf_path=psf_path,
        measurement_path=measurement_path,
    )


def find_waller_tutorial_paths(project_root: Path) -> tuple[Path, Path]:
    """Find the Waller tutorial sample PSF and raw-data files in common layouts."""

    candidates = [
        project_root / "data" / "external" / "tutorial",
        project_root / "data" / "external" / "waller_lab_diffusercam_tutorial" / "tutorial",
    ]

    for directory in candidates:
        psf_path = directory / "psf_sample.tif"
        measurement_path = directory / "rawdata_hand_sample.tif"
        if psf_path.exists() and measurement_path.exists():
            return psf_path, measurement_path

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find Waller tutorial sample files. Searched:\n{searched}")
