"""Minimal 2D lensless-imaging simulation.

This script demonstrates the shift-invariant toy model:

    y = psf * x + noise

using circular convolution so Wiener deconvolution is transparent in the
Fourier domain. It saves numeric metrics plus a visual montage under
``results/minimal_2d``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diffusercam_sim import (  
    centered_fft2,
    centered_ifft2,
    circular_convolve2d,
    psnr,
    residual_relative_l2,
    ssim,
)


RESULT_DIR = PROJECT_ROOT / "results" / "minimal_2d"


def make_synthetic_scene(size: int = 256) -> np.ndarray:
    """Generate a simple grayscale test scene with edges and smooth structure."""

    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle((26, 30, 96, 104), fill=215)
    draw.ellipse((138, 28, 225, 115), fill=175)
    draw.line((26, 185, 230, 142), fill=230, width=7)
    draw.rectangle((38, 138, 92, 214), outline=255, width=5)
    draw.polygon([(164, 168), (211, 222), (116, 224)], fill=150)

    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except OSError:
        font = ImageFont.load_default()
    draw.text((112, 108), "2D", fill=255, font=font)

    scene = np.asarray(img, dtype=np.float64) / 255.0
    yy, xx = np.mgrid[:size, :size]
    gradient = 0.18 * np.exp(-(((xx - 72) / 65) ** 2 + ((yy - 196) / 45) ** 2))
    return np.clip(scene + gradient, 0.0, 1.0)


def rick_scene(size: int = 256, seed: int = 7) -> np.ndarray:
    """Generate a rice-like grayscale scene."""

    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), 18)

    for _ in range(95):
        cx = int(rng.integers(8, size - 8))
        cy = int(rng.integers(8, size - 8))
        length = int(rng.integers(10, 22))
        width = int(rng.integers(3, 6))
        angle = float(rng.uniform(0.0, 180.0))
        value = int(rng.integers(145, 245))

        grain = Image.new("L", (2 * length, 2 * length), 0)
        grain_draw = ImageDraw.Draw(grain)
        box = (
            length - length // 2,
            length - width,
            length + length // 2,
            length + width,
        )
        grain_draw.ellipse(box, fill=value)
        grain = grain.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
        img.paste(grain, (cx - length, cy - length), grain)

    scene = np.asarray(img, dtype=np.float64) / 255.0
    yy, xx = np.mgrid[:size, :size]
    illumination = 0.85 + 0.18 * (xx / max(size - 1, 1)) + 0.08 * (yy / max(size - 1, 1))
    return np.clip(scene * illumination, 0.0, 1.0)


def averaging_psf(size: int, kernel_size: int = 5) -> np.ndarray:
    """Create a center-origin same-size averaging PSF/blurring filter."""

    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd")

    psf = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    radius = kernel_size // 2
    psf[center - radius : center + radius + 1, center - radius : center + radius + 1] = 1.0
    psf /= np.sum(psf)
    return psf


def diffuser_like_psf(size: int = 256, seed: int = 4) -> np.ndarray:
    """Create a deterministic synthetic speckle-like PSF.

    Not a physical diffuser simulator. It is a compact stand-in for a
    measured calibration PSF: random, extended, nonnegative, and normalized.
    The sparse caustic spots keep the first exercise visually interpretable
    while still making direct inversion unstable.
    """

    rng = np.random.default_rng(seed)
    psf = np.zeros((size, size), dtype=np.float64)
    spot_count = 100
    coords = rng.integers(0, size, size=(spot_count, 2))
    weights = rng.random(spot_count) ** 2

    for (row, col), weight in zip(coords, weights):
        psf[row, col] += weight
        if row > 0:
            psf[row - 1, col] += 0.20 * weight
        if row < size - 1:
            psf[row + 1, col] += 0.20 * weight
        if col > 0:
            psf[row, col - 1] += 0.20 * weight
        if col < size - 1:
            psf[row, col + 1] += 0.20 * weight

    psf = np.fft.fftshift(psf)
    psf += 1.0e-4 * np.mean(psf)
    psf /= np.sum(psf)
    return psf.astype(np.float64)


def add_gaussian_noise(signal: np.ndarray, snr_db: float, seed: int = 10) -> tuple[np.ndarray, float]:
    """Add zero-mean Gaussian noise at a requested SNR."""

    rng = np.random.default_rng(seed)
    signal_rms = float(np.sqrt(np.mean(signal * signal)))
    noise_sigma = signal_rms / (10.0 ** (snr_db / 20.0))
    noisy = signal + rng.normal(0.0, noise_sigma, signal.shape)
    return noisy.astype(np.float64), noise_sigma


def direct_inverse(measurement: np.ndarray, psf: np.ndarray, epsilon: float) -> np.ndarray:
    """Naive Fourier inverse with a tiny floor to avoid division by zero."""

    h_fft = centered_fft2(psf)
    y_fft = centered_fft2(measurement)
    x_fft = y_fft / np.where(np.abs(h_fft) < epsilon, epsilon, h_fft)
    return np.real(centered_ifft2(x_fft))


def direct_inverse_matlab(measurement: np.ndarray, psf: np.ndarray, epsilon: float) -> np.ndarray:
    """Naive inverse matching the prof K's ``G ./ (H + epsilon)`` line."""

    h_fft = centered_fft2(psf)
    y_fft = centered_fft2(measurement)
    x_fft = y_fft / (h_fft + epsilon)
    return np.real(centered_ifft2(x_fft))


def wiener_deconvolution(measurement: np.ndarray, psf: np.ndarray, regularization: float) -> np.ndarray:
    """Fourier-domain Wiener deconvolution."""

    h_fft = centered_fft2(psf)
    y_fft = centered_fft2(measurement)
    x_fft = np.conj(h_fft) * y_fft / (np.abs(h_fft) ** 2 + regularization)
    return np.real(centered_ifft2(x_fft))


def normalize_for_display(image: np.ndarray, percentile_clip: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Map an array to uint8 for visualization."""

    lo, hi = np.percentile(image, percentile_clip)
    if hi <= lo:
        hi = lo + 1e-9
    scaled = (image - lo) / (hi - lo)
    return np.uint8(np.clip(scaled, 0.0, 1.0) * 255.0)


def add_title(tile: Image.Image, title: str, height: int = 26) -> Image.Image:
    """Add a compact title strip above a tile."""

    out = Image.new("RGB", (tile.width, tile.height + height), "white")
    out.paste(tile.convert("RGB"), (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), title, fill=(20, 20, 20), font=ImageFont.load_default())
    return out


def save_montage(images: list[tuple[str, np.ndarray]], path: Path) -> None:
    """Save a labeled 2 by 3 montage."""

    tiles = []
    for title, array in images:
        if title.lower().startswith("psf"):
            view = normalize_for_display(np.log1p(array / max(float(array.max()), 1e-12)))
        else:
            view = normalize_for_display(array)
        tiles.append(add_title(Image.fromarray(view, mode="L"), title))

    tile_w, tile_h = tiles[0].size
    montage = Image.new("RGB", (3 * tile_w, 2 * tile_h), "white")
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, 3)
        montage.paste(tile, (col * tile_w, row * tile_h))
    montage.save(path)


def run_simulation(
    *,
    scene: np.ndarray,
    psf: np.ndarray,
    result_dir: Path,
    snr_db: float | None,
    fixed_noise_sigma: float | None,
    wiener_lambda: float,
    inverse_epsilon: float,
    matlab_inverse: bool,
    clip_measurement: bool,
    label: str,
) -> None:
    """Run one forward/inverse/Wiener simulation and save outputs."""

    result_dir.mkdir(parents=True, exist_ok=True)

    clean_measurement = circular_convolve2d(scene, psf)

    if fixed_noise_sigma is not None:
        rng = np.random.default_rng(10)
        measurement = clean_measurement + fixed_noise_sigma * rng.standard_normal(clean_measurement.shape)
        noise_sigma = fixed_noise_sigma
        snr_db_effective = 20.0 * np.log10(
            float(np.sqrt(np.mean(clean_measurement * clean_measurement))) / max(noise_sigma, 1e-12)
        )
    elif snr_db is not None:
        measurement, noise_sigma = add_gaussian_noise(clean_measurement, snr_db)
        snr_db_effective = snr_db
    else:
        raise ValueError("Either snr_db or fixed_noise_sigma must be provided")

    if clip_measurement:
        measurement = np.clip(measurement, 0.0, 1.0)

    if matlab_inverse:
        inverse = direct_inverse_matlab(measurement, psf, inverse_epsilon)
    else:
        inverse = direct_inverse(measurement, psf, inverse_epsilon)
    inverse_clipped = np.clip(inverse, 0.0, 1.0)

    wiener = wiener_deconvolution(measurement, psf, wiener_lambda)
    wiener_clipped = np.clip(wiener, 0.0, 1.0)
    predicted = circular_convolve2d(wiener_clipped, psf)
    residual = predicted - measurement

    metrics = {
        "experiment": label,
        "model": "same-size circular convolution: y = psf * x + noise",
        "image_size": int(scene.shape[0]),
        "snr_db_requested": snr_db,
        "snr_db_effective": snr_db_effective,
        "fixed_noise_sigma": fixed_noise_sigma,
        "noise_sigma": noise_sigma,
        "clip_measurement_to_0_1": clip_measurement,
        "wiener_lambda": wiener_lambda,
        "direct_inverse_epsilon": inverse_epsilon,
        "direct_inverse_formula": "G / (H + epsilon)" if matlab_inverse else "G / H with magnitude floor",
        "measurement_psnr_vs_clean_db": psnr(clean_measurement, measurement, data_range=1.0),
        "direct_inverse_psnr_db": psnr(scene, inverse_clipped, data_range=1.0),
        "direct_inverse_ssim": ssim(scene, inverse_clipped, data_range=1.0),
        "wiener_psnr_db": psnr(scene, wiener_clipped, data_range=1.0),
        "wiener_ssim": ssim(scene, wiener_clipped, data_range=1.0),
        "wiener_relative_residual_l2": residual_relative_l2(measurement, predicted),
    }

    np.savez_compressed(
        result_dir / "minimal_2d_arrays.npz",
        scene=scene,
        psf=psf,
        clean_measurement=clean_measurement,
        measurement=measurement,
        direct_inverse=inverse_clipped,
        wiener=wiener_clipped,
        residual=residual,
    )
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_montage(
        [
            ("Ground truth scene", scene),
            ("PSF, log display", psf),
            ("Noisy sensor measurement", measurement),
            ("Direct inverse, clipped", inverse_clipped),
            ("Wiener reconstruction", wiener_clipped),
            ("Forward residual", np.abs(residual)),
        ],
        result_dir / "montage.png",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {result_dir}")


def run_diffuser_like_simulation() -> None:
    """Run the DiffuserCam-style toy simulation."""

    size = 256
    run_simulation(
        scene=make_synthetic_scene(size),
        psf=diffuser_like_psf(size),
        result_dir=RESULT_DIR / "diffuser_like",
        snr_db=15.0,
        fixed_noise_sigma=None,
        wiener_lambda=1.0e-4,
        inverse_epsilon=1.0e-8,
        matlab_inverse=False,
        clip_measurement=False,
        label="diffuser_like_sparse_random_psf",
    )


def run_lecture_simulation() -> None:
    """Replicate the prof K's MATLAB inverse/Wiener filtering demo."""

    size = 256
    run_simulation(
        scene=rick_scene(size),
        psf=averaging_psf(size=size, kernel_size=5),
        result_dir=RESULT_DIR / "lecture_reference",
        snr_db=None,
        fixed_noise_sigma=0.01,
        wiener_lambda=0.01,
        inverse_epsilon=1.0e-3,
        matlab_inverse=True,
        clip_measurement=True,
        label="lecture_5x5_average_psf",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("diffuser", "lecture", "both"),
        default="both",
        help="Which simulation to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"diffuser", "both"}:
        run_diffuser_like_simulation()
    if args.mode in {"lecture", "both"}:
        run_lecture_simulation()


if __name__ == "__main__":
    main()
