"""Synthetic scenes, PSFs, and noise models for simulation notebooks/scripts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont


FloatArray = NDArray[np.floating]


def make_synthetic_scene(size: int = 256) -> FloatArray:
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


def make_diffuser_like_psf(size: int = 256, seed: int = 4, spot_count: int = 100) -> FloatArray:
    """Create a deterministic sparse caustic PSF for lensless toy problems."""

    rng = np.random.default_rng(seed)
    psf = np.zeros((size, size), dtype=np.float64)
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


def add_gaussian_noise(signal: FloatArray, snr_db: float, seed: int = 10) -> tuple[FloatArray, float]:
    """Add zero-mean Gaussian noise at a requested SNR."""

    rng = np.random.default_rng(seed)
    signal_rms = float(np.sqrt(np.mean(signal * signal)))
    noise_sigma = signal_rms / (10.0 ** (snr_db / 20.0))
    noisy = signal + rng.normal(0.0, noise_sigma, signal.shape)
    return noisy.astype(np.float64), noise_sigma
