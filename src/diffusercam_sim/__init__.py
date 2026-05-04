"""Small simulation utilities for DiffuserCam-style lensless imaging."""

from .admm import admm_total_variation
from .fft_ops import centered_fft2, centered_ifft2, circular_convolve2d
from .iterative import center_support_mask, fista, projected_gradient_descent
from .linear_ops import PaddedLinearConvolution, padded_linear_convolution_shape
from .metrics import mse, psnr, residual_relative_l2, ssim
from .phantoms import (
    add_gaussian_noise,
    make_averaging_psf,
    make_diffuser_like_psf,
    make_rice_like_scene,
    make_synthetic_scene,
)
from .waller_data import find_waller_tutorial_paths, preprocess_waller_tutorial_sample

__all__ = [
    "add_gaussian_noise",
    "admm_total_variation",
    "center_support_mask",
    "centered_fft2",
    "centered_ifft2",
    "circular_convolve2d",
    "fista",
    "find_waller_tutorial_paths",
    "mse",
    "PaddedLinearConvolution",
    "padded_linear_convolution_shape",
    "projected_gradient_descent",
    "preprocess_waller_tutorial_sample",
    "psnr",
    "make_averaging_psf",
    "make_diffuser_like_psf",
    "make_rice_like_scene",
    "make_synthetic_scene",
    "residual_relative_l2",
    "ssim",
]
