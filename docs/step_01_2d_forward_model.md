# Step 1: 2D Lensless Forward Model

## Goal

Build intuition for the simplest lensless imaging model before introducing real calibration data, cropping, sensor effects, or 3D depth.

## Incoherent Image Formation

For spatially incoherent light, intensities add linearly. If a point emitter at one scene location creates a shifted copy of the same sensor pattern, then the sensor image is approximately a convolution:

```text
y = h * x + n
```

- `x`: unknown scene intensity.
- `h`: point spread function (PSF), measured by imaging a point source.
- `y`: sensor measurement.
- `n`: noise and model error.

The more general linear model is:

```text
y = Hx + n
```

The convolution approximation is the special case where `H` is built from shifted copies of one PSF.

## Why the PSF Matters

The PSF is the system's calibration fingerprint. If you know how one point maps to the sensor, and if shifted points map to shifted PSFs, then a full scene is a weighted sum of shifted PSFs. Reconstruction is possible because the PSF tells us how scene structure was mixed.

## Fourier View

For the first simulation, we assume circular convolution. That makes the forward model diagonal in the Fourier domain:

```text
Y = H X + N
```

where `H` is the Fourier transform of the PSF. This is why the code can implement the forward model and Wiener reconstruction with FFTs.

## Why Direct Inversion Fails

The direct inverse is:

```text
X_hat = Y / H
```

When `|H|` is small, both sensor noise and calibration error are divided by a tiny number. Those frequencies blow up. This is the central reason computational imaging needs regularization.

## Wiener Reconstruction

Wiener/Tikhonov deconvolution stabilizes the inverse:

```text
X_hat = conj(H) Y / (|H|^2 + lambda)
```

`lambda` prevents extreme amplification where the PSF transfer function is weak. Larger `lambda` suppresses noise but blurs details; smaller `lambda` sharpens but can amplify noise.

## What This Step Deliberately Simplifies

- Uses circular convolution rather than padded linear convolution with sensor cropping.
- Uses a synthetic sparse caustic PSF rather than a measured DiffuserCam PSF.
- Uses Gaussian noise only.
- Clips reconstructions to `[0, 1]` for display and metrics.
- Does not yet model saturation, quantization, dark current, Bayer sampling, PSF mismatch, or depth.

These simplifications are useful because they isolate the first idea: a calibrated PSF defines the forward operator, and stable inversion requires regularization.
