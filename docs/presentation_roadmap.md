# Presentation Roadmap

This talk should make one clean claim: the repo reproduces the core 2D DiffuserCam reconstruction pipeline and uses controlled experiments to explain why regularization and calibration matter.

## 1. Motivation: Why Lensless Imaging?

Start from the big picture. Lensless cameras replace bulky focusing optics with a thin mask, diffuser, coded aperture, or bare-sensor geometry plus computation. The review article by Boominathan, Robinson, Waller, and Veeraraghavan frames the attraction as compactness, lower weight/cost, scalable fabrication, wide field of view, privacy-preserving raw measurements, and computational/compressive imaging opportunities.

Presentation line: a lensless image is not immediately human-readable; it is a measurement that becomes an image only after we solve an inverse problem.

## 2. DiffuserCam Idea

DiffuserCam places a diffuser close to the image sensor. A point in the scene produces a distinctive sensor pattern, the point spread function (PSF). Under the 2D shift-invariant approximation, every scene point contributes a shifted copy of the same PSF.

Core equation:

```text
b = h * x + n
```

where `x` is the unknown scene, `h` is the calibrated PSF, `b` is the measured sensor image, and `n` is noise/model mismatch.

## 3. PSF Calibration

The PSF is the calibration object that turns the hardware into a known linear system. In the Waller tutorial sample, the measured PSF and raw hand data are preprocessed by background subtraction, power-of-two box downsampling, and L2 normalization.

Presentation line: calibration quality is reconstruction quality. If the PSF shifts, crops, saturates, or no longer matches the camera, the inverse problem solves the wrong system.

## 4. 2D Forward Model

Move from circular convolution to padded linear convolution:

```text
b = A x + n
```

The operator `A` pads the scene, applies FFT-based convolution, and crops the simulated sensor measurement. The adjoint `A^H` is verified numerically with an inner-product check.

Presentation line: the forward model is the part we can test directly. If `A` and `A^H` are wrong, every iterative solver becomes theater.

## 5. Inverse-Problem Instability

In the Fourier domain:

```text
B = H X + N
```

Naive inversion divides by `H`. Frequencies where `|H|` is small amplify noise and calibration error, so the reconstruction can look worse than the measurement.

Presentation line: the hard part is not applying an inverse; it is deciding which information is trustworthy enough to invert.

## 6. Wiener Filtering

Wiener deconvolution stabilizes direct inversion:

```text
X_hat = conj(H) B / (|H|^2 + lambda)
```

The parameter `lambda` trades sharpness for stability.

## 7. PGD and FISTA

Projected gradient descent solves:

```text
minimize 0.5 ||A x - b||_2^2
subject to x >= 0
```

FISTA accelerates this update with momentum. This is the intuitive bridge between the forward model and modern optimization.

## 8. TV Regularization and ADMM

Total variation assumes images are often piecewise smooth with sparse gradients:

```text
minimize 0.5 ||A x - b||_2^2 + tau ||Psi x||_1
```

ADMM splits the problem into simpler subproblems: data consistency, finite-difference sparsity, and nonnegativity. This mirrors the structure of the Waller ADMM tutorial.

## 9. Robustness Experiments

Show diagnostics that make the project feel honest:

- noise sweep,
- PSF mismatch,
- PSF shift,
- crop error,
- saturation,
- quantization,
- TV-strength tuning,
- stopping diagnostics.

Presentation line: robustness experiments are where the repo stops being just a demo and starts behaving like a study.

## 10. Limitations and Next Steps

Be clear about current limits:

- 3D reconstruction is not robust enough for the main result yet.
- Current examples use tutorial/sample data, not self-built hardware.
- Future work is to improve the 3D prior and build a Raspberry Pi DiffuserCam with measured calibration data.

## Source Anchors

- DiffuserCam tutorial: https://waller-lab.github.io/DiffuserCam/tutorial.html
- Waller FISTA tutorial: https://waller-lab.github.io/DiffuserCam/tutorial/GD.html
- Waller ADMM tutorial: https://waller-lab.github.io/DiffuserCam/tutorial/ADMM.html
- Recent advances in lensless imaging: https://doi.org/10.1364/OPTICA.431361
