# DiffuserCam 2026 Study Workspace

This repo is a learning and simulation workspace for DiffuserCam-style lensless imaging. The clean-up-2D branch focuses on the 2D DiffuserCam: forward modeling, inverse-problem instability, Wiener filtering, projected-gradient/FISTA reconstruction, Waller-Lab sample data test, TV-ADMM, and robustness diagnostics.

## Start Here

Notebooks 01-07:

```text
notebooks/01_forward_model_wiener.ipynb
notebooks/02_padded_linear_pgd_fista.ipynb
notebooks/03_support_constraints_l2_regularization.ipynb
notebooks/04_waller_lab_sample_data.ipynb
notebooks/05_total_variation_admm.ipynb
notebooks/06_admm_diagnostics_parameter_sweeps.ipynb
notebooks/07_robustness_parameter_tuning_diagnostics.ipynb
```

Each notebook explains the math, runs the simulation, saves figures, and computes quantitative metrics. The reusable implementation lives in:

```text
src/diffusercam_sim/
```

The old command-line milestones have been folded into the notebook sequence: Notebook 01 owns the circular-convolution/Wiener demos, and Notebook 02 owns the padded-linear PGD/FISTA demo.

## What Exists

```text
notebooks/                  Main implementation sequence
src/diffusercam_sim/        Reusable FFT, forward-model, solver, metric, data-loading, and visualization utilities
data/external/tutorial/     Waller-Lab 2D sample PSF and raw hand measurement
results/                    Regenerated figures, metrics, and notebook outputs
```

## Scope

For this branch, present the 2D DiffuserCam workflow:

1. Lensless imaging motivation.
2. DiffuserCam idea: a diffuser turns each point source into a calibrated PSF pattern.
3. PSF calibration and measured raw data.
4. 2D shift-invariant forward model: `b = A x + n`.
5. Why direct inversion is unstable.
6. Wiener filtering as the first stabilized inverse.
7. Projected gradient descent and FISTA for constrained least squares.
8. TV regularization and ADMM for sharper piecewise-smooth reconstructions.
9. Robustness experiments: noise, PSF mismatch, shifts, crop errors, saturation, quantization, and parameter tuning.
10. Limitations and next steps: more robust 3D reconstruction and building a Raspberry Pi DiffuserCam.

## Roadmap

### Phase 1: Incoherent 2D Forward Model

- Interpret a lensless camera as a linear system: `y = Hx + n`.
- Use a measured or synthetic point spread function (PSF) as the impulse response of the system.
- Start with the shift-invariant approximation: `y = psf * x + noise`.
- Learn why Fourier-domain direct inversion can be unstable.
- Implement Wiener inverse filtering.

### Phase 2: Iterative Reconstruction

- Implement projected gradient descent for `0.5 ||Hx - y||_2^2`.
- Add nonnegativity projection because scene intensities cannot be negative.
- Implement Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for faster convergence.
- Compare convergence curves and residuals against Wiener deconvolution.

### Phase 3: Robustness Experiments

- Sweep SNR, PSF mismatch, PSF shifts, cropping errors, saturation, quantization, and regularization strength.
- Save metrics as CSV/JSON and generate comparison montages.

### Phase 4: Regularization

- Add Tikhonov regularization in the Fourier and iterative views.
- Add total variation regularization and compare against nonnegative FISTA.
- Introduce ADMM after the simpler projected-gradient view is solid.

### Future Work: 3D DiffuserCam Model

- Simulate `y = sum_z H_z x_z + n` using depth-dependent PSFs.
- Reconstruct a small depth volume.
- Evaluate lateral recovery and depth localization.
- Keep this out of the main presentation claims until the implementation is robust.

### Future Work: Hardware Preparation

- Plan Raspberry Pi 5 and Pi Camera Module 3 DiffuserCam implementation.

## Runnable Milestones

The repo now has one presentation path instead of parallel notebook and script paths:

- Notebook 01 runs the circular-convolution/Wiener milestone, including the DiffuserCam-style sparse random PSF toy model and the professor-reference 5x5 averaging PSF demo.
- Notebook 02 runs the padded linear convolution milestone with projected gradient descent and FISTA.

Notebook outputs are written under:

```text
results/notebooks/
```

## Notebook Environment

To use the notebooks interactively, create or use an environment with:

```text
jupyterlab
ipykernel
numpy
pillow
matplotlib
```

## Primary References

- DiffuserCam project: https://waller-lab.github.io/DiffuserCam/
- DiffuserCam tutorial: https://waller-lab.github.io/DiffuserCam/tutorial.html
- Gradient descent / FISTA tutorial: https://waller-lab.github.io/DiffuserCam/tutorial/GD.html
- ADMM tutorial: https://waller-lab.github.io/DiffuserCam/tutorial/ADMM.html
- Recent advances in lensless imaging: https://doi.org/10.1364/OPTICA.431361
- LenslessPiCam docs: https://lensless.readthedocs.io/en/latest/index.html
