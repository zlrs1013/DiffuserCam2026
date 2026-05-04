# Notebook Lessons

These notebooks are the primary learning interface for the DiffuserCam study project. The reusable implementation stays in `src/diffusercam_sim`, while each notebook explains the theory and runs one simulation step.

For the cleanup/presentation branch, notebooks 01-07 are the main path. Notebooks 08-09 are future-work references only because the 3D implementation is still experimental.

## Notebooks

1. `01_forward_model_wiener.ipynb`
   - Circular convolution toy model.
   - Direct inverse filtering.
   - Wiener deconvolution.
   - Professor-reference averaging PSF demo.

2. `02_padded_linear_pgd_fista.ipynb`
   - Padded linear convolution.
   - Sensor cropping.
   - Adjoint operator check.
   - Projected gradient descent.
   - FISTA.

3. `03_support_constraints_l2_regularization.ipynb`
   - Support constraints on the padded reconstruction grid.
   - L2-regularized objective and gradient derivation.
   - FISTA comparisons with nonnegativity, support, L2, and support + L2.
   - L2 strength sweep.

4. `04_waller_lab_sample_data.ipynb`
   - Load the Waller-Lab sample PSF and raw hand measurement.
   - Match tutorial preprocessing: background subtraction, box downsampling, and L2 normalization.
   - Run our padded FISTA implementation on measured data.
   - Evaluate objective/residual convergence without ground truth.

5. `05_total_variation_admm.ipynb`
   - Explain total variation as an image-gradient sparsity prior.
   - Derive the ADMM variable split used by the Waller tutorial.
   - Implement and test TV-ADMM on the Waller sample PSF/raw hand measurement.
   - Compare FISTA, ADMM without TV, and ADMM with different TV strengths.

6. `06_admm_diagnostics_parameter_sweeps.ipynb`
   - Portfolio-style ADMM diagnostic workflow.
   - Iteration snapshots showing when structure appears.
   - TV-strength sweep with residual and TV plots.
   - Predicted measurement and residual visualization.
   - Runtime-vs-quality comparison against FISTA.

7. `07_robustness_parameter_tuning_diagnostics.ipynb`
   - Real-data diagnostics: residual histogram, residual structure, Fourier-domain error, convergence rate, and dynamic range checks.
   - Robustness sweeps for PSF mismatch, PSF shifts, crop error, saturation, quantization, noise, and TV strength.
   - Practical parameter tuning guide for FISTA iteration count, ADMM penalties, TV weight, and stopping criteria.
   - Controlled ground-truth benchmark with PSNR, SSIM, reconstruction error, support error, and edge preservation.

## Future-Work Notebooks

8. `08_3d_forward_model_depth_reconstruction.ipynb`
   - Derive the 3D incoherent forward model `b = sum_z A_z x_z + n`.
   - Simulate depth-dependent PSFs and a layered 3D scene.
   - Reconstruct a depth volume with projected FISTA and L2 regularization.
   - Evaluate volume NMSE, volume PSNR, lateral projection SSIM, depth winner accuracy, and depth leakage.
   - Test 3D robustness under noise, shifted PSFs, and wrong depth calibration order.

9. `09_waller_lab_3d_real_data.ipynb`
   - Load the original Waller-Lab DiffuserCam `example_psfs.mat` and `example_raw.png`.
   - Apply settings-file bias corrections and CPU-friendly lateral/axial downsampling.
   - Build real depth-dependent convolution operators and verify the 3D adjoint.
   - Run a baseline projected-FISTA 3D reconstruction.
   - Diagnose depth slices, depth energy, depth winner maps, sensor prediction, and residual structure.

## Launching

This workspace keeps runnable milestones inside the notebook sequence. Use an environment with Jupyter installed, then start from the repo root:

```powershell
jupyter lab
```

If you install notebook tools later, useful packages are:

```text
jupyterlab
ipykernel
matplotlib
```

Notebook figures and saved diagnostics use Matplotlib for labeled, presentation-ready plots.

## Presentation Notebook Pattern

Each 2D notebook should read like a compact Waller-style tutorial:

1. Conceptual intro: what physical or computational issue this notebook isolates.
2. Mathematical model: the objective, operator, constraint, or update rule.
3. Implementation: the minimum code needed to connect theory to the repo utilities.
4. Results: presentation-ready figures with clear titles, labels, and captions.
5. Observations: what changed, what failed, and why.
6. Next steps: how the next notebook extends the idea.
