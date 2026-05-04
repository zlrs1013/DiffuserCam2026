"""Build Notebook 07 for robustness and parameter tuning experiments."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("notebooks/07_robustness_parameter_tuning_diagnostics.ipynb")


def md(text: str) -> dict:
    text = dedent(text).strip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


def code(text: str) -> dict:
    text = dedent(text).strip("\n")
    lines = [line + "\n" for line in text.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


cells = [
    md(
        r"""
        # 07 - Robustness Experiments, Parameter Tuning, and Ground-Truth Benchmarks

        Notebook 06 showed that ADMM can recover the Waller-Lab hand sample quickly. Notebook 07 asks the more practical question:

        **How fragile is the reconstruction when the model, data, or parameters are imperfect?**

        This matters before hardware. A real DiffuserCam system will have PSF misalignment, background error, sensor saturation, quantization, noise, cropping mistakes, and parameter choices that are never exactly optimal.

        This notebook has four parts:

        1. Real-data diagnostics for the Waller sample.
        2. Robustness experiments: PSF mismatch, PSF shifts, crop error, saturation, quantization, noise, and TV strength.
        3. A practical parameter tuning guide for FISTA and ADMM.
        4. A controlled simulation with known ground truth, so we can compute PSNR, SSIM, reconstruction error, support error, and edge preservation.
        """
    ),
    md(
        r"""
        ## Key Idea

        The reconstruction algorithm solves an inverse problem using an assumed forward model:

        $$ b \approx A x $$

        where `b` is the measured sensor image, `x` is the object estimate, and `A` is built from the measured PSF.

        Robustness experiments intentionally break one part of this chain at a time:

        - PSF shift tests whether calibration alignment matters.
        - Saturation tests whether the sensor stayed in its linear range.
        - Quantization tests whether bit depth matters.
        - Noise tests whether the solver amplifies measurement uncertainty.
        - Regularization sweeps test whether our prior assumptions are too weak or too strong.

        A good reconstruction pipeline should not only produce an image. It should also tell us **why** that image is trustworthy or questionable.
        """
    ),
    md("## Setup"),
    code(
        r"""
        from pathlib import Path
        import sys
        import json
        from time import perf_counter

        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt


        def find_project_root(start: Path) -> Path:
            for candidate in [start, *start.parents]:
                if (candidate / 'src' / 'diffusercam_sim').exists():
                    return candidate
            raise RuntimeError('Could not find project root containing src/diffusercam_sim')


        PROJECT_ROOT = find_project_root(Path.cwd().resolve())
        SRC_DIR = PROJECT_ROOT / 'src'
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        from diffusercam_sim import (
            PaddedLinearConvolution,
            admm_total_variation,
            find_waller_tutorial_paths,
            fista,
            make_diffuser_like_psf,
            make_synthetic_scene,
            preprocess_waller_tutorial_sample,
            psnr,
            residual_relative_l2,
            ssim,
        )
        from diffusercam_sim.admm import total_variation_anisotropic

        RESULT_DIR = PROJECT_ROOT / 'results' / 'notebooks' / '07_robustness_parameter_tuning_diagnostics'
        RESULT_DIR.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': '#263238',
                'axes.labelcolor': '#263238',
                'axes.titleweight': 'bold',
                'axes.titlesize': 11,
                'axes.labelsize': 10,
                'xtick.color': '#263238',
                'ytick.color': '#263238',
                'grid.color': '#d9e0e3',
                'grid.linewidth': 0.8,
                'legend.frameon': True,
                'legend.framealpha': 0.95,
                'legend.facecolor': 'white',
                'font.size': 10,
                'figure.max_open_warning': 80,
            }
        )

        PROJECT_ROOT
        """
    ),
    code(
        r"""
        def save_figure(fig, path: Path):
            fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
            return fig


        def normalize_l2(image: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(np.ravel(image)))
            return np.asarray(image, dtype=np.float64) / max(norm, 1e-12)


        def normalize_sum(image: np.ndarray) -> np.ndarray:
            image = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
            return image / max(float(np.sum(image)), 1e-12)


        def display_scale(image: np.ndarray, percentiles=(1, 99.7)) -> np.ndarray:
            image = np.asarray(image, dtype=np.float64)
            low, high = np.percentile(image, percentiles)
            if high <= low:
                return np.zeros_like(image)
            return np.clip((image - low) / (high - low), 0.0, 1.0)


        def plot_image_grid(items, path: Path, columns: int = 3, title: str | None = None):
            rows = int(np.ceil(len(items) / columns))
            fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.4 * rows), constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            for ax, (label, image) in zip(axes, items):
                ax.imshow(display_scale(image), cmap='gray', vmin=0, vmax=1)
                ax.set_title(label)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axes[len(items):]:
                ax.axis('off')
            if title is not None:
                fig.suptitle(title, fontsize=15, fontweight='bold')
            return save_figure(fig, path)


        def add_panel_note(ax, text: str, y: float = 0.05):
            ax.text(
                0.03,
                y,
                text,
                transform=ax.transAxes,
                fontsize=8.7,
                va='bottom',
                bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': '#cfd8dc', 'alpha': 0.92},
            )


        def plot_table(rows: list[dict], columns: list[tuple[str, str]], path: Path, title: str):
            table_data = [[row[key] for key, _ in columns] for row in rows]
            column_labels = [label for _, label in columns]
            fig, ax = plt.subplots(figsize=(1.65 * len(columns), 0.46 * (len(rows) + 3)), constrained_layout=True)
            ax.axis('off')
            fig.suptitle(title, fontsize=14, fontweight='bold')
            table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            table.scale(1.0, 1.3)
            for (row_index, col_index), cell in table.get_celld().items():
                cell.set_edgecolor('#cfd8dc')
                if row_index == 0:
                    cell.set_facecolor('#e8eef2')
                    cell.set_text_props(weight='bold')
                elif row_index % 2 == 0:
                    cell.set_facecolor('#f7f9fa')
            return save_figure(fig, path)


        def gradient_magnitude(image: np.ndarray) -> np.ndarray:
            row_diff = np.diff(image, axis=0, append=image[-1:, :])
            col_diff = np.diff(image, axis=1, append=image[:, -1:])
            return np.sqrt(row_diff * row_diff + col_diff * col_diff)


        def edge_preservation_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
            ref_edges = gradient_magnitude(reference).ravel()
            est_edges = gradient_magnitude(estimate).ravel()
            ref_edges = ref_edges - np.mean(ref_edges)
            est_edges = est_edges - np.mean(est_edges)
            denom = float(np.linalg.norm(ref_edges) * np.linalg.norm(est_edges))
            return float(np.dot(ref_edges, est_edges) / max(denom, 1e-12))


        def support_error(reference: np.ndarray, estimate: np.ndarray, threshold: float = 0.08) -> float:
            ref_support = reference > threshold * max(float(reference.max()), 1e-12)
            est_support = estimate > threshold * max(float(estimate.max()), 1e-12)
            union = np.logical_or(ref_support, est_support)
            if not np.any(union):
                return 0.0
            intersection = np.logical_and(ref_support, est_support)
            return float(1.0 - np.sum(intersection) / np.sum(union))


        def normalized_mse(reference: np.ndarray, estimate: np.ndarray) -> float:
            diff = np.asarray(reference) - np.asarray(estimate)
            return float(np.sum(diff * diff) / max(np.sum(reference * reference), 1e-12))
        """
    ),
    md(
        r"""
        ## Load the Waller-Lab Measured Sample

        We reuse the Waller tutorial sample data from earlier notebooks:

        - `psf_sample.tif`: measured point-source PSF.
        - `rawdata_hand_sample.tif`: measured sensor output for the hand object.

        The preprocessing is the same as before: background subtraction, power-of-two downsampling, and L2 normalization.
        """
    ),
    code(
        r"""
        psf_path, rawdata_path = find_waller_tutorial_paths(PROJECT_ROOT)
        sample = preprocess_waller_tutorial_sample(psf_path, rawdata_path, downsample_factor=1 / 8)
        operator = PaddedLinearConvolution(sample.psf)

        sample_info = {
            'processed_shape': sample.psf.shape,
            'padded_shape': operator.padded_shape,
            'raw_shape': sample.psf_raw.shape,
            'background_estimate': sample.background,
            'adjoint_inner_product_error': operator.adjoint_inner_product_error(),
        }

        sample_info
        """
    ),
    code(
        r"""
        plot_image_grid(
            [
                ('Measured PSF, processed', sample.psf),
                ('Measured sensor image, processed', sample.measurement),
                ('Raw sensor image, original scale', sample.measurement_raw),
            ],
            RESULT_DIR / '01_waller_inputs.png',
            columns=3,
            title='Waller-Lab sample inputs',
        )
        """
    ),
    md(
        r"""
        ## Part 1 - Real-Data Diagnostics

        For real measured data, we usually do **not** know the ground-truth object. That means PSNR and SSIM are not available. Instead, we use diagnostics that ask whether the estimate is physically and numerically plausible:

        - Does the predicted sensor image `A x_hat` resemble the measured sensor image `b`?
        - Is the residual `A x_hat - b` random-looking or structured?
        - Are there high-frequency residuals that indicate PSF mismatch?
        - Is the raw measurement saturated or compressed?
        - Does the residual stop improving with iterations?
        """
    ),
    code(
        r"""
        admm_parameters = {
            'mu1': 1e-6,
            'mu2': 1e-5,
            'mu3': 4e-5,
        }
        baseline_tau = 1e-9

        baseline_admm = admm_total_variation(
            operator=operator,
            measurement=sample.measurement,
            iterations=40,
            record_every=1,
            tau=baseline_tau,
            **admm_parameters,
        )

        baseline_prediction = operator.forward_padded(baseline_admm.padded_estimate)
        baseline_residual = baseline_prediction - sample.measurement

        baseline_metrics = {
            'relative_residual_l2': residual_relative_l2(sample.measurement, baseline_prediction),
            'tv_value': total_variation_anisotropic(baseline_admm.padded_estimate),
            'max_reconstruction_value': float(baseline_admm.reconstruction.max()),
        }

        baseline_metrics
        """
    ),
    code(
        r"""
        plot_image_grid(
            [
                ('Measured sensor b', sample.measurement),
                ('Predicted sensor A x_hat', baseline_prediction),
                ('Residual magnitude |A x_hat - b|', np.abs(baseline_residual)),
                ('ADMM-TV reconstruction', baseline_admm.reconstruction),
            ],
            RESULT_DIR / '02_real_data_prediction_residual.png',
            columns=2,
            title='Real-data forward-model consistency check',
        )
        """
    ),
    code(
        r"""
        def plot_real_data_diagnostics(measurement, prediction, residual, raw_measurement, history, path: Path):
            fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
            fig.suptitle('Real-data diagnostics beyond visual reconstruction', fontsize=15, fontweight='bold')

            axes[0, 0].hist(residual.ravel(), bins=80, color='#5d6fb3', alpha=0.9)
            axes[0, 0].set_title('1. Residual histogram')
            axes[0, 0].set_xlabel('Residual value: A x_hat - b')
            axes[0, 0].set_ylabel('Pixel count')
            axes[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
            add_panel_note(axes[0, 0], 'centered and narrow is good; skew or heavy tails suggest mismatch')

            residual_fft = np.fft.fftshift(np.fft.fft2(residual))
            measurement_fft = np.fft.fftshift(np.fft.fft2(measurement))
            fourier_ratio = np.log10(np.abs(residual_fft) + 1e-12) - np.log10(np.abs(measurement_fft) + 1e-12)
            im = axes[0, 1].imshow(fourier_ratio, cmap='magma')
            axes[0, 1].set_title('2. Fourier-domain residual ratio')
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])
            fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04, label='log residual FFT - log measurement FFT')

            raw = np.asarray(raw_measurement, dtype=np.float64)
            axes[1, 0].hist(raw.ravel(), bins=120, color='#1b7f79', alpha=0.9)
            axes[1, 0].axvline(np.percentile(raw, 99.9), color='#b24c63', linewidth=2, label='99.9 percentile')
            axes[1, 0].set_title('3. Raw measurement dynamic range')
            axes[1, 0].set_xlabel('Raw sensor value')
            axes[1, 0].set_ylabel('Pixel count')
            axes[1, 0].legend(loc='best')
            add_panel_note(axes[1, 0], 'pile-up at sensor max would indicate saturation')

            iterations = np.array([record.iteration for record in history])
            residuals = np.array([record.relative_residual_l2 for record in history])
            improvement = np.r_[np.nan, -np.diff(residuals)]
            axes[1, 1].plot(iterations, residuals, marker='o', linewidth=2.2, label='relative residual')
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_title('4. Convergence rate')
            axes[1, 1].set_xlabel('ADMM iteration')
            axes[1, 1].set_ylabel('Relative residual, log scale')
            twin = axes[1, 1].twinx()
            twin.plot(iterations[1:], improvement[1:], color='#d17a22', marker='s', linewidth=1.6, label='residual drop')
            twin.set_ylabel('Residual decrease per iteration')
            add_panel_note(axes[1, 1], 'flattening suggests diminishing returns')

            for ax in axes.ravel():
                ax.grid(True, which='major')
                ax.spines[['top', 'right']].set_visible(False)

            return save_figure(fig, path)


        plot_real_data_diagnostics(
            sample.measurement,
            baseline_prediction,
            baseline_residual,
            sample.measurement_raw,
            baseline_admm.history,
            RESULT_DIR / '03_real_data_diagnostics.png',
        )
        """
    ),
    md(
        r"""
        ### Reading These Diagnostics

        The residual image and residual histogram should be interpreted together. A small residual norm is useful, but a **structured residual** often means the model is missing something systematic: PSF alignment error, crop error, background error, saturation, nonlinear sensor response, or a failure of the shift-invariant convolution approximation.

        The Fourier residual plot is especially useful for lensless systems. If the residual concentrates in particular frequency bands, the issue may be deconvolution/model mismatch rather than random sensor noise.
        """
    ),
    md(
        r"""
        ## Part 2 - Robustness Experiments

        We now perturb the reconstruction inputs one factor at a time. For measured data, we still do not have ground truth, so we report:

        - relative residual: `||A x_hat - b||_2 / ||b||_2`
        - TV value: `||D x_hat||_1`
        - max reconstructed value: useful for spotting intensity-scale instability
        - runtime

        Important caveat: a lower residual is not automatically a better image. A wrong model can sometimes fit corrupted data well. That is why we also inspect reconstructions visually and later run a ground-truth simulation.
        """
    ),
    code(
        r"""
        def shift_psf(psf: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
            return normalize_l2(np.roll(np.roll(psf, row_shift, axis=0), col_shift, axis=1))


        def crop_error_psf(psf: np.ndarray, pixels: int = 3) -> np.ndarray:
            cropped = psf[pixels:-pixels, pixels:-pixels]
            damaged = np.pad(cropped, ((pixels, pixels), (pixels, pixels)), mode='constant')
            return normalize_l2(damaged)


        def mixed_mismatch_psf(psf: np.ndarray, strength: float = 0.18) -> np.ndarray:
            synthetic_square = make_diffuser_like_psf(max(psf.shape), seed=23, spot_count=80)
            synthetic = synthetic_square[: psf.shape[0], : psf.shape[1]]
            return normalize_l2((1.0 - strength) * psf + strength * normalize_l2(synthetic))


        def saturate_measurement(measurement: np.ndarray, percentile: float = 99.4) -> np.ndarray:
            level = float(np.percentile(measurement, percentile))
            return normalize_l2(np.clip(measurement, None, level))


        def quantize_measurement(measurement: np.ndarray, bits: int = 6) -> np.ndarray:
            shifted = measurement - float(np.min(measurement))
            scaled = shifted / max(float(np.max(shifted)), 1e-12)
            levels = 2**bits - 1
            quantized = np.round(levels * scaled) / levels
            return normalize_l2(quantized)


        def add_noise_by_snr(measurement: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
            rng = np.random.default_rng(seed)
            rms = float(np.sqrt(np.mean(measurement * measurement)))
            sigma = rms / (10.0 ** (snr_db / 20.0))
            noisy = measurement + rng.normal(0.0, sigma, measurement.shape)
            return normalize_l2(np.maximum(noisy, 0.0))


        def run_fista_case(name: str, psf: np.ndarray, measurement: np.ndarray, iterations: int = 70, l2: float = 0.0) -> dict:
            case_operator = PaddedLinearConvolution(psf)
            start = perf_counter()
            result = fista(
                operator=case_operator,
                measurement=measurement,
                iterations=iterations,
                record_every=max(1, iterations // 10),
                l2_regularization=l2,
            )
            seconds = perf_counter() - start
            prediction = case_operator.forward_padded(result.padded_estimate)
            return {
                'name': name,
                'method': 'FISTA',
                'iterations': iterations,
                'seconds': seconds,
                'operator': case_operator,
                'measurement': measurement,
                'result': result,
                'prediction': prediction,
                'residual': prediction - measurement,
                'relative_residual_l2': residual_relative_l2(measurement, prediction),
                'tv_value': total_variation_anisotropic(result.padded_estimate),
                'max_value': float(result.reconstruction.max()),
            }


        robustness_cases = [
            ('baseline', sample.psf, sample.measurement),
            ('PSF shift +2 rows', shift_psf(sample.psf, 2, 0), sample.measurement),
            ('PSF shift +2 cols', shift_psf(sample.psf, 0, 2), sample.measurement),
            ('PSF crop error', crop_error_psf(sample.psf, pixels=3), sample.measurement),
            ('PSF mixed mismatch', mixed_mismatch_psf(sample.psf, strength=0.12), sample.measurement),
            ('measurement saturated', sample.psf, saturate_measurement(sample.measurement, percentile=99.2)),
            ('measurement 6-bit', sample.psf, quantize_measurement(sample.measurement, bits=6)),
            ('noise 30 dB', sample.psf, add_noise_by_snr(sample.measurement, 30, seed=1)),
            ('noise 20 dB', sample.psf, add_noise_by_snr(sample.measurement, 20, seed=2)),
            ('noise 10 dB', sample.psf, add_noise_by_snr(sample.measurement, 10, seed=3)),
        ]

        robustness_results = [run_fista_case(name, psf, measurement) for name, psf, measurement in robustness_cases]

        summary_rows = [
            {
                'case': row['name'],
                'residual': f"{row['relative_residual_l2']:.4f}",
                'tv': f"{row['tv_value']:.3f}",
                'max': f"{row['max_value']:.4f}",
                'sec': f"{row['seconds']:.2f}",
            }
            for row in robustness_results
        ]

        summary_rows
        """
    ),
    code(
        r"""
        plot_table(
            summary_rows,
            columns=[('case', 'Case'), ('residual', 'Rel. residual'), ('tv', 'TV value'), ('max', 'Max'), ('sec', 'Seconds')],
            path=RESULT_DIR / '04_robustness_summary_table.png',
            title='Robustness sweep summary, FISTA reconstructions',
        )
        """
    ),
    code(
        r"""
        def plot_robustness_summary(results, path: Path):
            names = [row['name'] for row in results]
            residuals = np.array([row['relative_residual_l2'] for row in results])
            tv_values = np.array([row['tv_value'] for row in results])
            max_values = np.array([row['max_value'] for row in results])
            x = np.arange(len(names))

            fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.5), sharex=True, constrained_layout=True)
            fig.suptitle('How each perturbation changes reconstruction diagnostics', fontsize=15, fontweight='bold')

            axes[0].bar(x, residuals, color='#1b7f79')
            axes[0].set_ylabel('Relative residual')
            axes[0].set_title('1. Sensor fit')
            add_panel_note(axes[0], 'lower means the model fits the provided sensor image')

            axes[1].bar(x, tv_values, color='#b24c63')
            axes[1].set_ylabel('TV value')
            axes[1].set_title('2. Image-gradient activity')
            add_panel_note(axes[1], 'higher often means more texture/artifacts')

            axes[2].bar(x, max_values, color='#5d6fb3')
            axes[2].set_ylabel('Max reconstruction value')
            axes[2].set_title('3. Intensity-scale stability')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(names, rotation=35, ha='right')

            for ax in axes:
                ax.grid(True, axis='y')
                ax.spines[['top', 'right']].set_visible(False)

            return save_figure(fig, path)


        plot_robustness_summary(robustness_results, RESULT_DIR / '05_robustness_metric_bars.png')
        """
    ),
    code(
        r"""
        plot_image_grid(
            [(row['name'], row['result'].reconstruction) for row in robustness_results],
            RESULT_DIR / '06_robustness_reconstruction_grid.png',
            columns=5,
            title='Visual effect of model and measurement perturbations',
        )
        """
    ),
    md(
        r"""
        ### Robustness Observations

        These experiments should be read as a **failure-mode map**:

        - PSF shifts and crop errors test calibration alignment.
        - Saturation and quantization test whether the measurement still behaves like a linear intensity image.
        - Noise tests whether the solver is amplifying measurement uncertainty.

        For future hardware, this section tells us what to protect first: rigid diffuser-sensor geometry, unsaturated RAW captures, stable cropping, and consistent preprocessing.
        """
    ),
    md(
        r"""
        ## Part 3 - Parameter Tuning Guide

        There is no universal best parameter set. Parameters are tied to data normalization, PSF scaling, noise level, crop convention, and desired visual behavior.

        Still, there is a practical workflow:

        1. Use FISTA first as a cheap baseline.
        2. Watch residual vs iteration and stop when improvement becomes small.
        3. Use ADMM-TV when nonnegativity and TV control are important.
        4. Tune `tau / mu2` as the effective TV shrinkage threshold.
        5. Change `mu1`, `mu2`, and `mu3` only after the basic reconstruction is stable.

        The Waller-style ADMM split has:

        - `mu1`: penalty tying the full convolution variable to the object variable.
        - `mu2`: penalty tying the TV split variable to image gradients.
        - `mu3`: penalty tying the nonnegative split variable to the object variable.
        - `tau`: weight of the TV prior.

        The most interpretable TV knob is:

        $$ \tau / \mu_2 $$

        because that ratio is the soft-threshold used in the TV shrinkage step.
        """
    ),
    code(
        r"""
        fista_tuning = fista(
            operator=operator,
            measurement=sample.measurement,
            iterations=250,
            record_every=5,
        )

        admm_tuning = admm_total_variation(
            operator=operator,
            measurement=sample.measurement,
            iterations=60,
            record_every=1,
            tau=baseline_tau,
            **admm_parameters,
        )


        def plot_stopping_diagnostics(fista_result, admm_result, path: Path):
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
            fig.suptitle('Stopping criteria: residual decrease vs iteration', fontsize=15, fontweight='bold')

            for result, label, color in [(fista_result, 'FISTA', '#5d6fb3'), (admm_result, 'ADMM-TV', '#d17a22')]:
                iterations = np.array([record.iteration for record in result.history])
                residuals = np.array([record.relative_residual_l2 for record in result.history])
                improvement = np.r_[np.nan, -np.diff(residuals)]
                axes[0].plot(iterations, residuals, marker='o', linewidth=2.2, label=label, color=color)
                axes[1].plot(iterations[1:], improvement[1:], marker='o', linewidth=2.2, label=label, color=color)

            axes[0].set_yscale('log')
            axes[0].set_title('1. Residual curve')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Relative residual, log scale')
            add_panel_note(axes[0], 'use this to compare total progress')

            axes[1].set_title('2. Marginal improvement')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Residual drop since previous record')
            add_panel_note(axes[1], 'stop when improvement is small and image is stable')

            for ax in axes:
                ax.grid(True, which='major')
                ax.spines[['top', 'right']].set_visible(False)
                ax.legend(loc='best')

            return save_figure(fig, path)


        plot_stopping_diagnostics(fista_tuning, admm_tuning, RESULT_DIR / '07_stopping_diagnostics.png')
        """
    ),
    code(
        r"""
        tau_values = [1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8]
        tau_results = []
        for tau in tau_values:
            result = admm_total_variation(
                operator=operator,
                measurement=sample.measurement,
                iterations=15,
                record_every=15,
                tau=tau,
                **admm_parameters,
            )
            record = result.history[-1]
            tau_results.append(
                {
                    'tau': tau,
                    'tau_over_mu2': tau / admm_parameters['mu2'],
                    'relative_residual_l2': record.relative_residual_l2,
                    'tv_value': record.tv_value,
                    'max_value': float(result.reconstruction.max()),
                    'result': result,
                }
            )


        def plot_tau_tuning(rows, path: Path):
            ratios = np.array([row['tau_over_mu2'] for row in rows])
            residual = np.array([row['relative_residual_l2'] for row in rows])
            tv_values = np.array([row['tv_value'] for row in rows])
            max_values = np.array([row['max_value'] for row in rows])

            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
            fig.suptitle('ADMM TV tuning: sweep tau / mu2', fontsize=15, fontweight='bold')
            for ax, y, title, ylabel, color in [
                (axes[0], residual, '1. Sensor fit', 'Relative residual', '#1b7f79'),
                (axes[1], tv_values, '2. Smoothness', 'TV value', '#b24c63'),
                (axes[2], max_values, '3. Intensity scale', 'Max reconstruction value', '#5d6fb3'),
            ]:
                ax.plot(ratios, y, marker='o', linewidth=2.3, color=color)
                ax.set_xscale('log')
                ax.set_xlabel('tau / mu2, log scale')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, which='both')
                ax.spines[['top', 'right']].set_visible(False)
            add_panel_note(axes[0], 'do not optimize this alone')
            add_panel_note(axes[1], 'lower means smoother')
            add_panel_note(axes[2], 'large jumps suggest instability')
            return save_figure(fig, path)


        plot_tau_tuning(tau_results, RESULT_DIR / '08_tau_tuning.png')
        """
    ),
    code(
        r"""
        mu_sweep_rows = []
        mu_variants = []
        for key in ['mu1', 'mu2', 'mu3']:
            for scale in [0.3, 1.0, 3.0]:
                params = dict(admm_parameters)
                params[key] = admm_parameters[key] * scale
                mu_variants.append((f'{key} x{scale:g}', params))

        for label, params in mu_variants:
            result = admm_total_variation(
                operator=operator,
                measurement=sample.measurement,
                iterations=12,
                record_every=12,
                tau=baseline_tau,
                **params,
            )
            record = result.history[-1]
            mu_sweep_rows.append(
                {
                    'case': label,
                    'mu1': f"{params['mu1']:.1e}",
                    'mu2': f"{params['mu2']:.1e}",
                    'mu3': f"{params['mu3']:.1e}",
                    'residual': f"{record.relative_residual_l2:.4f}",
                    'tv': f"{record.tv_value:.3f}",
                    'max': f"{float(result.reconstruction.max()):.4f}",
                }
            )

        plot_table(
            mu_sweep_rows,
            columns=[('case', 'Case'), ('mu1', 'mu1'), ('mu2', 'mu2'), ('mu3', 'mu3'), ('residual', 'Residual'), ('tv', 'TV'), ('max', 'Max')],
            path=RESULT_DIR / '09_mu_sweep_table.png',
            title='Small ADMM penalty sweep, 12 iterations each',
        )
        """
    ),
    md(
        r"""
        ### Practical Parameter Guide

        **FISTA iteration count**

        Start with 50 to 100 iterations. Plot residual vs iteration. Increase only if the residual is still dropping meaningfully and the image is improving visually. Stop when residual improvement is small or artifacts begin to dominate.

        **ADMM `mu1`, `mu2`, `mu3`**

        Use a known-good baseline first. For this normalized Waller sample, the working baseline is:

        ```text
        mu1 = 1e-6
        mu2 = 1e-5
        mu3 = 4e-5
        ```

        Interpretation:

        - Increase `mu1` if the convolution-consistency split seems too loose.
        - Increase `mu2` if the TV split should follow image gradients more tightly.
        - Increase `mu3` if nonnegativity should be enforced more aggressively.

        But tune these carefully. The ADMM penalties affect numerical behavior, not just visual regularization.

        **TV weight `tau`**

        Tune `tau / mu2`. Too small gives weak denoising and artifacts. Too large can erase fingers, edges, and small features.

        **Stopping criteria**

        A practical stopping rule should combine residual improvement, stable visual reconstruction, stable max intensity, and residual structure diagnostics.
        """
    ),
    md(
        r"""
        ## Part 4 - Controlled Simulation with Known Ground Truth

        Measured data teaches realism, but it does not provide ground truth. Now we create a synthetic object, simulate the sensor measurement with a known PSF, add controlled noise, and reconstruct.

        Because the true object is known, we can compute PSNR, SSIM, normalized reconstruction error, support error, and edge preservation correlation.
        """
    ),
    code(
        r"""
        def make_rectangular_truth(shape: tuple[int, int]) -> np.ndarray:
            rows, cols = shape
            square = make_synthetic_scene(size=max(rows, cols))
            start_row = (square.shape[0] - rows) // 2
            start_col = (square.shape[1] - cols) // 2
            truth = square[start_row : start_row + rows, start_col : start_col + cols]
            truth = truth / max(float(truth.max()), 1e-12)
            return truth.astype(np.float64)


        def add_sensor_noise(measurement: np.ndarray, snr_db: float, seed: int = 10) -> tuple[np.ndarray, float]:
            rng = np.random.default_rng(seed)
            rms = float(np.sqrt(np.mean(measurement * measurement)))
            sigma = rms / (10.0 ** (snr_db / 20.0))
            noisy = measurement + rng.normal(0.0, sigma, measurement.shape)
            return np.maximum(noisy, 0.0), sigma


        truth = make_rectangular_truth(sample.psf.shape)
        physical_psf = normalize_sum(sample.psf)
        truth_operator = PaddedLinearConvolution(physical_psf)
        clean_synthetic_measurement = truth_operator.forward(truth)
        synthetic_measurement, synthetic_noise_sigma = add_sensor_noise(clean_synthetic_measurement, snr_db=30, seed=99)

        synthetic_info = {
            'truth_shape': truth.shape,
            'measurement_snr_db': 30,
            'noise_sigma': synthetic_noise_sigma,
            'psf_sum': float(physical_psf.sum()),
        }

        synthetic_info
        """
    ),
    code(
        r"""
        plot_image_grid(
            [
                ('Ground-truth object', truth),
                ('Physical PSF, sum-normalized', physical_psf),
                ('Clean synthetic measurement', clean_synthetic_measurement),
                ('Noisy synthetic measurement, 30 dB', synthetic_measurement),
            ],
            RESULT_DIR / '10_synthetic_forward_problem.png',
            columns=2,
            title='Controlled simulation with known truth',
        )
        """
    ),
    code(
        r"""
        synthetic_fista = fista(
            operator=truth_operator,
            measurement=synthetic_measurement,
            iterations=160,
            record_every=10,
            truth=truth,
        )

        synthetic_admm = admm_total_variation(
            operator=truth_operator,
            measurement=synthetic_measurement,
            iterations=45,
            record_every=5,
            tau=2e-4,
            mu1=1e-3,
            mu2=3e-3,
            mu3=1e-2,
        )

        synthetic_admm_prediction = truth_operator.forward_padded(synthetic_admm.padded_estimate)
        """
    ),
    code(
        r"""
        def evaluate_against_truth(name: str, reconstruction: np.ndarray, prediction: np.ndarray) -> dict:
            reconstruction_clipped = np.clip(reconstruction, 0.0, 1.0)
            return {
                'method': name,
                'psnr_db': psnr(truth, reconstruction_clipped, data_range=1.0),
                'ssim': ssim(truth, reconstruction_clipped, data_range=1.0),
                'normalized_mse': normalized_mse(truth, reconstruction_clipped),
                'support_error': support_error(truth, reconstruction_clipped),
                'edge_corr': edge_preservation_correlation(truth, reconstruction_clipped),
                'relative_residual_l2': residual_relative_l2(synthetic_measurement, prediction),
                'tv_value': total_variation_anisotropic(truth_operator.pad(reconstruction_clipped)),
            }

        synthetic_metrics = [
            evaluate_against_truth('FISTA 160 iter', synthetic_fista.reconstruction, truth_operator.forward_padded(synthetic_fista.padded_estimate)),
            evaluate_against_truth('ADMM-TV 45 iter', synthetic_admm.reconstruction, synthetic_admm_prediction),
        ]

        synthetic_table_rows = [
            {
                'method': row['method'],
                'psnr': f"{row['psnr_db']:.2f}",
                'ssim': f"{row['ssim']:.3f}",
                'nmse': f"{row['normalized_mse']:.4f}",
                'support': f"{row['support_error']:.3f}",
                'edge': f"{row['edge_corr']:.3f}",
                'residual': f"{row['relative_residual_l2']:.4f}",
            }
            for row in synthetic_metrics
        ]

        plot_table(
            synthetic_table_rows,
            columns=[('method', 'Method'), ('psnr', 'PSNR dB'), ('ssim', 'SSIM'), ('nmse', 'Norm. MSE'), ('support', 'Support error'), ('edge', 'Edge corr.'), ('residual', 'Residual')],
            path=RESULT_DIR / '11_synthetic_metric_table.png',
            title='Ground-truth benchmark metrics',
        )
        """
    ),
    code(
        r"""
        plot_image_grid(
            [
                ('Ground truth', truth),
                ('FISTA reconstruction', synthetic_fista.reconstruction),
                ('ADMM-TV reconstruction', synthetic_admm.reconstruction),
                ('FISTA absolute error', np.abs(synthetic_fista.reconstruction - truth)),
                ('ADMM absolute error', np.abs(synthetic_admm.reconstruction - truth)),
                ('Truth edge magnitude', gradient_magnitude(truth)),
                ('FISTA edge magnitude', gradient_magnitude(synthetic_fista.reconstruction)),
                ('ADMM edge magnitude', gradient_magnitude(synthetic_admm.reconstruction)),
            ],
            RESULT_DIR / '12_synthetic_reconstruction_errors.png',
            columns=4,
            title='Ground-truth reconstruction and error diagnostics',
        )
        """
    ),
    code(
        r"""
        def plot_ground_truth_convergence(fista_result, path: Path):
            iterations = np.array([record.iteration for record in fista_result.history])
            residuals = np.array([record.relative_residual_l2 for record in fista_result.history])
            psnr_values = np.array([record.psnr_db for record in fista_result.history])
            ssim_values = np.array([record.ssim for record in fista_result.history])

            fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
            fig.suptitle('Ground-truth benchmark convergence, FISTA', fontsize=15, fontweight='bold')

            axes[0].plot(iterations, residuals, marker='o', linewidth=2.2, color='#1b7f79')
            axes[0].set_yscale('log')
            axes[0].set_title('1. Sensor residual')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Relative residual, log scale')

            axes[1].plot(iterations, psnr_values, marker='o', linewidth=2.2, color='#5d6fb3')
            axes[1].set_title('2. PSNR')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('PSNR, dB')

            axes[2].plot(iterations, ssim_values, marker='o', linewidth=2.2, color='#b24c63')
            axes[2].set_title('3. SSIM')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('SSIM')
            add_panel_note(axes[2], 'truth metrics can peak before residual is minimal')

            for ax in axes:
                ax.grid(True, which='major')
                ax.spines[['top', 'right']].set_visible(False)

            return save_figure(fig, path)


        plot_ground_truth_convergence(synthetic_fista, RESULT_DIR / '13_ground_truth_convergence.png')
        """
    ),
    md(
        r"""
        ## Final Takeaways

        Robustness work changes how we think about reconstruction quality:

        - A pretty image is not enough.
        - A low residual is not enough.
        - A solver comparison without ground truth can be misleading.
        - Calibration, cropping, saturation, and preprocessing can dominate algorithm choice.

        For the real Waller sample, diagnostics help us identify whether the model is self-consistent. For the synthetic benchmark, PSNR/SSIM/support/edge metrics tell us whether that consistency actually corresponds to ground-truth recovery.

        The next major milestone after this is a deeper calibration notebook: simulate point-source PSF measurement, introduce calibration errors deliberately, and connect those errors to the robustness failures observed here.
        """
    ),
    code(
        r"""
        summary = {
            'sample_info': sample_info,
            'baseline_metrics': baseline_metrics,
            'robustness_results': [
                {
                    'name': row['name'],
                    'relative_residual_l2': row['relative_residual_l2'],
                    'tv_value': row['tv_value'],
                    'max_value': row['max_value'],
                    'seconds': row['seconds'],
                }
                for row in robustness_results
            ],
            'tau_results': [
                {
                    'tau': row['tau'],
                    'tau_over_mu2': row['tau_over_mu2'],
                    'relative_residual_l2': row['relative_residual_l2'],
                    'tv_value': row['tv_value'],
                    'max_value': row['max_value'],
                }
                for row in tau_results
            ],
            'mu_sweep_rows': mu_sweep_rows,
            'synthetic_info': synthetic_info,
            'synthetic_metrics': synthetic_metrics,
        }

        (RESULT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(NOTEBOOK_PATH)
