"""Step 2: padded linear convolution with projected GD and FISTA.

This script follows the operator convention used in the Waller Lab tutorial:

1. Store real-space images with their origin at the center.
2. Pad the scene and PSF to a power-of-two grid large enough for linear
   convolution.
3. Use FFTs on the padded grid and crop the sensor-sized measurement.
4. Reconstruct by minimizing ``0.5 ||A x - b||_2^2`` with ``x >= 0``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diffusercam_sim import (  # noqa: E402
    PaddedLinearConvolution,
    add_gaussian_noise,
    fista,
    make_diffuser_like_psf,
    make_synthetic_scene,
    projected_gradient_descent,
    psnr,
    residual_relative_l2,
    ssim,
)
from diffusercam_sim.viz import save_convergence_plot, save_montage  # noqa: E402


RESULT_DIR = PROJECT_ROOT / "results" / "iterative_2d"


def write_history_csv(path: Path, method: str, records: list[object]) -> None:
    """Write solver diagnostics to a CSV file."""

    fieldnames = ["method", "iteration", "objective", "relative_residual_l2", "psnr_db", "ssim"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["method"] = method
            writer.writerow(row)


def final_record(records: list[object]) -> dict[str, float | int | None]:
    """Return the final diagnostic record as a plain dictionary."""

    return asdict(records[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256, help="Scene and PSF width/height.")
    parser.add_argument("--snr-db", type=float, default=35.0, help="Gaussian measurement SNR in dB.")
    parser.add_argument("--iterations", type=int, default=80, help="Number of PGD/FISTA iterations.")
    parser.add_argument("--record-every", type=int, default=5, help="Metric logging interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    scene = make_synthetic_scene(args.size)
    psf = make_diffuser_like_psf(args.size)
    operator = PaddedLinearConvolution(psf)

    clean_measurement = operator.forward(scene)
    measurement, noise_sigma = add_gaussian_noise(clean_measurement, args.snr_db)

    adjoint_error = operator.adjoint_inner_product_error()
    if adjoint_error > 1e-10:
        raise RuntimeError(f"Adjoint check failed with relative error {adjoint_error:.3e}")

    pgd_result = projected_gradient_descent(
        operator=operator,
        measurement=measurement,
        iterations=args.iterations,
        truth=scene,
        record_every=args.record_every,
    )
    fista_result = fista(
        operator=operator,
        measurement=measurement,
        iterations=args.iterations,
        truth=scene,
        record_every=args.record_every,
    )

    pgd_predicted = operator.forward_padded(pgd_result.padded_estimate)
    fista_predicted = operator.forward_padded(fista_result.padded_estimate)
    pgd_crop_predicted = operator.forward(pgd_result.reconstruction)
    fista_crop_predicted = operator.forward(fista_result.reconstruction)
    pgd_residual = pgd_predicted - measurement
    fista_residual = fista_predicted - measurement

    write_history_csv(RESULT_DIR / "pgd_history.csv", pgd_result.method, pgd_result.history)
    write_history_csv(RESULT_DIR / "fista_history.csv", fista_result.method, fista_result.history)

    summary = {
        "model": "padded linear convolution with centered sensor crop",
        "image_shape": scene.shape,
        "padded_shape": operator.padded_shape,
        "snr_db_requested": args.snr_db,
        "noise_sigma": noise_sigma,
        "iterations": args.iterations,
        "record_every": args.record_every,
        "adjoint_inner_product_relative_error": adjoint_error,
        "step_size": pgd_result.step_size,
        "pgd_final": final_record(pgd_result.history),
        "fista_final": final_record(fista_result.history),
        "pgd_padded_estimate_forward_residual_l2": residual_relative_l2(measurement, pgd_predicted),
        "fista_padded_estimate_forward_residual_l2": residual_relative_l2(measurement, fista_predicted),
        "pgd_cropped_reconstruction_forward_residual_l2": residual_relative_l2(measurement, pgd_crop_predicted),
        "fista_cropped_reconstruction_forward_residual_l2": residual_relative_l2(measurement, fista_crop_predicted),
        "pgd_final_psnr_db_recomputed": psnr(scene, np.clip(pgd_result.reconstruction, 0.0, 1.0)),
        "fista_final_psnr_db_recomputed": psnr(scene, np.clip(fista_result.reconstruction, 0.0, 1.0)),
        "pgd_final_ssim_recomputed": ssim(scene, np.clip(pgd_result.reconstruction, 0.0, 1.0)),
        "fista_final_ssim_recomputed": ssim(scene, np.clip(fista_result.reconstruction, 0.0, 1.0)),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    np.savez_compressed(
        RESULT_DIR / "iterative_2d_arrays.npz",
        scene=scene,
        psf=psf,
        clean_measurement=clean_measurement,
        measurement=measurement,
        pgd_reconstruction=pgd_result.reconstruction,
        fista_reconstruction=fista_result.reconstruction,
        pgd_residual=pgd_residual,
        fista_residual=fista_residual,
    )

    save_montage(
        [
            ("Ground truth scene", scene),
            ("PSF, log display", psf),
            ("Noisy sensor measurement", measurement),
            ("PGD reconstruction", pgd_result.reconstruction),
            ("FISTA reconstruction", fista_result.reconstruction),
            ("FISTA residual magnitude", np.abs(fista_residual)),
        ],
        RESULT_DIR / "montage.png",
        columns=3,
    )
    save_convergence_plot(
        [
            ("PGD", pgd_result.history),
            ("FISTA", fista_result.history),
        ],
        RESULT_DIR / "convergence.png",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {RESULT_DIR}")


if __name__ == "__main__":
    main()
