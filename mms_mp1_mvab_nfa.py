# -*- coding: utf-8 -*-
"""
MMS1-4 MP1 MVAB + NFA
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import os

import cdflib
import numpy as np
import pandas as pd


START_TIME = "2016-01-08 09:22:38.443"
END_TIME = "2016-01-08 09:22:49.373"

R_E_KM = 6371.0

DEFAULT_DATA_DIR = Path("data") / "20160108"

DATA_DIR = Path(
    os.environ.get("MMS_DATA_DIR", str(DEFAULT_DATA_DIR))
).expanduser()

FGM_DIR = DATA_DIR / "fgm" / "brst"
MEC_DIR = DATA_DIR / "mec" / "srvy"


def normalize(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(v)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot normalize a zero or invalid vector.")

    return v / norm


def vector_angle_deg(a, b) -> float:
    a_u = normalize(a)
    b_u = normalize(b)
    cosine = np.clip(np.dot(a_u, b_u), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _cdf_epoch_to_datetime(epoch_values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        cdflib.cdfepoch.to_datetime(epoch_values)
    )


def _collect_cdf_series(
    files,
    var_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    all_time = []
    all_data = []

    for file in files:
        cdf = cdflib.CDF(str(file))
        time = _cdf_epoch_to_datetime(cdf.varget("Epoch"))
        mask = (time >= start) & (time <= end)

        if not np.any(mask):
            continue

        values = np.asarray(cdf.varget(var_name))

        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError(
                f"Unexpected shape for {var_name} in {file}: {values.shape}"
            )

        all_time.append(time[mask])
        all_data.append(values[mask, :3])

    if not all_time:
        raise FileNotFoundError(
            f"No samples found for {var_name} between {start} and {end}."
        )

    time = pd.DatetimeIndex(
        np.concatenate([t.to_numpy() for t in all_time])
    )
    data = np.vstack(all_data)

    order = np.argsort(time.asi8, kind="stable")
    time = time[order]
    data = data[order]

    _, unique_idx = np.unique(time.asi8, return_index=True)
    unique_idx = np.sort(unique_idx)

    return time[unique_idx], data[unique_idx]


def mvab(b_vectors: np.ndarray):
    b = np.asarray(b_vectors, dtype=float)

    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError("MVAB input must have shape (N, 3).")

    if len(b) < 3:
        raise ValueError("Too few magnetic-field samples for MVAB.")

    if not np.all(np.isfinite(b)):
        raise ValueError("MVAB input contains NaN or Inf.")

    b_centered = b - np.mean(b, axis=0)
    covariance = (b_centered.T @ b_centered) / len(b_centered)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    ratio_mid_min = eigenvalues[1] / eigenvalues[0]
    ratio_max_mid = eigenvalues[2] / eigenvalues[1]

    return (
        eigenvalues,
        eigenvectors,
        float(ratio_mid_min),
        float(ratio_max_mid),
    )


def normal_gradient(
    normal_list: np.ndarray,
    positions_re: np.ndarray,
) -> np.ndarray:
    normals = np.asarray(normal_list, dtype=float)
    positions = np.asarray(positions_re, dtype=float)

    if normals.shape != (4, 3):
        raise ValueError("normal_list must have shape (4, 3).")

    if positions.shape != (4, 3):
        raise ValueError("positions_re must have shape (4, 3).")

    x = positions - np.mean(positions, axis=0)

    R = (x.T @ x) / 4.0
    R_inv = np.linalg.inv(R)

    diff_x = np.empty((6, 3), dtype=float)
    diff_n = np.empty((6, 3), dtype=float)

    for i, (a, b) in enumerate(combinations(range(4), 2)):
        diff_x[i] = x[a] - x[b]
        diff_n[i] = normals[a] - normals[b]

    return (1.0 / 16.0) * diff_n.T @ diff_x @ R_inv


def nfa(
    normal_list: np.ndarray,
    positions_re: np.ndarray,
):
    normals = np.asarray(normal_list, dtype=float)
    positions = np.asarray(positions_re, dtype=float)

    normals = normals / np.linalg.norm(
        normals,
        axis=1,
        keepdims=True,
    )

    normal_center = normalize(np.mean(normals, axis=0))

    grad_n = normal_gradient(normals, positions)

    projection = (
        np.eye(3)
        - np.outer(normal_center, normal_center)
    )

    projected_gradient = grad_n @ projection

    eigenvalues, eigenvectors = np.linalg.eig(projected_gradient)

    eig_scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )
    max_eval_imag = float(
        np.max(np.abs(eigenvalues.imag))
    )
    max_evec_imag = float(
        np.max(np.abs(eigenvectors.imag))
    )

    if (
        max_eval_imag > 1.0e-9 * eig_scale
        or max_evec_imag > 1.0e-8
    ):
        raise ValueError(
            "NFA eigensystem has non-negligible complex components: "
            f"max Im(eigenvalue)={max_eval_imag:.3e}, "
            f"max Im(eigenvector)={max_evec_imag:.3e}."
        )

    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    eigenvectors = eigenvectors / np.linalg.norm(
        eigenvectors,
        axis=0,
        keepdims=True,
    )

    alignment = np.abs(
        eigenvectors.T @ normal_center
    )
    normal_idx = int(np.argmax(alignment))

    principal_idx = [
        i for i in range(3)
        if i != normal_idx
    ]
    principal_idx.sort(
        key=lambda i: abs(eigenvalues[i]),
        reverse=True,
    )

    order = principal_idx + [normal_idx]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return (
        normal_center,
        eigenvalues,
        eigenvectors,
        projected_gradient,
    )


def load_fgm_native_window(
    probe: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    files = sorted(
        FGM_DIR.glob(f"*mms{probe}*.cdf")
    )

    if not files:
        raise FileNotFoundError(
            f"No MMS{probe} FGM CDF files found in {FGM_DIR}"
        )

    variable = f"mms{probe}_fgm_b_gse_brst_l2"

    return _collect_cdf_series(
        files,
        variable,
        start,
        end,
    )


def load_mec_position_at_time(
    probe: int,
    target_time: pd.Timestamp,
):
    files = sorted(
        MEC_DIR.glob(f"*mms{probe}*.cdf")
    )

    if not files:
        raise FileNotFoundError(
            f"No MMS{probe} MEC CDF files found in {MEC_DIR}"
        )

    margin = pd.Timedelta(minutes=5)
    variable = f"mms{probe}_mec_r_gse"

    time, position_km = _collect_cdf_series(
        files,
        variable,
        target_time - margin,
        target_time + margin,
    )

    t_ns = time.asi8.astype(np.float64)
    target_ns = float(target_time.value)

    if target_ns < t_ns[0] or target_ns > t_ns[-1]:
        raise ValueError(
            f"MMS{probe} MEC data do not bracket {target_time}."
        )

    position_at_mid_km = np.array([
        np.interp(
            target_ns,
            t_ns,
            position_km[:, component],
        )
        for component in range(3)
    ])

    return position_at_mid_km / R_E_KM


def normal_error_statistic(
    normals,
    lambda_ratios,
    sample_counts,
):
    normals = np.asarray(normals, dtype=float)

    normals = normals / np.linalg.norm(
        normals,
        axis=1,
        keepdims=True,
    )

    n_mean = normalize(np.mean(normals, axis=0))

    delta_each = np.linalg.norm(
        normals - n_mean,
        axis=1,
    )

    mean_distance_to_center = float(
        np.mean(delta_each)
    )

    mean_normal_difference = (
        2.0 * mean_distance_to_center
    )

    angles_deg = np.degrees(
        np.arccos(
            np.clip(
                normals @ n_mean,
                -1.0,
                1.0,
            )
        )
    )

    ratios = np.asarray(
        lambda_ratios,
        dtype=float,
    )
    counts = np.asarray(
        sample_counts,
        dtype=float,
    )

    ratio_mean = float(np.mean(ratios))
    sample_count_mean = float(np.mean(counts))

    if ratio_mean <= 1.0:
        raise ValueError(
            "Mean lambda2/lambda3 must be > 1."
        )

    if sample_count_mean <= 1.0:
        raise ValueError(
            "Too few FGM samples."
        )

    delta_stat = (
        np.sqrt(
            ratio_mean
            / (sample_count_mean - 1.0)
        )
        / (ratio_mean - 1.0)
    )

    return {
        "mean_normal": n_mean,
        "mean_normal_difference":
            mean_normal_difference,
        "mean_distance_to_center":
            mean_distance_to_center,
        "individual_normal_angles_deg":
            angles_deg,
        "mean_angle_to_center_deg":
            float(np.mean(angles_deg)),
        "mean_lambda2_over_lambda3":
            ratio_mean,
        "mean_sample_count":
            sample_count_mean,
        "statistical_normal_error":
            float(delta_stat),
        "statistical_normal_error_deg":
            float(np.degrees(delta_stat)),
        "observed_to_statistical_error_ratio":
            float(
                mean_normal_difference
                / delta_stat
            ),
    }


def main():
    start = pd.Timestamp(START_TIME)
    end = pd.Timestamp(END_TIME)
    midpoint = start + (end - start) / 2

    print("=" * 70)
    print("MMS MP1 MVAB + NFA analysis")
    print(
        f"Requested interval : "
        f"{START_TIME} -- {END_TIME} UT"
    )
    print(
        f"MEC position time  : "
        f"{midpoint} UT"
    )
    print("=" * 70)

    mvab_results = []
    normals = []

    print("\n[MVAB results]")

    for probe in range(1, 5):
        time_fgm, b_fgm = load_fgm_native_window(
            probe,
            start,
            end,
        )

        (
            eigenvalues,
            eigenvectors,
            ratio_mid_min,
            ratio_max_mid,
        ) = mvab(b_fgm)

        normal = eigenvectors[:, 0].copy()

        if normal[0] < 0:
            normal = -normal

        normals.append(normal)

        mvab_results.append({
            "probe": probe,
            "time_start": time_fgm[0],
            "time_end": time_fgm[-1],
            "n_samples": len(time_fgm),
            "eigenvalues": eigenvalues,
            "normal": normal,
            "lambda2_over_lambda3":
                ratio_mid_min,
            "lambda1_over_lambda2":
                ratio_max_mid,
        })

        print(f"MMS{probe}")
        print(
            f"  native FGM interval : "
            f"{time_fgm[0]} -- "
            f"{time_fgm[-1]} UT"
        )
        print(
            f"  native FGM samples  : "
            f"{len(time_fgm)}"
        )
        print(
            f"  MVAB normal (GSE)   : "
            f"{np.round(normal, 5).tolist()}"
        )
        print(
            f"  lambda2/lambda3     : "
            f"{ratio_mid_min:.3f}"
        )

    normals = np.asarray(normals)

    positions_re = np.vstack([
        load_mec_position_at_time(
            probe,
            midpoint,
        )
        for probe in range(1, 5)
    ])

    print(
        "\n[MEC positions at analysis midpoint]"
    )

    for probe in range(1, 5):
        print(
            f"MMS{probe}: "
            f"{np.round(positions_re[probe - 1], 6).tolist()} R_E"
        )

    (
        normal_center,
        eigenvalues_nfa,
        eigenvectors_nfa,
        projected_gradient,
    ) = nfa(
        normals,
        positions_re,
    )

    k_nonzero = eigenvalues_nfa[:2]
    e_nonzero = eigenvectors_nfa[:, :2]

    order = np.argsort(
        np.abs(k_nonzero)
    )

    kappa_1 = float(
        k_nonzero[order[0]]
    )
    kappa_2 = float(
        k_nonzero[order[1]]
    )

    e_1 = e_nonzero[:, order[0]].copy()
    e_2 = e_nonzero[:, order[1]].copy()

    if e_1[1] > 0:
        e_1 = -e_1

    if e_2[2] > 0:
        e_2 = -e_2

    e_3 = eigenvectors_nfa[:, 2].copy()

    if np.dot(e_3, normal_center) < 0:
        e_3 = -e_3

    R_1 = 1.0 / abs(kappa_1)
    R_2 = 1.0 / abs(kappa_2)

    print("\n[NFA results]")

    print(
        "Mean outward normal (GSE): "
        f"{np.round(normal_center, 6).tolist()}"
    )

    print(
        f"kappa_1 = {kappa_1:+.6f} R_E^-1, "
        f"R_1 = {R_1:.6f} R_E, "
        f"e_1 = {np.round(e_1, 6).tolist()}"
    )

    print(
        f"kappa_2 = {kappa_2:+.6f} R_E^-1, "
        f"R_2 = {R_2:.6f} R_E, "
        f"e_2 = {np.round(e_2, 6).tolist()}"
    )

    print(
        f"kappa_3 = "
        f"{eigenvalues_nfa[2]:+.6e} R_E^-1, "
        f"e_3 = {np.round(e_3, 6).tolist()}"
    )

    angle_e1_e2 = vector_angle_deg(
        e_1,
        e_2,
    )
    angle_e2_e3 = vector_angle_deg(
        e_2,
        e_3,
    )
    angle_e1_e3 = vector_angle_deg(
        e_1,
        e_3,
    )

    e1_dot_e2_abs = abs(
        float(
            np.dot(
                normalize(e_1),
                normalize(e_2),
            )
        )
    )

    lambda_ratios = [
        item["lambda2_over_lambda3"]
        for item in mvab_results
    ]

    sample_counts = [
        item["n_samples"]
        for item in mvab_results
    ]

    normal_quality = normal_error_statistic(
        normals,
        lambda_ratios,
        sample_counts,
    )

    print("\n[Quality / error checks]")

    print("MVAB lambda2/lambda3:")
    print(
        "  values = "
        + str([
            round(value, 3)
            for value in lambda_ratios
        ])
    )
    print(
        f"  minimum = "
        f"{min(lambda_ratios):.3f}"
    )
    print(
        f"  mean    = "
        f"{normal_quality['mean_lambda2_over_lambda3']:.3f}"
    )

    print(
        "\nFour-spacecraft normal "
        "difference vs MVAB error:"
    )
    print(
        "  individual angles to "
        "mean normal (deg) = "
        + str([
            round(value, 6)
            for value in normal_quality[
                "individual_normal_angles_deg"
            ]
        ])
    )
    print(
        "  mean angle to mean normal = "
        f"{normal_quality['mean_angle_to_center_deg']:.6f} deg"
    )
    print(
        "  <|Delta n|> = "
        f"{normal_quality['mean_normal_difference']:.6f}"
    )
    print(
        "  statistical MVAB error = "
        f"{normal_quality['statistical_normal_error']:.6f} "
        f"({normal_quality['statistical_normal_error_deg']:.6f} deg)"
    )
    print(
        "  difference/error ratio = "
        f"{normal_quality['observed_to_statistical_error_ratio']:.3f}"
    )

    print("\nNFA direction angles:")
    print(
        f"  angle(e1, e2) = "
        f"{angle_e1_e2:.6f} deg"
    )
    print(
        f"  angle(e2, e3) = "
        f"{angle_e2_e3:.6f} deg"
    )
    print(
        f"  angle(e1, e3) = "
        f"{angle_e1_e3:.6f} deg"
    )
    print(
        f"  |e1 dot e2|   = "
        f"{e1_dot_e2_abs:.8f}"
    )
    print(
        "  angle(e3, mean normal) = "
        f"{vector_angle_deg(e_3, normal_center):.6f} deg"
    )
    print(
        "  |90 - angle(e1,e2)| = "
        f"{abs(90.0 - angle_e1_e2):.6f} deg"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
