#!/usr/bin/env python3
"""Independent deterministic checks of the signed polygon stress model.

This verifier deliberately assembles the reference tensor with signed distances
to supporting lines, rather than the cross-product formula used in polygon.py.
It performs finite-difference checks; it does not rerun the Monte Carlo study.
Run from any directory: python code/verify_core.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from polygon import polygon_metrics, polygon_tensor, polygon_validity


def reference_tensor(vertices, center, k=1.0):
    """Assemble k * signed distance * edge length * tangent outer tangent."""
    vertices = np.asarray(vertices, dtype=float)
    center = np.asarray(center, dtype=float)
    result = np.zeros((2, 2))
    signed_weights = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        edge = end - start
        length = np.linalg.norm(edge)
        tangent = edge / length
        outward_normal = np.array([tangent[1], -tangent[0]])
        distance = np.dot((start + end) / 2 - center, outward_normal)
        signed_weights.append(distance * length)
        result += k * distance * length * np.outer(tangent, tangent)
    return result, np.asarray(signed_weights)


def independent_area(vertices):
    """Signed area from a triangle fan anchored at the first vertex."""
    vectors = np.asarray(vertices)[1:] - vertices[0]
    return sum(
        (left[0] * right[1] - left[1] * right[0]) / 2
        for left, right in zip(vectors[:-1], vectors[1:])
    )


def spin2(tensor):
    return complex(tensor[0, 0] - tensor[1, 1], 2 * tensor[0, 1])


def vertices_on_rays(radii, angles):
    return np.asarray(radii)[:, None] * np.column_stack(
        [np.cos(angles), np.sin(angles)]
    )


def normalized_components(vertices, center):
    tensor = np.asarray(polygon_tensor(vertices, center=center))
    value = spin2(tensor) / np.trace(tensor)
    return np.array([value.real, value.imag])


def audit_saved_data(data_directory):
    """Recompute reported point statistics from saved draws, without new draws."""
    report = {}
    noise_path = data_directory / "single_noise_samples.npz"
    tissue_path = data_directory / "tissue_cells.npz"
    if noise_path.exists():
        errors = []
        groups = 0
        with np.load(noise_path) as samples:
            for filename, signal in [("single_noise_calibration.csv", False),
                                     ("single_noise_detection.csv", True)]:
                with (data_directory / filename).open() as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    groups += 1
                    n = int(row["n"])
                    sigma = float(row["sigma_over_R"])
                    key = (f"signal_b{float(row['b_over_R']):g}" if signal
                           else f"n{n}_sigma{sigma:g}")
                    valid = samples[key + "_valid"]
                    a = samples[key + "_a"][valid]
                    assert len(a) == int(row["accepted"])
                    assert len(valid) == int(row["drawn"])
                    threshold = (1.959963984540054 * sigma if n == 4
                                 else 2 * np.sqrt(np.log(20) / n) * sigma)
                    threshold_column = "predicted_null_a95" if signal else "predicted_a95"
                    errors.append(abs(threshold - float(row[threshold_column])))
                    fraction = np.mean(a > threshold)
                    errors.append(abs(fraction - float(row["detection_rate" if signal else "false_positive_rate"])))
                    if not signal:
                        errors.extend([abs(np.mean(a) - float(row["mean_a"])),
                                       abs(np.quantile(a, .95) - float(row["empirical_a95"]))])
                    assert np.all((a >= 0) & (a <= 1 + 1e-13))
        assert max(errors) < 2e-13
        report["noise"] = {"groups_recomputed_from_saved_draws": groups,
                           "max_point_statistic_error": float(max(errors))}
    else:
        report["noise"] = {"status": "not checked: saved sample file absent"}

    if tissue_path.exists():
        errors = []
        with np.load(tissue_path) as tissue:
            w, areas = tissue["W"], tissue["area"]
            indices = tissue["lattice_indices"]
            reference = np.column_stack([2 * indices[:, 0] + indices[:, 1],
                                          np.sqrt(3) * indices[:, 1]])
            distances_squared = np.sum(reference**2, axis=1)
            pairs = tissue["nearest_neighbor_pairs"]
            with (data_directory / "tissue_realizations.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            grouped = {}
            for row in rows:
                if row["shape"] == "rhombus":
                    m = int(float(row["parameter"]))
                    low = -(m // 2)
                    mask = np.all((indices >= low) & (indices < low + m), axis=1)
                else:
                    mask = distances_squared < float(row["parameter"])**2
                assert mask.sum() == int(row["n"])
                realization = int(row["realization"])
                total = np.sum(w[realization, mask])
                area = np.sum(areas[realization, mask])
                null_square = np.sum(np.abs(w[realization, mask])**2)
                exact = {"sum_W_real": total.real, "sum_W_imag": total.imag,
                         "total_area": area, "delta_total": abs(total),
                         "area_normalized_delta": abs(total) / area,
                         "orientation_null_squared": null_square,
                         "area_normalized_null_squared": null_square / area**2}
                errors.extend(abs(value - float(row[name])) for name, value in exact.items())
                grouped.setdefault((row["shape"], row["target_n"]), []).append(exact)
            with (data_directory / "tissue_summary.csv").open() as stream:
                summary = list(csv.DictReader(stream))
            for row in summary:
                members = grouped[(row["shape"], row["target_n"])]
                for prefix, actual_key, null_key in [
                    ("integrated_", "delta_total", "orientation_null_squared"),
                    ("area_normalized_", "area_normalized_delta", "area_normalized_null_squared")
                ]:
                    actual = np.sqrt(np.mean([member[actual_key]**2 for member in members]))
                    null = np.sqrt(np.mean([member[null_key] for member in members]))
                    for name, value in [("rms", actual), ("null_rms", null), ("ratio", actual / null)]:
                        errors.append(abs(value - float(row[prefix + name])))
            a, b = w[:, pairs[:, 0]], w[:, pairs[:, 1]]
            correlation = np.mean(np.real(a * np.conj(b))) / np.sqrt(np.mean(abs(a)**2) * np.mean(abs(b)**2))
            metadata = json.loads((data_directory / "tissue_metadata.json").read_text())
            errors.append(abs(correlation - metadata["nearest_neighbor_correlation"]["value"]))
        assert max(errors) < 2e-11
        report["tissue"] = {"realization_window_rows_recomputed": len(rows),
                            "summary_rows_recomputed": len(summary),
                            "max_point_statistic_error": float(max(errors)),
                            "neighbor_correlation": float(correlation)}
    else:
        report["tissue"] = {"status": "not checked: saved tissue file absent"}

    ellipse_path = data_directory / "ellipse_scan.csv"
    if ellipse_path.exists():
        errors = []
        with ellipse_path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            n, eccentricity = int(row["n"]), float(row["eccentricity"])
            s = np.sqrt(1 - eccentricity**2)
            theta = 2 * np.pi * np.arange(n) / n
            regular = vertices_on_rays(np.ones(n), theta)
            edge = np.roll(regular, -1, axis=0) - regular
            hx2, hy2 = edge[:, 0]**2, edge[:, 1]**2
            normalized = np.mean((hx2 - s*s*hy2) / (hx2 + s*s*hy2))
            trace = n * s * np.sin(2 * np.pi / n)
            errors.extend([abs(trace - float(row["trace_Es"])),
                           abs(abs(normalized) - float(row["normalized_gap"])),
                           abs(trace * normalized - float(row["W_real"]))])
        assert max(errors) < 2e-13
        # Rotating the square before this affine map produces an isotropic rectangle.
        rotated_square = vertices_on_rays(np.ones(4), np.pi / 4 + np.arange(4) * np.pi / 2)
        rectangle_errors = [abs(spin2(polygon_tensor(rotated_square * [1, s], center=np.zeros(2))))
                            for s in [1, .5, .1, .01]]
        assert max(rectangle_errors) < 2e-13
        report["ellipse"] = {"rows_crosschecked_with_closed_normalized_expression": len(rows),
                             "max_expression_error": float(max(errors)),
                             "max_rotated_square_rectangle_deviator": float(max(rectangle_errors))}
    else:
        report["ellipse"] = {"status": "not checked: ellipse scan absent"}
    return report


def run_checks():
    report = {}
    zero = np.zeros(2)
    random = np.random.default_rng(71349)

    pentagon = vertices_on_rays(np.ones(5), 2 * np.pi * np.arange(5) / 5)
    assert bool(polygon_validity(pentagon, center=zero)["valid"])
    pentagram = pentagon[[0, 2, 4, 1, 3]]
    assert not bool(polygon_validity(pentagram, center=zero)["convex_ccw"])
    assert not bool(polygon_validity(pentagon[::-1], center=zero)["convex_ccw"])
    assert not bool(polygon_validity(pentagon, center=[3, -4])["center_inside"])
    concave = pentagon.copy()
    concave[1] *= .1
    assert not bool(polygon_validity(concave, center=zero)["convex_ccw"])
    batch = np.stack([pentagon, pentagram, pentagon[::-1], concave])
    assert polygon_validity(batch, center=zero)["valid"].tolist() == [True, False, False, False]
    report["domain_classifier"] = {"convex_accepted": True,
                                   "self_intersecting_pentagram_rejected": True,
                                   "clockwise_and_concave_rejected": True,
                                   "outside_center_identified": True,
                                   "batch_classification_consistent": True}

    # Irregular convex polygons, including centers outside their convex hull.
    tensor_errors, trace_errors, metric_errors = [], [], []
    for n in range(3, 13):
        angles = np.arange(n) * 2 * np.pi / n
        angles += random.uniform(-0.15, 0.15, n) * 2 * np.pi / n
        points = vertices_on_rays(np.ones(n), angles)
        points = points @ np.array([[1.2, 0.25], [0.0, 0.8]])
        for center in [zero, np.array([3.0, -4.0])]:
            expected, weights = reference_tensor(points, center, k=1.7)
            actual = np.asarray(polygon_tensor(points, center=center, k=1.7))
            tensor_errors.append(float(np.max(np.abs(actual - expected))))
            trace_errors.append(float(abs(np.trace(actual) - 3.4 * independent_area(points))))
            metrics = polygon_metrics(points, center=center, k=1.7)
            metric_errors.extend([
                float(abs(metrics["W"] - spin2(actual))),
                float(abs(metrics["delta"] - abs(spin2(actual)))),
                float(abs(metrics["trace"] - np.trace(actual))),
                float(abs(metrics["area"] - independent_area(points))),
                float(abs(metrics["anisotropy"] - abs(spin2(actual)) / np.trace(actual))),
                float(np.max(np.abs(np.asarray(metrics["tensor"]) - actual))),
            ])
            if np.linalg.norm(center) > 0:
                assert np.any(weights < 0), "Outside-center case did not exercise signed weights"
    assert max(tensor_errors) < 2e-13
    assert max(trace_errors) < 2e-13
    assert max(metric_errors) < 2e-13
    report["independent_tensor"] = {
        "max_absolute_tensor_error": max(tensor_errors),
        "max_absolute_trace_error": max(trace_errors),
        "max_metric_error": max(metric_errors),
        "polygon_cases": 20,
        "includes_negative_signed_edge_weights": True,
    }

    # A rigid transformation must transform E as Q E Q^T and W as exp(2i phi) W.
    angles = np.deg2rad([0, 43, 115, 174, 247, 313])
    points = vertices_on_rays([1.2, 0.9, 1.0, 1.1, 0.85, 1.05], angles)
    center = np.array([0.05, -0.08])
    phi = 0.731
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    translation = np.array([2.34, -4.56])
    original = np.asarray(polygon_tensor(points, center=center))
    transformed = np.asarray(polygon_tensor(
        points @ rotation.T + translation, center=rotation @ center + translation
    ))
    covariance_error = float(np.max(np.abs(transformed - rotation @ original @ rotation.T)))
    spin_error = float(abs(spin2(transformed) - np.exp(2j * phi) * spin2(original)))
    assert covariance_error < 2e-13 and spin_error < 2e-13
    report["rigid_transform"] = {"tensor_error": covariance_error, "spin2_error": spin_error}

    # Any ordered vertices on a circle give the signed tensor k A I.
    cyclic_errors = []
    for n in range(3, 13):
        for _ in range(7):
            angles = np.sort(random.uniform(0, 2 * np.pi, n))
            points = vertices_on_rays(np.full(n, 1.3), angles)
            tensor = np.asarray(polygon_tensor(points, center=zero, k=0.9))
            cyclic_errors.append(float(np.max(np.abs(tensor - 0.9 * independent_area(points) * np.eye(2)))))
    assert max(cyclic_errors) < 2e-13
    report["cyclic_polygons"] = {"cases": len(cyclic_errors), "max_isotropy_error": max(cyclic_errors)}

    # Central differences independently verify all radial coefficients.
    radial_errors, angular_errors, noise_errors = [], [], []
    noise_eigenvalues = {}
    radius, k, step = 1.37, 0.83, 1e-6
    for n in range(3, 13):
        angles = np.arange(n) * 2 * np.pi / n
        points = vertices_on_rays(np.full(n, radius), angles)
        for j in range(n):
            radial_direction = np.array([np.cos(angles[j]), np.sin(angles[j])])
            plus, minus = points.copy(), points.copy()
            plus[j] += step * radial_direction
            minus[j] -= step * radial_direction
            derivative = (spin2(polygon_tensor(plus, center=zero, k=k)) - spin2(polygon_tensor(minus, center=zero, k=k))) / (2 * step)
            expected = 2 * k * radius * np.sin(2 * np.pi / n) * np.exp(2j * angles[j])
            radial_errors.append(float(abs(derivative - expected)))
            angle_plus, angle_minus = angles.copy(), angles.copy()
            angle_plus[j] += step
            angle_minus[j] -= step
            plus = vertices_on_rays(np.full(n, radius), angle_plus)
            minus = vertices_on_rays(np.full(n, radius), angle_minus)
            derivative = (spin2(polygon_tensor(plus, center=zero, k=k)) - spin2(polygon_tensor(minus, center=zero, k=k))) / (2 * step)
            angular_errors.append(float(abs(derivative)))

        # IID coordinate errors with unit variance have covariance J J^T.
        jacobian = np.zeros((2, 2 * n))
        for j in range(n):
            for coordinate in range(2):
                plus, minus = points.copy(), points.copy()
                plus[j, coordinate] += step
                minus[j, coordinate] -= step
                jacobian[:, 2 * j + coordinate] = (normalized_components(plus, zero) - normalized_components(minus, zero)) / (2 * step)
        covariance = jacobian @ jacobian.T
        expected = np.diag([1 / radius**2, 0]) if n == 4 else 2 / (n * radius**2) * np.eye(2)
        noise_errors.append(float(np.max(np.abs(covariance - expected))))
        noise_eigenvalues[str(n)] = np.linalg.eigvalsh(covariance).tolist()
    assert max(radial_errors) < 2e-8
    assert max(angular_errors) < 2e-8
    assert max(noise_errors) < 2e-8
    report["linear_response"] = {
        "max_radial_derivative_error": max(radial_errors),
        "max_angular_derivative_magnitude": max(angular_errors),
        "finite_difference_step": step,
    }
    report["cartesian_noise_linear_covariance"] = {
        "fixed_known_center": True,
        "radius": radius,
        "max_covariance_error": max(noise_errors),
        "covariance_eigenvalues_per_unit_coordinate_variance": noise_eigenvalues,
        "n4_is_rank_one": True,
    }

    # Explicit noncyclic isotropic quadrilateral: r2 = r4 = 1.
    f = lambda r: r * (r * r - 1) / (r * r + 1)
    r1 = 1.2
    r3 = brentq(lambda r: f(r) + f(r1), 0.6, 1.0, xtol=1e-14)
    points = vertices_on_rays([r1, 1, r3, 1], np.arange(4) * np.pi / 2)
    tensor = np.asarray(polygon_tensor(points, center=zero))
    # These four axis intercepts are cyclic iff r1*r3 = r2*r4.
    cyclicity_gap = abs(r1 * r3 - 1)
    assert abs(spin2(tensor)) < 2e-13 and cyclicity_gap > 0.01
    report["noncyclic_isotropic_quadrilateral"] = {
        "radii": [r1, 1, float(r3), 1],
        "root_residual": float(abs(f(r3) + f(r1))),
        "W_real": spin2(tensor).real,
        "W_imag": spin2(tensor).imag,
        "cyclicity_gap_abs_r1r3_minus_1": float(cyclicity_gap),
    }

    # Identical radii, distinct angle sequences: the radius list does not determine W.
    radii = [1.01, 1, 1, 1, 1, 1]
    angles_a = np.deg2rad([0, 60, 120, 180, 240, 300])
    angles_b = np.deg2rad([0, 30, 120, 180, 240, 330])
    wa = spin2(polygon_tensor(vertices_on_rays(radii, angles_a), center=zero))
    wb = spin2(polygon_tensor(vertices_on_rays(radii, angles_b), center=zero))
    assert abs(wa - wb) > 1e-3
    report["same_radii_different_angles"] = {
        "radii": radii,
        "W_A": [wa.real, wa.imag],
        "W_B": [wb.real, wb.imag],
        "delta_A": float(abs(wa)),
        "delta_B": float(abs(wb)),
        "difference_abs": float(abs(wa - wb)),
    }

    # Crosscheck the embedded-cell routine against the closed axial coordinates
    # in the supplement, and verify the spin-2 quadratic coefficient off axis.
    from generate_tissue_windows import embedded_geometry
    geometry_errors, axial_errors, quadratic_errors = [], [], []
    for epsilon in [0, .05, .10, .15]:
        upper = np.array([[(1 - epsilon) / 2, (1 + epsilon) / (2 * np.sqrt(3))],
                          [-epsilon, (1 - epsilon**2) / np.sqrt(3)],
                          [-(1 + epsilon) / 2, (1 - epsilon) / (2 * np.sqrt(3))]])
        expected_vertices = np.vstack([upper, upper[::-1] * [1, -1]])
        center, vertices, _ = embedded_geometry(epsilon, 0)
        measured_vertices = vertices - center
        distances = np.linalg.norm(measured_vertices[:, None] - expected_vertices[None], axis=2)
        geometry_errors.append(float(np.max(np.min(distances, axis=1))))
        expected_tensor, _ = reference_tensor(expected_vertices, zero)
        axial_errors.append(float(abs(spin2(expected_tensor) - epsilon**2 / np.sqrt(3))))
    for psi in np.linspace(0, np.pi / 3, 7):
        coefficients = []
        for epsilon in [.005, .0025]:
            center, vertices, _ = embedded_geometry(epsilon, psi)
            tensor, _ = reference_tensor(vertices, center)
            coefficients.append(spin2(tensor) / epsilon**2)
        extrapolated = (4 * coefficients[1] - coefficients[0]) / 3
        quadratic_errors.append(float(abs(extrapolated - np.exp(2j * psi) / np.sqrt(3))))
    assert max(geometry_errors) < 2e-13
    assert max(axial_errors) < 2e-13
    assert max(quadratic_errors) < 2e-8
    report["embedded_cell"] = {"max_closed_axial_coordinate_error": max(geometry_errors),
                               "max_exact_axial_W_error": max(axial_errors),
                               "max_extrapolated_quadratic_coefficient_error": max(quadratic_errors),
                               "off_axis_directions_checked": len(quadratic_errors)}
    report["status"] = "all checks passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "verification.json")
    args = parser.parse_args()
    report = run_checks()
    report["saved_data_audit"] = audit_saved_data(Path(__file__).resolve().parents[1] / "data")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"])
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
