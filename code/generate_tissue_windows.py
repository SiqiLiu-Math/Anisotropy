#!/usr/bin/env python3
"""Reconstruct buffered Voronoi tissue and displaced-centre benchmarks.

This implementation was written for the revision from the equations in the
manuscript. It does not reproduce or silently reuse an unavailable historical
notebook. All lengths are in the explicitly stated model units and k = 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import scipy
from scipy.spatial import Voronoi

from polygon import polygon_metrics


ROOT = Path(__file__).resolve().parents[1]


def save_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def ccw(vertices):
    vertices = np.asarray(vertices, dtype=float)
    cross = np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                   - vertices[:, 1] * np.roll(vertices[:, 0], -1))
    return vertices if cross > 0 else vertices[::-1]


def circumcenter(a, b, c):
    return np.linalg.solve(2 * np.vstack([b - a, c - a]),
                           [np.dot(b, b) - np.dot(a, a),
                            np.dot(c, c) - np.dot(a, a)])


def embedded_geometry(epsilon, psi):
    theta = np.arange(6) * np.pi / 3
    neighbors = np.column_stack([np.cos(theta), np.sin(theta)])
    center = epsilon * np.array([np.cos(psi), np.sin(psi)])
    vertices = np.array([circumcenter(center, neighbors[j], neighbors[(j+1) % 6])
                         for j in range(6)])
    return center, ccw(vertices), neighbors


def run_embedded():
    rows = []
    for epsilon in np.r_[0., np.logspace(-4, np.log10(.15), 61)]:
        for psi in [0., np.pi / 6]:
            center, vertices, _ = embedded_geometry(epsilon, psi)
            values = polygon_metrics(vertices, center=center)
            leading = epsilon ** 2 / np.sqrt(3)
            rows.append(dict(epsilon=float(epsilon), psi_rad=float(psi),
                             delta=float(values["delta"]), leading=leading,
                             ratio=float(values["delta"] / leading) if leading else "",
                             area=float(values["area"])))
    for epsilon in [.05, .10, .15]:
        for psi in np.linspace(0, 2*np.pi, 361):
            center, vertices, _ = embedded_geometry(epsilon, psi)
            values = polygon_metrics(vertices, center=center)
            leading = epsilon ** 2 / np.sqrt(3)
            rows.append(dict(epsilon=epsilon, psi_rad=float(psi),
                             delta=float(values["delta"]), leading=leading,
                             ratio=float(values["delta"] / leading),
                             area=float(values["area"])))
    save_csv(ROOT / "data/embedded_response.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=True)
    for epsilon, color in [(0., ".6"), (.15, "#1871b0")]:
        center, vertices, neighbors = embedded_geometry(epsilon, np.pi / 6)
        polygon = np.vstack([vertices, vertices[0]])
        axes[0].plot(*polygon.T, color=color, label=rf"$\varepsilon={epsilon:g}$")
        axes[0].plot(*center, marker="x", color=color, ms=7)
    axes[0].scatter(*neighbors.T, color="#5a8c34", s=18, label="Neighbor generators")
    axes[0].set(aspect="equal", xlabel="$x$", ylabel="$y$")
    axes[0].legend(fontsize=10, loc="lower left", frameon=False)
    for psi, label, color in [(0., r"$\psi=0$", "#1871b0"),
                              (np.pi/6, r"$\psi=\pi/6$", "#ce6a25")]:
        filtered = [r for r in rows[:124] if r["psi_rad"] == psi and r["epsilon"] > 0]
        axes[1].loglog([r["epsilon"] for r in filtered], [r["delta"] for r in filtered],
                       color=color, label=label)
    eps = np.logspace(-4, np.log10(.15), 100)
    axes[1].loglog(eps, eps**2 / np.sqrt(3), "k--", lw=1, label=r"$\varepsilon^2/\sqrt{3}$")
    axes[1].set(xlabel=r"Displacement $\varepsilon$", ylabel=r"$\Delta\lambda$")
    axes[1].legend(fontsize=10, frameon=False)
    for epsilon, color in zip([.05, .10, .15], ["#6d9f40", "#1871b0", "#ce6a25"]):
        filtered = [r for r in rows[124:] if r["epsilon"] == epsilon]
        axes[2].plot(np.rad2deg([r["psi_rad"] for r in filtered]),
                     100*(np.array([r["ratio"] for r in filtered])-1),
                     color=color, label=rf"$\varepsilon={epsilon:g}$")
    axes[2].set(xlabel=r"Direction $\psi$ (degrees)", ylabel="Excess over quadratic law (%)",
                xlim=(0, 120), xticks=[0, 30, 60, 90, 120])
    axes[2].legend(fontsize=10, frameon=False)
    for label, axis in zip("ABC", axes):
        axis.text(-.14, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.spines[["right", "top"]].set_visible(False)
    fig.savefig(ROOT / "figures/embedded_response.pdf")
    fig.savefig(ROOT / "figures/embedded_response.png", dpi=220)
    plt.close(fig)
    checks = []
    for epsilon in [.05, .10, .15]:
        for psi in [0., np.pi/6]:
            center, vertices, _ = embedded_geometry(epsilon, psi)
            ratio = float(polygon_metrics(vertices, center=center)["delta"] / (epsilon**2/np.sqrt(3)))
            checks.append(dict(epsilon=epsilon, psi_rad=psi, ratio=ratio))
    return checks


def bootstrap_rms(real_square, null_square, bootstrap_indices):
    actual = np.sqrt(np.mean(real_square))
    null = np.sqrt(np.mean(null_square))
    replicas_real = np.sqrt(np.mean(real_square[bootstrap_indices], axis=1))
    replicas_null = np.sqrt(np.mean(null_square[bootstrap_indices], axis=1))
    ratio = actual/null
    return dict(rms=actual, rms_ci_low=np.quantile(replicas_real, .025),
                rms_ci_high=np.quantile(replicas_real, .975), null_rms=null,
                null_rms_ci_low=np.quantile(replicas_null, .025),
                null_rms_ci_high=np.quantile(replicas_null, .975), ratio=ratio,
                ratio_ci_low=np.quantile(replicas_real/replicas_null, .025),
                ratio_ci_high=np.quantile(replicas_real/replicas_null, .975))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=202609092)
    parser.add_argument("--eta", type=float, default=.05)
    args = parser.parse_args()
    for folder in ["data", "figures"]:
        (ROOT/folder).mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.bbox": "tight"})

    extent = 23
    lattice_ij = np.array([(i, j) for i in range(-extent, extent+1)
                           for j in range(-extent, extent+1)])
    reference_q = np.column_stack([2*lattice_ij[:, 0]+lattice_ij[:, 1],
                                    np.sqrt(3)*lattice_ij[:, 1]])
    radius_squared = np.sum(reference_q**2, axis=1)
    shell_values = np.unique(np.round(radius_squared, 9))
    windows = []
    for m in [2, 4, 7, 10, 15, 20]:
        low = -(m//2)
        membership = np.all((lattice_ij >= low) & (lattice_ij < low+m), axis=1)
        windows.append(dict(shape="rhombus", target_n=m*m, parameter=float(m),
                            mask=membership, n=int(membership.sum())))
        counts = np.array([np.sum(radius_squared <= shell + 1.e-7) for shell in shell_values])
        # At an equal distance from the target use the larger whole shell.
        difference = np.abs(counts-m*m)
        shell_index = np.where(difference == difference.min())[0][-1]
        radius = float(np.sqrt((shell_values[shell_index]+shell_values[shell_index+1])/2))
        membership = radius_squared < radius**2
        windows.append(dict(shape="disk", target_n=m*m, parameter=radius,
                            mask=membership, n=int(membership.sum())))
    selected = np.flatnonzero(np.any([w["mask"] for w in windows], axis=0))
    selected_ij = lattice_ij[selected]
    selected_map = {tuple(ij): j for j, ij in enumerate(selected_ij)}
    for window in windows:
        window["local"] = np.flatnonzero(window["mask"][selected])
    # Only lattice bonds whose endpoints belong to the 20 x 20 rhombus enter C1.
    core_local = next(w["local"] for w in windows if w["shape"] == "rhombus" and w["n"] == 400)
    core_set = set(core_local)
    pairs = []
    for local in core_local:
        ij = selected_ij[local]
        for step in [(1, 0), (0, 1), (1, -1)]:
            other = selected_map.get(tuple(ij + np.array(step)))
            if other in core_set:
                pairs.append([local, other])
    pairs = np.asarray(pairs, dtype=int)

    rng = np.random.default_rng(args.seed)
    all_w = np.empty((args.realizations, len(selected)), dtype=complex)
    all_area = np.empty((args.realizations, len(selected)))
    all_q = np.empty((args.realizations, len(selected), 2))
    raw = []
    minimum_buffer_steps = extent - int(np.max(np.abs(selected_ij)))
    boundary_neighbors = 0
    minimum_area = np.inf
    maximum_trace_error = 0.
    example_polygons = None
    for realization in range(args.realizations):
        q = reference_q + rng.normal(0, args.eta, reference_q.shape)
        vor = Voronoi(q)
        # A window cell adjacent to the outermost generator ring is forbidden.
        selected_set = set(selected.tolist())
        for a, b in vor.ridge_points:
            if a in selected_set and np.max(np.abs(lattice_ij[b])) == extent:
                boundary_neighbors += 1
            if b in selected_set and np.max(np.abs(lattice_ij[a])) == extent:
                boundary_neighbors += 1
        polygons = []
        for local, global_index in enumerate(selected):
            indices = vor.regions[vor.point_region[global_index]]
            if not indices or -1 in indices:
                raise RuntimeError("Unbounded polygon entered a measurement window")
            vertices = ccw(vor.vertices[indices])
            result = polygon_metrics(vertices, center=q[global_index])
            if float(result["area"]) <= 0:
                raise RuntimeError("Nonpositive cell area")
            all_w[realization, local] = result["W"]
            all_area[realization, local] = result["area"]
            minimum_area = min(minimum_area, float(result["area"]))
            maximum_trace_error = max(maximum_trace_error,
                                      float(abs(result["trace"] - 2*result["area"])))
            if realization == 0:
                polygons.append(vertices)
        all_q[realization] = q[selected]
        if realization == 0:
            example_polygons = polygons
        for window in windows:
            ids = window["local"]
            total_w = np.sum(all_w[realization, ids])
            total_area = float(np.sum(all_area[realization, ids]))
            null_square = float(np.sum(np.abs(all_w[realization, ids])**2))
            raw.append(dict(realization=realization, shape=window["shape"],
                            target_n=window["target_n"], n=window["n"],
                            parameter=window["parameter"],
                            sum_W_real=float(total_w.real), sum_W_imag=float(total_w.imag),
                            total_area=total_area, delta_total=float(abs(total_w)),
                            area_normalized_delta=float(abs(total_w)/total_area),
                            orientation_null_squared=null_square,
                            area_normalized_null_squared=null_square/total_area**2))
        if (realization+1) % 25 == 0:
            print(f"Completed {realization+1}/{args.realizations} tissue realizations", flush=True)
    if boundary_neighbors:
        raise RuntimeError("Measurement cells reached boundary generators")
    save_csv(ROOT/"data/tissue_realizations.csv", raw)
    np.savez_compressed(ROOT/"data/tissue_cells.npz", W=all_w, area=all_area,
                        generator_positions=all_q, lattice_indices=selected_ij,
                        selected_global_indices=selected, reference_generators=reference_q,
                        nearest_neighbor_pairs=pairs)
    bootstrap_rng = np.random.default_rng(args.seed+1)
    bootstrap_indices = bootstrap_rng.integers(0, args.realizations,
                                              size=(2000, args.realizations))
    aggregate = []
    for window in windows:
        subset = [r for r in raw if r["shape"] == window["shape"] and r["target_n"] == window["target_n"]]
        integrated = bootstrap_rms(np.array([r["delta_total"]**2 for r in subset]),
                                  np.array([r["orientation_null_squared"] for r in subset]),
                                  bootstrap_indices)
        normalized = bootstrap_rms(np.array([r["area_normalized_delta"]**2 for r in subset]),
                                   np.array([r["area_normalized_null_squared"] for r in subset]),
                                   bootstrap_indices)
        record = dict(shape=window["shape"], target_n=window["target_n"], n=window["n"],
                      parameter=window["parameter"], realizations=args.realizations)
        record.update({"integrated_"+key: float(value) for key, value in integrated.items()})
        record.update({"area_normalized_"+key: float(value) for key, value in normalized.items()})
        aggregate.append(record)
    save_csv(ROOT/"data/tissue_summary.csv", aggregate)
    za, zb = all_w[:, pairs[:, 0]], all_w[:, pairs[:, 1]]
    # Do not call separate bonds independent observations: bootstrap tissues.
    numerator = np.mean(np.real(za*np.conj(zb)), axis=1)
    norm_a = np.mean(np.abs(za)**2, axis=1)
    norm_b = np.mean(np.abs(zb)**2, axis=1)
    correlation = np.mean(numerator)/np.sqrt(np.mean(norm_a)*np.mean(norm_b))
    boot_c = np.mean(numerator[bootstrap_indices], axis=1)/np.sqrt(
        np.mean(norm_a[bootstrap_indices], axis=1)*np.mean(norm_b[bootstrap_indices], axis=1))
    correlation_summary = dict(value=float(correlation), ci_low=float(np.quantile(boot_c, .025)),
                               ci_high=float(np.quantile(boot_c, .975)), pair_count=len(pairs),
                               definition="mean Re(W_a conjugate(W_b)) / sqrt(mean |W_a|^2 mean |W_b|^2), all three unoriented nearest lattice bonds in 20x20 rhombus; bootstrap over tissues")

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.5), constrained_layout=True)
    ax = axes[0, 0]
    rhombus = next(w for w in windows if w["shape"] == "rhombus" and w["target_n"] == 49)
    disk = next(w for w in windows if w["shape"] == "disk" and w["target_n"] == 49)
    base_ids = np.flatnonzero(np.linalg.norm(reference_q[selected], axis=1) < disk["parameter"]+4)
    ax.add_collection(PolyCollection([example_polygons[i] for i in base_ids],
                                    facecolors="none", edgecolors=".82", linewidths=.45))
    ax.add_collection(PolyCollection([example_polygons[i] for i in disk["local"]],
                                    facecolors="#dbeafa", edgecolors="none", alpha=.6))
    ax.add_collection(PolyCollection([example_polygons[i] for i in rhombus["local"]],
                                    facecolors="none", edgecolors="#ce6a25", linewidths=.7))
    length_scale = .7 / np.quantile(np.abs(all_w[0, base_ids]), .9)
    sticks = []
    for i in base_ids:
        angle = .5*np.angle(all_w[0, i])
        half = .5*length_scale*abs(all_w[0, i])*np.array([np.cos(angle), np.sin(angle)])
        sticks.append([all_q[0, i]-half, all_q[0, i]+half])
    ax.add_collection(LineCollection(sticks, colors="#243b55", linewidths=.8))
    ax.autoscale_view()
    ax.set(aspect="equal", xlabel="$x$", ylabel="$y$")
    colors = {"rhombus": "#ce6a25", "disk": "#1871b0"}
    for shape in ["rhombus", "disk"]:
        subset = [r for r in aggregate if r["shape"] == shape]
        n = np.array([r["n"] for r in subset])
        color = colors[shape]
        for axis, prefix in [(axes[0, 1], "integrated_"), (axes[1, 0], "area_normalized_")]:
            values = np.array([r[prefix+"rms"] for r in subset])
            low = np.array([r[prefix+"rms_ci_low"] for r in subset])
            high = np.array([r[prefix+"rms_ci_high"] for r in subset])
            axis.errorbar(n, values, yerr=np.vstack([values-low, high-values]),
                          marker="o", ms=3, capsize=2, color=color, label=shape.capitalize())
            axis.plot(n, [r[prefix+"null_rms"] for r in subset], "--", color=color,
                      label=shape.capitalize()+" orientation null")
        ratio = np.array([r["integrated_ratio"] for r in subset])
        low = np.array([r["integrated_ratio_ci_low"] for r in subset])
        high = np.array([r["integrated_ratio_ci_high"] for r in subset])
        axes[1, 1].errorbar(n, ratio, yerr=np.vstack([ratio-low, high-ratio]),
                            marker="o", ms=3, capsize=2, color=color, label=shape.capitalize())
    axes[0, 1].set(xscale="log", yscale="log", xlabel="Cells in window, $N$",
                   ylabel=r"RMS $|\sum_\alpha W_\alpha|$")
    axes[0, 1].legend(fontsize=10, frameon=False)
    axes[1, 0].set(xscale="log", yscale="log", xlabel="Cells in window, $N$",
                   ylabel=r"RMS $(|\sum_\alpha W_\alpha|/\sum_\alpha A_\alpha)$")
    axes[1, 1].set(xscale="log", xlabel="Cells in window, $N$", ylabel="RMS / orientation-null RMS")
    axes[1, 1].axhline(1, ls=":", lw=1, color=".4")
    axes[1, 1].legend(fontsize=10, frameon=False)
    for label, axis in zip("ABCD", axes.flat):
        axis.text(-.14, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.spines[["right", "top"]].set_visible(False)
    fig.savefig(ROOT/"figures/tissue_scaling.pdf")
    fig.savefig(ROOT/"figures/tissue_scaling.png", dpi=220)
    plt.close(fig)

    embedded_checks = run_embedded()
    metadata = dict(seed=args.seed, realizations=args.realizations, eta=args.eta,
                    noise="Independent Gaussian coordinate displacements of every generator; standard deviation eta per Cartesian coordinate",
                    spacing=2, k=1, finite_generator_count=len(reference_q),
                    generator_index_range=[-extent, extent], minimum_buffer_index_steps=minimum_buffer_steps,
                    outer_ring_neighbors_of_measurement_cells=boundary_neighbors,
                    minimum_cell_area=minimum_area, maximum_trace_minus_2area=maximum_trace_error,
                    window_definition="Union of complete Voronoi cells selected by their unperturbed generator labels; rhombi use m x m lattice-index blocks; disks include complete radial shells with the nearest available count to m^2. Membership stays fixed across realizations.",
                    quantity="W=(E11-E22)+2i E12, E=sum signed cross(v_i-q,v_i+1-q) t_i tensor t_i; k=1. W is an integrated cell-assigned interfacial deviator, not an area-normalized stress.",
                    pressure="Uniform pressure adds no deviatoric contribution.",
                    orientation_null="Conditional independent uniform orientations with observed cell |W| fixed: expected |sum W|^2 equals sum |W|^2. This is a statistical reference, not a force-balanced alternative tissue.",
                    area_normalization="Compute |sum W| / sum area separately for every tissue realization, then RMS; no second area weighting of W.",
                    uncertainty="Percentile 95% bootstrap confidence intervals from 2000 resamples of independent entire tissue realizations; windows and cells within a tissue are correlated.",
                    nearest_neighbor_correlation=correlation_summary,
                    embedded_unit_convention="Six neighbors at radius 1; k=1; central generator displaced by epsilon; exact circumcenters recomputed. This unit convention differs from tissue lattice spacing 2.",
                    embedded_checks=embedded_checks,
                    interpretation_limits="One small disorder amplitude, equal generator weights, triangular reference lattice, finite windows and two shapes; no universal exponent or thermodynamic-limit claim.",
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__))
    (ROOT/"data/tissue_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"nearest_neighbor_correlation": correlation_summary,
                      "embedded_checks": embedded_checks,
                      "windows": [{"shape": r["shape"], "n": r["n"],
                                   "rms": r["integrated_rms"], "ratio": r["integrated_ratio"]}
                                  for r in aggregate]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
