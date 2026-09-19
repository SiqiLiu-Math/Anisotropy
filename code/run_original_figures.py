"""Regenerate William's Figures 5--7 with corrected, explicit protocols.

The plotting layouts and parameter families follow cells 19/21/23/25 of
WilliamResearchCode_annotated.ipynb.  Publication figure assets are preserved in the default output directory.
Figure 5 is NEW seeded sampling, conditioned on strict convexity and the fixed
mechanical center lying inside.  Candidate/accepted counts and Monte Carlo
uncertainties are exported; this conditioning is part of the experiment.

Run from any directory: python code/run_original_figures.py
Dependencies: numpy, matplotlib.  polygon.py must be next to this script.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Ellipse
from matplotlib.lines import Line2D

from polygon import polygon_tensor, polygon_metrics, polygon_validity

ROOT = Path(__file__).resolve().parents[1]
FIGURES, DATA = ROOT / "figures", ROOT / "data"
BASE_SEED = 20260909
SIGMAS = np.linspace(0, 0.1, 11)
TH6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
CELL_FILL, CELL_EDGE = "#A9D6F2", "#111111"
GREEN_D, ORANGE, BLUE, GREY, PURPLE = (
    "#217A3B", "#E8703A", "#2E6DB4", "#8A8A8A", "#7E57C2")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04, "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 3,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    "legend.fontsize": 10, "legend.frameon": False,
    "legend.handlelength": 1.6, "legend.labelspacing": 0.3,
    "lines.linewidth": 1.5, "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": True,
})


def panel_label(ax, letter, dx=-0.22, dy=1.14, size=16):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="top", ha="left")


def gap(tensor):
    """Eigenvalue gap of a real symmetric 2 by 2 tensor."""
    return np.hypot(tensor[..., 0, 0] - tensor[..., 1, 1],
                    2*tensor[..., 0, 1])


def independent_tensor(vertices, center=(0., 0.)):
    """Loop over supporting normals, independently of the vectorized core."""
    v = np.asarray(vertices) - np.asarray(center)
    answer = np.zeros((2, 2))
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        tangent = b-a
        length = np.linalg.norm(tangent)
        tangent /= length
        outward = np.array([tangent[1], -tangent[0]])
        d = np.dot(a, outward)
        answer += d*length*np.outer(tangent, tangent)
    return answer


def ellipse_polygon(n, eccentricity=0.):
    theta = 2*np.pi*np.arange(n)/n
    return np.column_stack((np.cos(theta),
                            np.sqrt(1-eccentricity**2)*np.sin(theta)))


def save_csv(name, rows):
    with (DATA / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)


def conditioned_sample(n, sigma, angular_sigma, count, seed):
    """Accept exactly count convex CCW polygons with center strictly inside.

    Angles are not sorted and vertices are not projected. Each candidate is a
    fresh draw from the notebook's Gaussian angular/radial family. Disallowed
    geometries are explicitly rejected. Only candidates through the final
    accepted sample enter the attempted count, even if a larger batch is drawn.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    theta0 = 2*np.pi*np.arange(n)/n
    chunks, accepted, attempted, rejected = [], 0, 0, 0
    max_error = 0.
    while accepted < count:
        batch = min(25000, max(64, int(1.2*(count-accepted))))
        theta = theta0 + angular_sigma*rng.standard_normal((batch, n))
        radius = 1 + sigma*rng.standard_normal((batch, n))
        v = np.stack((radius*np.cos(theta), radius*np.sin(theta)), axis=-1)
        valid = polygon_validity(v)["valid"]
        selected = np.flatnonzero(valid)
        need = count-accepted
        if len(selected) >= need:
            consumed = int(selected[need-1])+1
            selected = selected[:need]
        else:
            consumed = batch
        attempted += consumed
        rejected += consumed-len(selected)
        if len(selected):
            good = v[selected]
            tensor = polygon_tensor(good)
            for j in {0, len(good)//2, len(good)-1}:
                max_error = max(max_error, float(np.max(np.abs(
                    tensor[j]-independent_tensor(good[j])))))
            chunks.append(gap(tensor))
            accepted += len(selected)
        if attempted > 50*count:
            raise RuntimeError("Acceptance too low; inspect the geometry protocol")
    values = np.concatenate(chunks)
    assert len(values) == count and attempted == count+rejected
    assert max_error < 1e-12
    sd = float(np.std(values, ddof=1))
    return {
        "accepted": count, "attempted": attempted, "rejected": rejected,
        "acceptance_fraction": count/attempted,
        "mean_gap": float(np.mean(values)), "sample_sd": sd,
        "sem": sd/np.sqrt(count), "normal_assembly_max_error": max_error,
    }


def fitted_scan(panel, n, angular_sigma, count, angular_index=0):
    rows = []
    for j, sigma in enumerate(SIGMAS):
        seed = [BASE_SEED, ord(panel), int(n), angular_index, j]
        stats = conditioned_sample(n, sigma, angular_sigma, count, seed)
        rows.append(dict(panel=panel, n=int(n), sigma=float(sigma),
                         angular_sigma=float(angular_sigma),
                         seed_sequence=";".join(map(str, seed)), **stats))
    x = SIGMAS
    y = np.array([r["mean_gap"] for r in rows])
    sem = np.array([r["sem"] for r in rows])
    X = np.column_stack((x, np.ones_like(x)))
    A = np.linalg.inv(X.T@X)@X.T
    slope, intercept = A@y
    residual = y-X@np.array([slope, intercept])
    r2 = 1-float(residual@residual)/float((y-y.mean())@(y-y.mean()))
    # Independent sample means: propagate their measured Monte Carlo variance.
    # These errors do not account for finite-amplitude bias or fit curvature.
    mc_cov = (A*sem[None, :]**2)@A.T
    residual_cov = float(residual@residual)/(len(x)-2)*np.linalg.inv(X.T@X)
    # Parametric uncertainty of R² under the CLT for the independently sampled
    # means. It is not an assessment of whether a linear model is exact.
    rng = np.random.default_rng([BASE_SEED, 900, ord(panel), n, angular_index])
    bootstrap_y = y + rng.standard_normal((2000, len(y)))*sem
    bootstrap_fit = (A@bootstrap_y.T).T@X.T
    r2_draws = 1-np.sum((bootstrap_y-bootstrap_fit)**2, axis=1)/np.sum(
        (bootstrap_y-bootstrap_y.mean(axis=1, keepdims=True))**2, axis=1)
    fit = dict(panel=panel, n=int(n), angular_sigma=float(angular_sigma),
               fit="unweighted OLS with free intercept", sigma_min=0., sigma_max=.1,
               slope=float(slope), intercept=float(intercept), r_squared=r2,
               slope_mc_se=float(np.sqrt(mc_cov[0, 0])),
               intercept_mc_se=float(np.sqrt(mc_cov[1, 1])),
               slope_residual_se=float(np.sqrt(residual_cov[0, 0])),
               intercept_residual_se=float(np.sqrt(residual_cov[1, 1])),
               r_squared_mc_q025=float(np.quantile(r2_draws, .025)),
               r_squared_mc_q975=float(np.quantile(r2_draws, .975)))
    return rows, fit


def figure5():
    fig = plt.figure(figsize=(7.1, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.46,
                         left=.09, right=.985, top=.90, bottom=.09)
    axA = fig.add_subplot(gs[0, 0])
    examples = []
    for eps, xo in zip([0., .15, .35], [0., 2.5, 5.]):
        radius = np.ones(6); radius[0] += eps
        v = np.column_stack((radius*np.cos(TH6), radius*np.sin(TH6)))
        delta = float(gap(polygon_tensor(v)))
        examples.append(dict(epsilon=eps, gap=delta))
        axA.add_patch(plt.Circle((xo, 0), 1., fill=False, ls=(0, (3, 2)),
                                lw=.9, ec=GREY, zorder=1))
        axA.add_patch(MplPoly(v+[xo, 0], closed=True, fc=CELL_FILL,
                             ec=CELL_EDGE, lw=1.4, zorder=2))
        axA.plot(v[:, 0]+xo, v[:, 1], 'o', ms=3.4,
                 color=CELL_EDGE if eps == 0 else ORANGE, zorder=3)
        axA.plot(xo, 0, '+', ms=6, mew=1.3, color=CELL_EDGE, zorder=4)
        pass  # in-panel numeric label removed
        pass  # in-panel numeric label removed
    axA.set_xlim(-1.5, 6.5); axA.set_ylim(-2.35, 1.45)
    axA.set_aspect('equal'); axA.axis('off')
    panel_label(axA, 'A')
    pass  # panel subtitle removed
    all_rows, all_fits = [], []
    axB = fig.add_subplot(gs[0, 1])
    rows, fit = fitted_scan("B", 6, 0., 20000)
    all_rows.extend(rows); all_fits.append(fit)
    y = [r["mean_gap"] for r in rows]; sem = [r["sem"] for r in rows]
    axB.plot(SIGMAS, fit['slope']*SIGMAS+fit['intercept'], '-', color=BLUE,
             lw=1.4, zorder=1)
    axB.errorbar(SIGMAS, y, yerr=sem, fmt='o', ms=4.2, color=BLUE,
                 mfc='white', mew=1.2, elinewidth=.9, capsize=1.6, zorder=2)
    axB.set_xlabel(r"perturbation amplitude  $\sigma_\epsilon$")
    axB.set_ylabel(r"$\langle\Delta\lambda\rangle$")
    pass  # in-panel fit annotation removed
    axB.set_xlim(-.004, .104); axB.set_ylim(bottom=-.012)
    panel_label(axB, 'B')
    pass  # panel subtitle removed
    axC = fig.add_subplot(gs[1, 0])
    ns = np.arange(3, 11)
    cmap = plt.cm.viridis(np.linspace(.05, .92, len(ns)))
    slopes, slope_ses = [], []
    for col, n in zip(cmap, ns):
        rows, fit = fitted_scan("C", int(n), 0., 6000)
        all_rows.extend(rows); all_fits.append(fit)
        slopes.append(fit['slope']); slope_ses.append(fit['slope_mc_se'])
        axC.errorbar(SIGMAS, [r['mean_gap'] for r in rows],
                     yerr=[r['sem'] for r in rows], fmt='o', ms=3., color=col,
                     alpha=.85, elinewidth=.6, capsize=1.)
        axC.plot(SIGMAS, fit['slope']*SIGMAS+fit['intercept'], '-', color=col, lw=1.2)
    axC.set_xlabel(r"perturbation amplitude  $\sigma_\epsilon$")
    axC.set_ylabel(r"$\langle\Delta\lambda\rangle$")
    axC.set_xlim(-.004, .104)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(3, 10))
    cb = fig.colorbar(sm, ax=axC, pad=.015, fraction=.045, aspect=16)
    cb.set_label("number of sides  $n$", fontsize=10, labelpad=2)
    cb.set_ticks(list(range(3, 11)))
    cb.ax.tick_params(labelsize=7.5, width=.8, length=2.4)
    cb.outline.set_linewidth(.8)
    axins = axC.inset_axes([.67, .16, .31, .22])
    axins.errorbar(ns, slopes, yerr=slope_ses, fmt='o-', ms=3.2, lw=1.1,
                   color='#333333', capsize=1.)
    axins.set_xlabel(r"$n$", fontsize=7.5, labelpad=1)
    axins.set_ylabel("fit slope", fontsize=7.5, labelpad=2)
    axins.tick_params(labelsize=6.5, width=.7, length=2)
    axins.patch.set_alpha(0.)
    panel_label(axC, 'C')
    pass  # panel subtitle removed
    axD = fig.add_subplot(gs[1, 1])
    for j, (angular, col) in enumerate(zip([0., .02, .05, .10],
                                    ['#1f4e8c', '#2e8b57', '#d9822b', '#b03a48'])):
        rows, fit = fitted_scan("D", 6, angular, 4000, j)
        all_rows.extend(rows); all_fits.append(fit)
        axD.errorbar(SIGMAS, [r['mean_gap'] for r in rows],
                     yerr=[r['sem'] for r in rows], fmt='o', ms=3.4, color=col,
                     alpha=.9, elinewidth=.6, capsize=1.)
        axD.plot(SIGMAS, fit['slope']*SIGMAS+fit['intercept'], '-', color=col,
                 lw=1.2, label=r"$\sigma_\delta=%.2f$" % angular)
    axD.set_xlabel(r"perturbation amplitude  $\sigma_\epsilon$")
    axD.set_ylabel(r"$\langle\Delta\lambda\rangle$")
    axD.set_xlim(-.004, .104)
    axD.legend(title="angular disorder", title_fontsize=8,
               loc='upper left', borderpad=.2, frameon=False)
    panel_label(axD, 'D')
    pass  # panel subtitle removed
    save_figure(fig, "figure5")
    save_csv("legacy_figure5_samples.csv", all_rows)
    save_csv("legacy_figure5_fits.csv", all_fits)
    return dict(examples=examples, fits=all_fits,
                protocol="Gaussian candidates conditioned on strict convex CCW "
                         "boundary and mechanical center strictly inside; k=R=1",
                total_accepted=sum(r['accepted'] for r in all_rows),
                total_attempted=sum(r['attempted'] for r in all_rows),
                tensor_spotcheck_max_error=max(r['normal_assembly_max_error'] for r in all_rows),
                worst_acceptance=min(all_rows, key=lambda r: r['acceptance_fraction']))


def figure6():
    fig = plt.figure(figsize=(7.1, 2.6))
    gs = fig.add_gridspec(1, 3, wspace=.50, left=.070, right=.99,
                         top=.82, bottom=.20, width_ratios=[.85, 1., 1.])
    axA = fig.add_subplot(gs[0, 0])
    for e, y in zip([0., .6, .9], [0, -2.6, -4.9]):
        v = ellipse_polygon(6, e)
        b = np.sqrt(1-e**2)
        axA.add_patch(Ellipse((0, y), 2., 2*b, fill=False, ls=(0, (3, 2)),
                              lw=.9, ec=GREY, zorder=1))
        axA.add_patch(MplPoly(v+[0, y], closed=True, fc=CELL_FILL,
                             ec=CELL_EDGE, lw=1.4, zorder=2))
        axA.plot(v[:, 0], v[:, 1]+y, 'o', ms=3., color=CELL_EDGE, zorder=3)
        axA.plot(0, y, '+', ms=6, mew=1.3, color=CELL_EDGE, zorder=4)
        pass  # in-panel numeric label removed
    axA.set_xlim(-2., 2.); axA.set_ylim(-6.4, 1.4)
    axA.set_aspect('equal'); axA.axis('off')
    panel_label(axA, 'A')
    es = np.linspace(0, .999, 400)
    axB, axC = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    rows, error = [], 0.
    for n, col in zip([4, 5, 6, 8], [PURPLE, '#2e8b57', BLUE, '#d9822b']):
        vertices = np.array([ellipse_polygon(n, e) for e in es])
        metrics = polygon_metrics(vertices)
        raw, nor = metrics['delta'], metrics['anisotropy']
        for e, v, tensor, delta, ratio in zip(es, vertices, metrics['tensor'], raw, nor):
            error = max(error, float(np.max(np.abs(tensor-independent_tensor(v)))))
            rows.append(dict(n=n, eccentricity=float(e), gap=float(delta),
                             trace=float(np.trace(tensor)), normalized_gap=float(ratio)))
        axB.plot(es, raw, '-', lw=1.5, color=col, label=r"$n=%d$" % n)
        axC.plot(es, nor, '-', lw=1.5, color=col)
        if n == 6:
            i = int(np.argmax(raw))
            peak = dict(eccentricity=float(es[i]), gap=float(raw[i]))
    axB.set_xlabel(r"eccentricity  $e$"); axB.set_ylabel(r"$\Delta\lambda$")
    axB.set_xlim(0, 1); axB.set_ylim(bottom=0)
    axB.legend(loc='upper left', borderpad=.25, ncol=1, fontsize=10, frameon=False)
    panel_label(axB, 'B')
    pass  # panel subtitle removed
    axC.set_xlabel(r"eccentricity  $e$")
    axC.set_ylabel(r"$\Delta\lambda\,/\,\mathrm{tr}\,\mathbf{E}_s$")
    axC.set_xlim(0, 1); axC.set_ylim(0, 1.05)
    panel_label(axC, 'C')
    pass  # panel subtitle removed
    assert error < 1e-12
    save_figure(fig, "figure6")
    save_csv("legacy_figure6_ellipse.csv", rows)
    return dict(verified_points=len(rows), tensor_normal_assembly_max_error=error,
                sampled_hexagon_peak=peak,
                pentagon_normalized_at_e_0999999=float(polygon_metrics(
                    ellipse_polygon(5, .999999))['anisotropy']))


def circumcenter(a, b, c):
    """Original two-equation circumcenter construction for noncollinear triples."""
    matrix = 2*np.array([b-a, c-a])
    rhs = np.array([b@b-a@a, c@c-a@a])
    return np.linalg.solve(matrix, rhs)


def central_cell(center, ring):
    vertices = np.array([circumcenter(center, ring[i], ring[(i+1) % len(ring)])
                         for i in range(len(ring))])
    # Each candidate vertex must actually lie in the central Voronoi cell:
    # its distance to the center cannot exceed its distance to any neighbor.
    central_distance2 = np.sum((vertices-center)**2, axis=-1)
    other_distance2 = np.sum((vertices[:, None, :]-ring[None, :, :])**2, axis=-1)
    if np.any(other_distance2 < central_distance2[:, None]-1e-12):
        raise ValueError("Neighbor topology changed; consecutive circumcenters are invalid")
    return vertices


def figure7():
    ring = np.column_stack((np.cos(TH6), np.sin(TH6)))
    fig = plt.figure(figsize=(11.4, 3.35))
    gs = fig.add_gridspec(1, 3, wspace=.38, left=.055, right=.99,
                         top=.86, bottom=.20)
    axA = fig.add_subplot(gs[0, 0])
    cell = central_cell(np.zeros(2), ring)
    for i in range(6):
        tri = np.array([np.zeros(2), ring[i], ring[(i+1) % 6]])
        axA.add_patch(MplPoly(tri, closed=True, fc='none', ec=GREEN_D,
                             lw=.9, ls=(0, (3, 2)), zorder=1))
    axA.add_patch(MplPoly(cell, closed=True, fc=CELL_FILL, ec=CELL_EDGE, lw=1.6, zorder=2))
    for v in cell:
        axA.plot([v[0], 1.85*v[0]], [v[1], 1.85*v[1]], '-', color=CELL_EDGE, lw=1.6, zorder=2)
    axA.plot(ring[:, 0], ring[:, 1], 'o', ms=5., color=GREEN_D, zorder=4)
    axA.plot([0], [0], 'o', ms=5., color=GREEN_D, zorder=4)
    axA.plot(cell[:, 0], cell[:, 1], 'o', ms=4.4, color=ORANGE, zorder=5)
    axA.text(.10, .10, r"$\mathbf{q}_\alpha$", fontsize=11, color=GREEN_D)
    axA.text(1.05, .08, r"$\mathbf{q}_\beta$", fontsize=11, color=GREEN_D)
    axA.set_xlim(-1.55, 1.75); axA.set_ylim(-1.55, 1.55)
    axA.set_aspect('equal'); axA.axis('off')
    axA.legend(handles=[Line2D([], [], marker='o', ls='none', ms=5, color=GREEN_D,
                              label='mechanical center'),
                        Line2D([], [], marker='o', ls='none', ms=4.4, color=ORANGE,
                              label='tissue vertex')],
               loc='lower center', bbox_to_anchor=(.5, -.13), fontsize=9, handletextpad=.5, frameon=False)
    panel_label(axA, 'A')
    pass  # panel subtitle removed
    axB = fig.add_subplot(gs[0, 1])
    ang = np.linspace(0, 2*np.pi, 721)
    direction_rows, amplitude_rows = [], []
    error = 0.
    for eps, col in zip([.03, .05, .08], [PURPLE, BLUE, ORANGE]):
        vals = []
        for a in ang:
            center = eps*np.array([np.cos(a), np.sin(a)])
            v = central_cell(center, ring)
            tensor = polygon_tensor(v, center)
            assert polygon_validity(v, center)['valid']
            error = max(error, float(np.max(np.abs(tensor-independent_tensor(v, center)))))
            delta = float(gap(tensor))
            vals.append(delta)
            direction_rows.append(dict(epsilon=eps, psi_radians=float(a), gap=delta,
                                       quadratic=eps**2/np.sqrt(3),
                                       ratio_to_quadratic=delta/(eps**2/np.sqrt(3))))
        axB.plot(np.degrees(ang), np.array(vals)/(eps**2/np.sqrt(3)), '-',
                 lw=1.4, color=col, label=r"$\varepsilon=%.2f$" % eps)
    axB.set_xlabel(r"direction of displacement  $\psi$  (deg)")
    axB.set_ylabel(r"$\Delta\lambda\,/\,(\varepsilon^{2}/\sqrt{3})$")
    axB.set_xlim(0, 360); axB.set_xticks([0, 60, 120, 180, 240, 300, 360])
    axB.set_ylim(.9997, 1.0062)
    for a in [30, 90, 150, 210, 270, 330]:
        axB.axvline(a, color=GREY, lw=.6, ls=':', zorder=0)
    axB.legend(loc='upper center', borderpad=.25, ncol=3, columnspacing=.9,
               handlelength=1.2, fontsize=9, frameon=False)
    panel_label(axB, 'B')
    pass  # panel subtitle removed
    axC = fig.add_subplot(gs[0, 2])
    eps_v = np.linspace(0, .15, 61)
    for psi, col, lw, label in [(np.pi/6, ORANGE, 2.6, r"$\psi=30^\circ$ (toward vertex)"),
                               (0., BLUE, 1.3, r"$\psi=0^\circ$ (toward neighbor)")]:
        vals = []
        for eps in eps_v:
            center = eps*np.array([np.cos(psi), np.sin(psi)])
            v = central_cell(center, ring)
            tensor = polygon_tensor(v, center)
            assert polygon_validity(v, center)['valid']
            error = max(error, float(np.max(np.abs(tensor-independent_tensor(v, center)))))
            delta = float(gap(tensor))
            vals.append(delta)
            amplitude_rows.append(dict(epsilon=float(eps), psi_radians=psi, gap=delta,
                                       quadratic=eps**2/np.sqrt(3),
                                       relative_excess=delta/(eps**2/np.sqrt(3))-1 if eps else 0.))
        axC.plot(eps_v, vals, '-', lw=lw, color=col, label=label)
    axC.plot(eps_v, eps_v**2/np.sqrt(3), ls=(0, (4, 2.5)), lw=1.2,
             color='#111111', label=r"leading term $\varepsilon^{2}/\sqrt{3}$")
    axC.set_xlabel(r"displacement magnitude  $\varepsilon$")
    axC.set_ylabel(r"$\Delta\lambda$")
    axC.set_xlim(0, .15); axC.set_ylim(bottom=0)
    axC.legend(loc='upper left', borderpad=.25, fontsize=10, handlelength=1.6, frameon=False)
    maximum = max(r['relative_excess'] for r in amplitude_rows)
    pass  # in-panel numeric note removed
    panel_label(axC, 'C')
    assert error < 1e-12
    save_figure(fig, "figure7")
    save_csv("legacy_figure7_direction.csv", direction_rows)
    save_csv("legacy_figure7_amplitude.csv", amplitude_rows)
    return dict(direction_points=len(direction_rows), amplitude_points=len(amplitude_rows),
                tensor_normal_assembly_max_error=error, maximum_relative_excess=maximum,
                direction_modulation=[dict(epsilon=eps, maximum_excess_percent=100*max(
                    r['ratio_to_quadratic']-1 for r in direction_rows if r['epsilon']==eps))
                                      for eps in [.03, .05, .08]])


def main():
    global FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', default='reproduced_figures',
                        help='Diagnostic layouts; publication assets are preserved by default.')
    args = parser.parse_args()
    FIGURES = ROOT / args.output_directory
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    summary = dict(base_seed=BASE_SEED, numpy_version=np.__version__,
                   source="WilliamResearchCode_annotated.ipynb, cells 19/21/23/25",
                   historical_notebook_modified=False,
                   figure5=figure5(), figure6=figure6(), figure7=figure7())
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (DATA/'legacy_figures_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
