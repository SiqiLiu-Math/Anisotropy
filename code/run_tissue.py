#!/usr/bin/env python3
"""Verify tissue data and rebuild the original Figure 8 A--E storyline.

Default: reuse the supplied, independently checked 200-realization window data.
--regenerate: regenerate those data before checks and the additional disorder scan.
The historical Figure 8 source was unavailable; no historical values are inferred.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
from scipy.spatial import Voronoi

from polygon import polygon_metrics
from generate_tissue_windows import ccw, save_csv

ROOT = Path(__file__).resolve().parents[1]


def window_ids(ij, shape, parameter):
    if shape == 'rhombus':
        m = int(parameter)
        low = -(m // 2)
        mask = np.all((ij >= low) & (ij < low + m), axis=1)
    else:
        ref = np.column_stack([2*ij[:, 0] + ij[:, 1], np.sqrt(3)*ij[:, 1]])
        mask = np.sum(ref**2, axis=1) < parameter**2
    return np.flatnonzero(mask)


def independent_tensor(vertices, center):
    edge = np.roll(vertices, -1, axis=0) - vertices
    length = np.linalg.norm(edge, axis=1)
    tangent = edge / length[:, None]
    outward_normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    distance = np.einsum('ij,ij->i', vertices-center, outward_normal)
    return np.einsum('i,ij,ik->jk', length*distance, tangent, tangent)


def verify_window_data(data, metadata):
    # NpzFile decompresses an array on every indexing operation. Materialize
    # once before the per-cell verification loop.
    if hasattr(data, 'files'):
        data = {key: data[key] for key in data.files}
    ref = data['reference_generators']
    selected = data['selected_global_indices']
    rng = np.random.default_rng(metadata['seed'])
    errors = dict(generator=0., W=0., area=0., tensor_assembly=0.,
                  junction_force=0., half_edge_allocation=0., raw_window=0.)
    checked_junctions = 0
    for realization in range(data['W'].shape[0]):
        q = ref + rng.normal(0, metadata['eta'], ref.shape)
        vor = Voronoi(q)
        errors['generator'] = max(errors['generator'], float(np.max(np.abs(q[selected]-data['generator_positions'][realization]))))
        measurement_vertices = set()
        for local, index in enumerate(selected):
            ids = vor.regions[vor.point_region[index]]
            assert ids and -1 not in ids
            measurement_vertices.update(ids)
            v = ccw(vor.vertices[ids])
            m = polygon_metrics(v, q[index])
            independent = independent_tensor(v, q[index])
            errors['tensor_assembly'] = max(errors['tensor_assembly'], float(np.max(np.abs(independent-m['tensor']))))
            errors['W'] = max(errors['W'], float(abs(m['W']-data['W'][realization, local])))
            errors['area'] = max(errors['area'], float(abs(m['area']-data['area'][realization, local])))
        force = {v: np.zeros(2) for v in measurement_vertices}
        for (a, b), ridge in zip(vor.ridge_points, vor.ridge_vertices):
            if -1 in ridge or len(ridge) != 2:
                continue
            u, v = ridge
            if u not in measurement_vertices and v not in measurement_vertices:
                continue
            edge = vor.vertices[v]-vor.vertices[u]
            tension = np.linalg.norm(q[b]-q[a])
            edge_force = tension*edge/np.linalg.norm(edge)
            if u in force:
                force[u] += edge_force
            if v in force:
                force[v] -= edge_force
            n = (q[b]-q[a])/tension
            da = np.dot(vor.vertices[u]-q[a], n)
            db = np.dot(vor.vertices[u]-q[b], -n)
            errors['half_edge_allocation'] = max(errors['half_edge_allocation'], abs(da-tension/2), abs(db-tension/2))
        errors['junction_force'] = max(errors['junction_force'], max(float(np.linalg.norm(f)) for f in force.values()))
        checked_junctions += len(force)
    with (ROOT/'data/tissue_realizations.csv').open() as stream:
        raw = list(csv.DictReader(stream))
    for row in raw:
        r = int(row['realization'])
        ids = window_ids(data['lattice_indices'], row['shape'], float(row['parameter']))
        w = data['W'][r, ids]
        area = np.sum(data['area'][r, ids])
        quantities = dict(sum_W_real=np.sum(w).real, sum_W_imag=np.sum(w).imag,
                          total_area=area, delta_total=abs(np.sum(w)),
                          area_normalized_delta=abs(np.sum(w))/area,
                          orientation_null_squared=np.sum(abs(w)**2),
                          area_normalized_null_squared=np.sum(abs(w)**2)/area**2)
        for key, value in quantities.items():
            errors['raw_window'] = max(errors['raw_window'], abs(float(row[key])-value))
    if max(errors.values()) > 1.e-10:
        raise AssertionError(errors)
    return dict(maximum_absolute_errors=errors, regenerated_realizations=int(data['W'].shape[0]),
                checked_junctions=checked_junctions, checked_window_records=len(raw),
                cache_sha256=hashlib.sha256((ROOT/'data/tissue_cells.npz').read_bytes()).hexdigest())


def disorder_scan():
    """Actual Voronoi realizations: do not extrapolate from the cached eta=.05 run."""
    ij = np.array([(i, j) for i in range(-16, 17) for j in range(-16, 17)])
    ref = np.column_stack([2*ij[:, 0]+ij[:, 1], np.sqrt(3)*ij[:, 1]])
    core = np.flatnonzero(np.all((ij >= -10) & (ij < 10), axis=1))
    core_set = set(core.tolist())
    outer_set = set(np.flatnonzero(np.max(abs(ij), axis=1) == 16).tolist())
    examples, records, summary = {}, [], []
    minimum_area, max_trace_error = np.inf, 0.
    for eta in [0., .02, .05, .10, .15]:
        # Paired disorder fields make amplitude comparisons more efficient;
        # different tissues remain independent, and no cross-amplitude fit is used.
        rng = np.random.default_rng(202609102)
        all_delta = []
        for r in range(40):
            q = ref + eta*rng.normal(size=ref.shape)
            vor = Voronoi(q)
            if any((a in core_set and b in outer_set) or
                   (b in core_set and a in outer_set) for a, b in vor.ridge_points):
                raise RuntimeError('Disorder-scan cell reached the outer generator ring')
            gaps, cells, Ws = [], [], []
            for local, index in enumerate(core):
                ids = vor.regions[vor.point_region[index]]
                assert ids and -1 not in ids
                v = ccw(vor.vertices[ids])
                values = polygon_metrics(v, q[index])
                gap = float(values['delta'])
                gaps.append(gap)
                Ws.append(values['W'])
                minimum_area = min(minimum_area, float(values['area']))
                max_trace_error = max(max_trace_error, abs(float(values['trace']-2*values['area'])))
                records.append(dict(eta=eta, realization=r, i=int(ij[index, 0]),
                                    j=int(ij[index, 1]), delta=gap,
                                    W_real=float(values['W'].real), W_imag=float(values['W'].imag),
                                    area=float(values['area'])))
                if r == 0:
                    cells.append(v)
            if r == 0 and eta in [.02, .10]:
                examples[eta] = (q[core], ij[core], cells, np.asarray(Ws))
            all_delta.append(gaps)
        values = np.asarray(all_delta)
        replicate_means = values.mean(axis=1)
        boot = np.random.default_rng(202609103).integers(0, 40, (2000, 40))
        boot_mean = replicate_means[boot].mean(axis=1)
        summary.append(dict(eta=eta, mean=float(values.mean()),
                            cell_sd=float(values.std(ddof=0)),
                            mean_ci_low=float(np.quantile(boot_mean, .025)),
                            mean_ci_high=float(np.quantile(boot_mean, .975)),
                            realizations=40, cells_per_realization=400))
        print(f'Figure 8 disorder scan: eta={eta:.2f}, mean={values.mean():.6g}', flush=True)
    save_csv(ROOT/'data/tissue_disorder_cells.csv', records)
    save_csv(ROOT/'data/tissue_disorder_summary.csv', summary)
    return summary, examples, dict(seed=202609102, realizations=40, cells_per_realization=400,
                                  generator_count=len(ref), minimum_cell_area=minimum_area,
                                  maximum_trace_error=max_trace_error,
                                  outer_ring_neighbors_of_measurement_cells=0,
                                  amplitudes=[r['eta'] for r in summary],
                                  amplitudes_use_paired_disorder=True)


def shell_correlations(data):
    ij = data['lattice_indices']
    ids = window_ids(ij, 'rhombus', 20)
    lookup = {tuple(ij[index]): index for index in ids}
    # Reference-lattice separations: distance = 2 sqrt(di^2+di*dj+dj^2).
    shells = {}
    for di in range(-6, 7):
        for dj in range(-6, 7):
            s = di*di+di*dj+dj*dj
            if not (0 < s <= 16 and (di > 0 or (di == 0 and dj > 0))):
                continue
            for a in ids:
                b = lookup.get(tuple(ij[a]+[di, dj]))
                if b is not None:
                    shells.setdefault(s, []).append((a, b))
    R = data['W'].shape[0]
    boot = np.random.default_rng(202609093).integers(0, R, (2000, R))
    records = []
    for s, pairs in sorted(shells.items()):
        pairs = np.asarray(pairs)
        a, b = data['W'][:, pairs[:, 0]], data['W'][:, pairs[:, 1]]
        numerator = np.mean(np.real(a*np.conj(b)), axis=1)
        qa, qb = np.mean(abs(a)**2, axis=1), np.mean(abs(b)**2, axis=1)
        value = numerator.mean()/np.sqrt(qa.mean()*qb.mean())
        rep = numerator[boot].mean(axis=1)/np.sqrt(qa[boot].mean(axis=1)*qb[boot].mean(axis=1))
        records.append(dict(shell=s, reference_separation_in_spacings=np.sqrt(s),
                            correlation=float(value), ci_low=float(np.quantile(rep, .025)),
                            ci_high=float(np.quantile(rep, .975)), pairs_per_realization=len(pairs)))
    save_csv(ROOT/'data/tissue_correlations.csv', records)
    return records


def plot_figures(scan, examples, correlations, aggregate):
    plt.rcParams.update({'font.size':12, 'axes.labelsize':12, 'pdf.fonttype':42,
                         'xtick.labelsize':11, 'ytick.labelsize':11,
                         'ps.fonttype':42, 'savefig.bbox':'tight'})
    # Split into two figures: 8 = cell maps + amplitude scan; 8b = correlations + scaling
    # Local-anisotropy figure: three panels in one float
    figloc = plt.figure(figsize=(13.2, 4.2), layout='constrained')
    gloc = figloc.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15])
    fig2 = plt.figure(figsize=(10.4, 4.2), layout='constrained')
    grid2 = fig2.add_gridspec(1, 2)
    axes = [figloc.add_subplot(gloc[0, 0]), figloc.add_subplot(gloc[0, 1]),
            figloc.add_subplot(gloc[0, 2]),
            fig2.add_subplot(grid2[0, 0]), fig2.add_subplot(grid2[0, 1])]
    color_max = .40
    for axis, eta in zip(axes[:2], [.02, .10]):
        q, ij, cells, W = examples[eta]
        ids = np.flatnonzero(np.all((ij >= -4) & (ij < 4), axis=1))
        collection = PolyCollection([cells[i] for i in ids], array=abs(W[ids]),
                                    cmap='viridis', edgecolors='white', linewidths=.4,
                                    clim=(0, color_max))
        axis.add_collection(collection)
        sticks = []
        for i in ids:
            theta = .5*np.angle(W[i])
            half = 1.5*abs(W[i])*np.array([np.cos(theta), np.sin(theta)])
            sticks.append([q[i]-half, q[i]+half])
        axis.add_collection(LineCollection(sticks, colors='white', linewidths=.75))
        axis.autoscale_view()
        axis.set(aspect='equal')
        axis.set_axis_off()
    figloc.colorbar(collection, ax=axes[1], fraction=.046, pad=.02,
                  label=r'Cell $\Delta\lambda$ ($k=1$)', extend='max')
    eta = np.array([r['eta'] for r in scan])
    mean, sd = np.array([r['mean'] for r in scan]), np.array([r['cell_sd'] for r in scan])
    axes[2].fill_between(eta, np.maximum(0, mean-sd), mean+sd, color='#cddfee', label='Cell distribution: mean ± SD')
    axes[2].plot(eta, mean, 'o-', color='#226e9e', label='Mean')
    axes[2].set(xlabel=r'Generator disorder $\eta$', ylabel=r'Mean cell $\Delta\lambda$')
    axes[2].legend(fontsize=10, loc='upper left', frameon=False)
    x = np.array([r['reference_separation_in_spacings'] for r in correlations])
    c = np.array([r['correlation'] for r in correlations])
    low, high = np.array([r['ci_low'] for r in correlations]), np.array([r['ci_high'] for r in correlations])
    axes[3].errorbar(x, c, yerr=np.array([c-low, high-c]), fmt='o-', ms=4, capsize=3, color='#226e9e')
    axes[3].axhline(0, color='.5', lw=.7)
    axes[3].set(xlabel='Reference separation / lattice spacing', ylabel=r'Deviator correlation $C$', xlim=(.85, 4.1))
    for shape, color in [('rhombus', '#ce6a25'), ('disk', '#1871b0')]:
        rows = [r for r in aggregate if r['shape']==shape]
        n = np.array([int(r['n']) for r in rows])
        mean = np.array([float(r['integrated_rms']) for r in rows])
        low = np.array([float(r['integrated_rms_ci_low']) for r in rows])
        high = np.array([float(r['integrated_rms_ci_high']) for r in rows])
        axes[4].errorbar(n, mean, yerr=np.array([mean-low, high-mean]), fmt='o-', ms=4, capsize=3, color=color, label=shape.capitalize())
        axes[4].plot(n, [float(r['integrated_null_rms']) for r in rows], '--', color=color,
                     label=shape.capitalize()+' orientation reference')
    axes[4].set(xscale='log', yscale='log', xlabel='Cells in window, $N$',
                ylabel=r'RMS integrated gap $|\sum_\alpha W_\alpha|$')
    axes[4].legend(fontsize=10, ncol=1, loc='upper left', frameon=False)
    for label, axis in zip(['A', 'B', 'C', 'A', 'B'], axes):
        if label:
            axis.text(-.14, 1.10, label, transform=axis.transAxes,
                      fontweight='bold', fontsize=16)
        axis.spines[['top', 'right']].set_visible(False)
    figloc.savefig(ROOT/'figures/tissue_local.pdf')
    figloc.savefig(ROOT/'figures/tissue_local.png', dpi=250)
    fig2.savefig(ROOT/'figures/figure8b.pdf')
    fig2.savefig(ROOT/'figures/figure8b.png', dpi=250)
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), layout='constrained')
    for shape, color in [('rhombus', '#ce6a25'), ('disk', '#1871b0')]:
        rows = [r for r in aggregate if r['shape']==shape]
        n = np.array([int(r['n']) for r in rows])
        for ax, key in zip(axes, ['area_normalized_rms', 'integrated_ratio']):
            values = np.array([float(r[key]) for r in rows])
            lo = np.array([float(r[key+'_ci_low']) for r in rows])
            hi = np.array([float(r[key+'_ci_high']) for r in rows])
            ax.errorbar(n, values, yerr=np.array([values-lo, hi-values]), fmt='o-', ms=4, capsize=3, color=color, label=shape.capitalize())
        axes[0].plot(n, [float(r['area_normalized_null_rms']) for r in rows], '--', color=color)
    axes[0].set(xscale='log', yscale='log', xlabel='Cells in window, $N$',
                ylabel=r'RMS area-averaged gap $|\sum W|/\sum A$')
    axes[1].set(xscale='log', xlabel='Cells in window, $N$', ylabel='Integrated RMS / orientation-reference RMS')
    axes[1].axhline(1., ls=':', color='.5')
    for label, ax in zip('AB', axes):
        ax.text(-.10, 1.04, label, transform=ax.transAxes, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(ROOT/'figures/tissue_coarse_graining_SI.pdf')
    fig.savefig(ROOT/'figures/tissue_coarse_graining_SI.png', dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate', action='store_true')
    args = parser.parse_args()
    for name in ['data', 'figures', 'audit']:
        (ROOT/name).mkdir(exist_ok=True)
    if args.regenerate or not (ROOT/'data/tissue_cells.npz').exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name('generate_tissue_windows.py'))], check=True)
    with np.load(ROOT/'data/tissue_cells.npz') as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads((ROOT/'data/tissue_metadata.json').read_text())
    verification = verify_window_data(data, metadata)
    scan, examples, scan_metadata = disorder_scan()
    correlations = shell_correlations(data)
    with (ROOT/'data/tissue_summary.csv').open() as stream:
        aggregate = list(csv.DictReader(stream))
    plot_figures(scan, examples, correlations, aggregate)
    verification.update(disorder_scan=scan_metadata,
                        correlation_definition='Shells use unperturbed lattice labels; actual distance bins are not used. C=<Re(Wa conj(Wb))>/sqrt(<|Wa|^2><|Wb|^2>), 2000 bootstrap resamples of entire tissues.',
                        source_note='Window data generated in the previous revision; all 200 full realizations reproduced and all 2400 region records checked. Disorder scan and shell correlations newly computed for this figure.')
    (ROOT/'data/tissue_verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
