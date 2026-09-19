#!/usr/bin/env python3
"""Check the boundary identity and compare two mechanically distinct squeezes.

All k=1. Whole-cell windows use the half-interface stress of ordinary Voronoi
cells. Affine physical loading transforms BOTH vertices and force vectors,
uses determinant-one F, and is not a newly recomputed ordinary Voronoi diagram.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.spatial import Voronoi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from polygon import polygon_metrics
from generate_tissue_windows import ccw, save_csv
from run_tissue import window_ids

ROOT = Path(__file__).resolve().parents[1]


def patch_tensor(vor, q, selected):
    E, area, gaps, cells = np.zeros((2, 2)), 0., [], []
    for i in selected:
        ids = vor.regions[vor.point_region[i]]
        if not ids or -1 in ids:
            raise RuntimeError('Unbounded measurement cell')
        v = ccw(vor.vertices[ids])
        m = polygon_metrics(v, q[i])
        E += m['tensor']
        area += float(m['area'])
        gaps.append(float(m['delta']))
        cells.append(v)
    return E, area, np.asarray(gaps), cells


def boundary_tensor(vor, q, selected):
    """Independent exterior-ridge assembly; no polygon area/tensor routine."""
    inside = set(np.asarray(selected).tolist())
    B = np.zeros((2, 2))
    perimeter, l1, count = 0., 0., 0
    for (a, b), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if (a in inside) == (b in inside):
            continue
        if a not in inside:
            a, b = b, a
        if -1 in ridge or len(ridge) != 2:
            raise RuntimeError('Unbounded measurement boundary')
        ends = vor.vertices[ridge]
        length = np.linalg.norm(ends[1] - ends[0])
        normal = (q[b] - q[a]) / np.linalg.norm(q[b] - q[a])
        tangent = np.array([-normal[1], normal[0]])
        h = (ends.mean(axis=0) - q[a]) @ tangent
        B += length * h * (np.outer(tangent, normal) + np.outer(normal, tangent)) / 2
        perimeter += length
        l1 += length * abs(h)
        count += 1
    return B, perimeter, l1, count


def gap(E):
    return float(np.hypot(E[0, 0] - E[1, 1], 2 * E[0, 1]))


def affine_physical_tensor(cells, q_centers, F):
    """Assemble transformed half-interface forces from the baseline cells.

    Baseline assigned force per cell edge is d*t = half the full tension
    vector. It is transformed by F. Edge vectors also transform by F.
    No recomputation of generator-to-interface distances is used.
    """
    E = np.zeros((2, 2))
    area = 0.
    for v, center in zip(cells, q_centers):
        edge = np.roll(v, -1, axis=0) - v
        L = np.linalg.norm(edge, axis=1)
        t = edge / L[:, None]
        n = np.column_stack([t[:, 1], -t[:, 0]])
        d = np.einsum('ij,ij->i', v - center, n)
        transformed_edge = edge @ F.T
        transformed_half_force = (d[:, None] * t) @ F.T
        E += np.einsum('ni,nj->ij', transformed_half_force, transformed_edge)
        v_new = v @ F.T
        area += .5 * np.sum(v_new[:, 0] * np.roll(v_new[:, 1], -1)
                           - v_new[:, 1] * np.roll(v_new[:, 0], -1))
    return E, area


def main():
    metadata = json.loads((ROOT / 'data/tissue_metadata.json').read_text())
    with np.load(ROOT / 'data/tissue_cells.npz') as d:
        data = {k: d[k] for k in d.files}
    rng = np.random.default_rng(metadata['seed'])
    windows = [('rhombus', p) for p in [2, 4, 7, 10, 15, 20]]
    # Match the exact radial parameters recorded in the retained data.
    with (ROOT / 'data/tissue_realizations.csv').open() as f:
        rows = list(csv.DictReader(f))
    radial = sorted({float(r['parameter']) for r in rows if r['shape'] != 'rhombus'})
    windows += [('radial', p) for p in radial]
    records = []
    maxerr = maxbound = 0.
    for rep in range(metadata['realizations']):
        q = data['reference_generators'] + rng.normal(0, metadata['eta'], data['reference_generators'].shape)
        vor = Voronoi(q)
        for shape, param in windows:
            local = window_ids(data['lattice_indices'], shape, param)
            indices = data['selected_global_indices'][local]
            A = float(data['area'][rep, local].sum())
            W = data['W'][rep, local].sum()
            B, perimeter, l1, nedges = boundary_tensor(vor, q, indices)
            WB = B[0, 0] - B[1, 1] + 2j * B[0, 1]
            err = abs(W - WB)
            bound = l1 / (2 * A)
            actual = abs(W) / (2 * A)
            maxerr = max(maxerr, float(err))
            maxbound = max(maxbound, float(actual - bound))
            records.append(dict(realization=rep, shape=shape, parameter=param, cells=len(local),
                                area=A, perimeter=perimeter, boundary_edges=nedges,
                                W_real=float(W.real), W_imag=float(W.imag),
                                boundary_W_real=float(WB.real), boundary_W_imag=float(WB.imag),
                                identity_error=float(err), normalized_gap=float(actual),
                                deterministic_bound=float(bound)))
    save_csv(ROOT / 'data/boundary_identity.csv', records)
    if maxerr > 1e-10 or maxbound > 1e-12:
        raise AssertionError((maxerr, maxbound))

    # A regular 35x35 seed array, measured in the central 7x7 block.
    ij = np.array([(i, j) for i in range(-17, 18) for j in range(-17, 18)])
    ref = np.column_stack([2*ij[:, 0] + ij[:, 1], np.sqrt(3)*ij[:, 1]])
    core = np.flatnonzero(np.max(abs(ij), axis=1) <= 3)
    base_vor = Voronoi(ref)
    E0, A0, _, base_cells = patch_tensor(base_vor, ref, core)
    squeezes, examples = [], {}
    max_cell_error = max_affine_error = 0.
    for s in np.linspace(-.4, .4, 41):
        F = np.diag([np.exp(s), np.exp(-s)])
        q = ref @ F.T
        vor = Voronoi(q)
        E, A, gaps, cells = patch_tensor(vor, q, core)
        physical, physical_A = affine_physical_tensor(base_cells, ref[core], F)
        expected = F @ E0 @ F.T
        max_cell_error = max(max_cell_error, float(np.max(gaps)))
        max_affine_error = max(max_affine_error, float(np.max(abs(physical - expected))))
        squeezes.append(dict(s=float(s), cells=len(core), seed_squeeze_gap=gap(E)/A,
                             seed_squeeze_normalized=gap(E)/np.trace(E),
                             maximum_seed_cell_gap=float(max(gaps)),
                             physical_gap=gap(physical)/physical_A,
                             physical_normalized=gap(physical)/np.trace(physical),
                             exact_normalized=float(abs(np.tanh(2*s))),
                             tensor_mapping_error=float(np.max(abs(physical-expected)))))
        if np.isclose(s, .2):
            examples['seeds'] = cells
            examples['physical'] = [v @ F.T for v in base_cells]
    save_csv(ROOT / 'data/squeeze_controls.csv', squeezes)

    # Interior-generator changes cannot change regional stress while its
    # entire exterior interface geometry and adjacent generators stay fixed.
    measured = np.flatnonzero(np.max(abs(ij), axis=1) <= 9)
    perturbed = np.flatnonzero(np.max(abs(ij), axis=1) <= 4)
    localized = []
    for eta in [0., .05, .10, .20]:
        for rep in range(20):
            q = ref.copy()
            q[perturbed] += np.random.default_rng(np.random.SeedSequence([20260911, 19, rep])).normal(size=(len(perturbed), 2))*eta
            vor = Voronoi(q)
            E, A, gaps, cells = patch_tensor(vor, q, measured)
            B, perimeter, l1, count = boundary_tensor(vor, q, measured)
            localized.append(dict(eta=eta, realization=rep, cells=len(measured),
                                  perturbed_generators=len(perturbed), mean_local_gap=float(gaps.mean()),
                                  maximum_local_gap=float(gaps.max()), regional_gap=gap(E),
                                  normalized_gap=gap(E)/np.trace(E), boundary_norm=float(np.linalg.norm(B)),
                                  identity_error=float(np.max(abs(E-A*np.eye(2)-B)))))
    save_csv(ROOT / 'data/localized_generator_control.csv', localized)
    if max(r['regional_gap'] for r in localized) > 1e-9:
        raise AssertionError('Exterior boundary changed in localized control')

    plt.rcParams.update({'font.size': 12, 'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(2, 2, figsize=(9.3, 6.4), layout='constrained')
    for axis, key, label in zip(ax[0], ['seeds', 'physical'],
                               ['Squeeze generators; rebuild Voronoi', 'Deform interfaces and balanced forces']):
        collection = PolyCollection(examples[key], facecolors='#cfe3ee', edgecolors='#334e5c', linewidths=.6)
        axis.add_collection(collection)
        axis.autoscale_view()
        axis.set(aspect='equal', xlabel='$x$', ylabel='$y$')
    s = [r['s'] for r in squeezes]
    ax[1, 0].plot(s, [r['exact_normalized'] for r in squeezes], 'k-', label=r'Physical loading: $|\tanh(2s)|$')
    ax[1, 0].plot(s[::4], [r['physical_normalized'] for r in squeezes][::4], 'o', ms=4, color='#c5692c', label='Direct force-tensor sum')
    ax[1, 0].plot(s, [r['seed_squeeze_normalized'] for r in squeezes], color='#2679a3', label='Recomputed Voronoi')
    ax[1, 0].set(xlabel='Area-preserving squeeze $s$', ylabel='Regional normalized anisotropy')
    ax[1, 0].legend(fontsize=10, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=1, handletextpad=.5)
    subset = [r for r in records if r['cells'] in [400, 397]]
    for shape, color in [('rhombus', '#2679a3'), ('radial', '#c5692c')]:
        r = [x for x in subset if x['shape'] == shape]
        ax[1, 1].scatter([x['W_real'] for x in r], [x['boundary_W_real'] for x in r], s=8, alpha=.5, color=color, label=shape)
    lim = max(max(abs(r['W_real']) for r in subset), 1e-5)*1.1
    ax[1, 1].plot([-lim, lim], [-lim, lim], 'k--', lw=.8)
    ax[1, 1].set(xlabel=r'$\mathrm{Re}\sum W_\alpha$: all cells', ylabel=r'$\mathrm{Re}\,W_{\partial\Omega}$: boundary only')
    ax[1, 1].legend(fontsize=10, frameon=False)
    for axis, label in zip(ax.flat, 'ABCD'):
        axis.text(-.17, 1.12, label, transform=axis.transAxes, weight='bold', fontsize=16)
        axis.spines[['top', 'right']].set_visible(False)
    fig.savefig(ROOT / 'figures/regional_boundary.pdf')
    fig.savefig(ROOT / 'figures/regional_boundary.png', dpi=190)
    plt.close(fig)
    report = dict(boundary_windows=len(records), boundary_realizations=metadata['realizations'],
                  max_complex_boundary_identity_error=maxerr, maximum_bound_violation=maxbound,
                  maximum_squeezed_Bravais_cell_gap=max_cell_error, maximum_affine_tensor_error=max_affine_error,
                  squeeze_amplitudes=41, localized_controls=len(localized),
                  maximum_localized_regional_gap=max(r['regional_gap'] for r in localized),
                  maximum_localized_identity_error=max(r['identity_error'] for r in localized),
                  localized_eta_point2_mean_local_gap=float(np.mean([r['mean_local_gap'] for r in localized if r['eta']==.2])),
                  affine_force_convention='F f, determinant(F)=1; ordinary Voronoi is not recomputed',
                  provenance='Retained tissue cache regenerated from its recorded seed; boundary assembled independently from exterior ridges')
    (ROOT / 'data/boundary_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
