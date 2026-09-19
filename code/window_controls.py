"""Fixed-seed reference-label window controls (all stresses use k=1).

These helpers change computation reuse, not sampling. The coefficient-modulated
control is evaluated at fixed geometry and is not an equilibrium solution.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.spatial import Voronoi
from polygon import polygon_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
BASIS = np.array([[2., 0.], [1., np.sqrt(3.)]])


def ccw(v):
    c = v.mean(0)
    return v[np.argsort(np.arctan2(v[:, 1]-c[1], v[:, 0]-c[0]))]


def build(n, eta, seed, strain=0.):
    ij = np.array([(i, j) for i in range(n) for j in range(n)], float)
    ref = ij @ BASIS
    q = ref + np.random.default_rng(seed).normal(0, eta, ref.shape)
    if strain:
        q = q @ np.diag([1+strain, 1/(1+strain)]).T
    vor = Voronoi(q)
    W, A, edges = {}, {}, {}
    for index in range(len(q)):
        region = vor.regions[vor.point_region[index]]
        if not region or -1 in region:
            continue
        v = ccw(vor.vertices[region])
        metrics = polygon_metrics(v, q[index])
        W[index], A[index] = complex(metrics['W']), float(metrics['area'])
        x = v-q[index]
        nxt = np.roll(x, -1, axis=0)
        e = nxt-x
        tangent = e/np.linalg.norm(e, axis=1)[:, None]
        cross = x[:, 0]*nxt[:, 1]-x[:, 1]*nxt[:, 0]
        phi = np.arctan2(tangent[:, 1], tangent[:, 0])
        edges[index] = (cross, tangent, phi)
    return dict(n=n, ij=ij, q=q, vor=vor, W=W, area=A, edges=edges)


def polarized_W(tissue, beta):
    result = {}
    for index, (cross, tangent, phi) in tissue['edges'].items():
        coeff = 1+beta*np.cos(2*phi)
        E = np.einsum('n,ni,nj->ij', coeff*cross, tangent, tangent)
        result[index] = complex(E[0, 0]-E[1, 1]+2j*E[0, 1])
    return result


def select(tissue, kind, parameter):
    ij = tissue['ij']
    d = ij-(tissue['n']-1)/2
    if kind == 'label_box':
        mask = np.max(np.abs(d), axis=1) <= parameter
    elif kind == 'label_disk':
        mask = np.sum(d*d, axis=1) <= parameter**2
    elif kind == 'label_rotated_box':
        angle = np.radians(31.)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        mask = np.max(np.abs(d @ rotation.T), axis=1) <= parameter
    else:
        raise ValueError(kind)
    selected = np.flatnonzero(mask).tolist()
    if not all(i in tissue['W'] for i in selected):
        raise RuntimeError('An observation mask contains an unbounded cell')
    return selected


def neighbor_pairs(tissue, selected):
    lookup = {tuple(tissue['ij'][i].astype(int)): i for i in selected}
    pairs = []
    for i in selected:
        a, b = tissue['ij'][i].astype(int)
        for da, db in [(1, 0), (0, 1), (-1, 1)]:
            j = lookup.get((a+da, b+db))
            if j is not None:
                pairs.append((i, j))
    return np.asarray(pairs, dtype=int)


def measure(tissue, W, selected):
    w = np.array([W[i] for i in selected])
    pairs = neighbor_pairs(tissue, selected)
    left = np.array([W[i] for i in pairs[:, 0]])
    right = np.array([W[i] for i in pairs[:, 1]])
    mean = w.mean()
    a, b = left-mean, right-mean
    safe = lambda numerator, denominator: float(numerator/denominator) if denominator > 1.e-25 else float('nan')
    return dict(N=len(w), pairs=len(pairs), local=float(np.mean(abs(w))),
                regional=float(abs(w.sum())), sum_W_real=float(w.sum().real),
                sum_W_imag=float(w.sum().imag), area=float(sum(tissue['area'][i] for i in selected)),
                null=float(np.sqrt(np.sum(abs(w)**2))),
                corr=safe(np.real(left*np.conj(right)).sum(), np.sum(abs(left)*abs(right))),
                corr_centered=safe(np.real(a*np.conj(b)).sum(), np.sum(abs(a)*abs(b))),
                corr_centered_rms=safe(np.real(a*np.conj(b)).sum(), np.sqrt(np.sum(abs(a)**2)*np.sum(abs(b)**2))))


def boundary_stats(tissue, selected):
    selected = set(selected)
    perimeter, boundary_cells = 0., set()
    for (a, b), ridge in zip(tissue['vor'].ridge_points, tissue['vor'].ridge_vertices):
        if (a in selected) == (b in selected):
            continue
        if -1 in ridge or len(ridge) != 2:
            raise RuntimeError('Unbounded observation boundary')
        perimeter += np.linalg.norm(np.diff(tissue['vor'].vertices[ridge], axis=0)[0])
        boundary_cells.add(a if a in selected else b)
    return float(perimeter), len(boundary_cells)


def junction_residual(tissue, beta, selected):
    vor = tissue['vor']
    vertices = set(v for i in selected for v in vor.regions[vor.point_region[i]])
    force = {v: np.zeros(2) for v in vertices}
    scale = {v: 0. for v in vertices}
    for (a, b), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge or len(ridge) != 2:
            continue
        u, v = ridge
        edge = vor.vertices[v]-vor.vertices[u]
        tangent = edge/np.linalg.norm(edge)
        phi = np.arctan2(tangent[1], tangent[0])
        T = np.linalg.norm(tissue['q'][a]-tissue['q'][b])*(1+beta*np.cos(2*phi))
        for j, sign in ((u, 1), (v, -1)):
            if j in force:
                force[j] += sign*T*tangent
                scale[j] += T
    return max(float(np.linalg.norm(force[v])/scale[v]) for v in vertices)


def save_csv(name, rows):
    DATA.mkdir(exist_ok=True)
    with (DATA/name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(name, data):
    DATA.mkdir(exist_ok=True)
    (DATA/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def summarize(rows, parameter_name, value):
    r = np.sqrt(np.mean([x['regional']**2 for x in rows]))
    return dict(**{parameter_name: value}, N=rows[0]['N'], realizations=len(rows),
                local=float(np.mean([x['local'] for x in rows])), regional=float(r),
                per_cell=float(r/rows[0]['N']),
                ratio=float(r/np.mean([x['null'] for x in rows])),
                corr=float(np.mean([x['corr'] for x in rows])),
                corr_centered=float(np.mean([x['corr_centered'] for x in rows])),
                corr_centered_rms=float(np.mean([x['corr_centered_rms'] for x in rows])))


def fit_rows(rows, parameter_name, value):
    slope, intercept = np.polyfit(np.log([x['N'] for x in rows]), np.log([x['rms'] for x in rows]), 1)
    return dict(**{parameter_name: value}, N_min=min(x['N'] for x in rows),
                N_max=max(x['N'] for x in rows), fitted_slope=float(slope),
                fitted_intercept=float(intercept), interpretation='finite-range empirical log-log fit; no asymptotic law inferred')
