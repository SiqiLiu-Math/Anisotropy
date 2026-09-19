#!/usr/bin/env python3
"""Independently verify the delivered scientific data, without changing figures.

Run: python code/verify_package.py
Requires numpy, scipy, matplotlib (the latter is imported by experiment modules).
Reference tensors use outward supporting normals and midpoint distances, not the
cross-product expression in the production core. New regional mechanics are
checked separately by verify_regional.py. The removed uncertainty application
is not part of this revision.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi

from polygon import polygon_tensor
from verify_core import run_checks, audit_saved_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def read_csv(name):
    with (DATA/name).open() as stream:
        return list(csv.DictReader(stream))


def normal_tensor(vertices, center=(0., 0.)):
    """Batch supporting-normal construction, independently of signed cross sums."""
    v = np.asarray(vertices, dtype=float)
    c = np.asarray(center, dtype=float)
    E = np.zeros(v.shape[:-2] + (2, 2))
    for j in range(v.shape[-2]):
        a, b = v[..., j, :], v[..., (j+1) % v.shape[-2], :]
        edge = b-a
        length = np.sqrt(np.sum(edge*edge, axis=-1))
        tangent = edge/length[..., None]
        normal = np.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)
        d = np.sum(((a+b)/2-c)*normal, axis=-1)
        E += (length*d)[..., None, None]*tangent[..., :, None]*tangent[..., None, :]
    return E


def W_of(vertices, center=(0., 0.)):
    E = normal_tensor(vertices, center)
    return np.stack((E[..., 0, 0]-E[..., 1, 1], 2*E[..., 0, 1]), axis=-1)


def valid_independently(vertices, center=(0., 0.)):
    """Strict convexity and center inclusion via supporting-line distances."""
    v = np.asarray(vertices)-np.asarray(center)[..., None, :]
    scale = np.maximum(np.max(np.sum(v*v, axis=-1), axis=-1), 1.)
    valid = np.all(np.isfinite(v), axis=(-2, -1))
    n = v.shape[-2]
    for i in range(n):
        a, b = v[..., i, :], v[..., (i+1) % n, :]
        edge = b-a
        L = np.sqrt(np.sum(edge*edge, axis=-1))
        inward = np.stack((-edge[..., 1], edge[..., 0]), axis=-1)/L[..., None]
        threshold = 1e-12*scale/L
        valid &= L*L > 1e-12*scale
        valid &= np.sum(-a*inward, axis=-1) > threshold
        for j in range(n):
            if j not in (i, (i+1) % n):
                valid &= np.sum((v[..., j, :]-a)*inward, axis=-1) > threshold
    return valid


def wilson_interval(count, size):
    # Independently written Wilson score inversion at the usual 95% quantile.
    z = 1.959963984540054
    denominator = size + z*z
    half = z*np.sqrt(count*(size-count)/size + z*z/4)/denominator
    middle = (count + z*z/2)/denominator
    return middle-half, middle+half


def original_figure_checks():
    errors = []
    ellipse = read_csv('legacy_figure6_ellipse.csv')
    for row in ellipse:
        n, e = int(row['n']), float(row['eccentricity'])
        theta = np.arange(n)*2*np.pi/n
        v = np.column_stack((np.cos(theta), np.sqrt(1-e*e)*np.sin(theta)))
        E = normal_tensor(v)
        gap = np.linalg.norm(W_of(v)); trace = np.trace(E)
        errors += [abs(gap-float(row['gap'])), abs(trace-float(row['trace'])),
                   abs(gap/trace-float(row['normalized_gap']))]
    assert max(errors) < 2e-13
    ellipse_error = max(errors)

    # Reconstruct the central Voronoi region with Qhull, rather than the original
    # consecutive-triplet circumcenter routine. Its complete topology is checked.
    rows = read_csv('legacy_figure7_direction.csv') + read_csv('legacy_figure7_amplitude.csv')
    theta = np.arange(6)*np.pi/3
    neighbors = np.column_stack((np.cos(theta), np.sin(theta)))
    errors = []
    for row in rows:
        eps, psi = float(row['epsilon']), float(row['psi_radians'])
        c = eps*np.array([np.cos(psi), np.sin(psi)])
        vor = Voronoi(np.vstack((c, neighbors)))
        region = vor.regions[vor.point_region[0]]
        assert len(region) == 6 and -1 not in region
        v = vor.vertices[region]
        angle = np.arctan2(v[:, 1]-c[1], v[:, 0]-c[0])
        v = v[np.argsort(angle)]
        gap = np.linalg.norm(W_of(v, c))
        errors.append(abs(gap-float(row['gap'])))
    assert max(errors) < 2e-13
    embedded_error = max(errors)

    # Recompute all displayed finite-range fits from the exported means and SEM.
    means, fits = read_csv('legacy_figure5_samples.csv'), read_csv('legacy_figure5_fits.csv')
    fit_errors = []
    for f in fits:
        group = [r for r in means if r['panel']==f['panel'] and r['n']==f['n']
                 and r['angular_sigma']==f['angular_sigma']]
        x = np.array([float(r['sigma']) for r in group]); y = np.array([float(r['mean_gap']) for r in group])
        sem = np.array([float(r['sem']) for r in group]); X = np.column_stack((x, np.ones(len(x))))
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        A = np.linalg.pinv(X)
        covariance = A@np.diag(sem*sem)@A.T
        residual = y-X@coef
        r2 = 1-np.dot(residual,residual)/np.dot(y-y.mean(),y-y.mean())
        actual = dict(slope=coef[0], intercept=coef[1], r_squared=r2,
                      slope_mc_se=np.sqrt(covariance[0,0]), intercept_mc_se=np.sqrt(covariance[1,1]))
        fit_errors += [abs(v-float(f[k])) for k,v in actual.items()]
    assert max(fit_errors) < 2e-12

    # Four regimes cover the substantial conditioning correction, the triangle,
    # the high-count hexagonal experiment, and combined radial/angular noise.
    checks = [('C',10,.1,0.), ('C',3,.1,0.), ('B',6,.1,0.), ('D',6,.1,.1)]
    sample_reports = []
    for panel,n,sigma,angular in checks:
        row = next(r for r in means if r['panel']==panel and int(r['n'])==n
                   and float(r['sigma'])==sigma and float(r['angular_sigma'])==angular)
        seed = [int(s) for s in row['seed_sequence'].split(';')]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        count, attempted, chunks, total = int(row['accepted']), 0, [], 0
        while total < count:
            batch = min(25000, max(64, int(1.2*(count-total))))
            angle = 2*np.pi*np.arange(n)/n + angular*rng.standard_normal((batch,n))
            radius = 1+sigma*rng.standard_normal((batch,n))
            v = radius[...,None]*np.stack((np.cos(angle),np.sin(angle)), axis=-1)
            accepted = np.flatnonzero(valid_independently(v))
            need = count-total
            consumed = int(accepted[need-1])+1 if len(accepted)>=need else batch
            accepted = accepted[:need]
            attempted += consumed; total += len(accepted)
            chunks.append(np.linalg.norm(W_of(v[accepted]),axis=-1))
        samples = np.concatenate(chunks)
        assert attempted==int(row['attempted']) and count==len(samples)
        error = max(abs(samples.mean()-float(row['mean_gap'])),
                    abs(samples.std(ddof=1)-float(row['sample_sd'])))
        assert error < 2e-13
        sample_reports.append(dict(panel=panel,n=n,sigma=sigma,angular_sigma=angular,
                                   accepted=count,attempted=attempted,max_statistic_error=float(error)))
    return dict(figure6_points=len(ellipse), figure6_max_error=float(ellipse_error),
                figure7_points=len(rows), figure7_max_error=float(embedded_error),
                figure5_fits=len(fits),figure5_max_fit_error=float(max(fit_errors)),
                figure5_independently_regenerated_groups=sample_reports)


def tissue_extra_checks():
    # Recompute new disorder means and complete-tissue bootstrap intervals from
    # the saved 400-cell realizations. Within-tissue cells are never resampled.
    cells=read_csv('tissue_disorder_cells.csv'); summary=read_csv('tissue_disorder_summary.csv')
    errors=[]
    for row in summary:
        eta=float(row['eta']); group=[r for r in cells if float(r['eta'])==eta]
        R,N=int(row['realizations']),int(row['cells_per_realization'])
        assert len(group)==R*N
        values=np.array([float(r['delta']) for r in group]).reshape(R,N)
        W=np.array([complex(float(r['W_real']),float(r['W_imag'])) for r in group]).reshape(R,N)
        errors.append(np.max(abs(abs(W)-values)))
        assert min(float(r['area']) for r in group)>0
        boot=np.random.default_rng(202609103).integers(0,R,(2000,R))
        rep=values.mean(axis=1)[boot].mean(axis=1)
        actual=dict(mean=values.mean(),cell_sd=values.std(ddof=0),mean_ci_low=np.quantile(rep,.025),mean_ci_high=np.quantile(rep,.975))
        errors.extend(abs(float(row[k])-v) for k,v in actual.items())
    correlations=read_csv('tissue_correlations.csv')
    with np.load(DATA/'tissue_cells.npz') as d:
        ij=d['lattice_indices']; W=d['W']; R=len(W)
        ids=np.flatnonzero(np.all((ij>=-10)&(ij<10),axis=1))
        # Enumerate unordered pairs directly; this does not use the producer's
        # displacement-shell lookup. Each tissue remains the bootstrap unit.
        a,b=np.triu_indices(len(ids),k=1); left,right=ids[a],ids[b]
        offset=ij[right]-ij[left]
        shell=offset[:,0]**2+offset[:,0]*offset[:,1]+offset[:,1]**2
        # Match the fixed orientation convention used by the reported
        # normalization; the real numerator itself is symmetric in a and b.
        swap=(offset[:,0]<0)|((offset[:,0]==0)&(offset[:,1]<0))
        oldleft=left.copy(); left[swap]=right[swap];right[swap]=oldleft[swap]
        boot=np.random.default_rng(202609093).integers(0,R,(2000,R))
        for row in correlations:
            take=shell==int(row['shell']); aa,bb=W[:,left[take]],W[:,right[take]]
            assert int(take.sum())==int(row['pairs_per_realization'])
            num=np.real(aa*np.conj(bb)).mean(axis=1); qa=(abs(aa)**2).mean(axis=1);qb=(abs(bb)**2).mean(axis=1)
            value=num.mean()/np.sqrt(qa.mean()*qb.mean())
            rep=num[boot].mean(axis=1)/np.sqrt(qa[boot].mean(axis=1)*qb[boot].mean(axis=1))
            for k,v in dict(correlation=value,ci_low=np.quantile(rep,.025),ci_high=np.quantile(rep,.975)).items():
                errors.append(abs(v-float(row[k])))
    assert max(errors)<2e-12
    return dict(disorder_cell_records=len(cells),disorder_summary_rows=len(summary),
                correlation_shells=len(correlations),max_point_and_bootstrap_interval_error=float(max(errors)))


def main():
    report=dict(status='running')
    report['core']=run_checks()
    print('Core derivations and tensor checks passed.',flush=True)
    report['original_figures']=original_figure_checks()
    print('Original Figure 5–7 calculations and fitted statistics passed.',flush=True)
    report['saved_tissue_windows']=audit_saved_data(DATA)['tissue']
    report['tissue_disorder_and_correlations']=tissue_extra_checks()
    print('Tissue saved statistics and bootstrap intervals passed.',flush=True)
    report['status']='all checks passed'
    files=['code/polygon.py','code/run_original_figures.py','code/run_boundary_identity.py',
           'code/run_tissue.py','code/generate_tissue_windows.py','code/verify_core.py','code/verify_package.py',
           'code/run_regional_mechanics.py','code/verify_regional.py',
           'data/tissue_cells.npz','data/tissue_disorder_cells.csv','data/tissue_summary.csv']
    report['sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    report['scope']='Numerical and code verification of the supplied model and benchmark data; does not establish biological validity, empirical utility, or completeness of the literature.'
    (DATA/'package_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved data/package_verification.json',flush=True)


if __name__=='__main__':
    main()
