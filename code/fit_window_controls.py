#!/usr/bin/env python3
"""Descriptive paired-realization bootstrap for finite-range log-log slopes.

Resample entire realizations jointly across all nested masks. These intervals
measure Monte Carlo uncertainty for the tested masks, not an asymptotic law.
"""
import csv
import json
import numpy as np
from window_controls import DATA, save_json


def fit_group(records, group_key, value, gap_key, boundary=False, seed=0):
    rows=[r for r in records if r[group_key]==value]
    parameter='param' if boundary else 'parameter'
    params=sorted({float(r[parameter]) for r in rows})
    realizations=sorted({int(r['realization']) for r in rows})
    values=np.zeros((len(realizations),len(params)))
    sizes=np.zeros_like(values);boundary_counts=np.zeros_like(values)
    for r in rows:
        i,j=realizations.index(int(r['realization'])),params.index(float(r[parameter]))
        values[i,j]=float(r[gap_key])**2
        sizes[i,j]=float(r['N'])
        if boundary: boundary_counts[i,j]=float(r['nb'])
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(realizations),size=(2000,len(realizations)))
    rms=np.sqrt(values.mean(axis=0));bootrms=np.sqrt(values[indices].mean(axis=1))
    report=dict(group=value,N_min=float(sizes.mean(axis=0).min()),N_max=float(sizes.mean(axis=0).max()),
                realizations=len(realizations),bootstrap_draws=2000,bootstrap_seed=seed,
                interpretation='paired-realization bootstrap of descriptive finite-range log-log fit; not asymptotic uncertainty')
    for name,x,bootx in [('N',sizes.mean(0),sizes[indices].mean(1))]+([
                       ('boundary_cells',boundary_counts.mean(0),boundary_counts[indices].mean(1))] if boundary else []):
        slope=np.polyfit(np.log(x),np.log(rms),1)[0]
        xx=np.log(bootx); yy=np.log(bootrms)
        xx=xx-xx.mean(axis=1)[:,None]
        slopes=np.sum(xx*yy,axis=1)/np.sum(xx*xx,axis=1)
        report[name]=dict(slope=float(slope),q025=float(np.quantile(slopes,.025)),q975=float(np.quantile(slopes,.975)))
    return report


def main():
    report={}
    for n,(filename,key,gap,boundary) in enumerate([
        ('boundary_scaling_realizations.csv','name','gap',True),
        ('driven_scaling_realizations.csv','eps','regional',False),
        ('polarized_scaling_realizations.csv','beta','regional',False)]):
        with (DATA/filename).open() as handle: records=list(csv.DictReader(handle))
        values=list(dict.fromkeys(r[key] for r in records))
        report[filename]=[fit_group(records,key,value,gap,boundary,2026091500+10*n+i) for i,value in enumerate(values)]
    save_json('window_control_fit_bootstrap.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
