#!/usr/bin/env python3
"""Independently check raw/aggregate controls and representative tensor assembly."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.spatial import Voronoi
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'


def csv_rows(name):
    return list(csv.DictReader((DATA/name).open()))


def independent_realization(seed, eps=0., beta=0.):
    labels=np.array([(i,j) for i in range(30) for j in range(30)])
    q=np.column_stack((2*labels[:,0]+labels[:,1],np.sqrt(3)*labels[:,1])).astype(float)
    q+=np.random.default_rng(seed).normal(0,.05,q.shape)
    q=q@np.diag([1+eps,1/(1+eps)])
    vor=Voronoi(q)
    chosen=np.flatnonzero(np.max(abs(labels-14.5),axis=1)<=9)
    assert len(chosen)==324
    values={}
    for i in chosen:
        vertices=vor.vertices[vor.regions[vor.point_region[i]]]
        c=vertices.mean(0)
        vertices=vertices[np.argsort(np.arctan2(vertices[:,1]-c[1],vertices[:,0]-c[0]))]
        E=np.zeros((2,2))
        for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
            length=np.linalg.norm(b-a); t=(b-a)/length
            normal=np.array([t[1],-t[0]])
            distance=np.dot((a+b)/2-q[i],normal)
            # cos(2 phi)=tx^2-ty^2, independent of the angle-based production form.
            coefficient=1+beta*(t[0]**2-t[1]**2)
            E+=coefficient*length*distance*np.outer(t,t)
        values[i]=complex(E[0,0]-E[1,1],2*E[0,1])
    mean=sum(values.values())/len(chosen)
    accum=np.zeros(4)
    lookup={tuple(labels[i]):i for i in chosen}
    for i in chosen:
        for offset in [(1,0),(0,1),(-1,1)]:
            j=lookup.get(tuple(labels[i]+offset))
            if j is not None:
                a,b=values[i],values[j]
                accum+=[np.real(a*np.conj(b)),abs(a)*abs(b),
                        np.real((a-mean)*np.conj(b-mean)),abs(a-mean)*abs(b-mean)]
    w=np.array(list(values.values()))
    return dict(local=float(abs(w).mean()),regional=float(abs(w.sum())),
                null=float(np.sqrt(np.sum(abs(w)**2))),corr=accum[0]/accum[1],corr_centered=accum[2]/accum[3])


def main():
    report={};maximum=0.;raw_count=0
    for name,key,regional in [('driven','eps','regional'),('polarized','beta','reg')]:
        raw=csv_rows(name+'_realizations.csv');summary=json.loads((DATA/(name+'.json')).read_text())
        raw_count+=len(raw)
        for row in summary:
            rows=[r for r in raw if float(r[key])==row[key]]
            assert len(rows)==40 and all(int(r['N'])==324 for r in rows)
            gap=np.sqrt(np.mean([float(r['regional'])**2 for r in rows]))
            reconstructed={'local':np.mean([float(r['local']) for r in rows]),regional:gap,
                           'per_cell':gap/324,'ratio':gap/np.mean([float(r['null']) for r in rows])}
            reconstructed.update({stat:np.mean([float(r[stat]) for r in rows]) for stat in ['corr','corr_centered','corr_centered_rms']})
            maximum=max(maximum,max(abs(row[k]-v) for k,v in reconstructed.items()))
            first=rows[0]
            kwargs={'eps':row[key]} if name=='driven' else {'beta':row[key]}
            independent=independent_realization(int(first['seed']),**kwargs)
            maximum=max(maximum,max(abs(float(first[k])-v) for k,v in independent.items()))
        scaling=csv_rows(name+'_scaling_realizations.csv');raw_count+=len(scaling)
        for row in csv_rows(name+'_scaling_summary.csv'):
            rows=[r for r in scaling if r[key]==row[key] and r['parameter']==row['parameter']]
            assert len(rows)==25
            val=np.sqrt(np.mean([float(r['regional'])**2 for r in rows]))
            maximum=max(maximum,abs(val-float(row['rms'])))
    raw=csv_rows('boundary_scaling_realizations.csv');raw_count+=len(raw)
    for row in json.loads((DATA/'boundary_scaling.json').read_text()):
        group=[r for r in raw if r['name']==row['name'] and float(r['param'])==row['param']]
        assert len(group)==60
        for key in ['N','P','nb']:
            maximum=max(maximum,abs(row[key]-np.mean([float(r[key]) for r in group])))
        maximum=max(maximum,abs(row['rms']-np.sqrt(np.mean([float(r['gap'])**2 for r in group]))))
    protocol=json.loads((DATA/'window_control_protocol.json').read_text())
    for mask in protocol['masks'].values():
        labels=np.array(mask['selected_lattice_labels']);d=labels-16.5;p=mask['parameter']
        if mask['kind']=='label_box': valid=np.max(abs(d),axis=1)<=p
        elif mask['kind']=='label_disk': valid=np.sum(d*d,axis=1)<=p*p
        else:
            th=np.deg2rad(31); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
            valid=np.max(abs(d@R.T),axis=1)<=p
        assert np.all(valid)
        assert np.array_equal(labels[:,0]*34+labels[:,1],mask['selected_global_indices'])
    regular=independent_realization(5000,beta=.02) # geometry-disordered representative compared above
    # Exact regular hexagon with generator spacing 2: one-cell W=2 sqrt(3) beta.
    th=np.arange(6)*np.pi/3+np.pi/6
    v=(2/np.sqrt(3))*np.column_stack((np.cos(th),np.sin(th)))
    E=np.zeros((2,2))
    for a,b in zip(v,np.roll(v,-1,axis=0)):
        edge=b-a;L=np.linalg.norm(edge);t=edge/L;n=np.array([t[1],-t[0]])
        E+=(1+.02*(t[0]**2-t[1]**2))*L*np.dot(a,n)*np.outer(t,t)
    hexgap=324*np.hypot(E[0,0]-E[1,1],2*E[0,1])
    assert abs(hexgap-324*np.sqrt(12)*.02)<1.e-10
    residuals=csv_rows('polarized_realizations.csv')
    baseline=max(float(r['max_relative_junction_residual']) for r in residuals if float(r['beta'])==0)
    loaded=max(float(r['max_relative_junction_residual']) for r in residuals if float(r['beta'])==.4)
    assert baseline<1.e-10 and loaded>.01
    assert maximum<1.e-9,maximum
    report.update(status='passed',raw_records_reaggregated=raw_count,representative_full_windows_independently_assembled=12,
                  observation_masks_checked=len(protocol['masks']),max_absolute_metric_error=maximum,
                  regular_hexagon_324_cell_gap_beta002=hexgap,baseline_max_relative_junction_residual=baseline,
                  beta04_max_relative_junction_residual=loaded,
                  scope='All new plotted aggregate points checked against saved raw records. One full 324-cell realization per amplitude independently reassembled. This does not turn the prescribed fixed-geometry modulation into an equilibrium model.')
    (ROOT/'review').mkdir(exist_ok=True)
    (ROOT/'review/window_control_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
