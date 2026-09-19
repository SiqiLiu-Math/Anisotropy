#!/usr/bin/env python3
"""Force-balanced regional anisotropy in a prescribed central-spring network.

This extension explicitly relaxes the original equal-weight Voronoi closure. A reference
honeycomb supplies connectivity only. Its primal vertices relax under positive,
zero-rest-length springs, T_e=c_e L_e, with fixed outer vertices and uniform
pressure. The half-interface stress is computed from the *relaxed* forces.
No biological calibration or active time evolution is claimed.

Run: python code/run_regional_mechanics.py
Outputs: data/regional_* and figures/regional_application.*, regional_controls.*
"""
from __future__ import annotations
import argparse
from io import BytesIO
import csv
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial import Voronoi, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

ROOT=Path(__file__).resolve().parents[1]
C0=np.sqrt(3.)


def cross(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def clean_json(value):
    """Undefined isotropic-baseline coherence is JSON null, never NaN."""
    if isinstance(value,dict):return {key:clean_json(val) for key,val in value.items()}
    if isinstance(value,(tuple,list)):return [clean_json(val) for val in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def save_figure(fig,stem):
    """Render in memory, validate completion, then atomically replace outputs."""
    for suffix in ['pdf','png']:
        path=ROOT/'figures'/f'{stem}.{suffix}'
        temp=path.with_name(path.name+'.tmp')
        kwargs={'dpi':240} if suffix=='png' else {}
        with BytesIO() as buffer:
            fig.savefig(buffer,format=suffix,**kwargs)
            data=buffer.getvalue()
        if suffix=='pdf' and not data.rstrip().endswith(b'%%EOF'):
            raise RuntimeError(f'Incomplete PDF render: {stem}')
        temp.write_bytes(data)
        temp.replace(path)


def load_csv(path):
    rows=[]
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            for key,value in row.items():
                if key not in ['condition','region']:row[key]=float(value)
            rows.append(row)
    return rows


def replot_saved():
    """Replot saved realizations without recomputing the experiment."""
    p=make_patch(22)
    with np.load(ROOT/'data/regional_examples.npz') as d:
        if not np.allclose(p['x'],d['reference_vertices'],rtol=0,atol=1e-10):
            raise RuntimeError('Reference geometry differs from saved examples.')
        examples={}
        for name in ['coherent','permuted']:
            y=d[name+'_vertices'];c=d[name+'_coefficients']
            u,v=p['edges'].T;b=y[v]-y[u]
            E=(.5*c[:,None,None]*b[:,:,None]*b[:,None,:])[p['face_edges']].sum(1)
            vv=y[p['faces']];A=.5*cross(vv,np.roll(vv,-1,axis=1)).sum(1)
            examples[name]={'y':y,'E':E,'A':A,'W':E[:,0,0]-E[:,1,1]+2j*E[:,0,1]}
    make_figures(p,load_csv(ROOT/'data/regional_realizations.csv'),
                 load_csv(ROOT/'data/regional_radial_profiles.csv'),examples,
                 load_csv(ROOT/'data/regional_confinement.csv'),
                 load_csv(ROOT/'data/regional_uniform_activation.csv'),
                 load_csv(ROOT/'data/regional_affine_validation.csv'))


def save_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def make_patch(m=22):
    """Hexagonal patch with 1+3m(m+1) complete hexagonal faces."""
    ij=np.array([(i,j) for i in range(-m-2,m+3) for j in range(-m-2,m+3)
                 if max(abs(i),abs(j),abs(i+j))<=m+2])
    q=np.c_[2*ij[:,0]+ij[:,1],np.sqrt(3)*ij[:,1]]
    vor=Voronoi(q)
    selected=np.flatnonzero(np.max(np.c_[abs(ij),abs(ij.sum(1))],axis=1)<=m)
    vids=np.unique(np.concatenate([vor.regions[vor.point_region[i]] for i in selected]))
    assert -1 not in vids
    remap={v:i for i,v in enumerate(vids)}
    x=vor.vertices[vids];faces=[]
    for i in selected:
        vv=[remap[j] for j in vor.regions[vor.point_region[i]]]
        if cross(x[vv],np.roll(x[vv],-1,axis=0)).sum()<0:vv=vv[::-1]
        assert len(vv)==6
        faces.append(vv)
    faces=np.array(faces)
    all_edges=np.sort(np.c_[faces.ravel(),np.roll(faces,-1,axis=1).ravel()],axis=1)
    edges,inv,counts=np.unique(all_edges,axis=0,return_inverse=True,return_counts=True)
    fixed=np.unique(edges[counts==1]);free=np.setdiff1d(np.arange(len(x)),fixed)
    u,v=edges.T;d=x[v]-x[u]
    edge_r=np.linalg.norm((x[u]+x[v])/2,axis=1)
    return dict(m=m,x=x,faces=faces,edges=edges,face_edges=inv.reshape(faces.shape),
                counts=counts,fixed=fixed,free=free,q=q[selected],ij=ij[selected],
                edge_r=edge_r,edge_theta=np.arctan2(d[:,1],d[:,0]))


def coefficient_field(p,beta,phi=0.,radius=12.,width=2.,uniform=False):
    mask=np.ones(len(p['edges'])) if uniform else .5*(1-np.tanh((p['edge_r']-radius)/width))
    return C0*np.exp(beta*mask*np.cos(2*(p['edge_theta']-phi)))


def permute_coefficients(p,c,rng):
    """Exactly preserve coefficient multiset in each reference radial ring."""
    out=c.copy();bins=np.floor(p['edge_r']/2).astype(int)
    for b in np.unique(bins):
        at=np.flatnonzero(bins==b);out[at]=rng.permutation(c[at])
    assert np.array_equal(np.sort(out),np.sort(c))
    return out


def independent_edge_tensor(d,c):
    """Assemble from tension magnitudes and normalized tangents, independently."""
    L=np.linalg.norm(d,axis=1);t=d/L[:,None];T=c*L
    return .5*(T*L)[:,None,None]*t[:,:,None]*t[:,None,:]


def crossing_count(p,y):
    """Count proper intersections between nonincident straight edges.

    Candidate midpoint pairs use a bound greater than every possible sum of
    edge half-lengths. Collinear nonincident overlap is also rejected.
    """
    u,v=p['edges'].T;a=y[u];b=y[v];d=b-a;L=np.linalg.norm(d,axis=1)
    pairs=cKDTree((a+b)/2).query_pairs(float(L.max())+1e-8,output_type='ndarray')
    if len(pairs)==0:return 0
    i,j=pairs.T
    keep=(u[i]!=u[j])&(u[i]!=v[j])&(v[i]!=u[j])&(v[i]!=v[j])
    i,j=i[keep],j[keep]
    z1=cross(d[i],a[j]-a[i]);z2=cross(d[i],b[j]-a[i])
    z3=cross(d[j],a[i]-a[j]);z4=cross(d[j],b[i]-a[j])
    strict=(z1*z2 < -1e-20)&(z3*z4 < -1e-20)
    collinear=(abs(z1)<1e-10)&(abs(z2)<1e-10)&(abs(z3)<1e-10)&(abs(z4)<1e-10)
    projection=(a[j]-a[i])*d[i];p0=projection.sum(1)/(L[i]**2)
    p1=((b[j]-a[i])*d[i]).sum(1)/(L[i]**2)
    overlap=np.minimum(1,np.maximum(p0,p1))-np.maximum(0,np.minimum(p0,p1))>1e-9
    return int(np.sum(strict|(collinear&overlap)))


def solve_network(p,c,check_crossings=True):
    x=p['x'];u,v=p['edges'].T;n=len(x);free=p['free'];fixed=p['fixed']
    lap=coo_matrix((np.r_[c,c,-c,-c],(np.r_[u,v,u,v],np.r_[u,v,v,u])),shape=(n,n)).tocsr()
    y=x.copy();y[free]=spsolve(lap[free][:,free],-lap[free][:,fixed]@x[fixed])
    force=-(lap@y);d=y[v]-y[u];L=np.linalg.norm(d,axis=1)
    edge_E=.5*c[:,None,None]*d[:,:,None]*d[:,None,:]
    E=edge_E[p['face_edges']].sum(axis=1)
    verts=y[p['faces']];A=.5*cross(verts,np.roll(verts,-1,axis=1)).sum(axis=1)
    outgoing=np.roll(verts,-1,axis=1)-verts
    turns=cross(outgoing,np.roll(outgoing,-1,axis=1))
    W=E[:,0,0]-E[:,1,1]+2j*E[:,0,1];trace=E[:,0,0]+E[:,1,1]
    independent=independent_edge_tensor(d,c)[p['face_edges']].sum(axis=1)
    tension_sum=np.bincount(np.r_[u,v],weights=np.r_[c*L,c*L],minlength=n)
    force_relative=np.linalg.norm(force[free],axis=1)/tension_sum[free]
    crossing=crossing_count(p,y) if check_crossings else -1
    # For the complete graph, the positive boundary reaction moment equals the
    # full edge virial. Cells assign half of outer graph edges, so this identity
    # uses each graph edge ONCE rather than silently mixing partitions.
    boundary_moment=np.einsum('ni,nj->ij',y[fixed],(lap@y)[fixed])
    virial=(2*edge_E).sum(axis=0)
    initial_d=x[v]-x[u]
    initial_energy=.5*np.sum(c*np.sum(initial_d**2,axis=1))
    relaxed_energy=.5*np.sum(c*L**2)
    assert A.min()>0 and turns.min()>0 and L.min()>0
    assert crossing in (0,-1)
    assert force_relative.max()<1e-10
    assert relaxed_energy<=initial_energy+1e-9
    return dict(y=y,E=E,W=W,trace=trace,A=A,c=c,L=L,force=force,
                min_area=float(A.min()),min_turn=float(turns.min()),min_edge=float(L.min()),
                relative_force_residual=float(force_relative.max()),crossings=crossing,
                tensor_check_error=float(abs(E-independent).max()),
                boundary_virial_error=float(abs(boundary_moment-virial).max()),
                initial_energy=float(initial_energy),relaxed_energy=float(relaxed_energy),
                max_displacement=float(np.linalg.norm(y-x,axis=1).max()))


def region_metrics(p,s,mask):
    W=s['W'][mask];A=s['A'][mask].sum();local=np.abs(W).sum();summed=W.sum()
    gap=abs(summed)/A
    return dict(n=int(mask.sum()),area=float(A),regional_gap=float(gap),
                local_mean_gap=float(local/A),
                coherence=float(abs(summed)/local) if local>1e-10 else float('nan'),
                normalized_gap=float(abs(summed)/s['trace'][mask].sum()),
                Dxx=float(summed.real/A),Dxy2=float(summed.imag/A),
                major_axis_degrees=float(np.rad2deg(np.angle(summed)/2)) if abs(summed)>1e-10 else 0.)


def region_masks(p):
    r=np.linalg.norm(p['q'],axis=1)
    # Radial thresholds are deliberately between lattice shells: floating-point
    # roundoff must not decide membership of crystallographically equal sites.
    return {'inner':r<6.1,'domain':r<12.1,'core':r<24.1,
            'exterior':(r>18.1)&(r<24.1)}


def uniform_bulk_tensor(beta,phi=0.):
    """Exact periodic honeycomb solution at fixed primitive lattice vectors."""
    # Reference outgoing bonds at an A vertex; every edge is counted once in
    # one two-vertex primitive cell of area 2sqrt(3).
    a=(2/np.sqrt(3))*np.c_[np.cos(np.pi/2+np.arange(3)*2*np.pi/3),
                                    np.sin(np.pi/2+np.arange(3)*2*np.pi/3)]
    c=C0*np.exp(beta*np.cos(2*(np.arctan2(a[:,1],a[:,0])-phi)))
    shift=-np.sum(c[:,None]*a,axis=0)/c.sum();b=a+shift
    sigma=np.einsum('n,ni,nj->ij',c,b,b)/(2*np.sqrt(3))
    W=sigma[0,0]-sigma[1,1]+2j*sigma[0,1]
    return sigma,shift,float(abs(W)),float(abs(W)/np.trace(sigma))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--realizations',type=int,default=40)
    parser.add_argument('--seed',type=int,default=2026091107)
    parser.add_argument('--plot-only',action='store_true',help='Render figures from saved data without simulations.')
    args=parser.parse_args()
    if args.plot_only:
        replot_saved();return
    rng=np.random.default_rng(args.seed)
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    p=make_patch(22);masks=region_masks(p)
    betas=[0.,.1,.25,.5,.75]
    rows=[];checks=[];radial_rows=[];examples={};coherent_solutions={}
    for beta in betas:
        c=coefficient_field(p,beta)
        cases=[('coherent',0,c)]
        if beta>0:cases += [('permuted',i,permute_coefficients(p,c,rng)) for i in range(args.realizations)]
        for condition,i,cc in cases:
            s=solve_network(p,cc)
            if condition=='coherent':coherent_solutions[beta]=s
            for region,mask in masks.items():
                rows.append(dict(beta=beta,condition=condition,realization=i,region=region,
                                 **region_metrics(p,s,mask)))
            checks.append(dict(beta=beta,condition=condition,realization=i,
                               **{key:s[key] for key in ['min_area','min_turn','min_edge',
                               'relative_force_residual','crossings','tensor_check_error',
                               'boundary_virial_error','initial_energy','relaxed_energy','max_displacement']}))
            if beta==.5:
                for lo,hi in zip(np.arange(0,36,3),np.arange(3,39,3)):
                    rr=np.linalg.norm(p['q'],axis=1)
                    mask=(rr>=lo+.05)&(rr<hi+.05) if lo else rr<hi+.05
                    radial_rows.append(dict(condition=condition,realization=i,radius_low=float(lo),
                                            radius_high=float(hi),**region_metrics(p,s,mask)))
                if condition=='coherent' or i==0:examples[condition]=s
    save_csv(ROOT/'data/regional_realizations.csv',rows)
    save_csv(ROOT/'data/regional_mechanical_checks.csv',checks)
    save_csv(ROOT/'data/regional_radial_profiles.csv',radial_rows)

    # Confinement is a separate finite-patch sensitivity study: same material
    # activation radius, strength, width and central observation region.
    confinement=[]
    for m in [14,18,22,28,36,48]:
        pp=make_patch(m);ss=solve_network(pp,coefficient_field(pp,.5))
        mask=region_masks(pp)['domain']
        confinement.append(dict(patch_radius=m,patch_cells=len(pp['faces']),
                                 boundary_min_radius=float(np.linalg.norm(pp['x'][pp['fixed']],axis=1).min()),
                                 patch_area=float(ss['A'].sum()),**region_metrics(pp,ss,mask),
                                 relative_force_residual=ss['relative_force_residual'],
                                 min_turn=ss['min_turn'],crossings=ss['crossings']))
    save_csv(ROOT/'data/regional_confinement.csv',confinement)

    # Uniform directional activation connects to an analytic periodic limit.
    uniform=[]
    for beta in betas:
        ss=solve_network(p,coefficient_field(p,beta,uniform=True))
        sigma,shift,gap,normalized=uniform_bulk_tensor(beta)
        mm=region_metrics(p,ss,masks['domain'])
        uniform.append(dict(beta=beta,analytic_gap=gap,analytic_normalized_gap=normalized,
                            analytic_sigma_xx=float(sigma[0,0]),analytic_sigma_yy=float(sigma[1,1]),
                            sublattice_shift_x=float(shift[0]),sublattice_shift_y=float(shift[1]),**mm))
    save_csv(ROOT/'data/regional_uniform_activation.csv',uniform)

    # An affine primal-network transformation is an exact equilibrium and
    # stress-transformation check, separate from squeezing Voronoi generators.
    baseline=coherent_solutions[0.];affine=[]
    for strain in [-.3,-.1,.0,.1,.3]:
        F=np.diag([np.exp(strain),np.exp(-strain)]);xx=p['x']@F.T
        edges=p['edges'];d=xx[edges[:,1]]-xx[edges[:,0]]
        edgeE=.5*C0*d[:,:,None]*d[:,None,:];E=edgeE[p['face_edges']].sum(1)
        vv=xx[p['faces']];area=.5*cross(vv,np.roll(vv,-1,axis=1)).sum(1)
        sigma=E/area[:,None,None];expected=F@F.T/np.linalg.det(F)
        u,v=edges.T;force=np.zeros_like(xx);np.add.at(force,u,C0*d);np.add.at(force,v,-C0*d)
        affine.append(dict(strain=strain,tensor_error=float(abs(sigma-expected).max()),
                            force_residual=float(np.linalg.norm(force[p['free']],axis=1).max()),
                            normalized_gap=float(np.tanh(2*abs(strain)))))
    save_csv(ROOT/'data/regional_affine_validation.csv',affine)

    # Persist representative coordinates and coefficients for independent
    # verification/replotting; raw summary rows contain every realization.
    np.savez_compressed(ROOT/'data/regional_examples.npz',reference_vertices=p['x'],faces=p['faces'],
                        edges=p['edges'],face_edges=p['face_edges'],fixed=p['fixed'],free=p['free'],
                        reference_centers=p['q'],coherent_vertices=examples['coherent']['y'],
                        coherent_coefficients=examples['coherent']['c'],
                        permuted_vertices=examples['permuted']['y'],permuted_coefficients=examples['permuted']['c'])
    meta=dict(model='positive zero-rest-length central springs on fixed honeycomb connectivity',
              mechanical_law='T_e = c_e L_e',baseline_c=C0,reference_generator_spacing=2.,
              patch_axial_radius=22,patch_cells=len(p['faces']),vertices=len(p['x']),edges=len(p['edges']),
              fixed_vertices=len(p['fixed']),free_vertices=len(p['free']),
              regions={key:int(mask.sum()) for key,mask in masks.items()},
              beta_values=betas,activation_radius=12.,activation_transition_width=2.,
              preferred_axis_radians=0.,permutation_ring_width=2.,realizations=args.realizations,
              seed=args.seed,numpy_version=np.__version__,scipy_version=scipy.__version__,
              parameter_selection='Exploratory probes at beta 0.1-1.0 and patch radii 14-48 informed selection; not preregistered.',
              uncertainty='Permuted controls: empirical 2.5-97.5 percentiles across independent permutations; not confidence intervals on the mean.',
              interpretation='Material coefficient distribution is matched before relaxation; relaxed lengths and tensions are not held fixed.',
              limitations=['Fixed topology and outer boundary; no T1 events or time evolution.',
                           'No experimentally fitted constitutive rule or biological validation.',
                           'Final networks do not generally obey T=k times original generator separation; use half-interface stress.',
                           'A local coefficient anisotropy can relax; confinement controls residual regional stress.'])
    (ROOT/'data/regional_metadata.json').write_text(json.dumps(meta,indent=2))
    make_figures(p,rows,radial_rows,examples,confinement,uniform,affine)
    summary=dict(metadata=meta,mechanical_validation={
        'max_relative_force_residual':max(r['relative_force_residual'] for r in checks),
        'min_face_area':min(r['min_area'] for r in checks),'min_face_turn':min(r['min_turn'] for r in checks),
        'min_edge_length':min(r['min_edge'] for r in checks),'crossings_all_runs':sum(r['crossings'] for r in checks),
        'max_independent_tensor_error':max(r['tensor_check_error'] for r in checks),
        'max_boundary_virial_error':max(r['boundary_virial_error'] for r in checks),
        'max_confinement_relative_force_residual':max(r['relative_force_residual'] for r in confinement),
        'max_main_or_confinement_relative_force_residual':max([r['relative_force_residual'] for r in checks]+[r['relative_force_residual'] for r in confinement])},
        beta_half={region:summarize_group(rows,.5,region) for region in masks},confinement=confinement,uniform=uniform)
    (ROOT/'data/regional_summary.json').write_text(json.dumps(clean_json(summary),indent=2,allow_nan=False))
    print(json.dumps(clean_json(summary),indent=2,allow_nan=False))


def summarize_group(rows,beta,region):
    r=[v for v in rows if v['beta']==beta and v['region']==region]
    coherent=next(v for v in r if v['condition']=='coherent')
    perm=[v for v in r if v['condition']=='permuted']
    return dict(coherent=coherent,permuted={key:dict(mean=float(np.mean([v[key] for v in perm])),
                median=float(np.median([v[key] for v in perm])),
                q025=float(np.quantile([v[key] for v in perm],.025)),
                q975=float(np.quantile([v[key] for v in perm],.975)))
                for key in ['regional_gap','local_mean_gap','coherence','normalized_gap','Dxx','Dxy2']})


def group_curve(rows,key,region='domain'):
    betas=sorted(set(r['beta'] for r in rows));coh=[];means=[];lo=[];hi=[]
    for beta in betas:
        rr=[r for r in rows if r['region']==region and r['beta']==beta]
        coh.append(next(r[key] for r in rr if r['condition']=='coherent'))
        vals=[r[key] for r in rr if r['condition']=='permuted']
        if not vals:vals=[coh[-1]]
        means.append(np.mean(vals));lo.append(np.quantile(vals,.025));hi.append(np.quantile(vals,.975))
    return np.array(betas),np.array(coh),np.array(means),np.array(lo),np.array(hi)


def make_figures(p,rows,radial_rows,examples,confinement,uniform,affine):
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'pdf.fonttype':42,
                         'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
    blue='#176b9b';orange='#c86a2c'
    fig,axs=plt.subplots(2,3,figsize=(11.6,7.0),layout='constrained')
    vmax=.45
    for ax,name,title in zip(axs[0,:2],['coherent','permuted'],['Coherent coefficients','Permuted coefficients']):
        s=examples[name];local=np.abs(s['W'])/s['A'];r=np.linalg.norm(p['q'],axis=1)
        mask=r<26.1;poly=PolyCollection(s['y'][p['faces'][mask]],array=local[mask],cmap='magma',
                                      edgecolors='#bbbbbb',linewidths=.22,clim=(0,vmax))
        ax.add_collection(poly)
        centers=s['y'][p['faces']].mean(1);angle=np.angle(s['W'])/2
        lengths=np.where(local>1e-8,.65,0.)
        directions=np.c_[np.cos(angle),np.sin(angle)]*lengths[:,None]
        seg=np.stack([centers-directions/2,centers+directions/2],axis=1)
        ax.add_collection(LineCollection(seg[mask],colors='white',linewidths=.65))
        circle=plt.Circle((0,0),12,fill=False,color='#46c9ce',ls='--',lw=1.1);ax.add_patch(circle)
        ax.set(aspect='equal',xlim=(-26,26),ylim=(-26,26),xlabel='$x$',ylabel='$y$')
    fig.colorbar(poly,ax=list(axs[0,:2]),shrink=.85,label='Local stress gap / $k$',extend='max')
    ax=axs[0,2]
    for region,style in [('domain','-'),('core','--')]:
        b,c,m,lo,hi=group_curve(rows,'regional_gap',region)
        ax.plot(b,c,'o'+style,color=blue,label='Coherent, '+('127 cells' if region=='domain' else '517 cells'))
        ax.plot(b,m,'s'+style,color=orange,label='Permuted, '+('127 cells' if region=='domain' else '517 cells'))
        ax.fill_between(b,lo,hi,color=orange,alpha=.12)
    ax.set(xlabel='Directional coefficient amplitude $\\beta$',ylabel='Regional stress gap / $k$')
    ax.legend(fontsize=10,loc='upper left', frameon=False)
    for ax,key,label in [(axs[1,0],'local_mean_gap','Area-weighted local gap / $k$'),
                          (axs[1,1],'coherence','Directional coherence $C$')]:
        b,c,m,lo,hi=group_curve(rows,key)
        ax.plot(b,c,'o-',color=blue,label='Coherent');ax.plot(b,m,'s-',color=orange,label='Permuted')
        ax.fill_between(b,lo,hi,color=orange,alpha=.17)
        ax.set(xlabel='Directional coefficient amplitude $\\beta$',ylabel=label)
        if key=='coherence':ax.set(ylim=(-.02,1.04));pass
        else:ax.legend(fontsize=10, frameon=False)
    ax=axs[1,2]
    distance=[r['boundary_min_radius'] for r in confinement]
    ax.plot(distance,[r['regional_gap'] for r in confinement],'o-',color=blue)
    ax.set(xlabel='Distance to nearest fixed junction',ylabel='Regional stress gap / $k$')
    pass  # panel F note removed
    for letter,ax in zip('ABCDEF',axs.flat):
        ax.text(-.14,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=16)
    save_figure(fig,'regional_application')
    plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(8.1,3.3),layout='constrained')
    ax=axs[0];beta=np.array([r['beta'] for r in uniform]);exact=np.array([r['analytic_gap'] for r in uniform])
    ax.plot(beta,exact,'k-',label='Exact periodic solution')
    ax.plot(beta,[r['regional_gap'] for r in uniform],'o',color=blue,label='Finite patch: central region')
    ax.set(xlabel='Uniform coefficient amplitude $\\beta$',ylabel='Regional stress gap / $k$');ax.legend(fontsize=10, frameon=False)
    ax=axs[1]
    rb=sorted(set(r['radius_low'] for r in radial_rows));coh=[];mean=[];low=[];high=[]
    for radius in rb:
        rr=[r for r in radial_rows if r['radius_low']==radius]
        coh.append(next(r['Dxx'] for r in rr if r['condition']=='coherent'))
        vals=[r['Dxx'] for r in rr if r['condition']=='permuted'];mean.append(np.mean(vals))
        low.append(np.quantile(vals,.025));high.append(np.quantile(vals,.975))
    rr=np.array(rb)+1.5
    ax.plot(rr,coh,'o-',color=blue,label='Coherent');ax.plot(rr,mean,'s-',color=orange,label='Permuted')
    ax.fill_between(rr,low,high,color=orange,alpha=.17);ax.axvline(12,color='.45',ls=':',lw=1)
    ax.axhline(0,color='.65',lw=.7)
    ax.set(xlabel='Reference radial position',ylabel=r'Regional $(\sigma_{xx}-\sigma_{yy})/k$')
    ax.text(.04,.94,r'$\beta=0.5$',transform=ax.transAxes,va='top')
    ax.legend(fontsize=10,loc='upper right', frameon=False)
    for letter,ax in zip('AB',axs):ax.text(-.14,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=16)
    save_figure(fig,'regional_controls')
    plt.close(fig)


if __name__=='__main__':main()
