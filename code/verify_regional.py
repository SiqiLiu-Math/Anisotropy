#!/usr/bin/env python3
"""Independent checks for the regional revision.

No manuscript polygon or network force implementations are imported for the
verification formulas. Weighted cells are constructed by independent half-plane
clipping; network checks consume saved arrays and independently differentiate
the energy and assemble boundary-reaction moments.
"""
from pathlib import Path
import json
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def clip(poly, normal, offset):
    """Keep x.normal <= offset, with counterclockwise polygon ordering."""
    output = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = a @ normal-offset, b @ normal-offset
        ina, inb = da <= 1e-12, db <= 1e-12
        if ina:
            output.append(a)
        if ina != inb:
            output.append(a+(b-a)*(da/(da-db)))
    return np.asarray(output)


def weighted_cell(q, weights, owner):
    poly = np.array([[-50., -50.], [50., -50.], [50., 50.], [-50., 50.]])
    for j in np.argsort(np.sum((q-q[owner])**2, axis=1)):
        if j == owner:
            continue
        normal = 2*(q[j]-q[owner])
        offset = q[j]@q[j]-q[owner]@q[owner]+weights[owner]-weights[j]
        poly = clip(poly, normal, offset)
        if len(poly) < 3:
            raise AssertionError('Empty cell')
    return poly


def normals_tensor(v, q):
    """Normal-projector assembly independent of the cross-product kernel."""
    edges = np.roll(v, -1, axis=0)-v
    lengths = np.linalg.norm(edges, axis=1)
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))/lengths[:, None]
    midpoints = (v+np.roll(v, -1, axis=0))/2
    distances = np.einsum('ni,ni->n', midpoints-q, normals)
    A = sum(abs(np.linalg.det(np.stack((v[j]-v[0],v[j+1]-v[0]))))/2 for j in range(1,len(v)-1))
    E = sum((length*d*(np.eye(2)-np.outer(n,n)) for length,d,n in zip(lengths,distances,normals)), np.zeros((2,2)))
    return E, A, edges, lengths, normals, midpoints


def check_weighted_identity():
    ij = np.array([(i,j) for i in range(-5,6) for j in range(-5,6)])
    q0 = np.column_stack((2*ij[:,0]+ij[:,1],np.sqrt(3)*ij[:,1]))
    chosen = set(np.flatnonzero(np.max(abs(ij),axis=1)<=2).tolist())
    rng = np.random.default_rng(542174)
    cases = []
    for weighted in (False, True):
        for realization in range(4):
            q = q0+.15*rng.normal(size=q0.shape)
            weights = .25*rng.normal(size=len(q)) if weighted else np.zeros(len(q))
            E = np.zeros((2,2)); B = np.zeros((2,2)); A = 0.; localerr = 0.; bound = 0.
            for i in chosen:
                v = weighted_cell(q,weights,i)
                Ei, Ai, edges, lengths, normals, mids = normals_tensor(v,q[i])
                Bcell = np.zeros((2,2))
                for edge,L,n,m in zip(edges,lengths,normals,mids):
                    t = edge/L
                    tangential_offset = np.dot(m-q[i], t)
                    term = L*tangential_offset*np.outer(t,n)
                    Bcell += term
                    power = np.sum((q-m)**2,axis=1)-weights
                    power[i] = np.inf
                    neighbor = int(np.argmin(abs(power-(np.sum((q[i]-m)**2)-weights[i]))))
                    if neighbor not in chosen:
                        B += term
                        bound += L*abs(tangential_offset)
                localerr = max(localerr,float(np.max(abs(Ei-Ai*np.eye(2)-Bcell))))
                E += Ei; A += Ai
            err = float(np.max(abs(E-A*np.eye(2)-B)))
            gap = float(np.diff(np.linalg.eigvalsh(E))[0])
            assert err < 2e-10 and localerr < 2e-11
            assert gap <= bound+2e-10
            cases.append(dict(weighted=weighted,realization=realization,cells=len(chosen),maximum_local_identity_error=localerr,regional_identity_error=err,regional_gap=gap,bound=bound))
    return cases


def check_bravais_null():
    ij = np.array([(i,j) for i in range(-4,5) for j in range(-4,5)])
    q0 = np.column_stack((2*ij[:,0]+ij[:,1],np.sqrt(3)*ij[:,1]))
    center = int(np.flatnonzero(np.all(ij==0,axis=1))[0])
    records=[]
    for s in (-.4,0.,.4):
        for shear in (-.25,0.,.25):
            F=np.array([[np.exp(s),shear],[0,np.exp(-s)]])
            q=q0@F.T
            v=weighted_cell(q,np.zeros(len(q)),center)
            E,A,edges,L,n,m=normals_tensor(v,q[center])
            midpoint_tangent=np.max(abs(np.einsum('ni,ni->n',m-q[center],edges/L[:,None])))
            err=float(np.max(abs(E/A-np.eye(2))))
            assert err<1e-11 and midpoint_tangent<1e-11
            records.append(dict(s=s,shear=shear,sides=len(v),tensor_isotropy_error=err,maximum_tangential_midpoint_offset=float(midpoint_tangent)))
    return records


def node_gradient(y,edges,c):
    gradient=np.zeros_like(y)
    for (a,b),cc in zip(edges,c):
        force=cc*(y[a]-y[b])
        gradient[a]+=force;gradient[b]-=force
    return gradient


def all_face_metrics(y,faces,edges,c):
    lookup={tuple(sorted(e)):cc for e,cc in zip(edges,c)}
    tensors=[];areas=[];minhalf=np.inf
    for face in faces:
        p=y[face];E=np.zeros((2,2))
        area=sum(np.linalg.det(np.stack((p[j]-p[0],p[j+1]-p[0])))/2 for j in range(1,len(p)-1))
        for j,a in enumerate(face):
            b=face[(j+1)%len(face)];ell=y[b]-y[a];L=np.linalg.norm(ell)
            tangent=ell/L;T=lookup[tuple(sorted((a,b)))]*L
            E += T*L*np.outer(tangent,tangent)/2
            for k in range(len(p)):
                if k in (j,(j+1)%len(p)):continue
                r=p[k]-p[j]
                minhalf=min(minhalf,ell[0]*r[1]-ell[1]*r[0])
        assert area>0
        areas.append(area);tensors.append(E)
    assert minhalf>0
    return np.asarray(tensors),np.asarray(areas),minhalf


def no_crossing_check(y,edges):
    """Brute-force AABB filtering, independent of production KD-tree filter."""
    a=y[edges[:,0]];b=y[edges[:,1]]
    low=np.minimum(a,b);high=np.maximum(a,b)
    count=0
    def orientation(p,q,r):
        d=q-p;v=r-p
        return d[...,0]*v[...,1]-d[...,1]*v[...,0]
    for i in range(len(edges)):
        candidates=np.flatnonzero(np.all(low[i]<=high[i+1:]+1e-11,axis=1)&np.all(low[i+1:]<=high[i]+1e-11,axis=1))+i+1
        if not len(candidates):continue
        keep=np.all(edges[candidates]!=edges[i,0],axis=1)&np.all(edges[candidates]!=edges[i,1],axis=1)
        j=candidates[keep]
        if not len(j):continue
        o1=orientation(a[i],b[i],a[j]);o2=orientation(a[i],b[i],b[j])
        o3=orientation(a[j],b[j],a[i]);o4=orientation(a[j],b[j],b[i])
        # AABB overlap + orientation tests include touching/collinear overlap.
        contact=(o1*o2<=1e-22)&(o3*o4<=1e-22)
        count+=int(contact.sum())
    assert count==0
    return count


def representative_graph_checks():
    path=ROOT/'data/regional_examples.npz'
    if not path.exists():return {'status':'Representative network arrays unavailable'}
    data=dict(np.load(path));x=data['reference_vertices'];faces=data['faces'];edges=data['edges']
    fixed=data['fixed'];free=data['free'];counts=np.zeros(len(edges),dtype=int)
    lookup={tuple(sorted(e)):i for i,e in enumerate(edges)}
    for face in faces:
        for a,b in zip(face,np.roll(face,-1)):
            counts[lookup[tuple(sorted((a,b)))]]+=1
    assert np.all((counts==1)|(counts==2))
    assert np.array_equal(np.unique(edges[counts==1]),fixed)
    reports=[];rng=np.random.default_rng(2026091141)
    for condition in ('coherent','permuted'):
        y=data[condition+'_vertices'];c=data[condition+'_coefficients']
        assert np.array_equal(y[fixed],x[fixed]) and np.all(c>0)
        g=node_gradient(y,edges,c)
        E,A,minhalf=all_face_metrics(y,faces,edges,c)
        assert np.max(abs(g[free]))<2e-10
        maxgradient=0.
        # Finite differences of local energy avoid subtracting the large total energy.
        for v in np.r_[rng.choice(free,20,replace=False),rng.choice(fixed,10,replace=False)]:
            neighbor=[];cs=[]
            for (a,b),cc in zip(edges,c):
                if a==v:neighbor.append(b);cs.append(cc)
                elif b==v:neighbor.append(a);cs.append(cc)
            neighbor=np.array(neighbor);cs=np.array(cs)
            for axis in (0,1):
                plus=y[v].copy();minus=y[v].copy();h=1e-4
                plus[axis]+=h;minus[axis]-=h
                ep=.5*np.sum(cs*np.sum((plus-y[neighbor])**2,axis=1))
                em=.5*np.sum(cs*np.sum((minus-y[neighbor])**2,axis=1))
                fd=(ep-em)/(2*h)
                maxgradient=max(maxgradient,float(abs(fd-g[v,axis])))
        assert maxgradient<1e-8
        moment=y[fixed].T@g[fixed]
        d=y[edges[:,1]]-y[edges[:,0]]
        full=sum((cc*np.outer(dd,dd) for cc,dd in zip(c,d)),np.zeros((2,2)))
        exterior=.5*sum((cc*np.outer(dd,dd) for cc,dd in zip(c[counts==1],d[counts==1])),np.zeros((2,2)))
        virialrel=float(np.max(abs(moment-full))/np.max(abs(full)))
        partitionerr=float(np.max(abs(E.sum(0)+exterior-full)))
        assert virialrel<1e-11 and partitionerr<1e-9
        theta=.713;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        shift=np.array([.317,-.826]);yr=y@R.T+shift
        Er,Ar,_=all_face_metrics(yr,faces,edges,c)
        gr=node_gradient(yr,edges,c)
        rotationerr=float(np.max(abs(Er-np.einsum('ij,njk,lk->nil',R,E,R))))
        assert rotationerr<1e-10 and np.max(abs(gr-g@R.T))<1e-10
        crossings=no_crossing_check(y,edges)
        reports.append(dict(condition=condition,faces=len(faces),vertices=len(x),maximum_free_energy_gradient=float(np.max(abs(g[free]))),finite_difference_gradient_error=maxgradient,relative_boundary_virial_error=virialrel,half_interface_partition_identity_error=partitionerr,minimum_convex_halfplane_test=float(minhalf),nonincident_contacts=crossings,rotation_tensor_error=rotationerr))
    # Coefficients, including each radial-shell histogram, must agree exactly.
    rad=np.linalg.norm((x[edges[:,0]]+x[edges[:,1]])/2,axis=1)
    bins=np.floor(rad/2).astype(int)
    a=data['coherent_coefficients'];b=data['permuted_coefficients']
    for shell in np.unique(bins):
        idx=bins==shell
        assert np.array_equal(np.sort(a[idx]),np.sort(b[idx]))
    # Final edge tensions are deliberately NOT claimed to be matched.
    ta=a*np.linalg.norm(data['coherent_vertices'][edges[:,1]]-data['coherent_vertices'][edges[:,0]],axis=1)
    tb=b*np.linalg.norm(data['permuted_vertices'][edges[:,1]]-data['permuted_vertices'][edges[:,0]],axis=1)
    return dict(cases=reports,coefficient_histograms_match_in_every_reference_ring=True,final_tension_histograms_match=bool(np.allclose(np.sort(ta),np.sort(tb))),final_tension_rms_sorted_difference=float(np.sqrt(np.mean((np.sort(ta)-np.sort(tb))**2))))


def independent_affine_checks():
    data=dict(np.load(ROOT/'data/regional_examples.npz'))
    x=data['reference_vertices'];edges=data['edges'];faces=data['faces'];free=data['free']
    c=np.full(len(edges),np.sqrt(3.));out=[]
    for F in (np.eye(2),np.array([[1.2,.25],[0,.9]]),np.diag([np.exp(.3),np.exp(-.3)])):
        y=x@F.T
        E,A,_=all_face_metrics(y,faces,edges,c)
        expected=F@F.T/np.linalg.det(F)
        err=float(np.max(abs(E/A[:,None,None]-expected)))
        grad=float(np.max(abs(node_gradient(y,edges,c)[free])))
        assert err<1e-10 and grad<1e-10
        out.append(dict(F=F.tolist(),determinant=float(np.linalg.det(F)),maximum_stress_transport_error=err,maximum_free_energy_gradient=grad))
    return out



def independently_check_reported_metrics():
    data=dict(np.load(ROOT/'data/regional_examples.npz'))
    rows=list(csv.DictReader((ROOT/'data/regional_realizations.csv').open()))
    radial=np.linalg.norm(data['reference_centers'],axis=1)
    masks={'inner':radial<6.1,'domain':radial<12.1,'core':radial<24.1,'exterior':(radial>18.1)&(radial<24.1)}
    err=0.;checked=0
    for condition in ('coherent','permuted'):
        E,A,_=all_face_metrics(data[condition+'_vertices'],data['faces'],data['edges'],data[condition+'_coefficients'])
        for region,mask in masks.items():
            target=next(r for r in rows if float(r['beta'])==.5 and r['condition']==condition and int(r['realization'])==0 and r['region']==region)
            tensor=E[mask].sum(0);area=A[mask].sum()
            eig=np.linalg.eigvalsh(tensor);regional=(eig[1]-eig[0])/area
            cell_eig=np.linalg.eigvalsh(E[mask]);local=np.sum(cell_eig[:,1]-cell_eig[:,0])/area
            measured={'area':area,'regional_gap':regional,'local_mean_gap':local,'coherence':regional/local,'normalized_gap':(eig[1]-eig[0])/eig.sum(),'Dxx':(tensor[0,0]-tensor[1,1])/area,'Dxy2':2*tensor[0,1]/area}
            err=max(err,max(abs(float(target[k])-v) for k,v in measured.items()))
            checked+=1
    assert err<1e-9
    uniform=list(csv.DictReader((ROOT/'data/regional_uniform_activation.csv').open()))
    uniformerr=0.
    # Independent explicit scalar form for phi=0 and vertical/oblique bonds.
    for row in uniform:
        beta=float(row['beta']);a=np.exp(beta/2);b=np.exp(-beta)
        xx=a;yy=3*a*b/(2*a+b)
        uniformerr=max(uniformerr,abs(xx-float(row['analytic_sigma_xx'])),abs(yy-float(row['analytic_sigma_yy'])))
    assert uniformerr<1e-12
    return dict(representative_region_rows_checked=checked,maximum_reported_metric_error=float(err),periodic_analytic_rows=len(uniform),maximum_periodic_scalar_formula_error=float(uniformerr))


def main():
    (ROOT/'review').mkdir(parents=True, exist_ok=True)
    report=dict(weighted_halfplane_identity=check_weighted_identity(),bravais_halfplane_null=check_bravais_null(),representative_graph=representative_graph_checks(),general_affine_transport=independent_affine_checks(),reported_metrics=independently_check_reported_metrics())
    (ROOT/'review/independent_regional_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
