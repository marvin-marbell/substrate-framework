"""Physical source-BRIGHT multiplets/all virtual curvature, including dark lower states."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from rich_spatial_holonomy_20261014_model import build, momentum_basis, fourier_source

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave8_20261014_curvature'


def second_order(e, v, h1, h2, group):
    b=v[:,group]
    other=np.flatnonzero(np.abs(e-e[group[0]])>1e-8)
    a=v[:,other].conj().T@(h1@b)
    return b.conj().T@(h1@b), b.conj().T@(h2@b)+2*a.conj().T@(a/(e[group[0]]-e[other])[:,None]),other


def main():
    data=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    p=json.loads((ROOT/data['parameter_input']).read_text())
    model=build(p)
    reference=np.load(ROOT/data['state_input'])['neutral_reference']
    assert reference.shape[1]==1, 'Own recorded reference is rank1; no silent multiplet truncation'
    g=reference[:,0];eref=float(np.vdot(g,model['H']@g).real)
    h1=-p['charge']/p['sites']*sum(model['edgeJ'])
    h2=-p['charge']**2/p['sites']**2*sum(link+link.getH() for link in model['links'])
    u0=momentum_basis(model,6,0);e0,v0=eigh((u0.getH()@model['H']@u0).toarray())
    first0,second0,_=second_order(e0,v0,(u0.getH()@h1@u0).toarray(),(u0.getH()@h2@u0).toarray(),np.array([0]))
    refcurve=float(second0[0,0].real)
    s2=model['S2'];ident=sp.eye(s2.shape[0],format='csr')
    half=(s2-3.75*ident)@(s2-8.75*ident)/24
    records=[]
    for momentum in range(3):
        sources=fourier_source(model,momentum)
        for number,operators,target in [(7,sources,momentum),(5,[s.getH() for s in sources],(-momentum)%3)]:
            u=momentum_basis(model,number,target)
            e,v=eigh((u.getH()@model['H']@u).toarray())
            columns=np.column_stack([s@g for s in operators])
            amplitude=v.conj().T@(u.getH()@columns)
            weight=np.sum(abs(amplitude)**2,axis=1)
            firstbright=int(np.flatnonzero(weight>1e-10)[0])
            group=np.flatnonzero(abs(e-e[firstbright])<1e-8)
            first,second,other=second_order(e,v,(u.getH()@h1@u).toarray(),(u.getH()@h2@u).toarray(),group)
            slopes,r=eigh(first)
            basis=u@v[:,group]
            density=amplitude[group]@amplitude[group].conj().T
            strength=float(np.trace(density).real)
            source_s2=float(np.trace(density@(basis.conj().T@(s2@basis))).real/strength)
            source_half=float(np.trace(density@(basis.conj().T@(half@basis))).real/strength)
            branches=[]
            for i,slope in enumerate(slopes):
                curve=float(np.vdot(r[:,i],second@r[:,i]).real)
                branches.append(dict(flux_slope=float(slope),flux_curvature=curve,
                                     same_reference_excess_inverse_ring_inertia=(curve-refcurve)*(3*p['spacing'])**2/p['charge']**2))
            lower=other[e[other]<e[group[0]]]
            # Deliberately wrong only-higher virtual truncation misses lower DARK intermediate states.
            b=v[:,group];a=v[:,lower].conj().T@((u.getH()@h1@u)@b)
            missed=2*a.conj().T@(a/(e[group[0]]-e[lower])[:,None])
            records.append(dict(number=number,momentum_index=momentum,target_momentum=target,
                                sector_floor_gap=float(e[0]-eref),physical_bright_gap=float(e[firstbright]-eref),
                                bright_multiplet_rank=len(group),source_weight=strength,source_weighted_S2=source_s2,
                                inclusive_source_halfspin_probability=source_half,all_other_virtual_count=len(other),
                                lower_dark_virtual_count=len(lower),wrong_higher_only_curvature_error=float(np.linalg.norm(missed)),
                                branches=branches))
    result=dict(kind='NEW_BRIGHT_PHYSICAL_SOURCE_MULTIPLET_AND_ALL_VIRTUAL_RING_CURVATURE',
                parameters=p,neutral_flux_slope=float(first0[0,0].real),neutral_flux_curvature=refcurve,
                reference_frame_S2=float(np.vdot(g,s2@g).real),records=records,
                scope='Actual bright multiplet, no source state filtered in physics; halfspin polynomial is inclusive receiver only. Analytic same-neutral-reference flux curvature, not one-cell family curvature, fitted mass or free3D inertia. m0 nonzero opposite velocities are not a rest minimum. Full geometry orbital/field NoEther remains absent.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
