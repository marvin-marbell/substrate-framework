"""Independent fresh-state actual MPO field/current consumer; no evolved input."""
from pathlib import Path
import json
import hashlib
import numpy as np
from tenpy.networks.mpo import MPOEnvironment
from rich_periodic_tensor_20261015_model import GaugeTensor, matrix_element
from rich_periodic_photon_20261015_model import geometry, source_geometry, loops
ROOT=Path(__file__).resolve().parent
P='rich_review_wave33_20261016_'


def apply(op, state):
    out=state.copy()
    op.apply_naively(out)
    out.canonical_form(renormalize=False)
    return out


def inner(a,b):
    return complex(a.overlap(b))


def run():
    model=GaugeTensor()
    seed=model.seed()
    image=apply(model.H,seed)
    psi=seed.add(image,1.,-.12)
    psi.canonical_form(renormalize=False)
    psi.norm/=np.sqrt(inner(psi,psi).real)
    hpsi=apply(model.H,psi)
    energy=matrix_element(psi,model.H,psi).real
    variance=inner(hpsi,hpsi).real-energy**2
    q=[1,1,0]
    geo,ops,contact=model.sources(q)
    field=apply(ops[1],psi)
    current=apply(ops[3],psi)
    # H(A) uses the source convention exp(+i direction coefficient A).
    coeff=geo['coefficients'][1]
    def twisted(a):
        terms=model.diag_terms+[(x*np.exp(1j*d*coeff[e]*a),t) for x,t,e,d in model.hop_terms]
        return model.mpo(terms)
    eps=1e-4
    ep=matrix_element(psi,twisted(eps),psi)
    em=matrix_element(psi,twisted(-eps),psi)
    first=(ep-em)/(2*eps)
    second=(ep+em-2*energy)/eps**2
    jmean=matrix_element(psi,ops[3],psi)
    kmean=matrix_element(psi,contact[1][1],psi)
    hf=apply(model.H,field)
    fh=apply(ops[1],hpsi)
    comm=hf.add(fh,1.,-1.)
    right=current.copy();right.norm*=1j/16
    residual=comm.add(right,1.,-1.)
    wrong=comm.add(right,1.,1.)
    # Nontrivial mixed bra/ket prefactors: old MPO contraction must fail.
    bra=current.copy();bra.norm*=2.7
    ket=current.copy();ket.norm*=.31
    identity=model.mpo([(1.,[('Id',0)])])
    exact=inner(bra,ket)
    fixed=matrix_element(bra,identity,ket)
    old=complex(MPOEnvironment(bra,identity,ket).full_contraction(0))
    positions,edges,B=geometry()
    loop=np.array(loops())
    geometry_results={'sites':len(positions),'arcs':len(edges),'incidence_rank':int(np.linalg.matrix_rank(B)),'cycle_rank':24-int(np.linalg.matrix_rank(B)),
        'loop_divergence_max':int(np.abs(loop@B.T).max()),'loop_lengths':{str(n):int(sum(np.count_nonzero(loop,axis=1)==n)) for n in [2,4]},
        'transverse_divergence':float(np.max(np.abs(B@geo['coefficients'][:2].T))),
        'density_longitudinal_error':float(np.linalg.norm(B@geo['coefficients'][2]-np.linalg.norm(geo['discrete_gradient'])*geo['density']))}
    findings={'kind':'OWN_FRESH_TENSOR_FIELD_CURRENT_CONTRACTION_NOT_GROUND_STATE','input':'fresh Neel seed then normalized (1-.12H)|seed>, not any stored evolved state',
        'energy':energy,'full_H_variance':variance,'field_norm2':inner(field,field).real,'current_norm2':inner(current,current).real,
        'field_current_overlap':[inner(field,current).real,inner(field,current).imag],
        'twist_first_derivative_error':float(abs(first-jmean)),'twist_contact_second_derivative':float(second.real),'contact':float(kmean.real),'twist_contact_error':float(abs(second-kmean)),
        'commutator_relation':'[H,F_T]=i J_T/16','commutator_residual_norm2':inner(residual,residual).real,'wrong_sign_residual_norm2':inner(wrong,wrong).real,
        'mixed_norm':{'overlap':[exact.real,exact.imag],'corrected':[fixed.real,fixed.imag],'old_unnormalized':[old.real,old.imag],'corrected_error':abs(fixed-exact),'old_error':abs(old-exact)},
        'geometry':geometry_results,
        'sources_SHA256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in [Path(__file__).name,'rich_periodic_tensor_20261015_model.py','rich_periodic_photon_20261015_model.py']}}
    (ROOT/(P+'tensor_findings.json')).write_text(json.dumps(findings,indent=2)+'\n')
    print(json.dumps(findings),flush=True)
    assert findings['mixed_norm']['corrected_error']<1e-9
    assert findings['mixed_norm']['old_error']>1e-4
    assert findings['commutator_residual_norm2']<1e-12
    assert findings['wrong_sign_residual_norm2']>1e-5
    assert findings['twist_contact_error']<1e-5
    assert geometry_results['density_longitudinal_error']<1e-12


if __name__=='__main__':run()
