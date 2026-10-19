"""Cheap actual-law test of proposed Z=PhiParity*(-1)^TOTAL N_{0,2}.
Z is NOT physical particle-hole charge conjugation C. Vector hops have no Phi.
Actual neutral fullPhi1806 Hamiltonian and raw dressed sources are used.
"""
from __future__ import annotations
from pathlib import Path
import json,resource,sys,time
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_factors import load_h,PhiHamiltonian
from rich_composite_geometry_20261015_cloud_probe import neutral_state,source,inner


def signs(counts):return -1. if sum(counts[4*site+category] for site in (0,2) for category in range(4))%2 else 1.


def z_state(primitive,state):
    return {key:signs(primitive.label(key)[:-1])*field[:,::-1] for key,field in state.items()}


def stats(primitive,state):
    transformed=z_state(primitive,state);norm=float(inner(state,state).real);overlap=inner(state,transformed)
    return {'raw_norm2':norm,'normalized_Z_expectation_real':float(overlap.real/norm),
        'normalized_Z_expectation_imag':float(overlap.imag/norm),
        'distance_to_Zplus_norm2':sum(float(np.linalg.norm(transformed[key]-field)**2) for key,field in state.items())/norm,
        'distance_to_Zminus_norm2':sum(float(np.linalg.norm(transformed[key]+field)**2) for key,field in state.items())/norm}


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    data=load(config['neutral_basis_snapshot']);primitive=primitive_for(data);p=data['p'];coupling=load(config['neutral_coupling_snapshot']);states=np.load(config['neutral_states_snapshot'])
    h=load_h(config['neutral_H_snapshot'])
    if not isinstance(h,PhiHamiltonian):raise TypeError('Actual declared fullPhi factor snapshot required, no substitute toy')
    n=h.n;nf=h.nphi;parity=np.array([signs(row['counts']) for row in coupling['labels']])
    d=sparse.diags(parity);reflection=sparse.csr_matrix((np.ones(nf),(np.arange(nf),np.arange(nf-1,-1,-1))),shape=(nf,nf))
    z=sparse.kron(d,reflection,format='csr')
    full=sparse.kron(h.c0,sparse.eye(nf),format='csr')+sparse.kron(h.cphi,sparse.diags(h.grid),format='csr')+sparse.kron(sparse.eye(n),sparse.csr_matrix(h.hphi),format='csr')
    defect=z@full@z-full;even_field_odd_count=(h.c0-d@h.c0@d)/2
    vector_counterexample=sparse.kron(even_field_odd_count,sparse.eye(nf),format='csr')
    omega_vector=states['states'][:,0];omega=neutral_state(data,coupling,omega_vector);zomega=z_state(primitive,omega)
    result={'kind':'ACTUAL_PHI_SUBLATTICE_Z_SYMMETRY_DISCRIMINATOR_NOT_CHARGE_CONJUGATION','parameters':p,
        'law':'Hhop=gPhi(F0+F0dag)+i v sum_ia R_i_a(Fi-Fidag), v=.6. Scalar hops are Phi-odd, vector hops Phi-EVEN. All quark AND white hops flip total fermion N_{0,2}; electric/Hphi/rigid/plaquette preserve it. Therefore Z H Z-H=-2 Hvector, not0 unless that actual term vanishes.',
        'domain':'ACTUAL own1806-state neutralJ0/Phi7/R10/PW19 model, no two-level reduction. Neutral family is a direct sum of material counts tensor ALL7Phi coordinates, so it is Z-invariant; a nonzero compressed defect is a genuine counterexample to full primitive Z symmetry, not an arbitrary noncommuting projection.',
        'actual_dimension':h.shape[0],
        'Z_squared_minus_I_Frobenius':float(sparse.linalg.norm(z@z-sparse.eye(h.shape[0]))),
        'full_H_Frobenius':float(sparse.linalg.norm(full)),
        'Z_H_Z_minus_H_Frobenius':float(sparse.linalg.norm(defect)),
        'Phi_even_count_odd_C0_component_Frobenius':float(sparse.linalg.norm(vector_counterexample)),
        'defect_plus_twice_even_hopping_Frobenius':float(sparse.linalg.norm(defect+2*vector_counterexample)),
        'scalar_Cphi_bipartite_anticommutator_Frobenius':float(sparse.linalg.norm(d@h.cphi@d+h.cphi)),
        'Phi_Hamiltonian_reflection_defect_Frobenius':float(np.linalg.norm(reflection@h.hphi@reflection-h.hphi)),
        'actual_Omega':stats(primitive,omega),
        'actual_H_Z_commutator_on_Omega_norm':float(np.linalg.norm(full@(z@omega_vector)-z@(full@omega_vector))),
        'sources':{}}
    for site in (0,1):
        spec=([site,site,site],'half_triplet',1);state=source(primitive,omega,spec);transformed=z_state(primitive,state)
        other=source(primitive,zomega,spec);sign=-1 if site in (0,2) else 1
        keys=state.keys()|other.keys();zero=np.zeros((10,nf),complex)
        error=sum(float(np.linalg.norm(transformed.get(key,zero)-sign*other.get(key,zero))**2) for key in keys)
        result['sources'][str(site)]={**stats(primitive,state),'Z_source_intertwining_norm2_error':error,'expected_creation_sign':sign,
            'meaning':'Actual RAW canonical source on dressedOmega, all finalPhi/R/PW kept; no native charged projection or assumption that Omega is a Phi-parity eigenstate.'}
    result['decision']='If observed defect is nonzero, withdraw the proposed simple Z-conservation explanation for this law. Physical particle-hole C symmetry/antiB is a DIFFERENT operator and is not falsified by this test. No assertion of another parity/5hop mechanism, Dirac doublet, binding, mobility no-go or rigid field-return gate.'
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
