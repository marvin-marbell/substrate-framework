"""Executable finite tensor-state construction and source-resolved variational response."""
from __future__ import annotations
import json
import pickle
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from tenpy.algorithms import dmrg
from rich_periodic_tensor_20261015_model import GaugeTensor, PREFIX, matrix_element, small_benchmark, encoded_charge_proof, physical_tensor_norm_benchmark

ROOT=Path(__file__).resolve().parent


def save(name, value):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def normalize(psi):
    psi.canonical_form(); psi.norm=1.
    return psi


def source_response(model, psi, energy, q, chi):
    geo, ops, contacts=model.sources(q)
    vectors=[]; means=[]; source_quality=[]
    for op in ops:
        mean=matrix_element(psi,op,psi); means.append(mean)
        v=psi.copy(); op.apply_naively(v)
        v=v.add(psi,1.,-mean)
        # Compression is explicit and its source-norm loss retained below.
        exact_norm=float(np.real(v.overlap(v)))
        trunc=v.compress_svd({'chi_max':chi,'svd_min':1e-12})
        vectors.append(v)
        source_quality.append({'exact_source_norm2':exact_norm,'compressed_source_norm2':float(np.real(v.overlap(v))),'discarded':float(trunc.eps)})
        print('source',q,len(vectors),exact_norm,float(np.real(v.overlap(v))),flush=True)
    G=np.array([[a.overlap(b) for b in vectors] for a in vectors])
    K=np.array([[matrix_element(a,model.H,b)-energy*G[i,j] for j,b in enumerate(vectors)] for i,a in enumerate(vectors)])
    hermiticity=float(np.linalg.norm(K-K.conj().T))
    G=(G+G.conj().T)/2; K=(K+K.conj().T)/2
    g,U=eigh(G); keep=g>max(1e-12,1e-10*g[-1]); Q=U[:,keep]/np.sqrt(g[keep])
    w,V=eigh(Q.conj().T@K@Q); C=Q@V
    overlaps=C.conj().T@G
    weights=np.einsum('ma,mb->mab',overlaps.conj(),overlaps)
    contact=np.array([[matrix_element(psi,contacts[p][r],psi) if contacts[p][r] is not None else 0. for r in range(2)] for p in range(2)])
    frequencies=np.array([0.,.5,1.,2.,3.,4.])
    eta=.05
    chi_full=np.array([sum((weights[m]/(z-w[m]+1j*eta)-weights[m].T/(z+w[m]+1j*eta)) for m in range(len(w))) for z in frequencies])
    loop_cov=[];loop_quality=[]
    for loop,count in model.loop_sources(q):
        mean=matrix_element(psi,loop,psi)
        v=psi.copy();loop.apply_naively(v);v=v.add(psi,1.,-mean)
        exact=float(np.real(v.overlap(v)))
        err=v.compress_svd({'chi_max':chi,'svd_min':1e-12})
        loop_cov.append([a.overlap(v) for a in vectors])
        loop_quality.append({'number_of_loops':count,'exact_norm2':exact,'compressed_norm2':float(np.real(v.overlap(v))),'discarded':float(err.eps)})
    loop_cov=np.asarray(loop_cov)
    np.savez_compressed(ROOT/(PREFIX+'q'+''.join(map(str,q))+'_response.npz'),gram=G,shifted_H=K,omega=w,coefficients=C,source_overlaps=overlaps,raw_joint_weights=weights,contact=contact,frequencies=frequencies,joint_chi=chi_full,current_twist_response=chi_full[:,2:,2:]+contact[None,:,:],source_means=np.array(means),winding_plaquette_joint_covariance=loop_cov)
    groups=[]
    for i,omega in enumerate(w):
        groups.append({'omega':float(omega),'F_T1':float(abs(overlaps[i,0])**2),'F_T2':float(abs(overlaps[i,1])**2),'j_T1':float(abs(overlaps[i,2])**2),'j_T2':float(abs(overlaps[i,3])**2)})
    return {'q_bits':q,'polarizations':geo['polarizations'].tolist(),'source_order':['F_T1','F_T2','jQ_T1','jQ_T2'],'source_gram_eigenvalues':g.tolist(),'source_rank':int(sum(keep)),'bands':groups,'source_quality':source_quality,'loop_order':['sine_Wilson_two_link_transverse_windings','sine_Wilson_four_link_plaquettes'],'loop_quality':loop_quality,'loop_joint_covariance_real':loop_cov.real.tolist(),'loop_joint_covariance_imag':loop_cov.imag.tolist(),'loop_scope':'Connected equal-time covariance, not an extra Hamiltonian term or a winding-free bulk dispersion. Projected insertion norm losses retained.','source_subspace':'Ritz in four compressed source-inserted MPS; NOT stationary full-H Lehmann poles','hermiticity_before_symmetry':hermiticity,'contact_real':contact.real.tolist(),'contact_imag':contact.imag.tolist()}


def run():
    start=time.time(); model=GaugeTensor()
    psi=model.seed()
    small=small_benchmark()
    initial={'benchmark':small,'charge_proof':encoded_charge_proof(),'MPO_bonds':model.H.chi,'seed_energy':float(model.H.expectation_value(psi).real),'seed_total_charge':psi.get_total_charge(only_physical_legs=True).tolist(),'library':'physics-tenpy1.1.1','method':'explicit JW, U(1) block-exact 32-node MPS/MPO; fresh Neel E0 then full-H image expansion; no old evolved state'}
    initial['actual_tensor_H_image_norm_benchmark']=physical_tensor_norm_benchmark(model)
    save('initial',initial); print(json.dumps(initial),flush=True)
    trajectory=[]
    # A finite polynomial in the FULL H seeds charge and flux bond sectors that
    # a two-node Gauss-symmetric product-seeded optimizer cannot create.
    step=model.step_mpo(.035)
    for k in range(1,9):
        err=step.apply(psi,{'compression_method':'zip_up','trunc_params':{'chi_max':32,'svd_min':1e-12}})
        normalize(psi)
        row={'image_step':k,'chi':max(psi.chi),'energy':float(model.H.expectation_value(psi).real),'discarded':float(err.eps),'charge':psi.get_total_charge(only_physical_legs=True).tolist()}
        trajectory.append(row); print(json.dumps(row),flush=True)
        if k==3:
            save('actual_tensor_checkpoint',{'kind':'ACTUAL_GAUGE_EXACT_MPO_IMAGE_MPS_PHYSICS','initial':initial,'trajectory':trajectory.copy(),'elapsed':time.time()-start})
    records=[]
    for chi in (32,64):
        if chi==64:
            # Explicit fresh full-H image expansion adds sectors at the larger bond.
            for _ in range(2):
                step.apply(psi,{'compression_method':'zip_up','trunc_params':{'chi_max':chi,'svd_min':1e-12}}); normalize(psi)
        model.H_MPO=model.H
        info=dmrg.run(psi,model,{'mixer':True,'max_N_sites_per_ring':32,'max_trunc_err':1.0,'max_sweeps':3,'min_sweeps':2,'max_E_err':1e-7,'trunc_params':{'chi_max':chi,'svd_min':1e-11},'lanczos_params':{'N_max':12},'combine':True})
        normalize(psi)
        E=float(model.H.expectation_value(psi).real)
        samples=[]
        rng=np.random.default_rng(222+chi)
        for _ in range(128):
            sigmas,amplitude=psi.sample_measurements(rng=rng)
            physical=[int(site.perm[int(s)]) for site,s in zip(model.sites,sigmas)]
            electric=[physical[i]-1 for i in model.field.values()]
            occ=[physical[i].bit_count() for i in model.matter.values()]
            gauss=np.asarray(occ)-1+np.asarray(__import__('rich_periodic_tensor_20261015_model').INCIDENCE)@electric
            samples.append({'physical_local_indices':physical,'amplitude_real':float(np.real(amplitude)),'amplitude_imag':float(np.imag(amplitude)),'nonzero_links':sum(e!=0 for e in electric),'Gauss_max':int(max(abs(gauss)))})
        save('chi'+str(chi)+'_samples',{'draws':samples,'outside_old_support_count':sum(s['nonzero_links']>5 for s in samples),'support_witness':'old up-to-four-hop support has <=4 nonzero links; old single-loop-plus-one-hop support <=5; any >5 is rigorously outside their union, independent of inherited evolved coefficients'})
        variance=float(model.H.variance(psi,exp_val=E))
        rec={'chi_max':chi,'actual_chi':max(psi.chi),'energy':E,'full_H_variance':variance,'full_H_residual':float(np.sqrt(max(variance,0))),'charge':psi.get_total_charge(only_physical_legs=True).tolist(),'rho':np.real(psi.expectation_value('Rho',sites=list(model.matter.values()))).tolist(),'rho2':np.real(psi.expectation_value('Rho2',sites=list(model.matter.values()))).tolist(),'electric2':np.real(psi.expectation_value('E2',sites=list(model.field.values()))).tolist(),'sweeps':len(info['sweep_statistics']['E'])}
        rec['DMRG_sweep_energy']=[float(x) for x in info['sweep_statistics']['E']]
        rec['DMRG_sweep_max_discarded']=[float(x) for x in info['sweep_statistics']['max_trunc_err']]
        rec['truncation_acceptance']='Exploratory bond-limited state permits up to1 discarded fraction; observed errors and FULL-H variance retained, no accuracy claim from successful process exit.'
        records.append(rec); save('state_checkpoint',{'trajectory':trajectory,'states':records}); print(json.dumps(rec),flush=True)
        with (ROOT/(PREFIX+'chi'+str(chi)+'_state.pkl')).open('wb') as f:pickle.dump(psi,f)
    responses=[source_response(model,psi,records[-1]['energy'],q,64) for q in ([1,0,0],[1,1,0])]
    off=model.seed(); offH=model.mpo(model.diag_terms)
    offE=float(offH.expectation_value(off).real)
    controls={'own_H_OFF_energy':offE,'own_H_OFF_full_variance':float(offH.variance(off,exp_val=offE)),'own_H_OFF_field_sources_zero':True,'own_H_OFF_current_and_contact_zero':'t=0 removes hopping derivative; not independent photon evidence','E_conservation':'OFF H is diagonal in every E; [Hoff,Ee]=0 exactly on full local dimensions','ON_small_CAR_Gauss':small}
    result={'kind':'FINITE_GAUGE_EXACT_TENSOR_STATE_AND_VARIATIONAL_JOINT_RESPONSE','input':json.loads((ROOT/(PREFIX+'input.json')).read_text()),'initial':initial,'trajectory':trajectory,'state_comparison':records,'responses':responses,'controls':controls,'limits':['Vacuum full-H residual is measured, not projected away. Response is finite source-MPS Ritz variational linearization, not exact stationary spectral function.','L2 has only pi momenta; no low-q/continuum/masslessness/propagating-photon inference.','Bond32/64 comparison is a material state-method diagnostic, not convergence proof.','No magnetic/Maxwell term and no fitted constants. Ecap1 raising operator nonunitary unchanged.'],'elapsed_seconds':time.time()-start}
    save('result',result); print('SCIENCE_COMPLETE',flush=True)


if __name__=='__main__':run()
