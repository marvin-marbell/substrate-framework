"""Material second construction: coherent gauge-invariant winding-loop tensor gas.

The loop factors prepare a state, not a magnetic term in the Hamiltonian.
Their product represents exponentially many configurations via tensor bonds;
full-H image expansion and DMRG then relax that NEW state with unchanged law.
"""
import json
import pickle
import time
from pathlib import Path
import numpy as np
from tenpy.algorithms import dmrg
from rich_periodic_tensor_20261015_model import GaugeTensor, PREFIX, EDGES, INCIDENCE, physical_tensor_norm_benchmark
from rich_periodic_tensor_20261015_run import save, normalize, source_response
ROOT=Path(__file__).resolve().parent


def run():
    start=time.time(); model=GaugeTensor(); psi=model.seed(); chi=128
    save('loopgas_tensor_norm_benchmark',physical_tensor_norm_benchmark(model))
    seen=set(); loop_records=[]
    for e,(a,b,mu) in enumerate(EDGES):
        f=next(j for j,(c,d,nu) in enumerate(EDGES) if c==b and d==a and nu==mu)
        pair=tuple(sorted((e,f)))
        if pair in seen:continue
        seen.add(pair)
        U=np.diag([1.,1.],-1)
        terms=[(1.,[('Id',0)])]
        for T in (U,U.T):
            terms.append((.25,[(model.register(model.field[j],T),model.field[j]) for j in pair]))
        loop=model.mpo(terms)
        err=loop.apply(psi,{'compression_method':'zip_up','trunc_params':{'chi_max':chi,'svd_min':1e-12}});normalize(psi)
        loop_records.append({'links':pair,'discarded':float(err.eps),'chi':max(psi.chi)})
    seed_energy=float(model.H.expectation_value(psi).real)
    save('loopgas_checkpoint',{'kind':'ACTUAL_COHERENT_GAUGE_LOOP_GAS_TENSOR','factors':loop_records,'energy':seed_energy,'physical_charge':psi.get_total_charge(only_physical_legs=True).tolist(),'ansatz':'product over12 disjoint periodic2linkloops of [1+.25(UeUf+Ue^dagUf^dag)] acting fresh Neel E0, with stated tensor truncations; no new term in H'})
    print('ACTUAL_LOOP_GAS',seed_energy,flush=True)
    step=model.step_mpo(.035);trajectory=[]
    for k in range(8):
        err=step.apply(psi,{'compression_method':'zip_up','trunc_params':{'chi_max':chi,'svd_min':1e-12}});normalize(psi)
        row={'step':k+1,'energy':float(model.H.expectation_value(psi).real),'discarded':float(err.eps)};trajectory.append(row);print(row,flush=True)
    model.H_MPO=model.H
    info=dmrg.run(psi,model,{'mixer':True,'max_N_sites_per_ring':32,'max_trunc_err':1.,'max_sweeps':4,'min_sweeps':2,'max_E_err':1e-7,'trunc_params':{'chi_max':chi,'svd_min':1e-11},'lanczos_params':{'N_max':12},'combine':True})
    normalize(psi);E=float(model.H.expectation_value(psi).real)
    with (ROOT/(PREFIX+'loopgas_state.pkl')).open('wb') as f:pickle.dump(psi,f)
    variance=float(model.H.variance(psi,exp_val=E))
    samples=[];rng=np.random.default_rng(222128)
    for _ in range(256):
        sigmas,amp=psi.sample_measurements(rng=rng)
        physical=[int(site.perm[int(s)]) for site,s in zip(model.sites,sigmas)]
        field=np.array([physical[i]-1 for i in model.field.values()]);occ=np.array([physical[i].bit_count() for i in model.matter.values()])
        samples.append({'physical_local_indices':physical,'amplitude_real':float(amp.real),'amplitude_imag':float(amp.imag),'nonzero_links':int(sum(field!=0)),'Gauss_max':int(max(abs(occ-1+INCIDENCE@field)))})
    save('loopgas_samples',{'draws':samples,'outside_old_support_count':sum(s['nonzero_links']>5 for s in samples),'support_witness':'any >5 nonzero links rigorously outside old <=4hop OR one<=4linkloop+onehop union'})
    state={'chi':max(psi.chi),'energy':E,'full_H_variance':variance,'full_H_residual':float(np.sqrt(max(variance,0))),'physical_charge':psi.get_total_charge(only_physical_legs=True).tolist(),'rho':np.real(psi.expectation_value('Rho',sites=list(model.matter.values()))).tolist(),'rho2':np.real(psi.expectation_value('Rho2',sites=list(model.matter.values()))).tolist(),'electric2':np.real(psi.expectation_value('E2',sites=list(model.field.values()))).tolist(),'sweep_E':[float(x) for x in info['sweep_statistics']['E']],'sweep_discarded':[float(x) for x in info['sweep_statistics']['max_trunc_err']]}
    save('loopgas_state_result',state);print(json.dumps(state),flush=True)
    # Preserve first-method response arrays before writing selected new-loopgas arrays.
    responses=[source_response(model,psi,E,q,chi) for q in ([1,0,0],[1,1,0])]
    save('loopgas_result',{'kind':'MATERIAL_COHERENT_WINDING_GAS_SECOND_ANSATZ','state':state,'loop_factors':loop_records,'image_trajectory':trajectory,'responses':responses,'elapsed_seconds':time.time()-start,'limits':'Full-H residual and tensor source truncation finite; no stationary photon or lowq inference. Loop state preparation is NOT a B term.'})
    print('LOOP_GAS_SCIENCE_COMPLETE',flush=True)


if __name__=='__main__':run()
