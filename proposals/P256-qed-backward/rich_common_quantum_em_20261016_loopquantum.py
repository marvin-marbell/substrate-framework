"""Actual live charge-field quantum vacuum on the loop-rich occupied sea.
The measured clamped potential is NOT inserted into H. No first-law B/L state,
old solved state, fitted frequency or changed field cutoff is reused.
"""
from pathlib import Path
import json,sys,time,resource,gc
import numpy as np
from scipy.sparse.linalg import eigsh
from rich_common_quantum_em_20261016_operator import Model,ChargeOperator
from rich_common_quantum_em_20261016_fieldprobe import supported_spectrum


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=config['out'];start=time.time();model=Model(config['matter_prefix']);h=model.hamiltonian();rng=np.random.default_rng(2221017);seed=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]);seed/=np.linalg.norm(seed);print(json.dumps({'stage':'FULL_live_quantum_field_on_closed_occupied_sea','dimension':h.shape[0],'body':model.n,'Phi':len(model.grid),'field_states':len(model.flux),'solver_eigenpairs':4,'solver_ncv':24,'no_inserted_clamped_potential':True}),flush=True)
    energies,states=eigsh(h,k=4,ncv=24,which='SA',tol=2e-9,maxiter=20000,v0=seed);order=np.argsort(energies);energies=energies[order];states=states[:,order];ground=states[:,0].copy();residuals=[float(np.linalg.norm(h@states[:,i]-energies[i]*states[:,i]))for i in range(len(energies))];np.savez_compressed(out+'_states.npz',energies=energies,states=states,residuals=residuals);del states,seed;gc.collect();print(json.dumps({'stage':'loop_rich_FULL_quantum_reference_complete','own_energy':float(energies[0]),'residuals':residuals,'seconds':time.time()-start}),flush=True)
    x=ground.reshape(model.n,len(model.grid),len(model.flux));hg=h@ground;moments=np.column_stack([(x*model.flux[None,None,:,a]).reshape(-1)for a in range(2)]);means=ground.conj()@moments;moments-=ground[:,None]*means;hm=h@moments-energies[0]*moments;g0=moments.conj().T@moments;g1=moments.conj().T@hm;g2=hm.conj().T@hm;np.savez_compressed(out+'_full_H_electric_moments.npz',zero=g0,first=g1,second=g2,means=means)
    wards=[]
    for a in range(2):
        m=model.flux[None,None,:,a];comm=1j*(h@(x*m).reshape(-1)-(hg.reshape(x.shape)*m).reshape(-1));force=model.observer('force',a)@ground;wards.append({'axis':a,'same_H_source_derivative_force_state_norm':float(np.linalg.norm(force)),'field_generator_commutator_residual':float(np.linalg.norm(comm-force))})
    probability=abs(x)**2;body=probability.sum(axis=(1,2));qmean=body@model.q;qcov=np.einsum('n,na,nb->ab',body,model.q,model.q)-np.outer(qmean,qmean);flat=x.reshape(-1,len(model.flux));rho=flat.conj().T@flat;rho_eigen=np.linalg.eigvalsh(rho);readout=model.readout(ground);matter=ChargeOperator(model.n,model.grid,model.hphi,model.flux,model.groups(),free=False);mg=matter@ground;cut=[]
    for a in range(2):
        loop=model.loop(ground,a);matter_defect=matter@loop-model.loop(mg,a);kinetic_comm=loop.reshape(x.shape)*model.field_energy[None,None,:]-model.loop((x*model.field_energy[None,None,:]).reshape(-1),a).reshape(x.shape);cut.append({'geometric_loop':a,'matter_coordinate_commutator_boundary_defect_norm':float(np.linalg.norm(matter_defect)),'electric_kinetic_coordinate_commutator_norm':float(np.linalg.norm(kinetic_comm))})
    result={'kind':'NEW_LOOP_RICH_FULL_QUANTUM_GAUSS_REFERENCE_AND_TWO_TRANSVERSE_RESPONSE','model':model.receipt,'own_selected_neutral_quantum_energy':float(energies[0]),'lowest_four_energies':energies.tolist(),'stationary_residuals':residuals,'reference':readout,'local_Q_covariance_Gauss_correlated':qcov.tolist(),'reduced_transverse_purity':float(np.trace(rho@rho).real),'reduced_transverse_entropy':float(-np.sum(rho_eigen[rho_eigen>0]*np.log(rho_eigen[rho_eigen>0]))),'complete_full_H_electric_Gram_eigenvalues':np.linalg.eigvalsh(g0).tolist(),'complete_full_H_electric_frequency_centroids':(np.diag(g1).real/np.diag(g0).real).tolist(),'complete_full_H_electric_frequency_RMS':np.sqrt(np.diag(g2).real/np.diag(g0).real).tolist(),'field_force_source_Ward':wards,'flux_cut_coordinate_defects':cut,'same_physical_parameters_as_first_law':True,'clamped_response_only_provenance_NOT_inserted_H':config['induction_probe_result'],'scope':'Actual new-H joint quantum field/occupied-sea state. The complete first-law N36 B/L protocol is a DISTINCT finite matter representation and is NOT spliced into this state. No second-law added-pair dynamics claimed. Four low eigenstates support stationarity, NOT a mode-count cutoff; exact full-H moments and supported response retain all finite-H virtuals.'};Path(out+'_reference_result.json').write_text(json.dumps(result,indent=2)+'\n')
    del moments,hm,matter,mg,flat,rho,x,hg,probability,body,qmean,qcov,rho_eigen,comm,force,loop,matter_defect,kinetic_comm,ground;gc.collect();result['supported_collective_response']=supported_spectrum(model,out,out,maximum=config['source_Krylov_solver_columns']);result['resources']={'seconds':time.time()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'loop_rich_live_quantum_reference_and_collective_response_complete',**result['resources']}),flush=True)


if __name__=='__main__':main()
