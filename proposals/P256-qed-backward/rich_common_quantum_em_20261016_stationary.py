"""Fresh stationary Gauss vacuum and actual finite-q transverse collective data.
The clamped-matter stiffness is a diagnostic, NOT a substituted photon H.
"""
from pathlib import Path
import json,sys,time,resource
import numpy as np
from scipy import sparse,linalg
from scipy.sparse.linalg import eigsh
from rich_common_quantum_em_20261016_operator import Model,ChargeOperator


def dense_phi(operator):
    return(sparse.kron(operator.c0,sparse.eye(operator.nphi))+sparse.kron(operator.cphi,sparse.diags(operator.grid))+sparse.kron(sparse.eye(operator.n),operator.hphi)).toarray()


def stiffness(model,out):
    h=model.clamped([0.,0.]);values,vectors=linalg.eigh(dense_phi(h),check_finite=False);ground=vectors[:,0];first=np.column_stack([model.clamped([0.,0.],(a,))@ground for a in range(2)]);components=vectors.conj().T@first;contact=np.array([[np.vdot(ground,model.clamped([0.,0.],(a,b))@ground).real for b in range(2)]for a in range(2)]);ground_space=values-values[0]<=1e-10;excited=~ground_space;paramagnetic=-2*np.real(components[excited].conj().T@(components[excited]/(values[excited]-values[0])[:,None]));total=contact+paramagnetic;first_ground=np.asarray([vectors[:,ground_space].conj().T@(model.clamped([0.,0.],(a,))@vectors[:,ground_space])for a in range(2)])
    np.savez_compressed(out+'_clamped_full_virtual_stiffness.npz',energies=values,ground=ground,Hprime_ground_components=components,contact=contact,paramagnetic=paramagnetic,total=total,ground_space_first_order=first_ground)
    return{'angle_INPUT':[0.,0.],'matter_dimension_all_virtuals':len(values),'ground_energy_own_conditional':float(values[0]),'ground_multiplicity_numerical_1e-10':int(ground_space.sum()),'regular_adiabatic_Hessian':bool(ground_space.sum()==1),'contact':contact.tolist(),'all_finite_matter_virtual_paramagnetic':paramagnetic.tolist(),'total_at_zero_phase':total.tolist(),'total_eigenvalues':np.linalg.eigvalsh(total).tolist(),'not_used_as_added_H_or_Dyson_kernel':True,'interpretation':'Only the full selected finite-matrix matter law at clamped phase0. If ground degenerate, exported ground-space first-order matrix matters; total is reduced resolvent, NOT a unique scalar Hessian. No primitive/untruncated stiffness claim.'}


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=config['out'];started=time.time();model=Model(config['matter_prefix']);h=model.hamiltonian();Path(out+'_assembly.json').write_text(json.dumps(model.receipt,indent=2)+'\n');rng=np.random.default_rng(2221016);seed=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]);seed/=np.linalg.norm(seed)
    print(json.dumps({'stage':'fresh_NEW_joint_H_stationary_solve','dimension':h.shape[0],'eigenpairs':20}),flush=True)
    values,vectors=eigsh(h,k=20,ncv=48,which='SA',tol=2e-9,maxiter=15000,v0=seed);order=np.argsort(values);values=values[order];vectors=vectors[:,order];ground=vectors[:,0];hground=h@ground;residuals=[float(np.linalg.norm(h@vectors[:,i]-values[i]*vectors[:,i]))for i in range(len(values))];np.savez_compressed(out+'_states.npz',energies=values,states=vectors,residuals=residuals)
    print(json.dumps({'stage':'fresh_NEW_stationary_states_complete','energy':float(values[0]),'maximum_residual':max(residuals),'seconds':time.time()-started}),flush=True)
    x=ground.reshape(model.n,len(model.grid),len(model.flux));probes=[];names=[]
    for a in range(2):probes.append((x*model.flux[None,None,:,a]).reshape(-1));names.append('transverse_integer_E_generator_'+str(a))
    for a in range(2):probes.append(model.loop(ground,a));names.append('actual_geometric_Wilson_loop_sine_'+str(a))
    for species,edge in(('quark',0),('white',2)):probes.append(model.observer('current',species=species,edge=edge)@ground);names.append(species+'_physical_current_edge_'+str(edge))
    probes.append((x*model.grid[None,:,None]).reshape(-1));names.append('quantum_Phi')
    probe=np.column_stack(probes);means=ground.conj()@probe;centered=probe-ground[:,None]*means;complete_gram=centered.conj().T@centered;amplitudes=vectors.conj().T@centered;low_gram=amplitudes[1:].conj().T@amplitudes[1:];missing=complete_gram-low_gram;first_moment=centered.conj().T@(h@centered-values[0]*centered)
    np.savez_compressed(out+'_collective_response.npz',excitation_energies=values-values[0],probe_names=np.asarray(names),means=means,positive_frequency_residue_amplitudes=amplitudes,negative_frequency_residue_amplitudes=amplitudes.conj(),complete_equal_time_Gram=complete_gram,retained_20_state_Gram=low_gram,unresolved_equal_time_Gram=missing,complete_energy_weighted_Gram=first_moment)
    force_ward=[]
    for a in range(2):
        m=model.flux[None,None,:,a];commutator=1j*(h@(x*m).reshape(-1)-(hground.reshape(x.shape)*m).reshape(-1));force=model.observer('force',a)@ground;force_ward.append({'axis':a,'residual':float(np.linalg.norm(commutator-force)),'force_state_norm':float(np.linalg.norm(force))})
    matter=ChargeOperator(model.n,model.grid,model.hphi,model.flux,model.groups(),free=False);matter_ground=matter@ground;cut_commutators=[]
    for a in range(2):
        loop=model.loop(ground,a);comm=matter@loop-model.loop(matter_ground,a);electric=loop.reshape(x.shape)*model.field_energy[None,None,:]-model.loop((x*model.field_energy[None,None,:]).reshape(-1),a).reshape(x.shape);cut_commutators.append({'actual_loop':a,'matter_loop_commutator_cut_defect_norm':float(np.linalg.norm(comm)),'electric_kinetic_loop_commutator_norm':float(np.linalg.norm(electric)),'meaning':'Uncut coordinate multiplication operators commute. Nonzero first norm is an explicitly measured flux-boundary compression contribution, NOT induced Maxwell dynamics.'})
    matrix=x.reshape(-1,len(model.flux));rho=matrix.conj().T@matrix;rho_values=np.linalg.eigvalsh(rho);prob=abs(x)**2;body=prob.sum(axis=(1,2));qmean=body@model.q;qcov=np.einsum('n,na,nb->ab',body,model.q,model.q)-np.outer(qmean,qmean);readout=model.readout(ground)
    bare=[];where={tuple(m):i for i,m in enumerate(model.flux)}
    for loop in model.geometry['actual_cycle_profiles']:
        states=[]
        for sign in(-1,1):
            index=where.get(tuple(sign*np.asarray(loop)))
            if index is not None:states.append({'flux':model.flux[index].tolist(),'own_bare_energy':float(model.field_energy[index]),'sine_raw_residue':.25})
        bare.append(states)
    result={'kind':'NEW_GAUSS_COUPLED_RICH_MATTER_SPATIAL_TWO_CHANNEL_CONDITIONAL_MODEL','model':model.receipt,'geometry':{key:model.geometry[key]for key in('q','profile_gram','actual_cycle_profiles','actual_cycle_profile_rank','cover_map_finite_kernel_index','cover_kernel_NOT_quotiented','electric_flux_count')},'own_selected_sector_reference_energy':float(values[0]),'stationary_residuals':residuals,'reference':readout,'local_Q_covariance_Gauss_longitudinal_correlated':qcov.tolist(),'reduced_transverse_purity':float(np.trace(rho@rho).real),'reduced_transverse_entropy':float(-np.sum(rho_values[rho_values>0]*np.log(rho_values[rho_values>0]))),'field_force_current_Ward':force_ward,'flux_cut_coordinate_commutator_diagnostics':cut_commutators,'collective_probes':names,'finite_q_partial_stationary_spectrum':[{'excitation':float(values[i]-values[0]),'raw_residues':(abs(amplitudes[i])**2).tolist()}for i in range(1,len(values))],'complete_field_generator_Gram_eigenvalues':np.linalg.eigvalsh(complete_gram[:2,:2]).tolist(),'retained_low_field_generator_Gram_eigenvalues':np.linalg.eigvalsh(low_gram[:2,:2]).tolist(),'complete_loop_Gram_eigenvalues':np.linalg.eigvalsh(complete_gram[2:4,2:4]).tolist(),'unresolved_20_state_weight_diagonal':np.diag(missing).real.tolist(),'bare_free_rotor_loop_comparator':bare,'own_bare_rotor_first_positive_energy':float(np.min(model.field_energy[model.field_energy>1e-12])),'partial_spectrum_NOT_photon_claim':'Two genuine dynamical spatial-transverse coordinates, actual joint-H poles/residues and unresolved weight. NOT two input polarization labels relabelled photon branches; no spin1/linear dispersion/labEM conclusion. Positive and negative frequency Hermitian response legs exported, not RWA. All finite-H virtuals retained in Hamiltonian/time response; only lowest20 spectral poles listed.'}
    result['clamped_matter_induced_stiffness']=stiffness(model,out);result['resources']={'elapsed_seconds':time.time()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'NEW_stationary_collective_consumer_complete',**result['resources']}),flush=True)


if __name__=='__main__':main()
