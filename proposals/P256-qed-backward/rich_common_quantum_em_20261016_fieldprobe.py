"""Actual holonomy landscape and source-resolved higher collective spectrum.
No forced zero-flux vacuum, no eigenstate-count verdict, no Dyson construction.
"""
from pathlib import Path
import gc,json,sys,time,resource
import numpy as np
from scipy import linalg,sparse
from scipy.sparse.linalg import eigsh,LinearOperator,cg
from rich_common_quantum_em_20261016_operator import Model
from rich_composite_geometry_20261015_factors import PhiHamiltonian


def nonbacktracking_support(model,ground,out):
    """Actual ordered support witnesses; full phase/resolvent sums ALL H orders."""
    paths={'triangle01_12_20':[(0,1),(1,1),(4,-1)],'square01_12_23_30':[(0,1),(1,1),(2,1),(3,-1)]};result=[]
    for category,name in((2,'down_quark'),(3,'charged_down_white')):
        for path_name,path in paths.items():
            state=ground.copy();circulation=np.zeros(2,int)
            for edge,direction in path:
                part=next(value for value in model.parts if value['edge']==edge and value['category']==category);c0=part['c0']if direction==1 else part['c0'].conj().T;cphi=part['cphi']if direction==1 else part['cphi'].conj().T;state=PhiHamiltonian(c0,cphi,np.zeros_like(model.hphi),model.grid)@state;circulation+=direction*np.asarray(part['shift'])
            elastic=np.vdot(ground,state);result.append({'charged_root':name,'nonbacktracking_path':path_name,'ordered_edges_and_directions':path,'actual_integer_field_circulation':circulation.tolist(),'inclusive_operator_output_norm2_NOT_probability':float(np.vdot(state,state).real),'elastic_ground_return_real':float(elastic.real),'elastic_ground_return_imag':float(elastic.imag),'support_units':'energy^(2*path_length), no mass/binding inference'})
    receipt={'actual_directed_loop_support':result,'no_extra_intermediate_count_mask':True,'fundamental_fixed_body_Phi_R_PW_cuts_still_apply':True,'meaning':'These are genuine nonbacktracking three/four-edge operator words, not two-edge U(1) backtracks. Support norm alone is NOT magnetic stiffness or a particle definition. All temporal CAR/sea orderings are in the separately calculated full-H phase/virtual response, not replaced by these particular ordered witnesses.'};Path(out+'_nonbacktracking_support.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def phase_landscape(model,out,resolvent=True,denominator=2):
    cycles=np.asarray(model.geometry['actual_cycle_profiles'],float);landscape=[];previous=None
    for holonomy in((2*np.pi*i/denominator,2*np.pi*j/denominator)for i in range(denominator)for j in range(denominator)):
        angles=np.linalg.solve(cycles,np.asarray(holonomy));h=model.clamped(angles);e,v=eigsh(h,k=2,which='SA',tol=2e-11,maxiter=30000,v0=previous);order=np.argsort(e);e=e[order];g=v[:,order[0]];previous=g;res=float(np.linalg.norm(h@g-e[0]*g));landscape.append({'integer_unit_cycle_holonomies_INPUT':list(holonomy),'angles':angles.tolist(),'own_clamped_energy':float(e[0]),'first_gap':float(e[1]-e[0]),'residual':res})
        print(json.dumps({'stage':'actual_holonomy_landscape','body_dimension':model.n,**landscape[-1]}),flush=True)
    Path(out+'_phase_landscape.json').write_text(json.dumps({'phase_landscape':landscape,'holonomy_grid_denominator':denominator,'phase_landscape_NOT_global_minimum_theorem':True},indent=2)+'\n')
    if not resolvent:return{'phase_landscape':landscape,'holonomy_grid_denominator':denominator,'phase_landscape_NOT_global_minimum_theorem':True}
    # Full implicit reduced resolvent at phase0: no finite excited-state window.
    h=model.clamped([0.,0.]);e,v=eigsh(h,k=2,which='SA',tol=2e-11,maxiter=30000,v0=previous);order=np.argsort(e);e=e[order];g=v[:,order[0]];first=np.column_stack([model.clamped([0.,0.],(a,))@g for a in range(2)]);means=g.conj()@first;rhs=first-g[:,None]*means
    loop_support=nonbacktracking_support(model,g,out)
    def action(x):return h@x-e[0]*x+g*np.vdot(g,x)
    a=LinearOperator(h.shape,matvec=action,rmatvec=action,dtype=np.dtype(complex));solutions=[];receipts=[]
    for axis in range(2):
        solution,info=cg(a,rhs[:,axis],rtol=2e-10,atol=0.,maxiter=max(3000,h.shape[0]*2));solutions.append(solution);receipts.append({'axis':axis,'cg_info':int(info),'true_residual':float(np.linalg.norm(a@solution-rhs[:,axis])),'all_finite_virtuals_via_full_H_resolvent':True})
        if info:raise RuntimeError('Full clamped virtual resolvent did not converge; preserved actual failure, no spectral/cut fallback')
    contact=np.array([[np.vdot(g,model.clamped([0.,0.],(aa,b))@g).real for b in range(2)]for aa in range(2)]);paramagnetic=-2*np.real(rhs.conj().T@np.column_stack(solutions));total=contact+paramagnetic;np.savez_compressed(out+'_clamped_resolvent.npz',ground=g,own_energies=e,Hprime_ground=first,solutions=np.column_stack(solutions),contact=contact,paramagnetic=paramagnetic,total=total)
    return{'phase_landscape':landscape,'holonomy_grid_denominator':denominator,'phase_landscape_NOT_global_minimum_theorem':True,'nonbacktracking_operator_support':loop_support,'phase0_stiffness':{'all_virtual_resolvents':receipts,'ground_first_gap':float(e[1]-e[0]),'contact':contact.tolist(),'paramagnetic':paramagnetic.tolist(),'total':total.tolist(),'eigenvalues':np.linalg.eigvalsh(total).tolist(),'regular_ground_required':'If first_gap is unresolved, matrix is not promoted to unique adiabatic Hessian.'}}


def supported_spectrum(model,reference_prefix,out,maximum=96):
    h=model.hamiltonian()
    with np.load(reference_prefix+'_states.npz')as z:ground=z['states'][:,0].copy();e0=float(z['energies'][0])
    x=ground.reshape(model.n,len(model.grid),len(model.flux));seed=[(x*model.flux[None,None,:,a]).reshape(-1)for a in range(2)];names=['integer_transverse_E_y','integer_transverse_E_z']
    coordinates=np.asarray([(2*np.pi*i/27,2*np.pi*j/27)for i in range(27)for j in range(27)]);fourier=np.exp(1j*model.flux@coordinates.T);phase_phi=[]
    for p in range(len(model.grid)):
        rho=x[:,p,:].conj().T@x[:,p,:];phase_phi.append(np.einsum('ij,ij->j',fourier.conj(),rho@fourier).real/27**2)
    phase_phi=np.asarray(phase_phi);np.savez_compressed(out+'_actual_phase_POVM.npz',angles=coordinates,Phi_values=model.grid,joint_Phi_phase_weights=phase_phi,phase_weights=phase_phi.sum(axis=0),electric_fluxes=model.flux)
    phase_summary={'definition':'Covariant Fourier phase POVM27x27, NOT an orthogonal position/phase PVM. Resolves all signed flux differences exactly in this cut. No phase recentering or forced zero-flux state.','total_weight':float(phase_phi.sum()),'marginal_phase_max_min_ratio':float(np.max(phase_phi.sum(axis=0))/np.min(phase_phi.sum(axis=0))),'minimum_raw_joint_weight':float(phase_phi.min())}
    for a in range(2):seed.append(model.loop(ground,a));names.append('geometric_loop_sine_'+str(a))
    for species,edge in(('quark',0),('white',2)):seed.append(model.observer('current',species=species,edge=edge)@ground);names.append(species+'_physical_current_'+str(edge))
    source=np.column_stack(seed);source-=ground[:,None]*(ground.conj()@source);q=np.zeros((h.shape[0],maximum),complex,order='F');hq=np.zeros_like(q,order='F');q[:,0]=ground;rank=1
    def append(vector):
        nonlocal rank
        value=np.array(vector,copy=True);original=float(np.vdot(value,value).real)
        for _ in range(2):value-=q[:,:rank]@np.conjugate(q[:,:rank].T@np.conjugate(value))
        norm=float(np.vdot(value,value).real)
        if original and norm>1e-24*original and rank<maximum:q[:,rank]=value/np.sqrt(norm);rank+=1
    for column in range(source.shape[1]):append(source[:,column])
    cursor=0
    while cursor<rank:
        hq[:,cursor]=h@q[:,cursor]
        if rank<maximum:append(hq[:,cursor])
        cursor+=1
        if cursor%16==0:print(json.dumps({'stage':'supported_joint_H_block_Krylov','acted_columns':cursor,'basis_rank':rank,'maximum_solver_window':maximum}),flush=True)
    q=q[:,:rank];hq=hq[:,:rank];small=np.column_stack([np.conjugate(q.T@np.conjugate(hq[:,i]))for i in range(rank)]);defect=float(np.linalg.norm(small-small.conj().T));energies,vectors=linalg.eigh(small,check_finite=False);coefficients=np.column_stack([np.conjugate(q.T@np.conjugate(source[:,i]))for i in range(source.shape[1])]);amplitudes=vectors.conj().T@coefficients;residuals=np.asarray([np.linalg.norm(hq@vectors[:,i]-energies[i]*(q@vectors[:,i]))for i in range(rank)]);gram=source.conj().T@source;reconstructed=amplitudes.conj().T@amplitudes;np.savez_compressed(out+'_supported_spectrum.npz',energies_relative_new_reference=energies-e0,residue_amplitudes=amplitudes,negative_frequency_amplitudes=amplitudes.conj(),Ritz_residuals=residuals,projected_H=small,orthonormal_source_coefficients=coefficients,complete_source_Gram=gram,reconstructed_source_Gram=reconstructed,source_names=np.asarray(names))
    peaks=[]
    for axis in range(2):
        selected=np.argsort(abs(amplitudes[:,axis])**2)[::-1][:8];peaks.append([{'relative_energy':float(energies[i]-e0),'raw_electric_residue':float(abs(amplitudes[i,axis])**2),'raw_other_probe_residues':(abs(amplitudes[i])**2).tolist(),'Ritz_residual':float(residuals[i])}for i in selected])
    return{'source_names':names,'actual_joint_state_phase_POVM':phase_summary,'solver_Krylov_rank':rank,'H_projection_Hermiticity_defect':defect,'complete_source_Gram_reconstruction_error':float(np.linalg.norm(gram-reconstructed)),'electric_source_dominant_supported_quadrature_poles':peaks,'interpretation':'Source-resolved block-Krylov quadrature of the SAME full finite joint H. Not a new physical state cut, not exact poles unless Ritz residual warrants it, not inferred photon branches. Both signed response legs saved. No mode-count conclusion from lowest20 eigenstates.'}


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=config['out'];start=time.time();model=Model(config['matter_prefix']);result={'model':model.receipt,'actual_cycle_field_circulation_rank_already_established':model.geometry['actual_cycle_profile_rank'],'phase_discriminator':phase_landscape(model,out,config['compute_full_clamped_resolvent'],config.get('holonomy_denominator',2))}
    if config.get('reference_prefix'):result['source_supported_collective_spectrum']=supported_spectrum(model,config['reference_prefix'],out,config['Krylov_solver_columns'])
    result['resources']={'seconds':time.time()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'actual_holonomy_supported_field_discriminator_complete',**result['resources']}),flush=True)


if __name__=='__main__':main()
