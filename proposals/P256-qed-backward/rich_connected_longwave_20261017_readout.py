"""Read completed SAME-law trajectories: full81 vs8probes, contact, energy/limits.
No rerun, receiver move, source/parameter change or photon-pole assignment.
"""
from pathlib import Path
import json,sys,time
import numpy as np
from rich_connected_longwave_20261017_phase import Phase
from rich_connected_longwave_20261017_wave import Layout,Observers,moments


def contact(model,state,tangent,control):
    r,fields,a,p=state;dr,df,da,dp=tangent;ex=model.expectation_matrices(fields);dex=model.expectation_matrices(df);ext=np.r_[ex,1.+0j];dx=np.r_[dex,0.+0j];fac=ext[model.f1]*ext[model.f2];dfac=dx[model.f1]*ext[model.f2]+ext[model.f1]*dx[model.f2];phase=np.ones(len(model.terms),complex);dphase=np.zeros(len(model.terms),complex);good=model.angles>=0
    if not control:phase[good]=np.exp(1j*model.ns[good]*a[model.angles[good]]);dphase[good]=1j*model.ns[good]*da[model.angles[good]]*phase[good]
    z=model.coeff*phase*r[model.cols,model.rows]*fac;dz=model.coeff*((dphase*r[model.cols,model.rows]+phase*dr[model.cols,model.rows])*fac+phase*r[model.cols,model.rows]*dfac);hop=model.hop;k=np.bincount(model.edges[hop],weights=-2*(model.ns[hop]/3)**2*z[hop].real,minlength=96);dk=np.bincount(model.edges[hop],weights=-2*(model.ns[hop]/3)**2*dz[hop].real,minlength=96);return k,dk


def budget(model,state,tangent,control):
    r,fields,a,p=state;dr,df,da,dp=tangent;_,_,_,_,_,e,_,energy,ex=model.evaluate(*state,control=control);jet=model.jet(state,tangent,control=control)[2];de=jet['Eint'];el=(np.eye(96)-model.P)@e;et=model.P@e;dl=(np.eye(96)-model.P)@de;dt=model.P@de
    base={'EMtotal':.5*float(e@e),'EMlong':.5*float(el@el),'EMtransverse':.5*float(et@et),'total':energy};response={'EMtotal':float(e@de),'EMlong':float(el@dl),'EMtransverse':float(et@dt),'total':jet['energy']}
    for kind in('Phi','INTERNAL_R','PW19'):
        base[kind+'_bare']=float(sum(np.trace(q@f['bare']).real for q,f in zip(fields,model.fields)if f['kind']==kind));response[kind+'_bare']=float(sum(np.trace(q@f['bare']).real for q,f in zip(df,model.fields)if f['kind']==kind))
    base['all_nonEM']=base['total']-base['EMtotal'];response['all_nonEM']=response['total']-response['EMtotal'];return base,response


def disjoint_scalar_flows(model,state):
    # Exact flows of TWO actual disjoint white scalar hopping terms, with localPhi.
    choices={}
    for start,end,edge,sign in model.g['colour_tree']:
        if start//4==end//4:choices.setdefault(start//4,(start,end,edge))
    cells=sorted(choices)[:2]
    if len(cells)!=2:raise ArithmeticError('No two disjoint local tree terms')
    def flow(s,cell):
        rho,fields,a,p=s;rho=rho.copy();fields=[x.copy()for x in fields];start,end,_=choices[cell];ids=np.array([16*start+11,16*end+11]);phi=np.trace(fields[3*cell]@np.diag(model.phigrid)).real;expect=2*rho[ids[0],ids[1]].real;angle=.17*.65*phi;u=np.eye(256,dtype=complex);u[np.ix_(ids,ids)]=np.cos(angle)*np.eye(2)-1j*np.sin(angle)*np.array([[0,1],[1,0]]);rho=u@rho@u.conj().T;ph=np.exp(-1j*.17*.65*expect*model.phigrid);fields[3*cell]=ph[:,None]*fields[3*cell]*ph.conj()[None,:];return rho,fields,a.copy(),p.copy()
    ab=flow(flow(state,cells[0]),cells[1]);ba=flow(flow(state,cells[1]),cells[0]);de=model.T@(model.density(ab[0])-model.density(ba[0]))/3
    aa=np.array([[0,1,0],[1,0,0],[0,0,0]],float);bb=np.array([[0,0,1],[0,0,0],[1,0,0]],float)
    return {'native_disjoint_white_scalar_terms':[{'cell':c,'physical_tree_edge':choices[c][2],'localPhi_field':3*c}for c in cells],'finite_flow_time_DIAGNOSTIC_not_wave_parameter':.17,'full_species_algebra_rho_commutator_norm':float(np.linalg.norm(ab[0]-ba[0])),'independent_localPhi_flow_commutator_norm':float(max(np.linalg.norm(x-y)for x,y in zip(ab[1],ba[1]))),'native_fullE_flow_commutator_norm':float(np.linalg.norm(de)),'hardglobal_onepair_sharedvacuum_COUNTEREXAMPLE_commutator_norm':float(np.linalg.norm(aa@bb-bb@aa)),'scope':'Bounded representation discriminator on actual disjoint local scalar terms, not a quantum-validity theorem or a second propagation protocol. No globalPAP operation used in supplier.'}


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text());out=cfg['out'];started=time.time();model=Phase();wave=json.loads(Path(cfg['wave']+'_result.json').read_text());read=json.loads(Path(cfg['preparation']+'_result.json').read_text());raw=np.load(cfg['wave']+'_sample_observers.npz');times=raw['times'];f=wave['source_strength'];t=np.asarray(model.g['probe_profiles_cycle_projected']);gram=t.T@t;probe=t@np.linalg.pinv(gram)@t.T;near=model.P-probe;uniform=np.asarray(model.g['uniform_profiles']);pu=uniform@np.linalg.pinv(uniform.T@uniform)@uniform.T;projectors={'full_cycle':model.P,'chosen8_probe':probe,'current_coupled_complement':near,'uniform_inside_full81':pu};arrays={'profiles':t,'probe_Gram':gram,**projectors};near_stats={};energy_stats={};spectra={};frequencies=np.array([0.,1.,2.,4.,8.,16.])
    for name in('full','active_cotree_Uidentity'):
        be=raw[name+'_base_Efull_edge'];re=raw[name+'_response_per_unit_source_Efull_edge'];bn=np.einsum('ab,mtb->mta',near,be);rn=np.einsum('ab,mtb->mta',near,re);arrays[name+'_nearfield_base']=bn;arrays[name+'_nearfield_response_unit']=rn;near_stats[name]=moments(bn,rn,f);bem=raw[name+'_base_cell_electric_energy'].sum(axis=-1);rem=raw[name+'_response_per_unit_source_cell_electric_energy'].sum(axis=-1);bt=raw[name+'_base_energy'];rt=raw[name+'_response_per_unit_source_energy'];energy_stats[name]={'EM':moments(bem,rem,f),'all_other_rich_matter_fields':moments(bt-bem,rt-rem,f),'total':moments(bt,rt,f)};source=raw[name+'_response_per_unit_source_q_ET'];trans=[]
        for omega in frequencies:trans.append(np.trapezoid(source*np.exp(1j*omega*times)[None,:,None],times,axis=1))
        trans=f*np.array(trans);arrays[name+'_finite_time_transform_qET']=trans;spectra[name]={'frequencies_INPUT_READOUT_GRID_NOT_photon_dispersion':frequencies.tolist(),'mean_real':trans.mean(axis=1).real.tolist(),'mean_imag':trans.mean(axis=1).imag.tolist(),'finite_window_and_sparse_time_quadrature_NOT_poles':True}
    # Local closed-cycle circulation cancels Helmholtz projection tails exactly.
    loops=[];loop_cells=[]
    for cell in range(4):
        for word in model.g['triangle_words']:
            if not all(model.g['links'][e]['start']//4==cell and model.g['links'][e]['end']//4==cell and model.g['links'][e]['image']==[0,0,0]for e,_ in word):continue
            row=np.zeros(96)
            for edge,sign in word:row[edge]+=sign
            loops.append(row);loop_cells.append(cell)
    triangles=np.array(loops);arrays['local_tetra_triangle_cycles']=triangles;arrays['local_tetra_triangle_cells']=np.array(loop_cells);circulation={}
    for name in('full','active_cotree_Uidentity'):
        be=raw[name+'_base_Efull_edge'];re=raw[name+'_response_per_unit_source_Efull_edge'];bc=np.einsum('la,mta->mtl',triangles,be);rc=np.einsum('la,mta->mtl',triangles,re);arrays[name+'_local_Ecirculation_base']=bc;arrays[name+'_local_Ecirculation_response_unit']=rc
        initial_current=raw[name+'_response_per_unit_source_current_edge'][:,0];flow=initial_current if name=='full'else initial_current@(model.T@model.d).T;tail=flow@model.P.T
        circulation[name]={'closed_local_cycle_response':moments(bc,rc,f),'source_initial_full_Edot_edge_mean':(f*flow.mean(axis=0)).tolist(),'source_initial_ETdot_Helmholtz_tail_edge_mean':(f*tail.mean(axis=0)).tolist(),'oppositecell2_local_cycle_initial_Edot_max':float(np.max(abs(flow@triangles[np.array(loop_cells)==2].T))),'meaning':'Full-H Ephysdot=Jphys; activecontrol Ephysdot=T D Jphys (charge-dependenttree route). HelmholtzP is spatially NONLOCAL even with all81directions: distant ET alone may be a projection tail compensated by EL. Local closed-cycle Ecirculation has zeroEL and no such gradient tail. These are generator-owned derivatives, not forced controlzero.'}
    paired=arrays['full_local_Ecirculation_response_unit']-arrays['active_cotree_Uidentity_local_Ecirculation_response_unit']
    local={'triangle_cells':loop_cells,'triangle_count':len(loops),'native_closed_cycle_Gauss_residual':float(np.linalg.norm(triangles@model.d.T)),'cycles_inside_fullP_residual':float(np.linalg.norm(triangles@model.P-triangles)),'circulation':circulation,'paired_full_minus_control_source_mean':(f*paired.mean(axis=0)).tolist(),'paired_MonteCarlo_standard_error':(f*paired.std(axis=0,ddof=1)/np.sqrt(len(paired))).tolist(),'limits':'Electric circulation is a local graph observable, not MaxwellB or onephoton. Source/operator/cell2receiver unchanged; do not infervelocity from transverse projector tails.'}
    layout=Layout(model)
    with np.load(cfg['wave']+'_native_initial_sample0.npz')as initial:local['native_disjoint_scalar_flow_probe']=disjoint_scalar_flows(model,layout.unpack(initial['state_and_source_tangent'][:layout.size]))
    Path(out+'_local_circulation.json').write_text(json.dumps(local,indent=2)+'\n')
    budgets={}
    for control,name in((False,'full'),(True,'active_cotree_Uidentity')):
        with np.load(cfg['wave']+'_'+name+'_sample0_states.npz')as data:states=data['state_and_source_tangent'];baseline=[];response=[];contacts=[];dcontacts=[];ratios=[]
        for y in states.T:
            s=layout.unpack(y[:layout.size]);ds=layout.unpack(y[layout.size:]);b,r=budget(model,s,ds,control);baseline.append(b);response.append(r);c,dc=contact(model,s,ds,control);contacts.append(c);dcontacts.append(dc);ratios.append(float(f*np.linalg.norm(ds[0])/np.linalg.norm(s[0])))
        arrays[name+'_sample0_contact']=np.array(contacts);arrays[name+'_sample0_contact_response_unit']=np.array(dcontacts);budgets[name]={'baseline':baseline,'source_derivative_per_unitf':response,'f_times_fermion_tangent_over_baseline_norm':ratios,'scope':'ONE authorquasitrajectory, not ensemble quantumenergy or exactstatefidelity. No numericalclipping.'}
    np.savez_compressed(out+'_native_readout.npz',times=times,frequencies=frequencies,**arrays);result={'kind':'COMPLETED_CONNECTED_RICH_SPATIAL_RESPONSE_INTERPRETATION','wave':cfg['wave'],'preparation':cfg['preparation'],'all81cycles_retained':True,'actual_probe_rank':int(np.linalg.matrix_rank(gram,tol=1e-10)),'complement_rank':int(np.linalg.matrix_rank(near,tol=1e-9)),'uniform_rank_INSIDE81':int(np.linalg.matrix_rank(pu,tol=1e-10)),'uniform_outside_full_cycle_norm':float(np.linalg.norm((np.eye(96)-model.P)@uniform)),'source_retained_projector_difference_from_full_cycle':float(np.linalg.norm(model.K@np.linalg.solve(model.G,model.K.T)-model.P)),'profile_projector_NOT_applied_to_dynamics':True,'nearfield_response':near_stats,'energy_exchange':energy_stats,'finite_window_supported_response':spectra,'sample0_energy_and_contact':budgets,'initial_native_joint_current':read['native_current_images'],'actual_prepared_not_stationary':{'fermion_commutator_norm':read['fermion_stationarity_residual'],'local_field_residuals':read['local_field_stationarity_residuals'],'warning':'Inhomogeneous hotpreparedstate, not vacuum. Matchedfull/control nonlinear drift carried. Legalq is not stationaryBlochdispersion.'},'runtime_limits':{'samples':wave['samples'],'times':wave['times'],'weak_source':f,'nonlinear_Of2_source_terms_omitted':True,'TWA_and_finite_quantum_field_Wigner_higher_moments_not_exact':True,'no_photon_velocity_poles_lifetime_or_fullquantum_entanglement_claim':True},'resources':{'seconds':time.time()-started}};Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'ACTUAL_spatial_response_readout_complete','probe_rank':result['actual_probe_rank'],'complement_rank':result['complement_rank'],'uniform_rank':result['uniform_rank_INSIDE81'],'seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
