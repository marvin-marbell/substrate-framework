"""New native-charge/3D-EM physical consumer. Frame variant never imported."""
import hashlib,json,pickle,resource,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh,LinearOperator,gmres,expm_multiply
from rich_spatial_colour_charge_20261015_snapshot import load as load_body
from rich_spatial_colour_em_20261015_join import from_body,assemble,angular,native_source,native_C,native_charge_receiver
from rich_spatial_colour_em_20261015_model import FactorOperator,observable
from rich_spatial_colour_em_20261015_fields import profiles,phase_kernel
from rich_spatial_colour_em_20261015_spectrum import _total_M_operator
from rich_spatial_colour_em_20261015_density import density,spatial_current,mechanical_operator,no_hop_law
from rich_spatial_colour_em_20261015_measure import field_action,joint_receiver,field_halfline


def norm(x):return float(sparse.linalg.norm(x)) if sparse.issparse(x) else float(np.linalg.norm(x))
def mean(v,a):return float(np.vdot(v,a@v).real)
def cj(v):return dict(real=np.asarray(v).real.tolist(),imag=np.asarray(v).imag.tolist())


def reference(model,h,case):
    body=model['base'];field=model['field'];labels=np.rint(2*body['operators']['neutral']['S'][2].diagonal().real).astype(int);lr,ur=np.linalg.eigh(body['rotor']['L'][2].toarray());lf,uf=np.linalg.eigh(field['F'][2]);u=np.kron(ur,uf);rq=(np.rint(2*lr).astype(int)[:,None]+np.rint(2*lf).astype(int)[None,:]).ravel()
    transformed=FactorOperator(h.nm,h.scalar,h.nrq,[(a,u.conj().T@b.toarray()@u,p) for a,b,p in h.terms],h.include_scalar);reduced,selected=_total_M_operator(transformed,labels,rq,0)
    rng=np.random.default_rng(19015);x=rng.normal(size=len(selected))+1j*rng.normal(size=len(selected));x/=norm(x);full=np.zeros(h.shape[0],complex);full[selected]=x;image=transformed@full;outside=image.copy();outside[selected]=0
    e,v=eigsh(reduced,k=case['reference_eigenstates'],which='SA',tol=case['eigen_tolerance'],ncv=max(28,3*case['reference_eigenstates']),maxiter=24000,v0=rng.normal(size=len(selected)))
    q,s,_=np.linalg.svd(v,full_matrices=False);q=q[:,s>1e-9*s[0]];small=q.conj().T@(reduced@q);e,r=np.linalg.eigh((small+small.conj().T)/2);v=q@r;ground=np.zeros(h.shape[0],complex);ground[selected]=v[:,0];ground=np.einsum('ab,mfb->mfa',u,ground.reshape(h.nm,h.nf,h.nrq),optimize=True).ravel()
    return ground,dict(energy=float(e[0]),display_energies=e.tolist(),total_M0_dimension=len(selected),full_dimension=h.shape[0],M_leakage=norm(outside),eigen_residual=norm(h@ground-e[0]*ground),meaning='Fresh exact-total-M0 neutral solve of SAME extended H; all integer J represented, no fixed body M or halfspin constraint. Display count is not current-virtual cutoff.')


def propagate(h,state,time,energy):
    shifted=h-energy*sparse.eye(h.shape[0],format='csr');return expm_multiply(-1j*time*shifted,state,traceA=-1j*time*(h.trace()-energy*h.shape[0])) if time else state.copy()


def solve(h,e,rhs,w,eta,tol):
    z=e+w+1j*eta;op=LinearOperator(h.shape,matvec=lambda x:z*x-h@x,dtype=complex);d=z-h.diagonal();pre=LinearOperator(h.shape,matvec=lambda x:x/d,dtype=complex);x,flag=gmres(op,rhs,M=pre,rtol=tol,atol=tol*1e-2,restart=64,maxiter=200)
    if flag:raise RuntimeError('Full retained EM resolvent nonconvergence '+str(flag))
    return x,norm(op@x-rhs)


def measurements(model,name,state):
    base=model['base'];op=base['operators'][name];field=model['field'];nr=base['rotor']['dimension'];x=state.reshape(op['Q'].shape[0],len(base['scalar']['grid']),nr,field['dimension']);scalar=np.sum(abs(x)**2,axis=(0,2,3));answer={key:mean(state,observable(model,name,matter=op[key])) for key in ['rho0','rho1','N','Q','C','B','PW_boundary']};answer.update(phi2=float(scalar@(base['scalar']['grid']**2)),scalar_edge=float(scalar[0]+scalar[-1]),field_energy=mean(state,observable(model,name,rq=np.kron(np.eye(nr),field['H_E']))),field_edge=mean(state,observable(model,name,rq=np.kron(np.eye(nr),field['boundary']))))
    return answer


def main():
    inp,out=map(Path,sys.argv[1:3]);case=json.loads(inp.read_text());p=json.loads((inp.parent/case['owner_input']).read_text());start=time.monotonic();bodyfile=inp.parent/p['native_body_cache'];digest=hashlib.sha256(bodyfile.read_bytes()).hexdigest()
    if digest!=p['native_body_cache_sha256']:raise ValueError('Pinned native primitive BODY mismatch')
    body=load_body(bodyfile);model=from_body(body,p);base=model['base'];field=model['field'];rotor=base['rotor'];h=assemble(model,'neutral');ground,ref=reference(model,h,case);saved={'neutral_reference':ground};result=dict(kind='WAVE19_NEW_PRIMARY_NATIVE_CHARGE_QUANTUM_EM_NOT_FRAME_VARIANT',parameters=p,case=case,reference=ref)
    def checkpoint(stage):
        result.update(stage=stage,resource=dict(seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**saved)
    checkpoint('fresh_extended_reference');e=ref['energy'];nr=rotor['dimension'];ne=field['dimension'];rng=np.random.default_rng(1919);jops=[angular(model,'neutral',a) for a in range(3)];probe=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]);probe/=norm(probe)
    result['rotation']=dict(reference_J2=sum(norm(a@ground)**2 for a in jops),full_all_M_HJ_action_relative=[norm(h@(a@probe)-a@(h@probe))/max(1.,norm(h@probe)) for a in jops])
    flux=model['coefficients']['body'][1]-model['coefficients']['body'][0];full=phase_kernel(rotor,base['geometry'],model['coefficients'],field,1,2/3,spin_axis=2);separate=np.kron(rotor['integrate'](base['geometry']['tangents'][:,1,2]).toarray(),np.eye(ne))@phase_kernel(rotor,base['geometry'],model['coefficients'],field,1,2/3)
    result['kernel']=dict(raw_electric_Gram=model['coefficients']['raw_electric_gram'].tolist(),closed_TE_norm=norm(flux[:3]),closed_TM_norm=norm(flux[3:]),wrong_separate_rotation_phase_difference=norm(full-separate),phase_units='Physical q=(2/3,-1/3,0,-1) times actual normalized-profile route integral times canonical Q; neither integer3 nor material/colour generator is EM charge.',profile_meaning='Both TE/TM triplets nonblind geometry, not two propagating plane-wave photons.')
    coulomb={};source_results={};cn=native_C(model,'neutral');states={}
    for name,op in base['operators'].items():
        direct=(model['V_self']*(op['rho0']@op['rho0']+op['rho1']@op['rho1'])/2+model['V_cross']*(op['rho0']@op['rho1']+op['rho1']@op['rho0'])/2);formula=((model['V_self']+model['V_cross'])*(op['Q']@op['Q'])+(model['V_self']-model['V_cross'])*op['density_delta2'])/4
        coulomb[name]=dict(self_cross_formula_error=norm(formula-direct),full_D2_minus_compressed_D_squared=norm(op['density_delta2']-op['density_delta']@op['density_delta']),minimum_eigenvalue=float(np.linalg.eigvalsh(formula.toarray()).min()) if formula.shape[0]<700 else None)
    result['Coulomb']=dict(Vself=model['V_self'],Vcross=model['V_cross'],base_old_U1_skin='NONE; actual Gaussian self and cross ADDED once, Q_em only.',sectors=coulomb)
    for name,values in base['source_matter'].items():
        source_results[name]={};hsource=assemble(model,name)
        for key,mat in values.items():
            insertion=native_source(model,name,key);raw=insertion@ground;strength=norm(raw)**2;metric=base['source_metric_full'][name][key];gx=ground.reshape(metric.shape[0],-1);fullnorm=float(np.sum(gx.conj()*(metric@gx)).real);state=raw/np.sqrt(strength);J=float(key.split(':')[1]);angular2=sum(norm(angular(model,name,a)@state)**2 for a in range(3))
            source_results[name][key]=dict(full_raw_strength=fullnorm,retained_raw_strength=strength,signed_omitted_norm_squared=fullnorm-strength,preparation_cost=mean(state,hsource)-e,J2=angular2,source_tensor_J=J,source_metric_not_probability=True)
            if key==case['source_key']:states[name]=state;saved[name+'_raw_normalized_source']=state
        if not name.startswith('anti'):
            c=native_C(model,name);anti='anti'+name;key=case['source_key'];a=native_source(model,anti,key)@(cn@ground);b=c@(native_source(model,name,key)@ground);source_results[name]['direct_C_source_defect']=norm(a-b)
    result['sources']=source_results;checkpoint('fresh_full_sources_all_M')
    currents=[assemble(model,'neutral',derivative=(mu,)) for mu in range(6)];images=[a@ground for a in currents];contact=np.zeros((6,6));derivatives=[];seconds=[];step=case['derivative_step']
    for mu in range(6):
        ext=np.zeros(6);ext[mu]=step;hp=assemble(model,'neutral',external=ext);hm=assemble(model,'neutral',external=-ext);k=assemble(model,'neutral',derivative=(mu,mu));derivatives.append(norm((hp@ground-hm@ground)/(2*step)+images[mu])/max(1.,norm(images[mu])));seconds.append(norm((hp@ground+hm@ground-2*(h@ground))/step**2-k@ground)/max(1.,norm(k@ground)))
        for nu in range(mu,6):contact[mu,nu]=contact[nu,mu]=mean(ground,assemble(model,'neutral',derivative=(mu,nu)))
    green={};residuals={}
    for w in [case['frequency'],-case['frequency']]:
        values=np.zeros((6,6),complex);rr=[]
        for nu,rhs in enumerate(images):
            x,r=solve(h,e,rhs,w,case['eta'],case['resolvent_tolerance']);rr.append(r)
            for mu,left in enumerate(images):values[mu,nu]=np.vdot(left,x)
        green[w]=values;residuals[str(w)]=rr
    chi=green[case['frequency']]+green[-case['frequency']].conj();result['response']=dict(retained_all_M_virtual_dimension=h.shape[0],frequency=case['frequency'],eta=case['eta'],six_profile_current_Gram=cj([[np.vdot(a,b) for b in images] for a in images]),contact=contact.tolist(),retarded=cj(chi),effective_action_kernel=cj(chi+contact),linear_residuals=residuals,current_derivative_errors=derivatives,positive_Hsecond_errors=seconds,sign='Physical delta j=-(K+chi) external. Canonical profile derivative J not Noether or local E. Full six-profile signed inverses, not low-eigenstate virtual sum.')
    checkpoint('six_profile_full_response')
    event={};positive='lepton';times=case['times'];labels_pos=None;stored_positive=None;commonphase=None
    for name in ['lepton','antilepton']:
        initial=states[name];hs=assemble(model,name);off=assemble(model,name,coupled=False);hopoff=no_hop_law(model,name,hs);curr=[assemble(model,name,derivative=(mu,)) for mu in range(6)];nm=base['operators'][name]['Q'].shape[0];nf=len(base['scalar']['grid']);sign=-1 if name.startswith('anti') else 1;unitary=expm(-1j*sign*case['kick']*(field['Pi'][2]+.37*field['Pi'][4]));kicked=field_action(initial,nm,nf,nr,unitary);labels,effects=native_charge_receiver(model,name);records=[];xs=[];bs=[];receivers=[]
        initial_measure=measurements(model,name,initial);kicked_measure=measurements(model,name,kicked);phase=mean(initial,hs)
        for t in times:
            baseline=propagate(hs,initial,t,phase);x=propagate(hs,kicked,t,phase);bm=measurements(model,name,baseline);xm=measurements(model,name,x);joint=joint_receiver(x,effects,base['scalar'],rotor,field);records.append(dict(time=t,norm=norm(x)**2,excess={key:xm[key]-bm[key] for key in bm},current_excess=[mean(x,a)-mean(baseline,a) for a in curr],joint_probability_sum=float(joint.sum()),joint=joint.tolist()));xs.append(x);bs.append(baseline);receivers.append(joint)
        frozen_b=propagate(off,initial,times[-1],phase);frozen_x=propagate(off,kicked,times[-1],phase);fb=measurements(model,name,frozen_b);fx=measurements(model,name,frozen_x);hop_b=propagate(hopoff,initial,times[-1],phase);hop_x=propagate(hopoff,kicked,times[-1],phase);rho=observable(model,name,matter=base['operators'][name]['rho0']);routes=[assemble(model,name,route_derivative=edge) for edge in range(2)];graph=1j*(hs@(rho@initial)-rho@(hs@initial))+sum((a@initial for a in routes),np.zeros_like(initial));k=np.array(case['wavevector']);rhok=density(model,name,k);local=[spatial_current(model,name,k,a) for a in range(3)];wrong=[spatial_current(model,name,k,a,include_mechanical=False) for a in range(3)];change=1j*(hs@(rhok@initial)-rhok@(hs@initial));ward=change+1j*sum((ka*(a@initial) for ka,a in zip(k,local)),np.zeros_like(initial));wrongward=change+1j*sum((ka*(a@initial) for ka,a in zip(k,wrong)),np.zeros_like(initial))
        momentum=observable(model,name,rq=np.kron(np.eye(nr),field['Pi'][2]));electric=observable(model,name,rq=np.kron(np.eye(nr),field['H_E']));force=1j*(hs@(momentum@initial)-momentum@(hs@initial));bare=1j*(electric@(momentum@initial)-momentum@(electric@initial));first=mean(initial,hs)
        event[name]=dict(source_protocol='Nonstationary FULL normalized raw source on new coupled neutral reference, NOT a pole/added-particle position.',initial_energy_variance_norm=norm(hs@initial-first*initial),field_only_Pi_impulse='Unitary on field only; field Q/current can shift immediately. Matter-only t0 invariance checked, not instantaneous matter reaction.',initial_matter_changes={key:kicked_measure[key]-initial_measure[key] for key in ['rho0','rho1','N','Q','C','B','phi2']},initial_current_changes=[mean(kicked,a)-mean(initial,a) for a in curr],records=records,charge_labels=labels,frozen_all_EM_paths_off_excess={key:fx[key]-fb[key] for key in fb},frozen_probe_current_image_norms=[norm(assemble(model,name,derivative=(mu,),coupled=False)@initial) for mu in range(6)],all_hops_off=dict(material_charge_motion_norm=norm(1j*(hopoff@(rho@initial)-rho@(hopoff@initial))),mechanical_current_image_norms=[norm(mechanical_operator(model,name,derivative=(mu,))@initial) for mu in range(6)],laboratory_density_excess=cj(np.vdot(hop_x,rhok@hop_x)-np.vdot(hop_b,rhok@hop_b))),material_graph_Ward=norm(graph),spatial_Fourier_Ward=norm(ward),wrong_no_mechanical_Fourier_Ward=norm(wrongward),physical_current_vs_Pi_force_defect=norm(force-curr[2]@initial),bare_projected_Pi_squared_cut_force=norm(bare),receiver='Native physical charge PVM times scalar-sign bin times ONE continuous field-sign compressed effect. Field effect is not a finite sharp projector; no simultaneous sharp6Q or source-overlap postselection.')
        saved[name+'_event']=np.array(xs);saved[name+'_baseline']=np.array(bs);saved[name+'_kicked']=kicked
        if name=='lepton':stored_positive=(initial,kicked,xs,bs,receivers,labels)
        else:
            c=native_C(model,'lepton');pos,pk,px,pb,pr,pl=stored_positive;phase_align=np.vdot(initial,c@pos);phase_align/=abs(phase_align);order=[next(i for i,q in enumerate(pl) if abs(q+anti)<1e-10) for anti in labels];expected=np.array(pr)[:,order,:,::-1]
            event[name]['actual_C']=dict(initial_phase_aligned_defect=norm(phase_align*initial-c@pos),kick_defect=norm(phase_align*kicked-c@pk),all_M_event_defects=[norm(phase_align*x-c@y) for x,y in zip(xs,px)],inclusive_reversed_charge_field_TV=[float(np.sum(abs(a-b))/2) for a,b in zip(receivers,expected)])
        checkpoint('event_'+name)
    effect=field_halfline(field,2)[1];result.update(event=event,field_cut=dict(CCR_defect=norm(field['Q'][2]@field['Pi'][2]-field['Pi'][2]@field['Q'][2]-1j*np.eye(ne)),Q_noncommutation=norm(field['Q'][0]@field['Q'][2]-field['Q'][2]@field['Q'][0]),bare_E_spectrum=np.linalg.eigvalsh(field['H_E']).tolist(),one_sign_effect_nonidempotence=norm(effect@effect-effect)),limits='Separate primary EM H; no Frame kinetic/spin/state/gaps imported. Native15/30/118 family not proved nested bandpair1; field/rotor/PW/scalar/native cut defects not full closure, no photons/freeCOM/Nature. Primitive Pi² only, no supplied B² or fitted restoration. Own new full6 response at one frequency; unfiltered source events not stationary charged scattering.')
    checkpoint('complete');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
