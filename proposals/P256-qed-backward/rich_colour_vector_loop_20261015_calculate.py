"""NEW whole-law source poles, rotor Noether symmetry, EM and field reactions."""
from __future__ import annotations
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh, expm_multiply, spsolve
from rich_colour_vector_loop_20261015_model import (
    build, law, clean, whole_observable, expanded_state, rotation_audit,
    cut_residuals, orientation, frame, mode, string)
from rich_quantum_common_field_20261015_sources import source_terms


def mean(state,op):
    return float(np.vdot(state,op@state).real)


def norm(x):
    return float(sparse.linalg.norm(x)) if sparse.issparse(x) else float(np.linalg.norm(x))


def complex_json(x):
    return dict(real=np.asarray(x).real.tolist(),imag=np.asarray(x).imag.tolist())


def save_sparse(stored,name,matrix):
    matrix=sparse.csr_matrix(matrix)
    for key in ['data','indices','indptr']:
        stored[name+'_'+key]=getattr(matrix,key)
    stored[name+'_shape']=matrix.shape


def lowest(h,count):
    k=min(count,h.shape[0]-2); start=np.random.default_rng(222).normal(size=h.shape[0])
    e,v=eigsh(h,k=k,which='SA',tol=1e-11,ncv=min(h.shape[0],max(4*k,80)),v0=start,maxiter=12000)
    # Complex ARPACK vectors need not be orthogonal inside exact degeneracies.
    # An orthonormal Ritz basis repairs the eigensolver, not the physical state.
    u,s,_=np.linalg.svd(v,full_matrices=False)
    basis=u[:,s>1e-10*s[0]]
    small=basis.conj().T@(h@basis)
    energies,rotation=np.linalg.eigh((small+small.conj().T)/2)
    return energies,basis@rotation


def dynamics(state,h,times,obs):
    current=state.copy(); previous=0.; states=[]
    for t in times:
        if t!=previous:
            current=expm_multiply((-1j*(t-previous))*h,current)
        states.append(current.copy()); previous=t
    states=np.column_stack(states)
    return states,{name:[mean(states[:,i],operator) for i in range(len(times))] for name,operator in obs.items()}


def full_source_strength(reference,neutral,ng,rotor,scalar,kind,spin,removal):
    terms=source_terms(kind,spin)
    if removal:
        converted=[]
        for operations,amplitude in terms:
            mapped=[]
            for index,_ in operations:
                m=(index%16)//8;s=(index%8)//4
                amplitude*=1 if m==s else -1;mapped.append((index^12,False))
            converted.append((mapped,amplitude))
        terms=converted
    targets=set(); entries=[]
    for j,mask in enumerate(neutral['masks']):
        for operations,amplitude in terms:
            step=string(mask,operations)
            if step is not None:
                targets.add(step[0]);entries.append((step[0],j,amplitude*step[1]))
    targets=sorted(targets);lookup={mask:i for i,mask in enumerate(targets)}
    source=sparse.csr_matrix(([v for m,j,v in entries],([lookup[m] for m,j,v in entries],[j for m,j,v in entries])),shape=(len(targets),len(neutral['masks'])))
    tensor=expanded_state(reference,neutral,ng,rotor,scalar)
    raw=neutral['P']@tensor.reshape(neutral['dimension'],-1)
    image=sparse.kron(source,sparse.eye(19),format='csr')@raw
    return dict(full_unprojected_CAR_strength=norm(image)**2,target_raw_masks=len(targets),
        original_mirror_assignments=2 if kind=='lepton' else 8,
        meaning='Both Hadamard components, including occupied-hole filling, act on actual interacting reference. No old raw .5/.25 strengths or source rescaling inherited.')


def inclusive_pvm(state,sector,geo,rotor,scalar,baseline):
    tensor=expanded_state(state,sector,geo,rotor,scalar)
    raw=(sector['P']@tensor.reshape(sector['dimension'],-1)).reshape(len(sector['masks']),19,rotor['dimension'],len(scalar['grid']))
    weight=abs(raw)**2
    pairlabels=np.array([sum(bool(mask&(1<<mode(f,0,s,c))) for f in range(2) for s in range(2) for c in range(4))-baseline for mask in sector['masks']])
    ls=np.array([l for l,m in rotor['basis']]); classes=[range(1),range(1,10),range(10,19)]; joint={}
    for pair in sorted(set(pairlabels)):
        for l in sorted(set(ls)):
            for a,indices in enumerate(classes):
                joint[f'pairs={pair};l={l};PW={a}']=float(weight[pairlabels==pair][:,list(indices)][:,:,ls==l].sum())
    return dict(joint_pair_orientation_PW_probabilities=joint,inclusive_sum=float(sum(joint.values())),
        scalar_grid_probabilities=weight.sum(axis=(0,1,2)).tolist(),
        meaning='Complete retained-Hilbert PVM marginals/joint outcomes, not probabilities inferred by normalizing source insertion strengths. PW0=singlet,1=3x3*,2=3*x3.')


def retarded(h,current,contact,state,frequencies,eta):
    estate=mean(state,h); image=current@state; ident=sparse.eye(h.shape[0],format='csr',dtype=complex); answer=[]; residuals=[]
    for w in frequencies:
        first=(w+estate+1j*eta)*ident-h
        second=(w-estate+1j*eta)*ident+h
        x=spsolve(first.tocsc(),image); y=spsolve(second.tocsc(),image)
        answer.append(np.vdot(image,x-y))
        residuals.append([norm(first@x-image),norm(second@y-image)])
    contact_mean=mean(state,contact)
    return dict(reference_energy=estate,retained_virtual_dimension=h.shape[0],spectral_JJ_strength=norm(image)**2,
        contact=contact_mean,frequencies=frequencies,eta=eta,retarded=complex_json(answer),
        kernel_with_contact=complex_json(np.array(answer)+contact_mean),linear_solve_residuals=residuals,
        meaning='Full sparse resolvents over EVERY retained eigenstate, both retarded orders and signed energy differences. No lowest-eigenstate cutoff in this response.')


def kernel_symmetries(rotor):
    fr=frame();mass=fr['mass'];gamma=fr['gamma'];iy=1j*fr['sigma'][1]
    t=np.kron(np.diag([1.,-1.]),iy);b=np.kron(iy,iy)
    parity=rotor['parity'];tr=rotor['T']
    eps=lambda a,b,c:(0 if len({a,b,c})<3 else 1 if (a,b,c) in [(0,1,2),(1,2,0),(2,0,1)] else -1)
    cov=[]
    for a in range(3):
        for c in range(3):
            cov.append(norm(rotor['L'][a]@rotor['n'][c]-rotor['n'][c]@rotor['L'][a]-1j*sum(eps(a,c,d)*rotor['n'][d] for d in range(3))))
    return dict(orientation_vector_commutator=max(cov),mass_vector_anticommutators=[norm(mass@g+g@mass) for g in gamma],
        P_frame_odd=[norm(mass@g@mass+g) for g in gamma],P_path_odd=[norm(parity@n@parity+n) for n in rotor['n']],
        T_frame_odd=[norm(t@g.conj()@t.conj().T+g) for g in gamma],T_path_even=[norm(tr@n.conjugate()@tr.conj().T-n) for n in rotor['n']],
        C_frame_odd=[norm(-b@g.T@b.conj().T+g) for g in gamma],
        interpretation='P: original mirror X (rotated Z), polar path n->-n, U unchanged on parity-transformed directed path; Wilson flux A is a scalar holonomy. C: Fock particle-hole plus i tau_y i sigma_y and PW block conjugation U->U*, A->-A, n unchanged. Transposition of colour bilinear makes oriented U-Udag odd, compensating C-odd Gamma. T: antiunitary (Zmirror tensor i sigma_y)K, orientation Y_lm conjugation, loop/scalar K; Gamma and (U-Udag)/(2i) each odd, n even. No P/C/T label substituted for actual C-law or physical rotation checks.')


def main():
    inp,out=map(Path,sys.argv[1:3]);p=json.loads(inp.read_text());started=time.time();model=build(p);params=model['parameters']
    rotor=model['rotor'];scalar=model['scalar'];nf=len(scalar['grid']);stored={};results={};laws={};spectra={};observables={}
    for name,sector in model['sectors'].items():
        op=model['operators'][name];geo=model['geometry'][name]
        h,j,k=law(sector,op,geo,rotor,scalar,params);laws[name]=(h,j,k)
        e,v=lowest(h,p['low_eigenstates']);spectra[name]=(e,v)
        obs={key:whole_observable(op[key],geo,rotor,scalar) for key in ['positive','pairs','N','Q','S2','C','B','boundary','matter_octet']}
        obs.update(J2=clean(sparse.kron(geo['J2'],sparse.eye(nf),format='csr')),
            L2=clean(sparse.kron(geo['L2'],sparse.eye(nf),format='csr')),
            orientation_boundary=clean(sparse.kron(geo['orientation_boundary'],sparse.eye(nf),format='csr')),
            phi=clean(sparse.kron(sparse.eye(geo['dimension']),sparse.csr_matrix(scalar['phi']),format='csr')),
            phi2=clean(sparse.kron(sparse.eye(geo['dimension']),sparse.csr_matrix(scalar['phi']@scalar['phi']),format='csr')),
            scalar_boundary=clean(sparse.kron(sparse.eye(geo['dimension']),sparse.csr_matrix(scalar['boundary']),format='csr')),
            current=j,contact=k)
        observables[name]=obs
        j2means=[mean(v[:,i],obs['J2']) for i in range(len(e))]
        results[name]=dict(dimension=h.shape[0],matter_gauge_dimension=sector['dimension'],raw_CAR_masks=len(sector['masks']),
            total_M=0. if name=='neutral' else .5,ground_energy=float(e[0]),ground={key:mean(v[:,0],value) for key,value in obs.items()},
            low_energies=e.tolist(),low_J2=j2means,eigen_residuals=[norm(h@v[:,i]-e[i]*v[:,i]) for i in range(len(e))],
            low_eigenvector_orthogonality=norm(v.conj().T@v-np.eye(v.shape[1])),
            gauss_error=sector['gauge_error'],raw_vector_Gauss_image_errors=op['gauss_vector'],
            H_hermiticity=norm(h-h.conj().T),J_hermiticity=norm(j-j.conj().T),K_hermiticity=norm(k-k.conj().T),
            H_J2_commutator=norm(h@obs['J2']-obs['J2']@h),H_frame_S2_commutator=norm(h@obs['S2']-obs['S2']@h),
            H_rotated_population_commutator=norm(h@obs['positive']-obs['positive']@h),
            physical_rotation=rotation_audit(sector,op,rotor,params),Sz_offdiagonal=geo['Sz_offdiagonal'])
        for key,value in [('H',h),('current',j),('contact',k),('P',sector['P'])]:save_sparse(stored,name+'_'+key,value)
        for key,value in obs.items():save_sparse(stored,name+'_observable_'+key,value)
        stored[name+'_low_eigenvalues']=e;stored[name+'_low_eigenvectors']=v
        stored[name+'_masks']=np.array(sector['masks'],np.uint64);stored[name+'_M_indices']=geo['indices']
    e0,v0=spectra['neutral'];reference=v0[:,0];stored['reference']=reference
    sources={};receivers={'neutral':reference};pole_states={}
    for name,matrices in model['sources'].items():
        e,v=spectra[name];sources[name]={};removal=name.startswith('anti');kind='lepton' if 'lepton' in name else 'baryon'
        for spin,source in matrices.items():
            state=source@reference;strength=norm(state)**2;weights=abs(v.conj().T@state)**2
            bright=np.flatnonzero(weights>p['bright_tolerance']*max(strength,1e-30))
            if not len(bright):raise RuntimeError('No source-bright state in computed low spectrum: '+name+'/'+spin)
            energy=float(e[bright[0]]);pole=abs(e-energy)<p['degeneracy_tolerance']
            receiver=v[:,pole]@(v[:,pole].conj().T@state);receiver/=norm(receiver)
            full=full_source_strength(reference,model['sectors']['neutral'],model['geometry']['neutral'],rotor,scalar,kind,float(spin),removal)
            sources[name][spin]=dict(raw_strength=strength,full_CAR=full,source_projection_strength_loss=full['full_unprojected_CAR_strength']-strength,
                lowest_bright_energy=energy,lowest_bright_gap=energy-e0[0],raw_pole_residue=float(weights[pole].sum()),
                conditional_pole_fraction=float(weights[pole].sum()/strength),computed_low_weights=weights.tolist(),
                uncomputed_high_energy_source_strength=strength-float(weights.sum()),
                source_total_J2=mean(state,observables[name]['J2'])/strength,
                source_J2_residual=norm(observables[name]['J2']@state-float(spin)*(float(spin)+1)*state),
                pole_observables={key:mean(receiver,value) for key,value in observables[name].items()},
                pole_eigen_residual=norm(laws[name][0]@receiver-energy*receiver))
            save_sparse(stored,name+'_source_'+spin,source);stored[name+'_source_state_'+spin]=state;stored[name+'_pole_'+spin]=receiver
            pole_states[name,spin]=receiver
            if spin=='0.5':receivers[name]=receiver
    cchecks={};cn=model['C']['neutral'];hn,jn,kn=laws['neutral']
    cchecks['neutral']=dict(H=norm(hn@cn-cn@hn),J=norm(jn@cn+cn@jn),K=norm(kn@cn-cn@kn),
        unitarity=norm(cn.conj().T@cn-sparse.eye(cn.shape[0])),reference_C_overlap=float(abs(np.vdot(reference,cn@reference))**2))
    save_sparse(stored,'Cneutral',cn)
    for name,anti in [('baryon','antibaryon'),('lepton','antilepton')]:
        c=model['C'][name];h,j,k=laws[name];ha,ja,ka=laws[anti]
        hp,jp,kp=law(model['sectors'][name],model['operators'][name],model['geometry'][name],rotor,scalar,params,p['pulse_angle'])
        ham,jam,kam=law(model['sectors'][anti],model['operators'][anti],model['geometry'][anti],rotor,scalar,params,-p['pulse_angle'])
        cchecks[name]=dict(H=norm(ha@c-c@h),J=norm(ja@c+c@j),K=norm(ka@c-c@k),finite_A_H=norm(ham@c-c@hp),
            finite_A_J=norm(jam@c+c@jp),finite_A_K=norm(kam@c-c@kp),unitarity=norm(c.conj().T@c-sparse.eye(c.shape[1])),
            direct_source_C={spin:norm(model['sources'][anti][spin]@cn-c@source) for spin,source in model['sources'][name].items()})
        save_sparse(stored,'C'+name,c)
    response={};em_events={};field_events={};cut_evidence={};current_cuts={};times=np.array(p['times'])
    for name,state in receivers.items():
        sector=model['sectors'][name];op=model['operators'][name];geo=model['geometry'][name];h,j,k=laws[name];obs=observables[name]
        response[name]=retarded(h,j,k,state,p['frequencies'],p['eta'])
        energy=mean(state,h);baseline={key:[mean(state,value)]*len(times) for key,value in obs.items()}
        angle=p['pulse_angle']*(-1 if name.startswith('anti') else 1)
        hp,_,_=law(sector,op,geo,rotor,scalar,params,angle)
        pulse=expm_multiply((-1j*p['pulse_duration'])*hp,state)
        states,data=dynamics(pulse,h,times,obs)
        pair_baseline=3 if name=='baryon' else 1 if name=='lepton' else 0
        em_events[name]=dict(angle=angle,duration=p['pulse_duration'],times=times.tolist(),energy_injection=mean(pulse,h)-energy,
            norm_error=abs(norm(pulse)-1),baseline=baseline,pulse=data,excess={key:(np.array(value)-baseline[key]).tolist() for key,value in data.items()},
            inclusive_initial=inclusive_pvm(pulse,sector,geo,rotor,scalar,pair_baseline),inclusive_final=inclusive_pvm(states[:,-1],sector,geo,rotor,scalar,pair_baseline),
            preparation='Actual finite external Q Wilson-flux H(A) pulse on SAME normalized pole/reference; evolve with SAME H0. Untouched baseline is stationary same-state H0 evolution, not a newly chosen baseline.')
        stored[name+'_EM_event']=states;stored[name+'_receiver']=state
        field_unitary=expm(-1j*p['field_kick']*scalar['P']);field_state=(state.reshape(geo['dimension'],nf)@field_unitary.T).ravel()
        field_states,fd=dynamics(field_state,h,times,obs)
        field_events[name]=dict(energy_injection=mean(field_state,h)-energy,norm_error=abs(norm(field_state)-1),times=times.tolist(),
            initial_matter_loop_changes={key:mean(field_state,obs[key])-mean(state,obs[key]) for key in ['positive','pairs','N','Q','S2','J2','L2','C','B','boundary','matter_octet','orientation_boundary']},
            baseline=baseline,pulse=fd,excess={key:(np.array(value)-baseline[key]).tolist() for key,value in fd.items()},
            inclusive_final=inclusive_pvm(field_states[:,-1],sector,geo,rotor,scalar,pair_baseline),
            preparation='Scalar-only unitary I_matter,loop,orientation tensor exp(-i kick Pphi). No field/body reset or species projection; actual vector H0 subsequent reaction.')
        stored[name+'_field_event']=field_states
        if name in ['neutral','baryon','lepton']:
            cut_evidence[name]=cut_residuals(state,sector,geo,rotor,scalar,params)
            current_cuts[name]=cut_residuals(state,sector,geo,rotor,scalar,params,derivative=1)
    cut_evidence['baryon_threehalf']=cut_residuals(pole_states['baryon','1.5'],model['sectors']['baryon'],model['geometry']['baryon'],rotor,scalar,params)
    off={}
    for name in ['neutral','baryon','lepton']:
        h,_,_=law(model['sectors'][name],model['operators'][name],model['geometry'][name],rotor,scalar,params,vector_scale=0.)
        e,v=lowest(h,min(12,p['low_eigenstates']));off[name]=dict(ground_energy=float(e[0]),ground_pairs=mean(v[:,0],observables[name]['pairs']),
            ground_orientation_boundary=mean(v[:,0],observables[name]['orientation_boundary']),population_commutator=norm(h@observables[name]['positive']-observables[name]['positive']@h))
    off['gaps']={name:off[name]['ground_energy']-off['neutral']['ground_energy'] for name in ['baryon','lepton']}
    ordering=sources['baryon']['1.5']['lowest_bright_gap']-sources['baryon']['0.5']['lowest_bright_gap']
    payload=dict(kind='NEW_COLOUR_VECTOR_H0_PATH_ORIENTATION_COMMON_ROOT_PHYSICAL_CONSUMER_NOT_REVIEW',parameters=p,
        root={key:value for key,value in model['root'].items() if key not in ['cliff','Phi','Z']},kernel_symmetries=kernel_symmetries(rotor),
        sectors=results,sources=sources,C=cchecks,current_response=response,EM_events=em_events,scalar_field_only_events=field_events,
        cut_residuals=cut_evidence,current_cut_residuals=current_cuts,vector_off_control=off,
        baryon_threehalf_minus_half_gap=ordering,spin_ordering=('threehalf lower' if ordering < -p['degeneracy_tolerance'] else 'halfspin lower' if ordering > p['degeneracy_tolerance'] else 'unresolved/degenerate at declared tolerance'),
        resource=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-started),
        scope='Explicit NEW Ymirror/frame-vector oriented quantum-colour transport is NONZERO in H0 and mixes rotated sea populations. One CAR32 occupied reference and full mixed-Hadamard lepton/uud plus direct C-removal sources. Paircut and finite physical orientation Galerkin projection are variational with measured residuals; PW19 and scalar-DVR cuts remain. J=L_path+S_frame is actual collective-path physical Noether generator, not full COM/three-quark spatial mechanics. No fitted spin penalty, additive mass/recenter, inherited scalar gaps/source strengths, QCD/proton/nature/photon/COM claim.')
    out.write_text(json.dumps(payload,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)
    print(json.dumps(dict(dimensions={name:v['dimension'] for name,v in results.items()},gaps={name:{spin:v['lowest_bright_gap'] for spin,v in ss.items()} for name,ss in sources.items()},raw_strengths={name:{spin:v['raw_strength'] for spin,v in ss.items()} for name,ss in sources.items()},spin_ordering=payload['spin_ordering'],threehalf_minus_half=ordering,C=cchecks,cut_residuals=cut_evidence,resource=payload['resource']),indent=2))


if __name__=='__main__':
    main()
