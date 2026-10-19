"""NEW selected13/pair1/l1 physical review. No saved author state or producer replay."""
from itertools import product, permutations
import gc
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh, expm_multiply, LinearOperator, gmres
import rich_colour_vector_loop_20261015_model as m


def norm(a):
    return float(sparse.linalg.norm(a)) if sparse.issparse(a) else float(np.linalg.norm(a))


def mean(v,a):return float(np.vdot(v,a@v).real)


def cj(a):return {'real':np.asarray(a).real.tolist(),'imag':np.asarray(a).imag.tolist()}


def low_space(h,count):
    vectors=[];raw=[]
    for seed,k in [(913,count),(497,count+4)]:
        e,v=eigsh(h,k=min(k,h.shape[0]-2),which='SA',tol=3e-12,ncv=min(h.shape[0],max(4*k,96)),v0=np.random.default_rng(seed).normal(size=h.shape[0]),maxiter=16000)
        order=np.argsort(e);e=e[order];v=v[:,order];vectors.append(v);raw.append(dict(seed=seed,energies=e.tolist(),raw_orthogonality=norm(v.conj().T@v-np.eye(v.shape[1]))))
    u,s,_=np.linalg.svd(np.column_stack(vectors),full_matrices=False);q=u[:,s>1e-7*s[0]];small=q.conj().T@(h@q);e,r=np.linalg.eigh((small+small.conj().T)/2);v=q@r
    return e,v,dict(independent_ARPACK=raw,union_rank=len(e),orthogonality=norm(v.conj().T@v-np.eye(v.shape[1])),max_eigen_residual=max(norm(h@v[:,i]-e[i]*v[:,i]) for i in range(len(e))))


def own_terms(kind,spin,removal=False,plus_only=False):
    result=[]
    if kind=='lepton':
        result=[([(m.mode(1,r,0,3),True)],1/np.sqrt(2)) for r in ([0] if plus_only else [0,1])]
    else:
        clebsch={(0,0,1):np.sqrt(2/3),(0,1,0):-1/np.sqrt(6),(1,0,0):-1/np.sqrt(6)} if spin==.5 else {(0,0,1):1/np.sqrt(3),(0,1,0):1/np.sqrt(3),(1,0,0):1/np.sqrt(3)}
        for colours in permutations(range(3)):
            sign=(-1)**sum(colours[i]>colours[j] for i in range(3) for j in range(i+1,3))
            for spins,amplitude in clebsch.items():
                for mirrors in ([(0,0,0)] if plus_only else product(range(2),repeat=3)):
                    ops=[(m.mode(f,r,s,c),True) for f,r,s,c in zip((0,0,1),mirrors,spins,colours)]
                    result.append((ops,sign*amplitude/np.sqrt(6*8)))
    if removal:
        converted=[]
        for ops,amplitude in result:
            mapped=[]
            for index,_ in ops:
                mirror=(index%16)//8;spin=(index%8)//4;amplitude*=1 if mirror==spin else -1;mapped.append((index^12,False))
            converted.append((mapped,amplitude))
        return converted
    return result


def source_image(model,reference,kind,spin,removal=False,plus_only=False):
    neutral=model['sectors']['neutral'];geo=model['geometry']['neutral'];rotor=model['rotor'];nf=len(model['scalar']['grid']);nr=rotor['dimension'];entries=[];targets=set()
    for col,mask in enumerate(neutral['masks']):
        for operations,amplitude in own_terms(kind,spin,removal,plus_only):
            step=m.string(mask,operations)
            if step is not None:targets.add(step[0]);entries.append((step[0],col,amplitude*step[1]))
    targets=sorted(targets);lookup={mask:i for i,mask in enumerate(targets)}
    source=sparse.csr_matrix(([v for _,_,v in entries],([lookup[x] for x,_,_ in entries],[c for _,c,_ in entries])),shape=(len(targets),len(neutral['masks'])))
    full=np.zeros((neutral['dimension']*nr,nf),complex);full[geo['indices']]=reference.reshape(geo['dimension'],nf);raw=neutral['P']@full.reshape(neutral['dimension'],-1)
    image=(sparse.kron(source,sparse.eye(19),format='csr')@raw).reshape(len(targets),19,nr,nf)
    return targets,image


def source_project(model,reference,name,spin,plus_only=False):
    kind='lepton' if 'lepton' in name else 'baryon';targets,image=source_image(model,reference,kind,spin,name.startswith('anti'),plus_only)
    target=model['sectors'][name];lookup={x:i for i,x in enumerate(targets)};selected=np.zeros((len(target['masks']),19,image.shape[2],image.shape[3]),complex)
    for row,mask in enumerate(target['masks']):
        if mask in lookup:selected[row]=image[lookup[mask]]
    gauge=target['P'].conj().T@selected.reshape(19*len(target['masks']),-1);gauge=gauge.reshape(target['dimension']*image.shape[2],image.shape[3]);state=gauge[model['geometry'][name]['indices']].ravel()
    return state,dict(full_raw_strength=norm(image)**2,retained_strength=norm(state)**2,projection_loss=norm(image)**2-norm(state)**2,mirror_assignments=1 if plus_only else 2 if kind=='lepton' else 8)


def full_action_audit(model,name,p):
    sector=model['sectors'][name];op=model['operators'][name];rotor=model['rotor'];scalar=model['scalar'];nm=sector['dimension'];nr=rotor['dimension'];nf=len(scalar['grid']);sm=m.scalar_matter(op,p);loop=p['electric']*op['C']/2-p['plaquette']*op['B']
    js=model['geometry'][name]['Jfull']
    def action(x):
        t=x.reshape(nm,nr,nf);out=(t.reshape(-1,nf)@scalar['H'].T).reshape(nm,nr,nf)
        out+=(loop@t.reshape(nm,-1)).reshape(nm,nr,nf)
        out+=np.einsum('rs,asf->arf',rotor['L2'].toarray()/(2*p['orientation_inertia']),t)
        st=(sm@t.reshape(nm,-1)).reshape(nm,nr,nf);out+=st@scalar['phi'].T
        for axis in range(3):
            vt=(op['vectors'][0][axis]@t.reshape(nm,-1)).reshape(nm,nr,nf);out+=p['vector_velocity']*np.einsum('rs,asf->arf',rotor['n'][axis].toarray(),vt)
        return out.reshape(nm*nr,nf)
    rng=np.random.default_rng(319);x=rng.normal(size=(nm*nr,nf))+1j*rng.normal(size=(nm*nr,nf));x/=norm(x);hx=action(x);defects=[]
    for j in js:defects.append(norm(action(j@x)-j@hx)/max(1.,norm(hx)))
    gen=.23*js[0]-.17*js[1]+.31*js[2];ux=expm_multiply(-1j*gen,x)
    return dict(full_pre_fixed_M_dimension=nm*nr*nf,H_J_action_relative=defects,finite_rotation_action=norm(action(ux)-expm_multiply(-1j*gen,hx))/max(1.,norm(hx)),meaning='All three physical generators act before fixed-M reduction. No dense whole-H matrix or frame-only substitute.')


def extension(model,name,state,p,chunk):
    sector=model['sectors'][name];geo=model['geometry'][name];rotor=model['rotor'];scalar=model['scalar'];nr=rotor['dimension'];nf=len(scalar['grid']);nm=sector['dimension']
    full=np.zeros((nm*nr,nf),complex);full[geo['indices']]=state.reshape(geo['dimension'],nf);tensor=full.reshape(nm,nr,nf);raw=sector['P']@tensor.reshape(nm,-1)
    present=set(sector['masks']);missing=set()
    for mask in present:
        for f,r,s,t,a,b in product(range(2),range(2),range(2),range(2),range(4),range(4)):
            if (a==3)!=(b==3):continue
            step=m.string(mask,[(m.mode(f,r,s,a),True),(m.mode(f,1-r,t,b),False)])
            if step is not None and step[0] not in present:missing.add(step[0])
    missing=sorted(missing);larger=m.orientation(p['orientation_lmax']+1);n_ext=[x[:,:nr].toarray() for x in larger['n']];results={}
    for derivative in [0,1]:
        pair2=0.;intersection2=0.
        for begin in range(0,len(missing),chunk):
            targets=missing[begin:begin+chunk];out=np.zeros((19*len(targets),larger['dimension'],nf),complex)
            for axis in range(3):
                image=(m.raw_vector(sector['masks'],model['loop'],axis,target_masks=targets,derivative=derivative)@raw).reshape(19*len(targets),nr,nf)
                out+=p['vector_velocity']*np.einsum('rs,asf->arf',n_ext[axis],image,optimize=True)
            pair2+=norm(out[:,:nr])**2;intersection2+=norm(out[:,nr:])**2
        orientation=np.zeros((nm,larger['dimension']-nr,nf),complex)
        for axis in range(3):
            image=(m.project(sector,m.raw_vector(sector['masks'],model['loop'],axis,derivative=derivative))@tensor.reshape(nm,-1)).reshape(nm,nr,nf)
            orientation+=p['vector_velocity']*np.einsum('rs,asf->arf',n_ext[axis][nr:],image,optimize=True)
        orientation2=norm(orientation)**2
        results['H0' if derivative==0 else 'current']=dict(missing_raw_pair_masks=len(missing),pair_retained_orientation=np.sqrt(pair2),orientation_retained_pair=np.sqrt(orientation2),pair_orientation_intersection=np.sqrt(intersection2),true_union_extension_norm=np.sqrt(pair2+orientation2+intersection2),wrong_pair_only_estimate=np.sqrt(pair2),meaning='Rectangular physical one-step images into full next orientation times missing-pair raw CAR/PW space; three orthogonal support pieces combined. Not eigen/resolvent bound or higher-PW/outside-DVR completion.')
    return results


def probabilities(model,name,state):
    sector=model['sectors'][name];geo=model['geometry'][name];rotor=model['rotor'];nf=len(model['scalar']['grid']);nr=rotor['dimension'];full=np.zeros((sector['dimension']*nr,nf),complex);full[geo['indices']]=state.reshape(geo['dimension'],nf)
    raw=(sector['P']@full.reshape(sector['dimension'],-1)).reshape(len(sector['masks']),19,nr,nf);w=abs(raw)**2;pairbase=3 if name=='baryon' else 1 if name=='lepton' else 0
    pairs=np.array([sum(bool(mask&(1<<m.mode(f,0,s,c))) for f,s,c in product(range(2),range(2),range(4)))-pairbase for mask in sector['masks']]);ls=np.array([l for l,z in rotor['basis']]);joint={}
    for pair in sorted(set(pairs)):
        for l in sorted(set(ls)):
            for pw,indices in enumerate([range(1),range(1,10),range(10,19)]):joint[f'{pair},{l},{pw}']=float(w[pairs==pair][:,list(indices)][:,:,ls==l].sum())
    return dict(sum=sum(joint.values()),joint=joint,scalar=w.sum(axis=(0,1,2)).tolist(),meaning='Complete inclusive commuting pair/orientation/PW projectors; no outgoing-source postselection.')


def response(h,j,k,state,frequencies,eta):
    energy=mean(state,h);image=j@state;values=[];residuals=[];flags=[]
    for w in frequencies:
        pair=[];rr=[];ff=[]
        for sign in [-1,1]:
            shift=w-sign*energy+1j*eta
            a=LinearOperator(h.shape,matvec=lambda x,s=sign,z=shift:z*x+s*(h@x),dtype=complex)
            diagonal=shift+sign*h.diagonal();pre=LinearOperator(h.shape,matvec=lambda x,d=diagonal:x/d,dtype=complex)
            x,flag=gmres(a,image,M=pre,rtol=3e-11,atol=1e-13,restart=100,maxiter=500);pair.append(x);rr.append(norm(a@x-image));ff.append(flag)
            if flag:raise RuntimeError('Full matrix-free resolvent did not converge: '+str((w,sign,flag,rr[-1])))
        values.append(np.vdot(image,pair[0]-pair[1]));residuals.append(rr);flags.append(ff)
    return dict(retained_dimension=h.shape[0],spectral_strength=norm(image)**2,contact=mean(state,k),retarded=cj(values),kernel=cj(np.array(values)+mean(state,k)),residuals=residuals,flags=flags,frequencies=frequencies,eta=eta,meaning='Two full signed matrixfree inverses, NOT a low-eigenstate spectral approximation.')


def main():
    inp,out=map(Path,sys.argv[1:3]);case=json.loads(inp.read_text());p=json.loads((inp.parent/case['owner_input']).read_text());started=time.monotonic();model=m.build(p);nf=len(model['scalar']['grid']);params=model['parameters'];laws={};obs={};spectra={};summary={};saved={}
    for name in model['sectors']:
        sec=model['sectors'][name];op=model['operators'][name];geo=model['geometry'][name];laws[name]=m.law(sec,op,geo,model['rotor'],model['scalar'],params)
        obs[name]={key:m.whole_observable(op[key],geo,model['rotor'],model['scalar']) for key in ['positive','pairs','N','Q','S2','C','B','boundary']}
        obs[name].update(J2=sparse.kron(geo['J2'],sparse.eye(nf),format='csr'),L2=sparse.kron(geo['L2'],sparse.eye(nf),format='csr'),orientation_boundary=sparse.kron(geo['orientation_boundary'],sparse.eye(nf),format='csr'),phi2=sparse.kron(sparse.eye(geo['dimension']),model['scalar']['phi']@model['scalar']['phi'],format='csr'))
        if not name.startswith('anti'):
            e,v,audit=low_space(laws[name][0],case['low_eigenstates']);spectra[name]=(e,v)
            summary[name]=dict(dimension=len(e)*0+laws[name][0].shape[0],low_space=audit,energies=e.tolist(),full_rotation=full_action_audit(model,name,params),H_population_commutator=norm(laws[name][0]@obs[name]['positive']-obs[name]['positive']@laws[name][0]),H_J2_commutator=norm(laws[name][0]@obs[name]['J2']-obs[name]['J2']@laws[name][0]),ground_J2_arbitrary=mean(v[:,0],obs[name]['J2']))
            saved[name+'_low_e']=e;saved[name+'_low_v']=v
        print(json.dumps({'built':name,'dimension':laws[name][0].shape[0]}),flush=True)
    reference=spectra['neutral'][1][:,0];e0=spectra['neutral'][0][0];saved['reference']=reference;source_reports={};receivers={'neutral':reference};pole_states={};cn=model['C']['neutral']
    for name in ['lepton','baryon']:
        e,v=spectra[name];source_reports[name]={}
        for spin in ([.5] if name=='lepton' else [.5,1.5]):
            raw,rawreport=source_project(model,reference,name,spin);helper=model['sources'][name][str(spin)]@reference;weights=abs(v.conj().T@raw)**2;bright=np.flatnonzero(weights>p['bright_tolerance']*norm(raw)**2)
            energy=e[bright[0]];group=abs(e-energy)<p['degeneracy_tolerance'];projected=v[:,group]@(v[:,group].conj().T@raw);residue=norm(projected)**2;state=projected/np.sqrt(residue);wrong,wrongreport=source_project(model,reference,name,spin,plus_only=True)
            anti='anti'+name;direct_anti,_=source_project(model,cn@reference,anti,spin);c=model['C'][name]
            groupj=v[:,group].conj().T@(obs[name]['J2']@v[:,group]);source_reports[name][str(spin)]=dict(raw=rawreport,helper_state_difference=norm(raw-helper),raw_strength=norm(raw)**2,pole_residue=residue,residue_le_strength=bool(residue<=norm(raw)**2+1e-10),complete_low_ground_group_size=int(group.sum()),group_J2_eigenvalues=np.linalg.eigvalsh((groupj+groupj.conj().T)/2).tolist(),pole_gap=float(energy-e0),source_J2=mean(raw,obs[name]['J2'])/norm(raw)**2,source_J2_residual=norm(obs[name]['J2']@raw-spin*(spin+1)*raw),pole_J2=mean(state,obs[name]['J2']),pole_J2_residual=norm(obs[name]['J2']@state-spin*(spin+1)*state),pole_eigen_residual=norm(laws[name][0]@state-energy*state),N=mean(raw,obs[name]['N'])/norm(raw)**2,Q=mean(raw,obs[name]['Q'])/norm(raw)**2,direct_C_source_defect=norm(direct_anti-c@raw),wrong_plus_only=dict(report=wrongreport,state_difference=norm(wrong-raw),norm_ratio=norm(wrong)**2/norm(raw)**2),uncomputed_source_strength=norm(raw)**2-float(weights.sum()),low_cluster_not_at_edge=bool(abs(e[-1]-energy)>p['degeneracy_tolerance']))
            saved[name+'_raw_'+str(spin)]=raw;saved[name+'_pole_'+str(spin)]=state;pole_states[name,str(spin)]=state
            if spin==.5:receivers[name]=state
    c_audits={};derivatives={};events={};responses={};cuts={};times=case['times']
    for name,state in receivers.items():
        h,j,k=laws[name];sec=model['sectors'][name];op=model['operators'][name];geo=model['geometry'][name];step=case['derivative_step'];hp,_,_=m.law(sec,op,geo,model['rotor'],model['scalar'],params,step);hm,_,_=m.law(sec,op,geo,model['rotor'],model['scalar'],params,-step)
        derivatives[name]=dict(J=norm((hp-hm)/(2*step)+j)/max(1.,norm(j)),positive_contact=norm((hp+hm-2*h)/step**2-k)/max(1.,norm(k)))
        responses[name]=response(h,j,k,state,case['frequencies'],case['eta']);cuts[name]=extension(model,name,state,params,case['extension_chunk']);cuts[name]['retained_eigen_residual']=norm(h@state-mean(state,h)*state)
        cuts[name]['full_H_minus_E_extension_norm']=float(np.hypot(cuts[name]['H0']['true_union_extension_norm'],cuts[name]['retained_eigen_residual']))
        cuts[name]['boundary_occupancies']={key:mean(state,obs[name][key]) for key in ['pairs','orientation_boundary','boundary']};cuts[name]['scalar_edge']=float(abs(state.reshape(geo['dimension'],nf)[:,[0,-1]]) .__pow__(2).sum())
        pulsed_h,_,_=m.law(sec,op,geo,model['rotor'],model['scalar'],params,case['pulse_angle']);pulse=expm_multiply(-1j*case['pulse_duration']*pulsed_h,state)
        fu=expm(-1j*case['field_kick']*model['scalar']['P']);field=(state.reshape(geo['dimension'],nf)@fu.T).ravel();baseline=[];em=[];fe=[];cdef=[];anti='anti'+name if name!='neutral' else 'neutral';c=model['C'][name];ha=laws[anti][0]
        ahp,_,_=m.law(model['sectors'][anti],model['operators'][anti],model['geometry'][anti],model['rotor'],model['scalar'],params,-case['pulse_angle']);paired_initial=c@state;anti_pulse=expm_multiply(-1j*case['pulse_duration']*ahp,paired_initial)
        c_audits[name]=dict(H=norm(ha@c-c@h),J=norm(laws[anti][1]@c+c@j),K=norm(laws[anti][2]@c-c@k),finite_A_H=norm(ahp@c-c@pulsed_h),paired_pulse_state_defect=norm(anti_pulse-c@pulse),initial_state_contract='Anti initial state is exactly C times this same source-projected pole, NOT an arbitrary independently selected degenerate ground column.')
        for t in times:
            b=expm_multiply(-1j*t*h,state);s=expm_multiply(-1j*t*h,pulse);f=expm_multiply(-1j*t*h,field);a=expm_multiply(-1j*t*ha,anti_pulse);baseline.append(b);em.append(s);fe.append(f);cdef.append(norm(a-c@s))
        def measurements(states):return [{key:mean(v,a)-mean(b,a) for key,a in obs[name].items()} for v,b in zip(states,baseline)]
        events[name]=dict(times=times,EM_excess=measurements(em),field_only_excess=measurements(fe),field_initial_nonfield_changes={key:mean(field,obs[name][key])-mean(state,obs[name][key]) for key in ['pairs','S2','J2','L2','C','B','boundary','orientation_boundary']},EM_injection=mean(pulse,h)-mean(state,h),field_injection=mean(field,h)-mean(state,h),C_same_initial_trajectory_defects=cdef,inclusive_EM_final=probabilities(model,name,em[-1]),inclusive_field_final=probabilities(model,name,fe[-1]),norms=[norm(s)**2 for s in em],same_state_baseline_stationarity=max(norm(b-np.exp(-1j*mean(state,h)*t)*state) for b,t in zip(baseline,times)))
        saved[name+'_EM_states']=np.column_stack(em);saved[name+'_field_states']=np.column_stack(fe);saved[name+'_baselines']=np.column_stack(baseline)
        print(json.dumps({'completed_physics':name,'extension':cuts[name]['H0']['true_union_extension_norm']}),flush=True)
    result=dict(kind='WAVE16_NEW_SELECTED13_NODE_PHYSICAL_REVIEW_NOT_PROTON_OR_ADMISSION',parameters=p,case=case,sectors=summary,sources=source_reports,current_contact=derivatives,retarded_response=responses,C=c_audits,events=events,extension=cuts,baryon_threehalf_minus_half=source_reports['baryon']['1.5']['pole_gap']-source_reports['baryon']['0.5']['pole_gap'],resource=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.monotonic()-started),scope='Actual H0 vector/collective path J, all mirror sources; same13 scalar nodes/pair1/l1. Matrixfree whole-space actions and inverses; chunked exact one-step omitted support. PW/DVR/full relative geometry/COM/photons not completed; lepton TrU polarizability is an interaction input, not matter SU3 charge.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**saved)
    print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
