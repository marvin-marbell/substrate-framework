"""Independent rebuilt physical consumers; no producer entry point or saved state import."""
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import expm_multiply
import rich_common_vector_20261015_model as cv
import rich_rotcov_induced_20261015_model as rc


def norm(a):
    return float(sparse.linalg.norm(a)) if sparse.issparse(a) else float(np.linalg.norm(a))


def mean(v,a):
    return float(np.vdot(v,a@v).real)


def complex_json(a):
    return {'real':np.asarray(a).real.tolist(),'imag':np.asarray(a).imag.tolist()}


def kernel(gaps,amps,frequencies,eta):
    residues=np.einsum('ni,nj->nij',amps.conj(),amps)
    return np.array([np.sum(residues/(w-gaps[:,None,None]+1j*eta)-residues.transpose(0,2,1)/(w+gaps[:,None,None]+1j*eta),axis=0) for w in frequencies])


def common(p,case):
    model=cv.build(p);nf=len(model['scalar']['grid']);loop=model['loop'];scalar=model['scalar']
    h0,_,_=cv.law(model['operators']['neutral'],scalar,p);e0,v0=eigh(h0);reference=v0[:,0]
    records={};saved={'neutral_reference':reference};frequencies=np.array(case['frequencies']);eta=case['eta']
    for name in ['neutral','lepton','antilepton','baryon','antibaryon']:
        base=model['sectors'][name];op=model['operators'][name];h,j,k=cv.law(op,scalar,p);e,v=eigh(h)
        if name=='neutral':state=reference;source_report=None
        else:
            raw=model['sources'][name]['0.5']@reference;weight=abs(v.conj().T@raw)**2;bright=np.flatnonzero(weight>1e-10);pole=abs(e-e[bright[0]])<1e-8
            state=v[:,pole]@(v[:,pole].conj().T@raw);state/=norm(state)
            source_report=dict(raw_strength=norm(raw)**2,raw_pole_residue=float(weight[pole].sum()),gap=float(e[bright[0]]-e0[0]),N=mean(raw,np.kron(op['N'],np.eye(nf)))/norm(raw)**2,Q=mean(raw,np.kron(op['Q'],np.eye(nf)))/norm(raw)**2)
        energy=mean(state,h);amps=np.zeros((len(e),3),complex);amps[:,2]=v.conj().T@j@state
        chi=kernel(e-energy,amps,frequencies,eta);contact=np.zeros((3,3));contact[2,2]=mean(state,k);chi+=contact
        reversed_chi=chi.copy();direct=amps.conj().T@amps;allg=[e-energy];alla=[amps];closure=[];receiver=[];maxc=0.;maxr=0.;maxder=0.;maxgauss=0.;second_signatures=set()
        reverse_base=cv.spin_reverse_sector(base);first_signatures=cv.reachable_signatures(base)
        for bi,signature in enumerate(first_signatures):
            target=cv.gauge_sector(cv.complete_masks(signature),loop);top=cv.operators(target,loop);th,_,_=cv.law(top,scalar,p);te,tv=eigh(th)
            js=[cv.current_block(base,target,a,p['vector_velocity'],nf) for a in range(3)];aa=np.column_stack([tv.conj().T@x@state for x in js]);gaps=te-energy
            chi+=kernel(gaps,aa,frequencies,eta);direct+=aa.conj().T@aa;allg.append(gaps);alla.append(aa)
            rt=cv.spin_reverse_sector(target);rjs=[cv.current_block(reverse_base,rt,a,p['vector_velocity'],nf) for a in range(3)];ra=np.column_stack([tv.conj().T@x@state for x in rjs]);reversed_chi+=kernel(gaps,ra,frequencies,eta)
            maxr=max(maxr,max(norm(rj-sign*jj) for rj,jj,sign in zip(rjs,js,[-1,1,-1])))
            anti=cv.conjugate_sector(target,loop);ah,_,_=cv.law(cv.operators(anti,loop),scalar,p)
            abase=cv.conjugate_sector(base,loop);ajs=[cv.current_block(abase,anti,a,p['vector_velocity'],nf) for a in range(3)];maxc=max(maxc,norm(ah-th),max(norm(a+b) for a,b in zip(ajs,js)))
            # Actual finite path sine for a rectangular current-connected block.
            a=case['derivative_step'];axis=0;terms=cv.vector_terms(axis)
            def finite(angle):
                converted=[]
                for x,y,value in terms:
                    f=x//16;c=x%4;q=([2/3]*3+[0.] if f==0 else [-1/3]*3+[-1.])[c]
                    converted.append((x,y,p['vector_velocity']*np.sin(q*angle)*value/q))
                raw=sparse.kron(cv.one_body(base['masks'],converted,target['masks']),sparse.eye(19),format='csr')
                return np.kron((target['P'].conj().T@raw@base['P']).toarray(),np.eye(nf))
            maxder=max(maxder,norm((finite(a)-finite(-a))/(2*a)+js[axis]))
            maxgauss=max(maxgauss,target['gauge_error']);closure.append(cv.closure(target));second_signatures.update(cv.reachable_signatures(target))
            conditioned=sum(z*x@state for z,x in zip(case['current_coefficients'],js));receiver.append((conditioned,te,tv,top))
        raw_strength=sum(norm(x[0])**2 for x in receiver);event=[]
        phi=scalar['phi'];field_bin=np.diag((scalar['grid']>0).astype(float))
        for time in case['times']:
            record=dict(time=time,probability=0.,phi2=0.,positive_phi_PVM=0.,loop_C=0.)
            for raw,te,tv,top in receiver:
                s=tv@(np.exp(-1j*te*time)*(tv.conj().T@raw))/np.sqrt(raw_strength);n=top['S2'].shape[0]
                record['probability']+=norm(s)**2;record['phi2']+=mean(s,np.kron(np.eye(n),phi@phi));record['positive_phi_PVM']+=mean(s,np.kron(np.eye(n),field_bin));record['loop_C']+=mean(s,np.kron(top['C'],np.eye(nf)))
            event.append(record)
        # Independent field-only displacement on the untouched same source pole.
        kicked=np.kron(np.eye(base['dimension']),expm(-1j*case['field_kick']*scalar['P']))@state;field_event=[]
        for time in case['times']:
            s=v@(np.exp(-1j*e*time)*(v.conj().T@kicked));b=v@(np.exp(-1j*e*time)*(v.conj().T@state))
            obs=np.kron(np.eye(base['dimension']),phi@phi)
            field_event.append(dict(time=time,norm=norm(s)**2,phi2_excess=mean(s,obs)-mean(b,obs)))
        gaps=np.concatenate(allg);amplitudes=np.vstack(alla);saved[name+'_gaps']=gaps;saved[name+'_amplitudes']=amplitudes;saved[name+'_state']=state
        records[name]=dict(source=source_report,virtual_states=len(gaps),minimum_signed_gap=float(gaps.min()),contact=contact.tolist(),chi=complex_json(chi),spin_reversed_chi=complex_json(reversed_chi),xy_imag_sign_reversal=float(np.max(abs(chi[:,0,1].imag+reversed_chi[:,0,1].imag))),spin_rotation_operator_defect=maxr,C_H_current_defect=maxc,finite_path_derivative_defect=maxder,Gauss_defect=maxgauss,H0_missing=max([cv.closure(base)]+closure),one_current_signatures=len(first_signatures),second_current_additional_signatures=len(second_signatures-set(first_signatures)-{cv.signatures(base['masks'][0])}),integrated_strength=complex_json(direct),conditional_raw_strength=raw_strength,conditional_receiver=event,field_only_event=field_event)
        print(json.dumps({'common':name,'virtuals':len(gaps),'raw_strength':raw_strength}),flush=True)
    return dict(kind='INDEPENDENT_REBUILT_COMMON_TWO_CURRENT_AND_CONDITIONAL_RECEIVERS',path=cv.path_derivation(p),records=records,frequencies=frequencies.tolist(),eta=eta,scope='All one-current-connected H0 blocks. NOT nonlinear finite-A closure, pair yield, spatial continuity, free COM or photon scattering.'),saved


def sector_eigensystems(model,number):
    numbers=model['N'].diagonal().real;mz=model['J'][2].diagonal().real;selected=np.flatnonzero(abs(numbers-number)<1e-8);blocks=[]
    for m in np.unique(np.round(2*mz[selected])/2):
        idx=selected[abs(mz[selected]-m)<1e-7];e,v=eigh(model['H'][idx][:,idx].toarray(),check_finite=False);blocks.append((idx,e,v))
    return blocks


def rotation(p,case):
    model=rc.build(p);h=model['H'];dim=h.shape[0];blocks=sector_eigensystems(model,2);idx,e,v=min(blocks,key=lambda x:x[1][0]);energy=e[0];ground=np.zeros(dim,complex);ground[idx]=v[:,0]
    j2=sum(x@x for x in model['J']);source=sum(z*x@ground for z,x in zip(case['source_coefficients'],model['sources']));strength=norm(source)**2;source/=np.sqrt(strength)
    prepared=expm_multiply(1j*case['boost']*model['X'][0],source);prepared=expm_multiply(-1j*case['displacement']*model['PX'][2],prepared)
    generator=model['P'][0]+.37*model['Q'][2]-.63*model['P'][4]+.29*model['Q'][3];kicked=expm_multiply(-1j*case['kick']*generator,prepared)
    allgap=[];allamp=[]
    for idx,e,v in blocks:
        probes=np.column_stack([(a@ground)[idx] for a in model['current']]);allgap.append(e-energy);allamp.append(v.conj().T@probes)
    gaps=np.concatenate(allgap);amps=np.vstack(allamp);contact=np.array([[mean(ground,a) for a in row] for row in model['contact']]);chi=kernel(gaps,amps,case['frequencies'],case['eta'])+contact
    rows=[];states=[];baselines=[];families=[model['S'],model['L']]+model['SL']+model['F'];labels=['S','L','supplier1','supplier2','TE','TM']
    # Inclusive commuting receiver: field-mode parity AND body physical Sz.
    dims=model['dims'];fu=model['field']['u'];occupancy=np.diag([(-1.)**(n[0]+n[3]) for n in model['field']['basis']]);parity=fu.conj().T@occupancy@fu
    fproj=(np.eye(dims[4])+parity)/2;field_receiver=sparse.kron(sparse.eye(np.prod(dims[:4])),sparse.csr_matrix(fproj),format='csr');spin=model['S'][2];spin_values=spin.diagonal().real;spin_receiver=sparse.diags((spin_values>0).astype(float));receiver=field_receiver@spin_receiver
    for time in case['times']:
        state=expm_multiply(-1j*time*h,kicked);baseline=expm_multiply(-1j*time*h,prepared);states.append(state);baselines.append(baseline)
        rows.append(dict(time=time,norm=norm(state)**2,energy_drift=mean(state,h)-mean(kicked,h),J=[mean(state,a) for a in model['J']],torque_balance=[mean(state,1j*rc.comm(h,a)) for a in model['J']],angular_excess={label:[mean(state,a)-mean(baseline,a) for a in ops] for label,ops in zip(labels,families)},supplier_energy_excess=mean(state,model['supplier_energy'])-mean(baseline,model['supplier_energy']),supplier_r2_excess=[mean(state,a)-mean(baseline,a) for a in model['supplier_r2']],field_energy_excess=mean(state,model['field_energy'])-mean(baseline,model['field_energy']),joint_receiver_probability=mean(state,receiver),baseline_joint_probability=mean(baseline,receiver),current_excess=[mean(state,a)-mean(baseline,a) for a in model['current']]))
    derivatives=[];seconds=[];step=case['derivative_step']
    for a in range(6):
        ext=np.zeros(6);ext[a]=step;hp=rc.shifted_h(model,ext);hm=rc.shifted_h(model,-ext);derivatives.append(norm((hp-hm)/(2*step)+model['current'][a])/max(1.,norm(model['current'][a])));seconds.append(norm((hp+hm-2*h)/step**2-model['contact'][a][a])/max(1.,norm(model['contact'][a][a])))
    # New cut2 quadrature-only witness: retained 0<->2 kinetic, not old producer.
    osc=rc.oscillator(2);quad=rc.quadrature(osc,p['length'],p['radial_quadrature']);kin=quad['u']@quad['kinetic'](p['reduced_mass'])@quad['u'].conj().T;vac=osc['basis'].index((0,0,0));two=osc['basis'].index((2,0,0));expected=-np.sqrt(2)/(4*p['reduced_mass']*p['length']**2)
    sup=rc.supplier(dict(p,supplier_path='constant_gauge'));se,sv=eigh(sup['H']);sg=sv[:,0];pure=[]
    for a in range(3):
        aa=sv.conj().T@sup['linear'][a]@sg;dia=2*mean(sg,sup['quadratic'][a][a]);para=float(-2*np.sum(abs(aa[1:])**2/(se[1:]-se[0])));pure.append(dict(contact=dia,virtual=para,residual=dia+para,fraction=(dia+para)/dia))
    result=dict(kind='INDEPENDENT_REBUILT_ROTCOV_FIELD_ONLY_CONSUMER',dimensions=dims,neutral_J2=mean(ground,j2),raw_source_strength=strength,unfiltered_source_J2=mean(source,j2),moving_J2=mean(prepared,j2),virtual_states=len(gaps),minimum_signed_gap=float(gaps.min()),contact=contact.tolist(),chi=complex_json(chi),current_derivative_defects=derivatives,positive_contact_derivative_defects=seconds,H_J_relative=[norm(rc.comm(h,a))/norm(h) for a in model['J']],left_spin_commutator=[norm(rc.comm(h,a))/norm(h) for a in model['S']],receiver_commutator=norm(rc.comm(field_receiver,spin_receiver)),receiver_idempotence=norm(receiver@receiver-receiver),event=rows,cut2_kinetic_witness=dict(actual=complex_json(kin[vac,two]),expected=expected,error=abs(kin[vac,two]-expected)),constant_R3_gradient_control=pure,scope='Initial frozen cut1 law. Complete retained N2 signed virtual response, same-H raw source and physical field-only kick with inclusive commuting receiver. Body-supplier longitudinal cross energy absent. Gauge fork excluded/not stable, Wave14.')
    frame=model['frame'];covariance=[]
    for axis in range(3):
        rep=np.array([[np.trace(d.conj().T@rc.comm(frame['S'][axis],s))/8 for d in frame['source']] for s in frame['source']])
        for b,source_op in enumerate(model['sources']):
            covariance.append(norm(rc.comm(model['J'][axis],source_op)-sum(rep[b,c]*model['sources'][c] for c in range(4))))
    theta=np.array([.17,-.23,.31]);generator=sum(theta[a]*model['J'][a] for a in range(3))
    rotated=expm_multiply(-1j*generator,prepared)
    result['physical_action']=dict(source_tensor_covariance=max(covariance),finite_H_rotation_on_moving_state=norm(h@rotated-expm_multiply(-1j*generator,h@prepared)),charged_2pi_minus=norm(expm_multiply(-2j*np.pi*model['J'][2],source)+source),angular_algebra=[norm(rc.comm(model['J'][a],model['J'][(a+1)%3])-1j*model['J'][(a+2)%3]) for a in range(3)])
    print(json.dumps({'rotation_dim':dim,'virtuals':len(gaps),'final_receiver':rows[-1]['joint_receiver_probability']}),flush=True)
    return result,dict(ground=ground,source=source,prepared=prepared,kicked=kicked,event_states=np.array(states),baseline_states=np.array(baselines),gaps=gaps,current_amplitudes=amps)


def main():
    inp,out=map(Path,sys.argv[1:3]);case=json.loads(inp.read_text());p=json.loads((inp.parent/case['owner_input']).read_text())
    result,saved=(common if case['case']=='common' else rotation)(p,case)
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**saved)


if __name__=='__main__':main()
