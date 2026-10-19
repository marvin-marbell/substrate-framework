"""Actual gauge-including quantum consumer and adverse source/metric controls."""
import json
from pathlib import Path
import sys
import resource
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye, kron, csr_matrix
from scipy.sparse.linalg import expm_multiply, eigsh
from rich_rotcov_gauge_20261015_model import build, shifted_h, london_profiles
from rich_rotcov_induced_20261015_calculate import mean, magnitude, groups
from rich_rotcov_induced_20261015_model import comm, oscillator, quadrature

def ground_state(model,n):
    selected=np.flatnonzero(abs(model['N'].diagonal().real-n)<1e-7)
    matrix=model['H'][selected][:,selected]
    values,vectors=eigsh(matrix,k=1,which='SA',tol=2e-12,v0=np.linspace(.2,1.,len(selected)))
    vector=np.zeros(model['H'].shape[0],complex);vector[selected]=vectors[:,0]
    return float(values[0]),vector


def spectrum_blocks(model,n,probes):
    """Yield ALL retained M eigenspaces; do not keep prior dense eigenvectors."""
    selected=np.flatnonzero(abs(model['N'].diagonal().real-n)<1e-7)
    jz=model['J'][2].diagonal().real
    for m in np.unique(np.round(jz[selected]*2)/2):
        idx=selected[abs(jz[selected]-m)<1e-6]
        es,u=eigh(model['H'][idx][:,idx].toarray(),check_finite=False,overwrite_a=True)
        yield es,m,idx,u,u.conj().T@probes[idx]


def receiver(v,model):
    """Exact retained total physical Jz / physical Q_TE,z joint PVM."""
    fz=sum(f[2] for f in model['field']['F']);q=model['field']['Q'][2]
    unused,u=eigh(q+10*fz);qvalues=np.diag(u.conj().T@q@u).real;fvalues=np.diag(u.conj().T@fz@u).real
    tensor=v.reshape(-1,model['dims'][4])@u.conj();prob=abs(tensor)**2
    bodyj=model['J'][2].diagonal().real.reshape(-1,model['dims'][4])[:,0]-fz[0,0].real
    joint={}
    for i,k in zip(*np.where(prob>1e-12)):
        key=f'{round(float(bodyj[i]+fvalues[k]),6)},{round(float(qvalues[k]),6)}';joint[key]=joint.get(key,0.)+float(prob[i,k])
    return dict(observables=['total_physical_Jz','physical_Q_TE_z'],joint=joint,probability_sum=float(np.sum(prob)),commutator=magnitude(comm(model['J'][2],model['Q'][2])))


def main():
    inp=Path(sys.argv[1]);out=Path(sys.argv[2]);p=json.loads(inp.read_text());model=build(p);h=model['H'];j=model['J'];j2=sum(a@a for a in j);dim=h.shape[0]
    energy,ground=ground_state(model,2)
    print(json.dumps(dict(stage='new_whole_neutral',dimensions=model['dims'],H_nonzeros=h.nnz,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,energy=energy,residual=magnitude((h-energy*eye(dim))@ground))),flush=True)
    raw=model['sources'][0]@ground;strength=float(np.vdot(raw,raw).real);source=raw/np.sqrt(strength);saved={};spectral={};grams={}
    for n,operators,label in [(3,model['sources'],'addition'),(1,[a.conj().T for a in model['sources']],'removal')]:
        probes=np.column_stack([a@ground for a in operators]);blocks=spectrum_blocks(model,n,probes);energies=[];amplitudes=[];poles=[]
        for es,m,idx,u,amps in blocks:
            energies.extend(es);amplitudes.extend(amps)
            for group in groups(es):
                weight=float(np.sum(abs(amps[group])**2))
                if weight>1e-9:
                    sub=u[:,group];poles.append(dict(gap=float(es[group[0]]-energy),M=float(m),strength=weight,J2=np.linalg.eigvalsh(sub.conj().T@(j2[idx][:,idx]@sub)).real.tolist()))
        es=np.array(energies);amps=np.array(amplitudes);order=np.argsort(es);es=es[order];amps=amps[order];grams[label]=amps.conj().T@amps
        spectral[label]=dict(retained_states=len(es),lowest_gap=float(es[0]-energy),bright_poles=sorted(poles,key=lambda x:x['gap'])[:24]);saved[label+'_energies']=es;saved[label+'_amplitudes']=amps
        del blocks
        print(json.dumps(dict(stage=label,retained_states=len(es),lowest_gap=float(es[0]-energy))),flush=True)
    probes=np.column_stack([a@ground for a in model['direct_original_current']]+[a@ground for a in model['covariant_current']]+[a@ground for a in model['Q']])
    neutral_es=[];neutral_amps=[]
    for es,m,idx,u,amps in spectrum_blocks(model,2,probes):
        neutral_es.extend(es);neutral_amps.extend(amps)
    neutral_es=np.array(neutral_es);neutral_amps=np.array(neutral_amps)
    order=np.argsort(neutral_es);neutral_es=neutral_es[order];neutral_amps=neutral_amps[order]
    positive=neutral_es-energy>1e-8;gaps=neutral_es[positive]-energy;weights=abs(neutral_amps[positive])**2
    response=[]
    for a in range(6):
        columns=[a,a+6,a+12];totals=np.sum(weights[:,columns],axis=0)
        first=[next((float(gap) for gap,weight in zip(gaps,weights[:,column]) if weight>1e-9),None) for column in columns]
        signed=-2*np.sum(weights[:,[a,a+6]]/gaps[:,None],axis=0)
        response.append(dict(family='TE' if a<3 else 'TM',channel=a%3,direct_current_total=float(totals[0]),covariant_current_total=float(totals[1]),field_Q_total=float(totals[2]),first_bright_gap=first,direct_signed_all_virtual=float(signed[0]),covariant_signed_all_virtual=float(signed[1]),physical_contact=mean(ground,model['contact'][a][a])))
    saved['neutral_energies']=neutral_es;saved['neutral_current_Q_amplitudes']=neutral_amps
    del u,amps,probes
    print(json.dumps(dict(stage='all_neutral_virtuals',retained_states=len(neutral_es))),flush=True)
    moving=expm_multiply(-1j*p['displacement']*model['PX'][0],source);moving=expm_multiply(1j*p['boost']*model['X'][1],moving)
    # Q is invariant under the London basis, so this preparation really is
    # independent field-only; no matter-dependent momentum kick is mislabelled.
    kick=model['Q'][1]+.7*model['Q'][3]
    kicked=expm_multiply(-1j*p['kick']*kick,moving);states=[];baselines=[];samples=[]
    keys={'body_spin':model['S'],'body_orbit':model['L'],'supplier1':model['physical_SL'][0],'supplier2':model['physical_SL'][1],'TE':model['physical_F'][0],'TM':model['physical_F'][1]}
    physical_supplier_mechanical=model['supplier_energy']+model['gauge_matter_coupling']
    for t in p['times']:
        state=expm_multiply(-1j*t*h,kicked);baseline=expm_multiply(-1j*t*h,moving);states.append(state);baselines.append(baseline)
        record=dict(time=t,norm=float(np.vdot(state,state).real),energy_drift=mean(state,h)-mean(kicked,h),total_J=[mean(state,a) for a in j],physical_angular={key:[mean(state,a) for a in ops] for key,ops in keys.items()},physical_angular_excess={key:[mean(state,a)-mean(baseline,a) for a in ops] for key,ops in keys.items()},physical_torques={key:[mean(state,1j*comm(h,a)) for a in ops] for key,ops in keys.items()},total_torque=[mean(state,1j*comm(h,a)) for a in j],radius_excess=mean(state,model['radius'])-mean(baseline,model['radius']),supplier_energy_excess=mean(state,model['supplier_energy'])-mean(baseline,model['supplier_energy']),electric_energy_excess=mean(state,model['field_energy'])-mean(baseline,model['field_energy']),direct_current_excess=[mean(state,a)-mean(baseline,a) for a in model['current']],covariant_current_excess=[mean(state,a)-mean(baseline,a) for a in model['covariant_current']],receiver=receiver(state,model))
        record['invariant_supplier_r2_excess']=[mean(state,a)-mean(baseline,a) for a in model['supplier_r2']]
        record['physical_supplier_mechanical_energy_excess']=mean(state,physical_supplier_mechanical)-mean(baseline,physical_supplier_mechanical)
        record['supplier_energy_excess_semantics']='London-coordinate bare Hs only, NOT alone transformed original bare energy or physical mechanical energy; see invariant_r2 and physical_supplier_mechanical fields.'
        samples.append(record)
    # All terms are retained in the actual law. These are explicitly wrong-law
    # adverse controls, propagated from exactly the same physical preparation.
    body=np.prod(model['dims'][:3]);nf=model['dims'][4]
    bh=p['charge']**2*kron(eye(body),kron(csr_matrix(model['born_huang']),eye(nf)),format='csr')/2
    bare_e=kron(eye(body*model['dims'][3]),csr_matrix(model['field']['E']),format='csr')
    controls={}
    for name in ['omit_Born_Huang','omit_all_electric_connections']:
        hcontrol=h-bh if name=='omit_Born_Huang' else h-model['field_energy']+bare_e
        final=expm_multiply(-1j*p['times'][-1]*hcontrol,kicked);controls[name]=dict(state_distance=float(np.linalg.norm(final-states[-1])),supplier_energy_change=mean(final,model['supplier_energy'])-mean(kicked,model['supplier_energy']),physical_J_drift=[mean(final,a)-mean(kicked,a) for a in j])
        del hcontrol,final
    off_h=h-model['gauge_matter_coupling']-model['field_energy']+bare_e
    offstate=expm_multiply(-1j*p['times'][-1]*off_h,kicked)
    controls['transverse_supplier_off']=dict(state_distance=float(np.linalg.norm(offstate-states[-1])),supplier_energy_change=mean(offstate,model['supplier_energy'])-mean(kicked,model['supplier_energy']),note='Longitudinal cross Coulomb remains; supplier energy need not remain fixed in this control.')
    del off_h,offstate
    print(json.dumps(dict(stage='full_event_and_omission_controls',samples=len(samples))),flush=True)
    constant=build(dict(p,gauge_profile='constant'));constant_e,constant_ground=ground_state(constant,2)
    cstate=expm_multiply(-1j*p['times'][-1]*constant['H'],expm_multiply(-1j*p['kick']*constant['Q'][1],constant_ground))
    constant_profile=[]
    for q in constant['supplier_quadratures']:
        original,radial,chi,gradient=london_profiles(q['points'],dict(p,gauge_profile='constant'));constant_profile.append(dict(residual_radial_A=float(np.linalg.norm(radial)),nonzero_chi=float(np.linalg.norm(chi))))
    controls['pure_gradient_static_comoving_basis']=dict(profile=constant_profile,matter_q_coupling_norm=magnitude(constant['gauge_matter_coupling']),note='Static A shifts absorbed by q+a-dependent London basis exactly; this is an operator/basis identity, NOT sufficient physical current/Ward restoration.',nonzero_physical_current_norm=[magnitude(a) for a in constant['current'][:3]],nonzero_dipole_current_norm=[magnitude(a) for a in constant['covariant_current'][:3]],actual_supplier_energy_change=mean(cstate,constant['supplier_energy'])-mean(constant_ground,constant['supplier_energy']),covariant_vs_original_current_ground=[magnitude((constant['covariant_current'][a]-constant['current'][a])@constant_ground) for a in range(3)])
    del constant,cstate,constant_ground
    physical_sum=[model['S'][a]+model['L'][a]+sum(f[a] for f in model['physical_SL']+model['physical_F']) for a in range(3)]
    differences=[model['covariant_current'][a]-model['current'][a] for a in range(6)]
    current_derivative=[];contact_derivative=[];step=1e-3
    for a in range(6):
        ext=np.zeros(6);ext[a]=step;hp=shifted_h(model,ext);hm=shifted_h(model,-ext)
        current_derivative.append(magnitude((hp-hm)/(2*step)+model['current'][a])/max(1.,magnitude(model['current'][a])))
        contact_derivative.append(magnitude((hp+hm-2*h)/step**2-model['contact'][a][a])/max(1.,magnitude(model['contact'][a][a])))
    result=dict(kind='NEW_LONDON_RADIAL_GAUGE_INCLUDING_QUANTUM_CONSTRUCTION_WITH_ADVERSE_SOURCE_TEST',dimensions=model['dims'],parameters=p,occupied=dict(energy=energy,J2=mean(ground,j2),J2_residual=magnitude(j2@ground)),source=dict(strength=strength,J2=mean(source,j2),moving_J2=mean(moving,j2),CAR=magnitude(grams['addition']+grams['removal'].T-np.eye(4)),spectral=spectral),Noether=dict(H_J=[magnitude(comm(h,a))/magnitude(h) for a in j],physical_angular_sum_equals_J=[magnitude(a-b) for a,b in zip(physical_sum,j)],physical_subsystem_PVM_commutator=magnitude(comm(model['physical_SL'][0][2],model['physical_F'][1][2])),claim='Only total physical Jz and Q_TE,z are used as a joint PVM; individual projected physical angular parts are NOT falsely assigned a joint PVM.'),gauge=dict(Born_Huang_eigenvalues=np.linalg.eigvalsh(model['born_huang']).real.tolist(),Born_Huang_H_norm=magnitude(bh),physical_current_derivative=current_derivative,physical_contact_derivative=contact_derivative,covariant_vs_original_current_operator=[magnitude(a) for a in differences],covariant_vs_original_current_ground=[magnitude(a@ground) for a in differences],physical_electric_source_equation_ground=[magnitude((1j*comm(h,model['physical_P'][a])-model['covariant_current'][a])@ground) for a in range(6)],interpretation='London phase absorbs static pure gradients before projection; electric Berry and Born-Huang are active. Remaining differences of the covariant dipole source, original-A continuum source and truncated Heisenberg source are measured, not fitted away. A nonzero difference is an adverse finite-projection result, not proof of gauge-complete Maxwell restoration.'),response=response,event=dict(preparation='Unfiltered new source, physical body displacement/boost, independent EM-only exp[-i kick(Q_TE,y+0.7Q_TM,x)]',external_J_impulse=[mean(kicked,a)-mean(moving,a) for a in j],samples=samples),controls=controls,limits=['Primitive charged bundle, CAR, core binding and electromagnetic profiles remain inputs','Full shell-supplier-central-ion longitudinal density uses the same softened Coulomb kernel; no old additive-only longitudinal law is called complete','Oscillator field cutoff1 has artificial bare gaps and CCR defects; no bulk/continuum/freeCOM/asymptotic photon claim','Static puregradient co-moving-basis equality is weaker than physical source Ward identity; exact operator/state adverse terms are retained','No fitted or subtracted curvature, no source/species filter or halfspin repair, no Nature prediction'])
    mechanical=oscillator(p['geometry_cut']);extended=oscillator(p['geometry_cut']+2)
    selected=[extended['basis'].index(n) for n in mechanical['basis']]
    exact=sum(a@a for a in extended['P'])[np.ix_(selected,selected)]/(2*p['reduced_mass']*p['length']**2)
    gq=quadrature(mechanical,p['length'],p['radial_quadrature']);actual=gq['u']@gq['kinetic'](p['reduced_mass'])@gq['u'].conj().T
    diagonal=np.diag([(sum(n)+1.5)/(2*p['reduced_mass']*p['length']**2) for n in mechanical['basis']])
    pp=sum(a@a for a in mechanical['P'])/(2*p['reduced_mass']*p['length']**2)
    zero=mechanical['basis'].index((0,0,0));entries={}
    for n in [(2,0,0),(0,2,0),(0,0,2)]:
        if n in mechanical['basis']:entries[str(n)]=float(actual[zero,mechanical['basis'].index(n)].real)
    result['mechanical_first_cause']=dict(actual_geometry_cut=p['geometry_cut'],continuum_gradient_vs_extended_Pp2P=magnitude(actual-exact),legacy_Tdiag_defect_on_NEW_operator=magnitude(actual-diagonal),PpP_squared_defect=magnitude(actual-pp),actual_retained_zero_to_two_entries=entries,supplier_kinetic='Same continuum-gradient quadrature, not frozen Noether diagonal formula',contact_mapping='NEW physical C=+d2H/dAext2; frozen legacy Noether contact=-d2H/dAext2. Signs are mapped, not weakened.')
    result['regulator']=dict(bare_E_eigenvalues=np.linalg.eigvalsh(model['field']['E']).real.tolist(),field_CCR_defects=[magnitude(comm(a,b)-1j*np.eye(nf)) for a,b in zip(model['field']['Q'],model['field']['P'])])
    result['runtime']=dict(H_nonzeros=h.nnz,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,H_Hermiticity_Frobenius=magnitude(h-h.conj().T),sparse_roundoff=model['sparse_roundoff'],note='Same full cutoff2/state space; NEW explicit primitive constructor receives ordinary dimension-relative CSR factory, no global rebinding or monkeypatch. Eigenspaces stream; removed finite-precision coefficient norms are recorded.')
    result['energy_semantics']=dict(supplier_energy_excess='London-coordinate bare Hs diagnostic, not physical mechanical energy. Same qualification applies to supplier_energy_change / actual_supplier_energy_change in omission/gradient controls.',physical_supplier_mechanical_energy_excess='Actual Udag Hs_original(A) U = Hs+AP minimal coupling, with continuum AP² products retained. Shared longitudinal cross-density separately belongs to whole H.',invariant_supplier_r2_excess='r² commutes with London U, genuine same-state supplier receiver.')
    result['source']['B_commutes_dipole_max']=max(magnitude(comm(b,d)) for b in model['sources'] for d in model['dipole'])
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),ground=ground,source=source,moving=moving,kicked=kicked,event_states=np.array(states),baseline_states=np.array(baselines),times=p['times'],**saved)
    print(json.dumps(dict(dims=model['dims'],neutral_J2=result['occupied']['J2'],source_J2=result['source']['J2'],addition_gap=spectral['addition']['lowest_gap'],H_J=result['Noether']['H_J'],BH_norm=result['gauge']['Born_Huang_H_norm'],source_adverse=result['gauge']['covariant_vs_original_current_ground'],gradient_coupling=controls['pure_gradient_static_comoving_basis']['matter_q_coupling_norm'],gradient_coordinate_bare_Hs_reaction=controls['pure_gradient_static_comoving_basis']['actual_supplier_energy_change'],final_invariant_supplier_r2_excess=samples[-1]['invariant_supplier_r2_excess'],final_physical_mechanical_energy_excess=samples[-1]['physical_supplier_mechanical_energy_excess'],final_radius_excess=samples[-1]['radius_excess'])))


if __name__=='__main__':main()
