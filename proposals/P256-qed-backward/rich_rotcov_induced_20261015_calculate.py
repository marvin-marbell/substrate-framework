"""Exact retained-sector source/virtuals and independent field-only event."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye, csr_matrix, kron
from scipy.sparse.linalg import expm_multiply, norm
from rich_rotcov_induced_20261015_model import build, shifted_h, profiles, supplier, comm, oscillator, symbasis


def mean(v,a):return float(np.vdot(v,a@v).real)
def magnitude(a):return float(norm(a)) if hasattr(a,'tocsr') else float(np.linalg.norm(a))
def groups(e,tol=2e-7):
    result=[]
    for i,x in enumerate(e):
        if not result or abs(x-e[result[-1][0]])>tol:result.append([i])
        else:result[-1].append(i)
    return result


def spectrum(model,n,probes=None,ground_only=False):
    """Every retained M block. No source/current virtual or species filtering."""
    h=model['H'];numbers=model['N'].diagonal().real;jz=model['J'][2].diagonal().real
    selected=np.flatnonzero(abs(numbers-n)<1e-7);records=[]
    for m in np.unique(np.round(jz[selected]*2)/2):
        idx=selected[abs(jz[selected]-m)<1e-6];sub=h[idx][:,idx].toarray();e,u=eigh(sub,check_finite=False,overwrite_a=True)
        if ground_only:
            v=np.zeros(h.shape[0],complex);v[idx]=u[:,0];records.append((float(e[0]),m,v))
        else:
            amps=u.conj().T@probes[idx] if probes is not None else None
            records.append((e,m,idx,u,amps))
    if ground_only:return min(records,key=lambda r:r[0])
    return records


def inclusive(v,model):
    # All primitive z generators were diagonalized by changes of basis only.
    systems=[model['S'][2],model['L'][2]]+[f[2] for f in model['SL']+model['F']]
    labels=np.array([a.diagonal().real for a in systems]);prob=abs(v)**2;result={}
    for i in np.flatnonzero(prob>1e-12):
        key=','.join(str(round(x,6)) for x in labels[:,i]);result[key]=result.get(key,0.)+float(prob[i])
    totals={}
    for key,value in result.items():
        m=str(round(sum(map(float,key.split(','))),6));totals[m]=totals.get(m,0.)+value
    return dict(order=['body_spin','body_orbital','supplier1_orbital','supplier2_orbital','TE_angular','TM_angular'],joint=result,total_Jz=totals,probability_sum=float(np.sum(prob)))


def supplier_response(p):
    s=supplier(p);e,u=eigh(s['H']);v=u[:,0];results=[]
    for a in range(6):
        b=s['linear'][a];contact=2*s['quadratic'][a][a];amps=u.conj().T@(b@v)
        para=float(2*np.sum(abs(amps[1:])**2/(e[0]-e[1:])))
        dia=mean(v,contact);step=1e-3
        direct=(eigh(s['H']+step*b+step**2*contact/2,eigvals_only=True)[0]+eigh(s['H']-step*b+step**2*contact/2,eigvals_only=True)[0]-2*e[0])/step**2
        results.append(dict(family='TE' if a<3 else 'TM',m_cartesian=a%3,diamagnetic=dia,signed_all_virtual=para,curvature=dia+para,direct_curvature=float(direct),current_brightness=float(np.sum(abs(amps[1:])**2))))
    state=v.reshape(s['primitive_dimension'],s['primitive_dimension']);rho=state@state.conj().T
    return dict(ground_energy=float(e[0]),energies=e.tolist(),one_flavour_purity=float(np.trace(rho@rho).real),correlation_energy=mean(v,s['coulomb']),channels=results,retained_virtual_count=len(e)-1)


def main():
    inp=Path(sys.argv[1]);out=Path(sys.argv[2]);p=json.loads(inp.read_text());model=build(p);h=model['H'];dim=h.shape[0];j=model['J'];j2=sum(a@a for a in j);ident=eye(dim,format='csr')
    e0,m0,ground=spectrum(model,2,ground_only=True)
    created_raw=model['sources'][0]@ground;source_strength=float(np.vdot(created_raw,created_raw).real);created=created_raw/np.sqrt(source_strength)
    sourceplus=np.column_stack([a@ground for a in model['sources']]);sourceminus=np.column_stack([a.conj().T@ground for a in model['sources']])
    spectral={};saved={};weights={}
    for n,probes,label in [(3,sourceplus,'addition'),(1,sourceminus,'removal')]:
        blocks=spectrum(model,n,probes);energies=[];amplitudes=[];poles=[]
        for es,m,idx,u,amps in blocks:
            energies.extend(es);amplitudes.extend(amps)
            for group in groups(es):
                residue=float(np.sum(abs(amps[group])**2))
                if residue>1e-9:
                    sub=u[:,group];js=j2[idx][:,idx];angular=np.linalg.eigvalsh(sub.conj().T@(js@sub)).real
                    poles.append(dict(gap=float(es[group[0]]-e0),M=float(m),multiplicity=len(group),strength=residue,J2=angular.tolist()))
        energies=np.array(energies);amplitudes=np.array(amplitudes);order=np.argsort(energies);energies=energies[order];amplitudes=amplitudes[order]
        gram=amplitudes.conj().T@amplitudes;weights[label]=gram
        spectral[label]=dict(retained_states=len(energies),lowest_sector_gap=float(energies[0]-e0),bright_poles=sorted(poles,key=lambda x:x['gap'])[:24],gram_real=gram.real.tolist(),gram_imag=gram.imag.tolist())
        saved[label+'_energies']=energies;saved[label+'_amplitudes']=amplitudes
        del blocks
    current_probes=np.column_stack([a@ground for a in model['current']]+[a@ground for a in model['Q']]);blocks=spectrum(model,2,current_probes)
    response=[];signed=np.zeros(6);contact=np.array([mean(ground,model['contact'][a][a]) for a in range(6)])
    for a in range(6):
        rows=[]
        for es,m,idx,u,amps in blocks:
            for k,energy in enumerate(es):
                gap=float(energy-e0);weight=float(abs(amps[k,a])**2);fieldweight=float(abs(amps[k,a+6])**2)
                if gap>1e-8:
                    signed[a]-=2*weight/gap
                    rows.append((gap,weight,fieldweight))
        response.append(dict(family='TE' if a<3 else 'TM',channel=a%3,current_total=float(sum(r[1] for r in rows)),first_current_bright_gap=next((r[0] for r in sorted(rows) if r[1]>1e-9),None),first_field_bright_gap=next((r[0] for r in sorted(rows) if r[2]>1e-9),None),contact=float(contact[a]),all_signed_virtual=float(signed[a]),static_curvature=float(contact[a]+signed[a])))
    del blocks
    # Actual spatial profiles and both angular families, not merely two labels.
    from scipy.spatial.transform import Rotation
    rot=Rotation.from_rotvec(np.array([.2,-.3,.4])).as_matrix();x=np.array([[.2,.3,.1],[-.4,.2,.5]])
    a,metric=profiles(x,p['field_width'],p['epsilon']);ar,unused=profiles(x@rot.T,p['field_width'],p['epsilon']);profile_error=[]
    for family in range(2):
        original=np.einsum('a,xai->xi',np.array([.4,-.2,.3]),a[:,3*family:3*family+3]);transformed=np.einsum('a,xai->xi',rot@np.array([.4,-.2,.3]),ar[:,3*family:3*family+3]);profile_error.append(magnitude(transformed-original@rot.T))
    derivative=[];second=[];step=1e-3
    for a in range(6):
        ext=np.zeros(6);ext[a]=step;hp=shifted_h(model,ext);hm=shifted_h(model,-ext)
        derivative.append(magnitude((hp-hm)/(2*step)+model['current'][a])/max(1.,magnitude(model['current'][a])))
        second.append(magnitude((hp+hm-2*h)/step**2-model['contact'][a][a])/max(1.,magnitude(model['contact'][a][a])))
    moving=expm_multiply(-1j*p['displacement']*model['PX'][0],created);moving=expm_multiply(1j*p['boost']*model['X'][1],moving)
    kick=model['P'][1]-model['Q'][2]+.7*model['P'][4]-.4*model['Q'][5]
    kicked=expm_multiply(-1j*p['kick']*kick,moving);samples=[];states=[];baseline_states=[]
    def embed(axis,operator):
        result=csr_matrix([[1.]])
        for a,n in enumerate(model['dims']):result=kron(result,csr_matrix(operator) if a==axis else eye(n),format='csr')
        return result
    gu=symbasis(model['geometry'])
    body_edge=embed(2,gu.conj().T@np.diag([sum(n)==p['geometry_cut'] for n in model['geometry']['basis']])@gu)
    su=symbasis(oscillator(p['supplier_cut']));ne=len(su)
    edge=su.conj().T@np.diag([sum(n)==p['supplier_cut'] for n in oscillator(p['supplier_cut'])['basis']])@su
    supplier_edges=[embed(3,np.kron(edge,np.eye(ne))),embed(3,np.kron(np.eye(ne),edge))]
    fu=model['field']['u'];field_edge=embed(4,fu.conj().T@np.diag([sum(n)==p['field_cut'] for n in model['field']['basis']])@fu)
    keys={'S':model['S'],'L':model['L'],'supplier1':model['SL'][0],'supplier2':model['SL'][1],'TE':model['F'][0],'TM':model['F'][1]}
    for t in p['times']:
        state=expm_multiply(-1j*t*h,kicked);baseline=expm_multiply(-1j*t*h,moving);states.append(state);baseline_states.append(baseline)
        angular={key:[mean(state,a) for a in ops] for key,ops in keys.items()};excess={key:[mean(state,a)-mean(baseline,a) for a in ops] for key,ops in keys.items()}
        record=dict(time=t,norm=float(np.vdot(state,state).real),energy_drift=mean(state,h)-mean(kicked,h),total_J=[mean(state,a) for a in j],angular=angular,angular_excess=excess,torques={key:[mean(state,1j*comm(h,a)) for a in ops] for key,ops in keys.items()},torque_balance=[mean(state,1j*comm(h,a)) for a in j],radius=mean(state,model['radius']),radius_excess=mean(state,model['radius'])-mean(baseline,model['radius']),supplier_r2=[mean(state,a) for a in model['supplier_r2']],supplier_r2_excess=[mean(state,a)-mean(baseline,a) for a in model['supplier_r2']],supplier_energy_excess=mean(state,model['supplier_energy'])-mean(baseline,model['supplier_energy']),field_energy_excess=mean(state,model['field_energy'])-mean(baseline,model['field_energy']),current_excess=[mean(state,a)-mean(baseline,a) for a in model['current']],inclusive=inclusive(state,model))
        tensor=state.reshape(model['dims']);fieldrho=np.einsum('abcdi,abcdj->ij',tensor,tensor.conj(),optimize=True);record['field_purity']=float(np.trace(fieldrho@fieldrho).real)
        record['cutoff_population']=dict(body_orbital=mean(state,body_edge),supplier_orbital=[mean(state,a) for a in supplier_edges],field=mean(state,field_edge),two_point_shape_edges=1.)
        samples.append(record)
    # Supplier-off is a different NEW law and is propagated, not interpreted as
    # the same ground. Keep exactly the same independently prepared event.
    poff=dict(p,supplier_coupling=0.);off=build(poff);offstate=expm_multiply(-1j*p['times'][-1]*off['H'],kicked)
    controls=dict(supplier_off_same_preparation_final=dict(supplier_energy_change=mean(offstate,off['supplier_energy'])-mean(kicked,off['supplier_energy']),field_energy_change=mean(offstate,off['field_energy'])-mean(kicked,off['field_energy'])),bare_E_eigenvalues=np.linalg.eigvalsh(model['field']['E']).tolist(),supplier_only=supplier_response(p),uncorrelated_supplier=supplier_response(dict(p,correlation=0.)),constant_A_pure_gauge_regulator_control=supplier_response(dict(p,supplier_path='constant_gauge')))
    controls['supplier_path_omission']={}
    for path in ['tangent','omit_TE']:
        pc=dict(p,supplier_path=path);control=build(pc)
        final=expm_multiply(-1j*p['times'][-1]*control['H'],kicked)
        controls['supplier_path_omission'][path]=dict(response=supplier_response(pc),same_preparation_final_supplier_energy_change=mean(final,control['supplier_energy'])-mean(kicked,control['supplier_energy']),same_preparation_final_field_energy_change=mean(final,control['field_energy'])-mean(kicked,control['field_energy']),state_distance_from_full=float(np.linalg.norm(final-states[-1])))
    ccr=[magnitude(comm(q,r)-1j*np.eye(model['dims'][4])) for q,r in zip(model['field']['Q'],model['field']['P'])]
    centre=expm_multiply(-2j*np.pi*j[2],created)
    result=dict(kind='NEW_JOINED_ROTCOV_CORRELATED_SUPPLIER_PRIMITIVE_E_FINITE_CONSTRUCTION',dimensions=model['dims'],parameters=p,occupied=dict(energy=e0,M=m0,J2=mean(ground,j2),J2_residual=magnitude(j2@ground),charge=mean(ground,model['N'])-2),source=dict(raw_strength=source_strength,unfiltered_J2=mean(created,j2),moving_J2=mean(moving,j2),charged_2pi_minus=magnitude(centre+created),spectral_CAR=magnitude(weights['addition']+weights['removal'].T-np.eye(4)),spectral=spectral),rotation=dict(H_J_relative=[magnitude(comm(h,a))/max(1.,magnitude(h)) for a in j],angular_algebra=[magnitude(comm(j[a],j[(a+1)%3])-1j*j[(a+2)%3]) for a in range(3)],spatial_TE_TM_covariance=profile_error,N_conservation=magnitude(comm(h,model['N'])),frame_only_failure=[magnitude(comm(h,a))/magnitude(h) for a in model['S']]),EM=dict(electric_metrics=metric.tolist(),primitive_E_only=True,installed_Maxwell_stiffness=False,current_derivative=derivative,contact_derivative=second,full_neutral_retained_virtuals=int(np.sum(abs(model['N'].diagonal().real-2)<1e-7)),response=response,CCR_frobenius_defects=ccr),event=dict(preparation='Unfiltered new charged source; physical body translation/boost; independent TE+TM field-only unitary; no species or angular filtering',external_J_impulse=[mean(kicked,a)-mean(moving,a) for a in j],samples=samples),controls=controls,limits=['Primitive charged Z2 bundle/CAR8 supplier and CAR4 frame, shell inertia/core mixing, bindings and neutral fixed ions are inputs, not Nature predictions','Supplier is the full one-particle-per-distinguishable-flavour sector of scalar CAR8 with s+p complete spatial shells; Coulomb softened and angular isotropically integrated','Two full j=1 transverse TE/TM families all m; finite localized profiles are NOT bulk or asymptotic photons','Primitive E oscillator-Galerkin regulator has bare discrete gaps and noncanonical CCR; neither counts as induced Maxwell or masslessness','No free COM, continuum/cutoff convergence, selfbinding, Lorentz/Dirac dynamics, species-purified halfspin or Nature claim','Uniform shell two internal channels co-located at same radius; local core mixing has zero geometric Peierls path; supplier uses actual 3D radial wavefunction motion','Fixed neutral ions and bindings define an external central relative-geometry model; total physical SO3 is retained, full translational invariance is not asserted'])
    frame=model['frame'];covariance=[]
    for a in range(3):
        rep=np.array([[np.trace(d.conj().T@comm(frame['S'][a],e))/8 for d in frame['source']] for e in frame['source']])
        for b,d in enumerate(model['sources']):covariance.append(magnitude(comm(j[a],d)-sum(rep[b,c]*model['sources'][c] for c in range(4))))
    result['source']['tensor_covariance_max']=max(covariance)
    result['source']['unfiltered_J2_residual']=magnitude((j2-.75*ident)@created)
    result['regulator']=dict(field_HE_equation_defect_operator=[magnitude(1j*comm(h,model['P'][a])-model['current'][a]) for a in range(6)],field_HE_equation_defect_ground=[magnitude((1j*comm(h,model['P'][a])-model['current'][a])@ground) for a in range(6)],bare_qdot_minus_P_ground=[magnitude((1j*comm(model['field_energy'],model['Q'][a])-model['P'][a])@ground) for a in range(6)],interpretation='Physical derivative current is distinct from truncated canonical Heisenberg source; CCR and local gauge Ward errors remain explicit. Total retained Noether angular balance is exact, continuum Maxwell/source equations are NOT claimed.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),ground=ground,source=created,moving=moving,kicked=kicked,event_states=np.array(states),baseline_states=np.array(baseline_states),times=p['times'],**saved)
    print(json.dumps(dict(dims=model['dims'],neutral_J2=result['occupied']['J2'],source_J2=result['source']['unfiltered_J2'],rotation=result['rotation']['H_J_relative'],addition_gap=spectral['addition']['lowest_sector_gap'],supplier_purity=controls['supplier_only']['one_flavour_purity'],TE_TM_supplier_curvature=[r['curvature'] for r in controls['supplier_only']['channels']],final_supplier_energy_excess=samples[-1]['supplier_energy_excess'],final_radius_excess=samples[-1]['radius_excess'])))


if __name__=='__main__':main()
