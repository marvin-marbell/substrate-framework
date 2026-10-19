"""NEW physical pullback, full occupied/source poles and mechanical EM event."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from math import factorial
import numpy as np
from scipy.linalg import eigh, expm
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply, norm as sparse_norm
from scipy.special import eval_hermite
from rich_frame_holonomy_20261014_model import angular
from rich_physical_noether_20261014_model import build, shifted_h, geometry_quadrature, rotation_geometry, sphere, electromagnetic_profile


def norm(a):
    return float(sparse_norm(a)) if hasattr(a,'tocsr') else float(np.linalg.norm(a))


def mean(v,a):
    return float(np.vdot(v,a@v).real)


def sector(model,n):
    numbers=np.real(model['N'].diagonal())
    index=np.flatnonzero(np.abs(numbers-n)<1e-7)
    values,vectors=eigh(model['H'][index][:,index].toarray())
    return index,values,vectors


def embed(v,index,dimension):
    out=np.zeros(dimension,complex); out[index]=v
    return out


def groups(values,tolerance=2e-8):
    result=[]
    for i,value in enumerate(values):
        if not result or abs(value-values[result[-1][0]])>tolerance:result.append([i])
        else:result[-1].append(i)
    return result


def pullback(osc,p,rotation):
    points,integrate=geometry_quadrature(osc,p)
    moved=points@rotation
    original=np.ones((len(points),len(osc['basis'])))
    rotated=original.copy()
    for col,ns in enumerate(osc['basis']):
        for axis,n in enumerate(ns):
            scale=np.sqrt(2**n*factorial(n))
            original[:,col]*=eval_hermite(n,points[:,axis]/p['length'])/scale
            rotated[:,col]*=eval_hermite(n,moved[:,axis]/p['length'])/scale
    # integrate() already includes the original polynomial on both sides.
    # Recover quadrature weights by integrating independent scalar monomials
    # through the first (vacuum) column, then use its exact Gram action.
    result=np.column_stack([integrate(rotated[:,col])[:,0] for col in range(len(osc['basis']))])
    return result


def frame_pullback(frame,axis,angle):
    fundamental=expm(-1j*angle*sum(axis[a]*angular(.5)[a] for a in range(3)))
    symmetric=np.zeros((4,3),complex); symmetric[0,0]=1; symmetric[1,1]=symmetric[2,1]=1/np.sqrt(2); symmetric[3,2]=1
    one=symmetric.conj().T@np.kron(fundamental,fundamental)@symmetric
    blocks=[np.ones((1,1)),fundamental.conj(),one.conj()]
    raw=np.zeros((14,14),complex); start=0
    for block in blocks:
        size=len(block); raw[start:start+size*size,start:start+size*size]=np.kron(block,np.eye(size)); start+=size*size
    physical=frame['P'].conj().T@np.kron(raw,np.eye(16))@frame['P']
    return physical,np.kron(raw,np.eye(16))


def inclusive(v,model):
    systems=[eigh(model['frame']['S'][2]),eigh(model['geometry']['L'][2]),eigh(model['field']['L'][2])]
    array=v.reshape(16,model['nb'],model['ng'],model['nf'])
    amplitudes=np.einsum('ai,abgf,gj,fk->ibjk',systems[0][1].conj(),array,systems[1][1].conj(),systems[2][1].conj(),optimize=True)
    probabilities=np.sum(np.abs(amplitudes)**2,axis=1)
    joint={}; total={}
    for i,sv in enumerate(systems[0][0]):
        for j,lv in enumerate(systems[1][0]):
            for k,fv in enumerate(systems[2][0]):
                prob=float(probabilities[i,j,k]); label=','.join(f'{x:.1f}' for x in (sv,lv,fv))
                joint[label]=joint.get(label,0.)+prob
                labelj=f'{sv+lv+fv:.1f}'; total[labelj]=total.get(labelj,0.)+prob
    return dict(joint_Sz_Lz_Fz={k:v for k,v in joint.items() if v>1e-10},total_Jz={k:v for k,v in total.items() if v>1e-10},probability_sum=float(np.sum(probabilities)))


def main():
    input_path=Path(sys.argv[1]); output=Path(sys.argv[2]); p=json.loads(input_path.read_text())
    model=build(p); h=model['H']; dim=h.shape[0]; frame=model['frame']; identity=eye(dim,format='csr')
    j=model['J']; j2=sum(a@a for a in j)
    index0,e0,v0=sector(model,2); indexp,ep,vp=sector(model,3); indexm,em,vm=sector(model,1)
    ground=embed(v0[:,0],index0,dim); reference_energy=float(e0[0])
    # Entire actual source, never a spin/energy-filtered initial state.
    created=model['source'][0]@ground; created/=np.linalg.norm(created)
    raw=frame['P']
    car=max(norm(a@b.conj().T+b.conj().T@a-(np.eye(16) if x==y else 0)) for x,a in enumerate(frame['source']) for y,b in enumerate(frame['source']))
    raw_gauss=max(norm(g@raw) for g in frame['G'])
    source_gauss=max(norm(g@d@raw) for g in frame['G'] for d in frame['raw_source'])
    bare_gram=norm(sum((raw.conj().T@d@raw).conj().T@(raw.conj().T@d@raw) for d in frame['bare']))
    commutators=[norm(h@a-a@h)/max(1.,norm(h)) for a in j]
    angular_algebra=[norm(j[a]@j[(a+1)%3]-j[(a+1)%3]@j[a]-1j*j[(a+2)%3]) for a in range(3)]
    axis=np.array([.3,.4,np.sqrt(.75)]); angle=.731
    geom=rotation_geometry(np.array([[.2,.3,-.1]]),sphere(4,8)[0],np.eye(2,dtype=complex),p['radii'][1],axis,angle)
    test_points=np.array([[.2,.3,-.1],[-.5,.1,.4],[.7,-.2,.3]])
    amplitude=np.array([.3,-.2,.6]); rotation=geom['rotation']
    avec,bvec=electromagnetic_profile(test_points,amplitude,p['field_width'])
    anew,bnew=electromagnetic_profile(test_points@rotation.T,rotation@amplitude,p['field_width'])
    actual_field_covariance=dict(A=norm(anew-avec@rotation.T),B=norm(bnew-bvec@rotation.T))
    # Right gauge rotates raw colour/material labels, not the physical shell.
    u=expm(.27j*np.array([[0,1],[1,0]])); gauge=expm(.39j*np.diag([1,-1]))
    colour=np.array([.4+.2j,-.3+.1j]); epsilon=np.array([[0,1],[-1,0]])
    density=np.outer(colour,colour.conj()); matrix=u@density@u.conj().T
    gauged=(u@gauge.conj().T)@(gauge@density@gauge.conj().T)@(gauge@u.conj().T)
    dressed=u@epsilon@colour.conj()
    gauged_source=(u@gauge.conj().T)@epsilon@(gauge@colour).conj()
    directions,unused=sphere(4,8)
    pauli=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    right_rotation=np.array([[np.trace(pauli[a]@gauge@pauli[b]@gauge.conj().T).real/2 for b in range(3)] for a in range(3)])
    before_routes=rotation_geometry(test_points,directions,u,p['radii'][1],axis,0.)['routes']
    gauge_routes=rotation_geometry(test_points,directions@right_rotation.T,u@gauge.conj().T,p['radii'][1],axis,0.)['routes']
    rotated_routes=rotation_geometry(test_points,directions,u,p['radii'][1],axis,angle)['routes']
    rotation_protocol=dict(raw_colour_gauge_source=norm(gauged_source-dressed),raw_colour_gauge_matrix=norm(gauged-matrix),raw_colour_gauge_routes=norm(gauge_routes-before_routes),physical_routes=norm(rotated_routes-before_routes@rotation.T),physical_matrix=norm((geom['lift']@u)@density@(geom['lift']@u).conj().T-geom['lift']@matrix@geom['lift'].conj().T),physical_source=norm((geom['lift']@u)@epsilon@colour.conj()-geom['lift']@dressed))
    ug=pullback(model['geometry'],p,geom['rotation'])
    fp=dict(p,length=1.)
    uf=pullback(model['field'],fp,geom['rotation'])
    ur,rawrot=frame_pullback(frame,axis,angle)
    pull=kron(csr_matrix(ur),kron(eye(model['nb']),kron(csr_matrix(ug),csr_matrix(uf))),format='csr')
    generator=sum(axis[a]*j[a] for a in range(3))
    rng=np.random.default_rng(1822); probe=rng.normal(size=dim)+1j*rng.normal(size=dim); probe/=np.linalg.norm(probe)
    rotation_residual=norm(pull@probe-expm_multiply(-1j*angle*generator,probe))
    vector_residuals={}
    for name in ['X','PX','Q','P']:
        ops=model[name]
        # Udag V U = R V for state rotation U=exp(-i theta J).
        vector_residuals[name]=max(norm(pull.conj().T@ops[a]@pull-sum(geom['rotation'][a,b]*ops[b] for b in range(3))) for a in range(3))
    centre=expm_multiply(-2j*np.pi*j[2],created)
    neutral_centre=expm_multiply(-2j*np.pi*j[2],ground)
    # Extract the genuine raw source tensor action, then test it on the full model.
    source_rep=[]; source_covariance=[]
    for a in range(3):
        rep=np.array([[np.trace(d.conj().T@(frame['S'][a]@e-e@frame['S'][a]))/8 for d in frame['source']] for e in frame['source']])
        source_rep.append(rep)
        for b,d in enumerate(model['source']):
            source_covariance.append(norm(j[a]@d-d@j[a]-sum(rep[b,c]*model['source'][c] for c in range(4))))
    source_j2=mean(created,j2); source_variance=norm((j2-source_j2*identity)@created)
    neutral_j2=mean(ground,j2)
    # Full adjacent-sector Lehmann response from this occupied reference.
    addition=np.column_stack([vp.conj().T@(d@ground)[indexp] for d in model['source']])
    removal=np.column_stack([vm.conj().T@(d.conj().T@ground)[indexm] for d in model['source']])
    z=p['spectral_z'][0]+1j*p['spectral_z'][1]
    green=addition.conj().T@((1/(z-(ep-reference_energy)))[:,None]*addition)
    green+=(removal.conj().T@((1/(z+(em-reference_energy)))[:,None]*removal)).T
    poles=[]
    j2p=j2[indexp][:,indexp]
    for group in groups(ep):
        residue=float(np.sum(np.abs(addition[group])**2))
        sub=vp[:,group]; angular_values=np.linalg.eigvalsh(sub.conj().T@(j2p@sub))
        if residue>1e-8 or len(poles)<2:
            poles.append(dict(gap=float(ep[group[0]]-reference_energy),multiplicity=len(group),source_residue=residue,J2_eigenvalues=angular_values.real.tolist()))
        if len(poles)>=12:break
    current_weights=[]
    for current in model['current']:
        amps=v0.conj().T@(current@ground)[index0]
        current_weights.append(dict(total=float(np.sum(np.abs(amps[1:])**2)),first_bright_gap=next((float(e0[i]-reference_energy) for i in range(1,len(e0)) if abs(amps[i])**2>1e-9),None)))
    derivative_step=2e-5; derivative=[]
    for a in range(3):
        ext=np.zeros(3); ext[a]=derivative_step
        numerical=(shifted_h(model,ext)-shifted_h(model,-ext))/(2*derivative_step)
        derivative.append(norm(numerical+model['current'][a])/max(1.,norm(model['current'][a])))
    # Physical geometry preparation is unitary and has no angular/pole filter.
    moving=expm_multiply(-1j*p.get('displacement',0.)*model['PX'][0],created)
    moving=expm_multiply(1j*p.get('boost',0.)*model['X'][1],moving)
    # The subsequent EM-only displacement is the measured external torque.
    kick_generator=model['P'][1]-model['Q'][2]
    kicked=expm_multiply(-1j*p['kick']*kick_generator,moving)
    initial_j=[mean(moving,a) for a in j]; after_j=[mean(kicked,a) for a in j]
    initial_energy=mean(kicked,h); samples=[]; states=[]
    for t in p['times']:
        state=expm_multiply(-1j*t*h,kicked); baseline=expm_multiply(-1j*t*h,moving)
        states.append(state)
        record=dict(time=t,norm=float(np.vdot(state,state).real),energy=mean(state,h),energy_drift=mean(state,h)-initial_energy,
                    total_J=[mean(state,a) for a in j],spin=[mean(state,a) for a in model['S']],orbital=[mean(state,a) for a in model['L']],field_angular=[mean(state,a) for a in model['F']],
                    centre=[mean(state,a) for a in model['X']],radius=mean(state,model['radius']),field_energy_excess=mean(state,model['field_energy'])-mean(baseline,model['field_energy']),
                    physical_current=[mean(state,a) for a in model['current']],physical_current_excess=[mean(state,a)-mean(baseline,a) for a in model['current']],inclusive=inclusive(state,model))
        amplitudes=state.reshape(16,model['nb'],model['ng'],model['nf'])
        geom_edge=np.array([sum(n)==p['geometry_cut'] for n in model['geometry']['basis']])
        field_edge=np.array([sum(n)==p['field_cut'] for n in model['field']['basis']])
        record['cutoff_population']=dict(geometry=float(np.sum(np.abs(amplitudes[:,:,geom_edge,:])**2)),field=float(np.sum(np.abs(amplitudes[:,:,:,field_edge])**2)),shape_edges=float(np.sum(np.abs(amplitudes[:,[0,-1],:,:])**2)))
        record['centre_baseline']=[mean(baseline,a) for a in model['X']]
        record['geometry_velocity']=[mean(state,1j*(h@a-a@h)) for a in model['X']]
        record['radial_charge_current']=mean(state,model['radial_current'])
        record['torque_balance']=[mean(state,1j*(h@a-a@h)) for a in j]
        record['torques']={key:[mean(state,1j*(h@a-a@h)) for a in model[key]] for key in ['S','L','F']}
        record['field_energy_rate']=mean(state,1j*(h@model['field_energy']-model['field_energy']@h))
        samples.append(record)
    c_error=norm(model['C']@h@model['C'].conj().T-h)/norm(h)
    radial_continuity=norm(1j*(h@model['Nt'][0]-model['Nt'][0]@h)+model['radial_current'])
    integer_frame=np.diag([float(abs(j-round(j))<1e-9) for j,_,_ in frame['basis']])
    odd_columns=np.flatnonzero([state.bit_count()%2 for state in range(16)])
    trivial_bundle_odd_weight=norm(np.kron(integer_frame,np.eye(16))@frame['P'][:,odd_columns])
    noether_adverse=[norm(h@a-a@h)/norm(h) for a in model['S']]
    result=dict(candidate='charged-material-shell-physical-noether-bundle',dimensions=dict(full=dim,raw_frame=224,physical_CAR=16,geometry=model['ng'],shape=model['nb'],field=model['nf'],neutral=len(index0),addition=len(indexp),removal=len(indexm)),
                raw=dict(Gauss=raw_gauss,source_Gauss=source_gauss,bare_physical_gram=bare_gram,CAR=car,rotation_preserves_Gauss=max(norm(g@rawrot@raw) for g in frame['G'])),
                physical_rotation=dict(H_J_relative=commutators,angular_algebra=angular_algebra,pullback_versus_generator=rotation_residual,vector_covariance=vector_residuals,actual_field_covariance=actual_field_covariance,configuration_protocol=rotation_protocol,raw_frame_pullback=norm(ur-expm(-1j*angle*sum(axis[a]*frame['S'][a] for a in range(3)))),charged_2pi_minus=norm(centre+created),neutral_2pi_plus=norm(neutral_centre-ground),source_covariance=max(source_covariance),source_rep_real=[a.real.tolist() for a in source_rep],source_rep_imag=[a.imag.tolist() for a in source_rep],geometric_route_rotation=geom['rotation'].tolist(),frame_only_rotation_fails_H=noether_adverse,trivial_SO3_bundle_odd_weight=trivial_bundle_odd_weight),
                occupied=dict(energy=reference_energy,ground_multiplicity=len(groups(e0)[0]),J2=neutral_j2,J2_variance=norm((j2-neutral_j2*identity)@ground),charge=mean(ground,model['N'])-2),
                charged=dict(unfiltered_source_J2=source_j2,unfiltered_source_J2_variance=source_variance,lowest_addition_gap=float(ep[0]-reference_energy),lowest_removal_gap=float(em[0]-reference_energy),source_poles=poles,addition_weight=(addition.conj().T@addition).real.tolist(),spectral_CAR_error=norm(addition.conj().T@addition+(removal.conj().T@removal).T-np.eye(4)),G_real=green.real.tolist(),G_imag=green.imag.tolist()),
                mechanical_EM=dict(metric=model['metric'],stiffness=model['stiffness'],omega=model['omega'],same_H_current_derivative=derivative,current_pair_response=current_weights,translation_current_norm=[norm(a) for a in model['translation_current']],rotation_current_norm=[norm(a) for a in model['rotation_current']],radial_charge_continuity=radial_continuity,total_charge_continuity=norm(h@model['N']-model['N']@h),C_invariance=c_error),
                event=dict(preparation='Unfiltered charged source, unitary geometry displacement/boost, then EM-only exp[-i kick (P_y-Q_z)]',moving_source_J2=mean(moving,j2),moving_source_J2_variance=norm((j2-mean(moving,j2)*identity)@moving),geometry_preparation_angular_impulse=(np.array(initial_j)-[mean(created,a) for a in j]).tolist(),initial_total_J=initial_j,postkick_total_J=after_j,external_angular_impulse=(np.array(after_j)-initial_j).tolist(),samples=samples),
                limits=['Z2 charged line bundle, primitive scalar CAR/colour, material inertia, binding/channel/shape law supplied; not derived Nature electron','Relative-centre dynamics only; free total CM factor removed; no relativistic dispersion or selfbinding-origin claim','Finite complete rotational oscillator shells and three-point radial shape; no cutoff convergence claim','One complete Gaussian toroidal l=1 Maxwell-mode triplet, not bulk/free photons','Finite field canonical commutators have boundary corrections, not falsely asserted canonical CCR','Actual C/current/pair/high virtuals retained within this declared single full CAR16 law'])
    output.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),neutral_ground=ground,unfiltered_charged_source=created,moving_source=moving,kicked=kicked,times=p['times'],event_states=np.array(states),neutral_energies=e0,addition_energies=ep,removal_energies=em,addition_amplitudes=addition,removal_amplitudes=removal)
    print(json.dumps(dict(dim=dim,rotation=rotation_residual,H_J=commutators,neutral_J2=neutral_j2,source_J2=source_j2,pole_gap=float(ep[0]-reference_energy),current_derivative=derivative,final_spin=samples[-1]['spin'],final_field=samples[-1]['field_angular'])))


if __name__=='__main__':main()
