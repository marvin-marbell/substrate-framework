"""Actual SAME extended-H nonstationary source/event/charge-phi-field POVM consumer."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,expm_multiply
from rich_spatial_colour_em_20261015_snapshot import load,source,charge_receiver,matrix
from rich_spatial_colour_em_20261015_model import assemble,angular,observable,MatterMap
from rich_spatial_colour_em_20261015_measure import drive,joint_receiver,field_action
from rich_spatial_colour_em_20261015_density import density,spatial_current,no_hop_law,mechanical_operator


def mean(state,operator):return float(np.vdot(state,operator@state).real)
def field_force_witness(h,operators,currents,state):
    """Actual Heisenberg force versus PRE-CUT physical current, exposing field cuts."""
    hstate=h@state;electric=operators['field_energy'];estate=electric@state;records=[]
    for mu,current in enumerate(currents):
        momentum=operators['Pi_field'+str(mu)];pstate=momentum@state
        force=1j*(h@pstate-momentum@hstate);physical=current@state
        bare_cut_force=1j*(electric@pstate-momentum@estate)
        records.append(dict(profile=mu,Heisenberg_force_norm=float(np.linalg.norm(force)),physical_current_image_norm=float(np.linalg.norm(physical)),force_minus_physical_current_defect=float(np.linalg.norm(force-physical)),bare_Pi_squared_cut_force_norm=float(np.linalg.norm(bare_cut_force))))
    return dict(profiles=records,meaning='Before field projection, Pi_dot=J=-partial H/partial Q for these canonical normalized profiles. Finite total-Hermite-shell projection need NOT preserve that canonical identity; even bare P Pi² P need not commute P Pi P. These actual defects are regulator evidence, not an extra force, a current patch or a photon claim.')




def propagate(h,state,dt,phase_energy):
    if not dt:return state.copy()
    def apply(vector):return h@vector-phase_energy*vector
    shifted=LinearOperator(h.shape,matvec=apply,matmat=apply,rmatvec=apply,dtype=complex)
    return expm_multiply(-1j*dt*shifted,state,traceA=-1j*dt*(h.trace()-phase_energy*h.shape[0]))


def scalar_measure(state,model,name):
    base=model['base'];scalar=base['scalar'];nm=base['operators'][name]['Q'].shape[0];x=state.reshape(nm,len(scalar['grid']),-1)
    marginal=np.sum(abs(x)**2,axis=(0,2));energy=np.vdot(x,np.einsum('fg,mgk->mfk',scalar['H'],x,optimize=True)).real
    return dict(phi=float(marginal@scalar['grid']),phi2=float(marginal@(scalar['grid']**2)),scalar_energy=float(energy),scalar_boundary=float(marginal[0]+marginal[-1]))


def measurement_operators(model,name,currents):
    base=model['base'];field=model['field'];rotor=base['rotor'];op=base['operators'][name];nr=rotor['dimension'];ne=field['dimension'];operators={}
    for key in ['Q','rho0','rho1','C','B','N','PW_boundary']:
        operators[key]=observable(model,name,matter=op[key])
    coulomb=((model['V_self']+model['V_cross'])*(op['Q']@op['Q'])+(model['V_self']-model['V_cross'])*op['density_delta2'])/4
    operators['Coulomb']=observable(model,name,matter=coulomb)
    operators['density_omitted_local_norm_squared']=observable(model,name,matter=(op['density_delta2']-op['density_delta']@op['density_delta'])/4)
    operators['field_energy']=observable(model,name,rq=sparse.kron(sparse.eye(nr),field['H_E']))
    operators['field_boundary']=observable(model,name,rq=sparse.kron(sparse.eye(nr),field['boundary']))
    operators['orientation_boundary']=observable(model,name,rq=sparse.kron(rotor['boundary'],sparse.eye(ne)))
    for mu in range(6):
        operators['Q_field'+str(mu)]=observable(model,name,rq=sparse.kron(sparse.eye(nr),field['Q'][mu]))
        operators['Pi_field'+str(mu)]=observable(model,name,rq=sparse.kron(sparse.eye(nr),field['Pi'][mu]))
        operators['current'+str(mu)]=currents[mu]
    for axis in range(3):
        operators['frame_S'+str(axis)]=observable(model,name,matter=op['S'][axis])
        operators['L_body'+str(axis)]=observable(model,name,rq=sparse.kron(rotor['L'][axis],sparse.eye(ne)))
        operators['F_field'+str(axis)]=observable(model,name,rq=sparse.kron(sparse.eye(nr),field['F'][axis]))
    return operators


def co_moving_continuity(h,rho0,routes,state):
    return dict(defect=float(np.linalg.norm(1j*(h@(rho0@state)-rho0@(h@state))+sum(route@state for route in routes))),route_image=float(np.linalg.norm(sum(route@state for route in routes))),meaning='dot Q_em,0 + sum directed physical route currents. No current is redefined from this commutator. Exact raw identity; measured retained-space defect is not suppressed.')


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent;started=time.time()
    state_json=root/request['state_json'];state_npz=root/request['state_npz'];metadata,data,model,laws=load(state_json,state_npz);base=model['base'];field=model['field'];p=model['parameters']
    times=np.array(request['times']);results={};stored={};reference=data['neutral_reference']
    input_hash={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [state_json,state_npz]}
    def checkpoint(stage):
        payload=dict(kind='ACTUAL_NEW_SPATIAL_CHARGE_PHI_COLOUR_QUANTUM_FIELD_REACTION',stage=stage,request=request,input_sha256=input_hash,parameters=p,results=results,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,protocol='UNFILTERED normalized raw full-source insertion; nonstationary paired baseline under identical H. This is not a source-pole protocol, and raw strength is not event probability.')
        out.write_text(json.dumps(payload,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)
    for case in request['cases']:
        name=case['name'];key=case.get('key');label=name if key is None else name+'_'+key;h=laws[name];op=base['operators'][name]
        model['parameters']=dict(p,field_kick=request['field_kick']*case.get('drive_sign',1.),drive_profile=request['drive_profile'])
        if key is None:initial=reference.copy();insertion=None;source_result={}
        else:
            insertion=source(data,model,name,key);raw=insertion@reference;strength=float(np.linalg.norm(raw)**2);initial=raw/np.sqrt(strength)
            metric=matrix(data,name+'_source_'+key+'_full_metric');full=mean(reference,observable(model,'neutral',matter=metric))
            source_result=dict(retained_raw_strength=strength,full_raw_strength=full,signed_omitted_source_norm_squared=full-strength,omitted_source_norm=float(np.sqrt(max(0.,full-strength))),preparation_energy=mean(initial,h)-metadata['reference_energy'],meaning='Raw CAR/source norm and preparation cost relative to NEW neutral reference; NOT a pole gap or probability. Signed source complement is preserved; sqrt guard concerns roundoff only.')
        kicked,drive_evidence=drive(model,name,initial);currents=[assemble(model,name,derivative=(mu,)) for mu in range(6)];operators=measurement_operators(model,name,currents)
        drive_evidence['applied_field_kick']=model['parameters']['field_kick']
        labels,effects=charge_receiver(data,model,name);rho0=operators['rho0'];routes=[assemble(model,name,route_derivative=edge) for edge in range(2)]
        phase_energy=mean(initial,h);base_state=initial.copy();pulse=kicked.copy();baseline=[];reaction=[];base_receiver=[];pulse_receiver=[];base_states=[];pulse_states=[];previous=0.
        for t in times:
            base_state=propagate(h,base_state,t-previous,phase_energy);pulse=propagate(h,pulse,t-previous,phase_energy);previous=t
            base_states.append(base_state);pulse_states.append(pulse)
            b={key:mean(base_state,operator) for key,operator in operators.items()};r={key:mean(pulse,operator) for key,operator in operators.items()}
            b.update(scalar_measure(base_state,model,name));r.update(scalar_measure(pulse,model,name))
            if insertion is not None:b['reverse_source_strength']=float(np.linalg.norm(insertion.H@base_state)**2);r['reverse_source_strength']=float(np.linalg.norm(insertion.H@pulse)**2)
            for axis in range(3):
                b['J_total'+str(axis)]=b['frame_S'+str(axis)]+b['L_body'+str(axis)]+b['F_field'+str(axis)]
                r['J_total'+str(axis)]=r['frame_S'+str(axis)]+r['L_body'+str(axis)]+r['F_field'+str(axis)]
            b['J2']=sum(float(np.linalg.norm(angular(model,name,axis)@base_state)**2) for axis in range(3));r['J2']=sum(float(np.linalg.norm(angular(model,name,axis)@pulse)**2) for axis in range(3))
            baseline.append(b);reaction.append(r)
            base_receiver.append(joint_receiver(base_state,effects,base['scalar'],base['rotor'],field));pulse_receiver.append(joint_receiver(pulse,effects,base['scalar'],base['rotor'],field))
        excess={key:[r[key]-b[key] for r,b in zip(reaction,baseline)] for key in baseline[0]}
        results[label]=dict(source=source_result,preparation='new neutral reference' if insertion is None else 'normalized FULL raw source insertion, not pole filtered',drive=drive_evidence,times=times.tolist(),baseline=baseline,reaction=reaction,excess=excess,charge_labels=labels,joint_baseline=np.array(base_receiver).tolist(),joint_reaction=np.array(pulse_receiver).tolist(),joint_TV=[float(np.sum(abs(r-b))/2) for r,b in zip(pulse_receiver,base_receiver)],joint_normalization_error=float(max(abs(x.sum()-1) for x in base_receiver+pulse_receiver)),norm_error=float(max(abs(np.linalg.norm(x)-1) for x in base_states+pulse_states)),co_moving_initial=co_moving_continuity(h,rho0,routes,initial),co_moving_final=co_moving_continuity(h,rho0,routes,pulse_states[-1]),current_t0_cut_jump=[excess['current'+str(mu)][0] for mu in range(6)],current_t0_interpretation='Untruncated coordinate-function E-impulse commutes J(Q); retained field compression need not. Any t0 jump is a measured electric-cut multiplication defect, not immediate matter reaction.')
        stored[label+'_initial']=initial;stored[label+'_kicked']=kicked;stored[label+'_baseline']=np.array(base_states);stored[label+'_reaction']=np.array(pulse_states)
        generator=sum(value*q for value,q in zip(request['drive_profile'],field['Q']))
        nm=op['Q'].shape[0];nf=len(base['scalar']['grid']);nr=base['rotor']['dimension']
        gstate=field_action(initial,nm,nf,nr,generator)
        results[label]['field_Ehrenfest_initial']=field_force_witness(h,operators,currents,initial)
        results[label]['field_Ehrenfest_final']=field_force_witness(h,operators,currents,pulse_states[-1])
        results[label]['J_G_cut_commutator_norms']=[float(np.linalg.norm(current@gstate-field_action(current@initial,nm,nf,nr,generator))) for current in currents]
        checkpoint('event_'+label)
        # Matched SAME initial state/control, no old reference or current inserted.
        frozen=assemble(model,name,coupled=False);fbase=propagate(frozen,initial,times[-1],phase_energy);fpulse=propagate(frozen,kicked,times[-1],phase_energy)
        frozen_result={key:mean(fpulse,operator)-mean(fbase,operator) for key,operator in operators.items() if key in ['rho0','rho1','C','B','Coulomb','frame_S2']}
        fs=scalar_measure(fpulse,model,name);bs=scalar_measure(fbase,model,name);frozen_result.update({key:fs[key]-bs[key] for key in fs})
        results[label]['frozen_paths']=dict(time=float(times[-1]),material_excess=frozen_result,zero_control_profile_probe_derivative=[0.]*6,meaning='All quantum Peierls AND charged mechanical A_rot couplings removed, including the profile probes. This CONTROL H retains Coulomb/body/bare-rotor/Pi² terms; same actual initial state and field-only kick, actual paired propagation, no reoptimized reference. Zero probe derivative defines this decoupled control, NOT a gauge-faithful physical density-current claim.')
        off=no_hop_law(model,name,h)
        results[label]['all_hops_off_vertex_motion_defect']=float(np.linalg.norm(1j*(off@(rho0@initial)-rho0@(off@initial))))
        off_base=propagate(off,initial,times[-1],phase_energy);off_pulse=propagate(off,kicked,times[-1],phase_energy)
        charge_initial=np.array([mean(initial,observable(model,name,matter=effect)) for effect in effects])
        charge_base=np.array([mean(off_base,observable(model,name,matter=effect)) for effect in effects])
        charge_pulse=np.array([mean(off_pulse,observable(model,name,matter=effect)) for effect in effects])
        mechanical=[mechanical_operator(model,name,derivative=(mu,)) for mu in range(6)]
        probe_density=density(model,name,request['wavevectors'][0]);lab_difference=np.vdot(off_pulse,probe_density@off_pulse)-np.vdot(off_base,probe_density@off_base)
        results[label]['all_hops_off_event']=dict(time=float(times[-1]),charge_initial=charge_initial.tolist(),charge_baseline=charge_base.tolist(),charge_reaction=charge_pulse.tolist(),co_moving_charge_baseline_TV_from_initial=float(np.sum(abs(charge_base-charge_initial))/2),co_moving_charge_reaction_TV_from_initial=float(np.sum(abs(charge_pulse-charge_initial))/2),co_moving_charge_pulse_vs_baseline_TV=float(np.sum(abs(charge_pulse-charge_base))/2),mechanical_current_excess=[mean(off_pulse,j)-mean(off_base,j) for j in mechanical],laboratory_density_wavevector=request['wavevectors'][0],laboratory_density_excess_real=float(lab_difference.real),laboratory_density_excess_imag=float(lab_difference.imag),meaning='ONLY actual material hopping is removed. The covariant charged rotor remains LIVE. Co-moving site-charge PVM must freeze; lab density and mechanical EM current may legitimately move with R. This is not the fully transverse-EM-off arm.')
        stored[label+'_all_hops_off_baseline']=off_base;stored[label+'_all_hops_off_reaction']=off_pulse
        spatial=[]
        for wavevector in request['wavevectors']:
            rho=density(model,name,wavevector);current=[spatial_current(model,name,wavevector,axis) for axis in range(3)]
            hop_only=[spatial_current(model,name,wavevector,axis,include_mechanical=False) for axis in range(3)]
            change=1j*(h@(rho@initial)-rho@(h@initial))
            defect=change+1j*sum(k*(j@initial) for k,j in zip(wavevector,current))
            wrong=change+1j*sum(k*(j@initial) for k,j in zip(wavevector,hop_only))
            spatial.append(dict(wavevector=wavevector,defect=float(np.linalg.norm(defect)),wrong_no_mechanical_defect=float(np.linalg.norm(wrong)),mechanical_current_image_norms=[float(np.linalg.norm((full-hop)@initial)) for full,hop in zip(current,hop_only)],density_norm=float(np.linalg.norm(rho@initial)),current_norms=[float(np.linalg.norm(j@initial)) for j in current],scope='Same-Gaussian full hopping+mechanical Fourier current BEFORE cuts. Wrongcase explicitly omits mechanical motion only. Any nonzero retained Ward defect is projection evidence, not patched current.'))
        results[label]['spatial_Fourier_Ward']=spatial;checkpoint('diagnosed_'+label)
        gauge=[]
        displacement=base['geometry']['displacement']/2
        for axis in range(3):
            coordinate=np.einsum('qia,a->qi',base['rotor']['R'],displacement)[:,axis]
            dipole=observable(model,name,matter=op['density_delta'],rq=sparse.kron(base['rotor']['integrate'](coordinate),sparse.eye(field['dimension'])))
            uniform=spatial_current(model,name,[0.,0.,0.],axis)
            hop=spatial_current(model,name,[0.,0.,0.],axis,include_mechanical=False)
            commutator=1j*(h@(dipole@initial)-dipole@(h@initial))
            gauge.append(dict(axis=axis,full_current_dipole_defect=float(np.linalg.norm(commutator-uniform@initial)),wrong_no_mechanical_defect=float(np.linalg.norm(commutator-hop@initial)),mechanical_image_norm=float(np.linalg.norm((uniform-hop)@initial)),meaning='Constant A=grad(alpha.r) first-order gauge/dipole identity BEFORE projection. Current is independently derived from full paths and charged rotor; this commutator is a DEFECT probe, never a patched current. Retained angular/field/fermion complements may remain.'))
        results[label]['constant_gradient_dipole_current']=gauge
        if name.startswith('anti'):
            positive=name[4:];partner=positive if key is None else positive+'_'+key
            c=MatterMap(matrix(data,positive+'_C_matter'),nf,nr,field['dimension'],field['C'])
            overlap=np.vdot(initial,c@stored[partner+'_initial'])
            pos_current=[assemble(model,positive,derivative=(mu,)) for mu in range(6)]
            mapped_charge=[next(i for i,q in enumerate(results[partner]['charge_labels']) if abs(q+anti_q)<1e-12) for anti_q in labels]
            expected_joint=np.array(results[partner]['joint_reaction'])[:,mapped_charge,:,::-1]
            results[label]['actual_C_event']=dict(field_drive_sign=case.get('drive_sign',1.),source_phase_real=float(overlap.real),source_phase_imag=float(overlap.imag),initial_C_error=float(np.linalg.norm(overlap*initial-c@stored[partner+'_initial'])),baseline_C_errors=[float(np.linalg.norm(overlap*a-c@b)) for a,b in zip(base_states,stored[partner+'_baseline'])],reaction_C_errors=[float(np.linalg.norm(overlap*a-c@b)) for a,b in zip(pulse_states,stored[partner+'_reaction'])],physical_current_C_odd_errors=[float(np.linalg.norm(overlap*(a@pulse_states[-1])+c@(b@stored[partner+'_reaction'][-1]))) for a,b in zip(currents,pos_current)],charge_phi_field_receiver_C_TV=[float(np.sum(abs(a-b))/2) for a,b in zip(pulse_receiver,expected_joint)],meaning='Independent raw CAR-removal source, its own actual anti H, opposite C-conjugate field-only kick, and actual propagation. Full source phase is measured, not imposed. EM charge and field-sign labels reverse; phi is C-even. No anti energy/state borrowed as an event initial condition.')
        print(json.dumps(dict(case=label,raw_source=source_result,max_material={key:max(abs(x) for x in excess[key]) for key in ['rho0','phi2','C','B']},continuity=results[label]['co_moving_initial'],peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)
    checkpoint('complete')


if __name__=='__main__':main()
