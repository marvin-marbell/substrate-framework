"""NEW physical calculation: Noether rotations, continuum current/contact, quantum field feedback."""
import json
from pathlib import Path
import numpy as np
from rich_spinor_rot_20261014_model import Parameters,build,solve,current,contact,amplitude,charged_green,vacuum_sources,rotate,multiplication
ROOT=Path(__file__).resolve().parent
PREFIX='rich_spinor_rot_20261014_actual1'


def encode(value):
    if isinstance(value,np.ndarray):return {'real':value.real.tolist(),'imag':value.imag.tolist()} if np.iscomplexobj(value) else value.tolist()
    if isinstance(value,complex):return {'real':value.real,'imag':value.imag}
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def groups(model):
    e=model['energies'];v=model['vectors'];par=model['parity'];j2=sum(a@a for a in model['angular'])
    result=[];start=0
    while start<len(e):
        end=start+1
        while end<len(e) and abs(e[end]-e[start])<1e-8:end+=1
        u=v[:,start:end]
        result.append({'energy':float(e[start]),'dimension':end-start,'indexes':list(range(start,end)),
                       'parity':np.real(np.diag(u.conj().T@par@u)),
                       'Jz':np.real(np.diag(u.conj().T@model['angular'][2]@u)),
                       'J2':np.real(np.diag(u.conj().T@j2@u))})
        start=end
    return result


def observe(params,tag,arrays):
    m=solve(build(params));h=m['H'];e=m['energies'];v=m['vectors'];gg=groups(m)
    even=next(a for a in gg if np.mean(a['parity'])>.99)
    odd=next(a for a in gg if np.mean(a['parity'])<-.99)
    low=even['indexes']+odd['indexes'];u=v[:,low];ground=even['indexes']
    zero=np.zeros(3);xyz=np.array([current(m,zero,np.eye(3)[i]) for i in range(3)])
    pjs=np.array([u.conj().T@a@u for a in xyz]);dim=len(low)
    speed2=sum(np.trace(a@a).real for a in pjs)/(3*dim)
    anti=max(np.linalg.norm(pjs[i]@pjs[j]+pjs[j]@pjs[i]-(2*speed2*np.eye(dim) if i==j else 0)) for i in range(3) for j in range(3))
    j=m['angular'];rotcomm=[np.linalg.norm(h@a-a@h) for a in j]
    algebra=np.linalg.norm(j[0]@j[1]-j[1]@j[0]-1j*j[2])
    vector_residual=[]
    for i,jj,k in [(0,1,2),(1,2,0),(2,0,1)]:
        vector_residual.append(np.linalg.norm(j[i]@xyz[jj]-xyz[jj]@j[i]-1j*xyz[k]))
    r=m['charged']['R'];noncomm=np.linalg.norm(r[0]@r[1]-r[1]@r[0])
    naive=np.eye(len(r[0]))-r[0]@r[0]
    correct=multiplication(m['charged'],1-m['charged']['r'][:,0]**2)
    data={'parameters':m['parameters'],'dimension':len(e),'multiplets':gg[:12],
          'even_indexes':ground,'odd_indexes':odd['indexes'],'parity_gap':float(odd['energy']-even['energy']),
          'projected_current':pjs,'current_speed_squared':float(speed2),
          'clifford_residual_if_nondark':float(anti/(2*speed2)) if speed2>1e-12 else None,
          'SO3_generator_algebra_residual':float(algebra),'H_Noether_commutator_xyz':rotcomm,
          'current_vector_covariance_residual_xyz':vector_residual,
          'projected_X_noncommutator_norm':float(noncomm),
          'before_vs_after_Galerkin_contact_difference':float(np.linalg.norm(correct-naive))}
    if tag in ('bare_L2','feedback_cut1'):
        axis=np.array([1,2,3])/np.sqrt(14);rotation=rotate(m,axis,.731)
        data['finite_rotation_H_invariance']=float(np.linalg.norm(rotation.conj().T@h@rotation-h))
    qin=np.array([0,0,.35]);qout=np.array([.35,0,0]);omega=.35;rows=[]
    for helicity in [-1,1]:
        inc=np.array([1,helicity*1j,0])/np.sqrt(2);jin=current(m,qin,inc)
        mats=[];restricted=[]
        for out in [np.array([0,1,0]),np.array([0,0,1])]:
            jout=current(m,qout,out);co=contact(m,qout,out,qin,inc)
            mats.append(np.array([[amplitude(m,a,b,jin,jout,co,omega) for a in ground] for b in ground]))
            restricted.append(np.array([[amplitude(m,a,b,jin,jout,co,omega,virtual=low) for a in ground] for b in ground]))
        mats=np.array(mats);restricted=np.array(restricted);rates=np.sum(abs(mats)**2,axis=(0,1))
        rows.append({'helicity':helicity,'amplitudes_full':mats,'amplitudes_lowest_parity_only':restricted,
                     'rates_by_initial_spin':rates,'spin_rate_asymmetry':float((rates[-1]-rates[0])/sum(rates)),
                     'spin_flip_rate_sum':float(np.sum(abs(mats[:,0,1])**2)+np.sum(abs(mats[:,1,0])**2)),
                     'relative_low_virtual_amplitude_error':float(np.linalg.norm(mats-restricted)/np.linalg.norm(mats))})
        if helicity==1:
            arrays[tag+'_incoming_current']=jin;arrays[tag+'_outgoing_current_z']=jout;arrays[tag+'_contact_z']=co
    data['full_spin_receiver']={'q_in':qin,'q_out':qout,'omega':omega,'eta':1e-5,'rows':rows}
    # Full source spectra, not early low4-state charged propagation.
    source=vacuum_sources(m);a=v.conj().T@source
    data['charged_G_vacuum']={'shape':[a.shape[1],a.shape[1]],'source_dimension':a.shape[1],
                              'all_addition_states':len(e),'spectral_sum_rule_residual':float(np.linalg.norm(a.conj().T@a-np.eye(a.shape[1]))),
                              'addition_threshold':float(e[0]),'reference':'N0 empty charged sphere; neutral l0; field vacuum; no removal poles'}
    arrays[tag+'_charged_amplitudes']=a;arrays[tag+'_charged_G_omega03']=charged_green(m,.3)
    if m['field'] is not None:
        number=np.kron(np.eye(len(m['body_H'])),m['field']['number'])
        data['ground_field_number']=np.real(np.diag(v[:,ground].conj().T@number@v[:,ground]))
        cnoise=[]
        for i in range(3):
            qfull=np.kron(np.eye(len(m['body_H'])),m['field']['Q'][i])
            qsq=np.kron(np.eye(len(m['body_H'])),m['field']['QQ'][i,i])
            cnoise.append({'Q_mean':np.real(np.diag(v[:,ground].conj().T@qfull@v[:,ground])),
                           'Q2_before_field_projection':np.real(np.diag(v[:,ground].conj().T@qsq@v[:,ground]))})
        data['ground_quantum_field_motion']=cnoise
    for key in ('H','energies','vectors','angular','spin','parity'):arrays[tag+'_'+key]=m[key]
    arrays[tag+'_current_xyz']=xyz
    return data


def main():
    arrays={};result={'law':'rotational-sphere-neutral-orientor-covariant-cavity-v1','branches':{}}
    branches=[('bare_L2',Parameters()),('bare_L3',Parameters(charged_lmax=3)),
              ('no_spin_transfer',Parameters(rotor_spin=0)),('no_orientation_transfer',Parameters(orientation=0)),
              ('feedback_cut1',Parameters(coupling=.45)),('feedback_cut2',Parameters(coupling=.45,field_cutoff=2))]
    for tag,params in branches:result['branches'][tag]=observe(params,tag,arrays)
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,default=encode,indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**arrays)
    print(json.dumps({tag:{'dimension':a['dimension'],'parity_gap':a['parity_gap'],
                          'Noether_max':max(a['H_Noether_commutator_xyz']),
                          'spin_asymmetry':a['full_spin_receiver']['rows'][1]['spin_rate_asymmetry'],
                          'rates':a['full_spin_receiver']['rows'][1]['rates_by_initial_spin'].tolist(),
                          'speed_squared':a['current_speed_squared'],
                          'clifford_residual':a['clifford_residual_if_nondark']} for tag,a in result['branches'].items()},indent=2))

if __name__=='__main__':main()
