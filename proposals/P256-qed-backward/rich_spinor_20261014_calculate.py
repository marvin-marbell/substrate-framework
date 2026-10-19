"""New physical experiment; source and output frozen by capture entrypoint."""
import json
from pathlib import Path
import numpy as np
from rich_spinor_20261014_model import (Parameters, build_sector, solve, current, contact,
    spin_current, scattering, charged_spectral, charged_green, lift)

ROOT=Path(__file__).resolve().parent
PREFIX='rich_spinor_20261014_actual1'


def encode(value):
    if isinstance(value,np.ndarray):
        return {'real':value.real.tolist(),'imag':value.imag.tolist()} if np.iscomplexobj(value) else value.tolist()
    if isinstance(value,complex): return {'real':value.real,'imag':value.imag}
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def multiplets(sector):
    e=sector['energies']; v=sector['vectors']; p=sector['parity']; jz=sector['angular'][2]
    groups=[]; start=0
    while start<len(e):
        end=start+1
        while end<len(e) and abs(e[end]-e[start])<1e-8: end+=1
        u=v[:,start:end]
        groups.append({'energy':float(e[start]),'dimension':end-start,'indexes':list(range(start,end)),
                       'parity':np.real(np.diag(u.conj().T@p@u)),
                       'total_Jz':np.real(np.diag(u.conj().T@jz@u))})
        start=end
    return groups


def analyze(params,tag,arrays):
    sector=solve(build_sector(params,1)); e=sector['energies']; v=sector['vectors']
    groups=multiplets(sector)
    even=next(a for a in groups if np.mean(a['parity'])>0.9)
    odd=next(a for a in groups if np.mean(a['parity'])<-.9)
    selected=even['indexes']+odd['indexes']; u=v[:,selected]
    js=np.array([current(sector,np.zeros(3),np.eye(3)[i]) for i in range(3)])
    projected=np.array([u.conj().T@j@u for j in js]); dim=len(selected)
    speed2=sum(np.trace(j@j).real for j in projected)/(3*dim)
    anti=[]
    for i in range(3):
        for j in range(3):
            target=2*speed2*np.eye(dim) if i==j else np.zeros((dim,dim))
            anti.append(np.linalg.norm(projected[i]@projected[j]+projected[j]@projected[i]-target))
    spinvec=np.real([v[:,a].conj()@sector['spin'][2]@v[:,a] for a in even['indexes']])
    qin=np.array([0,0,.35]); qout=np.array([.35,0,0]); omega=.35
    epsin=np.array([1,1j,0])/np.sqrt(2)
    amplitudes={}; rates=[]
    for label,epsout in [('y',np.array([0,1,0])),('z',np.array([0,0,1]))]:
        full=np.array([[scattering(sector,a,b,qin,epsin,qout,epsout,omega) for a in even['indexes']] for b in even['indexes']])
        reduced=np.array([[scattering(sector,a,b,qin,epsin,qout,epsout,omega,virtual=selected) for a in even['indexes']] for b in even['indexes']])
        amplitudes[label]={'full':full,'four_multiplet_only':reduced,'relative_virtual_omission':float(np.linalg.norm(full-reduced)/max(np.linalg.norm(full),1e-15))}
        rates.append(np.sum(abs(full)**2,axis=0))
    summed=np.sum(rates,axis=0)
    mz_one=np.zeros((12,12),complex); r=sector['geometry']['positions']
    for a in range(6):
        for b in range(6):
            phase=np.cross(r[b],r[a])[2]/2
            mz_one[2*a:2*a+2,2*b:2*b+2]=-1j*sector['orbital'][a,b]*phase*np.eye(2)
    mz=np.kron(lift(mz_one,sector['states']),np.eye(4))
    magnetic=np.real(np.diag(v[:,even['indexes']].conj().T@mz@v[:,even['indexes']]))
    neutral_exc=np.kron(np.eye(12),np.diag([0,1,1,1]))
    orbitmom=[v[:,a].conj()@neutral_exc@v[:,a] for a in selected]
    charge_continuity=[]
    for i in range(3):
        charge_continuity.append(np.linalg.norm(js[i]-1j*(sector['H']@sector['position'][i]-sector['position'][i]@sector['H'])))
    result={'parameters':sector['parameters'],'full_dimension':len(e),'multiplets':groups,
            'selected_indexes':selected,'even_dimension':len(even['indexes']),'odd_dimension':len(odd['indexes']),
            'parity_gap':float(odd['energy']-even['energy']),'projected_current':projected,
            'current_speed_squared':float(speed2),'anticommutator_max_residual':float(max(anti)),
            'normalized_clifford_residual':float(max(anti)/max(2*speed2,1e-15)),
            'neutral_orientation_excitation':np.real(orbitmom),'ground_spin_z':spinvec,
            'ground_orbital_magnetic_moment_z':magnetic,
            'primitive_spin_torque_norm':float(np.linalg.norm(sector['H']@sector['spin'][2]-sector['spin'][2]@sector['H'])),
            'total_rotation_commutator_norm':float(np.linalg.norm(sector['H']@sector['angular'][2]-sector['angular'][2]@sector['H'])),
            'current_position_identity_residual':charge_continuity,
            'scattering':amplitudes,'outgoing_pol_rates_by_initial_spin':np.array(rates),
            'summed_rate_by_initial_spin':summed,
            'spin_rate_asymmetry':float((summed[-1]-summed[0])/max(summed[-1]+summed[0],1e-15)),
            'q_in':qin,'q_out':qout,'omega':omega,'eta':.025}
    for key in ['H','vectors','energies','spin','angular','position','parity']:
        arrays[tag+'_'+key]=sector[key]
    arrays[tag+'_current_xyz']=js
    arrays[tag+'_spin_current_y_sz']=spin_current(sector,qin,np.array([0,1,0]),2)
    arrays[tag+'_contact_y_circular']=contact(sector,qout,np.array([0,1,0]),qin,epsin)
    if tag=='dynamic_positive':
        sectors={0:solve(build_sector(params,0)),1:sector,2:solve(build_sector(params,2))}
        data=charged_spectral(sectors,tuple(even['indexes']))
        for key,value in data.items(): arrays['charged_'+key]=value
        for number in (0,2):
            for key in ('H','vectors','energies','states'): arrays['N'+str(number)+'_'+key]=sectors[number][key]
        a=data['addition_amplitudes']; b=data['removal_amplitudes']
        sumrule=np.einsum('gna,gnb->ab',a.conj(),a)/a.shape[0]+np.einsum('gnb,gna->ab',b.conj(),b)/b.shape[0]
        result['charged_G']={'sector_dimensions':{str(n):len(sectors[n]['energies']) for n in sectors},
                             'CAR_sum_rule_residual':float(np.linalg.norm(sumrule-np.eye(12))),
                             'G_at_omega_0_3':charged_green(data,.3),
                             'addition_gap':float(np.min(data['addition_energies'])),
                             'removal_energy_min':float(np.min(data['removal_energies']))}
        arrays['charged_G_omega03']=charged_green(data,.3)
    return result


def main():
    arrays={}; result={'law':'octahedron-neutral-orientor-spin-transfer-v1','branches':{}}
    for tag,params in [('dynamic_positive',Parameters()),('dynamic_negative',Parameters(rotor_spin=-1.1)),
                       ('no_spin_orientation',Parameters(rotor_spin=0)),('no_orientation_transfer',Parameters(orientation=0))]:
        result['branches'][tag]=analyze(params,tag,arrays)
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,default=encode,indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**arrays)
    print(json.dumps({tag:{key:r[key] for key in ('parity_gap','even_dimension','odd_dimension','normalized_clifford_residual','spin_rate_asymmetry')} for tag,r in result['branches'].items()},indent=2))

if __name__=='__main__': main()
