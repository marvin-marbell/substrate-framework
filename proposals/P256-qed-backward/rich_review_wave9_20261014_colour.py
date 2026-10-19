"""NEW full total-Gauss/octet-loop/Pauli composite consumer; no QCD identification."""
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, expm, null_space
from rich_colour_composite_20261014_model import material_root, generators, build, hamiltonian, baryon_source

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave9_20261014_colour'


def mean(op,state):
    return float(np.vdot(state,op@state).real)


def evolve(e,v,state,time):
    return v@(np.exp(-1j*time*e)*(v.conj().T@state))


def main():
    p=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    root=material_root();ts=generators();m=build(p);h,j,k=hamiltonian(m)
    cp=m['colour']['P'];loop=m['loop'];gauss=m['colour']['G']
    e,v=eigh(h)
    ss,sv=eigh(m['S2']);half=sv[:,abs(ss-.75)<1e-8];threehalf=sv[:,abs(ss-3.75)<1e-8]
    eh,vh=eigh(half.conj().T@h@half);g=half@vh[:,0];eg=float(eh[0])
    e3=eigh(threehalf.conj().T@h@threehalf,eigvals_only=True)
    structural=max(float(np.linalg.norm(f@c-c@f)) for f in root['repeated'] for c in root['intertwiners'])
    gauss_error=max(float(np.linalg.norm(a@cp)) for a in gauss)
    raw_transfer=[]
    for particle in range(3):
        raw=np.zeros((513,513),complex)
        for a,b in product(range(3),repeat=2):
            em=np.zeros((3,3));em[a,b]=1
            factors=[em if i==particle else np.eye(3) for i in range(3)]
            matter=np.kron(np.kron(factors[0],factors[1]),factors[2])
            raw+=np.kron(matter,loop['U'][a,b])
        raw_transfer.append(raw)
    covariance=max(float(np.linalg.norm(a@u-u@a)) for a in gauss for u in raw_transfer)
    wrong_scalar_transfer=np.kron(np.eye(27),loop['W'])
    physical_matrix_difference=[float(np.linalg.norm(cp.conj().T@(u-wrong_scalar_transfer)@cp)) for u in raw_transfer]
    unitary_debt=np.block([[sum(loop['U'][c,a].conj().T@loop['U'][c,b] for c in range(3))-(np.eye(19) if a==b else 0) for b in range(3)] for a in range(3)])
    mattertotal=[sum(m['colour']['q'][i][a] for i in range(3)) for a in range(8)]
    matterC=sum(a@a for a in mattertotal)
    colour_octet=cp.conj().T@np.kron(matterC/3,np.eye(19))@cp
    octet=m['Pauli'].conj().T@np.kron(np.kron(np.eye(27),colour_octet),np.eye(3))@m['Pauli']
    ov,ou=eigh(octet);singlet=ou[:,ov<1e-8]
    forced_e=eigh(singlet.conj().T@h@singlet,eigvals_only=True)
    vacuum_gauss=sum((a+b)@(a+b) for a,b in zip(loop['L'],loop['R']))
    vg,vp=eigh(vacuum_gauss);vp=vp[:,vg<1e-8]
    vacuum_h=p['g2']/2*sum(p['lengths'])*loop['C']-p['magnetic']*loop['B']
    ve,vv=eigh(vp.conj().T@vacuum_h@vp);vacuum=vp@vv[:,0]
    sources=np.column_stack([baryon_source(m,site,vacuum) for site in range(3)])
    amps=v.conj().T@sources;source_weights=abs(amps)**2
    charge=sum(m['rho'])
    step=2e-5;derivatives=[];contacts=[]
    for edge in range(3):
        angles=np.zeros(3);angles[edge]=step
        hp=hamiltonian(m,angles)[0];hm=hamiltonian(m,-angles)[0]
        derivatives.append(float(np.linalg.norm((hp-hm)/(2*step)+j[edge])))
        contacts.append(float(np.linalg.norm((hp-2*h+hm)/step**2-k[edge])))
    continuity=[float(np.linalg.norm(1j*(h@rho-rho@h)-(j[(site-1)%3]-j[site]))) for site,rho in enumerate(m['rho'])]
    wrong_spin_current=float(np.linalg.norm(1j*(h@m['rho'][0]-m['rho'][0]@h)))
    centered=np.column_stack([op@g-g*np.vdot(g,op@g) for op in j]);t=v.conj().T@centered;gap=e-eg
    z=p['frequency']+1j*p['eta']
    spectral=t.conj().T@(t/(z-gap)[:,None])-(t.conj().T@(t/(z+gap)[:,None])).T
    xp=np.linalg.solve((z+eg)*np.eye(len(h))-h,centered);xm=np.linalg.solve((z-eg)*np.eye(len(h))+h,centered)
    direct=centered.conj().T@xp-(centered.conj().T@xm).T
    missing_negative=float(np.linalg.norm(spectral-centered.conj().T@xp))
    kick=expm(-1j*p['pulse']*m['rho'][0])@g
    packet=v@(np.exp(-p['source_tau']*(e-e[0]))*(v.conj().T@sources[:,0]));packet/=np.linalg.norm(packet)
    observables={'matter_octet':octet,'loop_C':m['loopobs']['C'],'loop_B':m['loopobs']['B'],'r2':m['r2'],'colocated':m['co'],'loop_boundary':m['loopobs']['boundary'],**{f'charge_{i}':rho for i,rho in enumerate(m['rho'])},**{f'J_{i}':op for i,op in enumerate(j)}}
    observables.update({f'triple_{site}':m['distributions'][m['positions'].index((site,site,site))] for site in range(3)})
    records={};saved={'half_reference':g,'vacuum':vacuum,'sources':sources}
    cut_h=h+p['hop']*sum(m['T'][i][2]+m['T'][i][2].conj().T for i in range(3));cut_e,cut_v=eigh(cut_h)
    for label,energies,vectors,initial in [('baryon_packet',e,v,packet),('EM_density_impulse',e,v,kick),('same_initial_chord_cut',cut_e,cut_v,kick)]:
        records[label]=[]
        delivered_observables={**observables,'J_2':np.zeros_like(j[2])} if label=='same_initial_chord_cut' else observables
        for time in p['times']:
            state=evolve(energies,vectors,initial,time)
            records[label].append(dict(time=time,norm=float(np.vdot(state,state).real),S2=mean(m['S2'],state),total_charge=mean(charge,state),**{name:mean(op,state) for name,op in delivered_observables.items()}))
        saved[label+'_initial']=initial;saved[label+'_final']=state
    co,cu=eigh(m['co']);not_co=cu[:,co<.5]
    apart=sum(m['distributions'][i] for i,pos in enumerate(m['positions']) if len(set(pos))==3)
    av,au=eigh(apart);separate=au[:,av>.5]
    no_co_e=eigh(not_co.conj().T@h@not_co,eigvals_only=True)[0]
    apart_e=eigh(separate.conj().T@h@separate,eigvals_only=True)[0]
    off_h=h-m['HE'];off_e,off_v=eigh(half.conj().T@off_h@half);off_g=half@off_v[:,0]
    result=dict(kind='NEW_TOTAL_GAUSS_DYNAMIC_NONABELIAN_COMPOSITE_CURRENT_AUDIT',parameters=p,
                root=dict(commutant_complex_dimension=root['commutant'].shape[1],commutant_residual=structural,
                          meaning='Chosen repeated quaternion2x3 module has I2 tensor M3 complex commutant; primitive multiplicity3 and subsequent SU3 gauging are INPUTS, not QCD.'),
                dimension=len(h),total_Gauss_dimension=m['nc'],Gauss_residual=gauss_error,tree_local_Gauss_error=m['local_gauss_error'],
                all_U_matter_transfer_total_Gauss_covariance=covariance,wrong_scalar_TrU_transfer_difference=physical_matrix_difference,
                projected_U_unitarity_defect=float(np.linalg.norm(unitary_debt)),matter_octet_projector_error=float(np.linalg.norm(octet@octet-octet)),
                spectrum=e.tolist(),actual_global_floor=float(e[0]),half_channel_ground_energy=eg,threehalf_channel_ground_energy=float(e3[0]),
                half_reference_residual=float(np.linalg.norm(h@g-eg*g)),S2_commutator=float(np.linalg.norm(h@m['S2']-m['S2']@h)),half_reference_S2=mean(m['S2'],g),
                charge_operator_identity=float(np.linalg.norm(charge-np.eye(len(h)))),half_source_residual=float(np.linalg.norm(m['S2']@sources-.75*sources)),
                source_Gram_real=(sources.conj().T@sources).real.tolist(),source_full_spectral_completeness=float(np.linalg.norm(amps.conj().T@amps-sources.conj().T@sources)),
                source_all_weights=source_weights.tolist(),source_all_energies_relative_own_loop_vacuum=(e-ve[0]).tolist(),
                forced_separate_matter_singlet=dict(dimension=singlet.shape[1],energy=float(forced_e[0]),energy_loss_vs_full_floor=float(forced_e[0]-e[0])),
                stationary_octet=mean(octet,g),stationary_loop_C=mean(m['loopobs']['C'],g),stationary_loop_boundary=mean(m['loopobs']['boundary'],g),
                localization=dict(exclude_colocated_cost_vs_half=float(no_co_e-eg),force_all_separate_cost_vs_half=float(apart_e-eg),half_r2=mean(m['r2'],g),half_colocated=mean(m['co'],g),electric_off_r2=mean(m['r2'],off_g),electric_off_colocated=mean(m['co'],off_g),meaning='Finite SAME N3 variational localization/extension costs, NOT asymptotic E1+E2 binding threshold.'),
                actual_EM_current_derivative=derivatives,actual_EM_contact_derivative=contacts,charge_continuity=continuity,
                wrong_constant_Noether_spin_current_continuity=wrong_spin_current,all_current_virtual_states=len(e),
                current_spectral_Gram=float(np.linalg.norm(t.conj().T@t-centered.conj().T@centered)),retarded_direct_error=float(np.linalg.norm(spectral-direct)),wrong_positive_only_error=missing_negative,
                trajectories=records,
                scope='Particle-only fixeduud three-constituent M=+1/2 sector, retaining both physical materialS=1/2 and3/2. Primitive quaternionspin/multiplicity/flavours/charges/topology/Galerkin1+3+bar3/g2/magnetic law. External EM density impulse is NOT dynamical photon or independent field preparation. Actual totalGauss allows matter8 x loop8 dressing; no rigid scalar spanpotential/QCD/proton/antiquark/Diracsea/asymptotic confinement or free3D NoEther claim.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**saved)
    print(json.dumps({key:result[key] for key in ('dimension','total_Gauss_dimension','Gauss_residual','all_U_matter_transfer_total_Gauss_covariance','projected_U_unitarity_defect','actual_global_floor','half_channel_ground_energy','threehalf_channel_ground_energy','stationary_octet','stationary_loop_C','stationary_loop_boundary','localization','charge_continuity','retarded_direct_error')},indent=2))


if __name__=='__main__':
    main()
