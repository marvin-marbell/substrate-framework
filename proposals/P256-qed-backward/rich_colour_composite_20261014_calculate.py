"""Actual stationary, baryon-source, current, motion and loop-reaction consumer."""
from __future__ import annotations
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh, expm, null_space
from rich_colour_composite_20261014_model import material_root, generators, build, hamiltonian, baryon_source


def norm(x):
    return float(np.linalg.norm(x))


def mean(v, a):
    return float(np.vdot(v,a@v).real)


def complex_json(a):
    return dict(real=np.asarray(a).real.tolist(),imag=np.asarray(a).imag.tolist())


def spectrum(h,s2):
    ss,sp=eigh(s2); es=[]; vs=[]; spins=[]
    for spin,value in ((.5,.75),(1.5,3.75)):
        p=sp[:,abs(ss-value)<1e-7]
        if p.shape[1]:
            e,v=eigh(p.conj().T@h@p); es.extend(e);vs.append(p@v);spins.extend([spin]*len(e))
    order=np.argsort(es)
    return np.array(es)[order],np.column_stack(vs)[:,order],np.array(spins)[order]


def dynamics(state, energies, vectors, times, observables):
    coefficients=vectors.conj().T@state
    states=vectors@(np.exp(-1j*np.outer(energies,times))*coefficients[:,None])
    data={key:[mean(states[:,i],op) for i in range(len(times))] for key,op in observables.items()}
    return states,data


def main():
    inp=Path(sys.argv[1]); out=Path(sys.argv[2]);p=json.loads(inp.read_text())
    model=build(p);h,j,k=hamiltonian(model);e,v,spins=spectrum(h,model['S2'])
    root=material_root();loop=model['loop'];cp=model['colour']['P'];ts=generators()
    structure=np.array([[[-2j*np.trace((a@b-b@a)@c) for c in ts] for b in ts] for a in ts]).real
    algebra=max(norm(a@b-b@a-1j*sum(structure[i,jj,k]*ts[k] for k in range(8))) for i,a in enumerate(ts) for jj,b in enumerate(ts))
    loopalgebra=max(norm(a@b-b@a-1j*sum(structure[i,jj,k]*loop['L'][k] for k in range(8))) for i,a in enumerate(loop['L']) for jj,b in enumerate(loop['L']))
    covariance=max(norm(loop['L'][a]@loop['U'][i,jj]-loop['U'][i,jj]@loop['L'][a]+sum(ts[a][i,k]*loop['U'][k,jj] for k in range(3))) for a,i,jj in product(range(8),range(3),range(3)))
    rcovariance=max(norm(loop['R'][a]@loop['U'][i,jj]-loop['U'][i,jj]@loop['R'][a]-sum(loop['U'][i,k]*ts[a][k,jj] for k in range(3))) for a,i,jj in product(range(8),range(3),range(3)))
    theta=np.array([.19,-.27,.13,.11,-.16,.23,.14,-.09]);g=expm(1j*sum(x*t for x,t in zip(theta,ts)))
    d=expm(1j*sum(x*t for x,t in zip(theta,loop['L'])))
    finitecov=max(norm(d@loop['U'][a,b]@d.conj().T-sum(g.conj().T[a,c]*loop['U'][c,b] for c in range(3))) for a,b in product(range(3),repeat=2))
    unitarydebt=np.block([[sum(loop['U'][c,a].conj().T@loop['U'][c,b] for c in range(3))-(np.eye(19) if a==b else 0) for b in range(3)] for a in range(3)])
    mattertotal=[sum(model['colour']['q'][i][a] for i in range(3)) for a in range(8)]
    mattercasimir=sum(a@a for a in mattertotal)
    octsmall=cp.conj().T@np.kron(mattercasimir/3,np.eye(19))@cp
    octet=model['Pauli'].conj().T@np.kron(np.kron(np.eye(27),octsmall),np.eye(3))@model['Pauli']
    vacuumgauss=sum((a+b)@(a+b) for a,b in zip(loop['L'],loop['R']))
    gv,gp=eigh(vacuumgauss);gp=gp[:,gv<1e-8]
    hv=p['g2']/2*sum(p['lengths'])*loop['C']-p['magnetic']*loop['B']
    ev,vv=eigh(gp.conj().T@hv@gp);vacuum=gp@vv[:,0]
    sources=np.column_stack([baryon_source(model,site,vacuum) for site in range(3)])
    half=np.flatnonzero(spins==.5); lowest=half[0];ground=v[:,lowest]
    de=e-e[lowest];sourceweights=abs(v.conj().T@sources)**2
    source3=baryon_source(model,0,vacuum,1.5)
    obs={**{f'n{i}':a for i,a in enumerate(model['n'])},**{f'charge{i}':a for i,a in enumerate(model['rho'])},**{f'J{i}':a for i,a in enumerate(j)},'r2':model['r2'],'colocated':model['co'],'electric':model['HE'],'magnetic':model['HB'],'loop_C':model['loopobs']['C'],'loop_B':model['loopobs']['B'],'loop_boundary':model['loopobs']['boundary'],'matter_octet':octet}
    obs.update({f'triple{i}':model['distributions'][model['positions'].index((i,i,i))] for i in range(3)})
    # Euclidean source preparation uses ALL retained energies, not a state cutoff.
    packet=v@(np.exp(-p['packet_tau']*(e-e[0]))*(v.conj().T@sources[:,0]));packet/=np.linalg.norm(packet)
    times=np.array(p['times']);packetstates,packetdata=dynamics(packet,e,v,times,obs)
    kicked=expm(-1j*p['pulse']*model['rho'][0])@ground
    pulsestates,pulsedata=dynamics(kicked,e,v,times,obs)
    # Variational separated subspaces are finite-cavity controls, NOT free coloured fragments.
    cv,cu=eigh(model['co']);separate=cu[:,cv<.5]
    esep=eigh(separate.conj().T@h@separate,eigvals_only=True)[0]
    allapart=sum(model['distributions'][i] for i,pos in enumerate(model['positions']) if len(set(pos))==3)
    av,au=eigh(allapart);ap=au[:,av>.5];eapart=eigh(ap.conj().T@h@ap,eigvals_only=True)[0]
    ov,op=eigh(octet);sing=op[:,ov<1e-7]
    esing=eigh(sing.conj().T@h@sing,eigvals_only=True)
    hcut=h+p['hop']*sum(model['T'][i][2]+model['T'][i][2].conj().T for i in range(3))
    ecut,vcut,scut=spectrum(hcut,model['S2']);cutground=vcut[:,np.flatnonzero(scut==.5)[0]]
    cutkick=expm(-1j*p['pulse']*model['rho'][0])@cutground
    cutobs={**obs,'J2':np.zeros_like(j[2])}
    _,cutdata=dynamics(cutkick,ecut,vcut,times,cutobs)
    hzero=h-model['HE'];ezero,vzero,szero=spectrum(hzero,model['S2']);zg=vzero[:,np.flatnonzero(szero==.5)[0]]
    amplitudes=np.array([v.conj().T@a@ground for a in j]);weights=abs(amplitudes)**2
    response=[]
    for omega in p['frequencies']:
        z=omega+1j*p['eta'];chi=np.array([[sum(amplitudes[a].conj()*amplitudes[b]/(z-de)-amplitudes[b].conj()*amplitudes[a]/(z+de)) for b in range(3)] for a in range(3)])
        response.append(dict(omega=omega,retarded=complex_json(chi),with_contact=complex_json(chi+np.diag([mean(ground,a) for a in k]))))
    continuity=[]
    for site in range(3):
        # Oriented edges 0->1,1->2,2->0: incoming minus outgoing.
        continuity.append(norm(1j*(h@model['rho'][site]-model['rho'][site]@h)-(j[(site-1)%3]-j[site])))
    pairspin=[np.array([[0,1],[1,0]],complex)/2,np.array([[0,-1j],[1j,0]],complex)/2,np.diag([.5,-.5])]
    pairtotal=[np.kron(s,np.eye(2))+np.kron(np.eye(2),s) for s in pairspin]
    pairvalues=eigh(sum(s@s for s in pairtotal),eigvals_only=True)
    result=dict(schema=1,kind='finite particle-only repeated-frame SU3 gauge composite hypothesis',parameters=p,
        dimensions=dict(material=6,commutant_complex=root['commutant'].shape[1],loop=19,colour_loop=513,gauss_colour_loop=model['nc'],raw_position_colour_loop_Mhalf=model['raw_dim'],fermionic_Mhalf=len(h),spin_half=int(sum(spins==.5)),spin_threehalf=int(sum(spins==1.5))),
        root=dict(assumption='Primitive quaternion field acts irreducibly on a complex two-component physical material-spin module, repeated EXACTLY three times; new law, not derived from old irreducible Clifford4 or a Dirac gamma ansatz.',commutant='I2 tensor M3(C), unitary U3, determinant-one SU3; gauging SU3 is a further primitive local law',commutant_residual=max(norm(f@c-c@f) for f in root['repeated'] for c in root['intertwiners']),su3_algebra=algebra,commutant_generators_rank=int(np.linalg.matrix_rank(np.column_stack([x.reshape(-1) for x in root['intertwiners']])))),
        gauge=dict(total_gauss=max(norm(g@cp) for g in model['colour']['G']),all_local_gauss_after_exact_tree_elimination=model['local_gauss_error'],left_algebra=loopalgebra,left_U_covariance=covariance,right_U_covariance=rcovariance,finite_group_covariance=finitecov,U_unitarity_defect_frobenius=norm(unitarydebt),U_unitarity_defect_operator=float(np.max(abs(eigh(unitarydebt,eigvals_only=True)))),projection_debt='Only loop irreps 1,3,bar3 retained. Exact gauge covariance survives; U is NOT unitary and omitted 6,bar6,8 transitions remain. Tree electric representations are NOT independently truncated.',matter_octet_projector_error=norm(octet@octet-octet)),
        stationary=dict(vacuum_energy=float(ev[0]),energy=float(e[lowest]),vacuum_subtracted=float(e[lowest]-ev[0]),S2=mean(ground,model['S2']),charge=sum(mean(ground,a) for a in model['rho']),observables={key:mean(ground,a) for key,a in obs.items()},probability_by_ordered_positions=[dict(positions=list(pos),probability=mean(ground,a)) for pos,a in zip(model['positions'],model['distributions'])],energies=e.tolist(),spins=spins.tolist(),ground_residual=norm(h@ground-e[lowest]*ground)),
        pauli=dict(identical_up_flavour='Particles 0,1 are identical u; exact antisymmetry includes positions, colour, loop and material spin. Particle2 is d; fixed uud and primitive charges are INPUTS, not an emergent proton projector.',odd_parity=-1,S2_commutator=norm(h@model['S2']-model['S2']@h),source_norms=np.diag(sources.conj().T@sources).real.tolist(),half_source_residual=norm(model['S2']@sources-.75*sources),threehalf_source_residual=norm(model['S2']@source3-3.75*source3),ordinary_two_fermion_spin2=pairvalues.tolist(),spin_scope='Physical material-frame SU2, not flavour isospin. Fixed routed triangle has no continuous orbital rotation symmetry; full spacetime Noether spin remains open.'),
        binding=dict(diagnostic='Variational energy required to exclude all co-located triples, and to force three separate sites, in SAME gauge Hilbert space. No unphysical E1+E2 coloured asymptotic threshold.',exclude_colocated_energy=float(esep),exclude_colocated_cost=float(esep-e[lowest]),all_separate_energy=float(eapart),all_separate_cost=float(eapart-e[lowest]),gauss_electric_off_energy=float(ezero[0]),gauss_electric_off_r2=mean(zg,model['r2']),gauss_electric_off_colocated=mean(zg,model['co'])),
        source=dict(definition='epsilon_abc times uud symmetric-u spin-CG half source, local triple creation on own gauge-loop vacuum; source creates odd charge1. No species projection of H.',spectral_energies_relative_vacuum=(e-ev[0]).tolist(),weights_by_site=sourceweights.tolist(),sumrules=sourceweights.sum(axis=0).tolist(),spin_threehalf_source_weights=(abs(v.conj().T@source3)**2).tolist()),
        current=dict(definition='J_edge=-dH/dA_edge for Peierls phase exp(i q_flavour A); contact=d2H/dA2. Primitive flavour charges 2/3,2/3,-1/3.',continuity_residuals=continuity,transition_gaps=de.tolist(),transition_weights=weights.tolist(),all_retained_eigenstates=len(e),retarded=response,contact=[mean(ground,a) for a in k]),
        motion=dict(times=times.tolist(),packet_preparation='Own epsilon source, exp[-tau(H-E0)] with full retained spectrum; not inserted baryon hopping.',source_packet=packetdata,EM_density_kick=pulsedata,pulse_initial_loop_delta=mean(kicked,model['loopobs']['C'])-mean(ground,model['loopobs']['C'])),
        controls=dict(force_matter_singlet_only=dict(dimension=sing.shape[1],energy=float(esing[0]),energy_loss=float(esing[0]-e[0])),chord_cut=dict(energy=float(ecut[0]),pulse=cutdata)),
        debts=['Particle-only fixed-three-constituent sector: no antiquarks, Dirac sea, pair production, renormalization or proton claim.','Finite triangle and severe loop Galerkin cutoff; not continuum QCD, asymptotic confinement proof, continuous COM mass or a scattering cross section.','Primitive material frame multiplicity3, fermionic CAR, two flavours and fractional EM charges, graph, electric coupling and plaquette magnetic coefficient are inputs.','S1/2 and S3/2 coexist without imposed exchange or fitted spin penalty; no unique proton/delta splitting.','External EM probe only: no dynamical EM photon, joined electron/photon theory or physical proton legs.'])
    out.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),H=h,HE=model['HE'],HB=model['HB'],S2=model['S2'],e=e,v=v,spins=spins,GaussP=cp,Pauli=model['Pauli'],vacuum=vacuum,sources=sources,ground=ground,current=np.array(j),contact=np.array(k),rho=np.array(model['rho']),matter_octet=octet,loop_C=model['loopobs']['C'],loop_B=model['loopobs']['B'],packet_states=packetstates,pulse_states=pulsestates,positions=np.array(model['positions']))
    print(json.dumps(dict(dimensions=result['dimensions'],gauge=result['gauge'],stationary=result['stationary']['observables'],binding=result['binding'],continuity=continuity),indent=2))


if __name__=='__main__':
    main()
