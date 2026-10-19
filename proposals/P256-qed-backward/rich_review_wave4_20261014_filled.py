"""Independent occupied spectral source and exact CAR-Fock physical pair consumer."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from scipy.linalg import eigh
from rich_filled_symmetry_20261014_model import Parameters, build, hamiltonian, spectrum, vertex, contact, charged_green, pair_response, pair_creation
from rich_review_wave4_20261014_tools import complex_pair, save


def car(bits: int, mode: int, create: bool) -> tuple[int,int] | None:
    if bool((bits>>mode)&1)==create:
        return None
    return bits^(1<<mode), (-1)**((bits&((1<<mode)-1)).bit_count())


def one_body(operator: np.ndarray, state: dict[int,complex]) -> dict[int,complex]:
    out: dict[int,complex] = {}
    for bits,amplitude in state.items():
        for origin in range(len(operator)):
            removed = car(bits,origin,False)
            if removed is None:
                continue
            for destination in range(len(operator)):
                coefficient = operator[destination,origin]
                if abs(coefficient)<1e-14:
                    continue
                created = car(removed[0],destination,True)
                if created is not None:
                    new,sign = created
                    out[new] = out.get(new,0j)+coefficient*removed[1]*sign*amplitude
    return out


def fock_pair_probe(p: Parameters,k: np.ndarray,omega: float,eta: float) -> dict[str,Any]:
    reduced = replace(p,charged_lmax=1,neutral_lmax=0,field_cutoff=0,internal_coupling=0.)
    model = build(reduced)
    energies,modes = spectrum(model,k)
    occupied = np.flatnonzero(energies<0);empty = np.flatnonzero(energies>0)
    sea = sum(1<<int(index) for index in occupied)
    sea_energy = float(np.sum(energies[occupied]))
    cache: dict[int,float] = {}
    def excitation(bits: int) -> float:
        if bits not in cache:
            cache[bits] = float(sum(energies[i] for i in range(len(energies)) if (bits>>i)&1)-sea_energy)
        return cache[bits]
    zero = np.zeros(3);epsin=np.array([.4,.7,-.2]);epsout=np.array([-.3,.5,.6])
    jin = modes.conj().T@vertex(model,k,k,epsin)@modes
    jout = modes.conj().T@vertex(model,k,k,epsout)@modes
    co = modes.conj().T@contact(model,k,k,zero,epsin,zero,epsout)@modes
    particle = 2;hole = -1;pid=int(empty[particle]);hid=int(occupied[hole])
    removed = car(sea,hid,False)
    assert removed is not None
    added = car(removed[0],pid,True)
    assert added is not None
    target = added[0];target_phase = removed[1]*added[1]
    pair_gap = excitation(target);outgoing=omega-pair_gap
    terms=[];source_weights=[];pair_sources=[]
    for first,second,w in [(jin,jout,omega),(jout,jin,-outgoing)]:
        source = one_body(first,{sea:1+0j})
        source[sea] = source.get(sea,0j)-np.trace(first[np.ix_(occupied,occupied)])
        source = {bits:amp for bits,amp in source.items() if abs(amp)>1e-13}
        source_weights.append(float(sum(abs(amp)**2 for amp in source.values())))
        pair_sources.append(len(source))
        resolved = {bits:amp/(w-excitation(bits)+1j*eta) for bits,amp in source.items()}
        received = one_body(second,resolved)
        # Normal-order the actual second current relative to the SAME filled sea.
        disconnected = np.trace(second[np.ix_(occupied,occupied)])*resolved.get(target,0j)
        terms.append(target_phase*(received.get(target,0j)-disconnected))
    contact_fock = target_phase*one_body(co,{sea:1+0j}).get(target,0j)
    direct = contact_fock+sum(terms)
    supplied = pair_creation(model,k,zero,epsin,zero,epsout,omega,particle,hole,eta)
    particle_only = supplied['contact']+sum(row['particle'] for row in supplied['time_orders'])
    plus_bits = car(sea,pid,True);minus_bits=car(sea,hid,False)
    assert plus_bits is not None and minus_bits is not None
    addition_energy=excitation(plus_bits[0]);removal_energy=excitation(minus_bits[0])
    return {'parameters':model['parameters'],'one_particle_modes':len(energies),'occupied_particles':len(occupied),'sea_energy':sea_energy,
            'actual_Nplus1_energy':addition_energy,'actual_Nminus1_energy':removal_energy,'actual_pair_energy':pair_gap,
            'charge_changes':{'addition':p.charge,'removal':-p.charge,'pair':0},
            'current_source_full_one_pair_counts':pair_sources,'current_source_Gram_weights':source_weights,
            'pair_Gram_from_band_formula':[float(np.linalg.norm(op[np.ix_(empty,occupied)])**2) for op in [jin,jout]],
            'CAR_Fock_contact':complex_pair(contact_fock),'CAR_Fock_time_order_amplitudes':[complex_pair(x) for x in terms],
            'CAR_Fock_total_amplitude':complex_pair(direct),'author_full_pair_amplitude':complex_pair(supplied['amplitude']),
            'complete_pair_direct_Fock_error':float(abs(direct-supplied['amplitude'])),'wrong_particle_only_error':float(abs(direct-particle_only)),
            'incoming_frequency':omega,'outgoing_frequency':outgoing,'scope':'Actual reduced microscopic16-mode material, not projection of rich576 result. Exact CAR signs/source/normal-ordering on filled8-particle Fock sea; every accessible one-pair Fock intermediate state retained. This contrary operator probe independently discriminates omitting the physical hole-scattering contribution.'}


def full_probe(model: dict,k: np.ndarray,q: np.ndarray,z: complex) -> dict:
    energies,modes = spectrum(model,k);occ=energies<0;empty=~occ;source=model['source']
    gram_add = source.conj().T@modes[:,empty]@modes[:,empty].conj().T@source
    gram_remove = source.conj().T@modes[:,occ]@modes[:,occ].conj().T@source
    full = charged_green(model,k,z)
    direct = source.conj().T@np.linalg.solve(z*np.eye(len(energies))-hamiltonian(model,k),source)
    response = pair_response(model,k,q,.61+.19j)
    ep,vp=spectrum(model,k+q);em,vm=spectrum(model,k-q)
    plus=np.array([vp[:,ep>0].conj().T@vertex(model,k+q,k,d)@modes[:,occ] for d in np.eye(3)])
    reverse=np.array([modes[:,empty].conj().T@vertex(model,k,k-q,d)@vm[:,em<0] for d in np.eye(3)])
    delta=ep[ep>0,None]-energies[None,occ];delta_reverse=energies[empty,None]-em[None,em<0]
    tensor=np.zeros((3,3),complex)
    for i in range(3):
        for j in range(3):
            tensor[i,j]=np.sum(plus[i].conj()*plus[j]/(.61+.19j-delta))-np.sum(reverse[i]*reverse[j].conj()/(.61+.19j+delta_reverse))
    ward=sum(q[axis]*vertex(model,k+q,k,d) for axis,d in enumerate(np.eye(3)))-model['params'].charge*(hamiltonian(model,k+q)-hamiltonian(model,k))
    c=model['C'];hm=hamiltonian(model,-k)
    ph_error=np.linalg.norm(c@hm.T@c.conj().T+hamiltonian(model,k))
    zero=np.zeros(3);e0,v0=spectrum(model,zero)
    near=np.flatnonzero(e0>0)[:2];negative=np.flatnonzero(e0<0)[-2:]
    low=np.column_stack([v0[:,negative],v0[:,near]])
    rest_current=np.array([low.conj().T@vertex(model,zero,zero,d)@low for d in np.eye(3)])
    wrong_empty=full['addition']
    return {'parameters':model['parameters'],'modes':len(energies),'occupied':int(occ.sum()),'sea_energy_per_k':float(energies[occ].sum()),
            'minimum_rung_eigenvalue':float(eigh(model['M'],eigvals_only=True)[0]),'actual_minimum_add_remove_gap':float(np.min(np.abs(energies))),
            'paired_band_sum_error':float(np.max(np.abs(energies+energies[::-1]))),'Fock_charge_conjugation_matrix_error':float(ph_error),
            'full_retarded_G_direct_error':float(np.linalg.norm(full['retarded']-direct)),
            'time_ordered_vs_retarded_difference':float(np.linalg.norm(full['time_ordered']-direct)),
            'wrong_empty_addition_only_G_error':float(np.linalg.norm(wrong_empty-direct)),
            'actual_add_remove_CAR_closure_error':float(np.linalg.norm(gram_add+gram_remove-source.conj().T@source)),
            'addition_source_weight':float(np.trace(gram_add).real),'removal_source_weight':float(np.trace(gram_remove).real),
            'pair_tensor_direct_error':float(np.linalg.norm(response['tensor']-tensor)),
            'pair_source_Gram_eigenvalues':np.linalg.eigvalsh(response['summed_weight']).tolist(),
            'pair_minimum_energy':float(delta.min()),'macrocell_Ward_error':float(np.linalg.norm(ward)),
            'lowest_splus_sminus_rest_current_norms':[float(np.linalg.norm(op)) for op in rest_current],
            'scope':'Literal rich576-mode Gaussian Fock sea with independent per-fermion INTERNAL vector/orientor levels, not separate neutral/photon field. One-k pair integrand requires actual BZ measure for bulk response. Time-ordered removal uses -i eta; retarded both use +i eta. Opposite hole charge is established relative this filled reference, not a Lorentz/QED positron identification.'}


def main() -> None:
    inputs=json.loads(Path(sys.argv[1]).read_text());p=Parameters(**inputs['parameters']);k=np.array(inputs['k']);q=np.array(inputs['q'])
    result=full_probe(build(p),k,q,complex(*inputs['z']))
    fock=fock_pair_probe(p,k,inputs['omega'],inputs['eta'])
    save('filled',{'kind':'INDEPENDENT_NEW_FILLED_REFERENCE_AND_EXACT_CAR_CONSUMER','inputs':inputs,'full_rich':result,'exact_Fock_contrary_material':fock,
                   'scope':'NEW bipartite all-offdiagonal microscopic rung/bridge law; scalar spectral symmetry is structural by chosen geometry, not fitted mu(k). Filled reference changes physical sources and holes; current lowestpair leading vector remains adverse, not repaired by labels.'})

if __name__=='__main__':
    main()
