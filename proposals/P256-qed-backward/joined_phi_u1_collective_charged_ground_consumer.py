"""All-native symmetric63 density-lock superexchange with COMPLETE frozen H action.

This is a finite variational STATE in the untruncated fixed-Phi Hamiltonian,
not a compressed Hamiltonian substituted for H or a full-Phi ground claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterator
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import joined_phi_u1_neutral_ground_graph_consumer as native_graph
import joined_phi_u1_collective_charged_ground_controls as controls

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_collective_charged_ground_'
Flow = tuple[tuple[int, int], ...]
Bath = tuple[tuple[int, int], ...]
Vertex = tuple[int, Bath, Flow]
State = dict[Vertex, complex]
sys.set_int_max_str_digits(0)

PROOF = {
 'basis': 'Auxiliary basis is the normalized sum of all 63!/prod n_mask! labelled-copy tensor occupations with a given multiset. All copies separately have 128 of256 CAR modes. A bath transition mask A->B has coefficient sqrt(n_A*(n_B+1)) times the literal single-copy CAR sign/amplitude; A=B has n_A. Fixed even particle parity permits commuting distinct-copy bilinears without a missing cross-copy sign. Active copy is distinct.',
 'flow': 'Vertices label integer192 edgeflow k. rho=Dk, Pi=Q^T k (129 orthonormal components in a projected lattice/coset, NOT Z^129). Q Q^T+D^T Lplus D=I gives HE+HC=64||k||^2 in conditional nativeinteger alpha0/beta0 seed sector. Aggregation occurs BEFORE squaring; independent-copy HC/HE is wrong. Full beta and minimizing beta OPEN.',
 'seed': 'Hadamard colour rotation diagonalizes Gamma0 exactly in algebra. Every species fills the lower two colours at every site, N=128 each. Aggregate occupancy128/site, Ref4 ->rho0, Pi0. This state is not an SU4 local singlet. Basis rotation is unitary on all256 CAR modes, not a truncation.',
 'construction': 'Density-lock superexchange source is P_(k=0) Hhop^2 seed with its seed component removed. It sums ALL native reverse paths, colours, active and symmetric63 bath transitions with normalized multiplicities. It is a correlated two-hop source, not two selected auxiliary labels/loops. First try seed plus this densitylocked source. Material second design adds the complete allnative one-hop bridge: a three-direction variational polynomial state, not a separable onePH state. H on EVERY source direction is explicitly computed to every destination, including three-copy, higher-occupation and new Fourier exteriors. No output truncation.',
 'neutral': 'Tensor EXACT spectral wedge128 of Pplus=1_(0,infinity)(literal full-native h) for physical -Lambda h. Its exact spectral residual is zero. Archived floating Wplus is a diagnostic only; its offband defect is NOT multiplied by Lambda^2 and called small.',
 'scope': 'Frozen bosonic potential is a common additive constant only. Cartesian/Haar dynamics and normalized physical Phi/colour/Gauss determinant-line lift, Berry connection and envelope remain UNPAID. Current Theta-phase and Pi-translation are different parameters. Finite Ritz spacings are NOT spectral gaps. No stationarity, fullPhi ground, photon pole, commoncone or oldsource adoption.'
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def colour_frame(frame: np.ndarray) -> np.ndarray:
    # R = [[I,I],[-I,I]]/sqrt2. Use exact integer signs /2; no zero cropping.
    signs = np.array([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])
    return np.asarray([signs.T@m@signs/2 for m in frame])


class Graph:
    def __init__(self, native: dict[str, Any], profiles: list[dict[str, Any]]) -> None:
        self.native = native
        self.base = sum(3 << (4*x) for x in range(64))
        self.seed: Vertex = (self.base, (), ())
        self.entries = [native_graph.terms(p, native) for p in profiles]
        self.seed_onsite = sum((1 if role == 0 else 63)*sum(value.real for dest,k,value in self.local(role,self.base) if dest==self.base and not k) for role in (0,1))

    @lru_cache(maxsize=None)
    def local(self, role: int, mask: int) -> tuple[tuple[int, Flow, complex], ...]:
        return tuple(native_graph.local_action(mask, self.entries[role]))

    def moves(self, vertex: Vertex, hopping_only: bool = False) -> Iterator[tuple[Vertex, complex]]:
        active,bath,k = vertex
        for dest,shift,value in self.local(0,active):
            if hopping_only and not shift: continue
            yield (dest,bath,native_graph.add_flow(k,shift)),value
        counts = dict(bath)
        counts[self.base] = 63-sum(counts.values())
        for mask,count in counts.items():
            if not count: continue
            for dest,shift,value in self.local(1,mask):
                if hopping_only and not shift: continue
                if dest == mask:
                    yield (active,bath,native_graph.add_flow(k,shift)),count*value
                    continue
                destcount = counts.get(dest,0)
                nextcounts = dict(counts)
                nextcounts[mask] -= 1
                nextcounts[dest] = destcount+1
                nextbath = tuple(sorted((m,n) for m,n in nextcounts.items() if m!=self.base and n))
                yield (active,nextbath,native_graph.add_flow(k,shift)),math.sqrt(count*(destcount+1))*value

    def action(self, state: State) -> State:
        result: State = defaultdict(complex)
        for vertex,amplitude in state.items():
            k = vertex[2]
            # Shift by Eseed to avoid loss of variance from the huge additive onsite.
            result[vertex] += (64*sum(n*n for _,n in k)-self.seed_onsite)*amplitude
            for dest,value in self.moves(vertex):
                result[dest] += amplitude*value
        return {v:a for v,a in result.items() if a!=0}

    def locked_second(self, first: State) -> State:
        result: State = defaultdict(complex)
        for vertex,amplitude in first.items():
            for dest,value in self.moves(vertex,True):
                if not dest[2] and dest!=self.seed:
                    result[dest] += amplitude*value
        return {v:a for v,a in result.items() if a!=0}

    def density(self, vertex: Vertex) -> np.ndarray:
        active,bath,_ = vertex
        rho = np.array([((active>>(4*x))&15).bit_count()-2 for x in range(64)])
        for mask,count in bath:
            rho += count*np.array([((mask>>(4*x))&15).bit_count()-2 for x in range(64)])
        return rho


def inner(a: State,b: State) -> complex:
    if len(a)>len(b): return inner(b,a).conjugate()
    real=math.fsum((value.conjugate()*b.get(vertex,0)).real for vertex,value in a.items())
    imag=math.fsum((value.conjugate()*b.get(vertex,0)).imag for vertex,value in a.items())
    return complex(real,imag)


def normalize(state: State) -> State:
    norm = math.sqrt(inner(state,state).real)
    assert norm>0
    return {v:a/norm for v,a in state.items()}


def combine(states: list[State], coefficients: np.ndarray) -> State:
    result: State = defaultdict(complex)
    for state,c in zip(states,coefficients):
        for v,a in state.items(): result[v]+=complex(c)*a
    return {v:a for v,a in result.items() if a!=0}


def state_receipt(state: State, action: State, graph: Graph) -> dict[str, Any]:
    n=inner(state,state).real
    energy=inner(state,action).real/n
    norm=inner(action,action).real/n
    exterior=math.fsum(abs(a)**2 for v,a in action.items() if v not in state)/n
    higher=math.fsum(abs(a)**2 for v,a in action.items() if sum(count for _,count in v[1])>=3)/n
    return dict(norm=n,energy_shift=energy,energy=energy+graph.seed_onsite,variance=norm-energy*energy,complete_action_norm_squared_shifted=norm,exterior_action_norm_squared=exterior,three_or_more_modified_aux_escape_norm_squared=higher,input_vertices=len(state),complete_action_destinations=len(action),stationary=False)


def run(output: Path) -> dict[str, Any]:
    started=time.monotonic()
    names=['joined_phi_u1_neutral_ground_graph_delivery_receipt_20261010.json','joined_phi_u1_neutral_ground_graph_final_20261010.json','joined_phi_u1_neutral_ground_graph_final_20261010.npz','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json']
    inputs={name:(ROOT/name).read_bytes() for name in names}
    assert sha(inputs[names[0]])=='7de498d59c4a4c8ab2dad6cdb70f6a019164b8d40f711850bd5e418d56d97447'
    delivery=json.loads(inputs[names[0]])
    for name in names[1:3]:
        assert sha(inputs[name])==delivery['exact_artifacts'][name]['sha256']
    data=field.load_mean(g_override=[float(g) for g in native_graph.G],mean_descriptor=json.loads(inputs[names[3]])['input_descriptor'])
    original=data['native']; native=dict(original); native['frame']=colour_frame(original['frame'])
    scale = -native['frame'][0,0,0].real
    assert scale > 0 and np.array_equal(native['frame'][0],scale*np.diag([-1,-1,1,1]))
    graph=Graph(native,data['profiles'][1:3])
    seed={graph.seed:1+0j}
    first=graph.action(seed)
    assert abs(first.pop(graph.seed,0))<1e-8
    first=normalize(first)
    locked=normalize(graph.locked_second(first))
    multiplicity_controls=controls.run(graph,first,native_graph.previous.car_source.car,field.native_paths(data['profiles'][2],native))
    states=[seed,first,locked]
    actions=[graph.action(state) for state in states]
    h=np.array([[inner(s,a) for a in actions] for s in states])
    gram=np.array([[inner(a,b) for b in actions] for a in actions])
    assert np.linalg.norm(h-h.conj().T)<1e-7
    assert np.linalg.norm(np.array([[inner(a,b) for b in states] for a in states])-np.eye(3))<1e-8
    seed_result=state_receipt(seed,actions[0],graph)
    # First material design: strict density-locked superexchange (no first shell).
    e_lock,c_lock=np.linalg.eigh(h[np.ix_([0,2],[0,2])])
    coeff_lock=np.array([c_lock[0,0],0,c_lock[1,0]])
    lock_state=combine(states,coeff_lock);lock_action=combine(actions,coeff_lock)
    lock_result=state_receipt(lock_state,lock_action,graph)
    del lock_action
    # Material repair: permit virtual collective electric excursions, no crop of H.
    energies,coeffs=np.linalg.eigh(h);coeff=coeffs[:,0]
    chosen=combine(states,coeff);chosen_action=combine(actions,coeff)
    chosen_result=state_receipt(chosen,chosen_action,graph)
    checkpoint=dict(stage='COMPLETE_H_ACTION_CALCULATED_PENDING_CONTROLS',seed=seed_result,densitylocked_attempt=lock_result,virtual_electric_superexchange=chosen_result,gram_variance=float((coeff.conj()@gram@coeff-energies[0]**2).real))
    output.with_name(output.stem+'_checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
    assert chosen_result['energy']<seed_result['energy']-1
    assert chosen_result['exterior_action_norm_squared']>1
    assert chosen_result['three_or_more_modified_aux_escape_norm_squared']>1
    assert abs(chosen_result['variance']-(coeff.conj()@gram@coeff-energies[0]**2).real)<1e-5
    # Discriminating controls on actual vertices, not copied expected fields.
    witness=next(v for v in locked if len(v[1])==2)
    assert np.array_equal(graph.density(witness),np.zeros(64)) and not witness[2]
    independent=0.0
    for mask,count in witness[1]:
        one=next(v for v in first if any(m==mask for m,_ in v[1]))
        independent+=count*64*sum(n*n for _,n in one[2])
    assert independent>1
    moved=next(v for v in first if v[1])
    nmod=sum(n for _,n in moved[1]);assert nmod==1
    assert graph.base.bit_count()==128
    assert all(v[0].bit_count()==128 and all(m.bit_count()==128 for m,_ in v[1]) and sum(n for _,n in v[1])<=63 for s in states for v in s)
    gauss_error=0.0;identity_error=0.0
    for v in chosen:
        k=native_graph.dense_flow(v[2]);rho=graph.density(v)
        gauss_error=max(gauss_error,float(np.linalg.norm(native['D']@k-rho)))
        full=64*(np.linalg.norm(native['Q'].T@k)**2+rho@native['Lplus']@rho)
        identity_error=max(identity_error,abs(float(full)-64*np.dot(k,k)))
    assert gauss_error==0 and identity_error<1e-7
    # Incoming electric offset is a separate conditional sector, not discarded.
    incoming=.37*(native['Q']@native['Q'].T)[:,0]
    k=native_graph.dense_flow(moved[2])
    incoming_transfer=64*(np.dot(incoming+k,incoming+k)-np.dot(incoming,incoming))
    zero_transfer=64*np.dot(k,k)
    assert abs(incoming_transfer-zero_transfer)>1e-3
    qprobes=[]
    loops=native_graph.previous.plaquettes(native)
    for qi in ((1,0,0),(2,0,0)):
        v=np.cos(2*np.pi*native['sites']@np.array(qi)/4)@loops
        v=v/np.linalg.norm(v)
        characteristic=sum(abs(a)**2*np.exp(1.3j*np.dot(v,native_graph.dense_flow(key[2]))) for key,a in chosen.items())
        phase_current: State=defaultdict(complex);phase_contact: State=defaultdict(complex)
        for vertex,amp in chosen.items():
            for dest,value in graph.moves(vertex,True):
                shift=native_graph.add_flow(dest[2],tuple((e,-n) for e,n in vertex[2]))
                dot=sum(v[e]*n for e,n in shift)
                phase_current[dest]+=1j*dot*amp*value
                phase_contact[dest]-=dot*dot*amp*value
        pi_current=sum(abs(a)**2*(128*np.dot(v,native_graph.dense_flow(key[2])))**2 for key,a in chosen.items())
        qprobes.append(dict(q_integer=qi,q_physical_over_inverse_a=[2*math.pi*q/4 for q in qi],connected_electric_Weyl_weight=1-abs(characteristic)**2,Theta_phase_current_action_norm_squared=inner(phase_current,phase_current).real,Theta_phase_contact_action_norm_squared=inner(phase_contact,phase_contact).real,Pi_translation_current_action_norm_squared=float(pi_current),Pi_translation_contact=128.,pole='NOT_CLAIMED'))
    state_rows=[dict(active_mask=str(v[0]),auxiliary_multiset=[(str(m),n) for m,n in v[1]],integer192_flow=v[2],amplitude=[a.real,a.imag]) for v,a in chosen.items()]
    neutral=json.loads(inputs[names[0]])['neutral_result']
    result: dict[str, Any]
    result=dict(kind='ALLNATIVE_SYMMETRIC63_DENSITYLOCK_SUPEREXCHANGE_COMPLETE_H_VARIATIONAL_STATE_NOT_GROUND',proof=PROOF,law=dict(N=4,V=64,canonical_each=128,charged_copies=64,auxiliary_copies=63,Ref=4,HE=64,HC=64,Lambda='2^8192',chi='2^-4096',Lambda54='2^10240',neutral='EXACT_Pplus_wedge128'),seed=seed_result,densitylocked_attempt=lock_result,virtual_electric_superexchange=chosen_result,chosen_coefficients=[[float(c.real),float(c.imag)] for c in coeff],matrix_shifted_h=[[ [float(z.real),float(z.imag)] for z in row] for row in h],complete_action_gram=[[ [float(z.real),float(z.imag)] for z in row] for row in gram],controls=dict(wrong_independent_copy_densitylocked_energy=independent,proper_densitylocked_HEHC=0,gauss_error=gauss_error,fullHEHC_identity_error=identity_error,incoming_transfer=float(incoming_transfer),wrong_zero_incoming_transfer=float(zero_transfer),wrong_omit_exterior_variance=chosen_result['variance']-chosen_result['exterior_action_norm_squared'],wrong_aux_multiplicity_seed_ratio=63,all256_CAR_allnative_AP=True),spatial_probes=qprobes,neutral=dict(exact_spectral_residual='0',floating_offband_defect_over_Lambda2=neutral['observed_binary_offband_residual_over_Lambda2'],floating_diagnostic_not_physical_residual=True),state=state_rows,numerics='Floating evaluation at pinned binary section; exact basis/symmetry/HEHC derivations, not outward energy certificate.',scientific_verdict='Correlated finite conditional variational energy improvement only. Residual assessed against balanced seed; not stationary and no photon pole. FullPhi lift/Gauss/Berry and minimizing beta OPEN.',elapsed_seconds=time.monotonic()-started)
    result['residual_change_vs_balanced_seed']=chosen_result['variance']-seed_result['variance']
    result['accepted_as_stationary_ground']=False
    result['discriminating_multiplicity_CAR_native_controls']=multiplicity_controls
    result['all63_executed_state_density_correlation']=controls.density_correlations(graph,chosen)
    upper_mask=sum(12 << (4*x) for x in range(64))
    def ph_order(v: Vertex) -> int:
        return (v[0]&upper_mask).bit_count()+sum(n*(m&upper_mask).bit_count() for m,n in v[1])
    source_max=max(map(ph_order,chosen));action_max=max(map(ph_order,chosen_action))
    ph3_norm=math.fsum(abs(a)**2 for v,a in chosen_action.items() if ph_order(v)>=3)
    assert source_max<=2 and action_max<=3 and ph3_norm>1
    source_flows={key[2] for key in chosen}
    result['complete_h_occupation_Fourier_tail_payment']=dict(source_max_upper_colour_particles=source_max,action_max_upper_colour_particles=action_max,three_or_more_PH_action_norm_squared=ph3_norm,above_three_PH_action='EXACT_ZERO: onsite diagonal and native one-body hopping change upper occupancy by at most one, HEHC diagonal',Fourier_exterior_action_norm_squared=math.fsum(abs(a)**2 for v,a in chosen_action.items() if v[2] not in source_flows),all_exteriors_summed_before_squaring=True)
    result['full_frozen_energy_decomposition']=dict(common_exact_neutral='-Lambda Tr(Pplus h), SAME exact neutral state for both charged trials',common_frozen_bosonic='NEW Jordan quartic + alignment + plaquette at the pinned Phi0/identity links',displayed_energy_scope='The numerical energy fields are charged+HE+HC. Add the same neutral and frozen bosonic constants to obtain complete fixed-Phi H energies; they cancel in the gain. The numerical variance is the complete fixed-Phi H variance because the exact neutral spectral factor has zero residual.',not_included='Cartesian/Haar kinetic on a normalized physical Phi/Gauss/Berry extension: not yet constructed or paid')
    captured={Path(__file__).name:Path(__file__).read_bytes(),**field.CAPTURED}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.parent==ROOT:captured.setdefault(path.name,path.read_bytes())
    result['source_sha256']={k:sha(raw) for k,raw in captured.items()}
    result['input_sha256']={k:sha(raw) for k,raw in inputs.items()}
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    assert all((ROOT/name).read_bytes()==raw for name,raw in inputs.items())
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in captured.items():z.writestr('executed_source/'+name,raw)
        for name,raw in inputs.items():z.writestr('inputs/'+name,raw)
    output.write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):receipt=run(args.output)
    print(json.dumps({k:receipt[k] for k in ('kind','seed','densitylocked_attempt','virtual_electric_superexchange','residual_change_vs_balanced_seed','elapsed_seconds')},indent=2))
