"""All64-copy correlated character resummation; complete frozen H norm, not ground."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_collective_charged_ground_consumer as old

ROOT = Path(__file__).resolve().parent
Flow = old.Flow
Local = dict[tuple[int, Flow], complex]
INPUTS = ['joined_phi_u1_collective_charged_ground_delivery_receipt_20261010.json', 'joined_phi_u1_collective_charged_ground_final_20261010.json', 'joined_phi_u1_collective_charged_ground_final_20261010_natural_exit.json', 'joined_phi_u1_collective_charged_ground_final_20261010.zip', 'joined_phi_u1_collective_charged_ground_final_20261010_preexecution.zip', 'joined_phi_u1_neutral_ground_graph_delivery_receipt_20261010.json', 'joined_phi_u1_neutral_ground_graph_final_20261010.json', 'joined_phi_u1_neutral_ground_graph_final_20261010.npz', 'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json', 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json', 'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json']

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def action(graph: old.Graph, role: int, state: Local) -> Local:
    result: Local = defaultdict(complex)
    onsite = sum(a for m,k,a in graph.local(role,graph.base) if m == graph.base and not k)
    for (mask,k),a in state.items():
        result[mask,k] -= onsite*a
        for dest,shift,b in graph.local(role,mask):
            result[dest,old.native_graph.add_flow(k,shift)] += a*b
    return {v:a for v,a in result.items() if a != 0}

def zero_moments() -> tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    return np.zeros(192),np.zeros((192,192)),np.zeros(192),0.

def moments(rows: list[tuple[Flow,complex]], divisor: complex) -> tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    m,M,t,f = zero_moments()
    for k,w0 in rows:
        w = w0/divisor
        assert abs(w.imag) < 1e-12
        v = old.native_graph.dense_flow(k);r = float(v@v)
        m += w.real*v;M += w.real*np.outer(v,v);t += w.real*r*v;f += w.real*r*r
    return m,M,t,f

def add_moments(a: Any,b: Any) -> Any:
    m,M,t,f = a;u,S,v,g = b
    return (m+u,M+S+np.outer(m,u)+np.outer(u,m),t+v+np.trace(M)*u+np.trace(S)*m+2*M@u+2*S@m,float(f+g+2*np.trace(M)*np.trace(S)+4*np.sum(M*S)+4*t@u+4*v@m))

def total_moments(local: list[Any], counts: list[int]) -> Any:
    result = zero_moments()
    for item,count in zip(local,counts):
        for _ in range(count):result = add_moments(result,item)
    return result

def local_pair(a: Local,b: Local,ha: Local,hb: Local) -> dict[str,Any]:
    # Each source occupation has ONE character. This is tested, not assumed.
    assert len({mask for mask,k in a}) == len(a)
    assert set(a) == set(b)
    source_by_mask = {mask:k for mask,k in a}
    rows = [(k,a[mask,k].conjugate()*b[mask,k]) for mask,k in a]
    s = sum(w for k,w in rows);assert abs(s.imag)<1e-10 and s.real>0
    X: dict[Flow,complex] = defaultdict(complex)
    Cvec = np.zeros(192,dtype=complex);C2 = 0j
    for (mask,k),amp in hb.items():
        if mask not in source_by_mask:continue
        source_k = source_by_mask[mask]
        delta = old.native_graph.add_flow(k,tuple((e,-n) for e,n in source_k))
        value = a[mask,source_k].conjugate()*amp
        X[delta] += value
        if not delta:
            v = old.native_graph.dense_flow(k);Cvec += value*v;C2 += value*(v@v)
    D = sum(v.conjugate()*hb.get(key,0j) for key,v in ha.items())
    Do = sum(v.conjugate()*hb.get(key,0j) for key,v in ha.items() if key[0] not in source_by_mask)
    Df = sum(v.conjugate()*hb.get(key,0j) for key,v in ha.items() if key[0] in source_by_mask and key[1] != source_by_mask[key[0]])
    return dict(s=s,X=dict(X),Cvec=Cvec,C2=C2,D=D,Do=Do,Df=Df,moments=moments(rows,s))

def contract(states: list[list[Local]], images: list[list[Local]], counts: list[int], incoming: np.ndarray) -> dict[str,np.ndarray]:
    n = len(states[0]);matrices = {name:np.zeros((n,n),dtype=complex) for name in ('S','H','H2','occupation_tail','Fourier_tail')}
    pairs = [[ [local_pair(states[r][a],states[r][b],images[r][a],images[r][b]) for b in range(n)] for a in range(n)] for r in range(len(counts))]
    for a in range(n):
        for b in range(n):
            ps = [pairs[r][a][b] for r in range(len(counts))]
            overlap = math.prod(p['s']**c for p,c in zip(ps,counts))
            mom = total_moments([p['moments'] for p in ps],counts);m,M,t,f = mom
            e = np.trace(M)+2*incoming@m
            e2 = f+4*incoming@t+4*incoming@M@incoming
            h = 64*e*overlap;hh = 4096*e2*overlap;ot = 0j;ft = 0j;cross = 0j
            for r,(pr,c) in enumerate(zip(ps,counts)):
                if not c:continue
                weight = overlap/pr['s']
                h += c*pr['X'].get((),0j)*weight
                hh += c*pr['D']*weight
                ot += c*pr['Do']*weight;ft += c*pr['Df']*weight
                remain = counts.copy();remain[r] -= 1
                mr,Mr,_,_ = total_moments([p['moments'] for p in ps],remain)
                cross += c*weight*(pr['C2']+2*pr['Cvec']@mr+pr['X'].get((),0j)*np.trace(Mr)+2*incoming@(pr['Cvec']+pr['X'].get((),0j)*mr))
                for q,(pq,d) in enumerate(zip(ps,counts)):
                    multiplicity = c*(d-int(q==r))
                    if not multiplicity:continue
                    # <h phi_a|phi_b>_r <phi_a|h phi_b>_q, Fourier constant.
                    reverse = pairs[r][b][a]['X']
                    value = sum(z.conjugate()*pq['X'].get(k,0j) for k,z in reverse.items())
                    wrongflow = sum(z.conjugate()*pq['X'].get(k,0j) for k,z in reverse.items() if k)
                    weight2 = overlap/(pr['s']*pq['s'])
                    hh += multiplicity*weight2*value;ft += multiplicity*weight2*wrongflow
            # Reverse electric/h order obtained from the conjugate opposite branch.
            matrices['S'][a,b] = overlap;matrices['H'][a,b] = h
            matrices['H2'][a,b] = hh
            matrices['occupation_tail'][a,b] = ot;matrices['Fourier_tail'][a,b] = ft
            # Temporarily keep E*h, symmetrized after all branch pairs are known.
            if 'mixed' not in matrices:matrices['mixed'] = np.zeros((n,n),dtype=complex)
            matrices['mixed'][a,b] = cross
    matrices['H2'] += 64*(matrices['mixed']+matrices['mixed'].conj().T)
    for name,value in matrices.items():
        if name != 'mixed':assert np.linalg.norm(value-value.conj().T)<1e-5, name
    return matrices

def evaluate(matrices: dict[str,np.ndarray], coeff: np.ndarray) -> dict[str,Any]:
    values = {name:float(np.real(coeff.conj()@value@coeff)) for name,value in matrices.items()}
    norm = values['S'];energy = values['H']/norm;h2 = values['H2']/norm
    return dict(norm=norm,energy_shift=energy,complete_shifted_H_norm_squared=h2,variance=h2-energy*energy,occupation_exterior_norm_squared=values['occupation_tail']/norm,Fourier_exterior_norm_squared=values['Fourier_tail']/norm,stationary=False)

def explicit_control(graph: old.Graph,incoming: np.ndarray) -> dict[str,Any]:
    # Independent labelled two-copy full-native H consumer, not a toy Hamiltonian.
    sources: list[Local] = []
    for role in (0,1):
        first = [(m,k,a) for m,k,a in graph.local(role,graph.base) if k]
        chosen = [first[0],next(x for x in first if x[1] == tuple((e,-n) for e,n in first[0][1]))]
        sources.append({(graph.base,()):1+0j,**{(m,k):(.2+.07*j)*a/abs(a) for j,(m,k,a) in enumerate(chosen)}})
    images = [action(graph,r,s) for r,s in enumerate(sources)]
    predicted = evaluate(contract([[s] for s in sources],[[h] for h in images],[1,1],incoming),np.ones(1))
    state: dict[Any,complex] = defaultdict(complex)
    for (m,k),a in sources[0].items():
        for (n,l),b in sources[1].items():state[m,n,old.native_graph.add_flow(k,l)] += a*b
    image: dict[Any,complex] = defaultdict(complex)
    onsite = graph.seed_onsite
    onsite_local = [sum(v for m,k,v in graph.local(r,graph.base) if m==graph.base and not k) for r in (0,1)]
    for (m,n,k),a in state.items():
        v = old.native_graph.dense_flow(k)
        image[m,n,k] += complex(64*(v@v+2*incoming@v)*a)
        for role,mask in enumerate((m,n)):
            image[m,n,k] -= onsite_local[role]*a
            for dest,shift,b in graph.local(role,mask):
                key = (dest,n,old.native_graph.add_flow(k,shift)) if role==0 else (m,dest,old.native_graph.add_flow(k,shift))
                image[key] += a*b
    norm = math.fsum(abs(a)**2 for a in state.values())
    energy = sum(a.conjugate()*image.get(v,0j) for v,a in state.items()).real/norm
    h2 = math.fsum(abs(a)**2 for a in image.values())/norm
    assert abs(energy-predicted['energy_shift'])<1e-8 and abs(h2-predicted['complete_shifted_H_norm_squared'])<1e-5
    exterior = math.fsum(abs(a)**2 for v,a in image.items() if v not in state)/norm
    assert abs(exterior-predicted['occupation_exterior_norm_squared']-predicted['Fourier_exterior_norm_squared'])<1e-5
    independent = 0.
    for (m,n,k),a in state.items():
        k0 = next(kk for mm,kk in sources[0] if mm==m);k1 = next(kk for mm,kk in sources[1] if mm==n)
        independent += abs(a)**2*64*(sum(v*v for _,v in k0)+sum(v*v for _,v in k1))/norm
    proper = sum(abs(a)**2*64*sum(v*v for _,v in k) for (m,n,k),a in state.items())/norm
    assert abs(proper-independent)>1e-3
    # Independent labelled two-bath cat covariance, using actual native CAR masks.
    toy = sources[1]
    occupied = [key for key in toy if key[1]]
    opposite = dict(toy)
    opposite[occupied[0]],opposite[occupied[1]] = toy[occupied[1]],toy[occupied[0]]
    branches = [toy,opposite]
    explicit: dict[Any,complex] = defaultdict(complex)
    for branch in branches:
        for (m,k),a in branch.items():
            for (n,l),b in branch.items():explicit[m,n,old.native_graph.add_flow(k,l)] += a*b
    toy_norm = math.fsum(abs(a)**2 for a in explicit.values())
    mean0 = np.zeros(64);mean1 = np.zeros(64);pair = np.zeros(64)
    for (m,n,k),a in explicit.items():
        rho0 = np.array([((m>>(4*x))&15).bit_count()-2 for x in range(64)])
        rho1 = np.array([((n>>(4*x))&15).bit_count()-2 for x in range(64)])
        w = abs(a)**2/toy_norm
        mean0 += w*rho0;mean1 += w*rho1;pair += w*rho0*rho1
    reference = pair-mean0*mean1
    inferred = density_correlations(branches,np.ones(2),[{(graph.base,()):1+0j}]*2,2)
    covariance_error = float(np.max(np.abs(reference-np.array(inferred['any_two_distinct_auxiliary_connected_onsite_density_covariance']))))
    assert covariance_error<1e-12 and np.max(np.abs(reference))>1e-6
    return dict(labelled_actual_full_native_two_copy_destinations=len(image),energy=energy,full_H_norm_squared=h2,exterior=exterior,contraction_error=abs(h2-predicted['complete_shifted_H_norm_squared']),wrong_independent_electric=independent,proper_aggregate_electric=proper,unused_64copy_seed_offset=onsite,labelled_two_bath_covariance_error=covariance_error,labelled_two_bath_nonzero_covariance=float(np.max(np.abs(reference))))

def density_correlations(states: list[Local],coeff: np.ndarray,active: list[Local],counts: int) -> dict[str,Any]:
    mean = np.zeros(64);pair = np.zeros(64);norm = 0j
    for a in range(len(states)):
        for b in range(len(states)):
            rows = [(mask,states[a][mask,k].conjugate()*states[b][mask,k]) for mask,k in states[a]]
            s = sum(w for mask,w in rows);sa = sum(v.conjugate()*active[b][key] for key,v in active[a].items())
            rho = sum((w*np.array([((mask>>(4*x))&15).bit_count()-2 for x in range(64)]) for mask,w in rows),np.zeros(64,dtype=complex))
            weight = coeff[a].conjugate()*coeff[b]*sa
            norm += weight*s**counts
            mean += (weight*s**(counts-1)*rho).real
            pair += (weight*s**(counts-2)*rho*rho).real
    covariance = pair/norm.real-(mean/norm.real)**2
    assert np.max(np.abs(covariance))>1e-12
    return dict(all63_exact_permutation_symmetry=True,any_two_distinct_auxiliary_connected_onsite_density_covariance=covariance.tolist(),max_abs=float(np.max(np.abs(covariance))),wrong_independent_copy_prediction=0.)

def run(output: Path) -> dict[str,Any]:
    started = time.monotonic();inputs = {name:(ROOT/name).read_bytes() for name in INPUTS}
    delivery = json.loads(inputs[INPUTS[0]]);previous = json.loads(inputs[INPUTS[1]])
    assert sha(inputs[INPUTS[0]])=='c44ab260702856ae6d9b7b19cd2ab69f950d49d69a31ea4742e989fb49b0ea03'
    for name in INPUTS[1:5]:assert sha(inputs[name])==delivery['exact_artifact_sha256'][name]
    capsule = json.loads(inputs[INPUTS[2]]);assert capsule['returncode']==0 and capsule['natural_exit'] and capsule['all_sources_inputs_unchanged']
    predecessor_source = Path(old.__file__).read_bytes()
    assert sha(predecessor_source)==delivery['exact_artifact_sha256'][Path(old.__file__).name]
    with zipfile.ZipFile(io.BytesIO(inputs[INPUTS[3]])) as archive:
        assert archive.read('executed_source/'+Path(old.__file__).name)==predecessor_source
    with zipfile.ZipFile(io.BytesIO(inputs[INPUTS[4]])) as archive:
        assert archive.read(Path(old.__file__).name)==predecessor_source
    assert previous['virtual_electric_superexchange']['variance']>1e6 and previous['complete_h_occupation_Fourier_tail_payment']['Fourier_exterior_action_norm_squared']>1e6
    data = old.field.load_mean(g_override=[float(g) for g in old.native_graph.G],mean_descriptor=json.loads(inputs[INPUTS[8]])['input_descriptor'])
    native = dict(data['native']);native['frame'] = old.colour_frame(native['frame']);graph = old.Graph(native,data['profiles'][1:3])
    incoming = .37*(native['Q']@native['Q'].T)[:,0]
    HEHC_identity_error = float(np.linalg.norm(native['Q']@native['Q'].T+native['D'].T@native['Lplus']@native['D']-np.eye(192)))
    assert HEHC_identity_error<1e-10 and np.linalg.norm(native['D']@incoming)<1e-10
    # A termwise integer continuity proof covers every composed H destination.
    continuity_checks = 0
    for entries in graph.entries:
        for create,annihilate,k,value in entries:
            rho = np.zeros(64)
            rho[create//4] += 1;rho[annihilate//4] -= 1
            assert np.array_equal(native['D']@old.native_graph.dense_flow(k),rho)
            continuity_checks += 1
    control = explicit_control(graph,incoming)
    directions: list[list[Local]] = [];direction_images: list[list[Local]] = []
    parity = (-1.)**np.sum(native['sites'],axis=1)
    for role in (0,1):
        first = {(m,k):a for m,k,a in graph.local(role,graph.base) if k};norm = math.sqrt(sum(abs(a)**2 for a in first.values()))
        assert len(first)==768 and len({m for m,k in first})==768
        f = {v:a/norm for v,a in first.items()}
        biased: Local = {}
        for (mask,k),a in f.items():
            rho = np.array([((mask>>(4*x))&15).bit_count()-2 for x in range(64)])
            assert np.array_equal(native['D']@old.native_graph.dense_flow(k),rho)
            biased[mask,k] = a*float(parity@rho)/2
        assert sum(abs(a)**2 for a in biased.values())>.1
        ds = [{(graph.base,()):1+0j},f,biased];directions.append(ds);direction_images.append([action(graph,role,s) for s in ds])
    # Material source pays complete virtual-electric processes across ALL64 copies.
    # Pair branch sum restores checkerboard inversion and entangles every bath pair.
    trials = [];selected: Any = None
    for c in (-.15,-.3,-.45,-.6,-.8,-1.):
        states = [];images = []
        for role in (0,1):
            role_states = [];role_images = []
            for sign in (-1,1):
                coeff = np.array([1.,c,sign*.35*c])
                s: Local = defaultdict(complex);h: Local = defaultdict(complex)
                for j in range(3):
                    for key,a in directions[role][j].items():s[key] += coeff[j]*a
                    for key,a in direction_images[role][j].items():h[key] += coeff[j]*a
                local_norm = math.sqrt(sum(abs(a)**2 for a in s.values()))
                role_states.append({v:a/local_norm for v,a in s.items()});role_images.append({v:a/local_norm for v,a in h.items() if a!=0})
            states.append(role_states);images.append(role_images)
        matrices = contract(states,images,[1,63],np.zeros(192));receipt = evaluate(matrices,np.ones(2));receipt['c'] = c;trials.append(receipt)
        if selected is None or receipt['energy_shift']<selected[0]['energy_shift']:selected = (receipt,states,images,matrices)
    chosen,states,images,matrices = selected
    upper_mask = sum(12<<(4*x) for x in range(64))
    assert all((m&upper_mask).bit_count()<=1 for role in states for s in role for m,k in s)
    assert max((m&upper_mask).bit_count() for role in images for h in role for m,k in h)==2
    offset = evaluate(contract(states,images,[1,63],incoming),np.ones(2))
    assert abs(offset['complete_shifted_H_norm_squared']-chosen['complete_shifted_H_norm_squared'])>1e-3
    # The nonzero bias is a named correlated trial, NOT a proved optimum.
    unbiased_states = [];unbiased_images = []
    for role in (0,1):
        us = dict(directions[role][0]);uh = dict(direction_images[role][0])
        for key,a in directions[role][1].items():us[key] = us.get(key,0j)+chosen['c']*a
        for key,a in direction_images[role][1].items():uh[key] = uh.get(key,0j)+chosen['c']*a
        ln = math.sqrt(sum(abs(a)**2 for a in us.values()))
        unbiased_states.append([{key:a/ln for key,a in us.items()}])
        unbiased_images.append([{key:a/ln for key,a in uh.items() if a!=0}])
    unbiased = evaluate(contract(unbiased_states,unbiased_images,[1,63],np.zeros(192)),np.ones(1))
    unbiased['distinct_aux_density_covariance'] = 0.
    unbiased['bias_optimized'] = False
    covariance = density_correlations(states[1],np.ones(2),states[0],63)
    full_tail = chosen['occupation_exterior_norm_squared']+chosen['Fourier_exterior_norm_squared']
    assert full_tail>1 and chosen['variance']>=full_tail-1e-5
    output.with_name(output.stem+'_checkpoint.json').write_text(json.dumps(dict(trials=trials,chosen=chosen,controls=control,correlations=covariance),indent=2)+'\n')
    assert chosen['energy_shift']<previous['virtual_electric_superexchange']['energy_shift']-1
    first_global = graph.action({graph.seed:1+0j})
    multiplicity = old.controls.run(graph,first_global,old.native_graph.previous.car_source.car,old.field.native_paths(data['profiles'][2],native))
    paths = old.field.native_paths(data['profiles'][2],native)
    seam_path = next(path for path in paths if path['seam']<0)
    seam_hop = np.array([[complex(float(r),float(i)) for r,i in row] for row in old.native_graph.previous.exact_hop(seam_path,native['frame'])])
    wrong_hop = np.array([[complex(float(r),float(i)) for r,i in row] for row in old.native_graph.previous.exact_hop({**seam_path,'seam':1},native['frame'])])
    seam_wrong_distance = float(np.linalg.norm(seam_hop-wrong_hop)**2)
    assert seam_wrong_distance>1
    actual_mask = next(m for m,k in states[1][0] if k)
    full_local = {(m,k):a for m,k,a in graph.local(1,actual_mask)}
    nearest_only = {(m,k):a for m,k,a in old.native_graph.local_action(actual_mask,[term for term in graph.entries[1] if len(term[2])<2])}
    omitted_twohop = math.fsum(abs(a-nearest_only.get(key,0j))**2 for key,a in full_local.items())
    assert omitted_twohop>1
    result: dict[str,Any] = dict(kind='ALLNATIVE_ALL63_CORRELATED_CHARACTER_RESUMMATION_FULL_H_EXTERIOR_PAID_NOT_GROUND',old_failed_stationarity_control=previous['virtual_electric_superexchange'],trials=trials,chosen=chosen,conditional_nonzero_incoming_Pi=offset,correlations=covariance,independent_labelled_full_H_control=control,CAR_multiplicity_AP_native_controls=multiplicity)
    result['unbiased_material_fork_control'] = unbiased
    result['bias_scope'] = 'Bias magnitude.35 is a fixed correlated construction input, not a variational optimum. If unbiased energy is lower, keep that fact and its exactlyzero distinct-copy density covariance; no correlation/ground optimization claim.'
    result['independent_native_path_wrongcontrols'] = dict(dropping_all_two_link_action_on_actual_PH_mask_norm_squared=omitted_twohop,removing_AP_seam_phase_matrix_distance_squared=seam_wrong_distance,termwise_exact_integer_continuity_checks=continuity_checks)
    result['stationarity_change'] = dict(energy_gain_vs_previous=previous['virtual_electric_superexchange']['energy_shift']-chosen['energy_shift'],complete_variance_change=chosen['variance']-previous['virtual_electric_superexchange']['variance'],old_stationarity=False,new_stationarity=False)
    result['complete_shared_character_branch_matrices'] = {name:[[[float(z.real),float(z.imag)] for z in row] for row in value] for name,value in matrices.items()}
    result['HEHC_projector_identity_error'] = HEHC_identity_error
    result['absolute_charged_HEHC_energy'] = graph.seed_onsite+chosen['energy_shift']
    result['absolute_charged_HEHC_H_norm_squared'] = chosen['complete_shifted_H_norm_squared']+2*graph.seed_onsite*chosen['energy_shift']+graph.seed_onsite**2
    result['material_change'] = dict(source='sum_sign tensor_product_64 [seed + c ALLNATIVE_firsthop + sign*.35*c checkerboard_density_weighted_ALLNATIVE_firsthop]. Each factor normalized separately. One distinct active and63 EXACT permutation symmetric auxiliaries; integer characters sum BEFORE HEHC square. This is not a linear combination of the old3 global source directions.',max_source_upper_colour_particles=64,max_action_upper_colour_particles=65,above65_occupation_action='EXACT ZERO: each native one-body action changes PH order by at most1 and electric energy is diagonal',local_source_characters=769,global_labelled_terms=str(769**64),symmetric_bath_occupation_orbits=str(769*math.comb(831,63)),allocated_local_full_action_destinations=[[len(h) for h in role] for role in images],global_bruteforce_allocated=False)
    result['exact_contraction'] = dict(norm='Source occupation uniquely specifies its integer character: local overlap Laurent polynomial is CONSTANT. Verified on all769 local occupations.',Hnorm='Full local h images retain every CAR/AP native destination. Sum_i h_i norm uses constant term of X_ba(delta)^* X_ab(delta), including nonzero loop characters. Aggregate electric norm uses exact independent-tensor raw moments through order4 with nonzero means; electric/h cross uses local0/1/2 flow moments and spectator mean/second moments. Branch pairs are contracted before normalized variance.',tails='Occupation exterior: local action mask absent from source, sum over64 copies. Fourier exterior: source mask with wrong character, INCLUDING cross-copy equal-loop interference; all nonzero Laurent coefficients contracted. Source has max64PH, complete next65PH occupation/Fourier image paid; no Ritz spacing/internal variance substitute.',independence='Two-copy labelled direct native H consumer independently checks full norm, incoming Pi, aggregate electric square and complete exterior.')
    result['wrongcontrols'] = dict(omitted_full_tail_wrong_variance=chosen['variance']-full_tail,missing_tail=full_tail,stationarity_would_be_wrong=True,independent_copy_covariance_wrong=0.,proper_density_covariance=covariance['max_abs'])
    result['physical_scope'] = dict(frozenPhi_native_integer_beta0_only=True,full_beta_minimum='OPEN',normalized_SU4_Phi_Gauss_Berry_Haar_fullPhi_lift='UNPAID',neutral='EXACT functional Pplus=1_(0,infinity)(literal fullnative h), spectral wedge128 for physical -Lambda h. No floating neutral defect multiplied by Lambda^2.',Gamma0='LowerGamma0 algebraic seed is NOT physical Phi0 eigenvalue1; basis Hadamard is unitary on all256 modes and fullnative physical onsite is read literally.',full_energy='Displayed shift is charged+aggregate HEHC minus old.seed_onsite. Add same exact neutral and frozen bosonic constants as predecessor; complete conditional variance unchanged.',ground=False,stationary=False,photon=False)
    result['volume_local_control'] = dict(reached=False,reason='This resums up to64 charged-copy excursions but each single-copy factor has at mostone spatial PH. At fixed species count as V grows it is NOT extensive in spatial volume: upper occupancy density <=64/V. Thus neither finite variance reduction nor a tiny fixed residual supplies q->0 uniform control.',exact_local_electric_correction='For product branch: E||sum k_i||^2=sum trace M_i+2 sum_{i<j}mu_i.mu_j; E||sum k_i||^4 includes all pair second-moment and third/mean connected terms. Opposite checkerboard cat has nonzero all63 pair density covariance. Independent-copy HEHC is rejected by actual labelled consumer.',remaining_source_directions='Within-copy twoPH and higher linked spatial virtual-electric excursions at every site; fullnative second-hop occupation/loop-Fourier coefficients are material paid exterior witnesses in this experiment, not a global no-go.')
    # Persist actual local source and complete-action material witnesses, not just norms.
    result['local_sources'] = [[ [dict(mask=str(m),flow=k,amplitude=[a.real,a.imag]) for (m,k),a in s.items()] for s in role] for role in states]
    witnesses = []
    for role in (0,1):
        source_masks = {m for m,k in states[role][0]}
        occ = sorted(((abs(a)**2,m,k,a) for (m,k),a in images[role][0].items() if m not in source_masks),reverse=True)[:8]
        for weight,m,k,a in occ:witnesses.append(dict(role=role,mask=str(m),integer_flow=k,local_action_amplitude=[a.real,a.imag],norm_squared=weight,upper_colour_particles=(m&sum(12<<(4*x) for x in range(64))).bit_count()))
    assert witnesses;result['new_material_exterior_witnesses'] = witnesses
    captured = dict(old.field.CAPTURED)
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            path = Path(name).resolve()
            if path.suffix=='.py' and path.parent==ROOT:captured[path.name] = path.read_bytes()
    result['source_sha256'] = {k:sha(v) for k,v in captured.items()};result['input_sha256'] = {k:sha(v) for k,v in inputs.items()};result['elapsed_seconds'] = time.monotonic()-started
    assert all((ROOT/k).read_bytes()==v for k,v in {**captured,**inputs}.items())
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for k,v in captured.items():archive.writestr('executed_source/'+k,v)
        for k,v in inputs.items():archive.writestr('inputs/'+k,v)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args = parser.parse_args()
    with threadpool_limits(limits=1):result = run(args.output)
    print(json.dumps({k:result[k] for k in ('kind','chosen','correlations','wrongcontrols','elapsed_seconds')},indent=2))
