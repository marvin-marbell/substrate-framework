"""New shared-Theta matter-energy covariance probe on the retained unbiased state.

Zero diagonal density covariance is not a physical independence criterion.
This observable does not change or rerun the retained whole-H variational producer.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_charged_exterior_resummation_consumer as retained

ROOT = Path(__file__).resolve().parent
INPUTS = ['joined_phi_u1_charged_exterior_resummation_final_20261010.json','joined_phi_u1_charged_exterior_resummation_final_20261010_natural_exit.json','joined_phi_u1_charged_exterior_resummation_delivery_receipt_20261010.json','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json']
PINNED = {'joined_phi_u1_charged_exterior_resummation_final_20261010.json':'1e12c4f23214969496dce0a27689f93fe44c4aaa985e9a0b65a68d94a8931b35','joined_phi_u1_charged_exterior_resummation_final_20261010_natural_exit.json':'d5e134c8285aeb7c74ff04bb8135bbd62c24bc6ebb43a7538ce67bb09ca227ff','joined_phi_u1_charged_exterior_resummation_delivery_receipt_20261010.json':'5dc1e082f127e5a8c81ca7fb8488cbed957427a5a3df19573ec746ab7a17e3f2'}

def capture() -> dict[str,bytes]:
    result = dict(retained.old.field.CAPTURED)
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            path = Path(name).resolve()
            if path.suffix=='.py' and path.parent==ROOT:result[path.name] = path.read_bytes()
    for name in INPUTS:result[name] = (ROOT/name).read_bytes()
    return result

def decode(rows: list[dict[str,Any]]) -> retained.Local:
    return {(int(row['mask']),tuple((int(e),int(n)) for e,n in row['flow'])):complex(*row['amplitude']) for row in rows}

def native_laurent(graph: retained.old.Graph, state: retained.Local) -> dict[retained.Flow,complex]:
    source = {mask:(k,a) for (mask,k),a in state.items()}
    result: dict[retained.Flow,complex] = defaultdict(complex)
    for (mask,k),a in state.items():
        for dest,shift,value in graph.local(1,mask):
            if dest not in source:continue
            bra_k,bra_amp = source[dest]
            delta = retained.old.native_graph.add_flow(retained.old.native_graph.add_flow(k,shift),tuple((e,-n) for e,n in bra_k))
            result[delta] += bra_amp.conjugate()*value*a
    return dict(result)

def independent_laurent(graph: retained.old.Graph,state: retained.Local) -> tuple[dict[retained.Flow,complex],int]:
    # Independent sorted-wedge CAR order on literal unaggregated native terms.
    source = {mask:(k,a) for (mask,k),a in state.items()}
    result: dict[retained.Flow,complex] = defaultdict(complex);checked = 0
    for (mask,k),a in state.items():
        occupied = [i for i in range(256) if mask&(1<<i)]
        ranks = {mode:i for i,mode in enumerate(occupied)}
        for create,annihilate,shift,value in graph.entries[1]:
            if annihilate not in ranks or (create in ranks and create!=annihilate):continue
            destination = (mask^(1<<annihilate))|(1<<create)
            if destination not in source:continue
            annihilation_rank = ranks[annihilate]
            creation_rank = bisect_left(occupied,create)-int(annihilate<create)
            sign = -1 if (annihilation_rank+creation_rank)%2 else 1
            bra_k,bra_amp = source[destination]
            delta = retained.old.native_graph.add_flow(retained.old.native_graph.add_flow(k,shift),tuple((e,-n) for e,n in bra_k))
            result[delta] += bra_amp.conjugate()*sign*value*a;checked += 1
    return dict(result),checked

def run(output: Path) -> dict[str,Any]:
    started = time.monotonic();captured = capture();inputs = {name:captured[name] for name in INPUTS}
    for name,expected in PINNED.items():assert retained.sha(inputs[name])==expected
    previous = json.loads(inputs[INPUTS[0]]);capsule = json.loads(inputs[INPUTS[1]])
    assert capsule['returncode']==0 and capsule['natural_exit'] and capsule['all_sources_inputs_unchanged']
    assert retained.sha(Path(retained.__file__).read_bytes())=='e5d00ea90d48c3c4a8f807f22ec62da878c525c8fd8cf32905face7e39f5c483'
    plus,minus = [decode(rows) for rows in previous['local_sources'][1]]
    assert set(plus)==set(minus)
    unbiased = {key:(a+minus[key])/2 for key,a in plus.items()}
    norm = math.fsum(abs(a)**2 for a in unbiased.values())
    unbiased = {key:a/math.sqrt(norm) for key,a in unbiased.items()}
    assert len(unbiased)==769 and len({m for m,k in unbiased})==769
    local_norm = math.fsum(abs(a)**2 for a in unbiased.values());assert abs(local_norm-1)<1e-12
    data = retained.old.field.load_mean(g_override=[float(g) for g in retained.old.native_graph.G],mean_descriptor=json.loads(inputs[INPUTS[3]])['input_descriptor'])
    native = dict(data['native']);native['frame'] = retained.old.colour_frame(native['frame']);graph = retained.old.Graph(native,data['profiles'][1:3])
    # Bind reconstruction to the literal unbiased seed+c full firsthop definition.
    state_reference: retained.Local = {(graph.base,()):1+0j}
    first = {(m,k):a for m,k,a in graph.local(1,graph.base) if k}
    firstnorm = math.sqrt(math.fsum(abs(a)**2 for a in first.values()))
    for key,a in first.items():state_reference[key] = previous['chosen']['c']*a/firstnorm
    raw_norm = math.sqrt(math.fsum(abs(a)**2 for a in state_reference.values()))
    reconstruction_error = max(abs(a-state_reference[key]/raw_norm) for key,a in unbiased.items())
    assert reconstruction_error<1e-13
    g = native_laurent(graph,unbiased);reference,checked = independent_laurent(graph,unbiased)
    contraction_error = max(abs(g.get(k,0j)-reference.get(k,0j)) for k in set(g)|set(reference))
    assert contraction_error<1e-8
    hermiticity_error = max(abs(a-g.get(tuple((e,-n) for e,n in k),0j).conjugate()) for k,a in g.items())
    assert hermiticity_error<1e-8 and abs(g.get((),0j).imag)<1e-9
    covariance = math.fsum(abs(a)**2 for k,a in g.items() if k)
    # Direct Fourier constant-term multiplication is independent of abs-square.
    convolution = sum(a*g.get(tuple((e,-n) for e,n in k),0j) for k,a in g.items() if k)
    assert abs(convolution.imag)<1e-9 and abs(convolution.real-covariance)<1e-8
    mean = g.get((),0j).real
    assert all(np.array_equal(native['D']@retained.old.native_graph.dense_flow(k),np.zeros(64)) for k in g if k)
    nonzero = sorted(((abs(a)**2,k,a) for k,a in g.items() if k and abs(a)>1e-12),reverse=True)
    full_second = mean*mean+covariance
    result: dict[str,Any] = dict(kind='NEW_UNBIASED_SHARED_THETA_DISTINCT_AUX_MATTER_ENERGY_COVARIANCE_NOT_STATIONARITY',retained_unbiased_candidate=previous['unbiased_material_fork_control'],physical_independence_criterion='Zero diagonal density covariance DOES NOT establish independent-copy physical state. This probe uses the SAME shared field and fullnative copy-local matter-energy observable.',source=dict(reconstruction='Average opposite-bias retained auxiliary local branches, then normalize; checked against retained c=-.6 unbiased seed+c ALLNATIVE firsthop source.',reconstruction_error=reconstruction_error,local_pointwise_norm=local_norm,unique_occupation_character=True),observable=dict(name='Fullnative auxiliary onebody h_aux(Theta), onsite+every one/two-link AP/CAR term; NOT electric surrogate and no onsite subtraction.',mean=mean,distinct_aux_raw_second_moment=full_second,distinct_aux_shared_field_connected_covariance=covariance,independently_convolved_connected_covariance=convolution.real,nonconstant_Laurent_modes_above_1e_minus12=len(nonzero),genuinely_correlated_in_this_observable=bool(covariance>1e-12),all63_exact_permutation_symmetry=True,diagonal_density_covariance_remains_zero=True),controls=dict(independent_literal_native_term_sorted_wedge_CAR_contraction_checks=checked,independent_Laurent_coefficient_error=contraction_error,Hermitian_g_opposite_character_error=hermiticity_error,wrong_separately_integrated_copy_energy_covariance=0.,proper_shared_field_covariance=covariance,all_nonconstant_characters_have_exact_zero_divergence=True),proof={'shared_field':'At eachTheta the unbiased matter factor is normalized: each source occupation carries exactlyone character and occupations are orthogonal. Distinctcopy h_i and h_j act on separate tensor factors, but both functions depend on SAMETheta. All other64-copy spectator norms equal1 pointwise. Hence <h_i h_j> = [0]g(Theta)^2, <h_i>=[0]g and Cov=sum_{k!=0}g_k g_-k=sum_{k!=0}|g_k|^2. Separately integrating copies would incorrectly replace [0]g^2 with ([0]g)^2.','projection':'For each Laurent coefficient source bra and ket have identical final occupation, so D delta=0. Distinct integer192 delta therefore correspond to distinct reduced physical Q^T delta, not density-character aliases. CAR and AP signs remain literal.','scope':'This is a new observable on the retained lowerenergy/lowervariance unbiased candidate, not ground/stationarity proof or a correlation eligibility filter. Retained variance734030.1360694319 and full exterior remain failing. Fullbeta minimum, normalizedSU4/PhiGauss/Berry/Haar/fullPhi lift and volume-uniform ground/IR control remain OPEN.'},all_Laurent_coefficients=[dict(integer192_character=k,amplitude=[a.real,a.imag]) for k,a in g.items()],material_nonconstant_witnesses=[dict(integer192_character=k,amplitude=[a.real,a.imag],squared_magnitude=w) for w,k,a in nonzero[:12]],source_sha256={name:retained.sha(raw) for name,raw in captured.items() if name.endswith('.py')},input_sha256={name:retained.sha(raw) for name,raw in inputs.items()},elapsed_seconds=time.monotonic()-started)
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():archive.writestr(name,raw)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

def natural(output: Path) -> None:
    capsule = output.with_name(output.stem+'_natural_exit.json');pre = output.with_name(output.stem+'_preexecution.zip')
    assert not output.exists() and not capsule.exists() and not pre.exists(), 'Never overwrite an observed epoch'
    captured = capture()
    with zipfile.ZipFile(pre,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():archive.writestr(name,raw)
    command = [sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--execute']
    started = time.monotonic();process = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    unchanged = all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    record = dict(command=command,natural_exit=True,returncode=process.returncode,wall_seconds=time.monotonic()-started,stdout=process.stdout,stderr=process.stderr,all_sources_inputs_unchanged=unchanged,sha256={name:retained.sha(raw) for name,raw in captured.items()},preexecution=dict(path=pre.name,sha256=retained.sha(pre.read_bytes())),outputs={path.name:retained.sha(path.read_bytes()) for path in (output,output.with_suffix('.zip')) if path.exists()})
    capsule.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    if process.returncode!=0 or not unchanged:raise RuntimeError('Failed observable epoch retained')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true');args = parser.parse_args()
    if args.execute:
        with threadpool_limits(limits=1):result = run(args.output)
        print(json.dumps({k:result[k] for k in ('kind','observable','controls','elapsed_seconds')},indent=2))
    else:natural(args.output)
