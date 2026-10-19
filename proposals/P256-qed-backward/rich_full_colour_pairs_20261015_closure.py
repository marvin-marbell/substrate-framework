"""Exact pre-pairprojection original-law coefficient and physical-source witness.

The frozen actual2 law contains tau_x mirror coupling, NOT a Y-vector kinetic
term. The mirror frame Ci algebra is not present in the Hamiltonian. This
consumer records that distinction, coefficient closure at finite EM angle,
and complete original-CAR source equality without a pair cut.
"""
from itertools import product
from math import comb
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
import numpy as np
from rich_full_colour_pairs_20261015_model import rotate_to_original
from rich_colour_sea_20261014_model import mode,string
from rich_colour_composite_20261014_model import loop_field,epsilon


def add_string(states,operations,coefficient,target):
    for mask,value in states.items():
        step = string(mask,operations)
        if step is not None:target[step[0]] = target.get(step[0],0)+coefficient*value*step[1]


def physical_source(states,spin):
    amplitudes = {(0,0,1):np.sqrt(2/3),(0,1,0):-np.sqrt(1/6),(1,0,0):-np.sqrt(1/6)} if spin == .5 else {(0,0,1):1/np.sqrt(3),(0,1,0):1/np.sqrt(3),(1,0,0):1/np.sqrt(3)}
    result = {}
    for colours in product(range(3),repeat=3):
        sign = epsilon(*colours)
        if not sign:continue
        for spins,amplitude in amplitudes.items():
            operations = [(mode(f,0,s,c),True) for f,s,c in zip((0,0,1),spins,colours)]
            add_string(states,operations,sign*amplitude/np.sqrt(6),result)
    return {mask:value for mask,value in result.items() if abs(value)>1e-15}


def witness(root):
    p = json.loads((root/'rich_full_colour_pairs_20261015_actual1_input.json').read_text());loop = loop_field()
    o = np.array([[1,1],[1,-1]])/np.sqrt(2);rotation = np.kron(o,np.eye(57));tau = np.diag([1,-1]);number = np.kron(tau,np.eye(57))
    coefficients = {};arrays = {};g = p['scalar_coupling'];r = p['loop_route']
    u = np.block([[loop['U'][a,b] for b in range(3)] for a in range(3)])
    for q in [2/3,-1/3]:
        for angle in [0.,.8,-.8,1.37]:
            phase = np.exp(1j*q*angle);w = phase*u+phase.conjugate()*u.conj().T
            dw = 1j*q*(phase*u-phase.conjugate()*u.conj().T);ddw = -q*q*w
            # These are precisely forward/backward and t+t^dagger in the
            # frozen original-mirror raw law, BEFORE any pair projection.
            t = g*np.eye(57)+g*r*w/2;dt = g*r*dw/2;ddt = g*r*ddw/2
            z = np.zeros((57,57),complex)
            original = dict(H=np.block([[z,t],[t.conj().T,z]]),J=-np.block([[z,dt],[dt.conj().T,z]]),K=np.block([[z,ddt],[ddt.conj().T,z]]))
            expected = dict(H=np.kron(tau,t),J=-np.kron(tau,dt),K=np.kron(tau,ddt))
            key=f'q{q}_A{angle}';coefficients[key] = dict(W_hermiticity=float(np.linalg.norm(w-w.conj().T)))
            for name,matrix in original.items():
                rotated = rotation.conj().T @ matrix @ rotation
                coefficients[key][name] = dict(exact_rotation_error=float(np.linalg.norm(rotated-expected[name])),rotated_mirror_commutator=float(np.linalg.norm(rotated @ number-number @ rotated)),original_tau_x_commutator=float(np.linalg.norm(matrix @ np.kron(np.array([[0,1],[1,0]]),np.eye(57))-np.kron(np.array([[0,1],[1,0]]),np.eye(57)) @ matrix)))
                arrays[key+'_'+name+'_original'] = matrix;arrays[key+'_'+name+'_rotated'] = rotated
    sea = sum(1 << mode(f,1,s,c) for f,s,c in product(range(2),range(2),range(3)))
    original_reference = rotate_to_original(sea);source_results = {}
    pair_probabilities = np.zeros(13)
    for mask,value in original_reference.items():
        positive = sum(bool(mask & (1 << mode(f,0,s,c))) for f,s,c in product(range(2),range(2),range(3)))
        pair_probabilities[positive] += abs(value)**2
    for spin in [.5,1.5]:
        actual = physical_source(original_reference,spin)
        rotated = {mask:value/np.sqrt(8) for mask,value in physical_source({sea:1.},spin).items()}
        predicted = {}
        for mask,value in rotated.items():
            for original,coefficient in rotate_to_original(mask).items():predicted[original] = predicted.get(original,0)+value*coefficient
        keys = sorted(set(actual)|set(predicted));aa = np.array([actual.get(mask,0) for mask in keys]);bb = np.array([predicted.get(mask,0) for mask in keys])
        source_results[str(spin)] = dict(full_original_CAR_source_equality_error=float(np.linalg.norm(aa-bb)),raw_norm_squared=float(np.vdot(aa,aa).real),original_mask_count=len(keys),rotated_mask_count=len(rotated),omitted_source_norm_squared=float(np.linalg.norm(aa-bb)**2))
        arrays['source_'+str(spin)+'_original_masks'] = np.array(keys,np.uint32);arrays['source_'+str(spin)+'_actual'] = aa;arrays['source_'+str(spin)+'_predicted'] = bb
    result = dict(kind='PRE_PAIRPROJECTION_FROZEN_ACTUAL2_OPERATOR_AND_FULL_ORIGINAL_CAR_SOURCE_WITNESS',same_law=True,law_changed=False,scalar_only_control=False,coefficient_witnesses=coefficients,source_witnesses=source_results,reference_original_pair_probabilities=pair_probabilities.tolist(),reference_pair_distribution_binomial_error=float(np.linalg.norm(pair_probabilities-np.array([comb(12,k)/4096 for k in range(13)]))),interpretation='The full original actual2 Hamiltonian already commutes with total mirror tau_x and each rotated mirror/flavour/spin population. Its original-mirror paircut fluctuations are basis/variational-projection correlations, NOT physical interband pair creation. Exact full-pair source/current closure is valid but is not rich Dirac pair feedback. Ordered Ymirror tensor Fi vectors appear only in ordered_frame metadata, not in actual2 law(). No Yvector was changed to X to gain solvability.',unpaid='A genuinely noncommuting vector/Dirac kinetic and its pair-connected current would be a DIFFERENT model and remains unpaid; not supplied or implied here.')
    path = root/'rich_full_colour_pairs_20261015_closure1_result.json';path.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(path.with_suffix('.npz'),**arrays)
    print(json.dumps(dict(same_law=True,source_witnesses=source_results,largest_rotation_error=max(x[name]['exact_rotation_error'] for x in coefficients.values() for name in ['H','J','K']),largest_mirror_commutator=max(x[name]['rotated_mirror_commutator'] for x in coefficients.values() for name in ['H','J','K']),pair_distribution_error=result['reference_pair_distribution_binomial_error'])))


def capture(root):
    prefix='rich_full_colour_pairs_20261015_closure1'
    names=['rich_full_colour_pairs_20261015_closure.py','rich_full_colour_pairs_20261015_model.py','rich_full_colour_pairs_20261015_calculate.py','rich_full_colour_pairs_20261015_equations.json','rich_full_colour_pairs_20261015_actual1_input.json','rich_colour_sea_20261014_model.py','rich_colour_composite_20261014_model.py']
    frozen={name:(root/name).read_bytes() for name in names};before={name:hashlib.sha256(value).hexdigest() for name,value in frozen.items()}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');command=[sys.executable,str(root/'rich_full_colour_pairs_20261015_closure.py')]
    started=time.time();process=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
    stdout=root/(prefix+'_stdout.txt');stderr=root/(prefix+'_stderr.txt');stdout.write_text(process.stdout);stderr.write_text(process.stderr)
    outputs=[stdout,stderr]+[root/(prefix+'_result'+suffix) for suffix in ['.json','.npz'] if (root/(prefix+'_result'+suffix)).exists()]
    after={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    receipt=dict(command=command,returncode=process.returncode,natural_exit=True,source_unchanged=before==after,source_before=before,source_after=after,elapsed_seconds=time.time()-started,outputs={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs})
    exitpath=root/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n');capsule=root/(prefix+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in frozen.items():archive.writestr(name,value)
        for path in outputs+[exitpath]:archive.write(path,path.name)
    print(process.stdout);print(process.stderr,file=sys.stderr);print(json.dumps(dict(returncode=process.returncode,capsule_sha256=hashlib.sha256(capsule.read_bytes()).hexdigest())))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    root=Path(__file__).resolve().parent
    if '--capture' in sys.argv:capture(root)
    else:witness(root)
