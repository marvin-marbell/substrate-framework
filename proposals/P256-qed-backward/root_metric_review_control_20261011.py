"""Independent NEW exact-identity and float-price review; never execute suppliers."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from fractions import Fraction as Q
from pathlib import Path
import subprocess
import time
import zipfile
from typing import Any
import numpy as np
import sympy as s

ROOT = Path(__file__).resolve().parent
STEM = 'root_metric_review_actual1_20261011'
INPUTS = [
    'normalized_root_graph_actual3_corrected_20261011_receipt.json',
    'connected_root_exterior_actual2_corrected_20261011_receipt.json',
    'connected_root_exterior_mixed_actual2_corrected_20261011_receipt.json',
    'connected_root_exterior_joint_metric_actual1_20261011_receipt.json',
    'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json',
    'local_so5_ground_upper_complete_state_consumer_20261011.py',
    'connected_root_exterior_graph_interface_corrected_20261011.json',
    'root_metric_review_binding_20261011.json',
    'root_metric_review_language_20261011.json',
]

def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def save(path: Path, value: Any) -> None:
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')

def car(mode: int, modes: int) -> np.ndarray:
    a = np.zeros((1 << modes, 1 << modes), dtype=np.int64)
    for mask in range(1 << modes):
        if mask & (1 << mode):
            a[mask ^ (1 << mode), mask] = (-1)**((mask & ((1 << mode)-1)).bit_count())
    return a

def exact_controls() -> dict[str, Any]:
    # Integer matrices: CAR/contact rejection is exact, not thresholded.
    aa = [car(p, 4) for p in range(4)]
    eye = np.eye(16, dtype=np.int64)
    for p, q in itertools.product(range(4), repeat=2):
        assert np.array_equal(aa[p]@aa[q].T+aa[q].T@aa[p], int(p == q)*eye)
    unsigned = [abs(a) for a in aa]
    assert np.any(unsigned[0]@unsigned[1].T+unsigned[1].T@unsigned[0])
    t0, t1 = aa[0].T@aa[2], aa[1].T@aa[3]
    assert np.array_equal(t0@t1, t1@t0)
    total = t0+t1
    coherent = total@total
    dropped = t0@t0+t1@t1
    assert np.any(coherent-dropped)
    assert np.array_equal(coherent, 2*(t0@t1))
    contact = aa[2].T@aa[0]@aa[0].T@aa[2]
    wrong_contact = aa[2].T@aa[0].T@aa[0]@aa[2]
    assert np.any(contact-wrong_contact)
    # All 12870 number8 occupations; every spin-labelled endpoint transition.
    masks = [sum(1 << p for p in z) for z in itertools.combinations(range(16), 8)]
    checked = 0
    wrong_offset_witness = None
    for mask in masks:
        n = [(mask >> (4*j) & 15).bit_count() for j in range(4)]
        charges = [v-2 for v in n]
        E = [-sum(charges[:j+1]) for j in range(3)]
        q = n[0]+n[1]-4
        r0, r1 = Q(n[0]-n[1], 2), Q(n[2]-n[3], 2)
        assert E == [-r0-Q(q,2), -q, -r1-Q(q,2)]
        if q == 2 and r1 != 0 and wrong_offset_witness is None:
            wrong_offset_witness = {'mask': mask, 'q': q, 'correct_right': str(-r1-Q(q,2)),
                                    'wrong_right': str(-r1+Q(q,2))}
        for p, z in itertools.product(range(4, 8), range(8, 12)):
            if mask & (1 << z) and not mask & (1 << p):
                changed = mask ^ (1 << p) ^ (1 << z)
                nn = [(changed >> (4*j) & 15).bit_count() for j in range(4)]
                ee = [-sum(v-2 for v in nn[:j+1]) for j in range(3)]
                assert ee[0] == E[0] and ee[2] == E[2] and ee[1] == E[1]-1
                checked += 1
    assert wrong_offset_witness is not None
    # Whole source-ground P is not the rho-zero subspace.
    omega = s.Matrix([1, 0, 0])
    rho = s.diag(0, 0, 1)
    P = omega*omega.T
    P_rho0 = s.diag(1, 1, 0)
    returned = s.Matrix([1, 2, 0])
    correct_return = (s.eye(3)-P)*returned
    wrong_return = (s.eye(3)-P_rho0)*returned
    assert rho*correct_return == s.zeros(3, 1)
    assert correct_return.norm() == 2 and wrong_return.norm() == 0
    # Exact symbolic log coefficient; complex independent tags keep exchange fold.
    z, zb, w, wb, A, B, D, Dc, G = s.symbols('z zb w wb A B D Dc G')
    f = A*z*zb+B*w*wb+D*zb*w+Dc*z*wb+G*z*zb*w*wb
    series = s.Poly(s.expand(f-f*f/2), z, zb, w, wb)
    coefficient = series.coeff_monomial(z*zb*w*wb)
    assert s.expand(coefficient-(G-A*B-D*Dc)) == 0
    assert s.expand(coefficient.subs({B:A,D:A,Dc:A})-(G-2*A*A)) == 0
    a, b = s.symbols('a b', real=True)
    metric = 1+A*a*a+B*b*b+2*D*a*b+G*a*a*b*b
    Z = 1+A+B+2*D+G
    formula = (Z*(2*D+4*G)-4*(A+D+G)*(B+D+G))/Z**2
    assert s.simplify(s.diff(s.log(metric), a, b).subs({a:1,b:1})-formula) == 0
    assert (1,63,4*63,4*math.comb(63,2)) == (1,63,252,7812)
    assert (1,63,63,math.comb(63,2)) == (1,63,63,1953)
    return {'integer_CAR_anticommutators':16, 'unsigned_rejected':True,
            'coherent_history_squared_norm':int(np.sum(coherent*coherent)),
            'dropped_history_squared_norm':int(np.sum(dropped*dropped)),
            'contact_dropping_rejected':True, 'occupation_count':len(masks),
            'all4spin_forward_transitions':checked, 'wrong_offset_witness':wrong_offset_witness,
            'whole_ground_Q_return_norm':2, 'rho_proxy_return_norm':0,
            'symbolic_mixed_log_coefficient':str(coefficient),
            'repeated_fold_G_minus_2A_squared':True,'symbolic_joint_response':True,
            'scope':'Exact toy projector/CAR and exhaustive integer kinematics; not actual harmonic neutral image or all64 numerical source proof.'}

def numerical_controls() -> dict[str, Any]:
    r, same, mixed, joint, cap = [json.loads((ROOT/name).read_bytes()) for name in INPUTS[:5]]
    first, second = r['first_candidate'], r['second_candidate']
    t, u, W2, Z = [second[k] for k in ['t_all_roles','u_all_roles','W2','Z']]
    W = first['W']
    K = Q(first['K_V_complete_all64_sharedfield_exact'])
    eta = Q(cap['normalized_exact_source_residual_upper_exact'])
    formation = Q(cap['coefficient_and_action_error_upper_exact'])
    observation = second['charge2_source_interface']['source_residual_observation_not_recertification']
    assert eta == Q(1,65536) and formation == Q(1,262144)
    assert float(eta) > observation+float(formation) > observation
    delta_observed = 2*observation*float(K*K)
    delta_cap_exact = 2*eta*K*K
    raw = u*u*second['complete_H0_charge2_action_moments']['relative_second']+2*u*(u*t*W-t)*second['complete_H0_charge2_action_moments']['relative_first']+(u*t*W-t)**2*W2
    assert abs(raw-second['charge2_Riccati_residual_square_exact_identity_float']) < 1e-10
    paid_observation = (math.sqrt(raw)+abs(u)*delta_observed)**2
    cap_substituted = (math.sqrt(raw)+abs(u)*float(delta_cap_exact))**2
    source_cap = 2*float(eta)*(1+t*float(K))/math.sqrt(Z)
    full_cap_substituted = math.sqrt((cap_substituted+t*t*second['neutral_QVB_norm_square_upper_after_deducting_P_and_charge2']+(second['odd_charge1_and_charge3_residual_norm_upper']*math.sqrt(Z))**2)/Z)+source_cap
    assert cap_substituted > paid_observation
    assert full_cap_substituted > second['full_HU_minus_Ulambda_moment_paid_norm_upper_float']
    infidelity = Q(cap['selected_ground_infidelity_upper_exact'])
    distance = math.sqrt(2-2*math.sqrt(1-float(infidelity)))
    vector_upper_stable = math.sqrt(2*float(infidelity)/(1+math.sqrt(1-float(infidelity))))
    assert vector_upper_stable > 10000*float(infidelity)
    # Two local source tensors: <=2d by telescoping, before operator amplification.
    out = {'observed_eta':observation,'accepted_normalized_eta_exact':str(eta),
           'accepted_formation_exact':str(formation),'eta_ratio':float(eta)/observation,
           'deltaC_observed_FLOAT':delta_observed,'deltaC_cap_exact_given_exact_K':str(delta_cap_exact),
           'deltaC_cap_FLOAT':float(delta_cap_exact),'raw_charge2_square_FLOAT':raw,
           'observed_price_square_FLOAT':paid_observation,'certified_eta_substitution_square_FLOAT_NOT_INTERVAL':cap_substituted,
           'certified_eta_substitution_whole_norm_FLOAT_NOT_INTERVAL':full_cap_substituted,
           'cap_substitution_energy_price_FLOAT_NOT_INTERVAL':u*u*math.sqrt(W2)*float(delta_cap_exact)/Z,
           'infidelity_SQUARED_exact':str(infidelity),'vector_distance_upper_FLOAT':vector_upper_stable,
           'paired_vector_distance_upper_FLOAT':2*vector_upper_stable,
           'square_not_vector_rejected':True,'formation_not_double_added_to_accepted_eta':True,
           'remaining_interval_debt':'M21/M22 contractions, spin rotations, t/u formation and residual cancellations are not outward enclosed. Cap substitution alone is NOT a rigorous numerical bound.'}
    row = mixed['results'][0]
    Wa, Wb = row['physical_first_source_weights']
    D = complex(*row['actual_cross_path_Gram_D'])
    Gab = row['complete_all64_relative_G_T_U_and_selected_V']['G']
    assert abs(D.real-59.50627108787278) < 1e-10 and abs(D)>0
    assert abs((Gab-Wa*Wb-abs(D)**2)-(Gab-Wa*Wb)) > 3000
    metrics = {}
    for name in ['same_edge_physical1','unequal_shared_paths_physical1']:
        x = joint[name]
        A,B,d,G = [x[k] for k in ['G1_a','G1_b','actual_coherent_overlap','G21']]
        z = 1+A+B+2*d+G
        response = (z*(2*d+4*G)-4*(A+d+G)*(B+d+G))/z**2
        assert abs(z-x['joint_physical_normalizer']) < 1e-15
        assert abs(response-x['physical_connected_log_tag_response']) < 1e-15
        wrong = (1+A)*(1+B)
        assert abs(z-wrong)>1e-4
        metrics[name] = {'Z_FLOAT':z,'independent_Z_wrong_FLOAT':wrong,'response_FLOAT':response,
                         'degree22_FLOAT':G-A*B-d*d,'source_overlap_FLOAT':d}
    assert math.e*27*abs(metrics['same_edge_physical1']['response_FLOAT'])>8
    out['joint_metrics_FLOAT'] = metrics
    out['mixed_raw_log_coefficient_FLOAT'] = Gab-Wa*Wb-abs(D)**2
    out['mixed_drop_D_wrong_coefficient_FLOAT'] = Gab-Wa*Wb
    out['actual_D_FLOAT'] = [D.real,D.imag]
    out['chosen_constant_envelope_load_FLOAT'] = math.e*27*abs(metrics['same_edge_physical1']['response_FLOAT'])
    out['repeated_multiplicity_moment_reduction'] = []
    for path in same['paths']:
        rows=path['same_copy_and_distinct_copy_kernels']
        mult=path['all64_exact_multiplicities']
        assert mult == {'same_active':1,'same_auxiliary':63,'different_active_auxiliary':252,'different_auxiliary_auxiliary':7812}
        for key in ['G','T_relative','U_relative_complete_H0']:
            value=sum(mult[n]*rows[n][key] for n in mult)
            assert math.isclose(value,path['two_root_all64_complete_H0'][key],rel_tol=1e-14)
        out['repeated_multiplicity_moment_reduction'].append({'length':path['length'],'matches_FLOAT':True})
    return out

def worker() -> None:
    start=time.monotonic()
    result={'kind':'INDEPENDENT_NEW_ROOT_METRIC_ADVERSE_AND_CERTIFIED_CAP_SEMANTICS_CONTROL',
            'exact_identities':exact_controls(),'FLOAT_diagnostics':numerical_controls(),
            'input_sha256':{n:digest((ROOT/n).read_bytes()) for n in INPUTS},
            'own_source_sha256':digest(Path(__file__).read_bytes()),
            'source_targets_executed':False,'all_assertions_reached':True,
            'elapsed_seconds':time.monotonic()-start,
            'no_physics_admission':['physicalPhi','globalground','allorder','current','phase','pole','commoncone','originalSOURCE']}
    save(ROOT/(STEM+'_receipt.json'),result)
    print(json.dumps(result['FLOAT_diagnostics'],indent=2),flush=True)

def outer() -> None:
    captures={n:(ROOT/n).read_bytes() for n in INPUTS+[Path(__file__).name]}
    binding=json.loads(captures['root_metric_review_binding_20261011.json'])
    # Reviewed immutable supplier source bytes included but never imported/executed.
    for epoch, row in binding['epochs'].items():
        with zipfile.ZipFile(ROOT/(epoch+'_executed.zip')) as supplier_archive:
            raw=supplier_archive.read(row['source'])
        assert digest(raw)==row['source_sha256']
        captures[row['source']]=raw
    def archive(path:Path, members:dict[str,bytes]) -> None:
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n,raw in members.items():z.writestr(n,raw)
        path.chmod(0o444)
    pre=ROOT/(STEM+'_preexecution.zip')
    archive(pre,captures)
    # Unique source/input archive precedes the independent worker execution.
    start=time.monotonic()
    run=subprocess.run(['/home/administrator/substrate-framework/.venv/bin/python','-B',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,capture_output=True,env=os.environ|{'PYTHONDONTWRITEBYTECODE':'1'})
    outputs={STEM+'_stdout.txt':run.stdout,STEM+'_stderr.txt':run.stderr}
    for n,raw in outputs.items():
        with (ROOT/n).open('xb') as f:f.write(raw)
    receipt=ROOT/(STEM+'_receipt.json')
    if receipt.exists(): outputs[receipt.name]=receipt.read_bytes()
    exit_record={'natural_returncode':run.returncode,'natural_exit':run.returncode>=0,
                 'elapsed_seconds':time.monotonic()-start,'preexecution_archive_sha256':digest(pre.read_bytes()),
                 'captured_sha256':{n:digest(v) for n,v in captures.items()},
                 'output_sha256':{n:digest(v) for n,v in outputs.items()},
                 'inputs_unchanged':all((ROOT/n).read_bytes()==raw for n,raw in captures.items()),
                 'only_NEW_review_worker_executed':True}
    save(ROOT/(STEM+'_exit.json'),exit_record)
    outputs[STEM+'_exit.json']=(ROOT/(STEM+'_exit.json')).read_bytes()
    archive(ROOT/(STEM+'_executed.zip'),captures|outputs)
    print(json.dumps(exit_record,indent=2),flush=True)
    raise SystemExit(run.returncode)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    worker() if parser.parse_args().worker else outer()
