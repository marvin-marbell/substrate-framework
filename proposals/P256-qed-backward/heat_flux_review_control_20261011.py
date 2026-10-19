"""New frozen mathematical consumer. Never invokes historical scientific CLIs."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from fractions import Fraction as F
from typing import cast

ROOT = Path(__file__).resolve().parent
TARGETS = [
    'charged_cell_heat_actual1_20261011_preexecution.zip',
    'charged_cell_heat_actual1_20261011_executed.zip',
    'charged_cell_heat_actual1_20261011_capsule.json',
    'charged_cell_heat_actual1_20261011_exit.json',
    'charged_cell_heat_actual1_20261011_receipt.json',
    'charged_cell_heat_interface_20261011.json',
    'charged_cell_heat_proof_20261011.json',
    'charged_cell_heat_helper_20261011.py',
    'charged_cell_heat_scientist_20261011.py',
    'flux_laplace_actual1_20261011_preexecution.zip',
    'flux_laplace_actual1_20261011_executed.zip',
    'flux_laplace_actual1_20261011_exit.json',
    'flux_laplace_actual1_20261011_receipt.json',
    'flux_laplace_proof_20261011.json',
    'flux_laplace_scientist_20261011.py',
    'flux_laplace_delivery_20261011.json',
    'flux_laplace_parent_error_decomposition_20261011.json',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def frozen_modules():
    # Zipimport executes declarations only; no original run/main/control function.
    archive = ROOT/'flux_laplace_actual1_20261011_preexecution.zip'
    sys.path.insert(0, str(archive))
    return tuple(importlib.import_module(name) for name in (
        'charged_cell_heat_helper_20261011', 'flux_laplace_scientist_20261011'))


def independent_exp(x):
    """Independent 220-bit rational alternating bracket with outward squaring."""
    x = F(x)
    assert x >= 0
    reductions = 0
    while x > F(1, 16):
        x /= 2
        reductions += 1
    term = total = F(1)
    for n in range(1, 81):
        term *= -x/n
        total += term
    upper = total
    lower = total-term*x/81
    scale = 1 << 220
    def rounded(a, b):
        lo = (a.numerator*scale)//a.denominator
        hi = -((-b.numerator*scale)//b.denominator)
        return F(lo, scale), F(hi, scale)
    lower, upper = rounded(lower, upper)
    for _ in range(reductions):
        lower, upper = rounded(lower*lower, upper*upper)
    return lower, upper


def matrix_heat_control(heat):
    import sympy as s
    K = s.diag(2, 7, 1000000)
    Y = s.Matrix([[s.Rational(1, 3), -s.I/4], [s.Rational(1, 5), s.Rational(1, 6)], [s.Rational(1, 1000000), s.I/1000000]])
    assert (s.eye(2)-Y.conjugate().T*Y).det() > 0
    def encode(z):
        re, im = s.expand_complex(z).as_real_imag()
        return {'re': [str(re)]*2, 'im': [str(im)]*2}
    moments = {name: [[encode(z) for z in row] for row in (Y.conjugate().T*K**k*Y).tolist()] for k, name in enumerate(('M0', 'M1', 'M2'))}
    interface = {'charged_floor_exact': '2', 'ground_distance_exact': ['0', '0'], 'roles': [
        {'role': 'control', 'orientation': orientation, 'moments': moments, 'ground_second_moment_trace_upper_exact': str(s.trace(Y.conjugate().T*K**2*Y))}
        for orientation in ('creation', 'annihilation')]}
    records = []
    for orientation in ('creation', 'annihilation'):
        for t in (F(0), F(1, 64), F(1, 8), F(1), F(4)):
            enclosed = heat.evaluate(interface, 'control', orientation, t)
            refs = [independent_exp(t*int(K[k, k])) for k in range(3)]
            for i, j in itertools.product(range(2), repeat=2):
                a, b = (i, j) if orientation == 'creation' else (j, i)
                for component in ('re', 'im'):
                    lo = hi = F(0)
                    for k in range(3):
                        z = s.expand_complex(s.conjugate(Y[k, a])*Y[k, b])
                        coefficient = F(str(s.re(z) if component == 're' else s.im(z)))
                        bounds = [coefficient*v for v in refs[k]]
                        lo += min(bounds)
                        hi += max(bounds)
                    actual = list(map(F, enclosed[i][j][component]))
                    assert actual[0] <= lo <= hi <= actual[1], (orientation, t, i, j, component)
            records.append({'orientation': orientation, 't_exact': str(t), 'all_complex_entries_enclosed': True})
    # Loewner residual has complex offdiagonal; entrywise nonnegative replacement fails.
    t = F(1, 8)
    a = F(5)
    residual_weights = []
    for x in (2, 7, 1000000):
        ex = independent_exp(t*x)
        ea = independent_exp(t*a)
        tangent = 1-t*(x-a)
        subtract = [tangent*v for v in ea]
        residual_weights.append((ex[0]-max(subtract), ex[1]-min(subtract)))
    imag_hi = F(0)
    for k in range(3):
        coefficient = F(str(s.im(s.conjugate(Y[k, 0])*Y[k, 1])))
        imag_hi += max(coefficient*v for v in residual_weights[k])
    assert imag_hi < 0
    return {'domain': 'exact rational contraction Y with positive diagonal K, including energy1000000; not canonical source replacement', 'enclosures': records, 'forbidden_entrywise_Loewner_sign_imaginary_upper_exact': str(imag_hi), 'complex_Gram_enclosure_checked': True, 'wrong_transpose_adverse_control_owned_by_parent': True}


def native_commutator_control():
    import sympy as s
    with zipfile.ZipFile(ROOT/'flux_laplace_actual1_20261011_preexecution.zip') as archive:
        profiles = json.loads(archive.read('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'))['profiles']
    ann = []
    for p in range(4):
        A = s.zeros(16)
        for mask in range(16):
            if (mask >> p) & 1:
                A[mask ^ (1 << p), mask] = (-1)**((mask & ((1 << p)-1)).bit_count())
        ann.append(A)
    A = s.diag(-1, 1, -1, 1)
    X = s.Matrix([[0, 1], [1, 0]])
    B = s.kronecker_product(X, s.diag(1, -1))
    C = s.zeros(2).row_join(s.I*X).col_join((-s.I*X).row_join(s.zeros(2)))
    r = s.diag(s.Rational(1, 2), s.Rational(1, 2), -s.Rational(1, 2), -s.Rational(1, 2))
    R = sum((ann[p].T*ann[q]*r[p, q] for p, q in itertools.product(range(4), repeat=2)), s.zeros(16))
    V = s.zeros(16, 5)
    for j, mask in enumerate((5, 3, 12, 10)):
        V[mask, j] = 1
    V[9, 4] = V[6, 4] = 1/s.sqrt(2)
    hole = s.zeros(16, 4)
    for p in range(4):
        hole[15 ^ (1 << p), p] = (-1)**p
    records = []
    for profile in profiles:
        h = cast(s.MutableDenseMatrix, sum((s.Rational(profile[key])*base for key, base in zip(('m_exact', 'c0_exact', 'd_exact'), (A, B, C))), s.zeros(4)))
        D = sum((ann[p].T*ann[q]*h[p, q] for p, q in itertools.product(range(4), repeat=2)), s.zeros(16))
        assert hole.T*D*hole == s.trace(h)*s.eye(4)-h.T
        assert hole.T*R*hole == -r.T
        for spectator in (0, 1, -1):
            totalR = R+spectator*s.eye(16)
            Hn = D+64*totalR**2
            for creation in (True, False):
                eps = 1 if creation else -1
                u = -s.Rational(eps, 2)
                Hc = D+64*(totalR+u*s.eye(16))**2
                for p in range(4):
                    cp = ann[p].T if creation else ann[p]
                    delta = eps*r[p, p]+u
                    change = sum(((cast(s.Expr,h[q, p]) if creation else -cast(s.Expr,h[p, q]))*(ann[q].T if creation else ann[q]) for q in range(4)), s.zeros(16))
                    assert Hc*cp-cp*Hn == change+64*cp*(2*delta*totalR+delta**2*s.eye(16))
                    if p < 2:
                        assert delta == 0
                        second_density = 64*((totalR+u*s.eye(16))**2*change-change*totalR**2)
                        assert second_density*V != s.zeros(16, 5)
                        cropped = sum(((cast(s.Expr,h[q, p]) if creation else -cast(s.Expr,h[p, q]))*(ann[q].T if creation else ann[q]) for q in range(2)), s.zeros(16))
                        assert (change-cropped)*V != s.zeros(16, 5)
        records.append({'role': profile['role'], 'all4_modes_both_signs_spectator0plusminus1': True, 'initial_delta0_verified': True, 'second_density_action_nonzero_on_native_slots': True, 'full_signed_number3_hole_D_R_verified': True, 'offsite_initial_action_crop_rejected': True})
    return records, ann, (A, B, C, r), profiles


def tensor_factorization_control(flux, ann_exact, bases, profiles):
    """Independent full 16^4 Fock propagation, actual native edge blocks, NEW toy g."""
    import numpy as np
    from scipy.linalg import expm
    import sympy as s
    source = json.loads((ROOT/'flux_laplace_actual1_20261011_receipt.json').read_text())
    ann = [np.array(a, dtype=complex) for a in ann_exact]
    masks = np.arange(1 << 16, dtype=np.uint32)
    g = np.zeros(16, complex)
    g[5] = 3/5
    g[10] = 4j/5
    initial = g
    for _ in range(3):
        initial = np.kron(initial, g)
    neutral = np.array([m for m in range(16) if m.bit_count() == 2])
    Hneutral = np.zeros((16, 16), complex)
    Hneutral[np.ix_(neutral, neutral)] = 3*(np.eye(6)-np.outer(g[neutral], g[neutral].conj()))
    spins = (0, 0, 1, 1)  # Both nonzero cross-spin axis2 blocks; not a factor16 scalar.
    tau = (F(1, 64), F(1, 128), F(3, 256))
    def global_op(vec, cell, mode, creation):
        # np.kron orders cell0 as the most significant nibble.
        bit = 4*(3-cell)+mode
        allowed = ((masks >> bit) & 1) == (0 if creation else 1)
        src = masks[allowed]
        # Global mode order is cell0,cell1,...; mode-bit order within each cell ascending.
        parity = np.zeros(src.shape, dtype=np.uint8)
        for v in range(cell):
            nibble = (src >> (4*(3-v))) & 15
            for p in range(4):
                parity ^= ((nibble >> p) & 1).astype(np.uint8)
        local = (src >> (4*(3-cell))) & ((1 << mode)-1)
        for p in range(mode):
            parity ^= ((local >> p) & 1).astype(np.uint8)
        out = np.zeros_like(vec)
        out[src ^ (1 << bit)] = vec[src]*(1-2*parity.astype(np.int8))
        return out
    def propagate(vec, local_exponentials):
        tensor = vec.reshape((16,)*4)
        for cell, mat in enumerate(local_exponentials):
            tensor = np.moveaxis(np.tensordot(mat, tensor, axes=(1, cell)), 0, cell)
        return tensor.reshape(-1)
    records = []
    wrong_equal_time_rejected = wrong_sign_rejected = 0
    max_difference = 0.0
    for role, profile in zip(('active', 'aux'), profiles):
        h = sum((s.Rational(profile[key])*base for key, base in zip(('m_exact', 'c0_exact', 'd_exact'), bases[:3])), s.zeros(4))
        h = np.array(h, dtype=complex)
        D = sum((ann[p].conj().T@ann[q]*h[p, q] for p, q in itertools.product(range(4), repeat=2)), np.zeros((16, 16), complex))
        R = sum((ann[p].conj().T@ann[p]*float(bases[3][p, p]) for p in range(4)), np.zeros((16, 16), complex))
        Hlocal = Hneutral.copy()
        for number, u in ((1, .5), (3, -.5)):
            sector = np.array([m for m in range(16) if m.bit_count() == number])
            charged = D+64*(R+u*np.eye(16))@(R+u*np.eye(16))+100*np.eye(16)
            Hlocal[np.ix_(sector, sector)] = charged[np.ix_(sector, sector)]
        expcache = {t: expm(-float(t)*Hlocal) for t in set(tau)|{sum(tau[i:j], F(0)) for i in range(3) for j in range(i+1, 4)}}
        native = {axis: np.array([[complex(s.sympify(z)) for z in row] for row in source['native_matrices_exact'][role][str(axis)]]) for axis in (1, 2)}
        forward = [native[1], native[2], native[1].conj().T, native[2].conj().T]
        for orientation in (-1, 1):
            edges = forward if orientation == 1 else [a.conj().T for a in forward]
            blocks = []
            for edge, M in enumerate(edges):
                a, b = (edge, (edge+1)%4) if orientation == 1 else ((edge+1)%4, edge)
                blocks.append(M[2*spins[a]:2*spins[a]+2, 2*spins[b]:2*spins[b]+2])
            for order in itertools.permutations(range(4)):
                vec = initial.copy()
                for stage, edge in enumerate(order):
                    a, b = (edge, (edge+1)%4) if orientation == 1 else ((edge+1)%4, edge)
                    hopped = np.zeros_like(vec)
                    for p, q in itertools.product(range(2), repeat=2):
                        hopped += blocks[edge][p, q]*global_op(global_op(vec, b, q, False), a, p, True)
                    vec = hopped
                    if stage < 3:
                        vec = propagate(vec, [expcache[tau[stage]]]*4)*np.exp(-64*(stage+1)*float(tau[stage]))
                direct = np.vdot(initial, vec)
                sign, counts, between, first = flux.order_data(order, orientation)
                heats = []
                zeros = []
                for v in range(4):
                    duration = sum((tau[j] for j in between[v]), F(0))
                    Y = np.column_stack([(ann[p].conj().T if first[v] else ann[p])@g for p in range(2)])
                    G = Y.conj().T@expcache[duration]@Y
                    G0 = Y.conj().T@Y
                    heats.append(G if first[v] else G.T)
                    zeros.append(G0 if first[v] else G0.T)
                def trace(kernels):
                    result = np.eye(2, dtype=complex)
                    for v in ((0, 1, 2, 3) if orientation == 1 else (0, 3, 2, 1)):
                        edge = v if orientation == 1 else (v-1)%4
                        result = result@kernels[v]@blocks[edge]
                    return np.trace(result)
                external = np.exp(-sum(64*(j+1)*float(t) for j, t in enumerate(tau)))
                factored = sign*trace(heats)*external
                difference = abs(direct-factored)
                max_difference = max(max_difference, float(difference))
                assert difference < 2e-11*max(1, abs(direct)), (role, orientation, order, direct, factored)
                wrong_equal = sign*trace(zeros)*external
                wrong_unsigned = trace(heats)*external
                wrong_equal_time_rejected += abs(direct-wrong_equal) > 1e-7
                wrong_sign_rejected += abs(direct-wrong_unsigned) > 1e-7
                records.append({'role': role, 'orientation': orientation, 'order': list(order), 'direct_fullFock_re': float(direct.real), 'direct_fullFock_im': float(direct.imag), 'factorization_difference': float(difference), 'charged_counts': counts})
    assert wrong_equal_time_rejected and wrong_sign_rejected
    return {'domain': 'NEW adversarial finite 16^4 CAR model; native actual role h and hopping blocks; artificial exact neutral g=(3|5>+4i|10>)/5 and positive charged shift100, NOT canonical g/H', 'controls_are_numerical_not_machine_certificate': True, 'tau_exact': list(map(str, tau)), 'spin_assignment': list(spins), 'all96_role_order_orientation_records': records, 'maximum_difference': max_difference, 'equal_time_substitution_rejected_cases': int(wrong_equal_time_rejected), 'omitted_global_sign_rejected_cases': int(wrong_sign_rejected)}


def coherence_control(flux):
    import sympy as s
    source = json.loads((ROOT/'flux_laplace_actual1_20261011_receipt.json').read_text())
    H = [s.Matrix([[s.Rational(v+2, 11), s.I*s.Rational(v+1, 17)], [-s.I*s.Rational(v+1, 17), s.Rational(v+3, 13)]]) for v in range(4)]
    GH = [s.diag(h, h) for h in H]
    rows = []
    for role in ('active', 'aux'):
        native = {axis: s.Matrix([[s.sympify(z) for z in row] for row in source['native_matrices_exact'][role][str(axis)]]) for axis in (1, 2)}
        forward = [native[1], native[2], native[1].conjugate().T, native[2].conjugate().T]
        for orientation in (-1, 1):
            edges = forward if orientation == 1 else [a.conjugate().T for a in forward]
            vertices = (0, 1, 2, 3) if orientation == 1 else (0, 3, 2, 1)
            total = 0
            for spins in itertools.product(range(2), repeat=4):
                product = s.eye(2)
                for v in vertices:
                    edge = v if orientation == 1 else (v-1)%4
                    a, b = (edge, (edge+1)%4) if orientation == 1 else ((edge+1)%4, edge)
                    product = product*H[v]*edges[edge][2*spins[a]:2*spins[a]+2, 2*spins[b]:2*spins[b]+2]
                total += s.trace(product)
            full = s.eye(4)
            for v in vertices:
                edge = v if orientation == 1 else (v-1)%4
                full = full*GH[v]*edges[edge]
            assert s.expand(total-s.trace(full)) == 0
            # Invoke the FROZEN implementation's interval trace, not a reconstructed substitute.
            target = flux.CZ
            for spins in itertools.product(range(2), repeat=4):
                blocks = []
                for edge, M in enumerate(edges):
                    a, b = (edge, (edge+1)%4) if orientation == 1 else ((edge+1)%4, edge)
                    blocks.append([[flux.cx.ci(M[2*spins[a]+p, 2*spins[b]+q]) for q in range(2)] for p in range(2)])
                target = flux.cx.ca(target, flux.contraction([[[flux.cx.ci(z) for z in row] for row in h.tolist()] for h in H], blocks, orientation))
            re, im = s.expand_complex(s.trace(full)).as_real_imag()
            for actual, value in zip(target, (re, im)):
                assert F(actual[0], flux.iv.Q) <= F(str(value)) <= F(actual[1], flux.iv.Q)
            wrong = 16*s.trace(H[0]*edges[0][:2, :2]*H[1]*edges[1][:2, :2]*H[2]*edges[2][:2, :2]*H[3]*edges[3][:2, :2])
            assert s.expand(total-wrong) != 0
            rows.append({'role': role, 'orientation': orientation, 'exact16_blocks_equal_full4_trace': True, 'frozen_interval_contraction_contains_exact_trace': True, 'factor16_scalar_shortcut_rejected': True, 'exact_full_trace': str(s.expand(total))})
    # Four insertions reach the neutral scalar lost by a rank5 crop. For TWO
    # insertions in the flux, endpoint stationarity makes neutral evolution cancel.
    # This witness therefore rejects GENERAL cropping, not that endpoint identity.
    ann = []
    for p in range(2):
        A = s.zeros(16)
        for mask in range(16):
            if (mask >> p) & 1:
                A[mask ^ (1 << p), mask] = (-1)**((mask & ((1 << p)-1)).bit_count())
        ann.append(A)
    state = s.eye(16)[:, 5]
    scalar = (s.eye(16)[:, 6]-s.eye(16)[:, 9])/s.sqrt(2)
    crop = s.eye(16)-scalar*scalar.T
    reached = ann[1].T*ann[0]*state
    full_value = (state.T*ann[0].T*ann[1]*reached)[0]
    cropped_value = (state.T*ann[0].T*ann[1]*crop*reached)[0]
    assert full_value == 1 and cropped_value == s.Rational(1, 2)
    return {'exact_controls': rows, 'neutral_scalar_four_insertion_full_exact': '1', 'neutral_scalar_four_insertion_cropped_exact': '1/2', 'neutral_crop_scope': 'c0dagger*c1*P*c1dagger*c0 on |5>; P deletes (|6>-|9>)/sqrt2, a reachable scalar. Two-insertion neutral ENDPOINT cancellation remains valid.'}


def control():
    started = time.monotonic()
    pins = {name: sha(ROOT/name) for name in TARGETS}
    assert pins['charged_cell_heat_interface_20261011.json'] == '41d524542ded29c183426646bbdecf2f34b613bee7dcd13bf98e2a427c9c330f'
    assert pins['flux_laplace_actual1_20261011_receipt.json'] == '6a1bbea3952c5e968968691d25c5f5bc08873623fbd5a113050e228a5ca45939'
    mismatches = []
    checked = 0
    for stem in ('charged_cell_heat', 'flux_laplace'):
        with zipfile.ZipFile(ROOT/f'{stem}_actual1_20261011_preexecution.zip') as archive:
            for name in archive.namelist():
                checked += 1
                if (ROOT/name).exists() and hashlib.sha256(archive.read(name)).hexdigest() != sha(ROOT/name):
                    mismatches.append(name)
    # Frozen target functions load ORIGINAL ZIP members, not evolved live sources.
    heat, flux = frozen_modules()
    matrix = matrix_heat_control(heat)
    commutator, ann, bases, profiles = native_commutator_control()
    tensor = tensor_factorization_control(flux, ann, bases, profiles)
    coherent = coherence_control(flux)
    assert pins == {name: sha(ROOT/name) for name in TARGETS}
    result = {'kind': 'INDEPENDENT_FROZEN_HEAT_FLUX_MATHEMATICAL_REVIEW_CONTROL', 'own_source_sha256': sha(Path(__file__)), 'input_sha256': pins, 'immutable_preexecution_members_compared': checked, 'source_capsule_mismatches': mismatches, 'original_CLIs_solvers_checkers_executed': False, 'target_pure_functions_called': ['charged_cell_heat_helper.evaluate', 'flux_laplace_scientist.order_data', 'flux_laplace_scientist.contraction'], 'matrix_functional_calculus': matrix, 'native_commutator': commutator, 'tensor_factorization': tensor, 'coherence_and_neutral_scope': coherent, 'elapsed_seconds': time.monotonic()-started}
    dump(ROOT/f'{PREFIX}_receipt.json', result)
    print(json.dumps({'kind': result['kind'], 'own_source_sha256': result['own_source_sha256'], 'all96_tensor_cases': len(tensor['all96_role_order_orientation_records']), 'max_tensor_difference': tensor['maximum_difference'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


def execute():
    request = {'kind': 'NEW_INDEPENDENT_MATHEMATICAL_REVIEW_CONTROL_REQUEST', 'source': Path(__file__).name, 'source_sha256': sha(Path(__file__)), 'inputs_sha256': {name: sha(ROOT/name) for name in TARGETS}, 'prohibitions': ['No original CLI/solver/root/checker', 'No heat/flux target edits', 'No phase claim'], 'command': [sys.executable, str(Path(__file__)), '--control', '--epoch', args.epoch]}
    dump(ROOT/f'{PREFIX}_request.json', request)
    pre = ROOT/f'{PREFIX}_preexecution.zip'
    with zipfile.ZipFile(pre, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in [Path(__file__).name, f'{PREFIX}_request.json', *TARGETS]:
            archive.write(ROOT/name, name)
    with (ROOT/f'{PREFIX}_stdout.txt').open('wb') as out, (ROOT/f'{PREFIX}_stderr.txt').open('wb') as err:
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        process = subprocess.Popen(request['command'], cwd=ROOT, stdout=out, stderr=err, env=env)
        returncode = process.wait()
    exitdata = {'kind': 'NEW_REVIEW_CONTROL_NATURAL_EXIT', 'pid': process.pid, 'returncode': returncode, 'natural_exit': True, 'preexecution_sha256': sha(pre), 'source_unchanged': sha(Path(__file__)) == request['source_sha256'], 'inputs_unchanged': request['inputs_sha256'] == {name: sha(ROOT/name) for name in TARGETS}}
    dump(ROOT/f'{PREFIX}_exit.json', exitdata)
    executed = ROOT/f'{PREFIX}_executed.zip'
    with zipfile.ZipFile(executed, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in [f'{PREFIX}_{suffix}' for suffix in ('request.json', 'stdout.txt', 'stderr.txt', 'exit.json', 'receipt.json')]:
            if (ROOT/name).exists():
                archive.write(ROOT/name, name)
        archive.write(Path(__file__), Path(__file__).name)
    dump(ROOT/f'{PREFIX}_capsule.json', {'preexecution_sha256': sha(pre), 'executed_sha256': sha(executed), 'exit_sha256': sha(ROOT/f'{PREFIX}_exit.json'), 'receipt_sha256': sha(ROOT/f'{PREFIX}_receipt.json') if (ROOT/f'{PREFIX}_receipt.json').exists() else None, 'returncode': returncode, 'natural_exit': True})
    print(json.dumps(exitdata), flush=True)
    return returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--epoch', required=True)
    args = parser.parse_args()
    PREFIX = 'heat_flux_review_' + args.epoch + '_20261011'
    if args.control:
        control()
    else:
        raise SystemExit(execute())
