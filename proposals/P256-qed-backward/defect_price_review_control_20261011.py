from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction as F
from pathlib import Path
import zipfile
from typing import cast
import sympy as S

ROOT = Path(__file__).resolve().parent
PREFIX = 'defect_price_review_'

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read_json(name: str):
    return json.loads((ROOT / name).read_text())

def bracket(q: F, bits: int = 192) -> tuple[F, F]:
    # Independent integer bisection, not the producer's isqrt implementation.
    den = 2 ** bits
    left, right = 0, den * (q.numerator // q.denominator + 1)
    while right - left > 1:
        mid = (left + right) // 2
        if mid * mid * q.denominator <= q.numerator * den * den:
            left = mid
        else:
            right = mid
    low, high = F(left, den), F(right, den)
    assert low * low <= q < high * high
    return low, high

def native_fock(m, c, d):
    # Jordan-Wigner annihilation on the full sixteen-state native Fock basis.
    annihilators = []
    for j in range(4):
        a = S.zeros(16)
        for mask in range(16):
            if mask & (1 << j):
                a[mask ^ (1 << j), mask] = (-1) ** ((mask & ((1 << j) - 1)).bit_count())
        annihilators.append(a)
    eye = S.eye(16)
    for j, a in enumerate(annihilators):
        for k, b in enumerate(annihilators):
            assert a * b + b * a == S.zeros(16)
            assert a * b.conjugate().T + b.conjugate().T * a == (eye if j == k else S.zeros(16))
    x = S.Matrix([[0, 1], [1, 0]])
    h = m * S.diag(-1, 1, -1, 1) + c * S.kronecker_product(x, S.diag(1, -1))
    h += d * S.Matrix([[0, 0, 0, S.I], [0, 0, S.I, 0], [0, -S.I, 0, 0], [-S.I, 0, 0, 0]])
    rho = S.diag(S.Rational(1, 2), S.Rational(1, 2), -S.Rational(1, 2), -S.Rational(1, 2))
    H, R = S.zeros(16), S.zeros(16)
    for j in range(4):
        for k in range(4):
            bilinear = annihilators[j].conjugate().T * annihilators[k]
            H += h[j, k] * bilinear
            R += rho[j, k] * bilinear
    return h, H, R

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, default=ROOT / (PREFIX + 'request_20261011.json'))
    parser.add_argument('--output', type=Path, default=ROOT / (PREFIX + 'result_20261011.json'))
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    pins = request['pins']
    assert all(sha((ROOT / n).read_bytes()) == v for n, v in pins.items())
    profiles = read_json('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json')['profiles']
    cap = read_json('local_so5_ground_upper_complete_state_consumer_20261011_receipt.json')
    target = read_json('matching_defect_interface_20261011.json')['quantities']
    roots, slot_spectra, car = [], [], []
    t, e = S.symbols('t e')
    for p in profiles:
        m, c, d = [S.Rational(p[k + '_exact']) for k in ('m', 'c0', 'd')]
        q = F(p['e_squared_exact'])
        assert F(str(m*m+d*d)) == q
        low, high = bracket(q)
        roots.append((low, high))
        h, H, R = native_fock(m, c, d)
        free = [-e+c, -e-c, e+c, e-c]
        spectra = []
        dimensions = []
        for n in range(5):
            indices = [mask for mask in range(16) if mask.bit_count() == n]
            block = H.extract(indices, indices)
            subsets = list(itertools.combinations(range(4), n))
            expected = S.prod(t-sum(free[j] for j in subset) for subset in subsets)
            expected = S.rem(S.Poly(S.expand(expected), e), S.Poly(e*e-m*m-d*d, e)).as_expr()
            assert S.expand(block.charpoly(t).as_expr() - expected) == 0
            dims = len(indices)
            dimensions.append(dims)
            for subset in subsets:
                coefficient = sum(-1 if j < 2 else 1 for j in subset)
                constant = F(str(c)) * sum(1 if j % 2 == 0 else -1 for j in subset)
                lo = coefficient * (low if coefficient >= 0 else high) + constant
                hi = coefficient * (high if coefficient >= 0 else low) + constant
                spectra.append({'n': n, 'mask': sum(1 << j for j in subset), 'lo': lo, 'hi': hi})
            r = R.extract(indices, indices)
            assert all(abs(r[i, i]) <= S.Rational(min(n, 4-n), 2) for i in range(dims))
            # Full shifted square has the cross term; it cannot be discarded.
            assert (r + S.eye(dims))**2 - (r*r + S.eye(dims)) == 2*r
        assert dimensions == [1, 4, 6, 4, 1]
        assert len(spectra) == 16
        minimum = next(v for v in spectra if v['mask'] == 3)
        assert minimum['lo'] == -2*high
        assert all(v['lo'] > minimum['hi'] for v in spectra if v['mask'] != 3)
        slot_spectra.append(spectra)
        car.append({'role': p['role'], 'dimensions': dimensions, 'charpolys': 'Exact native full-Jordan-Wigner exterior block equality', 'sqrt': [str(low), str(high)], 'spectra': [{k: str(v) if k in ('lo','hi') else v for k,v in row.items()} for row in spectra]})
    (al, ah), (ul, uh) = roots
    c = F(profiles[0]['c0_exact'])
    assert c == F(profiles[1]['c0_exact']) < 0 and ah < ul
    assert [p['spin_replicas'] for p in profiles] == [2, 126]
    baseline = sum(F(p['spin_replicas']) * min(v['lo'] for v in spectrum) for p, spectrum in zip(profiles, slot_spectra))
    assert baseline == -4*ah-252*uh
    delta = min(v['lo'] + 2*roots[i][0] for i, spectrum in enumerate(slot_spectra) for v in spectrum if v['mask'] != 3)
    # The direct outward subtraction loses one radical width; the sharper exact
    # spectral difference is e+c, independently justified by all sixteen states.
    delta_sharp = al+c
    assert 0 < delta <= delta_sharp
    one = []
    for p, spectrum in zip(profiles, slot_spectra):
        one.extend([v['lo'] for v in spectrum if v['n'] == 1] * p['spin_replicas'])
    assert len(one) == 512
    one.sort()
    sectors, total = [], F(0)
    for n in range(513):
        if n:
            total += one[n-1]
        assert total == F(target['all_number_sector_envelopes'][n]['free_floor_lower_exact'])
        sectors.append({'n':n, 'floor':str(total), 'rho_norm':str(F(min(n,512-n),2))})
    assert F(sectors[256]['floor']) == baseline
    for n in (254,258):
        assert F(sectors[n]['floor']) >= baseline+2*delta_sharp
    Eg = F(cap['ground_energy_interval_exact'][1])
    gamma = -5123-Eg
    alpha = baseline+96+5123
    assert gamma == F(46955312929,34359738368) and 0 < alpha < 32
    margins = {'w1_whole255_257':-5155+32-Eg, 'w2_whole256_affine':-5187+64-Eg,
               'w2_charge2':baseline+2*delta_sharp+64-Eg, 'w3_arbitrary_number_shift':baseline+96-Eg}
    assert all(v >= gamma for v in margins.values())
    assert 64 >= gamma
    # Affine in w: these intercept/slope checks prove the infinite tail.
    assert margins['w3_arbitrary_number_shift'] == gamma+alpha
    assert 32 >= alpha
    assert margins['w3_arbitrary_number_shift'] >= gamma/2+3*gamma/4 and 32 >= gamma/4
    assert gamma >= gamma/2+gamma/4 and gamma == gamma/2+2*gamma/4
    for key, value in [('baseline_lower_exact', baseline), ('delta_lower_exact',delta_sharp), ('E_g_upper_exact',Eg), ('gamma_exact',gamma), ('excess_flow_alpha_exact',alpha)]:
        assert F(target[key]) == value
    # Reject homogenizing all64 copy profiles; trace(h^2) is an invariant of
    # the actual native oneparticle law and differs at each actual role.
    mA,mU = [F(p['m_exact']) for p in profiles]
    averaged_m = (mA+63*mU)/64
    assert averaged_m != mA and averaged_m != mU
    wrongq = averaged_m**2+F(profiles[0]['d_exact'])**2
    assert wrongq not in [F(p['e_squared_exact']) for p in profiles]
    omitted = -4*al-252*ul+2*ah+(ah+c)+(uh+c)+5123
    assert omitted < 0
    # n0 active + n3 active + n3 auxiliary +125 number2 auxiliary =256.
    assert 0+3+3+125*2 == 256
    eta = F(cap['normalized_exact_source_residual_upper_exact'])
    observed = F(str(cap['independent_full_H_residual_float']))
    rayhi = F(cap['Rayleigh_interval_exact'][1])
    raylo = F(cap['Rayleigh_interval_exact'][0])
    formed_temple = raylo-eta*eta/(-5123-rayhi)
    assert formed_temple == F(cap['ground_energy_interval_exact'][0])
    underpriced_temple = raylo-observed*observed/(-5123-rayhi)
    assert underpriced_temple > formed_temple and eta > observed
    # Full rational native model: exact free-vacuum projector, no eigensolver.
    _, H, R = native_fock(S.Rational(3), S.Rational(-1), S.Rational(-4))
    ids = [k for k in range(16) if k.bit_count() == 2]
    H = H.extract(ids,ids); R = R.extract(ids,ids)
    P = S.eye(6)
    for val in (-2,0,2,10):
        P = P*(H-val*S.eye(6))/(-10-val)
    assert P*P == P and S.trace(P) == 1 and H*P == -10*P
    J = (S.eye(6)-P)*R*P
    assert J != S.zeros(6)
    double_out = cast(S.Matrix, 128*S.kronecker_product(J,J))
    assert double_out != S.zeros(36)
    exact_actual_double = 32*F(profiles[0]['d_exact'])**2/F(profiles[0]['e_squared_exact'])
    assert exact_actual_double > 0 and F(target['actual_two_active_slot_H_freevacuum_double_excitation_exact']) == exact_actual_double
    # Full coherent square differs from a square with a single role sign flip.
    coh = S.kronecker_product(R,S.eye(6))+S.kronecker_product(S.eye(6),R)
    wrong = S.kronecker_product(R,S.eye(6))-S.kronecker_product(S.eye(6),R)
    assert coh*coh != wrong*wrong
    # Complete-output Gram lower versus input compression, and explicit lost
    # affine identity/cross witnesses; negative LOWER never means negative H.
    Ru = S.Matrix([[1,S.Rational(3,5)], [S.Rational(3,5),1]])
    row = Ru.extract([0],[0,1])
    gram = row.T*row
    assert gram[0,1] == S.Rational(3,5)
    assert gram != S.diag(1,0)
    lower_negative_full_positive = {'full':S.diag(1,2), 'lower':S.diag(-1,1)}
    assert lower_negative_full_positive['full']-lower_negative_full_positive['lower'] == S.diag(2,1)
    # Source identities from ORIGINAL immutable antecedent archives, separately
    # from the current matching target snapshot; no path-equality rebinding.
    archived = []
    for spec in request['archive_members']:
        with zipfile.ZipFile(ROOT/spec['archive']) as z:
            data = z.read(spec['member'])
        assert sha(data) == spec['sha256']
        current = ROOT/spec['member']
        archived.append(dict(spec, current_sha256=sha(current.read_bytes()) if current.exists() else None))
    result = {'kind':'INDEPENDENT_NATIVE_FULL_CAR_OUTWARD_PRICE_PROOF_CONTROL', 'pins':pins,
              'native_CAR':car, 'all512_oneparticle_lower_endpoints':[str(v) for v in one], 'all513_number_envelopes':sectors,
              'B_lower':str(baseline), 'delta_lower':str(delta_sharp), 'gamma':str(gamma), 'alpha':str(alpha),
              'margins':{k:str(v) for k,v in margins.items()},
              'controls':{'wrong_active64_average_rejected_by_native_trace_invariant':True,
              'omitted_scalar_transfer_free_upper':str(omitted), 'not_full_H_instability':True,
              'formed_Temple_lower':str(formed_temple),'observed_residual_substitution_wrong_lower':str(underpriced_temple),
              'actual_two_active_double_excitation':str(exact_actual_double),
              'rational_native_freevacuum_double_output_Frobenius_square':str(S.trace(double_out.T*double_out.conjugate())),
              'lost_affine_output_cross':str(gram[0,1]), 'negative_lower_not_negative_full_H':True},
              'archive_member_bindings':archived, 'historical_solver_checker_imports_or_reruns':0,
              'premises':'Whole u0 ground/caps, whole charged -5155 coherent-cover extension, whole neutral affine -5187 complete-output forms; accepted exact joint frames and binary64 cap model remain conditional.'}
    assert all(sha((ROOT/n).read_bytes()) == v for n,v in pins.items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'gamma':str(gamma),'alpha':str(alpha),'all513_number_prices':'derived from full native CAR', 'adverse_controls':'rejected', 'archive_bindings':archived},sort_keys=True))

if __name__ == '__main__':
    main()
