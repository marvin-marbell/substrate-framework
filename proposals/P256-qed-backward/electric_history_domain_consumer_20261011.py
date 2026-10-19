"""New native per-word current-cutoff costs and unbounded-domain method witness."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_record(v: F) -> dict[str, Any]:
    return {'sha256': digest(str(v).encode()), 'numerator_bits': v.numerator.bit_length(),
            'denominator_bits': v.denominator.bit_length()}


def native_domain(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    ce = 8*sum((p['load']*t['length']*(abs(F(t['c0']))+abs(F(t['ci'])))
                for p in profiles[1:] for t in p['paths'] if t['axis'] == 0), F(0))
    w = [abs(F(p['onsite']))+2*sum((abs(F(t['c0']))+abs(F(t['ci'])) for t in p['paths']), F(0)) for p in profiles]
    b = 4*(lam*w[0]+w[1]+63*w[2])
    r = 60*chi+11192*lam+F(129, 10)*fast+262144
    b0 = 33*(1+10*r*4096)+16384*lam*4096
    # Three native insertion budgets: norm2, declared support/covariant jets.
    budgets = [b0]; norm2 = F(1)
    for a, m, tau_inv in [(F(1), 4, F(2)), (F(2), 6, F(4)), (F(1), 7, F(8))]:
        d = (4*chi+F(3, 20)*chi+64)*m
        new = 66*a*a*budgets[-1]+(66*d+16384*lam*4096*a*a)*norm2
        budgets.append(new); norm2 *= a*a
    rows = []
    for word_length, budget in enumerate(budgets):
        # A supported unit plaquette union can conservatively use m<=16;
        # translated support max distance8 gives tau>=1/4.
        h1, m, tau_inv = F(4), 16, F(4)
        tail_coefficient = (h1*ce)**2*2*m*budget*tau_inv/lam
        delta2 = F(1, 1024)
        threshold = (tail_coefficient/delta2).__ceil__()
        actual_upper = tail_coefficient/threshold
        assert actual_upper <= delta2 and threshold > 0
        rows.append({'word_length': word_length, 'Q_budget': exact_record(budget),
                     'current_tail_Ap_over_M_coefficient': exact_record(tail_coefficient),
                     'requested_error_squared': str(delta2),
                     'cutoff_M_sufficient': exact_record(F(threshold)),
                     'derived_tail_upper_at_M': exact_record(actual_upper)})
    # Arbitrary nonzero time factors, not evaluated massive native exponential.
    u, v = sp.symbols('u v', nonnegative=True)
    symbolic: Any = 66*u*sp.Rational(b0)+sp.Rational(16384*lam*4096)
    symbolic = sp.expand(264*v*symbolic+sp.Rational(4*16384*lam*4096))
    tail: Any = sp.expand(sp.Rational((4*ce)**2*2*16*4/lam)*symbolic)
    assert all(x > 0 for x in sp.Poly(tail, u, v).coeffs())
    return {'scope': 'native exact bound arithmetic on declared full-history words, not actual current measurements',
            'charged_C_e': str(ce), 'all65_B': exact_record(b), 'word_cutoff_rows': rows,
            'mixed_nonzero_time_tail_polynomial_terms': len(sp.Poly(tail, u, v).terms()),
            'mixed_nonzero_time_tail_polynomial_sha256': digest(str(tail).encode()),
            'u_v_mean_native_exp_Cstar_abs_time': True,
            'current_domain_each_dense_word_paid': True,
            'essential_selfadjointness_or_LJ_graph_paid': False}


def infinite_current_method() -> dict[str, Any]:
    # NON-NATIVE l2(N), J e_k=k e_k, Omega_k=sqrt3*2^-k.
    # Words of bounded unilateral shifts and diagonal phases are noncommuting.
    r = F(1, 4)
    rows = []
    for shift in (0, 1, 3, 5):
        current_norm2 = F(shift*shift)+F(8, 3)*shift+F(20, 9)
        errors = []
        for cutoff in (8, 16, 32):
            a = cutoff+1
            series = r**a*(F(a*a)/(1-r)+2*a*r/(1-r)**2+r*(1+r)/(1-r)**3)
            tail = 3*r**(-shift)*series
            assert 0 < tail < current_norm2
            errors.append({'M': cutoff, 'current_cutoff_squared_error': str(tail)})
        assert F(errors[0]['current_cutoff_squared_error']) > F(errors[1]['current_cutoff_squared_error'])
        rows.append({'shift_word_length': shift, 'current_word_norm_squared': str(current_norm2), 'tails': errors})
    assert F(rows[3]['tails'][0]['current_cutoff_squared_error']) > F(rows[0]['tails'][0]['current_cutoff_squared_error'])
    return {'scope': 'NON_NATIVE exact infinite-dimensional unbounded-current domain witness',
            'normalized_Omega_sum_3_over_4power': '1', 'rows': rows,
            'full_history_word_not_vacuum_cutoff_constant_rejected': True,
            'phase_times_shift_noncommute': 'For D e_k=(-1)^k e_k, D S=-S D; no compression between letters',
            'symmetric_J_on_dense_words_and_strong_cutoff_vectors': True,
            'native_ground_or_fullH_claim': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {p: digest((ROOT/p).read_bytes()) for p in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['profile_archive']) as archive:
        profiles = json.loads(archive.read(request['profile_member']))['primitive_profiles']
    data = {'schema': 1, 'kind': 'NEW_DENSE_CURRENT_HISTORY_DOMAIN_CONSUMER',
            'native_per_word_current_cutoff_cost': native_domain(profiles),
            'non_native_infinite_current_method': infinite_current_method(),
            'analytical_proof': json.loads((ROOT/request['proof']).read_text()),
            'source_input_sha256_before': before,
            'source_input_sha256_after': {p: digest((ROOT/p).read_bytes()) for p in request['pins']},
            'runtime': {'executable': sys.executable, 'elapsed_seconds': time.monotonic()-start}}
    assert before == data['source_input_sha256_after']
    with args.output.open('x') as stream:
        json.dump(data, stream, indent=2); stream.write('\n')
    print(json.dumps({'per_word_current_domain': True, 'native_ground_computed': False, 'photon': False}))


if __name__ == '__main__':
    main()
