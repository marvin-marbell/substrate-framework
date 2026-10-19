"""Independent dense-word unbounded-current cutoff and paid Q-budget controls."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import sympy as sp

sys.set_int_max_str_digits(0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dense_current() -> dict[str, Any]:
    # NON-NATIVE l2(N), J e_n=n^2 e_n, Omega_n=n^-3 (unnormalized).
    # Core includes finite basis vectors and finite shift/phase words on Omega.
    # Each shift word S^r Omega has finite J norm; J Omega is not in D(J).
    rows = []
    for r in (0, 5, 17):
        errors = []
        for m in (32, 64, 128):
            k = m-r
            upper = (1+F(r, k+1))**4/F(k)
            # Strict positive lower control, not fabricated equality to the infinite series.
            lower = sum((F(n**4, (n-r)**6) for n in range(m+1, 2*m+1)), F(0))
            assert 0 < lower < upper
            errors.append({'M': m, 'tail_squared_lower_partial': str(lower),
                           'tail_squared_upper': str(upper)})
        assert F(errors[-1]['tail_squared_upper']) < F(errors[0]['tail_squared_upper'])
        norm_formula = f'zeta(2)+{4*r}*zeta(3)+{6*r*r}*zeta(4)+{4*r**3}*zeta(5)+{r**4}*zeta(6)'
        rows.append({'shift_r': r, 'current_word_norm_squared_exact_formula': norm_formula,
                     'current_cutoff_tails': errors})
    assert F(rows[2]['current_cutoff_tails'][0]['tail_squared_lower_partial']) > F(rows[0]['current_cutoff_tails'][0]['tail_squared_lower_partial'])
    # Exact finite noncommuting words; diagonal phase D_n=(-1)^n obeys DS=-SD.
    size = 24
    J: Any = sp.diag(*[n*n for n in range(1, size+1)])
    D: Any = sp.diag(*[(-1)**n for n in range(1, size+1)])
    S: Any = sp.zeros(size)
    for n in range(size-1):
        S[n+1, n] = 1
    assert D*S == -S*D
    vectors = []
    base: Any = sp.Matrix([sp.Rational(1, n**3) for n in range(1, size+1)])
    for length in range(5):
        for letters in itertools.product((D, S), repeat=length):
            value = base
            for a in letters:
                value = a*value
            vectors.append(value)
    for p, q in zip(vectors, reversed(vectors)):
        assert (p.T*J*q)[0] == ((J*p).T*q)[0]
    return {'scope': 'NON_NATIVE_EXACT_INFINITE_CURRENT_CUTOFF_CAUCHY_AND_FINITE_FULL_WORD_SYMMETRY',
            'Omega_norm_squared': 'zeta(6)', 'J_Omega_norm_squared': 'zeta(2)',
            'J_Omega_in_D_J': False, 'J_squared_Omega_partial_norm_diverges': 'sum_(n<=K) n^2',
            'dense_core': 'span{finite basis vectors and all finite shift/phase words on Omega}',
            'cutoff': 'bounded selfadjoint J_M e_n=n^2 1_(n<=M)e_n',
            'dense_word_cutoff_rows': rows, 'finite_full_noncommuting_words': len(vectors),
            'cutoff_symmetry_pairings_exact': True,
            'closure_reason': 'Strong cutoff limits define symmetric operator on dense core; symmetry implies closability. No J^2 or essential-selfadjointness assumption used.',
            'vacuum_only_cutoff_budget_rejected': True, 'native_quantum_ground_claim': False}


def native_word_budgets(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    ce = 8*sum((F(p['load'])*t['length']*(abs(F(t['c0']))+abs(F(t['ci'])))
                for p in profiles[1:] for t in p['paths'] if t['axis'] == 0), F(0))
    r = 60*chi+11192*lam+F(129, 10)*fast+262144
    b0 = 33*(1+10*r*4096)+16384*lam*4096
    u: Any = sp.symbols('u0:4', positive=True)
    budget: Any = sp.Rational(b0)*u[0]
    norm2: Any = sp.Integer(1)
    for j, (a, jets) in enumerate(((1, 2), (3, 4), (2, 5))):
        d = (4*chi+F(3, 20)*chi+64)*jets*jets
        budget = sp.expand(66*a*a*budget+sp.Rational(66*d+16384*lam*4096*a*a)*norm2)
        norm2 *= a*a
        budget = sp.expand(u[j+1]*budget)
    # u0 is the INITIAL rightmost packet propagator (1 for exact ground); u3 is leading.
    tail: Any = sp.expand(sp.Rational((4*ce)**2*2*16*8/lam)*budget)
    polynomial: Any = sp.Poly(tail, *u)
    assert all(c > 0 for c in polynomial.coeffs())
    assert tail.has(u[0], u[-1])
    direct_cost: Any = sp.expand(tail.subs({factor: 1 for factor in u}))
    assert direct_cost > (4*ce)**2*2*16*8*sp.Rational(b0)/sp.Rational(lam)
    return {'scope': 'EXACT_NATIVE_DECLARED_FULL_WORD_COST_ARITHMETIC_NOT_EVOLUTION',
            'C_e': str(ce), 'insertion_operator_norms': [1, 3, 2],
            'tail_polynomial_terms': len(polynomial.terms()),
            'tail_polynomial_sha256': sha(str(tail).encode()),
            'tail_monomial_exponents': [list(x[0]) for x in polynomial.terms()],
            'norm_bound_squared': str(norm2), 'initial_packet_and_leading_propagators_retained': True,
            'u_meaning': 'exp(Cstar*abs(actual_interval)), retained symbolically; initial u0=1 only at exact ground',
            'native_Cstar_exponential_numerically_evaluated': False,
            'all_dense_word_tail_formula': '(H1*C_e)^2*2m*b_word/(lambda*tau*M)'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {key: sha(Path(row['path']).read_bytes()) for key, row in request['inputs'].items()}
    assert before == {key: row['sha256'] for key, row in request['inputs'].items()}
    with zipfile.ZipFile(request['profile_archive']) as archive:
        raw = archive.read(request['profile_member'])
        profiles = json.loads(raw)['primitive_profiles']
    receipt = {'schema': 1, 'kind': 'INDEPENDENT_DENSE_CURRENT_CLOSABILITY_AND_FULL_WORD_BUDGET_CONTROLS',
               'dense_current': dense_current(), 'native_full_word_budgets': native_word_budgets(profiles),
               'earlier_completed_epoch': request['earlier_completed_epoch'],
               'source_input_sha256_before': before,
               'source_input_sha256_after': {key: sha(Path(row['path']).read_bytes()) for key, row in request['inputs'].items()},
               'runtime': {'executable': sys.executable, 'elapsed_seconds': time.monotonic()-start},
               'scientific_admission': False, 'native_ground_or_fullH_evolution': False, 'photon': False}
    assert before == receipt['source_input_sha256_after']
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'dense_unbounded_current_cutoff_controls': True, 'native_ground': False, 'photon': False}))


if __name__ == '__main__':
    main()
