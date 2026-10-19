"""Exact inverse-form boundary controls; not a numerical native ground or phase."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import sympy as sp

ROOT = Path(__file__).resolve().parent


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def controls() -> dict[str, Any]:
    s: Any = sp.Symbol('s', real=True)
    a: Any = sp.Symbol('a', positive=True)
    g: Any = sp.Symbol('g', positive=True)
    h: Any = sp.diag(0, g)
    e: Any = sp.Matrix([[0, a], [a, 0]])
    shifted: Any = h + 128*s*e + 256*s*s*sp.eye(2)
    determinant: Any = sp.factor(shifted.det())
    assert sp.expand(determinant - 256*s*s*(g-64*a*a+256*s*s)) == 0
    # All-character positivity is equivalent to g>=64*a^2, not merely H>=0.
    allowed: Any = shifted.subs(g, 64*a*a)
    assert sp.factor(allowed.det()) == 65536*s**4
    wrong: Any = shifted.subs({g: 32, a: 1, s: sp.Rational(1, 16)})
    assert wrong[0, 0] > 0 and wrong.det() < 0
    inverse: Any = a*a/g
    assert inverse.subs(g, 64*a*a) == sp.Rational(1, 64)
    # Mean zero does not remove the entire degenerate ground block.
    hz: Any = sp.diag(0, 0, 64)
    ez: Any = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    omega: Any = sp.Matrix([1, 0, 0])
    p0: Any = sp.diag(1, 1, 0)
    assert (omega.T*ez*omega)[0] == 0 and (p0*ez*omega).norm() == 1
    bad_shift: Any = hz + 8*ez + sp.eye(3)
    assert (sp.Matrix([1, -1, 0]).T*bad_shift*sp.Matrix([1, -1, 0]))[0] == -14
    # Real quadratic control cannot silently delete an imaginary matrix cross.
    susceptibility: Any = sp.Matrix([[1, sp.I], [-sp.I, 1]])
    real_x: Any = sp.Symbol('x', real=True)
    real_y: Any = sp.Symbol('y', real=True)
    rv: Any = sp.Matrix([real_x, real_y])
    assert sp.expand((rv.T*susceptibility*rv)[0]) == real_x**2+real_y**2
    cv: Any = sp.Matrix([1, -sp.I])
    assert (cv.conjugate().T*susceptibility*cv)[0] == 4
    assert (cv.conjugate().T*cv)[0] == 2
    # E4 alone permits a unit source atom to collapse to zero; inverse control does not.
    collapsing = []
    for n in (4, 16, 256):
        energy = sp.Rational(1, n)
        fourth = sp.Integer(1)
        inv = 1/energy
        assert fourth == 1 and inv > sp.Rational(1, 64)
        collapsing.append({'E': str(energy), 'source_mass': '1', 'E_source_fourth_moment': '1',
                           'inverse_moment': str(inv), 'uniform_inverse_guard_pass': False})
    q: Any = sp.Symbol('q', positive=True)
    # Both an atom and an atomless continuum obey the same complex inverse upper.
    atom_inverse: Any = sp.Rational(1, 256)+sp.Rational(1, 256)
    continuum_inverse: Any = sp.log(2)/256+sp.Rational(1, 256)
    assert atom_inverse == sp.Rational(1, 128)
    assert sp.simplify(sp.Rational(1, 128)-continuum_inverse).is_positive
    atom_mass: Any = q/256
    continuum_mass: Any = q/256
    assert atom_mass == continuum_mass and 2*q/128 > atom_mass
    return {'kind': 'EXACT_NONNATIVE_ENERGY_RIESZ_WHOLEP0_COMPLEXIFICATION_METHOD_NOT_PHASE',
            'all_character_shift_determinant': str(determinant),
            'all_character_positivity_condition': 'g>=64*a^2',
            'face_inverse_upper': '1/64',
            'selected_positive_H_without_all_character_floor_rejected': True,
            'degenerate_wholeP0_vs_mean_rejected': True,
            'complex_factor_two_required_without_extra_symmetry': True,
            'imaginary_cross_retained': [['1', 'I'], ['-I', '1']],
            'E4_only_collapsed_zero_block_controls': collapsing,
            'allowed_atomic_measure': '(q/256)*delta_q+delta_256',
            'allowed_atomless_measure': '(1/256)*Lebesgue_[q,2q]+delta_256',
            'atomic_inverse_moment': str(atom_inverse),
            'atomless_inverse_moment': str(continuum_inverse),
            'upper_bound_does_not_force_atom_or_lowband_nonzero': True,
            'native_ground_fullH_backend': False,
            'native_inverse_theorem_or_transverse_IR_proved_by_this_method': False,
            'phase_stiffness_sharp_photon_commoncone': False,
            'campaign_complete': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    result = controls()
    result['source_input_sha256_before'] = before
    result['source_input_sha256_after'] = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    result['runtime'] = {'python': sys.executable, 'sympy': sp.__version__, 'seconds': time.monotonic()-started}
    assert result['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'exact_method_controls_completed': True, 'native_ground_or_phase': False}))


if __name__ == '__main__':
    main()
