"""Native slab deck covariance and full-cover heat-trace obstruction.

This is an ensemble-boundary control, not a native thermal-pressure or phase
calculation. It leaves full quantum Phi, colour and canonical CAR untouched.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from magnetic_boundary_slab_scientist_20261011 import FixedMagneticSlab

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def interior_deck(n: int) -> dict[str, Any]:
    slab = FixedMagneticSlab(n, 1)
    start = slab.index[(1, 1, 1)]
    py = slab.path(start, 1, 1)
    pz = slab.path(start, 2, 1)
    assert py is not None and pz is not None
    yz = slab.path(py['target'], 2, 1)
    zy = slab.path(pz['target'], 1, 1)
    assert yz is not None and zy is not None and yz['target'] == zy['target']
    k: dict[int, int] = {int(py['edges'][0]): 1, int(yz['edges'][0]): 1,
                         int(zy['edges'][0]): -1, int(pz['edges'][0]): -1}
    assert len(k) == 4 and all(edge in slab.dynamic for edge in k)
    assert not any(edge in slab.pinned for edge in k)
    column = sp.Matrix([k.get(edge, 0) for edge in slab.dynamic])
    DI = sp.Matrix([[int(value) for value in row] for row in slab.DI.toarray()])
    assert DI * column == sp.zeros(len(slab.interior), 1)
    assert (column.T * column)[0] == 4
    slab.accept_translation({edge: Fraction(value) for edge, value in k.items()})
    paths = 0
    changed_half_period_paths = 0
    for site in range(slab.v):
        for axis in range(3):
            for length in (1, 2):
                path = slab.path(site, axis, length)
                if path is None:
                    continue
                winding = sum(k.get(edge, 0) for edge in path['edges'])
                assert sp.exp(2 * sp.pi * sp.I * winding) == 1
                if sp.exp(sp.pi * sp.I * winding) != 1:
                    changed_half_period_paths += 1
                paths += 1
    assert changed_half_period_paths > 0
    return {'N': n, 'interior_face_base': [1, 1, 1],
            'integer_divfree_dynamic_deck': sorted(k.items()),
            'norm_squared': 4, 'interior_divergence_zero': True,
            'pinned_boundary_unchanged': True, 'native_one_two_paths': paths,
            'half_period_wrongcase_changed_paths': changed_half_period_paths,
            'Gauss_density_not_changed': True,
            'all64_charged_periodicity': 'Exact integer path phase; arbitrary full Phi/ordered SU4 coefficients unchanged. Neutral has no U1 phase; Ref/number/density operators untouched.',
            'quantum_field_space_dimension': len(slab.dynamic)-len(slab.interior),
            'dimension_uses_reviewed_fullrowrank_Dirichlet_incidence': True}


def ensemble_obstruction() -> dict[str, Any]:
    return {
        'hypotheses': 'Accepted finite-N lower-semibounded physical quotient Friedrichs H, no bare B2, arbitrary full Phi/SU4/all64 canonical+neutral+Ref; nonzero compact transverse packet in the physical Hilbert.',
        'native_commutation': 'T_j translates by 2pi*j*k on dynamic quotient. D_I k=0, pinned shift0, every charged phase unchanged and kinetic/collective HC unchanged. Hence T_j H=H T_j.',
        'orthogonal_family': 'Choose packet support in a ball of radius less than pi*||k||. Its integer deck translates have disjoint connection support, all canonical/colour/Phi components retained.',
        'positive_heat_diagonal': 'For any inverse_temperature>0, c=<psi,exp(-inverse_temperature*H)psi> is finite and strictly positive by spectral calculus; every normalized orthogonal translate has the same c.',
        'full_cover_trace': 'For every integer M, Tr exp(-inverse_temperature H)>=(2M+1)c, therefore the ordinary full-cover heat trace is INFINITE.',
        'not_a_phase_no_go': True,
        'finite_full_cover_partition_assumption_rejected': True,
        'required_next_boundary': 'Trace per native deck cell or equivalent regularization with affine Gauss/CAR deck gluing and integrated electric Bloch characters. This must be constructed, not silently beta0.',
        'ordinary_real_cover_Gibbs_density_computed': False,
        'raw_unwrapped_regular_field_admitted': False}


def free_character_control() -> list[dict[str, Any]]:
    b = sp.Symbol('b', real=True)
    x = sp.Symbol('x', real=True)
    rows = []
    for beta in (sp.Rational(1, 64), sp.Integer(1), sp.Integer(2)):
        cut = 4
        integrated = sum((sp.integrate(sp.exp(-64*beta*(j+b)**2), (b, 0, 1))
                          for j in range(-cut, cut)), sp.Integer(0))
        exact = sp.sqrt(sp.pi/(64*beta))
        finite_interval = sp.integrate(sp.exp(-64*beta*x*x), (x, -cut, cut))
        assert sp.simplify(integrated-finite_interval) == 0
        tail = exact*(1-sp.erf(8*sp.sqrt(beta)*cut))
        assert sp.simplify(integrated+tail-exact) == 0
        # The beta_Bloch=0 heat trace includes its zero momentum eigenvector,
        # while the integrated trace per cell is strictly less than1 here.
        if beta >= 1:
            assert exact < 1
        rows.append({'inverse_temperature': str(beta),
                     'integrated_character_heat_per_cell_exact': str(exact),
                     'finite_character_integral_exact': str(integrated),
                     'whole_tail_exact': str(tail),
                     'zero_Bloch_trace_lower': 1,
                     'zero_Bloch_substitution_rejected': bool(beta >= 1),
                     'toy_free_Pi_only_NOT_native_pressure': True})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert all(digest(ROOT/name) == pin for name, pin in request['pins'].items())
    result = {'kind': 'NATIVE_MAGNETIC_SLAB_NONCOMPACT_THERMAL_TRACE_BOUNDARY_NOT_PHASE',
              'source_sha256': digest(Path(__file__)),
              'native_deck': [interior_deck(n) for n in (4, 6, 8)],
              'full_quantum_heat_trace_obstruction': ensemble_obstruction(),
              'integrated_character_method_control': free_character_control(),
              'native_thermal_response_computed': False,
              'physical1_phase_stiffness_photon': False, 'campaign_complete': False}
    assert all(digest(ROOT/name) == pin for name, pin in request['pins'].items())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'N': [4, 6, 8], 'full_cover_partition': 'infinite',
                      'native_thermal_pressure': False, 'phase_or_pole': False}))


if __name__ == '__main__':
    main()
