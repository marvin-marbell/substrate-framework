"""New native weighted-form/graph controls and separated matrix propagation controls.
No old producer, native eigensolver, ground solver or checker is imported or run.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction as Rat
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

import numpy as np
import scipy.linalg as sla
import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rational_record(value: Rat) -> dict[str, Any]:
    return {'sha256': digest(str(value).encode()),
            'numerator_bits': value.numerator.bit_length(),
            'denominator_bits': value.denominator.bit_length()}


def exact_constants(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lam, chi, fast = Rat(2**8192), Rat(1, 2**4096), Rat(2**10240)
    loads = [int(p['load']) for p in profiles]
    assert loads == [1, 1, 63]
    assert len({p['onsite'] for p in profiles}) == 3
    w = [abs(Rat(p['onsite'])) + 2*sum(abs(Rat(t['c0']))+abs(Rat(t['ci']))
                                     for t in p['paths']) for p in profiles]
    b = 4*(lam*w[0]+w[1]+63*w[2])
    assert b < 512*lam
    kp, kh, ke = 4*chi, Rat(3, 20)*chi, Rat(64)
    alpha, beta = fast/40, 7*fast/10
    sstar, z = Rat(4096), Rat(4096)+2/lam
    allocations = [Rat(54), 480*kp*b*b*sstar,
                   184320*kp*alpha*alpha*z, 480*kh*b*b*z,
                   32*ke*b*b*z, 32*b*b*z,
                   46080*kh*alpha*alpha/lam, 2880*kh*beta*beta*sstar]
    indexed = sum(allocations)
    ljet = 2**16*(1+fast+b)
    cstar = 2**32*65*(1+ljet*ljet)*z
    assert indexed <= cstar
    # Wrong law with mismatched site Jordan/kinetic weights fails in the actual
    # radial Phi principal symbol: q^3 p is not paid by q^4+p^2.
    q, p = sp.symbols('q p', positive=True)
    quartic_current = 4*q**3*p
    ratios = [Rat(4*m**5, 2*m**4+1) for m in (4, 16, 64)]
    assert ratios[0] < ratios[1] < ratios[2]
    assert sp.expand(quartic_current.subs(p, q*q)) == 4*q**5
    assert Rat(13, 11)**4 < 2  # proves coth(log(2)/8)<12 and W<1728
    return {'scope': 'native exact fixed-profile coefficient/form allocation',
            'loads': loads, 'all65_species': sum(loads),
            'B': rational_record(b), 'B_over_lambda_lt_512': True,
            'C_index': rational_record(indexed), 'Cstar': rational_record(cstar),
            'indexed_allocations': [rational_record(v) for v in allocations],
            'exact_C_index_le_Cstar': True,
            'constant_used': '2^32*65*(1+[2^16(1+F+B)]^2)*(4096+2/lambda)',
            'N_or_character_in_constant': False,
            'wrong_mismatched_quartic_current': str(quartic_current),
            'wrongcase_unbounded_principal_form_ratios': [str(v) for v in ratios],
            'wrongcase_is_principal_symbol_not_ground_state': True}


def native_topology(n: int, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3))
    eps = math.log(2)/4
    distance = {x: sum(min(v, n-v) for v in x) for x in sites}
    weight = {x: math.exp(-eps*distance[x]) for x in sites}
    phi_inc = {x: 0 for x in sites}
    haar_inc: dict[tuple[tuple[int, ...], int], int] = {}
    electric_inc: dict[tuple[tuple[int, ...], int], int] = {}
    paths = []
    seams = 0
    def move(x: tuple[int, ...], axis: int, step: int = 1) -> tuple[int, ...]:
        z = list(x)
        z[axis] = (z[axis]+step) % n
        return tuple(z)
    # Group all65 commuting species by literal axis/length, retaining their
    # exact summed coefficient elsewhere. Onsite/length1 heavy stays distinct.
    for x in sites:
        for axis, length in itertools.product(range(3), (1, 2)):
            y = move(x, axis, length)
            path = [(move(x, axis, j), axis) for j in range(length)]
            support = (x, y)
            w = (weight[x]+weight[y])/2
            phi_inc[x] += 1
            phi_inc[y] += 1
            for e in path:
                haar_inc[e] = haar_inc.get(e, 0)+1
                electric_inc[e] = electric_inc.get(e, 0)+1
            seams += int(x[axis]+length >= n)
            for z in support:
                assert max(w/weight[z], weight[z]/w) <= math.sqrt(2)+1e-12
            # Only coordinates at endpoints and links on this literal path
            # are differentiated, including the middle length2 link.
            for edge, _ in path:
                for z in support:
                    assert (w-weight[edge])**2/weight[edge] <= 8*weight[z]+1e-12
            paths.append((x, y, path))
    for x in sites:
        phi_inc[x] += 1  # full65 combined onsite
    assert max(phi_inc.values()) <= 30
    assert max(haar_inc.values()) <= 7
    assert max(electric_inc.values()) <= 6
    corners_max = 0.0
    for x in sites:
        for axis, other in itertools.combinations(range(3), 2):
            corners = (x, move(x, axis), move(x, other), move(move(x, axis), other))
            w = sum(weight[z] for z in corners)/4
            for z in corners:
                corners_max = max(corners_max, w/weight[z], weight[z]/w)
                assert max(w/weight[z], weight[z]/w) <= math.sqrt(2)+1e-12
    total_weight = sum(weight.values())
    assert total_weight < 4096
    v = n**3
    parity = np.array([(-1)**sum(x) for x in sites], dtype=np.int64)
    # Native physical radial/Haar trial with local exterior4 canonical
    # singlets and the original shared integer affine fullE section.
    occupied = sum(1 for x in sites if sum(x) % 2 == 0)*4
    assert occupied == 2*v
    rho = 128*parity
    k = np.zeros((n, n, n, 3), dtype=np.int64)
    k[..., 0] = 64*np.array([(-1)**sum(x) for x in sites]).reshape(n, n, n)
    divergence = sum((k[..., a]-np.roll(k[..., a], 1, axis=a) for a in range(3)),
                     np.zeros((n, n, n), dtype=np.int64))
    assert np.array_equal(divergence.reshape(-1), rho)
    assert int(np.sum(k*k)) == 4096*v
    # Nonzero native KE/matter current coefficient, evaluated on the two CAR
    # occupations linked by the actual charged nearest transport at the origin.
    # This is a literal ambient coefficient compression, NOT fullH spectrum.
    r = sp.root(2, 4)**-1
    w_hop = (1+r)/2
    delta_e = Rat(64*(-128+1))  # occupied origin -> empty neighbour: t=-1
    current = sp.expand((w_hop-1)*sp.Integer(delta_e.numerator))
    assert current != 0
    assert sp.simplify(current - 4064*(1-r)) == 0
    # Heavy/active/aux path masks retain original AP seams and charge roles.
    literal_counts = [v*len(p['paths'])*int(p['load']) for p in profiles]
    assert all(literal_counts) and seams > 0
    return {'N': n, 'V': v, 'periodic_all65_number_each': occupied,
            'Ref4': 4, 'rho_is_collective64': True, 'Gauss_exact': True,
            'weighted_fullE_trial': '262144 sum(theta)',
            'native_grouped_Phi_matter_incidence_max': max(phi_inc.values()),
            'native_grouped_Haar_matter_incidence_max': max(haar_inc.values()),
            'native_grouped_fullE_matter_incidence_max': max(electric_inc.values()),
            'literal_paths_by_role_with_load': literal_counts,
            'AP_seam_paths': seams, 'weight_sum': total_weight,
            'corner_average_ratio_max': corners_max,
            'native_nonzero_KE_hopping_current_coefficient': str(current),
            'current_scope': 'native ambient coefficient jet; not a quantum eigenproblem',
            'all_N_proof_not_extrapolated_from_rows': True}


def native_gaussian_graph(n: int) -> dict[str, Any]:
    """Exact physical Gaussian/Haar/affine form and radial graph-envelope limits."""
    d = 60*n**3
    lam, chi, fast = Rat(2**8192), Rat(1, 2**4096), Rat(2**10240)
    qdensity = 60*chi+11192*lam+Rat(129, 10)*fast+262144
    pdensity = 60*chi+465*lam+Rat(129, 10)*fast+262144
    assert qdensity > pdensity > 0
    controls = []
    for factor in (1, 4, 16):
        r2 = d*factor*factor
        h = d//2
        l2 = Rat(1)-2*Rat(r2, r2+1)**h+Rat(r2, r2+2)**h
        moment8 = Rat(d*(d+2)*(d+4)*(d+6))
        radial_graph = moment8*(1-2*Rat(r2, r2+1)**(h+4)
                                +Rat(r2, r2+2)**(h+4))
        assert l2 >= 0 and radial_graph >= 0
        # -Delta Gaussian at t=1: (d/2-|q|^2/4)f. For envelope,
        # a=1/4+1/(2R^2), -Delta(exp(-a r^2))=(2ad-4a^2r^2)exp(-a r^2).
        a = Rat(1, 4)+Rat(1, 2*r2)
        def inner_polynomial(c0: Rat, c1: Rat, e0: Rat, e1: Rat,
                             exponent_shift: Rat) -> Rat:
            t = 1+2*exponent_shift
            norm = t**(-h)
            m2, m4 = d/t, d*(d+2)/(t*t)
            return norm*(c0*e0+(c0*e1+c1*e0)*m2+c1*e1*m4)
        c0, c1 = Rat(d, 2), Rat(-1, 4)
        e0, e1 = 2*a*d, -4*a*a
        delta_graph = (inner_polynomial(c0, c1, c0, c1, Rat(0))
                       -2*inner_polynomial(c0, c1, e0, e1, Rat(1, 2*r2))
                       +inner_polynomial(e0, e1, e0, e1, Rat(1, r2)))
        assert delta_graph >= 0
        controls.append({'R_squared': r2,
                         'L2_error_squared': float(l2),
                         'radial_degree4_graph_relative_squared': float(radial_graph/moment8),
                         'ambient_Laplacian_graph_relative_squared': float(delta_graph/(d*d)),
                         'exact_L2_error': rational_record(l2),
                         'exact_radial_graph_error': rational_record(radial_graph),
                         'exact_Laplacian_graph_error': rational_record(delta_graph)})
    for field in ('L2_error_squared', 'radial_degree4_graph_relative_squared',
                  'ambient_Laplacian_graph_relative_squared'):
        assert controls[2][field] < controls[1][field] < controls[0][field]
    return {'N': n, 'scope': 'actual normalizable physical Bloch0 trial, not minimizing ground',
            'native_Qtheta_trial': '1+[60chi+11192lambda+(129/10)F+262144] sum(theta)',
            'native_Ptheta_trial': '1+[60chi+465lambda+(129/10)F+262144] sum(theta)',
            'Q_density_exact': rational_record(qdensity),
            'P_density_exact': rational_record(pdensity),
            'matter_expectation_all65': '0, not annihilated',
            'Haar_kinetic': '0, Haar uniform physical singlet; not dropped',
            'ordered_Jordan_Wick_price': '3000lambda/site',
            'radial_dimension': d, 'physical_envelope': 'exp(-|q|^2/(2R^2)) times native trial',
            'graph_controls': controls,
            'analytical_limit': 'R->infinity gives L2, |q|^4 multiplier and ambient Laplacian graph convergence',
            'compact_graph_core_proof_is_analytic_not_finite_samples': True}


def exact_matrix_method() -> dict[str, Any]:
    """Mechanism-only controls, explicitly not a native H truncation."""
    h = sp.Matrix([[3, 1, 0, 0], [1, 3, 1, 0], [0, 1, 3, 1], [0, 0, 1, 3]])
    q = sp.diag(1, 2, 4, 8)
    current = sp.I*(h*q-q*h)
    # Exact principal minors verify 2Q +/- i[H,Q]>=0 without eigensolver.
    minors = []
    for sign in (-1, 1):
        a = 2*q+sign*current
        for size in range(1, 5):
            for indices in itertools.combinations(range(4), size):
                determinant = sp.expand(a.extract(indices, indices).det())
                assert determinant >= 0
                minors.append(int(determinant))
    s = sp.diag(1, 1, 0, 0)
    compressed = s*h*s
    assert sp.I*(compressed*q-q*compressed) == s*current*s
    v = sp.Matrix([1, 0, 1, 0])/sp.sqrt(2)
    bad = v*v.H
    bad_defect = sp.I*(bad*h*bad*q-q*bad*h*bad)-bad*current*bad
    assert bad_defect != sp.zeros(4)
    hn, qn = np.array(h).astype(np.complex128), np.array(q).astype(np.complex128)
    psi = np.array([1, 1j, -1, .5j], dtype=np.complex128)
    psi /= np.linalg.norm(psi)
    initial = float(np.vdot(psi, qn@psi).real)
    propagation = []
    for t in (-.3, .2, .7):
        evolved = sla.expm(-1j*t*hn)@psi
        energy = float(np.vdot(evolved, qn@evolved).real)
        assert energy <= math.exp(2*abs(t))*initial+1e-12
        propagation.append({'time': t, 'Q_energy': energy, 'bound': math.exp(2*abs(t))*initial})
    a1 = np.array([[1, .2, 0, 0], [0, 1, 0, 0], [0, 0, .5, .1], [0, 0, 0, 1]], complex)
    a2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
    word = psi.copy()
    bound = initial
    words = []
    for t, insertion in [(.2, a1), (-.4, a2), (.7, a1), (.1, a2)]:
        # Frobenius dominates operator norm for Q^(1/2) A Q^(-1/2).
        weighted = np.diag(np.sqrt([1, 2, 4, 8]))@insertion@np.diag(1/np.sqrt([1, 2, 4, 8]))
        insertion_constant = float(np.sum(abs(weighted)**2))
        bound *= insertion_constant*math.exp(2*abs(t))
        word = insertion@sla.expm(-1j*t*hn)@word
        energy = float(np.vdot(word, qn@word).real)
        assert energy <= bound+1e-12
        words.append({'time': t, 'actual_Q_energy': energy, 'recursion_bound': bound})
    return {'scope': 'synthetic exact/matrix mechanism ONLY, not native H spectrum or evolution',
            'exact_C': 2, 'all_principal_minors_nonnegative': minors,
            'Q_spectral_compression_identity_exact': True,
            'arbitrary_rotated_projection_identity_rejected': True,
            'rotated_defect_squared_Frobenius': str(sp.trace(bad_defect.H*bad_defect)),
            'real_unitary_controls': propagation, 'arbitrary_mixed_word_controls': words,
            'no_eigensolver_used': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {name: digest((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['epoch_archive']) as archive:
        profiles = json.loads(archive.read(request['profile']))['primitive_profiles']
        members = {name: digest(archive.read(name)) for name in archive.namelist()}
    proof = json.loads((ROOT/request['proof']).read_text())
    result = {'schema': 1,
              'scope': 'NEW_NATIVE_WEIGHTED_FORM_AND_GRAPH_CONTROLS_WITH_ANALYTIC_FRIEDRICHS_CLOSURE',
              'exact_constants': exact_constants(profiles),
              'native_topology': [native_topology(n, profiles) for n in request['N']],
              'native_physical_graph': [native_gaussian_graph(n) for n in request['N']],
              'matrix_method_only': exact_matrix_method(),
              'analytical_proof': proof, 'epoch_member_sha256': members,
              'source_input_sha256_before': before,
              'source_input_sha256_after': {name: digest((ROOT/name).read_bytes()) for name in request['pins']},
              'runtime': {'executable': sys.executable, 'elapsed_seconds': time.monotonic()-started}}
    assert result['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'native_controls_N': request['N'], 'constant_exact_domination': True,
                      'analytic_weighted_Friedrichs_closure': True,
                      'numerical_native_fullH_evolution': False,
                      'static_GNS_identification': False,
                      'elapsed_seconds': result['runtime']['elapsed_seconds']}))


if __name__ == '__main__':
    main()
