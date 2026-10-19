"""New exact native insertion costs and full noncommuting history controls.
Native fullH evolution/ground numerics are deliberately not computed.
"""
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

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact_record(x: F) -> dict[str, Any]:
    return {'sha256': sha(str(x).encode()), 'numerator_bits': x.numerator.bit_length(),
            'denominator_bits': x.denominator.bit_length()}


def native_constants(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    lam, chi, fast = F(2**8192), F(1, 2**4096), F(2**10240)
    w = [abs(F(p['onsite'])) + 2*sum((abs(F(r['c0'])) + abs(F(r['ci']))
                                     for r in p['paths']), F(0)) for p in profiles]
    assert [p['load'] for p in profiles] == [1, 1, 63]
    assert len({p['onsite'] for p in profiles}) == 3
    b = 4*(lam*w[0] + w[1] + 63*w[2])
    assert b < 512*lam
    kp, kh, ke = 4*chi, F(3, 20)*chi, F(64)
    z = 4096 + 2/lam
    jet = 2**16*(1+fast+b)
    cstar = 2**32*65*(1+jet**2)*z
    r = 60*chi+11192*lam+F(129, 10)*fast+262144
    b0 = 33*(1+10*r*4096)+16384*lam*4096
    # Declared support/jet budgets, not measurements of the minimizing family.
    # tau>=2^-ceil(R/4). The rational overbound pays the exact radical c_A.
    rows = []
    for radius, m, a, d1, d2 in [(2, 3, F(2), (F(3), F(4), F(2)), (F(8), F(7), F(2))),
                                (8, 5, F(3), (F(5), F(6), F(4)), (F(12), F(15), F(8)))]:
        tau_inv = F(2**((radius+3)//4))
        d = sum((k*j*j for k, j in zip((kp, kh, ke), d1)), F(0))
        # sqrt(k/tau)<=max(1,k)/tau for tau<=1;
        # m^.75 (2*tau^-1/lambda)^.25<=m for fixed displayed supports.
        assert lam >= 2*tau_inv
        c = sum((k*j2+2*max(F(1), k)*tau_inv*j1
                 for k, j1, j2 in zip((kp, kh, ke), d1, d2)), F(0)) + 4*a*b*m*m
        rows.append({'radius_from_weight_centre': radius, 'support_sites': m,
                     'tau_inverse_rational_upper': str(tau_inv), 'a': str(a),
                     'D1': [str(v) for v in d1], 'D2': [str(v) for v in d2],
                     'insertion_Q_coefficient': str(66*a*a),
                     'insertion_norm_coefficient': exact_record(66*d+16384*lam*4096*a*a),
                     'commutator_rational_upper': exact_record(c),
                     'flow_difference_cost_squared_per_unit_edge': str(tau_inv/64)})
    # Formal positive-polynomial recursion retains exp(Cstar |dt|) variables.
    # Integer overbudget for g^2 is 2a^2*g^2+2c_A^2*E (no massive exp rounded).
    u, v = sp.symbols('u v', nonnegative=True)
    energy: Any = sp.Integer(b0.numerator)/b0.denominator
    graph2: Any = sp.Integer(0)
    norm2: Any = sp.Integer(1)
    for factor, row in zip((u, v, u, v), (rows[0], rows[1], rows[0], rows[1])):
        a = F(row['a']); m = int(row['support_sites']); tau_inv = F(row['tau_inverse_rational_upper'])
        d1 = tuple(F(x) for x in row['D1']); d2 = tuple(F(x) for x in row['D2'])
        d = sum((k*j*j for k, j in zip((kp, kh, ke), d1)), F(0))
        c = sum((k*j2+2*max(F(1), k)*tau_inv*j1
                 for k, j1, j2 in zip((kp, kh, ke), d1, d2)), F(0))+4*a*b*m*m
        energy = factor*energy
        graph2 = sp.expand(2*sp.Rational(a*a)*graph2+2*sp.Rational(c*c)*energy)
        energy = sp.expand(66*sp.Rational(a*a)*energy+sp.Rational(66*d+16384*lam*4096*a*a)*norm2)
        norm2 *= sp.Rational(a*a)
    ep, gp = sp.Poly(energy, u, v), sp.Poly(graph2, u, v)
    assert all(x > 0 for x in ep.coeffs()) and all(x > 0 for x in gp.coeffs())
    return {'scope': 'native exact coefficient and declared domain-budget arithmetic',
            'B': exact_record(b), 'B0': exact_record(b0), 'Cstar': exact_record(cstar),
            'loads': [1, 1, 63], 'kinetic_coefficients': [exact_record(k) for k in (kp, kh, ke)],
            'insertion_costs': rows, 'mixed_word_recursion_length': 4,
            'Q_polynomial_positive_terms': len(ep.terms()), 'graph2_polynomial_positive_terms': len(gp.terms()),
            'Q_polynomial_sha256': sha(str(ep.as_expr()).encode()),
            'graph2_polynomial_sha256': sha(str(gp.as_expr()).encode()),
            'uniform_N_or_character_in_constants': False,
            'u_v_mean_exp_Cstar_times_not_numerical_native_evolution': True}


def car_transfer(mask: int, create: int, annihilate: int) -> tuple[int, int] | None:
    if not ((mask >> annihilate) & 1):
        return None
    sign = (-1)**((mask & ((1 << annihilate)-1)).bit_count())
    mask ^= 1 << annihilate
    if (mask >> create) & 1:
        return None
    sign *= (-1)**((mask & ((1 << create)-1)).bit_count())
    return mask | (1 << create), sign


def native_products_and_jets(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    # Actual complete coefficient tensor, not a gauge-invariant single entry.
    q: Any = sp.Symbol('q', real=True)
    a: Any = sp.Symbol('a', real=True)
    t: Any = sp.diag(sp.Rational(1, 2), -sp.Rational(1, 2), 0, 0)
    phi_x: Any = q*t
    phi_y: Any = (q+1)*t
    u1: Any = sp.Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    u2: Any = sp.diag(sp.I, -sp.I, 1, 1)
    assert sp.trace(t*t) == sp.Rational(1, 2) and u1*u2 != u2*u1
    gx, gy = u1, u2
    rows = []
    for profile in profiles:
        charged = profile['scale'] == 'charged'
        scale = -sp.Integer(2**8192) if not charged else sp.Integer(1)
        for row in profile['paths']:
            c0, ci = sp.Rational(row['c0']), sp.Rational(row['ci'])
            # Phi0=Phi_(axis+1)=the displayed independent-field slice;
            # other Phi components remain available in the actual domain.
            fx, fy = (c0+sp.I*ci)*phi_x, (c0+sp.I*ci)*phi_y
            length = int(row['length']); path = u1 if length == 1 else u1*u2
            coeff = scale*(fx*path+path*fy)/2
            transformed = scale*((gx*fx*gx.conjugate().T)*(gx*path*gy.conjugate().T)
                                  +(gx*path*gy.conjugate().T)*(gy*fy*gy.conjugate().T))/2
            assert sp.simplify(transformed-gx*coeff*gy.conjugate().T) == sp.zeros(4)
            # Ordered Haar derivatives at FIRST link; second position differs.
            dpath, ddpath = sp.I*t*path, -t*t*path
            dcoeff = scale*(fx*dpath+dpath*fy)/2
            ddcoeff = scale*(fx*ddpath+ddpath*fy)/2
            assert dcoeff != sp.zeros(4) and ddcoeff != sp.zeros(4)
            if length == 2:
                second = sp.I*u1*t*u2
                assert second != dpath
            phase = sp.exp(sp.I*a) if charged else sp.Integer(1)
            tensor = phase*coeff
            assert sp.simplify(sp.diff(tensor, a)-(sp.I*tensor if charged else sp.zeros(4))) == sp.zeros(4)
            assert sp.simplify(sp.diff(tensor, a, 2)-(-tensor if charged else sp.zeros(4))) == sp.zeros(4)
            # Smooth invariant radial envelope: bounded Phi/first/second jets.
            envelope = sp.exp(-(q*q+(q+1)**2))
            bounded = envelope*tensor
            assert sp.simplify(sp.diff(bounded, q)-(sp.diff(envelope, q)*tensor+envelope*sp.diff(tensor, q))) == sp.zeros(4)
            assert sp.simplify(sp.diff(bounded, q, 2)-(sp.diff(envelope, q, 2)*tensor
                    +2*sp.diff(envelope, q)*sp.diff(tensor, q))) == sp.zeros(4)
            # Full two-endpoint invariant f=exp[-sum_mu(||Phi_mu,x||HS^2+
            # ||Phi_mu,y||HS^2)] gives the displayed envelope on this slice.
            # 16 CAR entries and elementary sup(T^j exp(-T)) pay the global
            # norm/jet costs, not just the chosen tensor values.
            load_coeff = F(abs(int(scale)))*int(profile['load'])*(abs(F(row['c0']))+abs(F(row['ci'])))
            anorm = 32*load_coeff
            d1 = (768*load_coeff, 192*load_coeff, 64*load_coeff if charged else F(0))
            d2 = (30720*load_coeff, 960*load_coeff, 64*load_coeff if charged else F(0))
            kinetic = (F(4, 2**4096), F(3, 20*2**4096), F(64))
            dcost = sum((k*j*j for k, j in zip(kinetic, d1)), F(0))
            local_c = sum((k*j2+4*max(F(1), k)*j1
                           for k, j1, j2 in zip(kinetic, d1, d2)), F(0))
            # B<512lambda; radius<=4 =>tau^-1<=2, m<=26.
            local_c += 4*anorm*F(512*2**8192)*26*26
            rows.append({'role': profile['role'], 'load': profile['load'], 'axis': row['axis'],
                         'length': length, 'coefficient_sha256': sha(str(tensor).encode()),
                         'ordered_Haar_first_second_paid': True, 'fullE_first_second_paid': True,
                         'Phi_product_first_second_paid': True,
                         'actual_global_smooth_insertion_norm_upper': exact_record(anorm),
                         'actual_global_D1_upper': [exact_record(z) for z in d1],
                         'actual_global_D2_upper': [exact_record(z) for z in d2],
                         'actual_global_weighted_jet_cost': exact_record(dcost),
                         'actual_commutator_upper': exact_record(local_c),
                         'support_enlargement_sites_upper': 26, 'weight_centre_radius_upper': 4})
    # Same canonical active1 sector with remote spectators: sparse CAR products.
    # Each local basis has one mobile colour, 2V-1 fixed remote spectators.
    local_masks = [1 << j for j in (0, 4, 8)]
    matrices: list[Any] = []
    for create, annihilate in ((0, 4), (4, 8)):
        mat: Any = sp.zeros(3)
        for column, mask in enumerate(local_masks):
            action = car_transfer(mask, create, annihilate)
            if action is not None:
                state, sign = action
                mat[local_masks.index(state), column] = sign
        matrices.append(mat)
    x, y = matrices
    assert x*y != y*x and x*y == sp.Matrix([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    p: Any = sp.diag(1, 0, 1)
    assert p*x*y*p != p*x*p*y*p
    f: Any = sp.diag(1, 2, 3)
    assert f*x != x*f
    assert f*x*y == (f*x)*y and (x*y).conjugate().T == y.conjugate().T*x.conjugate().T
    # Closed plaquette h has h.path01=pi/2. Its shift of the reverse
    # Peierls multiplier contributes -i, retaining the actual CAR order.
    mx = sp.exp(sp.I*a)*x
    shifted_y = sp.exp(-sp.I*(a+sp.pi/2))*x.conjugate().T
    twisted = sp.simplify(mx*shifted_y)
    wrong = sp.simplify(mx*(sp.exp(-sp.I*a)*x.conjugate().T))
    assert sp.simplify(twisted+sp.I*wrong) == sp.zeros(3) and twisted != wrong
    return {'scope': 'native complete gauge-covariant tensor/ambient canonical CAR coefficient controls, not a ground or evolution',
            'tensor_jet_rows': rows, 'all65_load_sum': sum(int(p['load']) for p in profiles),
            'remote_spectators_per_species': '2V-1 for displayed active component; all other species exactly2V',
            'single_entry_not_claimed_gauge_invariant_state': True,
            'CAR_noncommutation_exact': str(x*y-y*x),
            'compression_product_defect_exact': str(p*x*y*p-p*x*p*y*p),
            'field_CAR_noncommutation_exact': str(f*x-x*f),
            'closed_Weyl_twisted_product_phase': '-i, not1',
            'compressed_products_and_omitted_Weyl_shift_rejected': True}


def spatial_gauss_ap(n: int) -> dict[str, Any]:
    sites = list(itertools.product(range(n), repeat=3)); index = {x: i for i, x in enumerate(sites)}
    v = n**3
    def move(x: tuple[int, ...], axis: int) -> tuple[int, ...]:
        z = list(x); z[axis] = (z[axis]+1) % n; return tuple(z)
    # Exact signed AP permutation commutes in the three axes; S_i^N=-I.
    ap_rows = 0
    for x in sites:
        for axis in range(3):
            z = x; sign = 1
            for _ in range(n):
                sign *= -1 if z[axis] == n-1 else 1; z = move(z, axis)
            assert z == x and sign == -1 and (-1)**(2*v) == 1
            for other in range(3):
                left = (-1 if x[axis] == n-1 else 1)*(-1 if move(x, axis)[other] == n-1 else 1)
                right = (-1 if x[other] == n-1 else 1)*(-1 if move(x, other)[axis] == n-1 else 1)
                assert move(move(x, axis), other) == move(move(x, other), axis) and left == right
            for length in (1, 2):
                z = x; signed = 1
                for _ in range(length):
                    signed *= -1 if z[axis] == n-1 else 1; z = move(z, axis)
                assert signed == (-1 if x[axis]+length >= n else 1)
                ap_rows += 1
    # Actual collective64 canonical occupation backgrounds, integer raw Gauss.
    rho = [128*((-1)**sum(x)) for x in sites]
    edges = {(x, axis): 0 for x in sites for axis in range(3)}
    for x in sites:
        edges[x, 0] = 64*((-1)**sum(x))
    def divergence(k: dict[tuple[tuple[int, ...], int], int]) -> list[int]:
        d = [0]*v
        for (x, axis), value in k.items():
            d[index[x]] += value; d[index[move(x, axis)]] -= value
        return d
    assert divergence(edges) == rho and sum(rho) == 0
    # Translation intertwines D and rho; affine D*Lplus covariance follows
    # because site/edge permutations commute DD* and its zero-sum inverse.
    checks = 0
    for axis in range(3):
        translated = {(move(x, axis), direction): value for (x, direction), value in edges.items()}
        translated_rho = [0]*v
        for x, value in zip(sites, rho): translated_rho[index[move(x, axis)]] = value
        assert divergence(translated) == translated_rho
        checks += 1
    # Closed raw plaquette flow; complete weighted fullE square retains cross.
    o = (0, 0, 0); ex, ey = move(o, 0), move(o, 1)
    h = {key: 0 for key in edges}; h[o, 0]=1; h[ex, 1]=1; h[ey, 0]=-1; h[o, 1]=-1
    assert not any(divergence(h))
    transfer = {key: 0 for key in edges}; transfer[o, 0]=-1
    drho = divergence(transfer)
    assert drho[index[o]] == -1 and drho[index[ex]] == 1
    active_mask = sum(((1 << (4*index[x]+colour))
                       for x in sites if sum(x) % 2 == 0 for colour in range(4)), 0)
    action = car_transfer(active_mask, 4*index[ex], 4*index[o])
    assert action is not None
    moved_mask, car_sign = action
    assert active_mask.bit_count() == moved_mask.bit_count() == 2*v
    active_delta = [sum((moved_mask >> (4*i+c)) & 1 for c in range(4))-
                    sum((active_mask >> (4*i+c)) & 1 for c in range(4)) for i in range(v)]
    assert active_delta == drho
    assert divergence({key: value+transfer[key] for key, value in edges.items()}) == [
        value+shift for value, shift in zip(rho, drho)]
    zz: Any = sp.Symbol('z', real=True)
    def theta(x: tuple[int, ...]) -> Any: return zz**sum(min(t, n-t) for t in x)
    delta: Any = sum((64*theta(x)*(2*value*transfer[x, axis]+transfer[x, axis]**2)
                     for (x, axis), value in edges.items()), sp.Integer(0))
    assert delta == -8128
    affine: dict[str, Any] = {'scope': 'analytic D*Lplus covariance; no numeric inverse for this N'}
    if n == 4:
        edge_list = list(edges)
        incidence: Any = sp.zeros(v, 3*v)
        for column, (x, axis) in enumerate(edge_list):
            incidence[index[x], column] = 1
            incidence[index[move(x, axis)], column] = -1
        laplacian = incidence*incidence.T
        grounded = laplacian[:v-1, :v-1]
        potential: Any = sp.zeros(v, 1)
        potential[:v-1, 0] = grounded.inv(method='DM')*sp.Matrix(drho[:v-1])
        tl = incidence.T*potential
        raw: Any = sp.Matrix([transfer[key] for key in edge_list])
        tt = raw-tl
        assert incidence*tt == sp.zeros(v, 1)
        assert tl[edge_list.index((o, 0))] == -sp.Rational(21, 64)
        exterior = [i for i, value in enumerate(raw) if value == 0 and tl[i] != 0]
        assert exterior and all(tt[i] == -tl[i] for i in exterior)
        # Induced covariant first/second jets retain occupation-connection
        # change t_L, while frozen ordinary jets use only t_T.
        native_jet2 = sum((theta(x)*(tt[i]+tl[i])**2
                          for i, (x, _) in enumerate(edge_list)), sp.Integer(0))
        frozen_jet2 = sum((theta(x)*tt[i]**2
                          for i, (x, _) in enumerate(edge_list)), sp.Integer(0))
        defect = sp.rem(sp.expand(frozen_jet2-native_jet2), 2*zz**4-1, zz)
        assert native_jet2 == 1 and defect != 0
        affine = {'scope': 'native exact N4 raw/affine charged coefficient, not a ground calculation',
                  'longitudinal_literal_edge': '-21/64', 'transverse_literal_edge': '-43/64',
                  'exterior_edges_cancelling_first_and_second_jets': len(exterior),
                  'complete_weighted_second_jet': '1',
                  'frozen_second_jet_defect_mod_native_weight': str(defect),
                  'frozen_connection_rejected': True, 'once_collective_rho_no_64_Coulomb_copies': True}
    # A translated patch has equal diameter but different distance costs.
    centre_distance = sum(min(t, n-t) for t in sites[-1])
    return {'N': n, 'canonical_particles_each_species': 2*v, 'Ref4_inert': True,
            'AP_path_checks': ap_rows, 'signed_AP_Nth_power_canonical_identity': True,
            'commuting_AP_translations': True, 'D_translation_intertwining_checks': checks,
            'collective64_raw_Gauss_exact': True, 'closed_plaquette_Dh_zero': True,
            'full_raw_weighted_E_delta': str(delta),
            'allowed_active1_CAR_transfer_sign': car_sign,
            'canonical_active_transfer_changes_fullrho_once': True,
            'native_affine_second_jets': affine,
            'centre_distance_not_patch_diameter': centre_distance,
            'affine_Bloch_character_transport': 'pullback by raw edge permutation; fixed fibre invariance not assumed',
            'native_ground_or_fullH_evolution_computed': False}


def psd_by_minors(matrix: Any) -> list[str]:
    values = []
    for size in range(1, matrix.rows+1):
        for indices in itertools.combinations(range(matrix.rows), size):
            det: Any = sp.simplify(matrix.extract(indices, indices).det())
            assert det >= 0, (indices, det)
            values.append(str(det))
    return values


def method_full_histories(depth: int) -> dict[str, Any]:
    """Exact non-native finite matrices; never called a native fullH truncation."""
    l: Any = sp.diag(0, 1, 2)
    q: Any = sp.Matrix([[3, 1, 0], [1, 4, 1], [0, 1, 6]])
    omega: Any = sp.Matrix([1, 0, 0])
    current = sp.I*(l*q-q*l)
    assert l*omega == sp.zeros(3, 1)
    minors = psd_by_minors(q)
    for sign in (-1, 1): minors += psd_by_minors(4*q+sign*current)
    a1: Any = sp.Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    a2: Any = sp.diag(1, -1, sp.Rational(1, 2))
    a3: Any = sp.Matrix([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
    assert a1*a2 != a2*a1
    letters: list[tuple[Any, int, Any, Any]] = []
    insertion_rows = []
    for a in (a1, a2, a3):
        norm2 = sum((sp.simplify(abs(z)**2) for z in a), sp.Integer(0))
        comm = l*a-a*l
        psd_by_minors(32*q-a.conjugate().T*q*a)
        psd_by_minors(64*q-comm.conjugate().T*comm)
        insertion_rows.append({'a2_upper': str(norm2), 'Q_insertion_upper': 32, 'commutator_squared_upper': 64})
        for time_index in (0, 1):
            # t=0 or pi/2, exp(+itL) is exact diag(1,i,-1).
            phase: Any = sp.diag(1, sp.I**time_index, (-1)**time_index)
            alpha = phase*a*phase.conjugate().T
            letters.append((alpha, time_index, norm2, a))
    total = 0; transcript = hashlib.sha256(); distinct = set()
    for size in range(depth+1):
        for word in itertools.product(range(len(letters)), repeat=size):
            operator: Any = sp.eye(3)
            energy: Any = (omega.conjugate().T*q*omega)[0]
            graph2: Any = sp.Integer(0); norm2: Any = sp.Integer(1)
            # Process complete evolved letters: U(-t) A U(t); each Q
            # propagator has upper3^7 when |t|=pi/2 because exp(2pi)<3^7.
            for letter_id in word:
                alpha, tid, insertion_norm2, _ = letters[letter_id]
                growth = 3**7 if tid else 1
                energy *= growth
                graph2 = 2*insertion_norm2*graph2+128*energy
                energy *= 32*growth
                norm2 *= insertion_norm2
                operator = alpha*operator
            vector = operator*omega
            actual_norm2 = (vector.conjugate().T*vector)[0]
            actual_energy = (vector.conjugate().T*q*vector)[0]
            actual_graph2 = ((l*vector).conjugate().T*(l*vector))[0]
            assert sp.simplify(norm2-actual_norm2) >= 0
            assert sp.simplify(energy-actual_energy) >= 0
            assert sp.simplify(graph2-actual_graph2) >= 0
            # Genuine full products on both sides: null-left-ideal inequality.
            for b in (a1, a2, a3):
                bnorm2 = sum((abs(z)**2 for z in b), sp.Integer(0))
                assert sp.simplify(bnorm2*actual_norm2-((b*vector).conjugate().T*(b*vector))[0]) >= 0
            transcript.update(str((word, tuple(vector), actual_energy, actual_graph2)).encode())
            distinct.add(tuple(vector)); total += 1
    # Positive feature compression does not preserve products: common whole
    # word X Y yields a transition through an omitted intermediate state.
    x: Any = sp.zeros(3); x[0, 1] = 1
    y: Any = sp.zeros(3); y[1, 2] = 1
    p: Any = sp.diag(1, 0, 1)
    assert p*x*y*p != p*x*p*y*p
    # Whole ground block is NOT mean subtraction in a degenerate ground space.
    degenerate: Any = sp.diag(0, 0, 1)
    extra_ground: Any = sp.Matrix([0, 1, 0])
    p0: Any = sp.diag(1, 1, 0)
    mean: Any = omega*omega.conjugate().T
    assert degenerate*extra_ground == sp.zeros(3, 1)
    assert (sp.eye(3)-p0)*extra_ground == sp.zeros(3, 1)
    assert (sp.eye(3)-mean)*extra_ground == extra_ground
    return {'scope': 'NON_NATIVE_EXACT_METHOD_ONLY_NO_NATIVE_SPECTRUM',
            'positive_current_principal_minors': minors,
            'whole_words_exercised': total, 'max_length': depth, 'distinct_word_vectors': len(distinct),
            'full_word_transcript_sha256': transcript.hexdigest(), 'insertion_controls': insertion_rows,
            'exact_graph_form_norm_and_left_multiplication_controls': True,
            'noncommuting_letters': True, 'positive_feature_compressed_product_rejected': True,
            'rankone_mean_is_not_wholeP0_rejected': True, 'eigensolver_or_native_time_backend_used': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    request = json.loads(args.request.read_text())
    before = {name: sha((ROOT/name).read_bytes()) for name in request['pins']}
    assert before == request['pins']
    with zipfile.ZipFile(ROOT/request['epoch_archive']) as archive:
        raw = archive.read(request['profile_member'])
        profiles = json.loads(raw)['primitive_profiles']
        profile_hash = sha(raw)
    proof = json.loads((ROOT/request['proof']).read_text())
    receipt = {'schema': 1, 'kind': 'NEW_FULL_NATIVE_HISTORY_COST_AND_WHOLE_WORD_CONSUMER',
               'native_exact_constants': native_constants(profiles),
               'native_full_products_jets': native_products_and_jets(profiles),
               'native_spatial_Gauss_AP': [spatial_gauss_ap(n) for n in request['N']],
               'method_only_whole_histories': method_full_histories(int(request['method_depth'])),
               'analytical_native_construction': proof,
               'profile_member_sha256': profile_hash,
               'source_input_sha256_before': before,
               'source_input_sha256_after': {name: sha((ROOT/name).read_bytes()) for name in request['pins']},
               'runtime': {'executable': sys.executable, 'elapsed_seconds': time.monotonic()-started}}
    assert receipt['source_input_sha256_after'] == before
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'native_N': request['N'],
                      'whole_method_words': receipt['method_only_whole_histories']['whole_words_exercised'],
                      'full_native_history_analytic_conditional': True,
                      'numerical_native_fullH_evolution': False,
                      'same_static_GNS_invariant_or_onto': False,
                      'elapsed_seconds': receipt['runtime']['elapsed_seconds']}))


if __name__ == '__main__':
    main()
