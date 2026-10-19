"""Independent finite-CAR/shared-field and rational domain/packet review.

No target scientist, producer, ground, root, checker, or source main is called.
The immutable reviewed epoch is distinct from any later scientific correction.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.sparse import bmat, csr_matrix, diags, eye, kron
from scipy.sparse.linalg import expm_multiply

import bulk_electric_phase_consumer as primitive

ROOT = Path(__file__).resolve().parent


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exclusive_json(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def car_lift(matrix: Any, masks: list[int]) -> Any:
    """Independent signed occupation action, including every local occupancy."""
    index = {mask: i for i, mask in enumerate(masks)}
    rows, cols, values = [], [], []
    for col, mask in enumerate(masks):
        for p, q in zip(*np.nonzero(matrix)):
            p, q = int(p), int(q)
            if not (mask >> q) & 1:
                continue
            sign = -1 if (mask & ((1 << q)-1)).bit_count() % 2 else 1
            intermediate = mask ^ (1 << q)
            if (intermediate >> p) & 1:
                continue
            if (intermediate & ((1 << p)-1)).bit_count() % 2:
                sign = -sign
            target = intermediate | (1 << p)
            assert target in index
            rows.append(index[target]); cols.append(col)
            values.append(sign*matrix[p, q])
    return csr_matrix((values, (rows, cols)), shape=(len(masks), len(masks)))


def generators() -> Any:
    answer = []
    for p, q in itertools.combinations(range(4), 2):
        real = np.zeros((4, 4), complex); real[p, q] = real[q, p] = .5
        imag = np.zeros((4, 4), complex); imag[p, q] = -.5j; imag[q, p] = .5j
        answer.extend((real, imag))
    for k in range(1, 4):
        answer.append(np.diag([1.]*k+[-float(k)]+[0.]*(3-k))/np.sqrt(2*k*(k+1)))
    basis = np.asarray(answer)
    assert np.max(np.abs(np.einsum('aij,bji->ab', basis, basis)-np.eye(15)/2)) < 1e-14
    return basis


def hopping(fields: Any, link: Any, coeff: Any, theta: float = 0.) -> Any:
    block = np.zeros((4, 4), complex)
    for mu in range(4):
        block += coeff[mu]*(fields[0, mu]@link+link@fields[1, mu])/2
    block *= np.exp(1j*theta)
    return np.block([[np.zeros((4, 4)), block], [block.conj().T, np.zeros((4, 4))]])


def jets(generator: Any, first: Any, second: Any, vector: Any) -> tuple[Any, Any, Any]:
    """One exact augmented ODE, with the complete second forcing."""
    n = len(vector); zero = csr_matrix((n, n), dtype=complex)
    block = bmat([[generator, zero, zero], [first, generator, zero], [second, 2*first, generator]], format='csr')
    answer = expm_multiply(block, np.concatenate((vector, np.zeros(2*n, complex))))
    return answer[:n], answer[n:2*n], answer[2*n:]


def permutation_sign(order: list[int]) -> int:
    return (-1)**sum(order[i] > order[j] for i in range(len(order)) for j in range(i+1, len(order)))


def charged_singlet(masks: list[int]) -> Any:
    index = {mask: i for i, mask in enumerate(masks)}
    n = len(masks); vector = np.zeros(n*n, complex)
    for left in itertools.combinations(range(4), 2):
        complement_left = [i for i in range(4) if i not in left]
        for right in itertools.combinations(range(4), 2):
            complement_right = [i for i in range(4) if i not in right]
            first = sum(1 << i for i in left)+sum(1 << (i+4) for i in right)
            second = sum(1 << i for i in complement_left)+sum(1 << (i+4) for i in complement_right)
            vector[n*index[first]+index[second]] = permutation_sign(list(left)+complement_left)*permutation_sign(list(right)+complement_right)/6
    assert abs(np.vdot(vector, vector)-1) < 1e-14
    return vector


def sparse_cap(matrix: Any) -> float:
    return float(np.sqrt(np.max(np.asarray(abs(matrix).sum(axis=0)))*np.max(np.asarray(abs(matrix).sum(axis=1)))))


def norm(matrix: Any) -> float:
    return float(np.linalg.norm(matrix))


def finite_controls(charged: Any, neutral: Any) -> dict[str, Any]:
    rng = np.random.default_rng(222101117)
    basis = generators()
    coordinates = .17*rng.normal(size=(2, 4, 15))
    fields = np.einsum('xma,aij->xmij', coordinates, basis)
    link = expm(1j*np.einsum('a,aij->ij', .31*rng.normal(size=15), basis))
    masks = [m for m in range(256) if m.bit_count() == 4]
    all_masks = list(range(256)); n = len(masks)
    identity = eye(n, format='csr')
    def both(matrix: Any) -> Any:
        single = car_lift(matrix, masks)
        return kron(single, identity, format='csr')+kron(identity, single, format='csr')
    rho_single = np.array([(m & 15).bit_count()-2 for m in masks])
    rho = (rho_single[:, None]+rho_single[None, :]).ravel()
    density = diags(2*rho*rho)
    def charged_generator(matrix: Any) -> Any:
        lifted = both(matrix)
        return density@lifted-lifted@density
    neutral_projector = np.diag([1.]*4+[0.]*4)
    def neutral_generator(matrix: Any) -> Any:
        return car_lift(matrix@neutral_projector-neutral_projector@matrix, all_masks)
    hc = hopping(fields, link, charged)
    hn = -hopping(fields, link, neutral)
    cg = charged_generator(hc); ng = neutral_generator(hn)
    c0 = charged_singlet(masks); n0 = np.zeros(256, complex); n0[15] = 1
    # Positive full-Fock restriction control; no tangent-only or fixed onsite n=2 truncation.
    full_h = car_lift(hc, all_masks)
    embedding = csr_matrix((np.ones(n), (masks, range(n))), shape=(256, n))
    intertwiner_error = norm((full_h@embedding-embedding@car_lift(hc, masks)).data)
    assert intertwiner_error == 0
    full_number = diags([m.bit_count() for m in all_masks])
    assert norm((full_number@full_h-full_h@full_number).data) == 0
    # Both circuits act on all0..4 endpoint occupancies in the exact canonical pair sector.
    tau, kappa = -.0078125, .025
    cv = expm_multiply(tau*cg, c0); nv = expm_multiply(kappa*ng, n0)
    assert abs(np.vdot(cv, cv)-1) < 1e-11 and abs(np.vdot(nv, nv)-1) < 1e-11
    assert norm(rho*cv) > .01
    local_occupancies = sorted(set((m & 15).bit_count() for m in masks))
    assert local_occupancies == list(range(5))
    # Local epsilon pairs are genuine singlets, despite unequal active/auxiliary masses.
    local_masks = [m for m in range(16) if m.bit_count() == 2]
    epsilon = np.zeros((6, 6), complex)
    for i, mask in enumerate(local_masks):
        complement = 15 ^ mask; j = local_masks.index(complement)
        order = [p for p in range(4) if mask >> p & 1]+[p for p in range(4) if complement >> p & 1]
        epsilon[i, j] = permutation_sign(order)/np.sqrt(6)
    singlet_error = max(norm(car_lift(t, local_masks).toarray()@epsilon+epsilon@car_lift(t, local_masks).toarray().T) for t in basis)
    assert singlet_error < 1e-13
    results = {}
    field_direction = np.zeros_like(fields); field_direction[0, 1] = basis[0]
    zero8 = np.zeros((8, 8), complex)
    directions = {
        'Cart': (hopping(field_direction, link, charged), zero8, -hopping(field_direction, link, neutral), zero8),
        'colour': (hopping(fields, 1j*basis[3]@link, charged), hopping(fields, -basis[3]@basis[3]@link, charged), -hopping(fields, 1j*basis[3]@link, neutral), -hopping(fields, -basis[3]@basis[3]@link, neutral)),
        'theta': (np.block([[np.zeros((4, 4)), 1j*hc[:4, 4:]], [-1j*hc[4:, :4], np.zeros((4, 4))]]), -hc, zero8, zero8),
    }
    largest_berry = 0.
    for name, (cdh, cd2h, ndh, nd2h) in directions.items():
        cdg = charged_generator(cdh); cd2g = charged_generator(cd2h)
        ndg = neutral_generator(ndh); nd2g = neutral_generator(nd2h)
        c, dc, d2c = jets(tau*cg, tau*cdg, tau*cd2g, c0)
        neut, dn, d2n = jets(kappa*ng, kappa*ndg, kappa*nd2g, n0)
        c_initial = complex(np.vdot(c0, cdg@c0)); n_initial = complex(np.vdot(n0, ndg@n0))
        assert c_initial == 0 and n_initial == 0
        cb, nb = complex(np.vdot(c, dc)), complex(np.vdot(neut, dn))
        cb_cap = tau*tau*sparse_cap(cg)*sparse_cap(cdg)
        nb_cap = kappa*kappa*sparse_cap(ng)*sparse_cap(ndg)
        assert abs(cb) <= cb_cap+1e-13 and abs(nb) <= nb_cap+1e-13
        largest_berry = max(largest_berry, abs(nb))
        joint = np.outer(neut, c)
        joint_first = np.outer(dn, c)+np.outer(neut, dc)
        joint_second = np.outer(d2n, c)+2*np.outer(dn, dc)+np.outer(neut, d2c)
        omitted_mixed = 2*np.outer(dn, dc)
        cross = 2*float((nb.conjugate()*cb).real)
        increment = norm(joint_first)**2-norm(dn)**2
        assert abs(increment-(norm(dc)**2+cross)) < 1e-10
        real_scalar_cross = 2*float(np.vdot(joint, joint_first).real)
        assert abs(real_scalar_cross) < 1e-10
        # Complete scalar/source Leibniz action; every mixed first/second term stays visible.
        scalar_first, scalar_second = -.37, .22
        full_action = scalar_second*joint+2*scalar_first*joint_first+joint_second
        wrong_scalar = scalar_second*joint+joint_second
        wrong_mixed = full_action-omitted_mixed
        wrong_second_norm = 0.
        if name != 'Cart':
            _, _, wrong_c2 = jets(tau*cg, tau*cdg, csr_matrix(cg.shape), c0)
            wrong_second_norm = norm(d2c-wrong_c2)
            assert wrong_second_norm > 1e-5
        if name != 'theta':
            assert norm(omitted_mixed) > 1e-6
            assert norm(full_action-wrong_mixed) > 1e-6
        assert norm(full_action-wrong_scalar) > 1e-4
        results[name] = {
            'initial_Berry_exact_by_occupation': [str(n_initial), str(c_initial)],
            'neutral_Berry': [nb.real, nb.imag], 'charged_Berry': [cb.real, cb.imag],
            'Duhamel_quadratic_caps': [nb_cap, cb_cap], 'charged_self': norm(dc)**2,
            'shared_tensor_cross': cross, 'direct_shared_increment': increment,
            'factorized_increment_error': abs(increment-(norm(dc)**2+cross)),
            'real_scalar_cross': real_scalar_cross,
            'omitted_mixed_second_action_error': norm(omitted_mixed),
            'omitted_scalar_first_Leibniz_error': norm(full_action-wrong_scalar),
            'omitted_generator_second_jet_error': wrong_second_norm,
        }
        if name == 'Cart':
            ndv = ndg@n0; contrary_input = (n0+1j*ndv/norm(ndv))/np.sqrt(2)
            contrary_initial = complex(np.vdot(contrary_input, ndg@contrary_input))
            assert abs(contrary_initial) > .01
            # Replacing the anti-Hermitian generator by iA breaks normalization and scalar cross.
            wrong_v, wrong_d, _ = jets(1j*kappa*ng, 1j*kappa*ndg, csr_matrix(ng.shape), n0)
            wrong_real_scalar = 2*float(np.vdot(wrong_v, wrong_d).real)
            assert abs(wrong_real_scalar) > 1e-5
            results['adverse_normalization'] = {'initial_Berry_when_occupation_hypothesis_removed': [contrary_initial.real, contrary_initial.imag], 'nonunitary_real_scalar_cross': wrong_real_scalar, 'nonunitary_norm': float(np.vdot(wrong_v, wrong_v).real)}
            # Gauge-invariant field phase exp(i gamma R²) keeps canonical/Gauss norm,
            # but changes the initial source jet hypothesis. It gives a NONZERO
            # shared-species cross, refuting an independence=>zero shortcut.
            gamma = .37
            phase = np.exp(1j*gamma*float(np.sum(coordinates**2)))
            phase_derivative = 2*gamma*coordinates[0, 1, 0]
            phase_c = phase*c
            phase_dc = phase*(dc+1j*phase_derivative*c)
            phase_joint_first = np.outer(dn, phase_c)+np.outer(neut, phase_dc)
            phase_cb = complex(np.vdot(phase_c, phase_dc))
            phase_cross = 2*float((nb.conjugate()*phase_cb).real)
            phase_direct = norm(phase_joint_first)**2-norm(dn)**2-norm(phase_dc)**2
            assert abs(phase_cross) > 1e-6 and abs(phase_direct-phase_cross) < 1e-10
            results['adverse_unitary_shared_cross'] = {
                'gauge_invariant_phase': 'exp(i gamma R²), alternative source only',
                'direct_shared_cross': phase_direct,
                'Berry_factorized_cross': phase_cross,
                'wrong_omitted_cross_error': abs(phase_cross),
                'charged_pointnorm': float(np.vdot(phase_c, phase_c).real),
            }
    assert largest_berry > 1e-9
    # Phase covariance verifies exact U1 Gauss and 2pi periodicity, not a neutrality surrogate.
    phase_results = []
    reference = None
    for theta in (0., .37, 1.91, 2*np.pi):
        current_g = charged_generator(hopping(fields, link, charged, theta))
        current = expm_multiply(tau*current_g, c0)
        expected = np.exp(1j*theta*rho)*cv
        error = norm(current-expected)
        assert error < 1e-10
        matrix_theta = np.block([[np.zeros((4, 4)), 1j*hopping(fields, link, charged, theta)[:4, 4:]], [-1j*hopping(fields, link, charged, theta)[4:, :4], np.zeros((4, 4))]])
        _, theta_jet, _ = jets(tau*current_g, tau*charged_generator(matrix_theta), -tau*current_g, c0)
        gauss_error = norm(theta_jet-1j*rho*current)
        assert gauss_error < 1e-10
        metrics = [float(np.vdot(current, both(hopping(fields, link, charged, theta))@current).real), norm(theta_jet)**2]
        if reference is None:
            reference = metrics
        assert np.max(np.abs(np.asarray(metrics)-reference)) < 1e-9
        phase_results.append({'theta': theta, 'covariance_error': error, 'Gauss_jet_error': gauss_error, 'selected_matter_and_fullE64': [metrics[0], 64*metrics[1]]})
    gr = expm(.11j*basis[2]); gc = expm(-.13j*basis[7])
    transformed = np.asarray([[gr@fields[0, mu]@gr.conj().T for mu in range(4)], [gc@fields[1, mu]@gc.conj().T for mu in range(4)]])
    bg = np.block([[gr, np.zeros((4, 4))], [np.zeros((4, 4)), gc]])
    su4_error = norm(hopping(transformed, gr@link@gc.conj().T, charged)-bg@hc@bg.conj().T)
    assert su4_error < 1e-12
    # An ordered two-link primitive has endpoints r,c, not the intermediate field.
    middle = expm(.23j*basis[8]); other = expm(.19j*basis[1]); gm = expm(.07j*basis[5])
    two_correct = hopping(transformed, (gr@link@gm.conj().T)@(gm@other@gc.conj().T), charged)
    two_reference = bg@hopping(fields, link@other, charged)@bg.conj().T
    two_error = norm(two_correct-two_reference)
    two_reversed_error = norm(hopping(fields, other@link, charged)-hopping(fields, link@other, charged))
    assert two_error < 1e-12 and two_reversed_error > .01
    # Nonmatching CAR hop in two distinct gates: TOTAL canonical number alone
    # is insufficient; each gate's separately conserved label is load bearing.
    masks16 = [mask for mask in range(1 << 16) if mask.bit_count() == 8]
    index16 = {mask: i for i, mask in enumerate(masks16)}
    conditional = cv.reshape((n, n)).sum(axis=1)
    conditional /= norm(conditional)
    two_gate = np.zeros(len(masks16), complex)
    for i, left in enumerate(masks):
        for j, right in enumerate(masks):
            two_gate[index16[left | (right << 8)]] = conditional[i]*conditional[j]
    cross_matrix = np.zeros((16, 16), complex)
    cross_matrix[0, 8] = cross_matrix[8, 0] = 1
    nonmatching = car_lift(cross_matrix, masks16)
    leakage = nonmatching@two_gate
    assert norm(leakage) > .1 and np.vdot(two_gate, leakage) == 0
    contrary_gate = (two_gate+leakage/norm(leakage))/np.sqrt(2)
    contrary_gate_expectation = float(np.vdot(contrary_gate, nonmatching@contrary_gate).real)
    assert abs(contrary_gate_expectation) > .1
    results['nonmatching_gate_number'] = {
        'full_global_canonical8_sector_dimension': len(masks16),
        'exact_original_expectation': str(np.vdot(two_gate, leakage)),
        'nonzero_hopping_action_norm': norm(leakage),
        'same_TOTAL_number_without_gate_labels_expectation': contrary_gate_expectation,
        'meaning': 'Conditional actual charged fibre columns remain in each gate number4. Crossgate hopping reaches orthogonal3/5 labels, at arbitrary finite circuit step.',
    }
    # Even Bloch momentum averaging cancels the electric linear shift; wrong one-sided packet does not.
    electric_base = 64*float(np.sum(rho*rho*abs(cv)**2))
    shift = .03125
    electric_plus = 64*float(np.sum((rho+shift)**2*abs(cv)**2))
    electric_minus = 64*float(np.sum((rho-shift)**2*abs(cv)**2))
    assert abs((electric_plus+electric_minus)/2-electric_base-64*shift*shift) < 1e-10
    wrong_one_sided_cross = 64*((1+shift)**2-1-shift**2)
    assert wrong_one_sided_cross == 128*shift
    return {'kind': 'INDEPENDENT_FINITE_SIGNED_CAR_AND_EXACT_BLOCK_ODE_CONTROLS', 'neutral_full_Fock_dimension': 256, 'charged_canonical_four_particle_pair_dimensions': [70, 70], 'charged_full_Fock_restriction_dimension_each': 256, 'charged_two_copy_coherent_dimension': 4900, 'full_Fock_intertwiner_error': intertwiner_error, 'local_occupancies_retained': local_occupancies, 'source_norms': [float(np.vdot(nv, nv).real), float(np.vdot(cv, cv).real)], 'singlet_error': singlet_error, 'finite_charged_rho_action': norm(rho*cv), 'actual_shared_coordinates': 120, 'colour_generators': 15, 'mixed_jet_controls': results, 'U1_phase_controls': phase_results, 'SU4_endpoint_covariance_error': su4_error, 'ordered_two_link_covariance_error': two_error, 'wrong_reversed_Wilson_error': two_reversed_error, 'Bloch_shift_even_average_error': abs((electric_plus+electric_minus)/2-electric_base-64*shift*shift), 'wrong_noncentered_Bloch_cross': wrong_one_sided_cross, 'scope': 'Finite two-copy controls enact the coherent charged source; all64 claims use separate exact multiplicities/operator caps, never numerical extrapolation.'}


def rational_review(profiles: Any, proofs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    g0, n0, m2, m4 = F(2**32), F(2**10), F(120), F(120*122)
    assert [(p['role'], p['count']) for p in profiles] == [('active', 1), ('auxiliary', 63)]
    literal = next(path['coefficients'] for path in profiles[0]['paths'] if path['axis'] == 0 and path['length'] == 1)
    coeffsum = sum(F(r)**2+F(i)**2 for r, i in literal)
    assert coeffsum < 2**10
    # Operator caps, over the whole CAR not just the initial source.
    hopping_cap_squared = F(128**2)*coeffsum
    density_cap = F(2*128**2)
    assert hopping_cap_squared < 2**32
    assert (2*density_cap)**2*hopping_cap_squared < g0**2
    for proof in proofs:
        request = proof['request']; eta = F(1, 2**request['eta_exponent'])
        chi = F(1, 2**4096); kappa = F(2**8192, 2**request['neutral_epsilon_exponent']); tau = F(request['tau'])
        # Derive counts and Hamiltonian coefficients first, then compare independent exact arithmetic.
        cart_self = 4*chi*F(120, 2)*tau**2*g0**2
        cart_cross = 4*chi*F(120, 2)*2*kappa**2*tau**2*n0**2*g0**2*m2*eta
        colour_self = F(3, 20)*chi*F(15, 2)*tau**2*g0**2*m2*eta
        colour_cross = F(3, 20)*chi*F(15, 2)*2*kappa**2*tau**2*n0**2*g0**2*m4*eta**2
        independent = {'Phi_charged_derivative': cart_self, 'Phi_neutral_charged_Berry_cross': cart_cross, 'colour_charged_derivative': colour_self, 'colour_neutral_charged_Berry_cross': colour_cross}
        target = proof['rational_increment']['positive_remainder_prices_per_site_exact']
        assert all(value == F(target[key]) for key, value in independent.items())
        assert cart_cross/cart_self == 2*kappa**2*n0**2*m2*eta
        assert colour_cross/colour_self == 2*kappa**2*n0**2*m4*eta/m2
        moments = {str(k): str(eta**k*math.prod(F(120+2*j) for j in range(k))) for k in range(1, 4)}
        leading = (4*tau+256*tau**2)*60*coeffsum*eta
        increment = leading+sum(F(v) for v in target.values())
        assert increment == F(proof['rational_increment']['increment_upper_per_site_exact']) and increment < leading/2 < 0
        rows.append({'eta_exponent': request['eta_exponent'], 'tau_exact': str(tau), 'four_field_prices_exact': {k: str(v) for k, v in independent.items()}, 'four_field_prices_match': True, 'moments_R2_R4_R6_exact': moments, 'per_site_increment_over_gain': str(increment/(-leading)), 'units': 'per site; total V times this, 120 Cart and15 SU4 per matching pair, V/2 pairs'})
    inventories = []
    for size in (4, 6, 8):
        sites = list(itertools.product(range(size), repeat=3))
        pairs = [(site, ((site[0]+1)%size, site[1], site[2])) for site in sites if sum(site)%2 == 0]
        labels = {site: i for i, pair in enumerate(pairs) for site in pair}
        assert len(labels) == size**3
        within = outside = 0
        for profile in profiles:
            for path in profile['paths']:
                for site in sites:
                    endpoint = list(site); endpoint[path['axis']] = (endpoint[path['axis']]+path['length'])%size
                    if labels[site] == labels[tuple(endpoint)]:
                        assert path['axis'] == 0 and path['length'] == 1 and sum(site)%2 == 0
                        within += profile['count']
                    else:
                        outside += profile['count']
        assert within == 64*size**3//2 and within+outside == 64*6*size**3
        inventories.append({'N': size, 'V': size**3, 'matching': size**3//2, 'canonical_per_species_and_neutral': 2*size**3, 'charged_species': 64, 'Ref4': 'inert tensor factor; no reference particle/source change', 'selected_directed_paths': within, 'nonmatching_zero_number_paths': outside})
    return {'whole_CAR_cap_squared_before_density': str(hopping_cap_squared), 'density_cap': str(density_cap), 'whole64_G0_cap_verified': True, 'independent_price_reproductions': rows, 'all_even_N_argument': 'Parity matching bijection exists for evenN. For N>=4 every length2 primitive and every other axis/negative matching orientation connects distinct gates; each per-species pair number4 is preserved by the full exponential, so the hopping expectation vanishes by an exact number eigenspace selection rule.', 'inventories': inventories, 'adverse_N2': 'length2 closes onto its source site at N2; endpoint-distinct number proof fails. The asserted N>=4 restriction is necessary.', 'no_field_freeze': 'Primitive scalar coefficients are read from archived frame; all60V realCart coordinates remain integrated with changed eta. Shared interacting Haar is not factored into independent links.'}


def packet_controls() -> dict[str, Any]:
    def bump(x: float) -> float:
        return np.exp(-1/(1-x*x)) if abs(x) < 1 else 0.
    i0 = quad(lambda x: bump(x)**2, -1, 1, epsabs=1e-12)[0]
    i1 = quad(lambda x: (2*x/(1-x*x)**2*bump(x))**2, -1, 1, epsabs=1e-11)[0]
    i2 = quad(lambda x: x*x*bump(x)**2, -1, 1, epsabs=1e-12)[0]
    coordinate_rows = []
    for width in (F(1, 32), F(1, 64)):
        kinetic = 64*i1/i0/float(width*width)
        momentum_energy = 64*float(width*width)*i2/i0
        coordinate_rows.append({'w_exact': str(width), 'coordinate_bump_64E2': kinetic, 'momentum_bump_Fourier_64E2': momentum_energy, 'source_like_coordinate_claim_upper': str(64*width*width), 'coordinate_uncertainty_lower_exact': str(16/(width*width)), 'normalized_bump_I0': i0})
        assert kinetic > float(16/(width*width)) and momentum_energy <= float(64*width*width)
    assert abs(coordinate_rows[1]['coordinate_bump_64E2']/coordinate_rows[0]['coordinate_bump_64E2']-4) < 1e-12
    # Actual native integral cycle graph: full divergence-zero dimension, not selected faces only.
    cycle_rows = []
    for size in (4, 6):
        box = primitive.IntegerBox(size); columns, chords = box.cycle_basis()
        trace = int(np.sum(columns.astype(np.int64)**2)); dimension = len(chords)
        width = F(1, 64*dimension*trace)
        momentum_bound = 64*trace*dimension*width**2
        assert momentum_bound == F(1, 64*dimension*trace)
        # If theta=B beta is used as a coordinate chart, inverse metric G^-1 obeys >=I/TrG.
        coordinate_lower = F(16*dimension, trace*width**2)
        assert coordinate_lower > momentum_bound
        assert all(not np.any(box.divergence(columns[:, j])) for j in range(dimension))
        cycle_rows.append({'N': size, 'cycle_dimension': dimension, 'TrG': trace, 'w': str(width), 'momentum_packet_64E2_upper': str(momentum_bound), 'misidentified_coordinate_packet_64E2_lower': str(coordinate_lower), 'wrong_over_claim_at_least': str(coordinate_lower/momentum_bound)})
    # Explicit nonorthonormal cycle chart: raw h(B^Ttheta) loses a Jacobian.
    test_b = np.asarray([[1., 0.], [0., 1.], [1., 1.]])
    gram = test_b.T@test_b; det = float(np.linalg.det(gram))
    assert abs(det-3) < 1e-12
    raw_norm = 1/np.sqrt(det); prefactor = det**.25
    assert abs(prefactor**2*raw_norm-1) < 1e-12
    return {'coordinate_vs_momentum_rows': coordinate_rows, 'w_inverse_square_contrary_ratio': 4., 'actual_cycle_rows': cycle_rows, 'Jacobian_contrary': {'G': gram.tolist(), 'detG': det, 'raw_h_Btranspose_theta_norm_squared': raw_norm, 'required_amplitude_prefactor': prefactor, 'corrected_norm_squared': prefactor**2*raw_norm}, 'correct_original_Hilbert_construction': 'Choose orthonormal transverse Q. Let M=B^TQ, alpha=Mx and d alpha=sqrt(detG) dx. For unitL2 real-even momentum g_w(beta), h=unitary inverseFourier(g_w), and scalar s(x)=detG^(1/4)h(Mx) has unitL2 in x. It is real Schwartz, not the coordinate compact bump. E s=B(-i partial_alpha)h times the Jacobian factor. Fourier fullE adds B beta to the fibre electric integer/longitudinal charge section; DB beta=0 preserves Gauss. Never differentiate g_w(beta) as a theta amplitude.', 'complete_shifted_action': '64 sum_e (E_e+(B beta)_e)^2 on each Bloch fibre includes E²+2(B beta).E+|B beta|² once, including the rho-dependent longitudinal section. Or in coordinate form retain s Dtheta²U+2(Dtheta s)Dtheta U+(Dtheta²s)U. Even beta weights cancel the linear shift for beta-independent fibre states; a noncentered weight cannot silently omit it.', 'increment_hypotheses': 'Both sources use the identical real normalized packet. Real scalar-gradient cross vanishes pointwise; scalar packet energy is identical and cancels. Equality with nativecompact increment needs theta-invariant trial increment in addition. This matching circuit has Uc(theta)=e^(i theta rho_r)Uc(0)e^(-i theta rho_r), initial rho0, field and onsite covariance and exact nonmatching-number zero; the independent phase panel exercises this. Unequal packets or complex scalar phases invalidate this cancellation.', 'scope': 'Corrected optional originalcover construction/domain and increment argument only; no unpriced originalcover absolute-negative admission or ground claim.'}


def verdict() -> dict[str, Any]:
    return {
        'native_compact_verdict': 'ACCEPTED_SCOPED: field-cross rational prices, exact fullGauss/canonical norm, and complete finiteN D(H) analytic action are supported under the enumerated load-bearing hypotheses. Attempt5 tiny negative gain is not a ground/phase/source admission.',
        'load_bearing_hypotheses': [
            'Same microscopic law, primitive couplings and source roles active1+aux63, full four-adjoint15 Cart basis tr(TaTb)=delta/2. eta changes the wave shape, not primitive Phi coefficients or an adopted frozen field.',
            'Even finite periodic N>=4, disjoint parity matching pairs; each neutral and each charged species canonical2V, each matching pair species number4,32 coherent local epsilon pairs; unchanged full Ref4 stays in its normalized neutral SU4-singlet sector and is acted on by identity, not replaced by a projected source mark.',
            'Both circuits are exact finite-CAR exponentials of anti-Hermitian generators, retaining all onsite occupations0..4 and full coherent/all64 density rho=sum(n_s-2). Pointwise fibre norm1, one shared60V Gaussian and the SAME interacting gauge-invariant Haar Gibbs wave, not independent seas/links.',
            'Initial neutral and charged derivative generators change original local species occupation labels, so the initial Berry expectation is zero. Neutrality or a C0/source-bound assertion does not replace this selection rule.',
            'Duhamel unitarity, not a small pointwise field assumption, gives |Berry|<=t²||A||||A_d||. Shared tensor first cross=2Re(conj(Berry_n)Berry_c), quadratic in EACH strength; counts120 Cart and15 SU4 perpair and coefficients4chi and3chi/20.',
            'Scalar Gaussian and compact colour wave amplitudes are real; pointnorm derivative is imaginary because U†DU anti-Hermitian. Only that yields real scalar-gradient cross0, not species neutrality. Neutral self/action cancels by tensor factor norm1, charged onsite stays paid.',
            'Native U1 source is the normalized constant periodic beta0 state, integer full electric representation with Gauss-longitudinal HC already included in fullE64; selected covariant exp(i theta) circuit has exact2pi periodicity. Originalcover packet is a DIFFERENT construction requiring the correction below.',
            'Complete second jets retain D²A insertion and both ordered DA insertions, allneutral/charged/crosscopy terms and scalar first/second Leibniz terms. At finiteN all scalar potentials, matter, derivatives and compact coefficient ratios are bounded by finite polynomials in60V Gaussian coordinates with finite moments; cutoff approximation places the source in D(H).',
        ],
        'Duhamel_proof': 'U†DU=t integral_0^1 e^(-stA) A_d e^(stA) ds. Initial expectation of A_d is0 by occupation labels. Integrating the commutator difference gives |Berry|<=|t| integral_0^1 (s|t| 2||A||||A_d||) ds=t²||A||||A_d||. This is finite-step on every field. For colour both norms carry R, hence cross moment R4; Cart derivative constant, hence R2.',
        'domain_proof': 'f_eta is normalized by (2pi eta)^(-15V), |f|² varianceeta in60V real coordinates; df/f=-q/(2eta), d²f/f=q²/(4eta²)-1/(2eta). Haar normalization is exact integral0<Z<=1 (Z>=exp(-24tV)), never numerical. Compact colour derivative ratios bounded24tV and48tV+576t²V². Circuit first/second caps polynomialdegree<=2; Jordandegree4,alignmentdegree2,matterdegree1; E R_all^(2k)=eta^k product_j(60V+2j). Complete H action L2 norm is a finite triangle sum for each finiteN; cutoff core approximation, not a claim D(H²), numerical residual/variance or N-uniform total variance.',
        'finding': {
            'severity': 'SCOPED_MATHEMATICAL_ERROR_IN_OPTIONAL_ORIGINAL_COVER_DESCRIPTION',
            'witness': 'physical_singlet_circuit_source_20261011.py:286 uses |B beta|² to price64E² for compact g_w(beta); physical_charged_descent_20261011.py:326,329 differentiates that bump in theta as if beta were coordinate. Both meanings cannot hold simultaneously. Prior bulk_electric_phase_consumer.py:228-236 and raw-flux boundary receipt use continuous electric momentum packets.',
            'trigger': 'Interpret g_w(beta) literally as coordinate compact bump in the theta differential action, while retaining the small electric bound1/(64dTrG).',
            'consequence': 'Kinetic energy grows as w^-2, not w²; current originalcover absolute price or its domain formula is not established as written. Raw h(B^Ttheta) also lacks the detG^(1/4) normalization Jacobian. Nativecompactbeta0 and its independent tiny sameH bound are unaffected.',
            'smallest_correction': 'Specify beta as electric cycle momentum and use its normalized real-even Fourier/Bloch transform in the actual transverse coordinates, with Jacobian. Retain the full shifted electric action and all envelope Leibniz terms; no beta derivative. Price added scalar energy separately in any absolute originalcover statement. Preserve the original reviewed epoch and produce a corrected source epoch, not a silent receipt overwrite.',
        },
        'not_claimed': ['D(H²)', 'numerical fullHvariance', 'stationary ground', 'fullPhi lifted selectedlocal or SO5 cancellation', 'phase/IR/pole/commoncone', 'originalSOURCE admission', 'campaign closure', 'all64 numeric exponential from two-copy controls'],
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); request_path = ROOT/args.request
    request = json.loads(request_path.read_bytes())
    raw = {name: (ROOT/name).read_bytes() for name in request['pins']}
    assert all(digest(raw[name]) == expected for name, expected in request['pins'].items())
    frame, scale = primitive.frozen.exact_frame(raw[primitive.INPUTS[0]])
    cfg = json.loads(raw[primitive.INPUTS[1]])['input_descriptor']['original_micro_config_UNCHANGED']
    assert cfg['a'] == 1/64 and cfg['auxiliary_count'] == 63
    profiles = primitive.frozen.profiles(cfg, frame, scale)
    band = primitive.frozen.RotationBand(a=cfg['a'], m=cfg['heavy_m'], r=cfg['heavy_r'], v=cfg['heavy_v'], improved=False)
    _, neutral_paths = primitive.frozen.scalar_paths(band)
    assert not band.improved and len(neutral_paths) == 3
    literal = next(path['coefficients'] for path in profiles[0]['paths'] if path['axis'] == 0 and path['length'] == 1)
    charged = np.asarray([complex(float(F(r)), float(F(i))) for r, i in literal])
    neutral = next(coeff for displacement, coeff in neutral_paths if displacement[0] == 1)
    proofs = [json.loads(raw[name]) for name in request['proofs']]
    result = {'request': request, 'source_input_hashes': {name: digest(data) for name, data in raw.items()}, 'actual_primitive_onsites': [{'role': p['role'], 'count': p['count'], 'onsite': str(p['onsite']/scale)} for p in profiles], 'rational_review': rational_review(profiles, proofs), 'finite_controls': finite_controls(charged, neutral), 'originalcover_controls': packet_controls(), 'verdict': verdict(), 'runtime': {'python': sys.version, 'numpy': np.__version__}, 'target_main_or_old_producer_called': False, 'natural_exit': 'external exclusive execution runner records actual exit'}
    assert all((ROOT/name).read_bytes() == data for name, data in raw.items())
    exclusive_json(ROOT/args.output, result)
    print(json.dumps({'native_compact_verdict': result['verdict']['native_compact_verdict'], 'optional_originalcover_finding': result['verdict']['finding']['severity'], 'price_matches': [r['four_field_prices_match'] for r in result['rational_review']['independent_price_reproductions']], 'w_inverse_square_contrary_ratio': result['originalcover_controls']['w_inverse_square_contrary_ratio']}))


if __name__ == '__main__':
    main()
