"""Full-field neutral matching and charged density-assisted canonical trial.

The physical source is an exact finite-CAR operator expression, not a dense
N4 simulation. Dense/sparse controls below enact actual CAR in independent
small sectors. All-N prices are rational analytic inequalities; floats are
never used as a physical-field integral or a stationarity certificate.
"""
from __future__ import annotations

from fractions import Fraction as Q
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags, bmat, eye, kron
from scipy.sparse.linalg import expm_multiply

import bulk_electric_phase_consumer as primitive

ROOT = Path(__file__).resolve().parent
PREFIX = 'physical_singlet_circuit_epoch4_'
PINS = {
    'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz': 'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json': '5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354',
    'joined_phi_u1_uniform_sharp_gauss_proof_20261010.json': 'fb99f8b2accadee1f868922a972d0dbbb46dacc06b88fef61dfe4bc8e3d44dfb',
}


def car(mask: int, p: int, q: int) -> tuple[int, int] | None:
    if not mask & (1 << q):
        return None
    sign = (-1)**((mask & ((1 << q)-1)).bit_count())
    mask ^= 1 << q
    if mask & (1 << p):
        return None
    sign *= (-1)**((mask & ((1 << p)-1)).bit_count())
    return mask | (1 << p), sign


def lift(h: Any, masks: list[int]) -> Any:
    index = {m: i for i, m in enumerate(masks)}
    rr, cc, vv = [], [], []
    for j, mask in enumerate(masks):
        for p, q in zip(*np.nonzero(h)):
            moved = car(mask, int(p), int(q))
            if moved is not None:
                rr.append(index[moved[0]])
                cc.append(j)
                vv.append(moved[1]*h[p, q])
    return csr_matrix((vv, (rr, cc)), shape=(len(masks), len(masks)))


def su4_basis() -> Any:
    rows = []
    for i in range(4):
        for j in range(i+1, 4):
            a = np.zeros((4, 4), complex)
            a[i, j] = a[j, i] = .5
            rows.append(a)
            a = np.zeros((4, 4), complex)
            a[i, j], a[j, i] = -.5j, .5j
            rows.append(a)
    for k in range(1, 4):
        a = np.diag([1.]*k+[-float(k)]+[0.]*(3-k))
        rows.append(a/np.sqrt(2*k*(k+1)))
    return np.array(rows)


def hop(fields: Any, u: Any, coefficients: Any, theta: float = 0.) -> Any:
    t = sum(coefficients[mu]*(fields[0, mu]@u+u@fields[1, mu])/2 for mu in range(4))
    t *= np.exp(1j*theta)
    return np.block([[np.zeros((4, 4)), t], [t.conj().T, np.zeros((4, 4))]])


def jet_exponential(a: Any, da: Any, d2a: Any, v: Any) -> tuple[Any, Any, Any]:
    """Exact block-Duhamel first/second jet, no finite-difference derivative."""
    n = len(v)
    z = csr_matrix((n, n), dtype=complex)
    block = bmat([[a, z, z], [da, a, z], [d2a, 2*da, a]], format='csr')
    out = expm_multiply(block, np.concatenate([v, np.zeros(2*n, complex)]))
    return out[:n], out[n:2*n], out[2*n:]


def epsilon_sign(indices: list[int]) -> int:
    return (-1)**sum(indices[i] > indices[j] for i in range(4) for j in range(i+1, 4))


def controls(coeff: Any, onsite: list[float], neutral_coeff: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(2221011)
    basis = su4_basis()
    assert np.max(abs(np.einsum('aij,bji->ab', basis, basis)-np.eye(15)/2)) < 1e-14
    fields = np.einsum('xma,aij->xmij', rng.normal(size=(2, 4, 15)), basis)
    u = expm(1j*np.einsum('a,aij->ij', rng.normal(size=15), basis))
    hc = hop(fields, u, coeff)
    masks = [sum(1 << j for j in comb) for comb in itertools.combinations(range(8), 4)]
    single = lift(hc, masks)
    d = len(masks)
    hs = kron(single, eye(d), format='csr')+kron(eye(d), single, format='csr')
    rho_single = np.array([(m & 15).bit_count()-2 for m in masks])
    rho = (rho_single[:, None]+rho_single[None, :]).ravel()
    r2 = diags(2*rho*rho)
    c = r2@hs-hs@r2
    index = {m: i for i, m in enumerate(masks)}
    v = np.zeros(d*d, complex)
    for left in itertools.combinations(range(4), 2):
        lc = [i for i in range(4) if i not in left]
        for right in itertools.combinations(range(4), 2):
            rc = [i for i in range(4) if i not in right]
            ma = sum(1 << i for i in left)+sum(1 << (i+4) for i in right)
            mb = sum(1 << i for i in lc)+sum(1 << (i+4) for i in rc)
            v[index[ma]*d+index[mb]] = epsilon_sign(list(left)+lc)*epsilon_sign(list(right)+rc)/6
    assert abs(np.vdot(v, v)-1) < 1e-14 and np.linalg.norm(rho*v) == 0
    tangent = c@v
    assert np.linalg.norm(tangent-2*hs@v) < 1e-12
    # Single-copy RDM .5I and genuinely non-Wick local number variance0.
    wedge_masks = [sum(1 << i for i in comb) for comb in itertools.combinations(range(4), 2)]
    local = np.zeros((6, 6), complex)
    for i, m in enumerate(wedge_masks):
        complement = 15 ^ m
        j = wedge_masks.index(complement)
        local[i, j] = epsilon_sign([k for k in range(4) if m >> k & 1]+[k for k in range(4) if complement >> k & 1])/np.sqrt(6)
    rdm2 = local@local.conj().T
    one_rdm = np.array([[np.trace(rdm2@lift(np.eye(4)[p:p+1].T@np.eye(4)[q:q+1], wedge_masks).toarray()) for q in range(4)] for p in range(4)])
    singlet_residual = max(np.linalg.norm(lift(t, wedge_masks).toarray()@local+local@lift(t, wedge_masks).toarray().T) for t in basis)
    a0 = lift(np.block([[fields[0, 0], np.zeros((4, 4))], [np.zeros((4, 4)), fields[1, 0]]]), masks)
    actual_onsite = onsite[0]*kron(a0, eye(d))+onsite[1]*kron(eye(d), a0)
    equal_onsite = onsite[1]*(kron(a0, eye(d))+kron(eye(d), a0))
    assert np.linalg.norm(equal_onsite@v) < 1e-12
    onsite_norm = float(np.linalg.norm(actual_onsite@v))
    assert onsite_norm > .01
    wrong_density = kron(diags(2*rho_single*rho_single), eye(d))+kron(eye(d), diags(2*rho_single*rho_single))
    wrong_c = wrong_density@hs-hs@wrong_density
    # They agree on first tangent, but NOT on finite source or other number sectors.
    density_wrong_norm = float(np.linalg.norm((c-wrong_c).data))
    eps = 2**-10
    finite = expm_multiply(-eps*c, v)
    wrong_finite = expm_multiply(-eps*wrong_c, v)
    density_wrong_state_gap = float(np.linalg.norm(finite-wrong_finite))
    charge_distribution = {str(k): float(np.sum(abs(finite[rho == k])**2)) for k in range(-4, 5)}
    assert density_wrong_state_gap > 1e-8 and charge_distribution['2'] > 0
    energy = float(np.vdot(finite, hs@finite).real)
    tangent_energy = -4*float(np.vdot(hs@v, hs@v).real)
    dt = 1e-5
    plus = expm_multiply(-dt*c, v)
    minus = expm_multiply(dt*c, v)
    derivative = float((np.vdot(plus, hs@plus)-np.vdot(minus, hs@minus)).real/(2*dt))
    hprime = np.block([[np.zeros((4, 4)), 1j*hc[:4, 4:]], [-1j*hc[4:, :4], np.zeros((4, 4))]])
    sp = kron(lift(hprime, masks), eye(d))+kron(eye(d), lift(hprime, masks))
    cp = r2@sp-sp@r2
    cpp = -c
    fv, j1, j2 = jet_exponential(-eps*c, -eps*cp, -eps*cpp, v)
    def theta_state(th: float) -> Any:
        hh = lift(hop(fields, u, coeff, th), masks)
        full = kron(hh, eye(d))+kron(eye(d), hh)
        return expm_multiply(-eps*(r2@full-full@r2), v)
    fp, fm = theta_state(dt), theta_state(-dt)
    jet1_error = float(np.linalg.norm((fp-fm)/(2*dt)-j1))
    jet2_error = float(np.linalg.norm((fp-2*fv+fm)/dt**2-j2))
    # Physical one-edge E=-i dtheta: nonzero complete second derivative.
    electric_action = -64*j2
    # Independent SU4 endpoint covariance and U1 matter/electric Gauss controls.
    gr = expm(.13j*basis[2]); gc = expm(-.17j*basis[7])
    transformed = np.array([[gr@fields[0, m]@gr.conj().T for m in range(4)], [gc@fields[1, m]@gc.conj().T for m in range(4)]])
    bigg = np.block([[gr, np.zeros((4, 4))], [np.zeros((4, 4)), gc]])
    su4_error = float(np.linalg.norm(hop(transformed, gr@u@gc.conj().T, coeff)-bigg@hc@bigg.conj().T))
    gauss_error = float(np.linalg.norm(j1-1j*rho*fv))
    # Full256 CAR neutral, every local number sector, no rank2 state projector.
    hn = -hop(fields, u, neutral_coeff)
    ppar = np.diag([1.]*4+[0.]*4)
    s = hn@ppar-ppar@hn
    alpha = 2**-12
    un = expm(-alpha*s)
    pn = un@ppar@un.conj().T
    neutral_energy = float(np.trace(hn@pn).real)
    allm = list(range(256))
    sn = lift(s, allm).toarray()
    hncar = lift(hn, allm).toarray()
    vn = np.eye(256, dtype=complex)[:, 15]
    fn = expm(-alpha*sn)@vn
    car_neutral_energy = float(np.vdot(fn, hncar@fn).real)
    sector_nontrivial = {str(k): float(np.linalg.norm(sn[np.ix_([m for m in allm if m.bit_count()==k], [m for m in allm if m.bit_count()==k])])) for k in range(9)}
    # Field jets: a real Cart coordinate, exact Duhamel and explicit Gaussian cross.
    df = np.zeros_like(fields); df[0, 1] = basis[0]
    dh = -hop(df, u, neutral_coeff)
    ds = dh@ppar-ppar@dh
    nv, nj1, nj2 = jet_exponential(csr_matrix(-alpha*sn), -alpha*lift(ds, allm), csr_matrix((256, 256)), vn)
    def neutral_field(th: float) -> Any:
        hh = -hop(fields+th*df, u, neutral_coeff)
        ss = hh@ppar-ppar@hh
        return expm(-alpha*lift(ss, allm).toarray())@vn
    nfp, nfm = neutral_field(dt), neutral_field(-dt)
    field_j1_error = float(np.linalg.norm((nfp-nfm)/(2*dt)-nj1))
    field_j2_error = float(np.linalg.norm((nfp-2*nv+nfm)/dt**2-nj2))
    coordinate = float(np.einsum('ij,ji', fields[0, 1], 2*basis[0]).real)
    full_p2_action = -(nj2-coordinate*nj1+(coordinate**2/4-.5)*nv)
    omission_gap = float(np.linalg.norm(full_p2_action+(coordinate**2/4-.5)*nv))
    # Charged Phi and colour jets use the full density generator.
    def charged_operator(ff: Any, uu: Any) -> Any:
        one = lift(hop(ff, uu, coeff), masks)
        both = kron(one, eye(d))+kron(eye(d), one)
        return r2@both-both@r2
    dcfield = charged_operator(df, u)
    cv, cj1, cj2 = jet_exponential(-eps*c, -eps*dcfield, csr_matrix(c.shape), v)
    cfp = expm_multiply(-eps*charged_operator(fields+dt*df, u), v)
    cfm = expm_multiply(-eps*charged_operator(fields-dt*df, u), v)
    charged_field_j1_error = float(np.linalg.norm((cfp-cfm)/(2*dt)-cj1))
    charged_field_j2_error = float(np.linalg.norm((cfp-2*cv+cfm)/dt**2-cj2))
    du = 1j*basis[1]@u
    ddu = -basis[1]@basis[1]@u
    dccolour = charged_operator(fields, du)
    ddccolour = charged_operator(fields, ddu)
    uv, uj1, uj2 = jet_exponential(-eps*c, -eps*dccolour, -eps*ddccolour, v)
    ufp = expm_multiply(-eps*charged_operator(fields, expm(1j*dt*basis[1])@u), v)
    ufm = expm_multiply(-eps*charged_operator(fields, expm(-1j*dt*basis[1])@u), v)
    colour_j1_error = float(np.linalg.norm((ufp-ufm)/(2*dt)-uj1))
    colour_j2_error = float(np.linalg.norm((ufp-2*uv+ufm)/dt**2-uj2))
    assert charged_field_j1_error < 1e-7 and charged_field_j2_error < 2e-5
    assert colour_j1_error < 1e-7 and colour_j2_error < 2e-5
    assert np.max(abs(one_rdm-np.eye(4)/2)) < 1e-14
    assert singlet_residual < 1e-14 and su4_error < 1e-12 and gauss_error < 1e-10
    assert energy < 0 and neutral_energy < 0 and abs(car_neutral_energy-neutral_energy) < 1e-12
    assert abs(derivative-tangent_energy) < .1
    assert jet1_error < 1e-7 and jet2_error < 2e-5 and field_j1_error < 1e-8 and field_j2_error < 2e-5
    assert density_wrong_norm > 1 and omission_gap > 1e-7
    result = dict(kind='ACTUAL_CAR_FINITE_FLOAT_CONTROLS_NOT_FULL_FIELD_ENERGY_INTEGRAL', charged_two_species_canonical_dimension=d*d, full_neutral_CAR_dimension=256, norm=float(np.vdot(finite, finite).real), RDM_half_identity_error=float(np.max(abs(one_rdm-np.eye(4)/2))), wedge_RDM_I6_over6_error=float(np.max(abs(rdm2-np.eye(6)/6))), nonGaussian_number_variance_actual=0, wrong_Wick_number_variance=1, SU4_singlet_residual=singlet_residual, actual_mass_difference_action_norm=onsite_norm, wrong_averaged_or_equal_roles_action_norm=float(np.linalg.norm(equal_onsite@v)), fullrho_wrong_percopy_operator_difference=density_wrong_norm, tangent_C_equals_2H_error=float(np.linalg.norm(tangent-2*hs@v)), negative_H_derivative_exact_formula=tangent_energy, observed_derivative=derivative, finite_charged_energy=energy, SU4_covariance_error=su4_error, U1_Gauss_jet_error=gauss_error, electric_second_derivative_action_norm=float(np.linalg.norm(electric_action)), first_jet_error=jet1_error, second_jet_error=jet2_error, neutral_energy_onebody=neutral_energy, neutral_energy_fullCAR=car_neutral_energy, full_neutral_number_sectors=sector_nontrivial, Phi_first_jet_error=field_j1_error, Phi_second_jet_error=field_j2_error, omitted_Phi_derivatives_action_gap=omission_gap, quadratic_charged_descent_rejected='RDM=.5I gives Tr(h RDM)=0 for every traceless quadratic singlecopy h, also after any singlecopy orbital rotation', diagnostic_steps=dict(charged=eps, neutral=alpha), physical_steps='NOT these diagnostic steps; exact symbolic physical alpha2^-20/delta2^-100')
    result.update(fullrho_wrong_finite_state_gap=density_wrong_state_gap, charged_finite_total_vertex_charge_distribution=charge_distribution, charged_Phi_first_jet_error=charged_field_j1_error, charged_Phi_second_jet_error=charged_field_j2_error, charged_colour_first_jet_error=colour_j1_error, charged_colour_second_jet_error=colour_j2_error)
    arrays = dict(charged_state=finite, charged_tangent=tangent, charged_E_second_action=electric_action, neutral_fullCAR_state=fn, neutral_Phi_p2_action=full_p2_action, local_singlet=local, fields=fields, colour=u)
    return result, arrays


def uniform_prices(neutral_coeff: Any, profiles: Any) -> dict[str, Any]:
    """Exact dyadic prices: no underflowed physical floats."""
    eta, chi, lam, bigf = Q(1, 2**4200), Q(1, 2**4096), Q(2**8192), Q(2**10240)
    alpha, delta, t = Q(1, 2**20), Q(1, 2**100), Q(2**7168)
    csum = sum(Q(float(z.real))**2+Q(float(z.imag))**2 for z in neutral_coeff)
    m2 = Q(15, 4)*csum
    # Conditional on U, vec(Tz) is a centred real Gaussian vector. Wick gives
    # E||Tz||HS^4 <= 3(E||Tz||HS²)^2; M2 is independent of U.
    assert csum < 2**8 and m2 > 1
    m4upper = Q(2**24)
    assert 3*m2*m2 < m4upper
    neutral_coefficient = -2*alpha*m2+Q(4, 3)*alpha**3*m4upper
    assert neutral_coefficient < -alpha
    neutral_upper = neutral_coefficient*lam/Q(2**2100)
    prices = dict(Phi_base=60*chi/eta, Jordan=3000*lam*eta**2, alignment=Q(9, 2)*bigf*eta, colour_base_and_cross=Q(27, 4)*chi*t, colour_potential=Q(7, 5)*bigf/t*(450+48*7168+24), all_field_circuit_and_cross=Q(2**30)*chi/eta, colour_circuit=Q(2**30)*chi, full_HE_HC=Q(2**30), all64_charged_matter_absolute=Q(2**24)/Q(2**2100))
    positive = sum(prices.values())
    total = neutral_upper+positive
    assert total < -Q(2**6070)
    # Baseline neutral parity + paired charges shares the same bosonic fields;
    # its exact matter0, electric0, nonnegative bosonic expectation.
    def log2_upper(q: Q) -> int:
        assert q > 0
        return q.numerator.bit_length()-q.denominator.bit_length()+1
    rows = []
    for n in (4, 6):
        v = n**3
        matching = [(x, y, z) for x in range(n) for y in range(n) for z in range(n) if (x+y+z)%2 == 0]
        targets = [((x+1)%n, y, z) for x, y, z in matching]
        assert len(set(matching+targets)) == v
        paths = {p['role']: p['count']*v*len(p['paths']) for p in profiles}
        rows.append(dict(N=n, V=v, matching_edges=len(matching), neutral_canonical_particles=2*v, charged_copies=64, particles_each_charged=2*v, charged_onsite_pairs=32*v, primitive_directed_charged_paths=paths, actual_even_parity_matching=True, strict_total_upper_log2_negative_magnitude_lower=6070+int(np.log2(v))))
        box = primitive.IntegerBox(n)
        cycles, _ = box.cycle_basis()
        gram_trace = int(np.sum(cycles.astype(np.int64)**2))
        rows[-1]['original_cover_packet'] = dict(cycle_dimension=2*v+1, actual_Gram_trace=gram_trace, compact_beta_halfwidth_exact=str(Q(1, 64*(2*v+1)*gram_trace)), additional_64E2_upper_exact=str(Q(1, 64*(2*v+1)*gram_trace)), normalized_full_dimensional=True, beta0_is_not_real_cover_vacuum=True)
    return dict(eta_trial='2^-4200', alpha_neutral='2^-20', epsilon_neutral='alpha/(Lambda sqrteta)=2^-6112', delta_charged='2^-100', epsilon_charged='delta/sqrteta=2^2000', microscopic_parameters_unchanged=True, neutral_M2_exact=str(m2), neutral_M4_upper_exact=str(m4upper), positive_prices_log2_upper={k: log2_upper(v) for k, v in prices.items()}, neutral_negative_magnitude_log2_lower=6072, total_energy_upper_per_site='less than -2^6070', baseline_exact_energy='nonnegative (same source before both circuits); every matter expectation0 and native electric0', Sharp_comparison='Sharp exact matter0 and bosonic plus electric nonnegative, not its upper used as a lower', all_even_N_ge4=True, cases=rows, full64_density_caps=dict(abs_rho_vertex=128, rho_r2_plus_rho_c2=32768, commutator_factor=65536), physical_floating_evaluation=False)


def construction(profiles: Any, neutral: Any, scale: Q) -> dict[str, Any]:
    return dict(
        kind='EXACT_FULL_PHI_GAUSS_CANONICAL_OPERATOR_SOURCE_AND_COMPLETE_D_H_DOMAIN_BOUND',
        source=[
            'Psi0=f_eta(q) Z_N(t)^(-1/2) exp(-t Ccolour(U)/2) tensor neutral parity wedge4/wedge0 tensor product of32 local wedge2xwedge2 epsilon singlets tensor Ref4 tensor native compact beta0 integer k0.',
            'f_eta(q)=(2pi eta)^(-15V) exp(-|q|²/(4eta)), eta2^-4200; t2^7168; Z_N(t)=integral_Haar exp(-t Ccolour) strictly positive. This exact integral-defined normalizer is not numerically evaluated.',
            'Neutral parity is + when x+y+z even; matching X=(x,x+axis0) for parity+. U_n,X=exp(-epsilon_n [dGamma(-Lambda hhop,n,X),dGamma(Pparity)]) acts on ALL local Fock sectors. Global source has neutral canonical2V, no reference rank2 projector.',
            'For each matching X let Hhop,c,X=sum_s=1..64 dGamma(T_s,X+adjoint), rho_y=sum_s(n_s,y-2), C_X=[rho_r^2+rho_c^2,Hhop,c,X]. U_c,X=exp(-epsilon_c C_X). All64 species and all0..4 local occupancies are acted on. Disjoint matching edges commute; this is one matching colour of a finite-depth circuit.',
            'Psi=(product_X U_c,X)(product_X U_n,X)Psi0. Multiplication by gauge-covariant pointwise unitary preserves pointwise matter norm1, so exact global shared norm1. All colour and charged U1 Gauss constraints commute with the circuit. Ref4 remains inert; charged intercell n1/n3 sectors have nonzero tangent and finite-epsilon leakage.',
            'T_X=sum_mu c_mu (Phi_mu,r U_X+U_X Phi_mu,c)/2 times exp(i theta_X) for charged, no U1 factor neutral. Two-link terms use endpoint fields and the ordered two-link Wilson product, NEVER a midpoint or a frozen frame.',
            'Original real cover: choose a spanning-tree integral cycle matrix B, d=2V+1, G=B^T B. Product packet g_w(beta)=product_j norm_w^(-1/2) exp[-1/(1-(beta_j/w)^2)] for |beta_j|<w, zero elsewhere, w=1/(64 d TrG). Exact norm_w is the positive 1D squared-bump integral. Then |B beta|²<=TrG d w² and 64E² extra<=1/(64dTrG). FullE cross priced by2 times circuit energy plus2 packet energy, already within2^30/site. Native compact beta0 is not a real-cover L2vacuum or a beta minimizer.'
        ],
        primitive_binding=dict(neutral_improved=False, neutral_physical_scale='-Lambda', charged_scale=1, charged_roles=[dict(role=p['role'], count=p['count'], fullPhi_onsite_scalar_exact=str(p['onsite']/scale), archived_frozen_frame_onsite_exact=str(p['onsite']), path_count=len(p['paths'])) for p in profiles], neutral_onsite_scalar=neutral['onsite'], neutral_paths=neutral['paths']),
        exact_energy_derivation=[
            'Because matching neutral projector has only diagonal and matched offdiagonal blocks, all nonmatching nearest neutral hoppings have expectation0. Neutral has no two-link paths (actual improved=False). Onsite term is odd under global q->-q whereas projector diagonal is even, hence its full Gaussian integral0 even though Phi0 appears in matching generator.',
            'Write T=sqrteta Tz; singularvalues sigma(Tz). Exact matched neutral expectation=-2 Lambda sqrteta sum_j sigma_j sin(2 alpha sigma_j). No eigenprojector defines the state; singularvalues are only used to evaluate its expectation. sin x>=x-x^3/6 yields per-site <=Lambda sqrteta[-2alpha M2+(4/3)alpha^3 M4].',
            'Independent radial endpoints imply M2=15/4 sum_mu|c_mu|² independent of U, even under the interacting colour Gibbs wave. Conditional on U, real and imaginary components of vec(Tz) form a centred real Gaussian with covariance Sigma. Wick gives E||Tz||HS4=(TrSigma)^2+2TrSigma²<=3M2²<2^24. Therefore M4=ETr[(Tz* Tz)^2]<=E||Tz||HS4<2^24. No unjustified pointwise32|z|² estimate is needed.',
            'Bosonic multiplication potentials are EXACTLY unchanged by the circuits: shared pointwise norm1. Jordan3000Lambda eta², alignment(9/2)Feta, colour potential <=(7/5)(F/t)(450+48logt+24). Baseline has matter0 and E0, hence nonnegative; no variational upper substituted for baseline lower.',
            'Charged matter, including all onsite and one/two paths, is bounded in absolute expectation by2^24 sqrteta/site. Per onsite/path CAR cap<=8||T||, coefficients<64, total all64 and6paths/site, Gaussian E|z|<=sqrt120;2^24 is conservative. Equal aux-pair onsite cancels only BEFORE the circuit; no cancellation asserted after it.',
            'The paired epsilon singlet is a reused ingredient from bosonic_original_law_join.md406-425, NOT a new source algebra or cancellation of a different sea. For the full-H charged first variation at zero chargedstep, C_X Psi=2Hhop,X Psi. Onsite and field jets preserve local occupations and are orthogonal to this leakage; other one/two hops have different endpoint occupation defects unless they are the selected hop/adjoint. Initial fullE annihilates the compact source. Thus its full-H first derivative is -4||Hhop,X Psi||². This does NOT certify finite-step monotonicity or stationarity; the strict final whole-H bound above instead independently prices every finite-step sector.',
            'Full derivative energy uses ||d(fU)||²<=2||df||²+2|f|²||dU||² for colour, and realpart<f*df,U*dU>=0 for radial Phi because U*dU antiHermitian (normalization derivative); either use the conservative2 bound. Source-jet caps below price neutral and full64 charged together with their cross terms, not independent-sea kinetic costs.'
        ],
        complete_action=[
            'For A_X=-epsilon generator, J_d U=integral_0^1 exp((1-s)A)(J_d A)exp(sA) ds. J_d² U includes the single J_d²A insertion plus both time-ordered J_dA insertions. Product circuit jets use Leibniz including every cross-copy and neutral/charged cross. No finite particle-hole tangent truncation is used: exp(A) is the exact finite-CAR exponential with its entire PH tower.',
            'Put R_X=|q_endpoints|/sqrteta (120 coordinates). From literal coefficients<64, local CAR count<=512 and density cap32768, ||A_n||<=2^10 alpha R_X, ||A_c||<=2^35 delta R_X. RealCart first jets <=(2^10 alpha+2^35 delta)/sqrteta<2^-9/sqrteta; second generator Cart jets0. Ordered colour/U1 first and second jets <=2^-9 R_X (finite product lengths<=2); unitary first jet is bounded by these, second jet <=2^-9 R_X+2^-18 R_X². These caps act in ALL local number sectors, not a wholeCAR operator cap independent of unbounded Phi.',
            'At each Cart coordinate at most one matching gate pair contributes; at each colour/electric edge at most one matching gate pair contributes. Sum over120 endpoint Cartcoordinates and15 colour generators, E R_X²=120 and E R_X4=14640, absorbs squared derivatives and cross terms into2^30 chi/eta perV for Phi,2^30 chi perV for colour circuit,2^30 perV for full64E². Full colour source energy cross already doubled to27chi t/4. This deliberately loose2^30 cap exceeds the above explicit summed moments.',
            'H Psi = -4chi sum_Cart[(d²f)Upsi0+2(df)(dU)psi0+f(d²U)psi0] plus -(3chi/20)sum_colour[(X²g)U+2(Xg)(XU)+g X²U] fpsi0 plus -64sum_edge f g dtheta²Upsi0 plus [VJ+FW+sum_charged dGamma(h_s)-Lambda dGamma(h_n)]Psi. Here U includes both complete circuits and native Gauss section, so HE+HC is once-paid fullE, not two independent terms.',
            'Ccolour<=24V, |X Ccolour|<=48V and |X² Ccolour|<=96V are finite compact bounds (sum over faces); g derivatives contain t,t². Radial derivatives are polynomial q/eta and q²/eta². Circuit jets above are polynomial R up todegree2. Matter action is degree1 q with literal finite CAR cap; Jordan is degree4 q; alignment degree2 q, colour potential compact bounded. Thus every displayed complete-H action term is square integrable with explicit Gaussian moments E R^2k=product_j(120+2j), finite positive Z_N(t), and finite colour/U1 domains. Psi belongs D(H) on each finite N, original packets chosen Schwartz. This is a complete analytic action/domain bound, NOT a numerical full-H residual or variance.',
            'A usable full action norm upper is the triangle sum of the L2 norms of the displayed polynomials: for each Cart derivative use |df/f|<=R/(2sqrteta), |d²f/f|<=R²/(4eta)+1/(2eta); colour ratios <=24tV and48tV+576t²V²; U jets above; matter <=2^30 Lambda sqrteta V R_all, VJ<=2^20 Lambda eta² V R_all4, alignment<=2^20 Feta V R_all², colour<=24FV. Gaussian moment products with dimension60V give all these finite numbers. This bound grows with V; no N-uniform variance claimed.'
        ],
        debt=['No numerical full-H variance computed; normupper is never called variance.', 'No stationary/minimizing ground, beta0 minimization, localSO5 cancellation, spectral pole, IRphase, commoncone or source adoption.', 'All-N strict energetic source is a variational milestone, not campaign closure.', 'Finite controls use rescaled fields and deliberately resolvable diagnostic steps; physical exponents handled by exact rational analytic pricing.'],
        next_physical_decision='Use this negative fullPhi/Gauss trial as a new comparison source; derive a stationary ground and same-ground insertion domains before positive Schur/spectral/phase claims.',
        campaign_complete=False)


def archive(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in files.items():
            z.writestr(name, raw)


def main() -> None:
    started = time.monotonic()
    inputs = {name: (ROOT/name).read_bytes() for name in PINS}
    assert all(hashlib.sha256(inputs[n]).hexdigest() == pin for n, pin in PINS.items())
    # Capture every loaded local Python dependency, including pure profile supplier.
    sources = {Path(__file__).name: Path(__file__).read_bytes()}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).parent == ROOT and str(path).endswith('.py'):
            sources[Path(path).name] = Path(path).read_bytes()
    pre = {**sources, **inputs}
    archive(ROOT/(PREFIX+'preexecution_20261011.zip'), pre)
    frame, scale = primitive.frozen.exact_frame(inputs[primitive.INPUTS[0]])
    cfg = json.loads(inputs[primitive.INPUTS[1]])['input_descriptor']['original_micro_config_UNCHANGED']
    assert cfg['a'] == 1/64 and cfg['auxiliary_count'] == 63
    profiles = primitive.frozen.profiles(cfg, frame, scale)
    band = primitive.frozen.RotationBand(a=cfg['a'], m=cfg['heavy_m'], r=cfg['heavy_r'], v=cfg['heavy_v'], improved=False)
    onsite_n, paths_n = primitive.frozen.scalar_paths(band)
    assert not band.improved and len(paths_n) == 3
    cp = next(p['coefficients'] for p in profiles[0]['paths'] if p['axis'] == 0 and p['length'] == 1)
    charged_coeff = np.array([complex(float(z[0]), float(z[1])) for z in cp])
    neutral_coeff = next(c for disp, c in paths_n if disp[0] == 1)
    measured, arrays = controls(charged_coeff, [float(p['onsite']/scale) for p in profiles], neutral_coeff)
    prices = uniform_prices(neutral_coeff, profiles)
    neutral_binding = dict(onsite=str(Q(float(onsite_n))), paths=[dict(displacement=disp.tolist(), coefficients=[[str(Q(float(z.real))), str(Q(float(z.imag)))] for z in c]) for disp, c in paths_n])
    proof = construction(profiles, neutral_binding, scale)
    proof['uniform_prices'] = prices
    proof['observed_controls'] = measured
    npz = ROOT/(PREFIX+'controls_20261011.npz')
    np.savez_compressed(npz, **arrays)
    proof_path = ROOT/(PREFIX+'proof_20261011.json')
    proof_path.write_text(json.dumps(proof, indent=2)+'\n')
    unchanged = all((ROOT/n).read_bytes() == raw for n, raw in pre.items())
    assert unchanged
    receipt = dict(kind='NEW_LITERAL_CAR_AND_EXACT_PRICE_CONSUMER', command=[sys.executable, Path(__file__).name], cwd=str(ROOT), runtime_seconds=time.monotonic()-started, python=sys.version, numpy=np.__version__, source_inputs_unchanged=unchanged, source_input_hashes={n: hashlib.sha256(raw).hexdigest() for n, raw in pre.items()}, output_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (npz, proof_path)}, stage='fullPhi/Gauss canonical variational source; parent scientific review pending', natural_exit='external runner records actual returncode; no inferred exit in this receipt', prices=prices, controls=measured)
    receipt_path = ROOT/(PREFIX+'receipt_20261011.json')
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    archive(ROOT/(PREFIX+'executed_20261011.zip'), {**pre, **{p.name: p.read_bytes() for p in (npz, proof_path, receipt_path)}})
    print(json.dumps(dict(stage=receipt['stage'], strict_upper=prices['total_energy_upper_per_site'], finite_charged_energy=measured['finite_charged_energy'], source_inputs_unchanged=unchanged)))


if __name__ == '__main__':
    main()
