#!/usr/bin/env python3
"""NEW spacetime-democratic induced Wilson/Haar law, NOT the continuous-time law.

The declared primitive quantum law is a four-dimensional Wilson Grassmann action
and normalized compact U(1) Haar links, with ZERO bare gauge action. Phi provides
only a conditional normalized Clifford frame. Finite-volume prepared-window
transfer, heavy determinant bounds, and the leading connected Compton coefficient
are distinguished from an infinite-volume phase or exact interacting S matrix.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, cast

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT = {name: (ROOT / name).read_bytes()
                          for name in ('covariant_induced_photon.py', 'sea_selection.py', 'rotation_band.py')}
SOURCE_AT_IMPORT = {name: hashlib.sha256(value).hexdigest() for name, value in SOURCE_BYTES_AT_IMPORT.items()}
from sea_selection import SELECTED

I = np.eye(4, dtype=complex)


@dataclass(frozen=True)
class Parameters:
    spatial_size: int = 16777216
    time_slices: int = 1048576
    beta_power: int = 150
    kappa_power: int = 250
    gram: tuple[float, ...] = (.4187253908699568, .0632221974274214,
                               .0632221974274214, .0632221974274214)
    masses: tuple[float, ...] = (.7, 1., 1.4)
    spacings: tuple[float, ...] = (.02, .01, .005)
    energy_ratios: tuple[float, ...] = (.15, .5, 1.)
    angles: tuple[float, ...] = (.4, 1.2, 2.2)


def phi_frame(parameters: Parameters) -> tuple[np.ndarray, dict[str, Any]]:
    phi = np.sqrt(parameters.gram)[:, None, None] * SELECTED
    normal = phi / np.sqrt(np.einsum('mij,mji->m', phi, phi).real / 4)[:, None, None]
    beta, alpha = normal[0], normal[1:]
    gamma = np.array([beta] + [1j * beta @ a for a in alpha])
    defect = max(np.linalg.norm(gamma[m] @ gamma[n] + gamma[n] @ gamma[m]
                               - (2 * I if m == n else 0)) for m in range(4) for n in range(4))
    if defect > 1e-12:
        raise ValueError('Actual normalized Phi frame fails the Clifford contract')
    return gamma, dict(gram=list(parameters.gram), clifford_defect=float(defect),
                      hermiticity_defect=float(np.linalg.norm(gamma-gamma.conj().transpose(0, 2, 1))),
                      status='Selected prepared frame CONDITIONAL; no new-law Phi/Root transfer claimed')


def loop_coefficient(gamma: np.ndarray) -> dict[str, Any]:
    """Enumerate all rooted directed length-four words, not an imported coefficient."""
    directions = [(mu, sign) for mu in range(4) for sign in (-1, 1)]
    closed = nonzero = backtracks = 0
    traces = []
    for word in itertools.product(directions, repeat=4):
        displacement = np.zeros(4, int)
        spin = I.copy()
        for mu, sign in word:
            displacement[mu] += sign
            spin = spin @ (I-sign*gamma[mu])
        if np.any(displacement):
            continue
        closed += 1
        value = float(np.trace(spin).real)
        if any(word[j][0] == word[(j+1) % 4][0] and
               word[j][1] == -word[(j+1) % 4][1] for j in range(4)):
            backtracks += 1
            if abs(value) > 1e-12:
                raise ArithmeticError('Backtracking Wilson projector did not vanish')
        if abs(value) > 1e-12:
            nonzero += 1
            traces.append(value)
    coefficient = -sum(traces)/6/4
    if nonzero != 48 or abs(coefficient-16) > 1e-12:
        raise ArithmeticError('Wrong plaquette sign or rooted/oriented multiplicity')
    return dict(closed_rooted_words_per_site=closed, vanishing_backtracking_words=backtracks,
                nonzero_plaquette_words_per_site=nonzero, individual_spin_traces=sorted(set(traces)),
                beta_over_N_kappa4=coefficient,
                derivation='Saux=-N log det(I-kH)=N sum k^n Tr(H^n)/n. Each geometric plaquette has four starts and two orientations, spin trace -8; (1/4)*4*(-8)*(U+Udagger)=-16 cos(theta).',
                winding='Even extents >=6 exclude length-four winding; ALL longer winding words included in the uniform norm tail.',
                norm_proof='For each mu, Hmu=2[Pminus Tmu+Pplus Tmu^dagger], norm 2 since spin projectors are orthogonal and commute with unitary shifts. Triangle inequality gives ||H||<=8 for every link field.')


def determinant_control(gamma: np.ndarray) -> dict[str, Any]:
    """Exact finite Haar-link Dirac determinant, independent of loop counting."""
    sites = list(itertools.product(range(2), repeat=4))
    index = {x: j for j, x in enumerate(sites)}
    rng = np.random.default_rng(222)
    phases = rng.uniform(-1., 1., (len(sites), 4))
    def hopping(values):
        h = np.zeros((64, 64), complex)
        for x, site in enumerate(sites):
            for mu in range(4):
                if site[mu] == 1:
                    continue
                neighbor = list(site)
                neighbor[mu] += 1
                y = index[tuple(neighbor)]
                sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
                link = np.exp(1j*values[x, mu])
                h[sx, sy] += (I-gamma[mu])*link
                h[sy, sx] += (I+gamma[mu])*link.conjugate()
        return h
    h, zero = hopping(phases), hopping(np.zeros_like(phases))
    plaquette_cost = 0.
    plaquettes = 0
    for x, site in enumerate(sites):
        for mu, nu in itertools.combinations(range(4), 2):
            if site[mu] or site[nu]:
                continue
            xm, xn = list(site), list(site)
            xm[mu] += 1
            xn[nu] += 1
            flux = phases[x, mu]+phases[index[tuple(xm)], nu]-phases[index[tuple(xn)], mu]-phases[x, nu]
            plaquette_cost += 1-math.cos(flux)
            plaquettes += 1
    cases = []
    for k in (.005, .01, .02):
        sign, logdet = np.linalg.slogdet(np.eye(64)-k*h)
        sign0, logdet0 = np.linalg.slogdet(np.eye(64)-k*zero)
        actual = float(logdet0-logdet)
        leading = 16*k**4*plaquette_cost
        remainder = (4*16/6)*(8*k)**6/(1-(8*k)**2)
        if abs(actual-leading) > 2*remainder:
            raise ArithmeticError('Actual determinant violates the all-loop uniform bound')
        cases.append(dict(kappa=k, determinant_sign_real=float(sign.real),
                          determinant_sign_imag=float(sign.imag), exact_action_difference=actual,
                          induced_plaquette_difference=leading, correct_sign_error=abs(actual-leading),
                          wrong_sign_error=abs(actual+leading), uniform_difference_tail=2*remainder))
    return dict(open_box=[2]*4, plaquettes=plaquettes,
                actual_hopping_norm=float(np.linalg.norm(h, 2)), cases=cases,
                meaning='Exact compact link phases in the primitive Dirac matrix; open boundaries exclude spurious short winding, not a small-volume photon-phase claim.')


def spin_half(gamma: np.ndarray) -> dict[str, Any]:
    generators = np.array([-.5j*gamma[2]@gamma[3], -.5j*gamma[3]@gamma[1],
                           -.5j*gamma[1]@gamma[2]])
    commutator = max(np.linalg.norm(generators[i]@generators[(i+1) % 3]
                                   -generators[(i+1) % 3]@generators[i]
                                   -1j*generators[(i+2) % 3]) for i in range(3))
    eig, vectors = np.linalg.eigh(generators[2])
    full_turn = (vectors*np.exp(-2j*math.pi*eig))@vectors.conj().T
    return dict(spin_eigenvalues=eig.tolist(), su2_commutator_defect=float(commutator),
                full_turn_plus_identity_defect=float(np.linalg.norm(full_turn+I)),
                origin='Spatial Spin(3) generators derived from the ACTUAL Phi Euclidean gamma products, not assigned endpoint spin labels.')


def bounds(p: Parameters) -> dict[str, Any]:
    v = p.spatial_size**3*p.time_slices
    beta = 10.**p.beta_power
    # Exact finite integer multiplicity; no overflowing float(N) approximation.
    n = 10**(p.beta_power+4*p.kappa_power)//16
    log_tail = math.log10(v)+p.beta_power+math.log10(8**6/12)-2*p.kappa_power
    delta = 10.**(-p.beta_power/3)
    # Temporal link increment <=2delta; spatial plaquette <=4delta.
    # sum <=4V*(4delta)^4 beta/24, intentionally conservative.
    compact_action_bound = 4*v*256*beta*delta**4/24
    qmin = 2*math.sin(math.pi/p.spatial_size)
    # Vacuum plus a single Fock quantum, every physical mode including prepared
    # torons. Loose inverse-gap bound, not a thermodynamic phase statement.
    gaussian_tail_log = math.log(32*v)-beta*qmin*delta**2/128
    # Leaving the chart from a boundary packet supported in |A|<delta/4 costs
    # temporal action >=beta delta^2/(32 T). Torus-volume/normalization entropy
    # and active determinant ratio are bounded before comparing exponents.
    active_mass0 = min(math.expm1(a*m) for a in p.spacings for m in p.masses)
    entropy = 16*v*math.log(16*math.pi*beta)+4*v*math.log(1+8/active_mass0)
    escape_log = entropy-beta*delta**2/(32*p.time_slices)
    weak_relative = 64*v/(beta*active_mass0**4*qmin**4)
    active_logdet_chart = -4*v*math.log1p(-8*delta/active_mass0)
    normalized_transfer_budget = 32*(compact_action_bound+active_logdet_chart)
    heavy_quadratic_relative_log10 = (math.log10(v)+math.log10(8**6*8)
                                      -2*p.kappa_power-2*math.log10(qmin))
    return dict(volume_four=v, auxiliary_count=str(n), auxiliary_count_decimal_digits=len(str(n)),
                kappa=f'1e-{p.kappa_power}', beta_induced=beta, bare_electric=0, bare_magnetic=0,
                uniform_logdet_tail_log10=log_tail,
                uniform_tail_formula='|R| <= (4VN/6)*(8k)^6/(1-(8k)^2), even bipartite box. det has positive real eigenvalues/complex conjugate pairs since Re D>=1-8k>0 and gamma5 hermiticity.',
                retained_bounded_observable_absolute_error_log10=log_tail+math.log10(4),
                auxiliary_bare_m0_log10=p.kappa_power-math.log10(2),
                auxiliary_normalized_D_singular_gap=f'1-8e-{p.kappa_power}',
                heavy_gap_formula='m0aux=1/(2k)-4; free pole Eaux=log(1+m0aux), no additional low-energy charged species. Singular-value gap D>=1-8k uniformly.',
                heavy_pole_energy_lattice=p.kappa_power*math.log(10)-math.log(2),
                chart_radius=delta, compact_vs_gaussian_slab_action_error=compact_action_bound,
                prepared_gaussian_chart_tail_log_upper=gaussian_tail_log,
                chart_escape_log_upper=escape_log, chart_entropy_upper=entropy,
                weak_coupling_volume_gap_budget=weak_relative,
                active_logdet_variation_on_chart=active_logdet_chart,
                active_chart_proof='The Hermitian part of Dstd=m0+4-H/2 is >=m0 for every unitary link field. ||D(A)-D(0)||<=8delta. Hence |log det D(A)-log det D(0)|<=-4V log(1-8delta/m0), no quenched active vacuum.',
                normalized_Hermite_transfer_error_budget=normalized_transfer_budget,
                heavy_quadratic_relative_error_log10=heavy_quadratic_relative_log10,
                heavy_derivative_proof='Differentiating each n-link loop twice costs at most n^2; sum_even n*(8k)^n from n=6 is bounded by 8*(8k)^6 for k this small. Divide by beta and the smallest nonzero photon qhat^2. This bounds the higher-loop quadratic consumer, including winding, rather than only vacuum action.',
                physical_momentum_resolution=[2*math.pi/(p.spatial_size*a) for a in p.spacings],
                physical_energy_resolution=[1/(p.time_slices*a) for a in p.spacings],
                finite_window='Open finite temporal slab, gauge-averaged Coulomb-chart Gaussian boundary packets, coexact photons and localized torons; no thermodynamic phase or exact asymptotic scattering claim.',
                compact_bound_proof='On |A_link|<=delta in temporal gauge, |1-cos x-x^2/2|<=x^4/24, |curl A|<=4delta. Gauge-averaging restores Gauss invariance. Outside paths must move at least delta/2 in one spatial link from supported boundary data: Cauchy-Schwarz and 1-cos x>=2x^2/pi^2 give temporal barrier; the stated 1/32 bound is weaker. Torus Haar volume, Gaussian normalization and active determinant ratio enter the explicit entropy term. This pays chart leakage rather than inferring a phase from a Hessian.',
                gaussian_bound_proof='Nonzero coexact vacuum variances <=1/(2 beta qmin); a one-photon Hermite factor is paid by weakening the exponent by 64. Torons are localized prepared boundary packets, not omitted zero modes.',
                perturbative_status='weak_coupling_volume_gap_budget is a conservative scale budget, NOT a proved all-orders connected-amplitude remainder. Tree Compton below is the exact leading connected coefficient; no exact finite-g scattering precision is promoted.')


def reflection_positive(p: Parameters) -> dict[str, Any]:
    kappas = [1/(2*(math.expm1(a*m)+4)) for a in p.spacings for m in p.masses]
    return dict(active_kappa_range=[min(kappas), max(kappas)],
                spatial_B_lower=min(1-6*k for k in kappas),
                positivity=True, bare_gauge_kernel='constant Haar convolution, positive rank-one operator',
                determinant_prefactor='Exact Fock transfer is det(Bplus)*Gamma(T1), not Gamma(T1) alone. Bplus>0 ensures a positive scalar. Its gauge-dependent value is retained by the microscopic Dirac determinant; only constant-field free filled-sea normalization cancels it.',
                proof='Wilson r=1 temporal hoppings are 2k Pplus/Pminus. Reflection exchanges psi and barpsi with gamma0; expanding each crossing hopping exponential is a sum of reflected Grassmann squares with nonnegative k powers. Positive-side action and its reflection factor separately. Haar integration pairs reflected gauge links with their adjoints. The spatial fermion B=I-k sum_j(Tj+Tjdagger) obeys B>=1-6k>0, giving the positive one-step Wilson transfer construction. Tensoring a FINITE Naux+1 species preserves it even for the enormous explicit Naux. Gauge averaging is an orthogonal Gauss projector. Integrating auxiliaries is taking the same positive expectation restricted to retained observables, so its nonlocal determinant action preserves RP; it is NOT licensed as a new one-step gauge-only Markov kernel.',
                source='https://projecteuclid.org/journals/communications-in-mathematical-physics/volume-113/issue-3/General-proof-of-Osterwalder-Schrader-positivity-for-the-Wilson-action/cmp/1104160284.pdf',
                source_access='Full PDF blocked by provider HTML challenge; primary Springer abstract checked, local crossing-projector and B positivity argument supplied independently.',
                nonlocal_time_price='Uniform full finite-slab logdet tail bounds normalized retained correlations; no truncation of higher-time loops into a polynomial propagator, hence no claimed ghost pole from such a truncation.',
                operator_scope='Finite spatial volume: bounded positive transfer on compact Haar L2 tensored with finite-dimensional fermion Fock space, not a finite-dimensional Haar Hilbert space. log(T) only on nonzero spectral subspace.')


def rotor_levels(beta: float) -> dict[str, Any]:
    """Exact sine-variable Fourier integrals with weighted tails, not bare tails."""
    cutoff = 12.
    denominator_lower = 2*math.exp(-.5)
    def jacobian(x):
        return 1/math.sqrt(1-x*x/(4*beta))
    def density(x):
        return math.exp(-x*x/2)*jacobian(x)
    denominator, denominator_quad_error = quad(density, -cutoff, cutoff, epsabs=1e-13)
    denominator_tail = math.sqrt(2)*2/cutoff*math.exp(-cutoff*cutoff/2)
    # Beyond sqrt(2 beta), exp(-x^2/2)<=exp(-beta); integrating the
    # exact singular Jacobian over its entire support gives 2pi sqrt(beta).
    outer_denominator_log = math.log(2*math.pi)+.5*math.log(beta)-beta
    levels, budgets = [], []
    for n in range(1, 5):
        def weighted_density(x):
            y = x*x/beta
            # beta[1-T_n(1-x^2/(2beta))], without subtracting close floats.
            polynomial = (.5, 2-.5*y, 4.5-3*y+.5*y*y,
                          8-10*y+4*y*y-.5*y*y*y)[n-1]
            return density(x)*x*x*polynomial
        numerator, numerator_quad_error = quad(weighted_density, -cutoff, cutoff, epsabs=1e-13)
        mean = numerator/denominator
        numerator_tail = n*n*math.sqrt(2)*(cutoff+1/cutoff)*math.exp(-cutoff*cutoff/2)
        outer_numerator_log = math.log(4*math.pi)+1.5*math.log(beta)-beta
        # The omitted outer endpoint terms remain recorded in log form, not
        # silently equated to zero by floating-point underflow.
        normalized_tail = (numerator_tail+mean*denominator_tail)/denominator_lower
        log_lipschitz = 1/(1-(mean+normalized_tail+1)/beta)  # Outer normalized tail <1.
        scaled_level_tail = normalized_tail*log_lipschitz
        quadrature_estimate = (numerator_quad_error+mean*denominator_quad_error)/denominator*log_lipschitz
        levels.append(-math.log1p(-mean/beta))
        budgets.append(dict(n=n, central_scaled_numerator=numerator,
                            weighted_inner_numerator_tail_upper=numerator_tail,
                            normalized_scaled_level_inner_tail_upper=scaled_level_tail,
                            outer_numerator_tail_log_upper=outer_numerator_log,
                            normalized_scaled_level_outer_tail_log_upper=max(outer_numerator_log, math.log(mean)+outer_denominator_log)+math.log(2/denominator_lower)+math.log(log_lipschitz),
                            quadrature_scaled_level_error_estimate=quadrature_estimate,
                            nonlinear_minus_beta_log_price=mean*mean/(2*beta*(1-mean/beta))))
    ratio = cutoff*cutoff/(4*beta)
    return dict(beta=beta, electric_levels_lattice=levels,
                beta_times_levels=[beta*x for x in levels], cutoff=cutoff,
                exact_formula='En=-log[I_n(beta)/I_0(beta)]; x=2sqrt(beta)sin(theta/2), density exp(-x^2/2)/sqrt(1-x^2/(4beta)). Integer Haar Fourier states have unit norm.',
                denominator_central=denominator, denominator_lower=denominator_lower,
                denominator_inner_tail_upper=denominator_tail,
                denominator_outer_tail_log_upper=outer_denominator_log,
                denominator_quadrature_error_estimate=denominator_quad_error,
                finite_beta_central_Jacobian_relative_price=ratio/(2*(1-ratio)**1.5),
                finite_beta_central_Chebyshev_relative_price=32*cutoff*cutoff/beta,
                weighted_moment_and_log_budgets=budgets,
                tail_proof='For X<=|x|<=sqrt(2beta), Jacobian<=sqrt2 and beta(1-cos(n theta))<=n^2*x^2/2. Two-sided M2 tail<=2(X+1/X)exp(-X^2/2). Beyond sqrt(2beta), use exp(-beta) times full Jacobian volume2pi sqrt(beta), and numerator<=2beta. Normalize with D>=2exp(-1/2), paying denominator tail separately. Central n<=4 Chebyshev polynomials and exact Jacobian retain finite-beta corrections. Quad error estimates are separately labeled, not claimed rigorous analytic bounds.',
                wrong_sign_control='beta<0 gives I_1(beta)/I_0(beta)<0: NOT a positive one-step electric transfer eigenvalue.')


def quantum_adverse_controls() -> dict[str, Any]:
    def eigen(beta):
        z = quad(lambda t: math.exp(beta*math.cos(t)-abs(beta)), -math.pi, math.pi)[0]
        return quad(lambda t: math.cos(t)*math.exp(beta*math.cos(t)-abs(beta)),
                    -math.pi, math.pi)[0]/z
    q = .1
    khat = 2*math.sin(q/2)
    return dict(positive_beta10_odd_transfer_eigenvalue=eigen(10),
                wrong_negative_beta10_odd_transfer_eigenvalue=eigen(-10),
                correct_democratic_energy=2*math.asinh(khat/2),
                imported_electric_factor2_energy=2*math.asinh(khat/(2*math.sqrt(2))),
                wrong_cone_ratio=math.asinh(khat/(2*math.sqrt(2)))/math.asinh(khat/2),
                meaning='Actual Haar Fourier odd eigenvalue flips sign; retaining an imported unequal temporal coefficient changes the consumer cone by ~29%, so neither a wrong sign nor a speed-fit passes.')


def photon_transfer(q: float, beta: float, order: int = 40) -> dict[str, Any]:
    """Actual positive Gaussian transfer matrix elements plus compact action price.

    Coulomb/Gauss quotient has two coexact polarizations per nonzero momentum.
    The full finite-box vacuum is the product of these normalized Hermite states.
    Its compact chart cutoff and omitted-chart price are in bounds(), not a phase.
    """
    khat = 2*math.sin(q/2)
    energy = 2*math.asinh(khat/2)
    width = math.sinh(energy)
    # Gaussian T kernel exp[-(x-y)^2/2-khat^2*(x^2+y^2)/4].
    covariance = np.linalg.inv(np.array([[1+khat*khat/2+width, -1.],
                                        [-1., 1+khat*khat/2+width]]))
    z, w = hermgauss(order)
    zz = np.array(list(itertools.product(z, repeat=2)))*math.sqrt(2)
    weights = np.outer(w, w).reshape(-1)/math.pi
    xy = zz@np.linalg.cholesky(covariance).T
    polynomials = np.array([np.ones(len(xy)), math.sqrt(2*width)*xy[:, 0],
                           (2*width*xy[:, 0]**2-1)/math.sqrt(2)])
    other = np.array([np.ones(len(xy)), math.sqrt(2*width)*xy[:, 1],
                     (2*width*xy[:, 1]**2-1)/math.sqrt(2)])
    matrix = np.einsum('ni,mi,i->nm', polynomials, other, weights)
    expected = np.diag(np.exp(-energy*np.arange(3)))
    residual = float(np.linalg.norm(matrix-expected))
    if residual > 2e-12:
        raise ArithmeticError('Quantum transfer does not match two-polarization dispersion')
    return dict(q=q, polarizations=2, energy_lattice=energy, energy_over_q=energy/q,
                photon_spectral_weight=1/(2*beta*math.sinh(energy)),
                external_LSZ_factor_without_charge=math.sqrt(energy/math.sinh(energy)),
                actual_Hermite_transfer_normalized=matrix.tolist(), quantum_transfer_residual=residual,
                state='psi0=(beta*sinh(E)/pi)^1/4 exp[-beta*sinh(E)Q^2/2]; psi1=sqrt(2beta*sinh(E))*Q psi0. For each epsilon dot qhat=0, multiply the full physical Gaussian vacuum by this normalized Hermite factor and average gauge orbits. Curl observable extracts it; longitudinal states are not a third photon.',
                propagator='G_T(p0,q)=1/[beta*(4sin(p0/2)^2+sum_j 4sin(qj/2)^2)]; positive spectral weight 1/(2beta sinh E).',
                compact_status='Finite-slab prepared full-state Gaussian transfer controlled by explicit chart action/leakage bounds; matrix is the Gaussian reference whose error is PAID, not a full compact spectral diagonalization.')


class WilsonConsumer:
    def __init__(self, gamma: np.ndarray, a: float, mass: float):
        self.gamma, self.beta, self.a, self.mass = gamma, gamma[0], a, mass
        self.m0 = math.expm1(a*mass)

    def inverse(self, p: np.ndarray) -> np.ndarray:
        return (self.m0+np.sum(1-np.cos(p)))*I + 1j*np.einsum('m,mij->ij', np.sin(p), self.gamma)

    def energy(self, spatial: np.ndarray) -> float:
        b = 1+self.m0+float(np.sum(1-np.cos(spatial)))
        s2 = float(np.sin(spatial)@np.sin(spatial))
        # Stable acosh(1+x), particularly for the external rest pole.
        x = ((b-1)**2+s2)/(2*b)
        return 2*math.asinh(math.sqrt(x/2))

    def pole(self, spatial: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        e = self.energy(spatial)
        p = np.array([1j*e, *spatial], complex)
        b = 1+self.m0+float(np.sum(1-np.cos(spatial)))
        w = b-math.cosh(e)
        alpha = np.array([1j*self.beta@g for g in self.gamma[1:]])
        h = w*self.beta + np.einsum('j,jab->ab', np.sin(spatial), alpha)
        values, vectors = np.linalg.eigh(h)
        u = vectors[:, 2:]
        if np.linalg.norm(self.inverse(p)@u) > 1e-11:
            raise ArithmeticError('External spinor is not an exact Wilson pole')
        return p, u, b, e

    def transfer_resolvent(self, p: np.ndarray, *, drop_negative: bool = False) -> np.ndarray:
        """Geometric spectral sum of the exact free positive Wilson transfer.

        The positive/negative time rank-two residues and the equal-time Wilson
        term reconstruct the whole lattice propagator by analytic continuation.
        This is not a continuum Dirac propagator substituted at the consumer.
        """
        spatial = np.real(p[1:])
        e = self.energy(spatial)
        b = 1+self.m0+float(np.sum(1-np.cos(spatial)))
        w = b-math.cosh(e)
        numerator = w*I-1j*np.einsum('j,jab->ab', np.sin(spatial), self.gamma[1:])
        plus = (numerator+self.beta*math.sinh(e))/(2*b*math.sinh(e))
        minus = (numerator-self.beta*math.sinh(e))/(2*b*math.sinh(e))
        equal_time = I/(2*b)+numerator/(2*b*math.sinh(e))
        forward, backward = np.exp(-e-1j*p[0]), np.exp(-e+1j*p[0])
        return equal_time+plus*forward/(1-forward)+(0 if drop_negative else minus*backward/(1-backward))

    def vertex(self, outgoing: np.ndarray, incoming: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        middle = (outgoing+incoming)/2
        return np.einsum('m,mij->ij', epsilon*np.cos(middle), 1j*self.gamma) + np.sum(epsilon*np.sin(middle))*I

    def contact(self, outgoing: np.ndarray, incoming: np.ndarray,
                out_pol: np.ndarray, in_pol: np.ndarray) -> np.ndarray:
        middle = (outgoing+incoming)/2
        coefficients = out_pol*in_pol
        return np.sum(coefficients*np.cos(middle))*I - 1j*np.einsum('m,mij->ij', coefficients*np.sin(middle), self.gamma)

    def photon_energy(self, k: np.ndarray) -> float:
        return 2*math.asinh(float(np.linalg.norm(2*np.sin(k/2)))/2)

    def polarizations(self, k: np.ndarray) -> list[np.ndarray]:
        qhat = 2*np.sin(k/2)
        _, _, vh = np.linalg.svd(qhat[None])
        return [np.array([0., *v], complex) for v in vh[1:]]

    def compton(self, ratio: float, angle: float, *, direction_axis: int = 2) -> dict[str, Any]:
        initial, ui, bi, ei = self.pole(np.zeros(3))
        wi = self.a*self.mass*ratio
        kin = np.zeros(3)
        kin[direction_axis] = 2*math.asin(math.sinh(wi/2))
        plane = (direction_axis+1) % 3
        direction = np.zeros(3)
        direction[direction_axis], direction[plane] = math.cos(angle), math.sin(angle)
        def conservation(r):
            return self.energy(kin-r*direction)+self.photon_energy(r*direction)-ei-wi
        r = cast(float, brentq(conservation, 0., 2*np.linalg.norm(kin), xtol=1e-15))
        kout = r*direction
        wo = self.photon_energy(kout)
        final, uf, bf, ef = self.pole(kin-kout)
        q, k = np.array([1j*wi, *kin]), np.array([1j*wo, *kout])
        ps, pu = initial+q, initial-k
        ss, su = self.transfer_resolvent(ps), self.transfer_resolvent(pu)
        spectral_defect = max(float(np.linalg.norm(self.inverse(ps)@ss-I)),
                              float(np.linalg.norm(self.inverse(pu)@su-I)))
        if spectral_defect > 1e-10:
            raise ArithmeticError('Exact positive-transfer spectral sum fails the Dirac inverse')
        input_pols, output_pols = self.polarizations(kin), self.polarizations(kout)
        scale = 2*math.sqrt(ei*ef/(bi*bf))*math.sqrt(wi/math.sinh(wi))*math.sqrt(wo/math.sinh(wo))
        def amplitude(inc, out, control='full'):
            s = self.vertex(final, ps, out)@ss@self.vertex(ps, initial, inc)
            u = self.vertex(final, pu, inc)@su@self.vertex(pu, initial, out)
            c = self.contact(final, initial, out, inc)
            matrix = s+u-c
            if control == 'drop_contact': matrix = s+u
            if control == 'drop_u': matrix = s-c
            if control == 'drop_negative_band':
                matrix = (self.vertex(final, ps, out)@self.transfer_resolvent(ps, drop_negative=True)@self.vertex(ps, initial, inc)
                          +self.vertex(final, pu, inc)@self.transfer_resolvent(pu, drop_negative=True)@self.vertex(pu, initial, out)-c)
            if control == 'drop_temporal_contact':
                spatial_in, spatial_out = inc.copy(), out.copy()
                spatial_in[0] = spatial_out[0] = 0
                matrix = s+u-self.contact(final, initial, spatial_out, spatial_in)
            normalization = scale if control != 'drop_residue' else 2*math.sqrt(ei*ef)
            return normalization*uf.conj().T@self.beta@matrix@ui
        squared = sum(float(np.linalg.norm(amplitude(x, y))**2) for x in input_pols for y in output_pols)/4
        qhat, khat = 2*np.sin(q/2), 2*np.sin(k/2)
        ward_in = max(float(np.linalg.norm(amplitude(qhat, y))) for y in output_pols)
        ward_out = max(float(np.linalg.norm(amplitude(x, khat))) for x in input_pols)
        double = float(np.linalg.norm(amplitude(qhat, khat)))
        dropped = dict(contact=max(float(np.linalg.norm(amplitude(qhat, y, 'drop_contact'))) for y in output_pols),
                       pole=max(float(np.linalg.norm(amplitude(qhat, y, 'drop_u'))) for y in output_pols),
                       temporal_contact=float(np.linalg.norm(amplitude(qhat, khat, 'drop_temporal_contact'))),
                       negative_band=max(float(np.linalg.norm(amplitude(qhat, y, 'drop_negative_band'))) for y in output_pols),
                       residue_squared=sum(float(np.linalg.norm(amplitude(x, y, 'drop_residue'))**2)
                                           for x in input_pols for y in output_pols)/4)
        h = 1e-6
        photon_group = (self.photon_energy(kout+h*direction)-self.photon_energy(kout-h*direction))/(2*h)
        recoil = kin-kout
        fermion_group = (self.energy(recoil+h*direction)-self.energy(recoil-h*direction))/(2*h)
        incoming_group = math.cos(kin[direction_axis]/2)/math.cosh(wi/2)
        # Phase-space expression with lattice dispersion/group velocities and all
        # external residues already in the invariant-amplitude convention.
        cross_a2_over_e4 = squared*r*r/(64*math.pi**2*ei*wi*incoming_group*ef*wo*abs(photon_group-fermion_group))
        continuum_out = ratio/(1+ratio*(1-math.cos(angle)))
        kn = (continuum_out/ratio)**2*(continuum_out/ratio+ratio/continuum_out-math.sin(angle)**2)/(32*math.pi**2*self.mass**2)
        physical_cross_over_e4 = self.a*self.a*cross_a2_over_e4
        spin_rotation = np.linalg.norm(ui.conj().T@ui-I[:2, :2])
        if max(ward_in, ward_out, double) > 2e-9:
            raise ArithmeticError(f'Same-current Wilson Ward failure: {ward_in}, {ward_out}, {double}')
        return dict(a=self.a, mass=self.mass, energy_ratio=ratio, angle=angle, axis=direction_axis,
                    outgoing_energy_over_mass=wo/(self.a*self.mass), exact_energy_defect=abs(conservation(r)),
                    spin_multiplicity=2, spin_normalization_defect=float(spin_rotation),
                    exact_transfer_resolvent_defect=spectral_defect,
                    fermion_pole_residues=[1/bi, 1/bf], photon_LSZ_factors=[math.sqrt(wi/math.sinh(wi)), math.sqrt(wo/math.sinh(wo))],
                    amplitude_squared_over_e4=squared, cross_section_over_e4=physical_cross_over_e4,
                    continuum_Klein_Nishina_over_e4=kn, cross_section_relative_to_KN=physical_cross_over_e4/kn,
                    Ward_in=ward_in, Ward_out=ward_out, double_Ward=double, adverse_controls=dropped,
                    normalization='External exact pole R*gamma0=(I+h/sinh E)/(2b), rank two positive; canonical spinors udagger u=1 carry b^(-1/2). Photon LSZ sqrt(E/sinh E)/sqrt(beta_ind). Rate uses exact recoil and lattice group velocities.',
                    scope='Exact leading connected coefficient in induced weak-coupling finite prepared window; not an exact all-orders scattering amplitude.')


def actual_free_transfer(gamma: np.ndarray) -> dict[str, Any]:
    """Positive Wilson time recurrence, not exp(-a*nearest-neighbor H)."""
    c = WilsonConsumer(gamma, .01, 1.)
    spatial = np.array([.004, .003, .002])
    _, u, b, e = c.pole(spatial)
    _, basis = np.linalg.eigh(gamma[0])
    plus, minus = basis[:, 2:], basis[:, :2]
    f = plus.conj().T@(1j*np.einsum('j,jab->ab', np.sin(spatial), gamma[1:]))@minus
    l = np.block([[np.eye(2), -f], [np.zeros((2, 2)), np.eye(2)]])
    t = l.conj().T@np.diag([1/b, 1/b, b, b])@l
    w = np.vstack((math.exp(e)*plus.conj().T@u, minus.conj().T@u))
    values = np.linalg.eigvalsh(t)
    residual = float(np.linalg.norm(t@w-math.exp(-e)*w))
    if residual > 1e-12 or min(values) <= 0:
        raise ArithmeticError('True Wilson time transfer fails positivity or pole')
    occupation_states = [tuple(j for j in range(4) if mask & (1 << j)) for mask in range(16)]
    fock_transfer = np.zeros((16, 16), complex)
    for row, occupied_out in enumerate(occupation_states):
        for column, occupied_in in enumerate(occupation_states):
            if len(occupied_out) == len(occupied_in):
                fock_transfer[row, column] = np.linalg.det(t[np.ix_(occupied_out, occupied_in)]) if occupied_out else 1.
    determinant_prefactor = b*b  # det(Bplus), Bplus is the TWO-spin block.
    fock_transfer *= determinant_prefactor
    number = np.array([len(occupied) for occupied in occupation_states])
    sea = float(np.linalg.eigvalsh(fock_transfer[np.ix_(number == 2, number == 2)])[-1])
    normalized_fock = fock_transfer/sea
    fock_eigenvalues = np.linalg.eigvalsh(normalized_fock)
    charged_gaps = [-math.log(float(np.linalg.eigvalsh(normalized_fock[np.ix_(number == n, number == n)])[-1]))/c.a for n in (1, 3)]
    if min(fock_eigenvalues) <= 0 or max(abs(gap-e/c.a) for gap in charged_gaps) > 1e-11:
        raise ArithmeticError('Actual exterior-power fermion transfer misses charged gaps')
    return dict(spatial_momentum=spatial.tolist(), positive_transfer_eigenvalues=values.tolist(),
                expected_eigenvalues=[math.exp(-e)]*2+[math.exp(e)]*2,
                exact_pole_time_recurrence_residual=residual,
                actual_Fock_transfer_dimension=16,
                exact_Fock_determinant_prefactor=determinant_prefactor,
                actual_normalized_Fock_transfer_eigenvalues=fock_eigenvalues.tolist(),
                actual_particle_hole_gaps_physical=charged_gaps,
                filled_sea_particle_number=2,
                factorization='T=Ldag diag(B^-1,B)L; L=[[I,-F],[0,I]], B=1+m0+sum_j(1-cos pj), F=Pplus(i sum gammaj sin pj)Pminus. State w=(u(t-1),v(t)).',
                many_body='Exact det(Bplus)*Gamma(T1), normalized by the filled two-eigenvalue (>1) sea, gives charged particle/hole gaps E/a and CAR exclusion. The scalar cancels ONLY in this fixed-background free normalization; it remains essential before gauge integration. This is NOT exp of the earlier continuous-time band.',
                gamma_basis='Fixed projectors (I+-gamma0)/2 from the actual normalized prepared Phi frame.')


def finite_photon_packet(p: Parameters, gamma: np.ndarray) -> dict[str, Any]:
    """Normalized TWO-polarization Gauss-physical packet on actual box momenta."""
    c = WilsonConsumer(gamma, .01, 1.)
    step = 2*math.pi/p.spatial_size
    target = np.array([0., 0., 2*math.asin(math.sinh(.005/2))])
    center = np.rint(target/step).astype(int)
    offsets = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    momenta = (center+offsets)*step
    weights = np.exp(-np.sum(offsets**2, axis=1)/2)
    weights /= np.sum(weights)
    energies = np.array([c.photon_energy(k) for k in momenta])
    mean = float(weights@energies)
    polarization_defect = max(abs(np.dot(2*np.sin(k/2), epsilon[1:]))
                              for k in momenta for epsilon in c.polarizations(k))
    times = [0, 4096, 65536, p.time_slices]
    overlaps = [complex(weights@np.exp(-1j*energies*t)) for t in times]
    return dict(integer_center=center.tolist(), grid_step=step, modes=len(weights), polarizations=2,
                norm=float(np.sum(weights)), transverse_Gauss_defect=float(polarization_defect),
                mean_energy_lattice=mean, energy_std_lattice=math.sqrt(float(weights@(energies-mean)**2)),
                center_energy_error=float(c.photon_energy(center*step)-.005),
                observation_times_lattice=times, actual_unitary_survival_probabilities=[abs(x)**2 for x in overlaps],
                actual_unitary_overlap_real=[x.real for x in overlaps],
                actual_unitary_overlap_imag=[x.imag for x in overlaps],
                quantum_state='For either epsilon(lambda,k), |f,lambda>=sum_grid sqrt(weight_k) a^dagger(lambda,k)|prepared Gaussian vacuum>; all terms are normalized coexact Hermite states. Discrete positive transfer energies determine the measured real-time overlap, not a classical Hessian.',
                scope='Finite prepared photon packet transfer; the Compton coefficient remains its explicitly labeled on-shell Born coefficient, not an uncomputed finite-packet all-orders probability.')

def run(p: Parameters) -> dict[str, Any]:
    gamma, frame = phi_frame(p)
    loop = loop_coefficient(gamma)
    beta = 10.**p.beta_power
    results = [WilsonConsumer(gamma, a, m).compton(ratio, angle)
               for a, m, ratio, angle in itertools.product(p.spacings, p.masses, p.energy_ratios, p.angles)]
    rotations = [WilsonConsumer(gamma, .01, 1.).compton(.5, 1.2, direction_axis=j) for j in range(3)]
    cone = []
    for a in p.spacings:
        c = WilsonConsumer(gamma, a, 1.)
        k = np.array([.003, .002, .001])
        e = c.energy(k)
        cone.append(dict(a=a, photon_energy_lattice=c.photon_energy(k),
                         fermion_relativistic_defect=(e*e-(a*a+float(k@k)))/(a*a+float(k@k)),
                         photon_cone_defect=c.photon_energy(k)/float(np.linalg.norm(k))-1,
                         massless_fermion_cone='At m0=0 both exact hypercubic symbols have leading p0^2+sum pj^2, coefficient 1; no fitted velocity. Massive finite-a O(am) and O(ap)^2 errors remain explicit.'))
    return dict(branch='NEW covariant Wilson/Haar induced law, conditional Phi frame', parameters=asdict(p),
                microscopic_action='S=sum_species barpsi [I-kappa H(U,gamma)] psi; dmu_gauge=product dtheta/(2pi). One active Wilson species plus explicitly counted massive auxiliaries. NO bare electric, magnetic, or plaquette coefficient.',
                source_at_import=SOURCE_AT_IMPORT, frame=frame, loops=loop, bounds=bounds(p),
                reflection_positive=reflection_positive(p), rotor=rotor_levels(beta),
                actual_finite_determinant=determinant_control(gamma), spin_half=spin_half(gamma),
                quantum_adverse_controls=quantum_adverse_controls(),
                photon_transfer=[photon_transfer(q, beta) for q in (.03, .1, .2)],
                common_cone=cone, compton=results,
                actual_free_Wilson_transfer=actual_free_transfer(gamma),
                finite_photon_packet=finite_photon_packet(p, gamma),
                hypercubic_consumer_rotation_spread=max(x['cross_section_over_e4'] for x in rotations)-min(x['cross_section_over_e4'] for x in rotations),
                current='Vmu(pprime,p)=i gammamu cos((pprime+p)mu/2)+I sin((pprime+p)mu/2); qhat_mu Vmu=D(pprime)-D(p). Cmunu=delta_munu[I cos(Pmu)-i gammamu sin(Pmu)] INCLUDING temporal Wilson seagull. M=Vs S Vs+Vu S Vu-C.',
                debt=['Fundamental quantum Grassmann/Haar law and chosen a,m0,kappa,Naux are inputs.',
                      'Selected Phi Clifford frame not yet transported from an actual fluctuating-Phi/Root quantum law.',
                      'No infinite-volume compact Coulomb-phase or sharp interacting-pole claim.',
                      'Leading connected finite-energy Compton coefficient calculated; all-orders connected remainder not proved by the weak-coupling scale budget.',
                      'Finite packets replace asymptotic plane waves; momentum resolution 2pi/(L a), finite temporal observation and prepared chart prices explicit.'])


def validate_consumer(result: dict[str, Any]) -> None:
    smallest = min(result['parameters']['spacings'])
    for case in result['compton']:
        ward = max(case['Ward_in'], case['Ward_out'], case['double_Ward'], 1e-18)
        for failure in ('contact', 'pole', 'temporal_contact', 'negative_band'):
            if case['adverse_controls'][failure] <= 1000*ward:
                raise ArithmeticError(f'Non-discriminating {failure} control')
        if case['a'] == smallest and abs(case['cross_section_relative_to_KN']-1) > .01:
            raise ArithmeticError('Finite-energy current consumer misses its low-cutoff window')
    sign = result['actual_finite_determinant']['cases'][0]
    if sign['wrong_sign_error'] <= sign['uniform_difference_tail']:
        raise ArithmeticError('Finite determinant cannot discriminate the wrong loop sign')
    if result['bounds']['normalized_Hermite_transfer_error_budget'] > 1e-15:
        raise ArithmeticError('Compact/active-sea prepared quantum transfer exceeds its window')
    if result['quantum_adverse_controls']['wrong_negative_beta10_odd_transfer_eigenvalue'] >= 0:
        raise ArithmeticError('Haar quantization control failed to reject the wrong sign')


def _jet_product(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    return [a[0]@b[0], a[1]@b[0]+a[0]@b[1], a[2]@b[0]+a[0]@b[2],
            a[3]@b[0]+a[0]@b[3]+a[1]@b[2]+a[2]@b[1]]


def _jet_sum(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    return [a[k]+b[k] for k in range(4)]


def log_first(x: float, y: float) -> float:
    midpoint = (x+y)/2
    z = (x-y)/(x+y)
    return 1/midpoint if abs(z) < 1e-15 else math.atanh(z)/(midpoint*z)


def _canonical_band_frames(blocks: list[np.ndarray], energies: np.ndarray,
                           gamma: np.ndarray, plus: np.ndarray, minus: np.ndarray,
                           mass0: float) -> tuple[list[np.ndarray], str]:
    projectors = []
    for block, energy in zip(blocks, energies):
        if not math.isfinite(energy) or energy <= 0:
            raise ArithmeticError('Canonical rank-two bands require a nonzero finite gap')
        lo, hi = math.exp(-energy), math.exp(energy)
        projectors.append(((hi*np.eye(4)-block)/(hi-lo), (block-lo*np.eye(4))/(hi-lo)))
    # A common graph seed, not independent eigenvector phases at each momentum.
    seeds = [np.eye(4)[:, 2*(band if mass0 >= 0 else 1-band):2*(band if mass0 >= 0 else 1-band)+2]
             for band in range(2)]
    seed_name = 'signed rest-frame beta seed'
    if any(np.linalg.eigvalsh((proj[band]@seeds[band]).conj().T@(proj[band]@seeds[band]))[0] < 1e-12
           for proj in projectors for band in range(2)):
        chirality = plus.conj().T@(gamma[0]@gamma[1]@gamma[2]@gamma[3])@minus
        common = np.vstack((np.eye(2), 1j*chirality.conj().T))/math.sqrt(2)
        seeds = [common, common]
        seed_name = 'common chirality quadrature seed across every momentum'
    frames = []
    for proj in projectors:
        columns = []
        for band in range(2):
            trial = proj[band]@seeds[band]
            values, vectors = np.linalg.eigh(trial.conj().T@trial)
            if not np.isfinite(values).all() or values[0] <= 1e-12:
                raise ArithmeticError('Canonical band seed fails rank-two receiver transport')
            columns.append(trial@((vectors/np.sqrt(values))@vectors.conj().T))
        frames.append(np.column_stack(columns))
    return frames, seed_name


def soft_photon_transfer_vertex(gamma: np.ndarray, spacing: float, mass: float,
                                pin: np.ndarray, q: np.ndarray, polarization: np.ndarray,
                                *, emission: bool) -> dict[str, Any]:
    """Actual one-soft-photon positive-transfer jet, charge1 g/sqrt(V) stripped."""
    c = WilsonConsumer(gamma, spacing, mass)
    outgoing = pin-q if emission else pin+q
    _, beta_basis = np.linalg.eigh(c.beta)
    plus, minus = beta_basis[:, 2:], beta_basis[:, :2]
    spin_f = np.array([plus.conj().T@g@minus for g in gamma[1:]])
    spin_bar = np.array([-minus.conj().T@g@plus for g in gamma[1:]])
    def free(momentum):
        b = 1+c.m0+float(np.sum(1-np.cos(momentum)))
        if b <= 0:
            raise ArithmeticError('Wilson B is not positive')
        f = np.einsum('j,jab->ab', 1j*np.sin(momentum), spin_f)
        bar = np.einsum('j,jab->ab', 1j*np.sin(momentum), spin_bar)
        block = np.block([[np.eye(2)/b, -f/b], [-bar/b, b*np.eye(2)+bar@f/b]])
        return b, f, bar, block
    bi, fi, _, ti = free(pin)
    bf, _, barf, tf = free(outgoing)
    middle = (pin+outgoing)/2
    b1 = float(polarization@np.sin(middle))
    f1 = np.einsum('j,jab->ab', 1j*polarization*np.cos(middle), spin_f)
    bar1 = np.einsum('j,jab->ab', 1j*polarization*np.cos(middle), spin_bar)
    inv1 = -b1*np.eye(2)/(bi*bf)
    derivative = np.block([[inv1, -(inv1@fi+f1/bf)],
                           [-(bar1/bi+barf@inv1),
                            b1*np.eye(2)+bar1@fi/bi+barf@inv1@fi+barf@f1/bf]])
    energies = np.array([c.energy(pin), c.energy(outgoing)])
    frames, convention = _canonical_band_frames([ti, tf], energies, gamma, plus, minus, c.m0)
    initial_values = np.repeat([math.exp(-energies[0]), math.exp(energies[0])], 2)
    final_values = np.repeat([math.exp(-energies[1]), math.exp(energies[1])], 2)
    raw = frames[1].conj().T@derivative@frames[0]
    omega = c.photon_energy(q)
    if not math.isfinite(omega) or omega <= 0:
        raise ArithmeticError('Soft-photon vertex requires a nonzero coexact frequency')
    vertex, static = np.empty((4, 4), complex), np.empty((4, 4), complex)
    for a, la in enumerate(final_values):
        for b, lb in enumerate(initial_values):
            if emission:
                lf, li = la*math.exp(-omega), lb
                multiplier = (math.sqrt(lb)+math.exp(-omega)*math.sqrt(la))/(math.sqrt(lb)+math.sqrt(la))
            else:
                lf, li = la, lb*math.exp(-omega)
                multiplier = (math.exp(-omega)*math.sqrt(lb)+math.sqrt(la))/(math.sqrt(lb)+math.sqrt(la))
            vertex[a, b] = -log_first(lf, li)*raw[a, b]*multiplier
            static[a, b] = -log_first(la, lb)*raw[a, b]
    physical_frame = np.column_stack((plus, minus))
    return dict(vertex_without_oscillator=vertex, charged_vertex_without_oscillator=vertex[:2, :2],
                static_log_vertex_without_oscillator=static,
                oscillator_normalization=1/math.sqrt(2*math.sinh(omega)), photon_energy=omega,
                incoming_momentum=pin.tolist(), outgoing_momentum=outgoing.tolist(), emission=emission,
                initial_canonical_frame=physical_frame@frames[0],
                final_canonical_frame=physical_frame@frames[1], frame_convention=convention,
                native_field_derivative=derivative,
                temporal_multiplier_proof='Exact sqrt(T_F) Kph sqrt(T_F) first jet, followed by actual -log divided difference. Common occupied-mode eta cancels, so this is the CAR bilinear kernel, including sea-to-particle and hole blocks; not a static derivative substituted for finite-frequency transfer.',
                coexact_defect=float(abs(polarization@(2*np.sin(q/2)))))


def connected_transfer_kernel(p: Parameters, gamma: np.ndarray, pin: np.ndarray,
                              qin: np.ndarray, qout: np.ndarray) -> dict[str, Any]:
    """Exact grade-two jet of positive Wilson Fock/photon transfer.

    Physical Fourier directions are derivatives, NOT noninteger compact-charge
    representations. The actual charge remains one. All connected paths at this
    grade are retained on the momentum graph; arbitrary disconnected sea paths
    cancel through the explicit occupied-minor Schur complement.
    """
    c = WilsonConsumer(gamma, .01, 1.)
    momentum_labels = list(itertools.product(range(-2, 3), repeat=2))
    momenta = np.array([pin+i*qin+j*qout for i, j in momentum_labels])
    index = {label: number for number, label in enumerate(momentum_labels)}
    count = len(momenta)
    zero = np.zeros((count, count), complex)
    shift_in, shift_out = zero.copy(), zero.copy()
    for column, (i, j) in enumerate(momentum_labels):
        if (i+1, j) in index:
            shift_in[index[(i+1, j)], column] = 1
        if (i, j-1) in index:
            shift_out[index[(i, j-1)], column] = 1
    # All selected wavepackets lie in the x-z plane: y is exactly coexact.
    epsilon = np.array([0., 1., 0.])
    _, beta_basis = np.linalg.eigh(gamma[0])
    plus, minus = beta_basis[:, 2:], beta_basis[:, :2]
    spin_f = [plus.conj().T@g@minus for g in gamma[1:]]
    spin_bar = [-minus.conj().T@g@plus for g in gamma[1:]]
    bjet = [(1+c.m0)*np.eye(count, dtype=complex), zero.copy(), zero.copy(), zero.copy()]
    fjet = [np.zeros((2*count, 2*count), complex) for _ in range(4)]
    barjet = [np.zeros_like(fjet[0]) for _ in range(4)]
    for axis in range(3):
        d = np.diag(np.exp(.5j*momenta[:, axis]))
        adj = np.diag(np.exp(-.5j*momenta[:, axis]))
        ain, aout = epsilon[axis]*shift_in, epsilon[axis]*shift_out
        mixed = -.5*(ain@aout+aout@ain)
        tj = [d@d, 1j*d@ain@d, 1j*d@aout@d, d@mixed@d]
        td = [adj@adj, -1j*adj@ain@adj, -1j*adj@aout@adj, adj@mixed@adj]
        bjet[0] += np.eye(count)-.5*(tj[0]+td[0])
        for grade in (1, 2, 3):
            bjet[grade] -= .5*(tj[grade]+td[grade])
        for grade in range(4):
            directed = .5*(tj[grade]-td[grade])
            fjet[grade] += np.kron(directed, spin_f[axis])
            barjet[grade] += np.kron(directed, spin_bar[axis])
    bj = [np.kron(value, np.eye(2)) for value in bjet]
    inverse0 = np.linalg.inv(bj[0])
    ij = [inverse0, -inverse0@bj[1]@inverse0, -inverse0@bj[2]@inverse0,
          inverse0@(bj[1]@inverse0@bj[2]+bj[2]@inverse0@bj[1]-bj[3])@inverse0]
    upper = _jet_product(ij, fjet)
    lower = _jet_product(barjet, ij)
    bottom = _jet_sum(bj, _jet_product(_jet_product(barjet, ij), fjet))
    native = [np.block([[ij[g], -upper[g]], [-lower[g], bottom[g]]]) for g in range(4)]
    # Stable, fixed spin labels in the charged/filled bands of actual T1.
    basis = np.zeros((4*count, 4*count), complex)
    energies = np.array([c.energy(k) for k in momenta])
    eigenvalues = np.empty(4*count)
    blocks = [native[0][np.ix_([2*j, 2*j+1, 2*count+2*j, 2*count+2*j+1],
                              [2*j, 2*j+1, 2*count+2*j, 2*count+2*j+1])] for j in range(count)]
    frames, seed_name = _canonical_band_frames(blocks, energies, gamma, plus, minus, c.m0)
    for number, energy in enumerate(energies):
        rows = [2*number, 2*number+1, 2*count+2*number, 2*count+2*number+1]
        basis[np.ix_(rows, range(4*number, 4*number+4))] = frames[number]
        eigenvalues[4*number:4*number+4] = [math.exp(-energy)]*2+[math.exp(energy)]*2
    tj = [basis.conj().T@value@basis for value in native]
    diagonal_defect = float(np.linalg.norm(tj[0]-np.diag(eigenvalues)))
    if not math.isfinite(diagonal_defect) or diagonal_defect > 1e-10:
        raise ArithmeticError('Canonical Wilson band basis fails native transfer')
    sea_modes = np.array([4*j+s for j in range(count) for s in (2, 3)])
    sea = sum(1 << int(mode) for mode in sea_modes)
    def excitation_energy(state):
        value = 0.
        for mode in range(4*count):
            if bool(state & (1 << mode)) != bool(sea & (1 << mode)):
                value += energies[mode//4]
        return value
    def first_fock(direction, state):
        # Exact CAR signs, exterior-power derivative, det(B) prefactor retained.
        # Its first derivative is zero for these nonzero momentum directions.
        result = {}
        eta = math.exp(-excitation_energy(state))
        for occupied in range(4*count):
            if not state & (1 << occupied):
                continue
            without = state ^ (1 << occupied)
            for empty in range(4*count):
                if without & (1 << empty) or abs(direction[empty, occupied]) < 1e-14:
                    continue
                changed = without | (1 << empty)
                parity = (state & ((1 << occupied)-1)).bit_count() + (without & ((1 << empty)-1)).bit_count()
                value = (-1 if parity % 2 else 1)*eta/eigenvalues[occupied]*direction[empty, occupied]
                result[changed] = result.get(changed, 0)+value
        return result
    wi, wo = c.photon_energy(qin), c.photon_energy(qout)
    field_in, field_out = 1/math.sqrt(2*math.sinh(wi)), 1/math.sqrt(2*math.sinh(wo))
    initial_momentum, final_momentum = index[(0, 0)], index[(1, -1)]
    nodes, weights = np.polynomial.legendre.leggauss(24)
    nodes, weights = (nodes+1)/2, weights/2
    def log_second(x, y, z):
        a = 1-nodes
        return -float(weights@(a/((x*a+nodes)*(y*a+nodes)*(z*a+nodes))))
    def integral(delta, duration):
        return duration*np.exp(.5j*delta*duration)*np.sinc(delta*duration/(2*math.pi))
    def pair_photon_multiplier(pair_energy, photon_energy, emission):
        root = math.exp(-pair_energy/2)
        if emission:
            return -log_first(root*root*math.exp(-photon_energy), 1.)*(1+math.exp(-photon_energy)*root)/(1+root)
        return -log_first(root*root, math.exp(-photon_energy))*(math.exp(-photon_energy)+root)/(1+root)
    def neutral_boundary_jet(habs, pair_energy, photon_energy):
        delta = pair_energy-photon_energy
        ratio = pair_photon_multiplier(pair_energy, photon_energy, True)/pair_photon_multiplier(pair_energy, photon_energy, False)
        return -habs*(ratio*math.exp(-delta/2)/(pair_energy+photon_energy)-math.expm1(-delta/2)/delta)
    response, effective, contact_only = np.zeros((2, 2), complex), np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    pair_response = np.zeros((2, 2, 2), complex)
    neutral_pair_response = np.zeros((2, 2), complex)
    word_envelope = np.zeros((2, 2))
    finite_unsigned = np.zeros((2, 2))
    prepared_response = np.zeros((2, 2), complex)
    prepared_zero_time_overlap = np.zeros((2, 2), complex)
    prepared_endpoint_difference = np.zeros((2, 2), complex)
    prepared_contact_response = np.zeros((2, 2), complex)
    maximum_virtual_energy = 0.
    virtual_numbers = []
    time = p.time_slices
    contexts: list[tuple[int, int, int | None]] = [(s, f, None) for s, f in itertools.product(range(2), repeat=2)]
    contexts += [(s, f, h) for s, f, h in itertools.product((0, 1, -1), range(2), range(2))]
    for spin_i, spin_f, hole in contexts:
        imode, fmode = 4*initial_momentum+spin_i, 4*final_momentum+spin_f
        initial = sea if spin_i == -1 else sea | (1 << imode)
        removed = imode if hole is None else 4*initial_momentum+2+hole
        final = (initial ^ (1 << removed)) | (1 << fmode)
        ei, ef = excitation_energy(initial)+wi, excitation_energy(final)+wo
        li, lf = math.exp(-ei), math.exp(-ef)
        gi, gf = math.exp(-excitation_energy(initial)), math.exp(-excitation_energy(final))
        # Two physical photon time orders: absorb incoming to vacuum; or emit
        # outgoing while incoming remains. Negative-band/Pauli states are actual
        # distinct occupied CAR states, not a continuum negative pole inserted.
        forward_in, reverse_out = first_fock(tj[1], initial), first_fock(tj[2].conj().T, final)
        forward_out, reverse_in = first_fock(tj[2], initial), first_fock(tj[1].conj().T, final)
        intermediates = []
        for a, b, photon_energy in ((forward_in, reverse_out, 0.), (forward_out, reverse_in, wi+wo)):
            for state in a.keys() & b.keys():
                ematter = excitation_energy(state)
                gm = math.exp(-ematter)
                phm = math.exp(-photon_energy)
                em, lm = ematter+photon_energy, gm*phm
                # S_F=positive sqrt(detB Gamma(T1)); K=S_F Kph S_F.
                smi = a[state]*field_in/(math.sqrt(gm)+math.sqrt(gi)) if photon_energy == 0 else a[state]*field_out/(math.sqrt(gm)+math.sqrt(gi))
                sfm = b[state].conjugate()*field_out/(math.sqrt(gf)+math.sqrt(gm)) if photon_energy == 0 else b[state].conjugate()*field_in/(math.sqrt(gf)+math.sqrt(gm))
                kmi = smi*(math.exp(-wi)*math.sqrt(gi)+phm*math.sqrt(gm))
                kfm = sfm*(phm*math.sqrt(gm)+math.exp(-wo)*math.sqrt(gf))
                hmi, hfm = -log_first(lm, li)*kmi, -log_first(lf, lm)*kfm
                intermediates.append((em, lm, phm, smi, sfm, kmi, kfm, hmi, hfm,
                                      (state ^ sea).bit_count() == 3))
                maximum_virtual_energy = max(maximum_virtual_energy, float(em))
        if hole is None:
            virtual_numbers.append(len(intermediates))
        common = np.array([mode for mode in range(4*count) if initial & final & (1 << mode)])
        without = initial ^ (1 << removed)
        parity = (initial & ((1 << removed)-1)).bit_count()+(without & ((1 << fmode)-1)).bit_count()
        sign = -1 if parity % 2 else 1
        fock_contact = sign*gi/eigenvalues[removed]*(tj[3][fmode, removed]
                        -np.sum((tj[1][fmode, common]*tj[2][common, removed]
                                 +tj[2][fmode, common]*tj[1][common, removed])/eigenvalues[common]))*field_in*field_out
        s2 = (fock_contact-sum(x[3]*x[4] for x in intermediates))/(math.sqrt(gi)+math.sqrt(gf))
        k2 = s2*(math.exp(-wi)*math.sqrt(gi)+math.exp(-wo)*math.sqrt(gf)) + sum(x[3]*x[4]*x[2] for x in intermediates)
        h2 = -log_first(lf, li)*k2-sum(log_second(lf, x[1], li)*x[5]*x[6] for x in intermediates)
        if hole is None:
            contact_only[spin_f, spin_i] = h2
            effective[spin_f, spin_i] = h2+.5*sum(x[7]*x[8]*(1/(ei-x[0])+1/(ef-x[0])) for x in intermediates)
        ordered = 0j
        unsigned = abs(h2*integral(ef-ei, time))
        for em, _, _, _, _, _, _, hmi, hfm, _ in intermediates:
            a, b = ef-em, em-ei
            ordered += hfm*hmi*(integral(a+b, time)-integral(a, time))/(1j*b)
            unsigned += abs(hfm*hmi*(integral(a+b, time)-integral(a, time))/(1j*b))
        value = np.exp(-1j*ef*time)*(-1j*h2*integral(ef-ei, time)-ordered)
        if hole is None:
            response[spin_f, spin_i] = value
            word_envelope[spin_f, spin_i] = time*abs(h2)+time*time/2*sum(abs(x[7]*x[8]) for x in intermediates)
            finite_unsigned[spin_f, spin_i] = unsigned
            for duration, target in ((time-1j, prepared_response), (-1j, prepared_zero_time_overlap)):
                ordered_prepared, endpoints = 0j, 0j
                for em, _, phm, _, _, _, _, hmi, hfm, is_pair in intermediates:
                    a, b = ef-em, em-ei
                    ordered_prepared += hfm*hmi*(integral(a+b, duration)-integral(a, duration))/(1j*b)
                    if not is_pair:
                        continue
                    matter_energy = em+math.log(phm)
                    pair_i = matter_energy-energies[initial_momentum]
                    pair_f = matter_energy-energies[final_momentum]
                    if phm == 1.:
                        initial_jet = neutral_boundary_jet(hmi, pair_i, wi)
                        final_jet = neutral_boundary_jet(hfm.conjugate(), pair_f, wo)
                    else:
                        initial_jet = -hmi/(pair_i+wo)
                        final_jet = -hfm.conjugate()/(pair_f+wi)
                    phase = np.exp(-1j*(em-ef)*duration)
                    endpoints += (-1j*initial_jet*hfm*integral(ef-em, duration)
                                  -1j*final_jet.conjugate()*hmi*phase*integral(em-ei, duration)
                                  +final_jet.conjugate()*initial_jet*phase)
                prefactor = math.exp((ei+ef)/2)*np.exp(-1j*ef*duration)
                base = -1j*h2*integral(ef-ei, duration)-ordered_prepared
                target[spin_f, spin_i] = prefactor*(base+endpoints)
                if duration == time-1j:
                    prepared_endpoint_difference[spin_f, spin_i] = prefactor*endpoints
                    prepared_contact_response[spin_f, spin_i] = prefactor*(-1j*h2*integral(ef-ei, duration))
        elif spin_i == -1:
            neutral_pair_response[spin_f, cast(int, hole)] = value
        else:
            pair_response[spin_f, cast(int, hole), spin_i] = value
    rows = [2*initial_momentum, 2*initial_momentum+1,
            2*count+2*initial_momentum, 2*count+2*initial_momentum+1]
    columns = [4*initial_momentum, 4*initial_momentum+1]
    ref_spin = np.column_stack((plus, minus))@basis[np.ix_(rows, columns)]
    ei, ef = energies[initial_momentum]+wi, energies[final_momentum]+wo
    # Complete two-spin particle AND hole blocks in each hard time channel.
    # Separate row/column norms survive rotations between intermediate states;
    # the same-M unsigned sum above is retained, but is not rotation-safe.
    hard_in_s = soft_photon_transfer_vertex(gamma, c.a, c.mass, pin, qin, epsilon, emission=False)
    hard_out_s = soft_photon_transfer_vertex(gamma, c.a, c.mass, pin+qin, qout, epsilon, emission=True)
    hard_out_u = soft_photon_transfer_vertex(gamma, c.a, c.mass, pin, qout, epsilon, emission=True)
    hard_in_u = soft_photon_transfer_vertex(gamma, c.a, c.mass, pin-qout, qin, epsilon, emission=False)
    jin_s, jout_s, jout_u, jin_u = [
        value['vertex_without_oscillator']*value['oscillator_normalization']
        for value in (hard_in_s, hard_out_s, hard_out_u, hard_in_u)]
    cauchy_paths = [(jout_s[:2, :2], jin_s[:2, :2]),
                   (jin_u[:2, :2], jout_u[:2, :2]),
                   (jin_u[:2, 2:], jout_u[2:, :2]),
                   (jout_s[:2, 2:], jin_s[2:, :2])]
    cauchy_channels = [np.linalg.norm(left, axis=1)[:, None]*np.linalg.norm(right, axis=0)[None, :]
                       for left, right in cauchy_paths]
    cauchy_envelope = time*np.abs(contact_only)+.5*time*time*sum(cauchy_channels)
    return dict(response=response, effective=effective, contact=contact_only,
                additional_pair_response=pair_response,
                neutral_pair_response=neutral_pair_response,
                prepared_response=prepared_response,
                prepared_zero_time_overlap=prepared_zero_time_overlap,
                prepared_neutral_endpoint_correction=prepared_endpoint_difference,
                prepared_contact_response=prepared_contact_response,
                maximum_grade2_virtual_energy_lattice=maximum_virtual_energy,
                prepared_state_contract='Positive charged range at BOTH boundaries contributes complex duration T-i, not a dropped halfstep. Neutral photon source is a Fourier Curl coordinate acting on the actual cold canonical quantum vacuum, followed by a neutral halfstep: its PH0 and PH2ph endpoint jets are retained. Source is explicitly positive-band filtered, so neutral grade2 state terms cannot remove a PH hole into the no-extra-PH recoil channel. Actual source/receiver Grams normalize the states; their even corrections enter grade4 because the reference overlap vanishes. Soft spectators remain thermal/inclusive; hard KMS and active thermal departures are separately priced.',
                absolute_time_word_envelope_over_g2_times_V=float(np.linalg.norm(word_envelope)),
                finite_time_unsigned_response_over_g2_times_V=float(np.linalg.norm(finite_unsigned)),
                rotation_safe_hard_Cauchy_envelope_over_g2_times_V=float(np.linalg.norm(cauchy_envelope)),
                rotation_safe_hard_Cauchy_channel_matrices=[value.tolist() for value in cauchy_channels],
                rotation_safe_hard_Cauchy_scope='Four actual positive-K hard channels: s/u particle blocks and s/u hole blocks. All two intermediate spin states retained in each row/column norm, not only matching nonzero entries. Oscillator factors included; physical g²/V remains stripped. Soft momentum shifts and source/receiver boundary placements require separate prices.',
                energies=[float(ei), float(ef)], energy_mismatch=float(ef-ei),
                virtual_counts=virtual_numbers, canonical_basis_defect=diagonal_defect,
                initial_Ref_spin_frame=ref_spin,
                canonical_spin_frame_convention=seed_name,
                source='Exact reference CAR occupied-minor Schur complement with detB retained. Only the free filled-sea scalar normalization is removed: photon-number receiver kernels, neutral coherences, occupied-minus contractions, and extra-PH backgrounds are NOT discarded. No full interacting state is declared Slater. Physical integer charge remains1.',
                time_kernel='K=S_F Kph S_F; H=-log K. First/second log divided differences include the temporal transfer contact. Actual finiteT Dyson double integrals, NOT T*sinc times an assumed asymptotic Born amplitude.')


def connected_packet_response(p: Parameters, gamma: np.ndarray,
                              statistical_twist: tuple[int, int, int] = (0, 0, 0)) -> dict[str, Any]:
    """Normalized finite-box electron/photon packets and physical Ref trace.

    Literal incoming B125 weights are retained. In the selected no-extra-PH
    bulk channel only its five x-offset modes survive momentum conservation;
    they are NOT renormalized or replaced by a postselected three-mode source.
    The electron/receiver beams and outgoing photon have their declared supports.
    Exact microscopic source admission is supplied by the full Gauss capsule.
    """
    c = WilsonConsumer(gamma, .01, 1.)
    electron_twist = math.pi*np.array(statistical_twist)/p.spatial_size
    step = 2*math.pi/p.spatial_size
    central_in = np.array([0., 0., 2*math.asin(math.sinh(.005/2))])
    direction = np.array([math.sin(1.2), 0., math.cos(1.2)])
    out_norm = cast(float, brentq(lambda r: c.energy(central_in-r*direction)+c.photon_energy(r*direction)-.015,
                                  0., .01, xtol=1e-15))
    incoming = step*np.rint(central_in/step)
    outgoing = step*np.rint(out_norm*direction/step)
    offsets = np.arange(-1, 2)
    amplitude = np.exp(-offsets*offsets/4)
    amplitude /= np.linalg.norm(amplitude)
    incoming_offsets = np.arange(-2, 3)
    photon_normalizer = float(np.exp(-incoming_offsets**2/2).sum())
    whole_incoming_offsets = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    whole_incoming_amplitude = np.exp(-np.sum(whole_incoming_offsets**2, axis=1)/4)/photon_normalizer**1.5
    incoming_amplitude = whole_incoming_amplitude.reshape(5, 5, 5)[:, 2, 2]
    receive_offsets = np.arange(-3, 4)
    receive_amplitude = np.exp(-receive_offsets*receive_offsets/(4*1.5**2))
    receive_amplitude /= np.linalg.norm(receive_amplitude)
    direction_offset = np.array([step, 0., 0.])
    result: np.ndarray = np.zeros((2, 4), complex)
    asymptotic = np.zeros_like(result)
    contacts = np.zeros_like(result)
    kernel_cache = {}
    bare_result = np.zeros_like(result)
    zero_time = np.zeros_like(result)
    endpoint_result = np.zeros_like(result)
    carrier = np.array([p.spatial_size/4, 0., 0.])
    records = []
    for a, b, d in itertools.product(range(3), range(5), range(3)):
        electron = electron_twist+offsets[a]*direction_offset
        qin = incoming+incoming_offsets[b]*direction_offset
        qout = outgoing+offsets[d]*direction_offset
        receiver_label = int(offsets[a]+incoming_offsets[b]-offsets[d])
        if receiver_label not in range(-3, 4):
            continue
        kernel = connected_transfer_kernel(p, gamma, electron, qin, qout)
        kernel_cache[(int(offsets[a]), b, d)] = kernel
        receiving = electron+qin-qout
        electron_phase = np.exp(-1j*electron@carrier)
        incoming_phase = np.exp(-1j*qin@carrier)
        outgoing_phase = np.exp(-1j*qout@carrier)
        receiver_phase = np.exp(-1j*receiving@carrier)
        weight = amplitude[a]*incoming_amplitude[b]*amplitude[d]*receive_amplitude[receiver_label+3]*electron_phase*incoming_phase*outgoing_phase.conjugate()*receiver_phase.conjugate()
        ref = kernel['initial_Ref_spin_frame'].conj().T/math.sqrt(2)
        result += weight*kernel['prepared_response']@ref
        bare_result += weight*kernel['response']@ref
        zero_time += weight*kernel['prepared_zero_time_overlap']@ref
        endpoint_result += weight*kernel['prepared_neutral_endpoint_correction']@ref
        mismatch = kernel['energy_mismatch']
        time_factor = -1j*p.time_slices*np.exp(-.5j*sum(kernel['energies'])*p.time_slices)*np.sinc(mismatch*p.time_slices/(2*math.pi))
        asymptotic += weight*time_factor*kernel['effective']@ref
        contacts += weight*kernel['prepared_contact_response']@ref
        records.append(dict(electron_grid_offset=int(offsets[a]), incoming_grid_offset=int(incoming_offsets[b]),
                            outgoing_grid_offset=int(offsets[d]), receiver_grid_offset=receiver_label,
                            weight_real=float(weight.real), weight_imag=float(weight.imag), energy_mismatch=mismatch,
                            virtual_CAR_counts=kernel['virtual_counts'],
                            canonical_basis_defect=kernel['canonical_basis_defect']))
    pair_rows = []
    neutral_control = 0.
    for hole_grid in range(-6, 7):
        pair: np.ndarray = np.zeros((2, 2), complex)
        hole_momentum = electron_twist+hole_grid*direction_offset
        for b, d in itertools.product(range(5), range(3)):
            receiver_label = hole_grid+int(incoming_offsets[b]-offsets[d])
            if receiver_label not in range(-3, 4):
                continue
            key = (hole_grid, b, d)
            if key not in kernel_cache:
                kernel_cache[key] = connected_transfer_kernel(p, gamma, hole_momentum,
                    incoming+incoming_offsets[b]*direction_offset, outgoing+offsets[d]*direction_offset)
            kernel = kernel_cache[key]
            neutral = kernel['neutral_pair_response']
            charged = kernel['additional_pair_response'][:, :, 0]*np.exp(1j*c.energy(hole_momentum)*p.time_slices)
            neutral_control = max(neutral_control, float(np.linalg.norm(charged-neutral)/np.linalg.norm(neutral)))
            pair += incoming_amplitude[b]*amplitude[d]*receive_amplitude[receiver_label+3]*neutral
        pair *= np.exp(1j*hole_momentum@carrier)
        pair_rows.append(dict(hole_grid=hole_grid, norm=float(np.linalg.norm(pair)),
                              response_real=np.real(pair).tolist(), response_imag=np.imag(pair).tolist()))
    pair_probability_coefficient = sum(row['norm']**2 for row in pair_rows)
    coefficient = float(np.linalg.norm(result))
    probability_coefficient = coefficient*coefficient
    physical_log_amplitude = -p.beta_power-math.log10(p.spatial_size**3)+math.log10(coefficient)
    return dict(initial_electron_modes=3, incoming_photon_modes=125,
                incoming_kinematically_retained_modes=5,
                incoming_retained_slice_probability=float(incoming_amplitude@incoming_amplitude),
                source_selection='Literal coherent B125 source. No input three-mode gate, no renormalization of its five contributing x modes.',
                outgoing_photon_modes=3, receiving_electron_modes=7,
                statistical_history=list(statistical_twist), electron_twist=electron_twist.tolist(),
                actual_carrier_centres=dict(electron=carrier.tolist(), incoming_photon=carrier.tolist(),
                                            outgoing_photon=carrier.tolist(), receiving_electron=carrier.tolist(), Ref=[0, 0, 0]),
                carrier_phase_contract='All four actual Fourier coefficients include exp(-ip dot X); bulk momentum-conserving products cancel those phases algebraically. Source boundary fields remain at the Ref and are not assigned that cancellation.',
                initial_electron_norm=float(amplitude@amplitude),
                incoming_photon_norm=float(whole_incoming_amplitude@whole_incoming_amplitude),
                outgoing_photon_norm=float(amplitude@amplitude), receiver_electron_norm=float(receive_amplitude@receive_amplitude),
                primitive_Ref_modes=4, initial_source_spin_factor='1/sqrt2; actual initial positive-band frame entangles Ref4 holes, not a chosen spin echo.',
                finiteT_connected_response_over_g2_times_V_real=np.real(result).tolist(),
                finiteT_connected_response_over_g2_times_V_imag=np.imag(result).tolist(),
                finiteT_connected_response_norm_over_g2_times_V=coefficient,
                physical_leading_amplitude_log10=physical_log_amplitude,
                physical_leading_probability_log10=2*physical_log_amplitude,
                probability_coefficient_over_g4_times_V2=probability_coefficient,
                maximum_grade2_virtual_energy_lattice=max(k['maximum_grade2_virtual_energy_lattice'] for k in kernel_cache.values()),
                bare_vs_prepared_response_relative=float(np.linalg.norm(result-bare_result)/coefficient),
                neutral_preparation_endpoint_relative=float(np.linalg.norm(endpoint_result)/coefficient),
                actual_zero_time_prepared_overlap_relative=float(np.linalg.norm(zero_time)/coefficient),
                bare_disconnected_one_PH_probability_coefficient=pair_probability_coefficient,
                bare_disconnected_one_PH_probability_log10=-2*p.beta_power-2*math.log10(p.spatial_size**3)+math.log10(pair_probability_coefficient),
                actual_neutral_source_discriminator_relative=neutral_control,
                actual_one_PH_background_rows=pair_rows,
                disconnected_scope='Actual bare positive-K finite-time gamma-to-gamma+PH receiver control for the y=z=0 incoming slice, present with neutral source too; original charged electron/Ref are spectators. Thirteen residual hole modes and fifteen photon terms retain the literal B125 amplitudes. Other incoming y/z offsets lead orthogonal PH-hole momenta, not an omitted cancellable scalar. This positive background is NOT charged-source connected Compton and is not the complete interacting inclusive photon-number kernel.',
                exact_finiteT_vs_asymptotic_response_relative=float(np.linalg.norm(result-asymptotic)/coefficient),
                dropping_actual_log_transfer_contact_relative=float(np.linalg.norm(contacts)/coefficient),
                actual_packet_terms=records,
                reference='Baseline Ref operator has charge1; Ref4 initial neutralized background. Hole is -1. Body packet centre at(L/4,0,0), reference worldline at0. Every position component uses the actual integer body-to-Ref Manhattan path, with y segment at reference x=z=0.',
                source_grade2_selection='In the selected no-extra-PH charged receiver, bulk momentum requires incoming y=z offsets zero, leaving ALL five x offsets. For these q_y=0/y-polarized photons the y string at x=z=0 transfers zero body x/z momentum, so pure/one-attachment terms cannot reach the nonzero recoil. For offplane projected-y input, a source X/Z attachment carries its nonzero body y offset; a Y attachment carries no body z recoil. The outgoing y attachment carries neither x nor z recoil. Neither the pure two-attachment nor one-attachment/one-current combination can supply the fixed x AND z receiver recoil with zero final y offset. The nonzero bosonic hard q_z makes the shortest-path geometric sum identical on both sides of the seam. Inclusive PH/radiation and photon-number coherences are not assigned this zero. Full-field/Ref departures require actual Root/Field prices.',
                localization_limit='Literal 125-mode incoming packet and finite three/seven-mode beam envelopes in the full box; carrier-centre separation is not compact support separation.')




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--kappa-power', type=int, default=400)
    args = parser.parse_args()
    start = time.monotonic()
    p = Parameters(kappa_power=args.kappa_power)
    result = run(p)
    validate_consumer(result)
    gamma, _ = phi_frame(p)
    result['connected_finite_packets_all_statistics'] = [connected_packet_response(p, gamma, cast(tuple[int, int, int], h))
        for h in itertools.product((0, 1), repeat=3)]
    result['reference_remainder_review_status'] = 'No claimed microscopic probability theorem from the historical coordinate-chart majorant. Actual quantum Gauss source Grams, energy-form/Duhamel bounds and Root/Field common-range join are separate obligations.'
    for case in result['compton']:
        case['actual_charge1_cross_section'] = case['cross_section_over_e4']/(10.**p.beta_power)/(10.**p.beta_power)
    result['physical_charge'] = dict(charge_representation=1, e_effective=10.**(-p.beta_power/2),
                                     alpha_effective=10.**(-p.beta_power)/(4*math.pi),
                                     status='Finite ultraweak interacting branch, NOT observed-alpha parameter closure')
    descriptor = dict(parameters=asdict(p), auxiliary_species=str(10**(p.beta_power+4*p.kappa_power)//16),
        charged_species_including_active=str(1+10**(p.beta_power+4*p.kappa_power)//16),
        law='Democratic 4D Wilson matter and bare compact Haar links; zero bare E/B/plaquette.',
        packet=dict(spacing=.01, mass=1., incoming_energy_lattice=.005, angle=1.2,
            electron_offsets=list(range(-1, 2)), incoming_offsets=list(itertools.product(range(-2, 3), repeat=3)),
            outgoing_offsets=list(range(-1, 2)), receiving_offsets=list(range(-3, 4)),
            electron_outgoing_envelope='exp(-n²/4), normalized',
            incoming_envelope='exp(-|n|²/4), normalized on ALL125 modes; no input postselection',
            receiving_envelope='exp(-n²/9), normalized',
            carrier_centre=[p.spatial_size//4, 0, 0], Ref_centre=[0, 0, 0], Ref_modes=4,
            body_charge=1, Ref_hole_charge=-1, initial_Ref_background=-4,
            all_statistical_torons=list(itertools.product((0, 1), repeat=3)),
            source_band='Actual canonical stable T1 projector, not bare gamma0 projector.',
            photon_precompensation='sqrt(2 beta sinh E)/qhat * exp(E) for BOTH joint halfsteps.',
            photon_clip=2e6, charged_energy_cut=.25,
            source_path='Shortest integer body-to-Ref Manhattan: x then z then y, y segment at x=z=0; exact half-box ties take the negative body-to-Ref coordinate direction.',
            receiver='Prepared electron/outgoing photon with inclusive unobserved soft states; source-adapted no-extra-real-PH channel is only a positive lower sector.'),
        actual_quantum_state_scope='Reference coefficients only here. Full law-dependent Xi and its reflected insertion Grams must be bound to actual Root/Field receipts, never retagged from kappa250.')
    result['input_descriptor'] = descriptor
    result['input_descriptor_sha256'] = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    result['runtime'] = dict(python=platform.python_version(), executable=sys.executable,
                             numpy=np.__version__, scipy=scipy.__version__, elapsed_seconds=time.monotonic()-start)
    result['source_snapshot_at_import'] = {name: value.decode() for name, value in SOURCE_BYTES_AT_IMPORT.items()}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=args.output, source_at_import=SOURCE_AT_IMPORT,
                         input=result['input_descriptor_sha256'], cases=len(result['compton']),
                         worst_Ward=max(x['Ward_in'] for x in result['compton']),
                         smallest_spacing_KN_ratios=[x['cross_section_relative_to_KN'] for x in result['compton'] if x['a']==min(p.spacings)],
                         compact_action_bound=result['bounds']['compact_vs_gaussian_slab_action_error'])))


if __name__ == '__main__':
    main()
