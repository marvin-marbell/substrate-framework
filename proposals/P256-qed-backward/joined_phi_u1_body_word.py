"""NEW full-N4 paired E0 Gaussian/CAR word consumer, not original SOURCE.

The finite contour word contains unbounded quotient electric kinetic energy,
all charged density Coulomb and both microscopic charged profiles.  Gaussian
mixture cofactors below are not interacting-Slater Wick.  Taylor polynomials
make the analytic-continuation integrals finite Fourier polynomials; replacing
them by exp(h(complex Theta)) is deliberately forbidden.  No normalization,
probability floor, private charged phase or old free-band output is used.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import joined_phi_u1_finite_car as joined
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OWN = Path(__file__).read_bytes()
CAPTURED = {**joined.SOURCE_BYTES, Path(__file__).name: OWN}
ARCHIVE = 'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'
ARCHIVE_SHA = 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6'
S = 2.
ELECTRIC_DIFFUSION = 128.  # HE=(UE/a)Pi²=64Pi², so heat variance=128*kappa.
PRIMITIVE = 'induced_phi_em_gaussian_variational.py'
PRIMITIVE_SHA = '986c34d93435be153d1b23600f4ffe01c600655a06da0bf8ab0943d6b10d7ef8'
PLAN = 'issue-222-induced-photon-delta13-continuation-20261008'
REVISION = 'f4e8ad04d46c6e0812cf213dd46e856c682e142a69df13092b04c75ffca413bc'


def packed(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return dict(real=value.real.tolist(), imag=value.imag.tolist()) if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): packed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [packed(v) for v in value]
    return value


def require(error: float, tolerance: float = 3e-9) -> None:
    if not np.isfinite(error) or error > tolerance:
        raise ArithmeticError(f'Authoritative identity defect {error} exceeds {tolerance}')


def contour_covariance(t: float, cuts: int, electric: bool = True) -> np.ndarray:
    """Exact closed Gaussian covariance at symmetric-word potential nodes.

    HE=64Pi² gives heat variance d*kappa, d=2*64=128 (NOT d=64).
    Endpoint G=exp(-theta²/(4s)): C_jk=s+d(S_min-(S_j+S_k)/2)-d²*S_j*S_k/(4s)
    at sum(kappa)=0. This is the analytic limit of the positive-heat
    tridiagonal inverse, NOT a Hermitian covariance. Forward and reverse
    nodes adjacent to the turn coincide because the two half-edges cancel.
    """
    if not electric:
        return np.full((2*cuts, 2*cuts), S, dtype=complex)
    delta = t/cuts
    times = 1j*delta*np.r_[np.arange(cuts)+.5, cuts-np.arange(cuts)-.5]
    earlier = times[np.minimum.outer(np.arange(2*cuts), np.arange(2*cuts))]
    d = ELECTRIC_DIFFUSION
    return S+d*(earlier-(times[:, None]+times[None, :])/2)-d**2*times[:, None]*times[None, :]/(4*S)


def covariance_factor(cov: np.ndarray) -> np.ndarray:
    # Exact duplicate turn nodes are one Gaussian variable, not a failed
    # matrix square root. Transpose-Cholesky preserves positive heat branches
    # and couples Pi/no-Pi through the same initial real Gaussian variable.
    size = len(cov)
    factor = np.zeros_like(cov)
    if np.all(cov == cov[0, 0]):
        factor[:, 0] = np.sqrt(cov[0, 0])
    else:
        turn = size//2
        keep = np.r_[np.arange(turn), np.arange(turn+1, size)]
        reduced = cov[np.ix_(keep, keep)]
        lower = np.zeros_like(reduced)
        for j in range(len(keep)):
            pivot = reduced[j, j]-np.sum(lower[j, :j]**2)
            if abs(pivot) < 1e-20:
                raise ArithmeticError('Unexpected non-turn contour covariance degeneracy')
            lower[j, j] = np.sqrt(pivot)
            for k in range(j+1, len(keep)):
                lower[k, j] = (reduced[k, j]-np.sum(lower[k, :j]*lower[j, :j]))/lower[j, j]
        factor[np.ix_(keep, keep)] = lower
        factor[turn] = factor[turn-1]
    require(float(np.max(np.abs(factor@factor.T-cov))), 1e-8)
    return factor


def heat_precision_control() -> dict[str, Any]:
    # First prove the endpoint/bridge inverse for genuine positive heat.
    kappas = np.array([.07, .11, .05])
    a = np.zeros((4, 4))
    a[0, 0] = a[-1, -1] = 1/(2*S)
    for j, k in enumerate(kappas):
        a[j:j+2, j:j+2] += np.array([[1., -1.], [-1., 1.]])/(ELECTRIC_DIFFUSION*k)
    actual = np.linalg.inv(a)
    total = float(kappas.sum())
    tau = np.r_[0., np.cumsum(kappas)]
    u = tau/total
    d = ELECTRIC_DIFFUSION
    formula = S+S*d*total/(d*total+4*S)*(1-2*u[:, None])*(1-2*u[None, :])+d*(tau[np.minimum.outer(np.arange(4), np.arange(4))]-tau[:, None]*tau[None, :]/total)
    require(float(np.max(np.abs(actual-formula))))
    c = contour_covariance(1/64, 3)
    b = covariance_factor(c)
    # A Hermitian factor generates BBdag rather than BBtranspose and fails.
    wrong = float(np.max(np.abs(b@b.conj().T-c)))
    if wrong < .01:
        raise ArithmeticError('Wrong complex-Gaussian covariance control was blind')
    # Pi-only raw paired E0 is exactly .5; free evolved density broadens.
    clock = 1/16
    variance = S+ELECTRIC_DIFFUSION**2*(clock/2)**2/(4*S)
    character = math.exp(-variance/2)
    omitted = math.exp(-S/2)
    wrong_half_variance = S+(ELECTRIC_DIFFUSION/2)**2*(clock/2)**2/(4*S)
    wrong_half_character = math.exp(-wrong_half_variance/2)
    # The SAME diffusion used by every contour factor must reproduce the
    # model-owned primitive Pi moments. Gauss-Hermite integrates the actual
    # real G derivative polynomial, rather than echoing supplied constants.
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    weights = weights/np.sqrt(np.pi)
    coordinates = np.sqrt(2*S)*nodes
    energy_density = (ELECTRIC_DIFFUSION/2)*(1/(2*S)-coordinates**2/(4*S**2))
    one_mean = float(weights@energy_density)
    one_square = float(weights@(energy_density**2))
    actual_HE_mean = 129*one_mean
    actual_HE_norm_square = 129*one_square+129*128*one_mean**2
    require(abs(actual_HE_mean-1032), 1e-8)
    require(abs(actual_HE_norm_square-1081536), 1e-7)
    return dict(positive_heat_inverse_defect=float(np.max(np.abs(actual-formula))),
                closed_covariance=c, transpose_factor_defect=float(np.max(np.abs(b@b.T-c))),
                wrong_Hermitian_covariance_defect=wrong,
                Pi_only_raw_paired_E0=.5,
                primitive_Pi_Gaussian_quadrature=dict(real_modes=129, density_variance=S,
                    heat_diffusion_used_by_word=ELECTRIC_DIFFUSION,
                    actual_HE_mean=actual_HE_mean, authoritative_HE_mean=1032,
                    actual_HE_norm_squared=actual_HE_norm_square, authoritative_HE_norm_squared=1081536,
                    mean_defect=abs(actual_HE_mean-1032), norm_squared_defect=abs(actual_HE_norm_square-1081536),
                    wrong_half_HE_mean=actual_HE_mean/2, wrong_half_HE_norm_squared=actual_HE_norm_square/4,
                    derivation='HE G/G=(diffusion/2)(M/(2s)-r²/(4s²)); 12-node Hermite on real density N(0,s), independence across all129 modes. The actual contour uses identical diffusion128.'),
                Pi_only_free_halfclock=dict(time=clock, Gaussian_density_variance=variance,
                    character_mean=character, omitted_Pi_character_mean=omitted,
                    omission_contrast=character-omitted,
                    wrong_half_electric_character=wrong_half_character,
                    wrong_half_electric_contrast=character-wrong_half_character),
                primitive_moment_identity='E exp(i a^T B z)=exp(-a^T C a/2); BB^T=C, z real standard normal; finite Fourier polynomial makes continuation integrable.')


def exterior(matrix: np.ndarray, count: int) -> np.ndarray:
    subsets = list(combinations(range(matrix.shape[0]), count))
    return np.asarray([[np.linalg.det(matrix[np.ix_(r, c)]) for c in subsets] for r in subsets])


def create_small(index: int, count: int, n: int) -> np.ndarray:
    low, high = list(combinations(range(n), count)), list(combinations(range(n), count+1))
    lookup = {c: j for j, c in enumerate(high)}
    out = np.zeros((len(high), len(low)))
    for j, c in enumerate(low):
        if index not in c:
            out[lookup[tuple(sorted((*c, index)))], j] = (-1.)**sum(k < index for k in c)
    return out


def small_car_control() -> dict[str, Any]:
    """Dense finite-CAR oracle derives the SAME-H paired orientation first."""
    rng = np.random.default_rng(2221009)
    n, count = 5, 2
    w, _ = np.linalg.qr(rng.normal(size=(n, count))+1j*rng.normal(size=(n, count)))
    generators = [rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)) for _ in range(4)]
    u = expm(.13*generators[0])@expm(.11*generators[1])
    left = expm(.09*generators[2])@expm(.07*generators[3])
    b = left@u
    o = w.conj().T@b@w
    formula = np.linalg.det(o)*(u-u@w@np.linalg.solve(o, w.conj().T@b))
    sea = np.array([np.linalg.det(w[list(c)]) for c in combinations(range(n), count)])
    gu, gl = exterior(u, count+1), exterior(left, count)
    dense = np.empty((n, n), complex)
    wrong = np.empty_like(dense)
    for i in range(n):
        ci = create_small(i, count, n)
        for j in range(n):
            cj = create_small(j, count, n)
            dense[i, j] = np.vdot(sea, gl@ci.T@gu@cj@sea)
            wrong[i, j] = np.vdot(sea, ci.T@exterior(b, count+1)@cj@sea)
    error = float(np.max(np.abs(dense-formula)))
    require(error)
    wrong_orientation = float(np.max(np.abs(dense-wrong)))
    if wrong_orientation < .01:
        raise ArithmeticError('Autocorrelation orientation control was blind')
    # Occupied basis rotation must not change the source cofactor.
    rotation, _ = np.linalg.qr(rng.normal(size=(count, count))+1j*rng.normal(size=(count, count)))
    wr = w@rotation
    oo = wr.conj().T@b@wr
    rotated = np.linalg.det(oo)*(u-u@wr@np.linalg.solve(oo, wr.conj().T@b))
    require(float(np.max(np.abs(rotated-formula))))
    return dict(CAR_modes=n, occupied=count, dense_SAME_H_paired_defect=error,
                wrong_source_autocorrelation_defect=wrong_orientation,
                occupied_basis_rotation_defect=float(np.max(np.abs(rotated-formula))),
                derivation='Gamma(L)c_i Gamma(U)c_jdag: c_i Gamma(U)=Gamma(U) sum_k U_ik c_k; determinant lemma gives detO [U-U W O^-1 Wdag L U]_ij, O=Wdag L U W. No Hermitian transpose of complex sampled U or L is permitted.')


class Species:
    def __init__(self, profile: dict[str, Any], data: dict[str, Any]):
        self.profile = profile
        native = data['native']
        paths = joined.old.native_paths(profile, native)
        self.x = np.asarray([p['source'] for p in paths], int)
        self.y = np.asarray([p['target'] for p in paths], int)
        self.lengths = np.asarray([p['length'] for p in paths], int)
        self.path_modes = np.asarray([np.sum(native['Q'][list(p['edges'])], axis=0) for p in paths])
        self.hops = np.asarray([p['seam']*np.einsum('m,mij->ij', p['coefficients'], native['frame']) for p in paths])
        self.onsite = float(profile['onsite'])*native['frame'][0]
        blocks = np.indices((4, 4))
        rows = (4*self.x[:, None, None]+blocks[0]).reshape(-1)
        cols = (4*self.y[:, None, None]+blocks[1]).reshape(-1)
        self.rows = np.r_[rows, cols, (4*np.arange(64)[:, None, None]+blocks[0]).reshape(-1)]
        self.cols = np.r_[cols, rows, (4*np.arange(64)[:, None, None]+blocks[1]).reshape(-1)]
        self.forward = self.hops.reshape(-1)
        self.reverse = self.hops.conj().reshape(-1)
        self.diagonal = np.broadcast_to(self.onsite, (64, 4, 4)).reshape(-1)
        h = self.h(np.zeros(129)).toarray()
        fields = np.broadcast_to(native['frame'], (64, 4, 4, 4))
        links = np.broadcast_to(np.eye(4), (192, 4, 4))
        literal = joined.assemble(profile, native, fields, links, np.zeros(192))
        require(float(np.max(np.abs(literal-h))))
        bands, basis = np.linalg.eigh(h)
        if not (bands[127] < 0 < bands[128]):
            raise ArithmeticError('Fixed negative mean Slater is not gapped rank128')
        self.w = basis[:, :128]
        self.p = self.w@self.w.conj().T
        # Real-Theta block row norm bound: all forward/reverse translates.
        hop_norms = np.linalg.norm(self.hops, ord=2, axis=(1, 2))
        row_bound = np.full(64, np.linalg.norm(self.onsite, 2))
        np.add.at(row_bound, self.x, hop_norms)
        np.add.at(row_bound, self.y, hop_norms)
        self.numerical_row_norm = float(np.max(row_bound))
        # Exact mean box has each g_mu<1. Clifford opnorm=sqrt(g_mu)<1;
        # l1 coefficient bound is rational and independent of real Theta.
        self.hmax_exact = abs(Fraction(profile['onsite']))+2*sum(
            (abs(Fraction(p['c0']))+abs(Fraction(p['ci'])) for p in profile['paths']), Fraction(0))
        self.hmax = float(np.nextafter(float(self.hmax_exact), math.inf))
        self.gap = float(min(-bands[127], bands[128]))
        self.literal_defect = float(np.max(np.abs(literal-h)))
        self.initial = np.column_stack((self.w, np.eye(256)[:, :4]))
        self.empty_source = (np.eye(256)-self.p)[:, :4]

    def h(self, theta: np.ndarray) -> csr_matrix:
        angles = self.path_modes@theta
        # Analytic continuation: reverse exp(-i angle), NOT conj(exp(i angle)).
        values = np.r_[self.forward*np.repeat(np.exp(1j*angles), 16),
                       self.reverse*np.repeat(np.exp(-1j*angles), 16), self.diagonal]
        return csr_matrix((values, (self.rows, self.cols)), shape=(256, 256))

    def polynomial(self, vectors: np.ndarray, theta: np.ndarray, kappa: complex, degree: int) -> np.ndarray:
        h = self.h(theta)
        result = vectors.copy()
        term = vectors
        for order in range(1, degree+1):
            term = (-kappa/order)*(h@term)
            result += term
        return result

    def source_derivative(self, theta: np.ndarray) -> complex:
        # Only row0 and its actual one/two-link neighbours enter the source.
        h = self.h(theta)
        return complex(.25*np.trace(h[:4]@self.empty_source))

    def mean_source_derivative(self, variance: complex) -> complex:
        phases = np.exp(-.5*variance*np.sum(self.path_modes**2, axis=1))
        values = np.r_[self.forward*np.repeat(phases, 16), self.reverse*np.repeat(phases, 16), self.diagonal]
        h = csr_matrix((values, (self.rows, self.cols)), shape=(256, 256))
        return complex(.25*np.trace(h[:4]@self.empty_source))


def full_inputs() -> tuple[dict[str, Any], list[Species], dict[str, bytes], np.ndarray]:
    raw = (ROOT/ARCHIVE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA:
        raise ArithmeticError('Pinned physical Phi descriptor changed')
    saved = json.loads(raw)['input_descriptor']['exact_NEW_allq_BO_mean']
    if not all(0 < Fraction(saved['parameters'][key]['upper']) < 1 for key in ('g0', 'gs')):
        raise ArithmeticError('Rational Clifford l1 bound requires exact root upper<1')
    centres = [Fraction(v) for v in saved['centre_exact_rational']]
    data = joined.old.load_mean(g_override=[centres[0], centres[1], centres[1], centres[1]],
        mean_descriptor=dict(law='NEW fixed mean charged Slaters times real quotient Gaussian; no original SOURCE certification inherited', exact_saved_Phi_mean=saved))
    native = data['native']
    if (native['N'], native['V'], native['UE'], native['a'], native['Q'].shape) != (4, 64, 1., 1/64, (192, 129)):
        raise ArithmeticError('Actual full N4 unwrapped quotient geometry required')
    if [p['load'] for p in data['profiles']] != [1, 1, 63]:
        raise ArithmeticError('Original microscopic loads changed')
    eig, vec = np.linalg.eigh(native['Lplus'])
    if not (abs(eig[0]) < 1e-9 and np.all(eig[1:] > 0)):
        raise ArithmeticError('Full63 positive density modes required')
    density_sqrt = vec[:, 1:]*np.sqrt(eig[1:])[None, :]
    species = [Species(p, data) for p in data['profiles'][1:]]
    require(float(np.max(np.abs(native['D']@native['Q']))))
    require(float(np.max(np.abs(native['Q'].T@native['Q']-np.eye(129)))))
    require(float(np.max(np.abs(density_sqrt@density_sqrt.T-native['Lplus']))))
    inputs = {ARCHIVE: raw}
    for key in ('root_receipt', 'full_receipt'):
        desc = data['descriptor'][key]
        inputs[Path(desc['path']).name] = Path(desc['path']).read_bytes()
    return data, species, inputs, density_sqrt


def sample_word(species: list[Species], theta: np.ndarray, alpha: np.ndarray,
                delta: float, degree: int, variant: str) -> tuple[complex, complex]:
    cuts = theta.shape[0]//2
    source_phase = np.exp(-np.sum(alpha[:cuts, 0])) if variant != 'omit_Ref_phase' else 1.+0j
    background = np.exp(-128*np.sum(alpha)) if variant != 'one_flavour' else np.exp(-2*np.sum(alpha))
    total = complex(background)
    active_source = 0j
    for index, sp in enumerate(species):
        if variant == 'one_flavour' and index == 1:
            continue
        vectors = sp.initial.copy() if index == 0 else sp.w.copy()
        for j in range(cuts):
            vectors = sp.polynomial(vectors, theta[j], 1j*delta/2, degree)
            vectors *= np.repeat(np.exp(alpha[j]), 4)[:, None]
            vectors = sp.polynomial(vectors, theta[j], 1j*delta/2, degree)
        forward = vectors
        for j in range(cuts):
            at = cuts+j
            vectors = sp.polynomial(vectors, theta[at], -1j*delta/2, degree)
            vectors *= np.repeat(np.exp(alpha[at]), 4)[:, None]
            vectors = sp.polynomial(vectors, theta[at], -1j*delta/2, degree)
        o = sp.w.conj().T@vectors[:, :128]
        sign, logabs = np.linalg.slogdet(o)
        det = complex(sign*np.exp(logabs))
        total *= det**(1 if index == 0 else 63)
        if index == 0:
            # SAME-H paired source, not <Epsi,U Epsi> autocorrelation.
            schur = forward[:4, 128:]-forward[:4, :128]@np.linalg.solve(o, sp.w.conj().T@vectors[:, 128:])
            active_source = complex(.25*np.trace(schur))
    return total*source_phase*active_source, total


def coulomb_baseline(species: list[Species], alpha: np.ndarray, variant: str) -> tuple[complex, complex]:
    """Exact Coul-only sample; its INTEGRAL, not each sample, is .5."""
    diagonal = np.repeat(np.exp(np.sum(alpha, axis=0)), 4)
    total = complex(np.exp(-(2 if variant == 'one_flavour' else 128)*np.sum(alpha)))
    source = 0j
    for index, sp in enumerate(species):
        if index == 1 and variant == 'one_flavour':
            continue
        o = sp.w.conj().T@(diagonal[:, None]*sp.w)
        sign, logabs = np.linalg.slogdet(o)
        total *= complex(sign*np.exp(logabs))**(1 if index == 0 else 63)
        if index == 0:
            source = complex(.25*np.trace(np.eye(4)-sp.w[:4]@np.linalg.solve(o, sp.w[:4].conj().T*diagonal[None, :4])))
    return total*source, total


def summary(values: np.ndarray) -> dict[str, Any]:
    mean = complex(np.mean(values))
    cov = np.cov(np.column_stack((np.real(values), np.imag(values))), rowvar=False, ddof=1)/len(values)
    return dict(mean=mean, standard_error_real=float(np.sqrt(cov[0, 0])),
                standard_error_imag=float(np.sqrt(cov[1, 1])), mean_covariance=cov,
                independent_antithetic_pairs=len(values),
                empirical_uncertainty_not_certified_confidence_bound=True)


def polynomial_price(species: list[Species], t: float, cuts: int, degree: int) -> dict[str, Any]:
    # On REAL fields exact exponentials are unitary; exterior powers lift
    # per-orbital error by count*(1+eps)^(count-1). Density/electric factors
    # remain exact unitary. Telescoping the finite real-coordinate word,
    # rather than taking sample-complex operator norms, bounds the integral.
    errors = []
    total = 0.
    for index, sp in enumerate(species):
        x = abs(t)*sp.hmax/(2*cuts)
        epsilon = math.exp(x)*x**(degree+1)/math.factorial(degree+1)
        occupied = 129 if index == 0 else 128
        copies = 1 if index == 0 else 63
        lifted = occupied*epsilon*math.exp((occupied-1)*epsilon)
        total += 4*cuts*copies*lifted
        errors.append(dict(role=sp.profile['role'], real_Theta_h_operator_upper=sp.hmax,
                           exact_rational_h_operator_upper=str(sp.hmax_exact),
                           orbital_polynomial_error=epsilon, occupied=occupied, copies=copies,
                           exterior_power_factor_error=lifted))
    word_error = math.expm1(total)
    return dict(degree=degree, real_coordinate_unitary_word_operator_error_upper=word_error,
                raw_paired_amplitude_polynomial_error_upper=word_error,
                control_variate_background_bias_upper=.5*word_error,
                combined_estimator_polynomial_bias_upper=1.5*word_error,
                envelope_evaluation='Analytic real-field formula with rational Hmax, evaluated binary64; arithmetic outward enclosure and source matrix roundoff are not certified.',
                rows=errors, excludes_Trotter_and_MC_error=True)


def density_controls(data: dict[str, Any], species: list[Species], density_sqrt: np.ndarray) -> dict[str, Any]:
    native = data['native']
    rho = np.r_[np.full(32, 128.), np.full(32, -128.)]
    rho[32] += 1
    rho[0] -= 1
    extreme = float(64*rho@native['Lplus']@rho)
    wrong_bound = 18432.
    if extreme <= wrong_bound:
        raise ArithmeticError('One-flavour Coulomb bound control was blind')
    # Co-local Refhole cancels the forward c0 creation density phase;
    # the remaining transition vacancy is NOT .5 sample-by-sample.
    rng = np.random.default_rng(12963)
    alpha = np.sqrt(-2j*64/65536)*(rng.normal(size=(4, 63))@density_sqrt.T)
    alphas = np.repeat(np.sum(alpha[:2], axis=0), 4)
    back = np.repeat(np.sum(alpha[2:], axis=0), 4)
    sp = species[0]
    u = np.diag(np.exp(alphas))
    b = np.diag(np.exp(alphas+back))
    o = sp.w.conj().T@b@sp.w
    source = .25*np.trace((u-u@sp.w@np.linalg.solve(o, sp.w.conj().T@b))[:4, :4])
    correct = source*np.exp(-np.sum(alpha[:2, 0]))
    vacancy = .25*np.trace((np.eye(256)-sp.w@np.linalg.solve(o, sp.w.conj().T@b))[:4, :4])
    require(abs(correct-vacancy))
    wrong = abs(source-vacancy)
    if wrong < 1e-5:
        raise ArithmeticError('Omitted Refhole Coulomb phase control was blind')
    kernel_clock = 2.**-25
    kernel = np.exp(-1j*kernel_clock*extreme)
    rho_one = np.r_[np.full(32, 2.), np.full(32, -2.)]
    rho_one[32] += 1
    rho_one[0] -= 1
    one_energy = float(64*rho_one@native['Lplus']@rho_one)
    kernel_wrong = np.exp(-1j*kernel_clock*one_energy)
    if abs(kernel-1) < .5 or abs(kernel-kernel_wrong) < .5:
        raise ArithmeticError('Full density HS kernel omissions did not discriminate')
    return dict(full_N4_extreme_source_sector_rho=rho, actual_extreme_Coulomb_energy=extreme,
                rejected_wrong_one_flavour_bound=wrong_bound,
                allcharged_uniform_operator_bound=34080768,
                actual_one_flavour_extreme_energy=one_energy,
                exact_density_HS_Fock_character_control=dict(time=kernel_clock,
                    full63_density_character=kernel, omitted_Coulomb_character=1.,
                    actual_omitted63_flavours_character=kernel_wrong,
                    Coulomb_omission_contrast=abs(kernel-1),
                    omission63_flavours_contrast=abs(kernel-kernel_wrong),
                    character_law='E exp(sqrt(-2i t64) z dot Lplus_sqrt.T rho)=exp(-it64 rho Lplus rho), actual full source-sector integer rho.'),
                Coul_only_sample_source_to_background_ratio=correct,
                Coul_only_omitted_Ref_phase_wrong_ratio=source,
                omitted_Ref_phase_defect=wrong,
                Coul_only_exact_integrated_raw_amplitude=.5,
                transition_vacancy_identity_defect=abs(correct-vacancy),
                identity='rho=n_active+sum63 n_aux-128-delta0 Refhole. [rho_x,E0]=0 since active c0dag and co-local R both change rho by opposite units; all four Ref labels retained.',
                density_modes=63, auxiliary_copies=63)


def campaign(data: dict[str, Any], species: list[Species], density_sqrt: np.ndarray,
             powers: list[int], cuts_list: list[int], samples: int, degree: int, seed: int) -> dict[str, Any]:
    rows = []
    contrast_rows = []
    for power in powers:
        clock = 2.**-power
        for cuts in cuts_list:
            variants = ['full']
            if power == min(powers) and cuts == cuts_list[len(cuts_list)//2]:
                variants += ['omit_Pi', 'omit_Coulomb', 'omit_Ref_phase', 'one_flavour']
            covariances = {v: contour_covariance(clock, cuts, v != 'omit_Pi') for v in variants}
            factors = {v: covariance_factor(c) for v, c in covariances.items()}
            expected = {v: -1j*clock*np.mean([species[0].mean_source_derivative(covariances[v][j, j]) for j in range(cuts)]) for v in variants}
            records: dict[str, list[complex]] = {v: [] for v in variants}
            bare: dict[str, list[complex]] = {v: [] for v in variants}
            bg: dict[str, list[complex]] = {v: [] for v in variants}
            rng = np.random.default_rng(seed+power*100+cuts)
            for sample in range(samples):
                z = rng.normal(size=(2*cuts, 129))
                density_z = rng.normal(size=(2*cuts, 63))
                a = np.empty((2*cuts, 64), complex)
                a[:cuts] = np.sqrt(-2j*clock/cuts*64)*(density_z[:cuts]@density_sqrt.T)
                a[cuts:] = np.sqrt(2j*clock/cuts*64)*(density_z[cuts:]@density_sqrt.T)
                accumulated = {v: np.zeros(3, complex) for v in variants}
                for antithetic in (1., -1.):
                    for variant in variants:
                        theta = factors[variant]@z
                        # Reflection of BOTH theta and density keeps their full
                        # Gaussian law; the two samples are one independent pair.
                        theta *= antithetic
                        alpha = np.zeros_like(a) if variant == 'omit_Coulomb' else antithetic*a
                        raw, background = sample_word(species, theta, alpha, clock/cuts, degree, variant)
                        derivative = -1j*clock*np.mean([species[0].source_derivative(theta[j]) for j in range(cuts)])
                        baseline, Coul_background = coulomb_baseline(species, alpha, variant)
                        residual = raw-baseline-.5*(background-Coul_background)-derivative
                        accumulated[variant] += np.array([residual, raw, background])/2
                for variant in variants:
                    records[variant].append(complex(.5+expected[variant]+accumulated[variant][0]))
                    bare[variant].append(complex(accumulated[variant][1]))
                    bg[variant].append(complex(accumulated[variant][2]))
            for variant in variants:
                rows.append(dict(time_exact=f'1/{1 << power}', time=clock, cuts=cuts,
                    variant=variant, raw_paired_amplitude=summary(np.asarray(records[variant])),
                    bare_MC_raw_amplitude=summary(np.asarray(bare[variant])),
                    background_closed_word=summary(np.asarray(bg[variant])),
                    exact_primitive_first_derivative_mean=expected[variant],
                    polynomial_error=polynomial_price(species, clock, cuts, degree),
                    estimator='raw-Coul_only_sample-.5*(background-Coul_background)-first_derivative + .5 + exactGaussian_first_derivative; exact Coul-only paired integral=.5 and both exact background integrals=1; polynomial background bias separately priced; no sample normalization.',
                    stage='Actual full-N4 all129 electric/all63 density finite symmetric unitary word, estimated using finite-polynomial Gaussian/CAR integrals; NOT unrestricted-Hamiltonian propagator certification'))
            for variant in variants[1:]:
                difference = np.asarray(records['full'])-np.asarray(records[variant])
                contrast_rows.append(dict(time=clock, cuts=cuts, full_minus=variant,
                                          paired_raw_amplitude_contrast=summary(difference)))
            print(json.dumps(packed(dict(time=clock, cuts=cuts, variants=variants,
                full_raw=rows[-len(variants)]['raw_paired_amplitude']))), flush=True)
    return dict(rows=rows, paired_omission_contrasts=contrast_rows)




def unrestricted_source_bound(species: list[Species], powers: list[int]) -> dict[str, Any]:
    """Exact bounded-intertwiner proof; no norm bound on unbounded Pi."""
    hmax = species[0].hmax_exact
    rows = []
    for power in powers:
        clock = Fraction(1, 1 << power)
        radius = 3*clock*hmax
        modulus = Fraction(1, 2)-radius
        if modulus <= 0:
            raise ArithmeticError('Unrestricted source survival disk is vacuous at selected clock')
        rows.append(dict(time_exact=str(clock), disk_centre_real='1/2', disk_centre_imag='0',
                         exact_disk_radius=str(radius), radius_decimal=float(radius),
                         exact_raw_modulus_lower=str(modulus), modulus_lower_decimal=float(modulus),
                         exact_raw_probability_lower=str(modulus**2), probability_lower_decimal=float(modulus**2)))
    return dict(stage='Analytic unrestricted full129-Pi/allcharged-Coulomb Hbody NEW paired source survival, not finiteword numerical amplitude transfer or current admission',
        exact_active_h_operator_upper=str(hmax), source_norm_squared_exact='1/2',
        source_Ref4_to_Ref3_operator_upper='1', source_commutator_operator_upper=str(hmax),
        raw_source_clocks=rows,
        derivation=[
            'Each fixed-number charged Fock sector is finite dimensional. Real unwrapped Theta in R129; HE=64(-Delta) selfadjoint on H². Periodic hPhi(Theta) and full rho Lplus rho are bounded Hermitian matrix multiplication. Bounded perturbation gives selfadjoint Hbg,Hsrc on common H², not a Pi cutoff.',
            'E0=.5 sum4 cdag(e0alpha) Ralpha is coordinate-independent, so HE E0=E0 HE on H². Full rho=n_active+sum63 n_aux-128-Refhole delta0 intertwines E0 exactly; therefore HC E0=E0 HC. Neutral heavy commutes with E0 and cancels without identifying its Lambda with charged energy.',
            'On Ref4, four Refholes are orthogonal and E0dag E0=(4-n_active,0)/4<=1. For K=Hsrc E0-E0 Hbg=.5 sum4 cdag(h_active(Theta)ealpha)Ralpha, Kdag K=.25 sum4 a(h ealpha)adag(h ealpha)<=.25 sum4 ||h ealpha||²<=Hmax² by CAR. These are Ref4-to-Ref3 bounds, not full arbitrary-Ref Hilbert bounds; no interacting Slater Wick is used.',
            'Duhamel for a bounded inter-sector intertwiner yields ||Usrc(t)E0-E0 Ubg(t)||<=|t| Hmax. The bounded commutator [Hbg,E0dag E0]=E0dag K-Kdag E0 has norm<=2Hmax, so ||E0 Ubg(t)Psi||² differs from initial1/2 by at most2|t|Hmax.',
            'With A=<E0 Ubg Psi,Usrc E0 Psi>, Cauchy gives |A-||E0 Ubg Psi||²|<=|t|Hmax. Hence |A-1/2|<=3|t|Hmax. Translation/AP mean rank128 fills two of four local active modes, giving exact initial1/2; the normalized real Gaussian does not change this.',
            'Rational microscopic l1 Hmax uses every onsite and one/two-link coefficient and sqrt(g_mu)<1 from the pinned exact mean box. All scalar phases have modulus1 on real Theta, independently of all129 modes. This proof includes HC and unbounded Pi by exact commutation, not omission or an operatornorm price for Pi.'],
        does_not_bound_unrestricted_minus_finite_word_amplitude=True,
        no_state_dependent_Pi_norm_claim=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    car = small_car_control()
    gaussian = heat_precision_control()
    data, species, inputs, density_sqrt = full_inputs()
    controls = density_controls(data, species, density_sqrt)
    initial = .25*np.trace((np.eye(256)-species[0].p)[:4, :4])
    if hashlib.sha256(CAPTURED[PRIMITIVE]).hexdigest() != PRIMITIVE_SHA:
        raise ArithmeticError('Model-owned kinetic/Coulomb coefficient epoch changed')
    require(abs(initial-.5))
    physical = unrestricted_source_bound(species, args.clock_powers)
    outputs = campaign(data, species, density_sqrt, args.clock_powers, args.cuts, args.samples, args.degree, args.seed)
    numerical = dict(Q=data['native']['Q'], Lplus=data['native']['Lplus'], density_sqrt=density_sqrt,
                     Phi_frame=data['native']['frame'], active_W=species[0].w, auxiliary_W=species[1].w,
                     active_path_modes=species[0].path_modes, auxiliary_path_modes=species[1].path_modes,
                     active_hops=species[0].hops, auxiliary_hops=species[1].hops)
    np.savez_compressed(args.numerical_capsule, **numerical)
    capsule_bytes = args.numerical_capsule.read_bytes()
    source_rows = {name: dict(sha256=hashlib.sha256(raw).hexdigest(), bytes_utf8=raw.decode(),
                  unchanged=(ROOT/name).read_bytes() == raw) for name, raw in CAPTURED.items()}
    input_rows = {name: dict(sha256=hashlib.sha256(raw).hexdigest(), unchanged=(ROOT/name).read_bytes() == raw) for name, raw in inputs.items()}
    if not all(v['unchanged'] for v in source_rows.values()) or not all(v['unchanged'] for v in input_rows.values()):
        raise RuntimeError('Captured scientific source/input epoch changed')
    descriptor = dict(plan=PLAN, revision=REVISION, preparation='NEW fixed negative mean charged Slaters rank128 each times normalized real129 quotient Gaussian densityvariance s2; Ref4 before E0, Ref3 source sector',
        law='Hbody=64 sum129 Pi²+dGamma_active hPhi(Theta)+sum63 dGamma_aux hPhi(Theta)+64 rho_allcharged Lplus rho_allcharged; no bare magnetic B²',
        paired_quantity='<E0 exp(-it Hbg) Psi, exp(-it Hsource) E0 Psi>; same-H E0=.5 sum4 c_active,0,alpha dag Ralpha',
        neutral='Same neutral heavy evolution cancels in paired inner product exactly, since E0 commutes with it; heavy Lambda is not charged energy',
        physical_mean=data['descriptor'], allTheta=129, allDensity=63, charged_profiles=2, copy_loads=[1, 63], Ref_before=4, Ref_after=3,
        scalar_U1_generator='I4, not SU4-normalized I4/sqrt8', AP='all three axes; literal one AND ordered two-link phases',
        clocks=args.clock_powers, cuts=args.cuts, samples=args.samples, degree=args.degree, seed=args.seed)
    primitive_lines = CAPTURED[PRIMITIVE].decode().splitlines()
    coefficients = dict(source=PRIMITIVE, sha256=PRIMITIVE_SHA,
        exact_line_witnesses={line: primitive_lines[line-1] for line in (162, 251, 301, 307, 444)},
        HE_coefficient=64, HC_coefficient=64, contour_heat_variance_coefficient=ELECTRIC_DIFFUSION,
        real_Gaussian_initial_HE_mean=1032, real_Gaussian_initial_HE_norm_squared=1081536,
        repaired_wrong_assumption='Own halfkinetic mapping HE=32 was wrong; parent precision1/(2g kappa) already implies heat covariance2g kappa. Earlier probes used wrong half electric coefficient. Final actual law retains UE1/a1/64, HC64 and sets contour heat variance128.')
    return dict(input_descriptor=descriptor, input_descriptor_sha256=hashlib.sha256(json.dumps(packed(descriptor), sort_keys=True).encode()).hexdigest(),
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__),
        small_CAR_orientation_control=car, complex_Gaussian_electric_controls=gaussian,
        authoritative_coefficient_binding=coefficients,
        full_allcharged_density_controls=controls, raw_initial_source_norm_squared=initial,
        profile_controls=[dict(role=sp.profile['role'], mean_spectral_gap=sp.gap,
            literal_mean_operator_defect=sp.literal_defect, full_rank=128, hmax=sp.hmax,
            hmax_exact_rational=str(sp.hmax_exact), numerical_row_norm=sp.numerical_row_norm,
            literal_one_link_translates=int(np.sum(sp.lengths == 1)),
            literal_two_link_translates=int(np.sum(sp.lengths == 2))) for sp in species],
        unrestricted_Hbody_source_survival=physical,
        science=outputs, numerical_capsule=dict(path=str(args.numerical_capsule), sha256=hashlib.sha256(capsule_bytes).hexdigest()),
        source_at_execution=source_rows, input_at_execution=input_rows,
        derivation=dict(electric='Positive heat precision proof and analytic closed-contour covariance given above; finite Fourier polynomial integrability follows exp(linear Gaussian) moments; HS is independent of Theta.',
            Coulomb='E_z exp(alpha dot rho)=exp(-i delta64 rho Lplus rho), alpha=sqrt(-2i delta64) Lplus_sqrt z, 63 real z; backwards sign conjugated. Full background -128 and source Refhole -delta0 on forward leg only.',
            finite_word='Each source-sector orbital polynomial is lifted by exterior powers, not exp(dGamma polynomial). Cofactor of fixed initial Slaters is exact sample-by-sample; Gaussian average is an interacting non-Gaussian many-CAR evolution.',
            unbounded_Pi='Implemented by exact untruncated Gaussian kinetic kernels. No bounded global Pi norm or nonexistent imaginary-noise probability law used.',
            sampling='Antithetic pairs independent across samples; covariance of real/imag mean reported. No certified MC confidence bound claimed.'),
        admission=dict(stage='NEW source: analytic unrestricted Hbody paired survival; actual full-N4 electric/matter/Coulomb finiteword numerical consumer separately',
            unrestricted_body_source_survival=True, unrestricted_body_numerical_amplitude_transfer=False,
            current_admitted=False, original_SOURCE_admitted=False,
            Phi_embedding='Parent-supplied conditional uniform arbitrary-state Phi comparison only; no old free-band probability transferred.',
            remaining_debts=['Unbounded-electric state-dependent Trotter error to unrestricted Hbody is not priced by cut convergence.',
                'Empirical Monte Carlo standard errors are not rigorous confidence intervals.',
                'Numerical mean Slater/projector/geometry matrices have floating roundoff, not interval proof.',
                'Parent must integrate conditional Phi uniform embedding; full current, photon pole/common cone and originalSOURCE remain open.']),
        elapsed_seconds=time.monotonic()-started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--numerical-capsule', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--clock-powers', type=int, nargs='+', default=[20, 18])
    parser.add_argument('--cuts', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--seed', type=int, default=20261009)
    args = parser.parse_args()
    if args.samples < 2 or args.degree < 1 or min(args.cuts) < 1:
        parser.error('At least two independent pairs, positive cuts and polynomial degree required')
    with threadpool_limits(limits=1):
        result = run(args)
    args.output.write_text(json.dumps(packed(result), indent=2)+'\n')
    print(json.dumps(packed(dict(input_descriptor_sha256=result['input_descriptor_sha256'],
          source_sha256=hashlib.sha256(OWN).hexdigest(), numerical_capsule=result['numerical_capsule'],
          admission=result['admission'], elapsed_seconds=result['elapsed_seconds']))), flush=True)


if __name__ == '__main__':
    main()
