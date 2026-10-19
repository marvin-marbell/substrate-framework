#!/usr/bin/env python3
"""Native 3D noncompact Maxwell-relative current Hessian and phase constants.

The CLI uses exact continuous-time segment evolution, independently perturbed
native endpoints and the contact-completed determinant. It is not a Monte Carlo
simulation or a finite-graph pole advertised as an infinite-volume phase.
Scientific execution belongs to the integration owner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.linalg import expm, expm_frechet

from bulk_quantum_selection import BulkQuantumLaw
from original_charged_principal import original_frame, portable
from original_compact_charge_bound import chiral_basis
from original_massive_photon_locality import (
    ContinuousGreen, array_hash, native_gauge_controls, rejected, source_hashes as locality_hashes,
)
from original_uniform_charge_gap import canonical_controls, require, residual, time_evolution
from rotation_band import RotationBand

ROOT = Path(__file__).resolve().parent
PLAN_REVISION = '0affb6059cb724fb5eaba4ec284bd2fb07cfdc25f045eff80327b48dbd624655'
TEMPLATE_REVISION = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def source_hashes() -> dict[str, str]:
    result = locality_hashes()
    for name in ('noncompact_maxwell_phase.py', 'noncompact_maxwell_phase.md',
                 'original_mass_density.py', 'noncompact_charge_threshold.py',
                 'noncompact_charge_threshold.md', 'noncompact_microscopic_charge_window.py',
                 'noncompact_microscopic_charge_window.md', 'original_slow_kinetic_bo.py',
                 'original_slow_kinetic_bo.md'):
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return result


def phase_constants(schur_lower: float = .01, speed: float = .1) -> dict[str, float]:
    """Explicit conditional family; numerical evaluation is not outward rounding.

    a=1,m=.2,r=.12,v=.3,kappa=2. The Schur lower bound is an explicit
    metric-family hypothesis, never inferred from the CLI's finite samples.
    The quartic/whole-half trace-norm supplier gives the Gram norm upper bound.
    """
    require(schur_lower > 0 and speed > 0, 'strict positive metric floor and bare speed')
    m, r, v = .2, .12, .3
    cx = math.sqrt((m + 6*r)**2 + 3*v*v)
    tstar = (2*math.sqrt(2)*cx)**(2/3)
    require(schur_lower <= tstar, 'Schur lower hypothesis compatible with Gram upper')
    mu = m*math.sqrt(schur_lower)
    ch = 3*(r + abs(v))*math.sqrt(tstar)
    b = math.sqrt(tstar)*(m + 6*r + 3*abs(v))
    gamma, xi = mu/4, mu/4
    alpha = math.log1p(xi/ch)
    # Uniform upper bounds on Appendix A's tilted onsite odd part and c.
    d0 = (m + 3*r)*math.sqrt(tstar)
    c_upper = (m + 3*r)*math.sqrt(tstar)
    valpha = d0 + ch*math.exp(alpha)
    qtail = max(c_upper, math.log(2), gamma + 8*valpha, gamma + 4)
    # qtail-c <= qtail: using the larger interval and operator bound is safe.
    mtail_upper = qtail
    delta = min(1/8, 1/(4*(b + mtail_upper + gamma + xi)))
    cbar = math.e/(delta*(mu-gamma-xi))
    p = 16/gamma*(1 + 2/math.expm1(alpha/2))**3
    jet = max((r + abs(v))*math.sqrt(tstar)/2, 1.)
    hlow = (16*(2*cbar)**3*jet**2*math.exp(6*alpha)*p*p
            + 8*(2*cbar)**2*jet*math.exp(4*alpha)*p)
    hell = mtail_upper*hlow + (16384*valpha**2 + 2048)/(qtail-gamma)
    # sup_{x>=0}(1+x)^4 exp(-alpha*x), with the boundary included.
    x4 = max(0., 4/alpha-1)
    p4 = (1+x4)**4*math.exp(-alpha*x4)
    area = 64.
    ce0 = 8*hell/(math.e**2*gamma**2)
    cb0 = 2*area**2*hell*p4
    tau = max(1., (alpha+1)/gamma)
    beta, temporal_rate = alpha/64, gamma/64
    xa, xt = max(0., 48/alpha-1), max(0., 48/gamma-1)
    log_p12a = 12*math.log1p(xa)-alpha*xa/4
    log_p12t = 12*math.log1p(xt)-gamma*xt/4
    log_r0 = 20*math.log(2)+math.log(hell)+alpha/2+log_p12a+log_p12t
    xs = max(5., 2/beta)
    spatial_max = xs*xs*math.exp(-beta*xs)
    temporal_max = 1/(math.e*temporal_rate*tau)
    log_seam = log_r0+math.log((spatial_max+temporal_max)/8)
    require(log_seam < math.log(sys.float_info.max)-8,
            'declared family constants fit float64; use symbolic formulas for smaller floors')
    seam = math.exp(log_seam)
    ce, cb = 2*ce0 + seam, 2*cb0 + seam
    he = 16*max(ce, cb/(speed*speed), 1.)
    hb, ue = speed*speed*he, 1/(2*he)
    rho = max(ce/he, cb/hb)
    require(ue > 0 and math.isfinite(hb) and rho <= 1/16*(1+1e-14),
            'fixed positive bare family with strict Maxwell dominance')
    return dict(a=1., m=m, r=r, v=v, kappa=2., schur_lower=schur_lower,
                Gram_Frobenius_upper=tstar, mu_lower=mu, Ch_upper=ch, B_upper=b,
                gamma=gamma, alpha=alpha, xi=xi, T0=1., q_tail=qtail,
                M_tail_upper=mtail_upper, delta_low=delta, Cbar=cbar,
                H_density=hell, C_area=area, P4=p4, CE_local=ce0, CB_local=cb0,
                seam_tail_prefactor=math.exp(log_r0),
                tau=tau, seam_uniform_upper=seam, log_seam_uniform_upper=log_seam,
                CE=ce, CB=cb, UE=ue, K=hb, electric_Hessian=he,
                bare_speed=speed, relative_Hessian_bound=rho,
                covariance_lower_ratio=1/(1+rho), covariance_upper_ratio=1/(1-rho),
                spectral_support_speed_upper=speed*(1+rho)/(1-rho))


def seam_allowance(size: int, period: float, constants: dict[str, float]) -> float:
    """Finite torus/circle tail INCLUDING cross terms and retained contact atoms."""
    return constants['seam_tail_prefactor']*(
        math.exp(-constants['alpha']*size/64)
        + math.exp(-constants['gamma']*period/64))


def curl(law: BulkQuantumLaw, source: np.ndarray) -> np.ndarray:
    columns = []
    for j, k in ((0, 1), (0, 2), (1, 2)):
        columns.append(source[:, j] + source[law.target[:, j], k]
                       - source[law.target[:, k], j] - source[:, k])
    return np.column_stack(columns)


def gradient(law: BulkQuantumLaw, site_source: np.ndarray) -> np.ndarray:
    return site_source[:, None] - site_source[law.target]


def native_jets(law: BulkQuantumLaw, h: np.ndarray, spatial: np.ndarray,
                *, omit_seams: bool = False) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.zeros_like(h), np.zeros_like(h)
    for x in range(law.volume):
        for j in range(3):
            if omit_seams and law.sites[x, j] == law.size-1:
                continue
            y = int(law.target[x, j])
            sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
            weight = spatial[x, j]
            first[sx, sy] += 1j*weight*h[sx, sy]
            first[sy, sx] -= 1j*weight*h[sy, sx]
            second[sx, sy] -= weight*weight*h[sx, sy]
            second[sy, sx] -= weight*weight*h[sy, sx]
    return first, second


def ordered_hessian(hamiltonians: list[np.ndarray], temporal: np.ndarray,
                    durations: np.ndarray, first: list[np.ndarray],
                    second: list[np.ndarray], temporal_probe: np.ndarray) -> dict[str, Any]:
    """Differentiate the actual ordered monodromy, not an equal-time resolvent.

    The upper triangular exponential carries Taylor jets in the SAME native
    generator. expm_frechet separately isolates the true local contact.
    """
    n = len(hamiltonians[0])
    u = np.eye(n, dtype=complex)
    du, ddu, contact_u = np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)
    prefixes = [u.copy()]
    for h, a0, dt, j, contact in zip(hamiltonians, temporal, durations, first, second):
        x = dt*(-h + 1j*np.diag(np.repeat(a0, 4)))
        dx = dt*(-j + 1j*np.diag(np.repeat(temporal_probe, 4)))
        ddx = -dt*contact
        block = np.zeros((3*n, 3*n), dtype=complex)
        for i in range(3):
            block[i*n:(i+1)*n, i*n:(i+1)*n] = x
        block[:n, n:2*n], block[n:2*n, 2*n:] = dx, dx
        block[:n, 2*n:] = ddx/2
        evolution = expm(block)
        e, de, dde = evolution[:n, :n], evolution[:n, n:2*n], 2*evolution[:n, 2*n:]
        if np.any(ddx):
            ec = expm_frechet(x, ddx, compute_expm=False)
        else:
            ec = np.zeros_like(u)
        contact_u = ec@u + e@contact_u
        ddu, du, u = dde@u + 2*de@du + e@ddu, de@u + e@du, e@u
        prefixes.append(u.copy())
    q = np.linalg.solve(np.eye(n) + u, np.eye(n))
    first_value = np.trace(q@du) - 2j*float(durations.sum())*float(temporal_probe.sum())
    full = np.trace(q@ddu - q@du@q@du)
    contact_value = np.trace(q@contact_u)
    residual(abs(float(full.imag))/max(1., abs(full.real)), 'real physical source Hessian', 2e-7)
    residual(abs(float(first_value.imag))/max(1., abs(first_value.real)),
             'background-completed real physical first source', 2e-7)
    return dict(first=float(first_value.real), full=float(full.real),
                contact=float(contact_value.real), bubble=float((full-contact_value).real),
                monodromy=u, source=q, prefixes=prefixes)


def first_source_response(hamiltonians: list[np.ndarray], temporal: np.ndarray,
                          durations: np.ndarray, first: list[np.ndarray],
                          temporal_probe: np.ndarray) -> float:
    """Physical first response without subtracting an extensive logdet."""
    n = len(hamiltonians[0])
    u, du = np.eye(n, dtype=complex), np.zeros((n, n), dtype=complex)
    for h, a0, dt, j in zip(hamiltonians, temporal, durations, first):
        x = dt*(-h + 1j*np.diag(np.repeat(a0, 4)))
        dx = dt*(-j + 1j*np.diag(np.repeat(temporal_probe, 4)))
        e, de = expm_frechet(x, dx)
        du, u = de@u + e@du, e@u
    value = np.trace(np.linalg.solve(np.eye(n)+u, du))
    value -= 2j*float(durations.sum())*float(temporal_probe.sum())
    residual(abs(float(value.imag))/max(1., abs(value.real)),
             'background-completed independent first response', 2e-7)
    return float(value.real)


def physical_logdet(hamiltonians: list[np.ndarray], temporal: np.ndarray,
                    durations: np.ndarray) -> float:
    u, _, phase = time_evolution(hamiltonians, temporal, durations)
    sign, value = np.linalg.slogdet(np.eye(len(u)) + u)
    residual(abs(complex(sign*np.exp(-1j*phase))-1),
             'same-background-two positive native determinant', 2e-7)
    return float(value)


def native_canonical_boundary(hamiltonians: list[np.ndarray], temporal: np.ndarray,
                              durations: np.ndarray, frame: np.ndarray,
                              gram: np.ndarray, green: ContinuousGreen) -> dict[str, Any]:
    """Actual centred U1/global-Haar density on noncommuting native histories.

    Positivity uses fixed beta AND chirality, not a one-body gap or positivity
    of individual heat factors. Arbitrary SU4 boundary twists are not admitted.
    """
    beta4 = chiral_basis(frame, gram, 1)[2]
    values, vectors = np.linalg.eigh(gram)
    inverse_root = (vectors/np.sqrt(values)[None, :])@vectors.T
    gammas = np.einsum('ij,jab->iab', inverse_root, frame)
    j4 = gammas[0]@gammas[1]@gammas[2]@gammas[3]
    n = len(hamiltonians[0])
    volume = n//4
    j = np.kron(np.eye(volume), j4)
    beta = np.kron(np.eye(volume), beta4)
    chirality_errors = [float(np.linalg.norm(j@h+h@j)) for h in hamiltonians]
    residual(max(chirality_errors), 'actual native chirality on EVERY noncommuting time slice')
    noncommuting = max(float(np.linalg.norm(a@b-b@a))
                      for a, b in zip(hamiltonians[:-1], hamiltonians[1:]))
    require(noncommuting > 1e-4, 'actual native histories are not a commuting heat-trace shortcut')
    schur = float(gram[0, 0]-gram[0, 1:]@np.linalg.solve(gram[1:, 1:], gram[1:, 0]))
    mu = .2*math.sqrt(schur)
    coercive = []
    for h, a0 in zip(hamiltonians, temporal):
        weighted = beta@(h-1j*np.diag(np.repeat(a0, 4)))
        coercive.append(float(np.linalg.eigvalsh((weighted+weighted.conj().T)/2)[0]))
    require(min(coercive) >= mu-2e-8, 'beta D = positive M plus anti-Hermitian part')
    u = green.u
    pseudo_error = float(np.linalg.norm(u.conj().T@j@u-j)/max(1., np.linalg.norm(u)**2))
    residual(pseudo_error, 'native ordered evolution J-pseudounitarity')
    # All 4V eigenvalues retained. This is an independent finite Fourier
    # evaluation of the canonical coefficient, not a selected filled band.
    spectrum = np.linalg.eigvals(u)
    period = float(durations.sum())
    radii = abs(spectrum)
    dichotomy_error = float(np.max(np.minimum(
        np.maximum(0., radii-math.exp(-mu*period)),
        np.maximum(0., math.exp(mu*period)-radii))))
    residual(dichotomy_error, 'native Floquet no-unit-circle dichotomy', 2e-7)
    require(int(np.sum(radii > 1)) == 2*volume, 'all-mode chiral pairing has exactly 2V large modes')
    count = 4*volume+1
    angles = 2*math.pi*np.arange(count)/count
    factors = 1+np.exp(1j*angles[:, None])*spectrum[None, :]
    logs = np.sum(np.log(abs(factors)), axis=1)
    phase_angles = (np.sum(np.angle(factors), axis=1)
                    -green.background_phase-2*volume*angles)
    phases = np.exp(1j*phase_angles)
    phase_error = float(np.max(abs(phases-1)))
    residual(phase_error, 'EVERY native discrete global-Haar node has positive centred density', 2e-7)
    direct_errors = []
    for index in np.linspace(0, count-1, 9, dtype=int):
        angle = float(angles[index])
        sign, logabs = np.linalg.slogdet(np.eye(n)+np.exp(1j*angle)*u)
        direct_phase = complex(sign*np.exp(-1j*(green.background_phase+2*volume*angle)))
        residual(abs(direct_phase-1), 'direct native centred-holonomy determinant positivity', 2e-7)
        direct_errors.append(abs(float(logabs)-float(logs[index])))
    residual(max(direct_errors), 'all-mode Fourier factors versus independently direct native determinants', 2e-7)
    maximum = float(logs.max())
    scaled = np.mean(phases*np.exp(logs-maximum))
    residual(abs(float(scaled.imag))/max(1e-100, abs(float(scaled.real))),
             'actual full native canonical Haar coefficient is real', 2e-7)
    require(float(scaled.real) > 0, 'actual full native canonical coefficient positive')
    canonical_log = maximum+math.log(float(scaled.real))
    uncentred_error = abs(complex(np.exp(1j*(green.background_phase)))-1)
    uncentred_failure = rejected(lambda: residual(uncentred_error,
        'direct native centred-holonomy determinant positivity', 2e-7), 'omitted canonical background two')
    # A native onsite colour twist demonstrates the forbidden boundary.
    # J4 lies in SU4 but anticommutes with beta. It is NOT a compatible
    # fixed-body field Haar path; treating it as an admitted temporal U1 kick
    # would make a strictly positive mass appear to guarantee a false density.
    onsite = (.2+3*.12)*frame[0]
    colour_u = j4@expm(-period*onsite)
    scalar_u = np.exp(.37j)*expm(-period*onsite)
    colour_singular = float(np.linalg.svd(np.eye(4)+colour_u, compute_uv=False)[-1])
    scalar_singular = float(np.linalg.svd(np.eye(4)+scalar_u, compute_uv=False)[-1])
    residual(float(np.linalg.norm(j4.conj().T@j4-np.eye(4))), 'actual forbidden twist unitary')
    residual(abs(complex(np.linalg.det(j4))-1), 'actual forbidden twist is SU4')
    require(scalar_singular > 1e-8, 'compatible scalar temporal twist positive-control has no zero')
    require(colour_singular < 1e-10, 'native gapped onsite law with generic SU4 twist can have a determinant zero')
    colour_failure = rejected(lambda: require(colour_singular > 1e-8,
        'admitted positive temporal density has no determinant zero'), 'blind generic-SU4 positivity')
    return dict(native_modes=n, noncommuting_history_commutator=noncommuting,
                beta_coercive_minimum=min(coercive), mu=mu,
                chirality_residual=max(chirality_errors), pseudo_unitarity_residual=pseudo_error,
                Floquet_dichotomy_error=dichotomy_error, Haar_nodes=count,
                Haar_rule='4V+1 exact roots select the entire 2V exterior coefficient',
                centred_node_phase_error=phase_error,
                direct_factor_log_errors=direct_errors, canonical_log_trace=canonical_log,
                grandcanonical_minus_canonical_log=float(logs[0])-canonical_log,
                omitted_background_failure=uncentred_failure,
                invalid_SU4_twist=dict(onsite_native_gap=float(np.min(abs(np.linalg.eigvalsh(onsite)))),
                    beta_commutator=float(np.linalg.norm(j4@beta4-beta4@j4)),
                    positive_scalar_twist_singular=scalar_singular,
                    colour_twist_singular=colour_singular, rejected=colour_failure,
                    scope='forbidden fixed-body boundary, not a legitimate full-field Haar history'),
                conditional_density='positive heat/Coulomb Gaussian times centred native determinant, conditioned on actual harmonic history and U1 Haar node',
                full_field_scope='heavy and field Haar integrated only by the actual isometric W preparation; no generic full-field determinant positivity asserted')


def native_case(size: int, centre: tuple[int, int, int], constants: dict[str, float],
                epsilon: float, seed: int, *, density_only: bool = False) -> dict[str, Any]:
    require(size >= 5, 'genuine three-axis N>=5 native graph')
    band = RotationBand(a=1., m=.2, r=.12, v=.3, improved=False)
    selected, frame = original_frame(band, size, np.asarray(centre)*math.pi/2, True)
    gram = np.asarray(selected['G'], dtype=float)
    law = BulkQuantumLaw(size, band=band)
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.asarray(centre)*math.pi/2)
    rng = np.random.default_rng(seed)
    phases = rng.normal(size=(3, law.volume, 3))*3.7
    temporal = rng.normal(size=(3, law.volume))*.23
    durations = np.asarray([.43, .61, .47])
    # Force a resolved background contact; no normalization phase is suppressed.
    temporal[0, 0] += (.71-2*np.sum(durations[:, None]*temporal))/(2*durations[0])
    h = [np.asarray(law.onebody(fields, links, a), dtype=complex) for a in phases]
    green = ContinuousGreen(h, temporal, durations)
    identities = green.identity_controls()
    gauge = native_gauge_controls(law, fields, links, phases, green, rng)
    canonical_density = native_canonical_boundary(h, temporal, durations, frame, gram, green)
    if density_only:
        return dict(size=size, modes=4*law.volume, spatial_dimension=3, centre=centre, Gram=gram,
                    phases_sha256=array_hash(phases), temporal_sha256=array_hash(temporal),
                    durations=durations, Green_source_identification=identities,
                    native_seam_Ward_controls=gauge, native_full_canonical_positive_density=canonical_density,
                    scope='actual native noncommuting-history positive canonical density only; no curvature probes or all64 samples in this mode')
    all64 = []
    for theta in np.ndindex(4, 4, 4):
        native, _ = original_frame(band, size, np.asarray(theta)*math.pi/2, True)
        g = np.asarray(native['G'], dtype=float)
        schur = float(g[0, 0]-g[0, 1:]@np.linalg.solve(g[1:, 1:], g[1:, 0]))
        require(schur >= constants['schur_lower'], 'sampled native metric satisfies DECLARED floor')
        require(float(np.linalg.norm(g)) <= constants['Gram_Frobenius_upper']*(1+1e-10),
                'sampled native Gram satisfies source-owned uniform upper')
        all64.append(dict(centre=theta, Schur=schur, Gram_norm=float(np.linalg.norm(g))))
    angle = 2*math.pi*law.sites[:, 0]/size
    magnetic = np.zeros((law.volume, 3))
    magnetic[:, 1] = np.cos(angle)/math.sqrt(law.volume/2)
    electric = np.sin(angle)/math.sqrt(law.volume/2)
    chi = np.cos(angle+.37)/math.sqrt(law.volume/2)
    ward = gradient(law, chi)
    zero_a = np.zeros_like(magnetic)
    zero_0 = np.zeros(law.volume)
    probe_rows = []
    base = physical_logdet(h, temporal, durations)
    for name, spatial, temporal_probe in (
        ('native_transverse_magnetic', magnetic, zero_0),
        ('native_longitudinal_electric', zero_a, electric),
        ('native_pure_gradient_Ward', ward, zero_0),
    ):
        jets = [native_jets(law, matrix, spatial) for matrix in h]
        first = [pair[0] for pair in jets]
        second = [pair[1] for pair in jets]
        actual = ordered_hessian(h, temporal, durations, first, second, temporal_probe)
        residual(np.linalg.norm(actual['monodromy']-green.u)/max(1., np.linalg.norm(green.u)),
                 'differentiated evolution has the independently supplied native endpoint', 2e-7)
        # Independent source: rerun onebody and ordered evolution at +/-epsilon.
        values, responses = [], []
        for sign in (-1., 1.):
            shifted = [np.asarray(law.onebody(fields, links, a+sign*epsilon*spatial), dtype=complex)
                       for a in phases]
            values.append(physical_logdet(shifted, temporal+sign*epsilon*temporal_probe, durations))
            shifted_first = [native_jets(law, matrix, spatial)[0] for matrix in shifted]
            responses.append(first_source_response(
                shifted, temporal+sign*epsilon*temporal_probe, durations,
                shifted_first, temporal_probe))
        logdet_differentiated = (values[0]-2*base+values[1])/(epsilon*epsilon)
        differentiated = (responses[1]-responses[0])/(2*epsilon)
        # The matter response owns these units, not an absolute floor or the
        # enormously larger bare Maxwell action. Ward uses its contact scale.
        scale = (max(1., abs(actual['contact'])) if name.endswith('Ward') else
                 max(abs(actual['full']), abs(differentiated), np.finfo(float).tiny))
        derivative_error = abs(differentiated-actual['full'])/scale
        residual(derivative_error, 'independently differentiated SAME native full-source Hessian', 4e-5)
        period = float(durations.sum())
        bnorm = period*float(np.linalg.norm(curl(law, spatial))**2)
        enorm = period*float(np.linalg.norm(gradient(law, temporal_probe))**2)
        qmin2 = 4*math.sin(math.pi/size)**2
        tail = seam_allowance(size, period, constants)*(1+1/qmin2)
        ce_here = 2*constants['CE_local']+tail
        cb_here = 2*constants['CB_local']+tail
        require(ce_here <= constants['CE'] and cb_here <= constants['CB'],
                'finite native time-circle seam fits the declared conservative bare family')
        envelope = ce_here*enorm + cb_here*bnorm
        bare = constants['electric_Hessian']*enorm + constants['K']*bnorm
        if name.endswith('Ward'):
            residual(abs(actual['full'])/max(1., abs(actual['contact'])),
                     'physical pure-gradient second Ward derivative', 2e-7)
            residual(float(np.linalg.norm(curl(law, spatial))), 'actual periodic gradient has zero curl', 1e-12)
            no_contact = rejected(lambda: residual(abs(actual['bubble'])/max(1., abs(actual['contact'])),
                         'physical pure-gradient second Ward derivative', 2e-7), 'missing native contact')
            wrong_first, _ = native_jets(law, h[0], spatial, omit_seams=True)
            repeated = np.repeat(chi, 4)
            correct = 1j*(repeated[:, None]-repeated[None, :])*h[0]
            seam_failure = rejected(lambda: residual(float(np.linalg.norm(wrong_first-correct)/max(1., np.linalg.norm(correct))),
                                   'actual periodic pure-gradient current vertex', 2e-7), 'omitted spatial seams')
            extra = dict(missing_contact_failure=no_contact, omitted_spatial_seam_failure=seam_failure)
        else:
            require(bare > 0, 'actual field-strength source has nonzero bare Maxwell action')
            # Keep the tiny correction separate: subtracting it from 1 would round it away.
            relative_correction = actual['full']/bare
            require(abs(actual['full']) <= envelope, 'actual native Hessian below PAID Maxwell envelope')
            require(abs(relative_correction) <= constants['relative_Hessian_bound'],
                    'actual native determinant leaves positive Maxwell-relative source Hessian')
            omitted_source = rejected(lambda: residual(abs(differentiated)/scale,
                             'independently differentiated SAME native full-source Hessian', 4e-5),
                             'Gaussian-only zero matter response')
            extra = dict(bare_Maxwell_Hessian=bare, paid_determinant_envelope=envelope,
                         signed_relative_determinant_correction=relative_correction,
                         log_abs_relative_correction=math.log(abs(actual['full']))-math.log(bare),
                         Gaussian_only_failure=omitted_source,
                         linearized_relative_inverse_correction=relative_correction/(1-relative_correction))
        probe_rows.append(dict(name=name, electric_source_norm2=enorm, magnetic_source_norm2=bnorm,
                               actual_first=actual['first'], actual_full_Hessian=actual['full'],
                               actual_contact=actual['contact'], actual_bubble=actual['bubble'],
                               independent_Hessian=differentiated, independent_relative_error=derivative_error,
                               logdet_second_difference=logdet_differentiated,
                               independent_method='centered difference of exact physical first-source response',
                               spatial_source_sha256=array_hash(spatial), temporal_source_sha256=array_hash(temporal_probe),
                               **extra))
    beta4 = chiral_basis(frame, gram, 1)[2]
    beta = np.kron(np.eye(law.volume), beta4)
    coercivity = min(float(np.linalg.eigvalsh((beta@matrix+matrix@beta)/2)[0]) for matrix in h)
    require(coercivity >= constants['mu_lower']-2e-8, 'actual native fixed-beta coercivity')
    return dict(size=size, spatial_dimension=3, modes=4*law.volume, centre=centre, Gram=gram,
                native_gradient=float(selected['gradient_residual']), actual_beta_coercivity=coercivity,
                phases_sha256=array_hash(phases), temporal_sha256=array_hash(temporal),
                unwrapped_phase_range=[float(phases.min()), float(phases.max())], durations=durations,
                Green_source_identification=identities, native_seam_Ward_controls=gauge,
                native_full_canonical_positive_density=canonical_density,
                all64_metric_samples=all64, probes=probe_rows,
                independent_complete_CAR_canonical=canonical_controls(frame, band, rng),
                scope='native exact conditional current Hessians, not a sampled photon measure; all64 samples are not an infinite-family proof')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--centre', type=int, nargs=3, default=[1, 2, 3])
    parser.add_argument('--schur-lower', type=float, default=.01)
    parser.add_argument('--bare-speed', type=float, default=.1)
    parser.add_argument('--epsilon', type=float, default=.002)
    parser.add_argument('--density-only', action='store_true',
                        help='Run the native canonical-positivity boundary without the expensive source-Hessian slice.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(args.epsilon > 0 and all(0 <= value < 4 for value in args.centre), 'legal native CLI inputs')
    before = source_hashes()
    constants = phase_constants(args.schur_lower, args.bare_speed)
    native = native_case(args.size, tuple(args.centre), constants, args.epsilon, 2221046,
                         density_only=args.density_only)
    require(before == source_hashes(), 'all native scientific source/proof bindings unchanged during run')
    result = dict(plan='issue-222-continuing-backward-20260930', assignment_revision=PLAN_REVISION,
                  template_revision=TEMPLATE_REVISION, work_item='campaign', requirements=['outcome', 'verification'],
                  runtime=dict(python=platform.python_version(), executable=sys.executable,
                               numpy=np.__version__, scipy=scipy.__version__), source_hashes=before,
                  phase_constants=constants, native=native, all_checks=True,
                  mode='canonical_density_only' if args.density_only else 'density_and_current_Hessian',
                  phase_curvature_exercised=not args.density_only,
                  theorem='conditional fixed-a uniform-Schur noncompact zeroT k!=0 Maxwell-comparable response; true canonical/harmonic measure retained',
                  claims=dict(sharp_photon_pole=False,_exact_renormalized_speed=False,
                              compact_phase=False, fixed_Lambda_uniform_BO=False,
                              finite_native_measure_sampling=False))
    rendered = json.dumps(portable(result), indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(rendered+'\n')
    print(rendered)


if __name__ == '__main__':
    main()
