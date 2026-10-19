#!/usr/bin/env python3
"""Cutoff-scaled canonical/harmonic positive photon family and exact unit consumer.

The weighted beta-cone/anchored kernel theorem belongs to the Wilson supplier.
This construction consumes it in dimensionless aH units, supplies a uniform
true-Gram admission, pays all seams, and converts to physical volume/field units.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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

from bulk_cutoff_matching import selected_gram
from bulk_quantum_selection import BulkQuantumLaw
from noncompact_maxwell_phase import native_jets
from noncompact_wilson_photon_envelope import primitive_wilson_energy_envelope
from original_charged_principal import original_frame
from original_compact_charge_bound import chiral_basis
from original_massive_photon_locality import ContinuousGreen
from rotation_band import RotationBand

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-synthesis-continuation-20261006'
TEMPLATE = '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe'


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def close(actual: Any, expected: Any, label: str, tolerance: float = 3e-8) -> float:
    error = float(np.linalg.norm(np.asarray(actual)-np.asarray(expected))/max(1., np.linalg.norm(expected)))
    require(math.isfinite(error) and error <= tolerance, f'{label}: {error}')
    return error


def rejected(callback: Any, label: str) -> str:
    try:
        callback()
    except AssertionError as failure:
        return f'{label}: {failure}'
    raise AssertionError(f'Wrong construction passed: {label}')


def metric_admission(a: float) -> dict[str, Any]:
    """Exact shifted-grid moment admission, not a fitted tighter metric box."""
    require(math.isfinite(a) and 0 < a <= .01, 'fixed near-Dirac cutoff 0<a<=.01')
    m, r, v = a*.2, .12, .3
    cx = math.sqrt((m+6*r)**2+3*v*v)
    upper = (2*math.sqrt(2)*cx)**(2/3)
    moments = np.diag([(m+3*r)**2+1.5*r*r, v*v/2, v*v/2, v*v/2])
    lower = moments/(2*math.sqrt(upper)*cx)
    return dict(a=a, dimensionless_mass=m, Cx=cx, Gram_Frobenius_upper=upper,
        exact_shifted_second_moment=moments.tolist(), Gram_Loewner_lower=lower.tolist(),
        Schur_floor=float(lower[0, 0]), G00_upper=upper, Gjj_upper=upper,
        N_metric=3, admission='full convex stationary Gram + exact N>=3 shifted trigonometric moments; all64 true wells')


def logsum(x: float, y: float) -> float:
    return max(x, y)+math.log1p(math.exp(-abs(x-y)))


def positive_family(a: float, speed: float = .1) -> dict[str, Any]:
    """Select the physical family from source-marked Wilson-energy moments."""
    require(math.isfinite(speed) and speed > 0, 'positive prescribed physical bare speed')
    admission = metric_admission(a)
    m, r, v = a*.2, .12, .3
    floor, upper = admission['Schur_floor'], admission['Gram_Frobenius_upper']
    tilt_root = math.sqrt(upper-floor)
    candidates = []
    for fraction in (.1, .2, .25, .3, .35, .4):
        core = primitive_wilson_energy_envelope(size_floor=64, m_hat=m, r=r, v=v, s_lower=floor,
            g00_upper=upper, spatial_diagonal_upper=(upper, upper, upper),
            tilt_root_upper=tilt_root, xi_fraction=fraction)
        mu, ch, odd = core['mu'], core['Ch'], core['odd_row']
        jet, anchor_jet = core['native_jet'], core['odd_jet']
        xi, alpha, gamma = core['xi'], core['alpha'], core['gamma']
        radius, cinv = core['cone_radius'], core['cone_angle_inverse']
        point, t0, h = core['point_kernel_bound'], core['T0'], core['H']
        triangle, contact = core['triangle'], core['native_contact']
        anchor_first, anchor_contact = core['anchor_first'], core['anchor_contact']
        ce0, cb0 = core['CE_bulk'], core['CB_bulk']
        mass_jet2, mass_contact = core['mass_only_jet_squared'], core['mass_only_contact']
        epsilon = max(ce0, cb0)/100
        tau = max(t0, (alpha+1)/gamma)
        logr = core['log_Rstar']
        bs, bt = alpha/64, gamma/64
        def tails(n: int) -> tuple[float, float]:
            seam = logr+2*math.log(n)-math.log(8)+logsum(-bs*n, -bt*tau*n*n)
            period = tau*n*n
            thermal = 5*math.log(n)+math.log(period*mass_jet2+mass_contact)-mu*period
            return seam, thermal
        onset = max(64, math.ceil(2/bs), math.ceil(math.sqrt(1/(bt*tau))),
                    math.ceil(math.sqrt(7/(2*mu*tau))))
        lo, hi = onset, onset
        while logsum(*tails(hi)) > math.log(epsilon):
            hi *= 2
        if logsum(*tails(lo)) <= math.log(epsilon):
            hi = lo
        else:
            while hi-lo > 1:
                mid = (hi+lo)//2
                if logsum(*tails(mid)) <= math.log(epsilon):
                    hi = mid
                else:
                    lo = mid
        ce, cb = ce0+epsilon, cb0+epsilon
        he = 16*max(ce, cb/(speed*speed), 1.)
        ue, magnetic = 1/(2*he), speed*speed*he
        rho = max(ce/he, cb/magnetic)
        wick = 128*point*point*jet*jet*3*math.exp(4*alpha)*(1+2/math.expm1(alpha))**3
        x2 = max(0., 2/alpha-1)
        p2 = (1+x2)**2*math.exp(-alpha*x2)
        score = 2*(8*h)**2*ue*((2*math.sqrt(3)/speed)*(128*p2)**2+1/(math.e*gamma))
        # ZeroT is taken at each N for this current capability. The mass-only
        # thermal remainder then vanishes; the remaining spatial score seam is paid.
        log_score_seam = (math.log(ue)+math.log(hi/(4*speed)+hi*hi/8)
                          +2*logr-2*bs*hi)
        score_seam = math.exp(log_score_seam)
        cj = wick+(score+score_seam)/(1-rho)
        candidates.append(dict(a=a, metric_admission=admission, speed=speed,
            units='all hatted moments use u=t/a, hhat=a hphys, theta=a Aphys, bhat=a a0phys',
            kernel_construction='source-marked Wilson-energy envelope; finite spatial zero mode retained',
            kernel_size_floor=64, cone_point_comparator=core['cone_point_candidate'],
            H_gain_at_same_inputs=core['H_gain'],
            mass_inverse_diagonal_upper=core['mass_inverse_diagonal_upper'],
            mass_diagonal_zero_mode_part=core['mass_diagonal_zero_mode_part'],
            source_interval_hat=core['source_interval'],
            mu_hat=mu, mu_physical=mu/a, odd_row_hat=odd, Ch_hat=ch, J_hat=jet,
            anchor_jet_hat=anchor_jet, xi_hat=xi, alpha_site=alpha,
            alpha_physical=alpha/a, gamma_hat=gamma, gamma_physical=gamma/a,
            cone_radius=radius, cone_angle_inverse=cinv, point_kernel=point,
            T0_hat=t0, H_hat=h, H_raw_physical=h/a,
            moment_terms=dict(triangle=triangle, contact=contact,
                              anchor_first=anchor_first, anchor_contact=anchor_contact),
            CE_bulk=ce0, CB_bulk=cb0, CE_physical=ce, CB_physical=cb,
            CE_raw_physical=a*ce, CB_raw_physical=cb/a,
            seam_and_mass_budget=epsilon, log_Rstar_hat=logr,
            mass_jet_squared_hat=mass_jet2, mass_contact_hat=mass_contact,
            N0=hi, tau_hat=tau, tau_physical=a*tau,
            log_paid_tail_at_N0=logsum(*tails(hi)), hE_physical_volume=he,
            UE_hat=ue, K_hat=magnetic, UE_physical=ue/a, K_physical=magnetic/a,
            bulk_UE1_physical=2*ue/a, bulk_K1_physical=magnetic/a,
            rho=rho, covariance_lower_ratio=1/(1+rho), covariance_upper_ratio=1/(1-rho),
            bare_field_g2=1/(he*speed), fixed_coupling_UV_claim=False,
            current_variance=dict(C_Wick_hat=wick, C_score_bulk_hat=score,
                C_score_spatial_seam_hat=score_seam, C_J_hat=cj,
                C_J_link_physical=cj/a**2, C_J_volume_mode_physical=cj/a**3,
                scope='effective true canonical/harmonic zeroT state; inherited instantaneous incidence argument; not a finite simulation or microscopic Gibbs equality'),
            xi_fraction=fraction))
    return max(candidates, key=lambda row: row['UE_hat'])


def packet_bounds(family: dict[str, Any], omega_min: float,
                  omega_max: float, physical_time: float) -> dict[str, float]:
    """Physical-volume CCR mode and same-current finite-resolution capability."""
    require(0 < omega_min <= omega_max and math.isfinite(omega_min+omega_max+physical_time),
            'finite nonzero physical photon frequency band')
    a, ue = family['a'], family['UE_hat']
    cj = family['current_variance']['C_J_volume_mode_physical']
    debt = ue*cj/omega_min**3
    residual = math.sqrt(ue*cj/omega_min)
    return dict(omega_min_physical=omega_min, omega_max_physical=omega_max,
        observation_time_physical=physical_time, observation_time_hat=physical_time/a,
        omega_min_hat=a*omega_min, CCR_mode_kinetic_coefficient=ue,
        current_variance_physical_volume=cj, annihilation_debt_upper=debt,
        creation_Gram_lower=1., creation_Gram_upper=1+debt,
        physical_energy_residual_upper=residual,
        finite_time_unnormalized_defect_upper=abs(physical_time)*residual,
        prepared_physical_energy_upper=omega_max*(1+debt)+residual)


def ordered_jets(hs: list[np.ndarray], temporal: np.ndarray, times: np.ndarray,
                 first: list[np.ndarray], contacts: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Actual ordered Taylor jets, retaining the native contact separately."""
    n = len(hs[0])
    u = np.eye(n, dtype=complex)
    du, ddu, cu = np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)
    for h, b, dt, j, c in zip(hs, temporal, times, first, contacts):
        generator = dt*(-h+1j*np.diag(np.repeat(b, 4)))
        dg, ddg = -dt*j, -dt*c
        block = np.zeros((3*n, 3*n), dtype=complex)
        for k in range(3):
            block[k*n:(k+1)*n, k*n:(k+1)*n] = generator
        block[:n, n:2*n], block[n:2*n, 2*n:] = dg, dg
        block[:n, 2*n:] = ddg/2
        eblock = np.asarray(expm(block))
        e, de, dde = eblock[:n, :n], eblock[:n, n:2*n], 2*eblock[:n, 2*n:]
        ec = np.asarray(expm_frechet(generator, ddg, compute_expm=False))
        cu = ec@u+e@cu
        ddu, du, u = dde@u+2*de@du+e@ddu, de@u+e@du, e@u
    return dict(u=u, du=du, ddu=ddu, contact=cu)


def canonical_response(jets: dict[str, np.ndarray], background: float) -> dict[str, float]:
    """Complete n/2 Haar coefficient, not a grand-canonical point response."""
    u, du, ddu, contact = (jets[key] for key in ('u', 'du', 'ddu', 'contact'))
    n = len(u)
    spectrum = np.linalg.eigvals(u)
    angles = 2*math.pi*np.arange(n+1)/(n+1)
    factors = 1+np.exp(1j*angles[:, None])*spectrum[None, :]
    logs = np.log(abs(factors)).sum(axis=1)
    phases = np.exp(1j*(np.angle(factors).sum(axis=1)-background-n/2*angles))
    phase_error = close(phases, np.ones(n+1), 'every actual centred scalar-U1 Haar node positive', 2e-7)
    weights = np.exp(logs-logs.max()); weights /= weights.sum()
    first, second, native_contact = [], [], []
    eye = np.eye(n)
    for angle in angles:
        twist = np.exp(1j*angle)
        source = np.linalg.solve(eye+twist*u, eye)
        s = np.trace(source@(twist*du))
        h = np.trace(source@(twist*ddu)-source@(twist*du)@source@(twist*du))
        c = np.trace(source@(twist*contact))
        close(s.imag, 0., 'real canonical conditional first score', 2e-7)
        close(h.imag, 0., 'real canonical conditional Hessian', 2e-7)
        first.append(s.real); second.append(h.real); native_contact.append(c.real)
    s, h, c = np.asarray(first), np.asarray(second), np.asarray(native_contact)
    score_mean = float(weights@s)
    score_covariance = float(weights@(s*s)-score_mean*score_mean)
    conditional_hessian = float(weights@h)
    canonical_hessian = conditional_hessian+score_covariance
    direct_errors = []
    for index in np.linspace(0, n, 7, dtype=int):
        sign, value = np.linalg.slogdet(eye+np.exp(1j*angles[index])*u)
        close(sign*np.exp(-1j*(background+n/2*angles[index])), 1., 'direct canonical positive determinant', 2e-7)
        direct_errors.append(close(value, logs[index], 'direct full determinant agrees with all-mode factors', 2e-7))
    return dict(full_modes=n, occupied=n//2, Haar_nodes=n+1,
        centred_phase_error=phase_error, direct_factor_error=max(direct_errors),
        canonical_log=float(logs.max()+math.log(float(np.exp(logs-logs.max()).mean()))),
        score_mean=score_mean, conditional_Hessian=conditional_hessian,
        score_covariance=score_covariance, full_canonical_Hessian=canonical_hessian,
        contact=float(weights@c), bubble_without_contact=float(weights@(h-c)),
        whole_canonical_not_replaced=True)


def native_consumer(a: float, size: int = 3, grid: int = 24, seed: int = 2221083) -> dict[str, Any]:
    """Real near-cutoff 3D full canonical current/contact/unit identities."""
    require(size >= 3 and grid >= 4, 'native 3D size>=3, selected continuum comparator grid>=4')
    physical = RotationBand(a=a, m=.2, r=.12, v=.3, improved=False)
    unit = RotationBand(a=1., m=a*.2, r=.12, v=.3, improved=False)
    law, unit_law = BulkQuantumLaw(size, band=physical), BulkQuantumLaw(size, band=unit)
    admission = metric_admission(a)
    lower = np.asarray(admission['Gram_Loewner_lower'])
    rows = []
    selected, frame = None, None
    for centre in itertools.product(range(4), repeat=3):
        result, fields = original_frame(physical, size, np.asarray(centre)*math.pi/2, True)
        gram = np.asarray(result['G'])
        schur = float(gram[0, 0]-gram[0, 1:]@np.linalg.solve(gram[1:, 1:], gram[1:, 0]))
        margin = float(np.linalg.eigvalsh(gram-lower)[0])
        require(margin >= -2e-10 and schur >= admission['Schur_floor']-2e-10,
                'true all64 full-ten Gram satisfies analytic moment admission')
        rows.append(dict(centre=list(centre), Schur=schur, Loewner_margin=margin,
                         full_Gram_gradient=result['gradient_residual']))
        if centre == (1, 2, 3):
            selected, frame = result, fields
    if selected is None or frame is None:
        raise AssertionError('native retained centre123')
    gram = np.asarray(selected['G'])
    fields = np.tile(frame, (law.volume, 1, 1, 1))
    links = law.toron_links(np.asarray([1, 2, 3])*math.pi/2)
    rng = np.random.default_rng(seed)
    harmonic = rng.normal(size=(3, 3))*.71
    phases = rng.normal(size=(3, law.volume, 3))*4.9+harmonic[:, None, :]
    bhat = rng.normal(size=(3, law.volume))*.37
    durations_hat = np.asarray([.31, .43, .37])
    hs = [law.onebody(fields, links, theta) for theta in phases]
    hats = [unit_law.onebody(fields, links, theta) for theta in phases]
    endpoint_error = max(close(a*h, hh, 'actual physical h_a equals hhat/a') for h, hh in zip(hs, hats))
    transverse = np.zeros((law.volume, 3))
    transverse[:, 1] = np.cos(2*math.pi*law.sites[:, 0]/size)/math.sqrt(law.volume/2)
    chi = np.cos(2*math.pi*law.sites[:, 0]/size+.37)/math.sqrt(law.volume/2)
    pure_gauge = chi[:, None]-chi[law.target]
    jp, cp = native_jets(law, hs[0], transverse)
    jh, ch = native_jets(unit_law, hats[0], transverse)
    first_unit = close(a*jp, jh, 'physical link current equals jhat/a')
    contact_unit = close(a*cp, ch, 'physical link contact equals chat/a')
    eps = .002
    # Aphys is the source here, so Peierls phases shift by a eps eta.
    plus = law.onebody(fields, links, phases[0]+a*eps*transverse)
    minus = law.onebody(fields, links, phases[0]-a*eps*transverse)
    current_A = close(a*jp, (plus-minus)/(2*eps), 'physical continuum-field current jA=jhat', 2e-7)
    contact_A = close(a*a*cp, (plus-2*hs[0]+minus)/(eps*eps), 'physical continuum-field contact cAA=a chat', 2e-6)
    wrong_current = rejected(lambda: close(jp, jh, 'physical continuum-field current jA=jhat'), 'forgot theta=a A source chain rule')
    wrong_endpoint = rejected(lambda: close(hs[0], hats[0], 'actual physical h_a equals hhat/a'), 'borrowed a1 energy normalization')
    beta = np.kron(np.eye(law.volume), chiral_basis(frame, gram, 1)[2])
    schur = rows[[row['centre'] for row in rows].index([1, 2, 3])]['Schur']
    mu = .2*math.sqrt(schur)
    coercive = min(float(np.linalg.eigvalsh((beta@h+h@beta)/2)[0]) for h in hs)
    require(coercive >= mu-2e-8, 'actual full near-Dirac beta coercivity')
    kappa = math.sqrt(.3**2*float(np.trace(gram[1:, 1:]))/(2*.2*.12*a*schur))+math.sqrt((gram[0, 0]-schur)/schur)
    kappa_only = rejected(lambda: require(kappa < 1, 'admit actual canonical positive family'), 'relative-kappa-only gate rejects the positive cutoff branch')
    responses = {}
    for name, probe in (('transverse', transverse), ('pure_gauge', pure_gauge)):
        jets = [native_jets(law, h, probe) for h in hs]
        data = ordered_jets(hs, bhat/a, a*durations_hat,
                            [j for j, _ in jets], [c for _, c in jets])
        response = canonical_response(data, 2*float(np.sum(durations_hat[:, None]*bhat)))
        if name == 'pure_gauge':
            close(response['full_canonical_Hessian'], 0., 'full canonical Ward with native contact', 2e-7)
            require(abs(response['bubble_without_contact']) > 1e-4, 'deleting physical contact fails the same canonical Ward boundary')
        responses[name] = response
    unit_jets = [native_jets(unit_law, h, transverse) for h in hats]
    unit_data = ordered_jets(hats, bhat, durations_hat,
                             [j for j, _ in unit_jets], [c for _, c in unit_jets])
    unit_response = canonical_response(unit_data, 2*float(np.sum(durations_hat[:, None]*bhat)))
    response_units = close(responses['transverse']['full_canonical_Hessian'],
                           unit_response['full_canonical_Hessian'], 'exact canonical response invariant under physical time/energy conversion')
    green_phys = ContinuousGreen(hs, bhat/a, a*durations_hat)
    green_hat = ContinuousGreen(hats, bhat, durations_hat)
    green_units = close(green_phys.kernel(.74*a, .17*a), green_hat.kernel(.74, .17), 'actual continuous inverse source kernel has no extra a multiplier')
    values, vectors = np.linalg.eigh(hs[0])
    filled = vectors[:, values < 0]
    require(filled.shape[1] == 2*law.volume, 'entire 2V static sea retained')
    projector = filled@filled.conj().T
    variance = float(np.trace(projector@jh@(np.eye(len(jh))-projector)@jh).real)
    variance_upper = float(sum((.12**2*gram[0, 0]+.3**2*gram[j+1, j+1])*np.sum(transverse[:, j]**2) for j in range(3)))
    require(variance <= variance_upper+2e-10, 'actual static physical-field current variance within native HS bound')
    continuum, residual, richness = selected_gram(a, grid)
    iso = np.asarray(continuum['G'])
    delta = .2*.12*a*iso[0, 0]/(.3**2*iso[1, 1])
    kv2 = 3*.3**2*iso[1, 1]/(2*.2*.12*a*iso[0, 0])
    applicability_error = close(kv2, 3/(2*delta), 'true near-Dirac relative applicability normalization')
    return dict(a=a, size=size, modes=4*law.volume, seed=seed, all64_metrics=rows,
        true_Gram=gram.tolist(), beta_coercivity_minimum=coercive, mu_physical=mu,
        relative_kappa=kappa, kappa_only_rejection=kappa_only,
        endpoint_unit_error=endpoint_error, current_link_unit_error=first_unit,
        contact_link_unit_error=contact_unit, current_A_finite_difference=current_A,
        contact_A_finite_difference=contact_A, wrong_current_rejected=wrong_current,
        wrong_endpoint_rejected=wrong_endpoint, canonical=responses,
        canonical_unit_identity_error=response_units, point_kernel_unit_identity_error=green_units,
        actual_static_JA_variance=variance, static_JA_variance_upper=variance_upper,
        selected_isotropic_comparator=dict(grid=grid, Gram=iso.tolist(), full10_gradient=residual,
            quadratic_rank=richness['quadratic_rank'], delta_W=delta, kappa_spatial_squared=kv2,
            applicability_error=applicability_error, Gordon_carrier_g=2/(1+delta),
            status='actual selected frozen isotropic carrier; no mechanical-spin g measurement or interval continuum certificate'),
        history_hashes=dict(phases=hashlib.sha256(phases.tobytes()).hexdigest(),
                            temporal=hashlib.sha256(bhat.tobytes()).hexdigest()),
        physical_durations=(a*durations_hat).tolist(), actual_harmonic_histories=harmonic.tolist(),
        scope='finite native 3D full canonical identity/current/contact/unit probe; not N0/T exhaustion or Gibbs photon simulation')


def source_hashes() -> dict[str, str]:
    names = ('noncompact_dirac_photon_envelope.py', 'noncompact_dirac_photon_envelope.md',
             'noncompact_wilson_photon_envelope.py', 'noncompact_wilson_photon_envelope.md',
             'noncompact_maxwell_phase.py', 'noncompact_maxwell_phase.md',
             'bulk_cutoff_matching.py', 'bulk_cutoff_matching.md', 'bulk_global_gram.md',
             'soft_budget.py', 'bulk_quantum_selection.py', 'original_charged_principal.py',
             'original_compact_charge_bound.py', 'original_massive_photon_locality.py',
             'noncompact_microscopic_charge_window.md', 'rotation_band.py')
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=float, default=.001)
    parser.add_argument('--speed', type=float, default=.1)
    parser.add_argument('--size', type=int, default=3)
    parser.add_argument('--gram-grid', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = source_hashes()
    family = positive_family(args.a, args.speed)
    receipt: dict[str, Any] = dict(topic='issue222-near-dirac-wilson-energy-physical-family', plan=PLAN,
        template_revision=TEMPLATE, work_item='campaign', requirements=['outcome', 'verification'],
        family=family, physical_packet=packet_bounds(family, .01, .02, 1.),
        native_consumer=native_consumer(args.a, args.size, args.gram_grid),
        source_sha256=before, runtime=dict(python=sys.version, scipy=scipy.__version__,
            numpy=np.__version__, platform=platform.platform()),
        campaign_complete=False, all_checks_pass=True,
        limitations=['No relative-kappa admission is used; cone/anchored theorem already belongs to Sharp supplier.',
            'Fixed positive coefficients are cutoff-dependent, not a fixed-coupling UV theorem or UE=.0001 admission.',
            'Kernel/current finite identities do not simulate the analytic large-N/large-T photon phase.',
            'Full-field result is the legal isometric near-ground/state-dynamics lift, not exact microscopic Gibbs equality.',
            'Carrier Gordon comparator is not joined physical spin/scattering or an exact renormalized speed/coupling.'])
    after = source_hashes()
    receipt['source_bindings_unchanged'] = before == after
    receipt['ending_source_sha256'] = after if before != after else None
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), all_checks_pass=True,
        source_bindings_unchanged=before == after, a=args.a, UE_physical=family['UE_physical'],
        K_physical=family['K_physical'], N0=family['N0'], rho=family['rho'],
        canonical_Ward=receipt['native_consumer']['canonical']['pure_gauge']['full_canonical_Hessian'],
        Gordon_carrier_g=receipt['native_consumer']['selected_isotropic_comparator']['Gordon_carrier_g']), indent=2))


if __name__ == '__main__':
    main()
