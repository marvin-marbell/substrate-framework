#!/usr/bin/env python3
"""SOURCE-bound first-form energy of the ORIGINAL long-pulse full-Fock reference.

The energy observable excludes, but the evolution RETAINS, mean Gauss density
squared. Its bounded commutator is paid. No Gaussian evolution, physical energy
shift, photon-number second moment, or actual-state graph transfer is asserted.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy.linalg import eigh
from improved_phi_band import C, UE, NSTAR, selected_band, ImprovedPhiLaw
from moving_physical_noether import PulseClock
from original_total_angular_transfer import primitive_flow, scalar_orbital
from prepared_neutral_slow_light_reference import ceil_log2

ROOT = Path(__file__).resolve().parent
CLOCK = 1008124871316564
ROUT = 2684354560000
PI_UPPER = F('3.14159265358979323846264338327950288419716939937511')
INITIAL_SHA = '64612effb995b3715d7dbc7d49cf0a1b7d3daef415986e730e32198924df47f4'
HARMONIC_SHA = '953c41379268fcd8b40588effc3f233966e6054772245d49e5acdf83aa87f646'
REFERENCE_SHA = '66b475ce7a259418a74d62790c1eef2d9173d37fec1b64fe4b895d5696512b2e'
MEAN_ARRAY_SHA = '63f3b4114689177dd5a224cee4135db0ea198c31e14bc1d535b88fb38fd45a68'
IDENTITY = 'PNFF-RAW64-coherent-Ref4-sea-Ref0-seaPlus4-ONE-globalfilter-v2'
PLAN_REVISION = 'a515ec257394da01e219692c75076125c6903434fb5e6d64edfa547d3156c6d9'


def exact(value: Any) -> F:
    if isinstance(value, dict):
        return F(int(value['exact_numerator_hex'], 16), int(value['exact_denominator_hex'], 16))
    return F(value)


def encoded(value: F | int) -> dict[str, str]:
    value = F(value)
    return dict(exact_numerator_hex=hex(value.numerator), exact_denominator_hex=hex(value.denominator))


def ceil_fraction(value: F) -> int:
    return -(-value.numerator // value.denominator)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def driven_energy_bound(initial_receipt: dict[str, Any],
                        harmonic_receipt: dict[str, Any] | None = None,
                        reference_capsule: dict[str, Any] | None = None,
                        *, clock: int = CLOCK, include_coulomb: bool = True) -> dict[str, Any]:
    """First-form bounds for the actual initial P components, NEVER renormalized.

    The CLI byte-binds all suppliers. This pure API checks their mathematical
    contracts; provenance is a separate byte-binding, not a caller assertion.
    Missing optional inputs are explicit and do not invent a coordinate graph.
    Pi below means canonical p_theta, theta=N A_h; physical p_h=N p_theta.
    """
    require(clock == CLOCK, 'Wrong fast/short clock: ORIGINAL quintic long pulse required')
    require(include_coulomb, 'Gauss longitudinal energy HC cannot be omitted')
    initial = initial_receipt['initial_energy']
    require(initial['identity'] == IDENTITY and initial['initial_only'], 'Actual ONE-filter initial energy required')
    require(initial_receipt.get('natural_exit') is True, 'Initial supplier lacks natural completion')
    n, volume = int(NSTAR), int(NSTAR)**3
    em0 = exact(initial['EM_initial_energy_upper'])
    hc0 = exact(initial['Coulomb_operator_upper'])
    require(em0 == 4*C*volume+2*UE*n*n*(volume+2)+3*UE/n, 'Original initial EM normalization differs')
    require(hc0 == UE*F(n*n*(4*volume+32), 16), 'Original full-Fock Coulomb/Ref density bound differs')
    require(exact(initial['actual_SOURCE_harmonic_energy_upper']) == 64*UE/n, 'Original harmonic kinetic coefficient differs')
    lam = int(initial['selected_log2_lambda_lower'])
    require(lam >= 1, 'Original chi=Lambda^-1/2 must be <=1')
    missing = []
    if reference_capsule is None:
        missing.append('SOURCE-bound A full-Fock operator-family/reference decomposition')
    else:
        reference = reference_capsule['science']['reference']
        require(reference['clock'] == clock and reference['original_Rout'] == ROUT, 'Reference pulse/guard changed')
        require('Qheavy only' in reference['Phi_fast'], 'Fast LR charges were not removed in this reference')
        require(reference_capsule['analytic_integration_premises']['mean_Gauss'].startswith('chi/2'), 'Retained mean Gauss term missing')
    # Literal averaging: sum_(i,j) ||clover_ji||² <= 2 ||curl A||²/a^4;
    # Cauchy over at most three j and contractive endpoint averages yield
    # ||Y||² <= 6 R² ||curl A||²/a². ORIGINAL a=1, |v_j|<=R.
    # sqrt(6)<3 chooses an exact rational majorant, not a continuum curl.
    cv = F(3*ROUT)
    a = cv/C
    omega_peak = 15*PI_UPPER/F(4*clock)
    tv_omega = 2*omega_peak
    kappa = a*omega_peak
    require(kappa < 1, 'Original sine-clover form is not coercive at this clock')
    # Actual mean Phi norm<=2 and encoded entry triangle<2^16 V.
    # Mechanical g0: gGG^-1<=160, JG<=16, mass-Y row<=2^16
    # (1+3R)sqrt(V), ||Q_LR||row<=32sqrt(V); 2^40 pays both
    # mean mechanical+Ref and scalar_orbital. All Peierls factors are unitary.
    m0 = F(2**16*volume)
    m1 = F(2**40*volume*(1+3*ROUT))
    # Keep B=chi/2 Q†gGG^-1 Q. Use chi<=1 ONLY in this uniform bound.
    # [B,Hfield]=0 (local SU4 preserves site number); no [B,Hfast]=0.
    b = F(2**16*volume)
    commutator_work = 2*b*clock*(m0+omega_peak*m1)
    proof_shift = m0+omega_peak*m1+m1/a
    # Fobs=Hfield+hmean-Omega(Gfield+Gmatter). Fobs+C0 >=0;
    # |<G>| <= a/(1-kappa) <Fobs+C0>. Work also includes i[B,Fobs].
    exponent = a*tv_omega/(1-kappa)
    # ln(2)>69/100 follows from 2*(1/3+1/81+1/1215)>69/100.
    require(2*(F(1, 3)+F(1, 81)+F(1, 1215)) > F(69, 100), 'Dyadic exponential majorant failed')
    gronwall_log = ceil_fraction(exponent/F(69, 100))
    shifted_upper = (em0+hc0+m0+proof_shift+commutator_work)*2**gronwall_log
    field_upper = shifted_upper/(1-kappa)
    # All coexact frequencies >=c/N. Zero harmonic is NOT an oscillator.
    first_number = 1+F(n, 1)/C*field_upper
    pi_theta_squared = F(n, 1)/UE*field_upper
    result: dict[str, Any] = dict(
        original_N=n, original_T=clock, original_UE=encoded(UE), original_c=encoded(C),
        original_K=encoded(C*C/(2*UE)), selected_log2_Lambda_lower=lam,
        field_energy_log2_upper=ceil_log2(field_upper), first_number_log2_upper=ceil_log2(first_number),
        harmonic_Pi_squared_log2_upper=ceil_log2(pi_theta_squared),
        field_energy_upper=encoded(field_upper), first_number_upper=encoded(first_number),
        harmonic_Pi_squared_upper=encoded(pi_theta_squared),
        harmonic_Pi_convention='p_theta=-i d/dtheta; theta=N Ah, p_h=N p_theta, Hh=UE/N p_theta²',
        Cv_upper=encoded(cv), Omega_peak_upper=encoded(omega_peak), TV_Omega_upper=encoded(tv_omega),
        kappa_upper=encoded(kappa), kappa_upper_float=float(kappa),
        gronwall_exponent_upper=encoded(exponent), gronwall_log2_upper=gronwall_log,
        M0_upper=encoded(m0), M1_upper=encoded(m1), retained_meanGauss_norm_upper_uniform_Lambda=encoded(b),
        retained_meanGauss_commutator_work_upper=encoded(commutator_work), proof_only_energy_shift=encoded(proof_shift),
        state_rows=['neutral', 'SOURCE_plus', 'SOURCE_minus', 'paired_SOURCE', 'normalized_true_photon_SOURCE'],
        Lambda_independent=True, field_includes=['coexact electric', 'literal noncompact magnetic', 'Gauss Coulomb', 'free harmonic kinetic'],
        energy_closed=reference_capsule is not None,
        fast_slow_identity='[Hfast,Fobs]=0 on flatP tensor ABI: fast Phi/colour/heavy/Root only; mean h/Gmatter on LR and A; B retained separately, [Hfast,B] NOT asserted',
        work_identity='d<Fobs>/dt=-Omega_dot<Gslow>+i<[B,Fobs]>; |<[B,Fobs]>|<=2||B||(M0+|Omega|M1)',
        domain='Symmetric weak forms on original Dirichlet Phi/colour union x all coexact/harmonic, full finite Fock; form-domain regularization then monotone limit',
        quantum_Noether='Gfield=Re sym(Pi_total.Y); longitudinal PiL=-D Lplus rho, lambdaC=-Lplus D†Y anchored at Ref. Expectation bound uses Cauchy, NOT a strong generator norm.',
        no_physical_energy_or_phase_shift=True, no_Gaussian_driven_state=True,
        no_driven_M2_M4_or_unbounded_actual_state_transfer=True,
        provenance=dict(initial_identity=IDENTITY, initial_receipt_sha256=INITIAL_SHA,
                        harmonic_receipt_sha256=HARMONIC_SHA if harmonic_receipt is not None else None,
                        reference_capsule_sha256=REFERENCE_SHA if reference_capsule is not None else None,
                        byte_verification='CLI verifies hashes; pure API contract checks alone are not byte provenance'))
    if harmonic_receipt is None:
        missing.append('Actual weighted ONE-filter initial theta-coordinate graph, not ptheta alone')
    else:
        require(harmonic_receipt.get('natural_exit') is True, 'Harmonic supplier lacks natural completion')
        graphs = harmonic_receipt['legal']['coherent_harmonic_initial_graphs']
        theta0 = exact(graphs['actual_ONE_fn_theta_norm_upper'])
        require(theta0 == 2 and exact(graphs['actual_ONE_fn_ptheta_norm_upper']) == 8, 'Original Gamma/weighted filter graph changed; broad harmonic input not authorized')
        require(harmonic_receipt['legal']['actual_prepared_neutral_identity'] == IDENTITY, 'Harmonic source differs')
        require('weighted fn/fq' in graphs['graph_transfer'], 'Position graph cannot be inferred from kinetic energy')
        # ||dot Ah psi|| <= 2 sqrt(UE E/V)+|Omega| Cv sqrt(2E/(KV)).
        # Minkowski in time; (x+y+z)²<=3(x²+y²+z²), integral|Omega|<8.
        ah_squared = 3*(theta0*theta0/F(n*n)+4*clock*clock*UE*field_upper/volume
                        +128*cv*cv*field_upper/(C*C/(2*UE)*volume))
        result.update(harmonic_A_squared_upper=encoded(ah_squared), harmonic_A_squared_log2_upper=ceil_log2(ah_squared),
                      harmonic_theta_squared_log2_upper=ceil_log2(n*n*ah_squared),
                      harmonic_coordinate_identity='dot Ah=2UE p_h/V-Omega mean(Y); Hh has no coordinate potential',
                      initial_theta_norm_upper=encoded(theta0),
                      initial_coordinate_authority='actual original Gamma sigma_theta=1; weighted theta*fn/fq before graph transfer, true creator Fourier pullthrough; no theta applied to norm-only error')
    result.update(closed=not missing, missing_prerequisites=missing)
    return result


def native_form_checks(mean_arrays: Path) -> dict[str, Any]:
    """New native Gauss/form witnesses; not a rerun of any supplier SCI."""
    band, frame, _ = selected_band()
    law: Any = ImprovedPhiLaw(3, band)
    v = law.volume
    fields = np.tile(frame, (v, 1, 1, 1))
    links = np.tile(np.eye(4, dtype=complex), (v, 3, 1, 1))
    xyz = law.sites.astype(float)-1
    alpha = np.clip(1-np.linalg.norm(xyz, axis=1)/3, 0, 1)
    velocity = alpha[:, None]*np.column_stack((-xyz[:, 1], xyz[:, 0], np.zeros(v)))
    r = float(np.max(abs(velocity)))
    gradient = np.zeros((3*v, v))
    for x in range(v):
        for j in range(3):
            gradient[3*x+j, int(law.target[x, j])] += 1
            gradient[3*x+j, x] -= 1
    lap = gradient.T@gradient
    values, vectors = eigh(lap)
    positive = values > 1e-9
    green = (vectors[:, positive]/values[positive])@vectors[:, positive].T
    harmonic = np.tile(np.eye(3), (v, 1))/math.sqrt(v)
    coeq = np.eye(3*v)-gradient@green@gradient.T-harmonic@harmonic.T
    rng = np.random.default_rng(22271007)
    native_ue, native_c = 2.**-20, .125
    k = native_c*native_c/(2*native_ue)
    rho = np.zeros(v)
    rho[0], rho[13] = 2, -2
    pi_l = -gradient@green@rho
    gauss_error = float(np.linalg.norm(-gradient.T@pi_l-rho))
    require(gauss_error < 1e-10, 'Native actual div PiL=rho sign failed')
    rows = []
    wrong_coulomb_defect = 0.
    wrong_sign_defect = 0.
    missing_harmonic_defect = 0.
    # Reduce flux until the missing-HC linear rho.lambda term dominates the
    # quadratic magnetic form; this is the actual adverse scaling mechanism.
    small_flux = 1e-10
    for scale in (small_flux, .07, 20.):
        phases = rng.standard_normal((v, 3))*scale
        y = primitive_flow(law, fields, links, phases, velocity, alpha)[2].ravel()
        curls = []
        for x in range(v):
            for j in range(3):
                for i in range(j+1, 3):
                    curls.append(phases[x, j]+phases[law.target[x, j], i]
                                 -phases[law.target[x, i], j]-phases[x, i])
        curl_squared = float(np.asarray(curls)@np.asarray(curls))
        require(float(y@y) <= 6*r*r*curl_squared+1e-20, 'Actual endpoint/clover energy inequality failed')
        lam = -green@gradient.T@y
        lam -= lam[13]
        p_t = coeq@rng.standard_normal(3*v) if scale != small_flux else np.zeros(3*v)
        p_h = harmonic@rng.standard_normal(3) if scale != small_flux else np.zeros(3*v)
        total = p_t+p_h+pi_l
        split = float((p_t+p_h)@y+rho@lam)
        direct = float(total@y)
        energy = native_ue*float(total@total)+k*curl_squared/2
        bound = 3*r/native_c*energy
        require(abs(split-direct) < 1e-10*(1+abs(direct)), 'Native Gauss/Noether form reconstruction failed')
        require(abs(direct) <= bound+1e-10, 'Quantum symmetric weak-form Cauchy/Young bound failed')
        wrong_sign_defect = max(wrong_sign_defect, abs(float((p_t+p_h)@y-rho@lam)-direct))
        missing_harmonic_defect = max(missing_harmonic_defect, abs(float(p_t@y+rho@lam)-direct))
        if scale == small_flux:
            no_hc_energy = native_ue*float((p_t+p_h)@(p_t+p_h))+k*curl_squared/2
            wrong_coulomb_defect = abs(direct)-3*r/native_c*no_hc_energy
        # Explicit Gaussian exp(-|z-z0|²/2+i p.z): sym(Pi,Y) has
        # real density p.Y; its electric form also contains |z-z0|².
        # The divergence correction belongs to the strong expression only.
        rows.append(dict(scale=scale, actual_sine_y_norm=float(np.linalg.norm(y)), curl_squared=curl_squared,
                         Noether_form=direct, reconstructed_form=split, energy=energy, form_bound=bound,
                         endpoint_clover_ratio=float(y@y)/(r*r*curl_squared),
                         symmetric_quantum_form_uses_real_Pi_pairing=True))
    require(wrong_coulomb_defect > 1e-14 and wrong_sign_defect > 1e-6 and missing_harmonic_defect > 1e-6,
            f'Adverse controls failed: HC={wrong_coulomb_defect}, sign={wrong_sign_defect}, harmonic={missing_harmonic_defect}')
    # Actual frozen mean cometric, original SU4 generators and actual mean
    # onsite h. Heavy and LR are disjoint factors, B can couple them.
    require(hashlib.sha256(mean_arrays.read_bytes()).hexdigest() == MEAN_ARRAY_SHA, 'Mean Gauss array bytes changed')
    with np.load(mean_arrays) as arrays:
        inverse = arrays['actual_cometric_GG'][:15, :15].copy()
    charges = np.asarray([np.kron(t, np.eye(4))+np.kron(np.eye(4), t) for t in law.generators])
    b = np.einsum('ab,aij,bjk->ik', inverse, charges, charges)/2
    h = law.onsite*frame[0]
    hfast = np.kron(-h, np.eye(4))
    fslow = np.kron(np.eye(4), h)
    fast_slow = float(np.linalg.norm(hfast@fslow-fslow@hfast))
    b_slow = float(np.linalg.norm(b@fslow-fslow@b))
    b_fast = float(np.linalg.norm(b@hfast-hfast@b))
    require(fast_slow < 1e-12, 'Actual heavy/LR tensor ABI commutation failed')
    require(b_slow > 1e-10 and b_fast > 1e-10, 'Retained Gauss commutator control failed to reject fake separation')
    commutator_bound = 2*float(np.linalg.norm(b, 2))*float(np.linalg.norm(fslow, 2))
    require(float(np.linalg.norm(b@fslow-fslow@b, 2)) <= commutator_bound+1e-10, 'Bounded retained-Gauss work bound failed')
    # Uniform harmonic A has exactly zero curl/Noether velocity. Position
    # cannot be bounded from free harmonic kinetic energy: broad Gaussians.
    uniform = np.tile(np.array([.7, -.4, .9]), (v, 1))
    zero_y = primitive_flow(law, fields, links, uniform, velocity, alpha)[2]
    require(float(np.linalg.norm(zero_y)) < 1e-12, 'Harmonic zero mode incorrectly confined')
    broad_variance = 3*10.**12/2
    broad_momentum_squared = 3/(2*10.**12)
    require(broad_variance > 4 and broad_momentum_squared < 64, 'Broad-position adverse state not discriminating')
    pulse = PulseClock(float(CLOCK), 2*math.pi)
    speed_check = abs(pulse.speed(CLOCK/2)-float(15*PI_UPPER/F(4*CLOCK)))
    require(speed_check < 1e-28, 'Actual PulseClock quintic normalization differs')
    return dict(native_N=3, native_UE=native_ue, native_c=native_c, native_K=k, legal_constants_not_substituted=True,
                actual_Gauss_sign_error=gauss_error, rows=rows,
                wrong_lambda_sign_defect=wrong_sign_defect, omitted_Coulomb_bound_violation=wrong_coulomb_defect,
                omitted_harmonic_Noether_defect=missing_harmonic_defect,
                actual_meanGauss=dict(mean_arrays_sha256=MEAN_ARRAY_SHA, fast_slow_commutator=fast_slow,
                                     retained_B_slow_commutator=b_slow, retained_B_fast_commutator=b_fast,
                                     bounded_slow_commutator_upper=commutator_bound,
                                     native_scope='Actual original local one-heavy/one-light SU4 charge restriction; no full-Fock state projection used in LEGAL proof'),
                uniform_harmonic_Noether_norm=float(np.linalg.norm(zero_y)),
                broad_position_wrong_input=dict(theta_squared=broad_variance, ptheta_squared=broad_momentum_squared,
                                                rejected_as_original_Gamma=True, kinetic_energy_does_not_bound_position=True),
                actual_quintic_peak_error=speed_check,
                law_mean_Phi_operator_norm=float(max(np.linalg.norm(f, 2) for f in frame)),
                actual_scalar_orbital_norm=float(np.linalg.norm(scalar_orbital(law, links, np.zeros((v, 3)), velocity), 2)))


def verified_json(path: Path, expected: str) -> dict[str, Any]:
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == expected, f'Frozen supplier changed: {path}')
    return json.loads(body)


def source_epoch() -> dict[str, str]:
    result = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.parent == ROOT and path.suffix == '.py':
                result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-receipt', type=Path, required=True)
    parser.add_argument('--harmonic-receipt', type=Path, required=True)
    parser.add_argument('--reference-capsule', type=Path, required=True)
    parser.add_argument('--mean-arrays', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    initial = verified_json(args.initial_receipt, INITIAL_SHA)
    harmonic = verified_json(args.harmonic_receipt, HARMONIC_SHA)
    reference = verified_json(args.reference_capsule, REFERENCE_SHA)
    before = source_epoch()
    bound = driven_energy_bound(initial, harmonic, reference)
    native = native_form_checks(args.mean_arrays)
    adverse = {}
    adverse_cases: list[tuple[str, dict[str, Any]]] = [
        ('wrong_fast_clock', dict(clock=6)),
        ('omitted_Coulomb_Gauss', dict(include_coulomb=False)),
    ]
    for name, kw in adverse_cases:
        try:
            driven_energy_bound(initial, harmonic, reference, **kw)
        except ValueError as error:
            adverse[name] = dict(rejected=True, reason=str(error))
        else:
            raise ValueError(f'Unsafe case accepted: {name}')
    require(before == source_epoch(), 'Loaded original source changed during execution')
    result = dict(schema='issue222-driven-slow-first-energy-v1', plan_revision=PLAN_REVISION,
                  bound=bound, native=native, adverse=adverse,
                  input_sha256={str(args.initial_receipt): INITIAL_SHA, str(args.harmonic_receipt): HARMONIC_SHA,
                                str(args.reference_capsule): REFERENCE_SHA, str(args.mean_arrays): MEAN_ARRAY_SHA},
                  loaded_sources=before, loaded_sources_stable=True,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  program_wall_seconds=time.monotonic()-started,
                  scope='Actual smaller fullFock reference under ORIGINAL long pulse, initial actual preparation only; not physical admission, ordinary T6/Born, or driven M2/M4')
    text = json.dumps(result, sort_keys=True)
    args.output.write_text(text)
    os.chmod(args.output, 0o600)
    print(text)


if __name__ == '__main__':
    main()
