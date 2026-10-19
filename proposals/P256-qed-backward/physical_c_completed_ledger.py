#!/usr/bin/env python3
"""NEW same-source C-completed physical ledger; frozen Alpha56 is unchanged.

The primitive B/F triangular comparison is made BEFORE matrix log. Operator
monotonicity of log controls hC; exponential conditioning is paid explicitly.
No active8k body expansion, occupation deletion or Gaussian-photon promotion.
Every occupation n has its own Gamma(T) order bound; charged and neutral use
ONE neutral lambda_N. Native currents/contact/source/receiver are actual Alpha56
consumers. The macro density ledger is regional AND requires the same kernel
congruences; it is not admission of the full reflected SF/Haar physical kernel.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import mpmath as mp
import numpy as np
from scipy.linalg import eigh, expm

ROOT = Path(__file__).resolve().parent
FROZEN_HASH = '56e1c611bffe17964644b47b6aca7ed496a195c8f4795e76e657ee2c01c3ff04'
CAPSULE_NAME = 'physical_alpha_compton_Cartesian_final_native_capsule.json'
SOURCE_BYTES = Path(__file__).read_bytes()
FROZEN_CAPSULE_BYTES = (ROOT/CAPSULE_NAME).read_bytes()
_snapshots = json.loads(FROZEN_CAPSULE_BYTES)['capsule']['source_snapshot_at_import']
if hashlib.sha256(_snapshots['physical_alpha_compton.py'].encode()).hexdigest() != FROZEN_HASH:
    raise RuntimeError('The selected physical law must be the actual frozen56 source')
_directory = tempfile.TemporaryDirectory(prefix='physical-C-ledger-frozen56-')
_snapshot = Path(_directory.name)
for _name, _content in _snapshots.items():
    (_snapshot/_name).write_text(_content)
sys.path.insert(0, str(_snapshot))
_spec = importlib.util.spec_from_file_location('_physical_C_frozen56', _snapshot/'physical_alpha_compton.py')
if _spec is None or _spec.loader is None:
    raise RuntimeError('The captured physical source cannot be imported')
alpha = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = alpha
_spec.loader.exec_module(alpha)
sys.path.remove(str(_snapshot))


def primitive_relative_bound(kappa: Any, radius: Any) -> dict[str, Any]:
    """Uniform all-U1/SU4, all-Cartesian Phi comparison against SAME Y=0.

    Tbase=Ldag diag(Bp^-1,Bm)L. After L0 congruence, only deltaF occurs:
    D0^1/2 Lphi L0^-1 D0^-1/2 has upper block Bp0^-1/2 deltaF Bm0^-1/2.
    Its norm AND inverse norm are <=1+df/b0. B order/inverse order gives
    exp(+/-eta)Tbase0 bounds, eta=-log(1-db/b0)+2log(1+df/b0).
    log is operator-monotone. Averaging h and its reflected inverse gives
    ||delta hC||<=eta. Exponential is NOT operator-monotone. Along
    Hs=H0+s deltaH, its relative derivative is integral_-1/2^1/2
    exp(u Hs)deltaH exp(-u Hs)du. Its norm is <=eta*2expm1(d/2)/d,
    d=log(condition)+2eta. Differential Loewner/Gronwall comparison
    then gives exp(+/-etaC)Tc0 without requiring absolute epsilon<1.
    """
    k, r = mp.mpf(kappa), mp.mpf(radius)
    b0 = 1/(2*k)-3
    db, df = k*(r+r*r), 4*k*r
    if b0 <= 0 or db >= b0:
        raise ValueError('This actual B comparison needs positive reference B and db<b0')
    eta_b = -mp.log1p(-db/b0)
    eta = eta_b+2*mp.log1p(df/b0)
    bmax, fmax = 1/(2*k)+3, mp.mpf(3)
    tmin = min(b0, 1/bmax)/(1+fmax)**2
    tmax = max(1/b0, bmax)*(1+fmax)**2
    condition = tmax/tmin
    epsilon = condition*eta*mp.exp(eta)
    spread = mp.log(condition)+2*eta
    eta_c = 2*eta*mp.expm1(spread/2)/spread
    # pref=mean(logA)-Tr(hC)/2. Each logB variation<=eta_b.
    pref_per_site = 2*eta_b+2*eta
    full_fock_per_site = pref_per_site+4*eta_c
    # Relative SF square roots, BEFORE any noncommuting normal heat.
    sf_epsilon = eta*mp.sqrt(condition)*mp.exp(eta/2)/2
    sf_all_log = pref_per_site/2+4*mp.log1p(sf_epsilon)
    return dict(B_reference_lower=b0, delta_B_upper=db, delta_F_upper=df,
                B_relative_log=eta_b, base_and_C_log_variation=eta,
                C_transfer_condition_ratio_upper=condition,
                absolute_spectral_epsilon_DIAGNOSTIC_not_used=epsilon, C_transfer_relative_log=eta_c,
                scalar_prefactor_log_per_site=pref_per_site,
                ALL_occupation_log_per_site=full_fock_per_site,
                raw_SF_onebody_relative_epsilon=sf_epsilon,
                raw_SF_ALL_occupation_relative_log_per_site=sf_all_log)


def macro_ledger(radius_power: int = 1490, normal_width_power: int = 1500) -> dict[str, Any]:
    """Actually price all flavours/occupations and common halfsteps, no NF arrays."""
    with mp.workdps(180):
        r, sigma = mp.power(10, -radius_power), mp.power(10, -normal_width_power)
        kbody, kaux = mp.mpf('.1256305951044483'), mp.mpf('1e-250')
        aux_count = 10**1150//16
        volume, nt = 2**72, 2**20
        body, auxiliary = primitive_relative_bound(kbody, r), primitive_relative_bound(kaux, r)
        delta = volume*(body['ALL_occupation_log_per_site']
                       +mp.mpf(aux_count)*auxiliary['ALL_occupation_log_per_site'])
        half_delta = delta/2
        sf_delta = volume*(body['raw_SF_ALL_occupation_relative_log_per_site']
                          +mp.mpf(aux_count)*auxiliary['raw_SF_ALL_occupation_relative_log_per_site'])
        epsilon = mp.expm1(delta)
        # p=Nt+2, m=p/2. Holder telescope of ACTUAL common congruenced K.
        p, m = nt+2, (nt+2)//2
        purifier = m*epsilon*mp.exp((m-1)*mp.log1p(epsilon))
        density_trace_upper = 4*purifier/(1-purifier)
        # Union bound for45V Gaussian coordinates, ||Ysite||<=r.
        # This is only the explicit reference normal oscillator, not coupled ground.
        gaussian_tail_log = mp.log(45*volume)-r*r/(45*sigma*sigma)
        encode = lambda value: mp.nstr(value, 80)
        return dict(law='ACTUAL frozen56 log-C/Aeven potential',
            kbody=encode(kbody), active_8k=encode(8*kbody), active_8k_series_available=False,
            auxiliary_kappa=encode(kaux), exact_Naux=str(aux_count), spatial_volume=str(volume),
            temporal_size=nt, analysis_radius_power=radius_power, normal_width_power=normal_width_power,
            radius_is_an_analysis_region_not_a_source_or_filter_gate=True,
            active={key: encode(value) for key, value in body.items()},
            auxiliary={key: encode(value) for key, value in auxiliary.items()},
            ALL_Fock_ANALYSIS_REGION_relative_log=encode(delta),
            ANALYSIS_REGION_positive_halfstep_relative_log=encode(half_delta),
            power_p=p, purifier_m=m, ANALYSIS_REGION_Holder_purifier_upper=encode(purifier),
            ANALYSIS_REGION_normalized_density_trace_distance_upper=encode(density_trace_upper),
            raw_SF_ANALYSIS_REGION_relative_log=encode(sf_delta),
            raw_SF_statement='D=exp((h0+dh)/2)exp(-h0/2), ||D-I||<=eta sqrt(condition) exp(eta/2)/2 by Duhamel. Actual halfFock factors satisfy SFphi=exp(dpref/2)Gamma(D)SF0. All occupations yield ||exp(dpref/2)Gamma(D)-I||<=expm1(sf_delta). Right Haar/charge-middle projection survives this relative FACTOR comparison, unlike kernel Loewner order. Left normal heat does not commute and is NOT silently admitted.',
            reference_normal_outside_region_probability_log_upper=encode(gaussian_tail_log),
            exact_occupation_statement='For n particles in4V modes: exp(+/-[2V eta_b+2V eta+n eta_C]) times SAME reference Gamma kernel. Every n=0..4V retained. Aux charge1 and Naux unchanged; active Ref4->3 uses n->n+1, not Q*n.',
            exact_halfstep_statement='Pointwise ALL-Fock Loewner order survives the SAME kinetic/temporal/gauge/Gauss/centre congruences and sqrt operator monotonicity. This is an actual relative theorem on those same kernels, not exponential operator-monotonicity. Norm(lambda_N) ratios are exp(+/-delta), so one common neutral lambda_N can be used.',
            exact_density_statement='The same reference kinetic/Haar compression must define BOTH kernels. Holder telescope uses ||K-K0||_p<=(exp(delta)-1)||K0||_p and all positive spectra, p=Nt+2. No normalized-source filter Gram is inserted or assumed.',
            P6_kernel_hypothesis='Requires actual K=R Tphi R with the SAME reference kinetic/Gauss/centre congruence. The supplied full-U1 reflected SF factor is NOT automatically of this form. This pointwise theorem is not yet a proved relative comparison of that SF/Haar kernel.',
            occupation_limit='This is an ALL-occupation operator bound, not a borrowed quasiparticle occupation suppression or an actual microscopic auxiliary determinant/thermal simulation.',
            normal_limit='Normal reference45/site gap1 oscillator union tail is priced. Promoting it to interacting chart probability requires the same global inside/outside trace join; outside is NOT silently discarded.',
            outside_limit='Eaux*n C-completes to retained2Eaux/site/species. FullU1 outside heat/rank projection and physical global compactness remain separate; no Root6911 receipt reused.')


def auxiliary_outside_and_domain_ledger() -> dict[str, Any]:
    """Fresh C-completed occupation and polynomial-domain audit, not Root reuse."""
    with mp.workdps(180):
        k, naux = mp.mpf('1e-250'), mp.mpf(10)**1150/16
        p = 2**20+2
        energy = mp.log(1/(2*k)-3)
        auxiliary_log_trace = mp.log(16)-2*p*energy
        # Explicit prospective outside coefficients, not a thermal admission.
        mass, wilson, current = mp.mpf('.7'), mp.mpf('.2'), mp.mpf('.3')
        fock_coupler = 2*(mass+3*(wilson+current))*(naux+1)
        quartic = mp.power(10, 4000)
        young = 3*fock_coupler**(mp.mpf(4)/3)/(mp.power(4,mp.mpf(4)/3)*mp.root(quartic/2,3))
        lower = 2*energy*naux-young+30
        polynomial = []
        for label, value in (('active',mp.mpf('.1256305951044483')),('auxiliary',k)):
            bmin = 1/(2*value)-3-value/4
            bstar = 1/(2*value)+3+2*value
            dstar = max(1/bmin,bstar)
            fstar = 4*(1+value)
            constant = bstar**2*dstar**2*fstar**4
            polynomial.append(dict(flavour=label,
                Fock_multiplier_norm_constant_per_site=mp.nstr(constant,80),
                Fock_multiplier_polynomial_degree_per_site=12))
        return dict(actual_auxiliary_E_TRANSFER=mp.nstr(energy,80),
            exact_ALL16_occupation_energies=['2Eaux']*16,
            exact_ALL16_auxiliary_outside_log_trace=mp.nstr(auxiliary_log_trace,80),
            derivation='C n Cdag=4-n, hence (E n+C E n Cdag)/2=2E I. Occupation0 is NOT zero-energy. No old E*n suppression or deleted occupation row is reused.',
            declared_outside_couplings=[str(mass),str(wilson),str(current)],
            declared_sigma_power=1500,declared_quartic_power=4000,
            fresh_ALL_Fock_Young_loss_per_site=mp.nstr(young,80),
            fresh_outside_form_lower_per_site=mp.nstr(lower,80),
            polynomial_normal_domain=polynomial,
            exact_domain_statement='At fixed compact gaugeA, Bglobal>0. bmax<=bstar(1+r)^2, 1+F<=fstar(1+r), and M=max(bmax,1/bglobal)*(1+F)^2 bounds Tbase and its inverse. Hence ||hC||<=log M and actual normal-ordered ALL-Fock multiplier norm<=constant^V*(1+r)^(12V) per flavour. Finite huge NF only increases a finite polynomial degree; no NF matrices are allocated.',
            normal_heat_hypothesis='For a declared Euclidean normal Mehler heat, sandwiching this finite polynomial is compact and Schwartz is a weak core at FIXED A. This is not certification of an unprovided global quotient/cometric/colour heat.',
            source_weak_core_hypothesis='In a declared reference frame source is bounded CAR creation times smooth compact Wilson path, independent of normalY, and preserves Schwartz normal-field core. Physical Dminus must remove pure-gauge Gamma(V) derivative. Actual temporal/SF/Haar source intertwining remains unproved.',
            P2_target_audit='Fixed-A normal compactness is NOT full U1 compactness: multiplication on compact-A direct integral is not compact merely because A is compact. Actual smoothing/reflected positive kernel is required. Positive compact finite-volume transfer may accumulate at0; this establishes no physical isolated mass pole or QED scattering.',
            full_U1_compactness_and_source_domain_admitted=False,
            outside_receipt_reuse=False)


def actual_relative_probes() -> dict[str, Any]:
    """Noncommuting full-Phi actual transfer and complete Fock Loewner probes."""
    basis = alpha.matrix_unit_basis()/2
    fields = np.einsum('smi,iab->smab', np.sin(np.arange(120).reshape(2, 4, 15)*.31), basis)
    fields *= .001/max(np.linalg.norm(point) for point in fields)
    links = np.array([[expm(1j*.17*(basis[(x+2*j)%15]+.3*basis[(9+x+j)%15]))
                      for j in range(3)] for x in range(2)])
    phases = np.array([[.23, -.17, .31], [-.13, .29, -.19]])
    rows = []
    for k in (.1256305951044483, .02, .005):
        mass = math.log(1/(2*k)-3)/.01
        law = alpha.CartesianCWilson(alpha.gamma_frame(), alpha.Parameters(mass=mass),
                                     np.array([[1, 0, 0], [0, 1, 1]]))
        t0, s0 = law.transfer(phases, np.zeros_like(fields), links)
        t1, s1 = law.transfer(phases, fields, links)
        with mp.workdps(100):
            bound = primitive_relative_bound(k, '.001')
        v, f = eigh(t0)
        inv = (f/np.sqrt(v))@f.conj().T
        relative_matrix = inv@t1@inv
        relative_matrix = (relative_matrix+relative_matrix.conj().T)/2
        rv, rf = eigh(relative_matrix)
        relative_log = (rf*np.log(rv))@rf.conj().T
        relative = rv
        actual = max(abs(np.log(relative)))
        assert actual <= float(bound['C_transfer_relative_log'])*(1+1e-7)
        # All n. Gamma preserves products and square roots, so the conditioned
        # Fock comparison is an exterior power of the ACTUAL relative onebody
        # matrix, not an ill-conditioned subtraction of two huge Fock kernels.
        occupation = []
        for number in range(9):
            generator, _ = alpha.second_quantized(law.change.conj().T@relative_log@law.change, number)
            fock_relative = expm(generator+(s1-s0)*np.eye(len(generator)))
            measured = eigh(fock_relative, eigvals_only=True)
            exponent = 4*float(bound['B_relative_log'])+4*float(bound['base_and_C_log_variation'])+number*float(bound['C_transfer_relative_log'])
            defect = max(abs(np.log(measured)))
            assert defect <= exponent*(1+1e-5)+3e-7
            occupation.append(dict(number=number,dimension=len(generator),actual_relative_log=defect,
                                   derived_relative_log_upper=exponent))
        sf_discriminator = reflected_middle_discriminator(law, phases, fields, links, bound)
        rows.append(dict(kappa=k, actual_onebody_relative_log=float(actual),
                         derived_onebody_relative_log_upper=float(bound['C_transfer_relative_log']),
                         scalar_prefactor_difference=s1-s0, ALL_occupations=occupation,
                         actual_reflected_middle_discriminator=sf_discriminator))
    outside = law.outside_potential(phases, fields, links, (.7, .2, .3))
    outside_rows = []
    radius = max(np.linalg.norm(point) for point in fields)
    for number in range(9):
        generator, _ = alpha.second_quantized(law.change.conj().T@outside['H']@law.change, number)
        spectrum = eigh(generator+outside['offset']*np.eye(len(generator)), eigvals_only=True)
        observed = float(max(abs(spectrum)))
        upper = 4.4*law.sites*radius
        assert observed <= upper*(1+1e-12)
        outside_rows.append(dict(number=number, dimension=len(generator),
                                 actual_linear_Fock_norm=observed, derived_norm_upper=upper))
    packet = law.source_packet_jet(phases, links, (np.zeros_like(phases), np.zeros_like(phases)))
    spatial = law.change.conj().T@packet['psi']
    _, masks4 = alpha.second_quantized(np.eye(8), 4)
    _, masks5 = alpha.second_quantized(np.eye(8), 5)
    lookup5 = {mask:index for index,mask in enumerate(masks5)}
    source = np.zeros((4,len(masks5),len(masks4)),complex)
    for column,mask in enumerate(masks4):
        for bit in range(8):
            if not mask & (1 << bit):
                source[:,lookup5[mask | (1 << bit)],column] += (
                    (-1)**((mask & ((1 << bit)-1)).bit_count())*spatial[bit])
    gram = sum(matrix.conj().T@matrix for matrix in source)
    source_bounds = eigh(gram,eigvals_only=True)
    assert max(source_bounds) <= 2*(1+1e-12)
    assert np.linalg.norm(spatial.conj().T@spatial-.5*np.eye(4)) < 1e-12
    return dict(rows=rows, field_radius=.001, actual_outside_ALL_occupation_linear_bounds=outside_rows,
                actual_unfiltered_Ref4_to_Ref3_source_Gram_extrema=[float(min(source_bounds)),float(max(source_bounds))],
                source_bound_derivation='SdagS=Tr(PsiPsi†)-dGamma(PsiPsi†); Psi†Psi=I4/2, hence 0<=SdagS<=2I for literal Ref4 source, independent of normalY and integerQ Wilson phases. This bounds the unfiltered source operator, not its Pe4 charged-half Gram.',
                actual_noncommuting_colour_and_all60_per_site_fields=True)


def reflected_middle_discriminator(law: Any, phases: np.ndarray, fields: np.ndarray,
                                   links: np.ndarray, bound: dict[str, Any]) -> dict[str, Any]:
    """Actual C-law SF/zero-electric charge-middle: expose range rotation."""
    h0 = law.potential(phases,np.zeros_like(fields),links)
    h1 = law.potential(phases,fields,links)
    rows = []
    for number in (4,5):
        g0,masks = alpha.second_quantized(h0['h'],number)
        g1,_ = alpha.second_quantized(h1['h'],number)
        # Physical site-major states. Ref4 neutral has2/site; Ref3 charged
        # has ONE -Q reference hole at site1, so E=0 requires2 at site0,3 at1.
        middle = [i for i,mask in enumerate(masks) if (mask & 15).bit_count()==2]
        sf0 = expm((g0+h0['pref']*np.eye(len(g0)))/2)[:,middle]
        sf1 = expm((g1+h1['pref']*np.eye(len(g1)))/2)[:,middle]
        q,_ = np.linalg.qr(sf0,mode='reduced')
        leaked = sf1-q@(q.conj().T@sf1)
        leakage = float(np.linalg.norm(leaked,'fro')**2)
        relative = float(np.linalg.norm(sf1-sf0,'fro')/np.linalg.norm(sf0,'fro'))
        exponent = law.sites*float(bound['scalar_prefactor_log_per_site'])/2
        exponent += number*math.log1p(float(bound['raw_SF_onebody_relative_epsilon']))
        factor_upper = math.expm1(exponent)
        assert relative <= factor_upper*(1+1e-10)+1e-12
        # Schatten4 factor -> Schatten2 positive-kernel comparison.
        s0,s1 = np.linalg.svd(sf0,compute_uv=False),np.linalg.svd(sf1-sf0,compute_uv=False)
        factor_ratio4 = float(np.linalg.norm(s1,4)/np.linalg.norm(s0,4))
        k0,k1 = sf0@sf0.conj().T,sf1@sf1.conj().T
        kernel_ratio2 = float(np.linalg.norm(k1-k0,'fro')/np.linalg.norm(k0,'fro'))
        kernel_upper2 = 2*factor_ratio4+factor_ratio4**2
        assert kernel_ratio2 <= kernel_upper2*(1+1e-10)+1e-12
        rows.append(dict(number=number,physical_carrier_dimension=len(masks),middle_dimension=len(middle),
            actual_changed_factor_weight_in_reference_kernel_null=leakage,
            actual_SF_relative_Schatten2=relative,derived_raw_SF_upper=factor_upper,
            actual_SF_relative_Schatten4=factor_ratio4,actual_kernel_relative_Schatten2=kernel_ratio2,
            factor_telescope_kernel_Schatten2_upper=kernel_upper2))
    return dict(rows=rows,
        exact_structural_falsifier='If changed-factor weight in reference kernel null is positive, no finite exp(delta) K0 Loewner upper bound can hold. Tphi order does NOT commute through SF-half*charge-middle*SF-half.',
        material_repair='Use actual relative SF factor, not kernel Loewner order. ||K1-K0||p/||K0||p <=2r+r², r=||F1-F0||2p/||F0||2p, valid including nulls and all singularvalues.',
        scope='Actual Alpha56 C-completed SF and physical zero-electric integer-charge middle discriminator. Normal field heat/full temporal Haar and macro NF are NOT simulated or admitted here; no gate is installed in the source.')


def actual_source_output_moment() -> dict[str, Any]:
    """Complete native same-source charged halfstep, moment and literal Pe4."""
    # This is the frozen actual consumer, with its complete positive eigenstates.
    p = alpha.Parameters(orders=(2,))
    reference = json.loads(FROZEN_CAPSULE_BYTES)
    gram = json.loads(reference['capsule']['input_json'])['NEW_law_inputs']['Phi_reference_gram']
    phi = np.sqrt(gram)[:, None, None]*alpha.SELECTED
    critical = dict(declared_critical_bare_mass_parameter=-2.0282053185612114)
    native = alpha.cartesian_C_native_source(p, alpha.charge_contract(p), critical, phi,
                                            photon_orders=(2,))
    row = native['records'][0]
    # Exact source moment is relative to ONE neutral eigenvalue in that consumer.
    moment_transfer = p.spacing*row['actual_source_energy_mean_physical']
    half = row['actual_charged_source_halfstep_Gram']
    if row['actual_coupled_charged_neutral_gap_physical'] < 0:
        raise ArithmeticError('The positive-output-energy Markov hypothesis failed')
    lower = half*max(0., 1-moment_transfer/4)
    raw = row['actual_primitive_Ref4_scalar_source_raw_Gram']
    jensen_moment = -math.log(half/raw)
    jensen_lower = half*max(0.,1-jensen_moment/4)
    assert moment_transfer <= jensen_moment*(1+1e-10)
    return dict(native=native, same_source_mean_output_energy_TRANSFER=moment_transfer,
        same_source_mean_output_energy_PHYSICAL=row['actual_source_energy_mean_physical'],
        literal_TRANSFER_cut=4., equivalent_PHYSICAL_cut=4/p.spacing,
        positive_energy_Markov_Pe4_halfstep_Gram_lower=lower,
        same_source_Jensen_first_output_moment_upper_TRANSFER=jensen_moment,
        same_source_Jensen_Pe4_halfstep_Gram_lower=jensen_lower,
        universal_positive_Pe4_sufficient_ratio=math.exp(-4),
        Jensen_proof='For the actual unfiltered SAME source spectral probability, x=K_C/lambda_N in[0,1], f(x)=-x log x is concave with f(0)=0. q=g_half/g_raw. Jensen gives unnormalized first output moment<=-g_half log q and conditional mean<=-log q; literal Pe4 halfGram>=g_half*(1+log q/4). Thus q>exp(-4) suffices, without source gate, quartergate, occupation deletion, or isolated pole.',
        P7_full_halfstep_log_energy_domain_theorem='For any ACTUAL bounded positive K_C, lambda_N>0, a=||K_C||/lambda_N finite, x|log x| and x(log x)^2 extend continuously at0. Their maxima on[0,a] are bounded respectively by max(1/e,a|log a|) and max(4/e²,a(log a)^2). Thus the ACTUAL full charged halfstep sqrt(K_C/lambda_N) S rho^(1/2) lies in the H_C operator domain on positive spectral support, and its first absolute TRANSFER output moment<=max(1/e,a|log a|)*Tr(rho SdagS)<=2 max(1/e,a|log a|). True Knull is killed by the ACTUAL halfstep, not hidden quarter-gating. This is a log-transfer spectral-domain theorem, NOT a physical Dminus/cometric current-domain certification or unfiltered-source domain assertion.',
        P7_remaining_hypothesis='The theorem still requires the declared full-U1/allPhi kernel to actually be a bounded positive operator. The missing macro kernel/cometric/centre construction is not supplied by this spectral theorem; q>exp(-4) must be established on the SAME global source to license positive Pe4 Gram.',
        proof='ALL charged eigenvalues lie below SAME neutral lambda_N in this actual native run. E=-log(K_C/lambda_N)>=0. For the actual source after actual complete charged sqrt(K_C/lambda_N), <P(E<=4)> >=1-<E>/4. No eigenstate/Ref/ph subtraction or changed source is used.',
        macro_limit='The actual finite moment/Markov bound is NOT a macro moment bound. Full physical-family same-source first output moment and receiver CP kernel remain to be derived from the global transfer; physical clock MEANS alone cannot supply them.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    started = time.time()
    result = dict(regional_relative_ledger=macro_ledger(), actual_relative_probes=actual_relative_probes(),
                  fresh_auxiliary_outside_domain_ledger=auxiliary_outside_and_domain_ledger())
    if args.native:
        result['actual_source_output_moment'] = actual_source_output_moment()
    inputs = dict(frozen_source=FROZEN_HASH, frozen_capsule=hashlib.sha256(FROZEN_CAPSULE_BYTES).hexdigest(),
                  radius_power=1490, normal_width_power=1500, native=args.native,
                  literal_transfer_cut=4., source='unchanged Ref4->Ref3 four-spin scalar Wilson path')
    encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
    result['execution'] = dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),
        source_bytes_at_IMPORT=SOURCE_BYTES.decode(), frozen_source_at_IMPORT=alpha.SOURCE_HASHES,
        frozen_source_bytes_at_IMPORT={key:value.decode() for key,value in alpha.SOURCE_BYTES.items()},
        input_json=encoded.decode(), input_sha256=hashlib.sha256(encoded).hexdigest(),
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__,
                     mpmath=mp.__version__, platform=platform.platform()),
        runtime_seconds=time.time()-started,
        own_startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES,
        frozen_startup_unchanged=all((_snapshot/key).read_bytes()==value for key,value in alpha.SOURCE_BYTES.items()))
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=args.output, source=result['execution']['source_at_IMPORT'],
        input_sha256=result['execution']['input_sha256'],
        regional_Fock_relative_log=result['regional_relative_ledger']['ALL_Fock_ANALYSIS_REGION_relative_log'],
        regional_density_trace_upper=result['regional_relative_ledger']['ANALYSIS_REGION_normalized_density_trace_distance_upper'],
        output_moment=(result['actual_source_output_moment']['same_source_mean_output_energy_TRANSFER']
                       if args.native else None),runtime_seconds=result['execution']['runtime_seconds'])))


if __name__ == '__main__':
    main()
