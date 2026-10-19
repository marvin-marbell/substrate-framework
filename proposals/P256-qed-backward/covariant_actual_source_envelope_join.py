#!/usr/bin/env python3
"""Join exact Source2a/Root6911 statistical, MF and physical source prices.

No source/preparation/band change. These are actual-law envelopes/reference
identities; an unknown full inserted physical Gram is NEVER filled by a reference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import mpmath as mp
import covariant_physical_statistical_source_envelope as statistical
import covariant_physical_mf_vacuum_winding as meanfield
import covariant_actual_source_band_consumer as consumer
import covariant_prepared_gauss_state as source
from covariant_induced_photon import Parameters,phi_frame

ROOT=Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_source_chart_price(p: Parameters,selected: dict[str,Any]) -> dict[str,Any]:
    """Source2a prescribed pullback band/Wilson recipe, no changed projector."""
    with mp.workdps(100):
        sigma=mp.mpf(10)**-2000;radius=mp.mpf(10)**-2100
        length=mp.mpf(p.spatial_size)
        pref=mp.exp(mp.mpf('.011')/2)
        path=96*length*sigma
        rows=[]
        for a in p.spacings:
            for mass in p.masses:
                am=mp.mpf(str(a))*mp.mpf(str(mass))
                k=1/(2*(mp.expm1(am)+4))
                bmin=mp.exp(am);bmax=bmin+6
                cap=24*bmax+48*max(1,1/bmin**2)
                dt=64*cap*sigma+(32*k*bmax+16*k*max(1,1/bmin**2))*(radius+8*radius**2)
                gap=2*mp.sinh(am)
                projector=2*dt/(gap-2*dt)
                # sqrt6 pays coherent l1 <=sqrt3 and Frobenius rank4/sqrt2.
                bound=pref*(path+mp.sqrt(6)*projector)
                rows.append(dict(a=a,m=mass,am=mp.nstr(am,20),
                    physical_band_separation=mp.nstr(gap,20),
                    onebody_deltaT_upper_log10=float(mp.log10(dt)),
                    reference_band_projector_upper_log10=float(mp.log10(projector)),
                    full_chart_source_operator_difference_upper_log10=float(mp.log10(bound))))
        return dict(original_operator=selected['macro']['source']['exact_operator'],
            exact_Dminus_physical_pullback=True,source_Pel_before_integer_Wilson_path=True,
            continuous_normal_and_colour_support=dict(sigma='1e-2000',r='1e-2100'),
            literal_SU4_colour_path_difference_upper_log10=float(mp.log10(path)),
            all_parameter_tuples=rows,worst_source_operator_difference_log10=max(row['full_chart_source_operator_difference_upper_log10'] for row in rows),
            original_filledRef4_CAR_squared_cap=selected['macro']['source']['filledRef4_CAR_chart_operator_norm_squared_upper'],
            normal_source_scope='The prescribed free Pel(k) is exactly transported by Q; it is not replaced by a new interacting projector. The Davis-Kahan rows are a conservative onebody departure envelope if comparing the same declared canonical band under normal/colour perturbations; no .019 universal gap.',
            coefficient_proof='U/V matrices are left/right unitaries independent of beam k BEFORE Wilson path. Frobenius/Parseval on all sites preserves original whole coherent source; path telescoping costs96 L sigma. The projector departure bound costs sqrt6 exp(.011/2) for any l2-normalized3 beams. No Gram1 asserted.',
            global_outside_source_normcap=8,actual_full_macro_source_Gram=None,macro_admission=False)


def active_reflected_gap_prices(p: Parameters) -> dict[str,Any]:
    """Actual active+Ref CAR conjugation prices, no auxiliary charge bound."""
    with mp.workdps(100):
        V=mp.mpf(p.spatial_size)**3;L=mp.mpf(p.spatial_size)
        beta=mp.mpf(10)**p.beta_power;delta=mp.mpf('1e-50')
        qmin=2*mp.sin(mp.pi/L)
        R=L*mp.mpf('1e-50')
        probability=(128*R/(315*mp.pi))**3
        rows=[]
        for a in p.spacings:
            for mass in p.masses:
                am=mp.mpf(str(a))*mp.mpf(str(mass));bmax=mp.exp(am)+6
                # Uniform onebody relative square-root bound, then literal
                # exterior multiplication. Tilting costs only e^am, NOT the
                # condition number of the full Fock transfer.
                sf=mp.mpf(2)**20*V*bmax**3*mp.exp(am)*delta
                rho=4*mp.exp(am)+3
                coulomb=rho**2*V**2/(2*beta*qmin**2)
                phase=32*mp.exp(am)*V*L*delta
                x=(2*V)**2*mp.exp(-2*am*p.time_slices)
                rows.append(dict(a=a,m=mass,
                    gap_onebody_conjugation_condition=mp.nstr(mp.exp(am),20),
                    active_relative_SF_gap_weighted_step_log10=float(mp.log10(sf)),
                    actual_active_Ref_Coulomb_projective_gap_weighted_log10=float(mp.log10(coulomb)),
                    actual_active_Ref_integer_phase_gap_weighted_log10=float(mp.log10(phase)),
                    canonical_gap_J_Nt_excitation_trace_log10=float(mp.log10(2*x)),
                    weighted_kernel_candidate_step_log10=float(mp.log10(sf+coulomb+phase))))
        return dict(all9_rows=rows,original_three_toron_f_squared_probability_log10=float(mp.log10(probability)),
            exact_positive_internal_gap_operator='In the actual free-band CAR frame, J_N=exp[-am(2j)] at neutral2V; charged2V+/-1 has exp[-am(2j+1)], with exact canonical multiplicities. ONE common neutral vacuum normalization. Radial heat and actual centre/stat blocks remain separate factors.',
            exact_conjugation_identity='J_N^(-1/2) Gamma_N(R) J_N^(1/2)=Gamma_N(j_one^(-1/2) R j_one^(1/2)); cond(j_one^(1/2))=exp(am). Hence ||Gamma_N(Rtilt)-I||<=expm1(N exp(am)||R-I||). This uses only onebody condition, never a global Fock Tmin.',
            SF_price_proof='Uniform Bmin>=exp(am), Bmax<=exp(am)+6, Tmin_one>=1/(16Bmax). Square-root Sylvester plus inverse onebody root bounds R-I by4096 Bmax^3 delta; exterior N<=2V+1 and actual scalar detB log price fit the displayed2^20 V Bmax^3 exp(am) delta.',
            Coulomb_price_proof='Each literal active+Ref rho_x is rank4 dGamma(Px)-2 plus at most one Ref charge. Gap conjugation costs norm<=4exp(am)+3. Sum|Lplus_xy|<=V²/qmin² gives ||Hlongtilt||<=rho_cap² V²/(2beta qmin²). Thus ||exp(-Hlongtilt)-I||<=expm1(price) without moving Pq through matter. Huge auxiliary excited charges are NOT assigned this cap.',
            integer_phase_proof='The same active+Ref integer Epath phase and gauge-gradient CAR word are sums of bounded onebody number operators. Gap tilting multiplies their declared6VLdelta/12VLdelta prices by at mostexp(am); displayed32exp(am)VLdelta pays both and Ref. Integer seams and all Floquet cycles remain.',
            unclosed_full_law_contract='These are reflected active-CAR coefficient bounds. They do not by themselves prove the all-species/normal/statistical full field kernel obeys G J_N domination or identify its inserted source traces; no normalized macro source moment is filled from them.',
            scalar_heat_power_rule='AFTER the actual same-G/J kernel contract, two error-edge loops use Tr J_N^Nt (1+canonical thermal price), ONE p from f² and actual Gaussian trace ratios; never2^(2V) times a weakened heat exponent.',
            macro_admission=False)


def run() -> dict[str,Any]:
    started=time.perf_counter()
    input_path=ROOT/'covariant_prepared_gauss_state_input_kappa400_sigma2000_r2100.json'
    cfg=json.loads(input_path.read_text())
    selected_path=ROOT/consumer.SELECTED
    selected=json.loads(selected_path.read_text())
    p=Parameters(**cfg['photon_parameters']);rp=source.root_transfer.RootParameters(**cfg['root_parameters'])
    gamma,_=phi_frame(p)
    statistics=statistical.statistical_prices()
    nonflat=statistical.nonflat_penalty_prices()
    source_stat=statistical.coherent_source_prices()
    mf=[meanfield.reference_budget(a,m,p.spatial_size,p.time_slices) for a in p.spacings for m in p.masses]
    chart=full_source_chart_price(p,selected)
    photon=consumer.gaussian_packet_heat_bound(selected['macro'],gamma)
    phi=source.root_transfer.charge_weighted_phi_budget(rp,p)
    aux=source.auxiliary_canonical_kernel_budget(p)
    gauss=source.canonical_gaussian_Gauss_budget(p)
    scalar=source.compact_photon_kernel_budget(p)
    active_gap=active_reflected_gap_prices(p)
    binding_names=[
        'covariant_actual_source_range_band_consumer_import_verified_20261009_capsule.json',
        'covariant_actual_real_gauss_car_kernel_final_verified_20261009_capsule.json',
        'covariant_actual_source_reference_band_final_verified_20261009_capsule.json',
        'covariant_physical_mf_vacuum_winding_reusable_import_20261009_capsule.json',
        'covariant_physical_statistical_source_envelope_kappa400_source2a_root6911_uniform_l2_capsule.json',
        'covariant_actual_source_spectral_strip_consumer_guarded_verified_20261009_capsule.json',
        'covariant_actual_source_upper_strip_exclusion_reference_star_verified_20261009_capsule.json',
        'covariant_parent_root_temporal_spatial_primitive_control_20261009.json']
    bindings={name:dict(sha256=digest(ROOT/name),data=json.loads((ROOT/name).read_text())) for name in binding_names}
    names=['covariant_prepared_gauss_state.py','covariant_root_transfer.py','covariant_canonical_gauss_boundary_kernel.py',
        'covariant_actual_source_envelope_join.py','covariant_physical_statistical_source_envelope.py',
        'covariant_physical_mf_vacuum_winding.py','covariant_actual_source_band_consumer.py',
        'covariant_actual_source_range_band_consumer.py','covariant_actual_real_gauss_car_kernel.py',
        'covariant_actual_source_reference_band.py','covariant_actual_source_spectral_strip_consumer.py',
        'covariant_actual_source_upper_strip_exclusion.py']
    hashes={name:digest(ROOT/name) for name in names}
    if hashes['covariant_prepared_gauss_state.py']!=consumer.EXPECTED_SOURCE or hashes['covariant_root_transfer.py']!=consumer.EXPECTED_ROOT:
        raise ValueError('Actual physical source epoch changed')
    return dict(governing_plan_revision=consumer.PLAN_REVISION,source_hashes=hashes,
        selected_original_capsule_sha256=digest(selected_path),original_input_sha256=digest(input_path),
        all9_exact_MF_reference_tuples=mf,actual_standalone_statistical_envelope=statistics,
        actual_nonflat_full_source_normalization=nonflat,actual_wholepacket_statistical_source_operator=source_stat,
        actual_physical_full_chart_source_operator=chart,
        actual_species_resolved_physical_Phi_projective=phi,
        actual_NF_kappa400_canonical_mixed_boundary=aux,
        actual_real_integer_Gauss_and_transformed_source=gauss,
        actual_positive_Coulomb_Gaussian_scalar_block=scalar,
        actual_full125_reference_photon_polynomial_heat_bound=photon,
        actual_source_operator_successor_bindings=bindings,
        active_reflected_gap_weighted_CAR_and_source_prices=active_gap,
        remaining_operator_contracts=[
            'The scalar Coulomb/Gaussian coefficient price is NOT yet a physical all-species field-kernel CAR norm domination by G times genuine MF/normal/statistical reference.',
            'FullPhi projective relative Schatten/power bounds retain the actual same-Q internal operator; no multiplication by unknown normalized source Gram.',
            'Gaussian weighted photon heat bound may be lifted only AFTER the SAME positive G/CAR kernel bound; no weighted BKS or scalar-to-Loewner lift.',
            'Original Pe proof witness needs the ACTUAL joint full-reference threshold strips; far-out/null components may be priced by delta/strip-distance leakage, not their irreducible entire raw norm. The full jumping Wilson-source momentum-fibre shell bound is not the theta-dependent body/toro operator strip.',
            'The same microscopic U_N(t) receiver consumer is a separate owner; no macro U backend invented'],
        actual_macro_M0=None,actual_macro_M1=None,actual_macro_Pe_Gram=None,actual_macro_normalized_photon_Gram=None,
        original_source_and_Pe_unchanged=True,macro_source_admitted=False,elapsed_seconds=time.perf_counter()-started)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run()
    args.output.write_text(json.dumps(source.portable(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),elapsed_seconds=report['elapsed_seconds'],
        worst_source_chart_error_log10=report['actual_physical_full_chart_source_operator']['worst_source_operator_difference_log10'],
        actual_macro_M0=None,macro_source_admitted=False)))


if __name__=='__main__':
    main()
