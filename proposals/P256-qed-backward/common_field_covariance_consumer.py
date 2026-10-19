#!/usr/bin/env python3
"""Execute an original-law breathing reference on the unchanged frozen packet.

This is a common-field reference consumer, not a native spin/ground acceptance
gate.  H-Href and G-Gref remain explicit operator ledgers.  A second invocation
consumes the produced canonical ABI without restoring or preparing any sea.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy
from scipy.optimize import brentq

from common_field_quadratic_reference import (
    _affine_centres, canonical_flow, continuous_reference_phase,
    current_six_projection, cutoff_graph_bounds, fixed_seed_potential,
    gaussian_polynomial_abi, mean_reference, modal_packet, radial_reference_observables,
    seed_packet,
)
from prepared_neutral_shared_photon_state import load_shared_native_cache
from raw_source_relative_noether_graph import runtime_binding

PLAN = dict(id='issue-222-mercury-synthesis-continuation-20261007',
            revision='f9a076e0785f47644962a38b973e6b87a329d6ad594a4345a4728f4c689aa3be',
            assignment_revision='bf56a5dd39fd60b6bafd47f9ba2e0d0eccf392e3af36d0150b15f1e92746b598',
            template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
            work_item='soft-spatial-closed-rotation',campaign_open=True)
EXPECTED_CACHE = '10d61adb4a374a05e45338522abefcda260ed25036396b1bea26c48c57e4bee3'
EXPECTED_MANIFEST = '90a6d7f34617540042406a07c5d8690cecdc456b2d585ee150c152c7d2e07f9a'
EXPECTED_GUARD = '5aeceaf4a7cb104346ebc5d1130e22bae6090b5cf96afcd297e02a6facc22128'
EXPECTED_LEGAL = '1330558494c1a30897d985ea9dee600fc2ed2c99ebc6be9e365e18c75a50653c'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, complex):
        return dict(real=value.real,imag=value.imag)
    if isinstance(value, dict):
        return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [portable(v) for v in value]
    return value


def short_observation(row: dict[str, Any]) -> dict[str, Any]:
    return {k:v for k,v in row.items() if k not in ('mode_mean_q','mode_mean_p')}


def jet_controls(model: Any, packet: dict[str, Any], reference: dict[str, Any], y0: np.ndarray) -> list[dict[str, Any]]:
    n=model.chart.normal_dim
    rng=np.random.default_rng(2221007)
    vectors=[y0/np.linalg.norm(y0),model.original.directions[:,0],reference['modes'][:,0],reference['modes'][:,-1]]
    for _ in range(2):
        q=rng.standard_normal(n);vectors.append(q/np.linalg.norm(q))
    centre=fixed_seed_potential(model,packet,np.zeros(n))
    e0=sum(centre.values());records=[]
    for index,direction in enumerate(vectors):
        raw_direction=model.chart.mean_basis@direction
        expected_h=float(raw_direction@reference['raw_hessian']@raw_direction)
        expected_g=float(direction@reference['gradient'])
        measurements=[]
        for step in (.0008,.0004):
            plus=sum(fixed_seed_potential(model,packet,step*direction).values())
            minus=sum(fixed_seed_potential(model,packet,-step*direction).values())
            second=(plus+minus-2*e0)/(step*step)
            first=(plus-minus)/(2*step)
            measurements.append(dict(step=step,first=first,second=second,
                                     second_relative_error=abs(second-expected_h)/max(1.,abs(expected_h)),
                                     first_absolute_error=abs(first-expected_g)))
        if min(row['second_relative_error'] for row in measurements)>1e-3:
            raise AssertionError('New analytic mean Hessian disagrees with literal original potential: '+repr((index,expected_h,measurements)))
        records.append(dict(direction=index,analytic_first=expected_g,analytic_second=expected_h,
                            literal_original_potential_controls=measurements))
    return records


def flow_controls() -> dict[str, Any]:
    k=np.array([-2.,0.,3.]);chi=.7;t=.125
    c,s,_=canonical_flow(k,chi,t)
    c2,s2,_=canonical_flow(k,chi,2*t)
    determinant=c*c+chi*k*s*s
    semigroup=float(max(np.max(np.abs(c2-(c*c-chi*k*s*s))),np.max(np.abs(s2-2*c*s))))
    if max(np.max(np.abs(determinant-1)),semigroup)>1e-12:
        raise AssertionError('Real symplectic reference lost zero/negative-mode evolution')
    # An independent direct quadrature checks the analytic affine action on a
    # restoring mode over several oscillations, not a principal-phase reset.
    reference=dict(eigenvalues=np.array([3.,7.]),chi=.7,E0=1.25)
    packet=dict(sigma=.4)
    modal=dict(gradient=np.array([.2,-.3]),Y0=np.array([.5,.4]))
    clock=4.;target=3.7
    nodes,weights=np.polynomial.legendre.leggauss(160)
    quadrature=0.
    for node,weight in zip(nodes,weights):
        q,p=_affine_centres(reference,modal,(node+1)*target/2,clock)
        quadrature+=weight*(float((reference['chi']*p@p-q@(reference['eigenvalues']*q))/2-modal['gradient']@q)-reference['E0'])*target/2
    analytic=continuous_reference_phase(reference,packet,modal,target,clock)
    defect=abs(analytic['affine_classical_action']-quadrature)
    if defect>1e-9:
        raise AssertionError('Canonical affine action omitted force/pulse phase: '+repr((analytic,quadrature)))
    return dict(negative_mode_cosine=c[0],zero_mode_sine_over_frequency=s[1],
                symplectic_determinants=determinant,semigroup_defect=semigroup,
                continuous_affine_action_control_error=defect,
                no_Hessian_clipping=True)


def residual_probes(model: Any, packet: dict[str, Any], reference: dict[str, Any],
                    modal: dict[str, Any], pressure_time: float, clock: float) -> list[dict[str, Any]]:
    rng=np.random.default_rng(2221008);rows=[]
    identity=np.eye(4*model.law.volume)
    ph=model.original.heavy_projector
    for t in (0.,pressure_time,3*pressure_time):
        abi=gaussian_polynomial_abi(reference,packet,modal,t,clock)
        covariance=1/(2*abi['modal_q_precision'].imag)
        for index in range(2):
            displacement=np.sqrt(covariance)*rng.standard_normal(len(covariance))
            qm=abi['modal_gaussian_centre_q']+displacement
            q=reference['modes']@qm
            polynomial=packet['a']+displacement@abi['modal_linear_coefficients']
            bank_prob=np.abs(polynomial)**2/np.sum(np.abs(polynomial)**2)
            parts=fixed_seed_potential(model,packet,q)
            exact=sum(parts.values())
            quadratic=reference['E0']+reference['gradient']@q+.5*qm@(reference['eigenvalues']*qm)
            point=model.chart.embedding(q)
            hl=model.law.onebody(point['phi'],point['links'],np.zeros((model.law.volume,3)))
            hh=model.original.massive_onebody(point['phi'],point['links'])
            heavy_out=(identity-ph)@hh@ph
            light_out2=[]
            for charge in (1,-1):
                p=model.fibre.projectors[charge]
                outside=(identity-p)@hl@p
                light_out2.append(float(np.linalg.norm(outside)**2))
            light_actual=sum(bank_prob[i]*float(np.trace(model.fibre.projectors[charge]@hl).real)
                             for i,charge in enumerate((1,-1)))
            coulomb_actual=float(bank_prob@cast(Any,np.diag(model.HC['neutral'])).real)
            rows.append(dict(time=t,index=index,normal_radius=float(np.linalg.norm(q)),
                             original_scalar_fixed_density_Taylor_remainder=float(exact-quadratic),
                             exact_held_bank_density_light_Coulomb_minus_reference_density=float(
                                 light_actual-parts['light_fixed_seed']+coulomb_actual-parts['fixed_coulomb']),
                             heavy_outside_PH_strong_norm=model.Lambda*float(np.linalg.norm(heavy_out)),
                             light_outside_PH_strong_norm=math.sqrt(float(bank_prob@np.asarray(light_out2))),
                             heavy_relative_SOURCE_even_CAR_commutator='EXACT ZERO by disjoint even CAR; field kinetic not included',
                             quadrature_not_uniform_full_H_remainder_bound=True))
    return rows


def legal_control(receipt: dict[str, Any]) -> dict[str, Any]:
    trial=receipt['raw_correlated_trial']
    l=int(trial['unchanged_selected_log2_Lambda_lower'])
    coefficients=trial['uncut_position_trace_coefficients_log2']
    # Exact integer log upper bounds at the inherited FINITE selected Lambda.
    initial_terms=[int(coefficients['soft'])-(7*l)//20,
                   int(coefficients['stiff'])-(19*l)//40]
    source_initial_tail=max(initial_terms)+1
    # Separate conditional covariance ENERGY from driven ordinary mean energy:
    # Ecov<=C Lambda^(3/8), K>=kappa0 Lambda gives TrSigma<=2C/kappa0 Lambda^-5/8.
    # Do not silently inherit an ordinary Lambda^3/8 window at mid pulse: the
    # unchanged supplier explicitly records a Lambda^1/2 driven common floor.
    return dict(selected_log2_Lambda_lower=l,
                legal_clock=receipt['selected_clock'],
                original_legal_raw_correlated_trial_initial_radial_tail_log2_upper=source_initial_tail,
                original_legal_cutoff_radius='Lambda^(-1/5)',
                source_initial_covariance='2^17 Lambda^(-3/4)+2^401 Lambda^(-7/8); SAME raw correlated trial, NOT native sigma rescaled',
                conditional_reference_covariance_energy='Ecov(0)<=C_N Lambda^(3/8), preserved by constant-Y0 linear forcing',
                conditional_physical_reference_lower_Hessian='Kref>=kappa0_N Lambda I, no clipping; formal fixed-seed comparison certificate still needed',
                conditional_uniform_reference_radial_covariance='TrSigma<=2 C_N/kappa0_N Lambda^(-5/8)',
                conditional_radial_tail='P(|q-mu|>=Lambda^(-1/5))<=2 C_N/kappa0_N Lambda^(-9/40)',
                conditional_covariance_coefficient_log2_required_below=(9*l)//40,
                actual_admitted_post_filter_covariance_NOT_inferred_from_raw_trial=True,
                mean_translation_and_original_common_Lambda_half_drive_floor_separate=True,
                native_proxy_pressure_is_NOT_legal_scale_failure=True,
                no_new_ground_no_new_state_no_native_full1e_minus4_gate=True)


def full_join_ledger() -> dict[str, Any]:
    return dict(
        reference='U_rho^-1 Wflat U_rho tensor IDENTITY on ALL Root/light/heavy/Ref/photon; U_rho psi=rho^(1/2)psi',
        Href='E0+g.q+chi p^2/2+q.Kref.q/2-Omega(t)Y0.p; original ordinary common E0/action retained',
        H_minus_Href=[
            'literal quartic/alignment/plaquette minus its mean Taylor polynomial',
            'actual fixed-Fock H minus integrated original packet density, including PH/multipair fluctuations',
            'offmean FULL normal/gauge cometric, physical density/half-density and gauge/fibre connection',
            'original light/Ref/Coulomb/coexact52/harmonic3 energy/current/contact; no harmonic Phi law added',
            'original beta=1/16 allRoot dynamics and actual SOURCE outsideP, no projected alias deletion',
            '-Omega*(FULL physical G-Y0.p), including DY, RAW clover colour-Haar, Gauss and all Ref/EM/body terms',
            'all64 original branch floors and support-boundary/mixing terms when extending this one native chart',
        ],
        Gref_scope='controlled constant mean Y0 plus full quintic Omega; linearJacobian DY and physical fibre/halfdivergence remain G-Gref',
        RAW_colour_Haar_1p219='legitimate nonzero Noether halfdivergence input, NOT a Gauss residual and NOT forced to zero',
        source='for ALL four literal words and both plus/minus maps: Btilde=Wdagger B W; ONE original initial SOURCE Z; neutral RIGHT retained',
        relative_equation='Rtilde_C Btilde-Btilde Rtilde_N-i d_t Btilde = Wdagger(H_C B-B H_N)W',
        charge='W is field-only, commutes every actual light/Ref charge; no charge marking or new species',
        Gauss='If W does not commute original constraints, Gauss_tilde=Wdagger Gauss W; no original-kernel/dressing admission inferred',
        G='Gtilde=Wdagger Gphysical W; canonical/metaplectic phase is common and NOT physical halfspin',
        current='Jtilde=Wdagger Joriginal W and contact likewise; operator equations unchanged, photon covariance/current residual still unpaid',
        heavy_PH='absolute heavy PH is measurable but disjoint EVEN heavy/source CAR commutator cancels in the SOURCE-relative equation; never discard field-kinetic DB',
        source_control='INSIDE tube: M1(tube)^2 TrSigma. OUTSIDE: retain 2||1outside B Psi||^2+2||B(mu)||^2 Poutside; a probability-only M0 bound requires a GLOBAL bounded-source certificate and is NOT inferred for noncompact Phi. Apply all4-word jets and same initial SOURCE Z.',
        next_decisive_construction='Consume this breathing ABI in the true full48-word SOURCE frame; evaluate all4 H/G/J SOURCE-relative residuals including actual DY/halfdivergence/offmean metric and root/current, then certify same-original LEGAL tube/energy comparison. No new preparation or Berry-pi substitute.',
        conclusion='enabling common physical-sector join, not an isolated covariance finish line; full original spatial2pi/spinhalf/current-scattering remains OPEN')


def consume(export: Path, receipt_path: Path, t: float) -> dict[str, Any]:
    receipt=json.loads(receipt_path.read_bytes())
    if sha(export)!=receipt['export_npz_sha256']:
        raise ValueError('Canonical reference export does not match its actual scientific receipt')
    with np.load(export,allow_pickle=False) as stored:
        k,modes,g,y,w,a=(stored[key].copy() for key in ('eigenvalues','modes','gradient','Y0','packet_w','packet_a'))
        scalars=stored['scalars'].copy();mq=stored['initial_mean_q'].copy();mp=stored['initial_mean_p'].copy()
    chi,sigma,radius,z,e0,clock=scalars
    reference=dict(eigenvalues=k,modes=modes,gradient=g,Y0=y,chi=chi,E0=e0)
    packet=dict(w=w,a=a,n=len(k),sigma=sigma,radius=radius,original_norm_squared=z,mean_q=mq,mean_p=mp)
    modal=modal_packet(reference,packet,y)
    abi=gaussian_polynomial_abi(reference,packet,modal,t,clock)
    core_variance=1/(2*abi['modal_q_precision'].imag)
    reconstructed=float(np.sum(abs(a)**2)+np.sum(core_variance[:,None]*abs(abi['modal_linear_coefficients'])**2))
    if abs(reconstructed-z)>1e-10*z:
        raise AssertionError('Consumed exact metaplectic packet lost its original global norm')
    c,s=abi['c'],abi['s']
    symplectic_defect=float(np.max(np.abs(c*c+chi*k*s*s-1)))
    if symplectic_defect>1e-10:
        raise AssertionError('Receipt-bound reference consumer lost canonical flow')
    return dict(scope='actual receipt-bound canonical reference consumer; no state preparation',
                input_export_sha256=sha(export),input_receipt_sha256=sha(receipt_path),time=t,
                original_global_norm_squared=z,metaplectic_core_reconstructed_norm_squared=reconstructed,
                all_normal_modes=len(k),symplectic_defect=symplectic_defect,
                actual_minimum_Gaussian_precision_imag=float(np.min(abi['modal_q_precision'].imag)),
                continuous_common_phase=abi['phase'],radial=short_observation(abi['moments']),
                original_cutoff_remainder=abi['cutoff_remainder'],
                assigned_intrinsic_spin=False,full_physical_2pi_claim=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-npz',type=Path)
    parser.add_argument('--cache-manifest',type=Path)
    parser.add_argument('--guard',type=Path)
    parser.add_argument('--legal-receipt',type=Path)
    parser.add_argument('--output-npz',type=Path)
    parser.add_argument('--output-json',type=Path)
    parser.add_argument('--runtime-archive',type=Path)
    parser.add_argument('--consume',type=Path)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--time',type=float,default=.0005)
    parser.add_argument('--clock',type=float,default=566.)
    args=parser.parse_args();started=time.perf_counter()
    if args.consume is not None:
        if args.receipt is None:
            parser.error('--consume requires --receipt')
        print(json.dumps(portable(consume(args.consume,args.receipt,args.time)),indent=2));return
    required=('cache_npz','cache_manifest','guard','legal_receipt','output_npz','output_json','runtime_archive')
    for name in required:
        if getattr(args,name) is None:
            parser.error('scientific run requires --'+name.replace('_','-'))
    for name,expected in (('cache_npz',EXPECTED_CACHE),('cache_manifest',EXPECTED_MANIFEST),
                          ('guard',EXPECTED_GUARD),('legal_receipt',EXPECTED_LEGAL)):
        if sha(getattr(args,name))!=expected:
            raise ValueError('Frozen supplier differs: '+name)
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    packet=seed_packet(model,data)
    if packet['boost_Omega_at_actual_cache_restore']!=0:
        raise ValueError('Controlled original zero-ended IC must not be silently deboosted')
    guard=json.loads(args.guard.read_bytes());y0=np.asarray(guard['Y0_normal'],float)
    reference=mean_reference(model,packet)
    modal=modal_packet(reference,packet,y0)
    controls=flow_controls();jets=jet_controls(model,packet,reference,y0)
    four=current_six_projection(reference,packet,y0,4)
    five=current_six_projection(reference,packet,y0,5)
    rule_difference=abs(four['outside_strong_Href_rate']-five['outside_strong_Href_rate'])
    if rule_difference>1e-4*max(1.,four['outside_strong_Href_rate']):
        raise AssertionError('Independent exact polynomial rules disagree on physical trial-space pressure')
    n,sigma,chi=packet['n'],packet['sigma'],reference['chi']
    initial=radial_reference_observables(reference,packet,modal,0.,args.clock)
    free_scale=math.sqrt(max(0.,packet['radius']**2-initial['radial_q_second_moment'])/(chi*chi*initial['Tr_p_covariance']))
    def pressure(t: float) -> float:
        return radial_reference_observables(reference,packet,modal,t,args.clock)['radial_q_second_moment']-packet['radius']**2
    upper=max(free_scale,1e-8)
    while pressure(upper)<0 and upper<args.clock:
        upper*=2
    crossing=0. if pressure(0.)>=0 else cast(float,brentq(pressure,0.,upper,xtol=1e-14))
    times=(0.,crossing/3,crossing,3*crossing,.01,args.clock/4,args.clock/2,3*args.clock/4,args.clock)
    observations=[]
    for t in times:
        row=radial_reference_observables(reference,packet,modal,t,args.clock)
        centre_q,centre_p=_affine_centres(reference,modal,t,args.clock)
        pulse=row['pulse_Omega']
        _,sine,integrals=canonical_flow(reference['eigenvalues'],chi,t)
        gradient_q=-chi*modal['gradient']*integrals[0]
        gradient_p=-modal['gradient']*sine
        Noe_q=centre_q-gradient_q
        Noe_p=centre_p-gradient_p
        row.update(bare_affine_mean_momentum_norm=float(np.linalg.norm(centre_p)),
                   boosted_affine_mean_momentum_norm=float(np.linalg.norm(centre_p-pulse*modal['Y0']/chi)),
                   SOURCE_relative_mean_field_velocity_norm=float(np.linalg.norm(chi*centre_p-pulse*modal['Y0'])),
                   pure_constant_gradient_mean_q_norm=float(np.linalg.norm(gradient_q)),
                   pure_Noe_mean_q_norm=float(np.linalg.norm(Noe_q)),
                   pure_Noe_bare_mean_p_norm=float(np.linalg.norm(Noe_p)),
                   pure_Noe_boosted_mean_p_norm=float(np.linalg.norm(Noe_p-pulse*modal['Y0']/chi)),
                   pure_Noe_SOURCE_relative_mean_field_velocity_norm=float(np.linalg.norm(chi*Noe_p-pulse*modal['Y0'])),
                   affine_gaussian_mean_q_norm=float(np.linalg.norm(centre_q)))
        observations.append(short_observation(row))
    probes=residual_probes(model,packet,reference,modal,crossing,args.clock)
    k=reference['eigenvalues']
    zero=int(np.sum(k==0));negative=int(np.sum(k<0));smallest=float(k.min())
    reference_stats={key:reference[key] for key in ('E0','mean_potential_parts','metric_defects','raw_symmetry_defect','flat_defect','uniform_defect','pieces','spectral')}
    reference_stats.update(smallest_Hessian_eigenvalue=smallest,largest_Hessian_eigenvalue=float(k.max()),
                           exact_zero_eigenvalue_count=zero,negative_eigenvalue_count=negative,
                           numerical_near_zero_count=int(np.sum(abs(k)<=32*np.finfo(float).eps*max(abs(k))*90)),
                           Hessian_eigenvalues_NEVER_clipped=True,mean_gradient_norm=float(np.linalg.norm(reference['gradient'])))
    if smallest>0:
        ecov=initial['covariance_energy']
        reference_stats['uniform_positive_reference_TrSigma_upper']=2*ecov/smallest
    legal=json.loads(args.legal_receipt.read_bytes())
    science=dict(scope='NEW all2430 original-law mean-quadratic/metaplectic reference on SAME unchanged native original packet',
                 plan=PLAN,native_law=dict(N=model.law.size,V=model.law.volume,Lambda=model.Lambda,chi=chi,
                     sigma=sigma,radius=packet['radius'],clock=args.clock,light_coefficient=1.,heavy_coefficient=-model.Lambda,
                     alignment_and_plaquette_Lambda_power='5/4',Phi_quartic_Lambda_power='1',
                     original_EM55_retained_in_full_R_H_not_replaced_by_Phi_oscillators=True),
                 original_initial_packet=dict(global_uncut_norm_squared=packet['original_norm_squared'],
                     original_coefficients14=data['preparation_state14'],fixed_bank_weights=packet['bank_weights'],
                     light_fixed_projector_ranks=[float(np.trace(model.fibre.projectors[q]).real) for q in (1,-1)],
                     heavy_fixed_projector_rank=float(np.trace(model.original.heavy_projector).real),
                     original_preparation_directions=data['seed_original_15'].shape,
                     original_static_boost_Omega=packet['boost_Omega_at_actual_cache_restore'],
                     no_coefficient_renormalization_no_regrounding_no_prep_change=True),
                 reference=reference_stats,new_literal_potential_jet_controls=jets,canonical_flow_controls=controls,
                 actual_current_six_strong_projection=four,independent_exact_Gaussian_rule_rate_difference=rule_difference,
                 initial_compact_graph_bounds=cutoff_graph_bounds(packet),native_reference_pressure_cross_time=crossing,
                 radial_covariance_and_separate_mean_observations=observations,
                 observed_original_H_minus_reference_probes=probes,
                 LEGAL_original_supplier_control=legal_control(legal),physical_sector_JOIN=full_join_ledger(),
                 decisions=dict(native_original_narrow_packet_breathes_before_quintic_mean_translation=True,
                     actual_six_already_contains_full2430_radial_not_six_coordinate_axes=True,
                     finite_fixed_polynomial_space_not_an_adapted_breathing_gaussian=True,
                     full_H_ref_remainder_or_physical_G_DY_loop_NOT_paid=True,
                     canonical_common_Maslov_or_Berry_phase_NOT_physical_halfspin=True,
                     full_native1e_minus4_NOT_an_acceptance_gate=True))
    arrays=dict(eigenvalues=k,modes=reference['modes'],gradient=reference['gradient'],Y0=y0,
                packet_w=packet['w'],packet_a=packet['a'],packet_b=packet['b'],
                initial_mean_q=packet['mean_q'],initial_mean_p=packet['mean_p'],
                scalars=np.asarray([chi,sigma,packet['radius'],packet['original_norm_squared'],reference['E0'],args.clock]),
                original_coefficients14=data['preparation_state14'])
    np.savez_compressed(args.output_npz,**arrays)
    binding=runtime_binding(args.runtime_archive)
    receipt=dict(science=science,export_npz_sha256=sha(args.output_npz),export_npz=str(args.output_npz),
                 source_sha256={Path(__file__).name:sha(Path(__file__)),
                     'common_field_quadratic_reference.py':sha(Path(__file__).with_name('common_field_quadratic_reference.py'))},
                 actual_input_sha256={name:sha(getattr(args,name)) for name in ('cache_npz','cache_manifest','guard','legal_receipt')},
                 runtime_binding=binding,actual_python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,
                 argv=sys.argv,wall_seconds=time.perf_counter()-started,
                 metadata_scope='explicit argv/source/scientific inputs/versions only; NO environment/credentials/configuration dump')
    args.output_json.write_text(json.dumps(portable(receipt),indent=2))
    print(json.dumps(dict(output_json=str(args.output_json),output_json_sha256=sha(args.output_json),
                         export_npz_sha256=receipt['export_npz_sha256'],pressure_cross_time=crossing,
                         smallest_Hessian_eigenvalue=smallest,negative_eigenvalues=negative,
                         radial_exit_lower_at_three_crossings=observations[3]['ORIGINAL_compact_reference_radial_exit_probability_lower'],
                         outside_current_six_reference_rate=four['outside_strong_Href_rate'],
                         wall_seconds=receipt['wall_seconds']),indent=2))


if __name__=='__main__':
    main()
