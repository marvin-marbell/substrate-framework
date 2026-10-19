#!/usr/bin/env python3
"""Original-law Dirichlet comparison: freeze ONLY light Phi/colour coefficients.

Original EM/Peierls, Coulomb, heavy/Root/neutral and physical Noether dynamics
remain. This comparison is not a new preparation or an admitted spin sector.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
import numpy as np
from improved_phi_band import NSTAR
from original_charged_principal import portable
from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch
from prepared_neutral_full_field import BETA
from prepared_neutral_phi_word_channels import coefficient_channels

C2_SHA = '10925ce66ab64c8f4c6a8fd8574baf183539be5827ac4dd495545357b28ab03e'
MEAN_SHA = '63f3b4114689177dd5a224cee4135db0ea198c31e14bc1d535b88fb38fd45a68'
MASSIVE_SHA = '1330558494c1a30897d985ea9dee600fc2ed2c99ebc6be9e365e18c75a50653c'
NORMAL_SHA = '07180616d355175aee1a4ff23c9b9dc1a2dd55a072144ed6073f28fce8560d77'
CCR_SHA = 'bcb3da759e33581dc3f8397db165f6be65ee9e983c58671bb88032cc0963a225'
AUX_SHA = '306a7608bce161a4c83ae94baf5254185de5d17b048fd36f3e17d4ea2bcff1c2'


def ceil_log2(value: F | int) -> int:
    value = F(value)
    if value <= 0:
        raise ValueError('Positive exact bound required')
    k = value.numerator.bit_length()-value.denominator.bit_length()
    power = F(2)**k
    return k if power >= value else k+1


def mean_light(model: Any, sample: dict[str, Any], first: np.ndarray | None = None,
               second: np.ndarray | None = None) -> np.ndarray:
    """SAME literal Peierls/harmonic/current/contact law, mean Phi/colour only."""
    return model.law.onebody(model.chart.mean_phi, model.chart.mean_links, sample['a'], first=first, second=second)


def mean_source(model: Any, sample: dict[str, Any], profile: str) -> np.ndarray:
    """Original raw source at mean Phi/colour; same theta, profile, quartet, Z."""
    return model.source_bank.seed_fibre.source_vectors_at(model.chart.mean_phi, model.chart.mean_links,
        sample['theta'], sample['a'], model.source_bank.profiles[profile], BETA)['source_matrix']


def light_lipschitz(law: Any, volume: int, frame_bound: F = F(2)) -> dict[str, Any]:
    """Global in A: all scalar Peierls factors have modulus one.

    Phi HS and link-exp displacement <=2*d in the original mass-horizontal
    chart. Triangle over individual encoded bilinears avoids assuming CAR.
    """
    onsite = abs(F.from_float(float(law.onsite)))
    rows = []
    for displacement, scalar in law.paths:
        length = sum(abs(int(x)) for x in displacement)
        coefficient = sum((abs(F.from_float(float(x.real)))+abs(F.from_float(float(x.imag))) for x in scalar), F())
        rows.append(dict(length=length, scalar_l1=coefficient))
    constant = 16*volume*(2*onsite+4*sum((r['scalar_l1']*(1+frame_bound*r['length']) for r in rows), F()))
    return dict(operator_Lipschitz_upper_fraction=str(constant), coefficient_log2_upper=ceil_log2(constant),
        full_encoded_bilinear_triangle=True, no_CAR_or_ground_assumption=True,
        phase_modulus_one_for_ALL_unbounded_A=True, mean_Phi_operator_norm_upper_fraction=str(frame_bound),
        paths=[dict(length=row['length'], scalar_l1_fraction=str(row['scalar_l1'])) for row in rows])


def fraction(value: dict[str, str]) -> F:
    return F(int(value['numerator_hex'],16),int(value['denominator_hex'],16))


def negative_power_log(coefficient: int, power: F, lambda_lower: int) -> int:
    """Combine powers FIRST. A lower Lambda bound cannot price growth."""
    if power > 0:
        raise ValueError('A positive Lambda power has no supplied upper bound')
    return coefficient-((-power.numerator*lambda_lower)//power.denominator)


def whole_dirichlet_mixed(control: dict[str, Any], graph_log: int,
                          lambda_lower: int) -> dict[str, Any]:
    """Magnetic IMS on the entire ORIGINAL D, not the analysis-only tube.

    d is the infimum of the original mass-normal distances to the finite
    class centres: a globally 1-Lipschitz scalar. Equal/overlapping centres
    do not add a rank factor. Use smooth Lipschitz approximants in H1_0,
    then dominated convergence; no second derivative of d is required.
    Gauss/Fock connection and the scalar C1 boost commute with d.
    """
    q2 = int(control['Dirichlet_actual_union_distance_second_moment_log2_upper'])
    d2_coefficient = q2+lambda_lower//2
    # Full105 weighted J has ||J||<=16, so gInv>=1/256. Thus
    # ||chi Pboost(d psi)||²<=512 chi E(d psi).
    # E(d psi)=Re<d²psi,Kfastpsi>+chi/2<|grad d|²_gInv>.
    # gInv<=160 and RD<=1. Product rule plus (a+b)²<=2(a²+b²)
    # gives 1024 chi RD sqrt<d²>||Kfastpsi||+81922 chi².
    first = negative_power_log(10+(d2_coefficient+1)//2+graph_log,F(-1,4),lambda_lower)
    second = negative_power_log(17,F(-1),lambda_lower)
    squared = max(first,second)+1
    return dict(full_Dirichlet_d_chiPboost_squared_log2_upper=squared,
        full_Dirichlet_d_chiPboost_norm_log2_upper=(squared+1)//2,
        strong_Kfast_coefficient_log2_upper=graph_log,
        distance_second_moment_coefficient_log2_upper=d2_coefficient,
        first_squared_Lambda_power='-1/4', second_squared_Lambda_power='-1',
        no_analysis_tube_cutoff=True, original_D_radius_upper=1,
        scalar_distance_Lipschitz_upper=1, full_metric_upper=256, full_cometric_upper=160,
        magnetic_doublecommutator='[d,[Kfast,d]]=chi (grad d)†gInvNN(grad d); Fock connection/boost/scalar potentials commute with d',
        domain='H1_0 of original regular union, scalar Lipschitz multiplier preserves Dirichlet trace; weak IMS via smooth approximants, actual psi in D(Kfast)',
        no_unbounded_whole_transfer=True)


def relative_phi_costs(volume: int, mixed: dict[str, Any],
                       lambda_lower: int, clock_log: int) -> dict[str, Any]:
    """Uniform full-Fock operator bounds, not a charged PH truncation.

    Encoded fermion norm<=1 suffices for the triangle. HS(t_a)=1/2,
    entrywise l1<=2, three species incl sparse Ref give sum||Q_a||²<=540V.
    Raw B<=16 sqrt(V), original source Z>=1/4; no normalization of B0.
    """
    v = ceil_log2(volume)
    # At the horizontal mean gNG=0. Full J perturbation bounds give
    # ||gNG||<=32d, ||A_N=-gNG gGG^-1||<=5120d<2^13d.
    # ||rQ_B0||_l2<=4*(64sqrtV)*sqrt(540V)<2^13 V.
    # ||gInvNN||<=160<2^8. Leading commutator <=2^34 V d chi P.
    cross = clock_log+34+v+mixed['full_Dirichlet_d_chiPboost_norm_log2_upper']
    # Derivative bounds: ||dJ||<=8, ||dg||<=256, ||dg^-1||<2^23,
    # ||dA_N||<2^28. Divergence adds 90V; log-Haar derivative
    # <=(105V/2)*160*256<2^22 V. Loose 2^60 V² pays both.
    derivative = negative_power_log(clock_log+60+2*v,F(-1,2),lambda_lower)
    # Keep the entire vertical term: it is NOT generically quadratic.
    # Q†gGG^-1 Q<=160*540V and source commutator <=2 CB times
    # that bound. A_N² completion is bounded using d<=RD<=1.
    vertical = negative_power_log(clock_log+30+2*v,F(-1,2),lambda_lower)
    square = negative_power_log(clock_log+64+2*v,F(-1,2),lambda_lower)
    combined = max(cross,derivative,vertical,square)+2
    return dict(actual_side_relative_Phi_kinetic_integral_log2_upper=combined,
        cross_log2_upper=cross, density_and_coefficient_derivative_log2_upper=derivative,
        full_vertical_Gauss_log2_upper=vertical, connection_square_log2_upper=square,
        source_operator_bound='64sqrt(V) after SAME original Z>=1/4; B0 not renormalized',
        charge_square_operator_upper='540V, full encoded triangle, all light/heavy/Ref retained',
        full_Phi_completion='chi/2 (P_N+A_N Q)†gInvNN(P_N+A_N Q)+chi/2 Q†gGG^-1(q)Q',
        scope='Constant-horizontal B0 relative Phi-form residual. Full microscopic Gauss/encoded-CAR identification and packet/Root intertwining are NOT inferred from this bound.',
        vertical_first_variation_not_erased=True, no_PH_or_BO_elimination=True,
        entire_original_D_used=True)


def actual_auxiliary_cost(auxiliary: dict[str, Any], volume: int,
                          lambda_lower: int, clock: int) -> dict[str, Any]:
    """Projected actual-state equation; never a reference-state graph gate."""
    bound = auxiliary['science']
    if F(bound['beta']) != F(1,16) or int(bound['original_clock']) != clock:
        raise ValueError('Original total beta and clock binding required')
    q_log = negative_power_log((ceil_log2(F(bound['leakage_squared_coefficient']))+1)//2,
                              F(-1,8),lambda_lower)
    cb_log = 6+(ceil_log2(volume)+1)//2
    forcing = ceil_log2(clock)-4+cb_log+q_log+1
    remainder = cb_log+q_log+1
    return dict(actual_Q_norm_log2_upper=q_log,
        original_P_D_Q_forcing_SOURCE_integral_log2_upper=forcing,
        bounded_SOURCE_projection_remainder_log2_upper=remainder,
        combined_auxiliary_norm_error_log2_upper=max(forcing,remainder)+1,
        full_flat_directsum_retained=8, colour_aliases_retained=64,
        physical_original_state_not_projected=True, no_reference_state_moment_gate=True,
        equation='i d(PPsiD)=P H_D P(PPsiD)+P D Q PsiD; ONLY original beta D mixes P/Q',
        CAR_scope='Exact ordinary even CAR on full flat-P direct sum, history loop-sign twists retained; not identical untwisted boundary conditions',
        unbounded_transfer='P commutes original coordinate/Pi; actual mixed graph applies directly to Ppsi. betaQ norm alone is NOT an unbounded derivative estimate',
        adverse_raw_commutator='||[D,B0]Ppsi||<=2beta CB is NOT small in Lambda; use projected actual inhomogeneous equation and paid Q remainder, never drop beta')


def legal_reference(science: dict[str, Any], law: Any, massive: dict[str, Any],
                    normal: dict[str, Any], ccr: dict[str, Any],
                    auxiliary: dict[str, Any]) -> dict[str, Any]:
    volume = int(NSTAR)**3
    constants = light_lipschitz(law, volume)
    logC = constants['coefficient_log2_upper']
    logT = ceil_log2(int(science['clock']))
    lam = int(science['selected_log2_Lambda_lower'])
    if fraction(normal['legal_N']['quotient_radius']) > 1:
        raise ValueError('Original Dirichlet radius RD<=1 required')
    if int(ccr['bindings']['selected_log2_lambda_lower']) != lam:
        raise ValueError('Initial creator addendum uses another Lambda')
    if ccr['frozen_legal_sha256'] != C2_SHA:
        raise ValueError('Initial creator addendum uses another original state')
    results = {}
    for label, key in (('neutral_and_SOURCE','full_pulse_neutral_and_SOURCE_actual_controls'),
                       ('true_photon_SOURCE','full_pulse_true_photon_unweighted_actual_controls')):
        control = science[key]
        q2 = int(control['Dirichlet_actual_union_distance_second_moment_log2_upper'])
        light_error = logT+logC+(q2+1)//2
        boundary = int(control['actual_whole_minus_D_norm_log2_upper'])
        source_initial = int(science['initial_mean_body_SOURCE_comparison']['initial_SOURCE_same_Z_error_squared_log2_upper'])
        initial_square = int(ccr['squared_norm_log2_upper']) if label=='true_photon_SOURCE' else source_initial
        initial = (initial_square+1)//2
        graph = (int(science['NEW_true_photon_initial_fast_graph']['propagated_strong_Kfast_norm_coefficient_log2_upper'])
                 if label=='true_photon_SOURCE' else int(massive['actual_regular_union_unweighted_graph_bridge']['second_graph_norm_coefficient_log2']))
        mixed = whole_dirichlet_mixed(control,graph,lam)
        phi = relative_phi_costs(volume,mixed,lam,logT)
        aux = actual_auxiliary_cost(auxiliary,volume,lam,int(science['clock']))
        results[label] = dict(actual_Dirichlet_light_freeze_norm_error_log2_upper=light_error,
            original_whole_to_Dirichlet_norm_error_log2_upper=boundary,
            initial_SAME_Z_BminusB0_norm_error_log2_upper=initial,
            initial_truephoto_independent_CCR_not_SOURCE_error_copy=label=='true_photon_SOURCE',
            bounded_light_comparison_error_log2_upper=max(light_error,boundary,initial)+2,
            full_Dirichlet_mixed_graph=mixed, relative_Phi_kinetic=phi,
            actual_auxiliary_projection=aux,
            paid_field_and_auxiliary_side_error_log2_upper=max(light_error,boundary,initial,
                phi['actual_side_relative_Phi_kinetic_integral_log2_upper'],
                aux['combined_auxiliary_norm_error_log2_upper'])+3,
            no_unbounded_whole_moment_or_creator_transfer_from_norm_error=True,
            preserved_timeconnection_and_Omega_start_normal_drift=True)
    return dict(LEGAL_N=int(NSTAR), constants=constants, clock=int(science['clock']), results=results,
        theorem='Uactual_D(t)−Uref_D(t)=−i integral Uref_D(t,s) DeltaHlight Uactual_D(s) ds; residual evaluated on ACTUAL Dirichlet state, not reference state',
        domain='ORIGINAL regular union Dirichlet; actual full512 histories and64 coherent aliases; no new tube, branch floor or superselection',
        reference='Hfast(actual Phi/metric/Gauss/heavy/Root)+H_EM(noncompact ALLmodes/harmonic)+HC(full Fock)+Hlight(meanPhi,meanColour,A), SAME Gphysical and C1 Vbeta/timeconnection',
        existence='Same Dirichlet closed-form domain; DeltaHlight bounded uniformly in all unbounded A. Bounded-perturbation evolution defines reference, not a BO ground.',
        projected_auxiliary='i d_t(P PsiD)=P H_D P(PPsiD)+P D Q PsiD; actual betaQ input is bound and priced; no new reference-code-flow moment assumption',
        current_observation='Original J/contact remain; mean-light reference uses SAME Peierls primitive. A-uniform difference and literal photon legs remain, no creator-weighted norm transfer.',
        common_drift='B0 is genuinely constant horizontal: [−Omega_start D_Y0,B0]=[beta_dot(a.q),B0]=0. Both remain COMMON ABSOLUTE dynamics.',
        bare_coexact_SOURCE='Original screened Green divA coexact word zero, harmonic retained; NOT generalized to filter/current/ordered-frame creator',
        preserved=['original fn/Gamma/globalZ and ordinary alias phases','true neutral RIGHT and FULL heavy Fock',
                   'all noncompact EM/harmonic, Coulomb and Root beta','−Omega_start DY and +beta_dot(a.q)'],
        unpaid=['full rH B0(f) packet light/Coul multipair/current intertwiner',
                'mixed Gauss/Fock microscopic sector identification beyond paid coefficient-form bound',
                'mechanical-core/Ref spatial finite group; actual omega smooth correction constant',
                'fixed LAB effect admission','quantum law/parameters'])


def native_observe(model: Any, data: dict[str, Any], mean_arrays: Any) -> dict[str, Any]:
    # This is a NEW first-variation observation, not a repeat of the sealed
    # mean cometric/noether SCI. It uses its exact frozen arrays as input.
    phi_rows = np.concatenate([np.arange(105*x,105*x+60) for x in range(model.law.volume)])
    gauge_phi = model.chart.mean_gauge[phi_rows]
    inertia_phi = gauge_phi.T@(model.chart.mass[phi_rows,None]*gauge_phi)
    inverse_GG = mean_arrays['actual_cometric_GG']
    radial_first = -2*inverse_GG@inertia_phi@inverse_GG
    rng = np.random.default_rng(2222042)
    q = rng.standard_normal(model.chart.normal_dim);q *= .0001/np.linalg.norm(q)
    records = []
    constants = light_lipschitz(model.law,model.law.volume)
    for scale in (1.,1000.):
        em = rng.standard_normal(55)*scale
        sample = model.basis_at(q,em)
        actual = model.law.onebody(sample['point']['phi'],sample['point']['links'],sample['a'])
        frozen = mean_light(model,sample)
        norm = float(np.linalg.norm(actual-frozen,ord=2))
        first = model.em['legs'][:,0].reshape(model.law.volume,3)
        j_actual = model.law.onebody(sample['point']['phi'],sample['point']['links'],sample['a'],first=first)
        j_ref = mean_light(model,sample,first=first)
        c_actual = model.law.onebody(sample['point']['phi'],sample['point']['links'],sample['a'],first=first,second=first)
        c_ref = mean_light(model,sample,first=first,second=first)
        sources = {}
        for profile in ('source19','SOFT'):
            original = model.source_bank.seed_fibre.source_vectors_at(sample['point']['phi'],sample['point']['links'],sample['theta'],sample['a'],model.source_bank.profiles[profile],BETA)['source_matrix']
            b0 = mean_source(model,sample,profile)
            _, metadata = coefficient_channels(data,'charged',profile)
            Z = metadata['global_SOURCE_normalizer']
            sources[profile] = dict(actual_BminusB0_operator_Frobenius_per_SAME_Z=float(np.linalg.norm(original-b0)/Z),
                no_B0_normalization_or_IC_replacement=True)
        records.append(dict(all55_EM_input_scale=scale,qnorm=float(np.linalg.norm(q)),light_full108_difference=norm,
            conservative_encoded_fullFock_bound=float(F(constants['operator_Lipschitz_upper_fraction']))*float(np.linalg.norm(q)),
            original_current_full108_difference=float(np.linalg.norm(j_actual-j_ref)),
            original_contact_full108_difference=float(np.linalg.norm(c_actual-c_ref)),sources=sources))
    return dict(new_first_radial_GG_inverse_variation_norm=float(np.linalg.norm(radial_first)),
        mean_inverse_GG_norm=float(np.linalg.norm(inverse_GG)),
        Gauss_inertia_radial_first_derivative='2 G_Phi† M_Phi G_Phi; gInvGG_prime=−2 gInvGG G_Phi†M_PhiG_Phi gInvGG at horizontal mean',
        adverse_claim_rejected='All vertical Gauss cometric variations are O(q²); uniform Phi radial variation is already first order',
        analytical_mean_horizontality='sqrtM G0=Qg R, N0=M^−1/2 Qn, Qn†Qg=0; gNN0=I,gNG0=0 and gInvGN0=0 EXACT by chart definition, not assumed offmean',
        relative_kinetic='A_N=−gNG gGG^−1=O(q); chi A_N Q·Pboost uses paid mixed graph, chi A_N²Q² is O(chi q²). Keep full vertical Q†gGG^−1(q)Q, or bound its first-order variation separately; no PH deletion.',
        common_drift='Constant horizontal B0 commutes with −Omega_start Y0·D_N and beta_dot(a.q); retain both in absolute fast/neutral dynamics, do NOT remove them from the state',
        native=records,no_native_weak_MC_IBP_or_spin_gate=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-npz','cache-manifest','C2-LEGAL','mean-arrays','massive','normal','initial-CCR','auxiliary','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    paths=((args.C2_LEGAL,C2_SHA),(args.mean_arrays,MEAN_SHA),(args.massive,MASSIVE_SHA),
           (args.normal,NORMAL_SHA),(args.initial_CCR,CCR_SHA),(args.auxiliary,AUX_SHA))
    if any(hashlib.sha256(path.read_bytes()).hexdigest()!=digest for path,digest in paths):
        raise ValueError('Exact frozen original-state/geometry/graph/CCR input bindings required')
    science=json.loads(args.C2_LEGAL.read_bytes())['science']
    massive=json.loads(args.massive.read_bytes());normal=json.loads(args.normal.read_bytes())
    ccr=json.loads(args.initial_CCR.read_bytes())
    auxiliary=json.loads(args.auxiliary.read_bytes())
    model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest);before=loaded_source_epoch()
    with np.load(args.mean_arrays) as arrays:
        native=native_observe(model,data,arrays)
    result=dict(legal=legal_reference(science,model.law,massive,normal,ccr,auxiliary),native=native,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(path):digest for path,digest in paths},
        cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        loaded_sources=before,loaded_sources_stable=before==loaded_source_epoch(),program_wall_seconds=time.monotonic()-start,
        scope='Actual new original-law light-only comparison construction, same-state Dirichlet unweighted transfer and independent native radial/current/source observations; not full particle/spin/Born admission')
    text=json.dumps(portable(result),sort_keys=True);args.output.write_text(text);os.chmod(args.output,0o600);print(text)


if __name__=='__main__':main()
