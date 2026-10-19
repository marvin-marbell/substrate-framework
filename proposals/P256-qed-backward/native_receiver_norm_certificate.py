"""Native fixed F2 norm bounds: actual compact SOURCE, not a retuned receiver.

The Gaussian mean-SOURCE function is used ONLY as a comparison vector. Actual
birth/ref frames, all centre paths, fixed seed CAR and all Noether extras stay in
F2. This native N3 factory certificate is not a statement about the LEGAL union
receiver, all64 physical-history error budgets or the original scattering law.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from noncompact_screened_gauss_dressing import source_paths
from functional_source_photon_bank import Full108ReferenceMarks
from sea_selection import SELECTED


def compact_frame_constants(model: Any) -> dict[str,Any]:
    """Analytic row-polar/least-intertwiner/polar SU4 guards on actual support.

    For row polars P=HQ, P0=H0Q0, ReTr((P-P0)(Q-Q0)^dagger)
    >=(smin(P)+smin(P0))*||Q-Q0||F²/2. Thus Gamma=2Q has
    ||deltaGamma||F <=4||deltaPhi||F/(2*smin(P0)-||deltaPhi||F).
    The exact mean Clifford commutator system has smallest nonzero singular2.
    """
    mean=model.chart.mean_phi;gram=np.einsum('xmij,xnji->xmn',mean,mean).real/4
    minimum=float(np.min(np.linalg.eigvalsh(gram)))
    coefficients=np.einsum('xmij,mji->xm',mean,SELECTED).real/4
    residual=float(np.linalg.norm(mean-coefficients[:,:,None,None]*SELECTED[None]))
    if minimum<.06 or residual>1e-12 or np.any(coefficients<=0):
        raise ValueError('Native certificate requires actual positive diagonal Clifford mean; not inferred for another factory')
    if np.linalg.norm(model.chart.mean_links-np.eye(4))>1e-12:
        raise ValueError('Native mean SOURCE identity links are not an arbitrary-field assumption')
    core=model.radius/2;delta=core/math.sqrt(2);s0=2*math.sqrt(.06);gap=1.99
    dsystem=8*delta/(2*s0-delta)
    least=2*math.sqrt(2)*dsystem/gap
    polar=2*least/(1-least)
    # Eigenangle determinant removal costs <=(1+1/sqrt(1-e²/4))*e.
    # 2.1 is a strict upper reserve for e<=.5; argdet stays within pi.
    if polar>=.5:raise ValueError('Actual compact core does not pay continuous SU4/Z4 lift guard')
    lipschitz=2.1*2/(1-least)*2*math.sqrt(2)/gap*8/(2*s0-delta)
    frame=lipschitz*delta;link=math.sqrt(.3)*core
    edge=2*frame+link;score_gap=4-2*math.sqrt(2)*edge
    if score_gap<=0:raise ValueError('Actual original whole-centre packet cannot be bounded near identity on this support')
    paths=source_paths(model.law,model.reference_site);length=max(map(len,paths))
    scalar=length*(6*math.exp(-18*score_gap)+6*math.exp(-72))
    # Full radius guard, even though comparison only uses cutoff-one core.
    outer=model.radius/math.sqrt(2);outer_d=8*outer/(2*s0-outer)
    outer_v=2*math.sqrt(2)*outer_d/gap
    full_gap=(gap-outer_d)**2-outer_d**2
    if full_gap<=1e-8 or .5-outer_v<=1e-8:raise ValueError('Actual full compact source chart/polar guard is unpaid')
    return dict(actual_mean_Gram_min=minimum,mean_Clifford_residual=residual,
        exact_mean_nonzero_intertwiner_singular=2.,outward_mean_Gram_lower=.06,outward_nonzero_singular_lower=gap,
        core_radius=core,original_compact_radius=model.radius,Phi_Frobenius_per_raw_coordinate_squared=.5,
        rowpolar_system_change_bound=dsystem,least_vector_change_bound=least,polar_change_bound=polar,
        SU4_determinant_lift_factor=2.1,birth_frame_Lipschitz_per_Phi_Frobenius=lipschitz,
        maximum_birth_frame_core_change=frame,actual_colour_link_core_change=link,
        centre_edge_core_change=edge,original_centre_score_gap_lower=score_gap,
        actual_maximum_original_path_length=length,original_whole_centre_path_error=scalar,
        full_compact_Gram_lower=((s0-outer)/2)**2,full_compact_intertwiner_squared_gap_lower=full_gap,
        full_compact_candidate_polar_min_lower=.5-outer_v,
        centre_gluing='F_x -> z_x F_x changes edge psi1/psi0 by conjugate(z_birth)*z_ref along path; exact character cancels birth/ref Z4. Continuous centre alignment is proof coordinates, not a nearest-centre source law.',
        Ref_action='Actual fullRef wedge3/bar4 andwedge1/4 action unitary; REMOVE ONLY in norm comparison, do not assign sigmaRef or alter actual F2')


def mean_source_coefficients(model: Any,data: dict[str,Any],f2: np.ndarray) -> dict[str,Any]:
    """Physical projected and BARE mean marks; NEVER assume raw positiveband."""
    raw=model.charged_coordinates@f2;hard=raw[:1680].reshape(2,4,2,7,15)
    point=model.chart.embedding(np.zeros(model.chart.normal_dim));fibre=model.source_bank.seed_fibre
    displacement=(model.law.sites-model.law.sites[model.reference_site])/model.law.size
    phases=np.concatenate((-displacement,displacement));count=len(phases)
    projected=np.zeros((count,fibre.charged_dimension,7,15),complex)
    bare_plus=np.zeros((model.law.volume,4,4,7,15),complex)
    bare_minus=np.zeros_like(bare_plus)
    for template,eta in enumerate(model.source_bank.profiles.values()):
        for word in range(4):
            source=fibre.source_vectors_at(point['phi'],point['links'],np.zeros(3),np.zeros((model.law.volume,3)),eta,np.eye(4)[word])
            S,T=source['full108_plus_coefficients'],source['full108_minus_coefficients']
            for site in range(model.law.volume):
                s=np.zeros_like(S);t=np.zeros_like(T);rows=slice(4*site,4*site+4);s[rows]=S[rows];t[:,rows]=T[:,rows]
                partial=Full108ReferenceMarks(s,t).physical_source(fibre)['source_matrix']
                for bank in range(2):projected[model.law.volume*bank+site]+=partial[:,bank,None,None]*hard[template,word,bank][None,:,:]
                bare_plus[site]+=S[rows][:,:,None,None]*hard[template,word,0][None,None,:,:]
                bare_minus[site]+=T[:,rows][:,:,None,None]*hard[template,word,1][None,None,:,:]
    normal=np.eye(7);normal[1:,1:]=model.original.directions.T@model.original.directions
    if np.linalg.norm(normal-np.eye(7))>1e-12:raise ValueError('Original normal-polynomial Gram must be owned, not renormalized')
    kernel=np.exp(-np.sum((phases[:,None]-phases[None,:])**2,axis=2)/4)
    mean_norm=0.
    for start in range(0,fibre.charged_dimension,24):
        c=projected[:,start:start+24];applied=(kernel@c.reshape(count,-1)).reshape(c.shape)
        mean_norm+=float(np.einsum('pfnk,nm,pfmk->',c.conj(),normal,applied,optimize=True).real)
    raw_norm=float(np.sum(abs(bare_plus)**2)+np.sum(abs(bare_minus)**2))
    # Floating packet(I) may round to1. True scalar differs by <=6e^-72
    # peredge. 1+1e-12 is a deliberately larger one-sided reserve.
    bare_upper=raw_norm*(1+1e-12)
    return dict(raw=raw,projected=projected,phases=phases,
        mean_projected_hard_norm_squared=mean_norm,mean_raw_HS_hard_norm_squared=raw_norm,
        mean_raw_HS_hard_norm_squared_upper=bare_upper,
        actual_raw_to_projected_CAR_squared_ratio=mean_norm/raw_norm,
        bare_equals_positiveband_NOT_assumed=True)


def original_extra_norm_bound(model: Any,data: dict[str,Any],raw: np.ndarray) -> dict[str,Any]:
    """Original gamma/commonodd/relativeodd tails with allPhi/52+harm3 graphs."""
    bank=model.noether_bank;coeff=bank.coefficients;gram=bank.scalar_gram()
    gamma=math.sqrt(float(gram[15,15].real))
    sine=np.sqrt(np.real(np.diag(gram[16:,16:])))
    rho=coeff['lambda_C']@bank._neutral_rho.T
    harmonic=np.linalg.norm(coeff['harmonic'],axis=1)/math.sqrt(2)
    displacement=(model.law.sites-model.law.sites[model.reference_site])/model.law.size
    relative=coeff['lambda_C']-coeff['lambda_C'][:,model.reference_site,None]-coeff['harmonic']@displacement.T
    prepared=[float(np.linalg.norm(data['preparation_state14'][7*b:7*b+7])) for b in range(2)]
    tail=raw[1680:].reshape(3,2,4,2);bounds=np.zeros(3)
    for template,eta in enumerate(model.source_bank.profiles.values()):
        norm=float(np.linalg.norm(eta));relative_norm=np.sqrt(np.sum(abs(eta)[None,:]**2*relative**2,axis=1))
        for word in range(4):
            for charge in range(2):
                bounds[0]+=abs(tail[0,template,word,charge])*norm*prepared[charge]*gamma
                weak=float(sine@(abs(rho[:,charge])+harmonic))
                bounds[1]+=abs(tail[1,template,word,charge])*norm*prepared[charge]*weak
                bounds[2]+=abs(tail[2,template,word,charge])*prepared[charge]*float(sine@relative_norm)
    return dict(original_even_gamma_norm_bound=float(bounds[0]),original_commonodd_density_harmonic_norm_bound=float(bounds[1]),
        original_relativeodd_norm_bound=float(bounds[2]),total_norm_bound=float(bounds.sum()),
        original_tail_coefficient_norm=float(np.linalg.norm(raw[1680:])),
        original_gamma_scalar_exact_norm=gamma,all_original_extras_retained=True,
        harmonic_gaussian_and_frozen_true_neutral_density_retained=True,
        common_SOURCE_raw_operator_bound='unitary birth/ref, |originalcentre|<=1; labelwordHS2/2=1; normeta and originalprepared14 branch norm. Original scalar gamma/sin exact52 Gram andtheta L2<=||h||/sqrt2. No SOURCE globalZ redefinition.')


def certificate(model: Any,data: dict[str,Any],f2: np.ndarray) -> dict[str,Any]:
    constants=compact_frame_constants(model);mean=mean_source_coefficients(model,data,f2)
    raw=mean['mean_raw_HS_hard_norm_squared_upper'];sigma=model.sigma;D=model.chart.normal_dim
    radial=2*constants['core_radius']**2/sigma**2
    if radial<=D:raise ValueError('Actual cutoff-one core cannot pay Gaussian normal tail')
    log_tail=-.5*(radial-D-D*math.log(radial/D))
    tail=raw*math.sqrt(61)*math.exp(log_tail/2)
    # Mass-normal contraction: local Phi raw map T has TrT<=60, ||T||<=1.
    # Wick fourth moment =sigma² TrT/4 R² +sigma²/2 d^T T d <=15.5sigma² R².
    birth=constants['birth_frame_Lipschitz_per_Phi_Frobenius']*sigma*math.sqrt(15.5*raw)
    scalar=constants['original_whole_centre_path_error']*math.sqrt(raw)
    extras=original_extra_norm_bound(model,data,mean['raw'])
    # Standard gamma_n accounts for arithmetic accumulation, NOT LAPACK
    # eigh/SVD or a rigorous enclosure of the stored mean projection. The
    # admission owner independently encloses the analytic mean without SVD.
    # These numerical candidates change NO source/state/receiver/metric.
    operations=2**30;unit_roundoff=2.**-52
    gamma=operations*unit_roundoff/(1-operations*unit_roundoff)
    rounding=gamma*2*model.law.volume*(1+raw)
    mean_lower=mean['mean_projected_hard_norm_squared']-rounding
    lower=math.sqrt(mean_lower-tail)-birth-scalar-extras['total_norm_bound']-rounding
    upper=math.sqrt(raw)+extras['total_norm_bound']+rounding
    output={key:value for key,value in mean.items() if key not in ('raw','projected','phases')}
    return dict(**output,analytic_constants=constants,original_extra_bounds=extras,
        Gaussian_cutoff_one_core_tail_log_upper=log_tail,mean_hard_norm_squared_outside_core_upper=tail,
        ACTUAL_birth_frame_hard_image_difference_norm_upper=birth,
        ACTUAL_centre_path_hard_image_difference_norm_upper=scalar,outward_numerical_norm_reserve=rounding,
        numerical_evaluation=dict(overcounted_arithmetic_operations=operations,unit_roundoff_upper=unit_roundoff,
            gamma_n=gamma,absolute_mean_quadratic_sum_upper=2*model.law.volume*raw,
            LAPACK_mean_projection_error_NOT_certified_by_gamma_n=True,
            independent_analytic_mean_interval_admission_REQUIRED=True),
        analytic_candidate_actual_compact_F2_norm_lower=lower,
        analytic_candidate_actual_compact_F2_norm_squared_lower=lower**2,
        analytic_candidate_actual_compact_F2_norm_upper=upper,
        analytic_candidate_actual_compact_F2_effect_operator_norm_upper=upper**2,
        analytic_candidate_rejects_native_rankone_E2=bool(lower>1),
        native_rankone_E2_NOT_POVM=None,
        original_F2_HARD_plus_ALLextras_not_renormalized_fitted_or_replaced=True,
        original_density_cancels_once_all52_allharm3_AND_actual_compact_cutoff=True,
        microscopic_effect_transfer_M_requires_independent_interval_admission=True,
        native_factory_scope='N3 cachedfieldfactory/sigma2^-13/radius.012/fixedseed108/Ref4 originalsource19/SOFT; not whole LEGAL E0kappa/all64 union/scattering disproof')
