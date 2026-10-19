#!/usr/bin/env python3
"""Full Cartesian outside angular smoothing with explicit Root parameter epochs.

Domain/proof (no T/Q graph split, no chart angular bound on the outside):
H=T+Q+V+Eaux*n+MF on the Friedrichs joint Dirichlet complement,
T=-Delta/2 in all D=60V Cartesian coordinates, Q=sum_link(-Delta_SU4),
V=Lambda sum_site r_site**4 >= lambda_eff*r**4, ||MF||<=L*r.
All inequalities below are uniform in the SAME compact U1 Wilson angles.

Use bounded smooth nondecreasing concave f_R(r**2) increasing to r**2,
0<=f_R<=r**2, 0<=f_R'<=1, f_R''<=0. Multiplication by f_R preserves
the joint Dirichlet/Gauss form domain, even at reentrant corners, and
Delta f_R<=2D. Weak testing yields the limiting weighted identity
Re<H psi,r**2 psi> = ||r grad psi||**2/2-D||psi||**2/2
+ integral r**2[V+Eaux*n+MF]|psi|**2+ integral r**2 |grad_colour psi|**2.
No H2 regularity, separate Cartesian/SU4 domains or TQ commutation.
On the negative-potential region, f_R<=r**2 gives
f_R(V-Lr)>=-L**2/(4*lambda_eff); elsewhere it is nonnegative.
Also ||f_R psi||<=||r**2 psi||, already finite by the ordinary form
lower bound below. Monotone convergence yields
||r grad psi||**2 <= 2||Hpsi||||r**2 psi||
+(D+L**2/(2*lambda_eff))||psi||**2. Separately, absorbing L*r
against V/2 gives ||r**2 psi||**2 <= 2/lambda_eff
(<psi,Hpsi>+s||psi||**2), s=3L**(4/3)/(4(2lambda_eff)**(1/3)).
Eaux*n stays positive in BOTH identities, without bounding its condition
number, commuting it through MF, or discarding ANY finite Fock sector.

Actual global L_index=-i sum_site(Phi_2 partial_Phi_3-Phi_3 partial_Phi_2)
has pointwise velocity <=r. Its world action fixes physical links and ALL
primitive matter/Ref components; it commutes local Gauss. In a chart,
Vframe'=Vframe Uspin, Y'=Uspin^dag Lambda Y Uspin and
X'=Uspin^dag X Uspin. Normal and colour RADII are invariant, centres fixed.
Thus joint chart/complement, zero trace, and its corners are invariant.
The chart Fock connection is NOT omitted: it is a coordinate rewrite of
this world action, not an additional free spin drive. A deliberately
conservative bounded connection ledger is reported separately.

For the bare H>=E>2, heat psi=e^-H/2 f has ||psi||<=e^-E/2,
||Hpsi||<=E e^-E/2 and <Hpsi,psi><=E e^-E. Polynomial weighted-form
prefactors use this BARE E; they are not computed by an arbitrary graph shift.
Actual exitedPhi OR colour carries exp(-Delta) at the endpoints: q is at most
exp(-E-Delta+epsilon), and the heat graph bound gains exp(-Delta/2).
With B=e^-H/2 P_Haar, polar BBdag gives ||L sqrt(Tout)||<=||LB||.
Full Gauss contracts because the WORLD action commutes Gauss, not via trace.

Hout need not commute L. Outside-to-PE uses X=Pout PE and
X=-sum_(n>=0) Tout**n F Bband**(-n-1), F=Pout deltaK PE.
For b=e^-.25 lambda0, q=||Tout||<b, M=||L sqrt(Tout)||,
||LX||<=||LF||/b+M sqrt(q)||F||/[b(b-q)]. This is an ordered
Sylvester/Neumann graph estimate, NOT multiplication of the chart Sylvester
identity by a noncommuting outside angular operator.

These are OPERATOR/Pe bounds, not proof that an actual normalized Pe source
exists. Field withdrew its .894663/.5 scalar/reference Gram claim:
Gaussian/action comparison cannot prove a microscopic inserted-Gauss/source
Gram, and NF*kappa**4 variance does not supply a conditional Fourier/minor
ratio. The actual macro Pe=.25 numerator has no nonzero lower bound.
Field's signed native U1-Gauss kappa.09 source is positive at Pe6, but its Pe=.25 numerator
is ZERO. No native retag, zero-source normalization or unconditional source
admission is made here; superseded executions are failed-claim audits only.
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
import time
from typing import Any

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import mpmath as mp
import numpy as np
import scipy
from scipy.linalg import eigh, expm

ROOT = Path(__file__).resolve().parent
FROZEN_ROOT_SHA = 'bd4d5e1d0e9fcfadef72cc9dba2cc4f7e6efacc6a577ee331fcd5b82574e5e49'
PLAN = 'issue-222-induced-photon-delta13-continuation-20261008'
REVISION = 'a7f0d27e781ceb4f644fb6a3a24ccd597441268e6a4bd1b32887dee65ec6a710'
SOURCE = Path(__file__).read_bytes()
LOADED_CODE_MATCH = sys._getframe().f_code == compile(SOURCE, __file__, 'exec',
                                                    dont_inherit=True, optimize=sys.flags.optimize)
if not LOADED_CODE_MATCH:
    raise ArithmeticError('Outside angular loaded code differs from captured source bytes')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArithmeticError(message)


def scalar_heat_prefactor(dimension: Any, lam: Any, load: Any, energy: Any) -> tuple[Any, Any, Any]:
    """Return polynomial Cartesian, colour and weighted-form shift constants."""
    shift = mp.mpf(3)/4*load**(mp.mpf(4)/3)/(2*lam)**(mp.mpf(1)/3)
    radial_fourth = mp.sqrt(2*(energy+shift)/lam)
    gradient_squared = 2*energy*radial_fourth+dimension+load**2/(2*lam)
    return mp.sqrt(gradient_squared), mp.sqrt(energy+shift), shift


def outside_thermal_budget(volume: int, naux: int, cap: int, trial: int,
                           delta: int, energy: Any, slices: int) -> dict[str, Any]:
    """Full primitive/SU4/Haar Schatten trace and nonzero-cross normalization."""
    primitive_edges = (7*(naux+1)-1)*volume+4
    dimension = 60*volume
    # The oscillator denominator is <=2**D since p*omega>log2.
    # All centre copies are already in the full SU4 heat trace, not selected.
    entropy = mp.mpf(primitive_edges+dimension)*mp.log(2)+3*volume*mp.log(mp.mpf('1.0001'))
    # Unit-time HS trace pays the full finite primitive Fock and ALL centre
    # copies. Raising this trace to p is coarse but valid even with field/A
    # dependence; no false tensor Schatten factorization is used.
    statistical_comparison_entropy = mp.mpf(primitive_edges+90*volume+2*(3*volume+3))*mp.log(2)
    epsilon = mp.mpf(1)/16
    residual = delta-cap-trial
    rows = []
    for p in (slices, slices+1):
        # All huge cancellation in exact INTEGERS before conversion to mp.
        bound = entropy-p*(energy+mp.mpf(residual)-epsilon)
        require(bound < -1, 'Full endpoint/Haar thermal entropy budget fails')
        statistical_bound = p*(mp.mpf(2*cap+trial-delta)+statistical_comparison_entropy+epsilon)
        require(statistical_bound < -1, 'Full nonflat-statistical comparison trace budget fails')
        rows.append(dict(power=p,
            normalized_ideal_outside_trace_ratio_log10_upper=str(mp.nstr(bound/mp.log(10),40)),
            nonflat_statistical_diagonal_comparison_trace_ratio_log10_upper=str(mp.nstr(statistical_bound/mp.log(10),40))))
    cross_relative_log = (-mp.mpf(delta)/2+mp.log(epsilon)+3*epsilon)/mp.log(10)
    # Schatten telescoping / reverse triangle, with (1+z)^p<=exp(pz).
    # z<=1/(4p) implies normalized density difference <=8p z.
    require(cross_relative_log < -mp.log10(16*(slices+1)**2), 'Cross relative Schatten/purifier budget is not small')
    cross_density_log = mp.log10(8*slices)+cross_relative_log
    cross_purifier_log = mp.log10(8)+cross_relative_log/2
    require(mp.log(180*volume) < slices+1, 'Radial product ground Gram lacks a positive lower bound')
    return dict(full_primitive_edges_exact=str(primitive_edges),
        full_primitive_entropy_log10=str(mp.nstr(mp.log10(mp.mpf(primitive_edges)*mp.log(2)),40)),
        endpoint_residual_exact=str(residual),thermal_rows=rows,
        statistical_comparison_proof='||exp(epsilon Dstat)||<=exp(epsilon) gives a genuine operator form comparison after the SAME positive congruence: K<=exp(epsilon)Kdiag. It is NOT scalar RN for sources. Nonflat-statistical Kdiag has exp(-Delta) endpoint. Unit-time heat/Haar HS trace<=2^E*2^(90V)*4^(3V+3)*exp(cap); this includes every finite primitive occupation/cycle and overcounts all centre copies. Tr A^p<=(Tr A)^p, and actual neutral trial lambda>=exp(-cap-trial), give the displayed comparison-trace ratio. This does NOT identify Tr(Pnonflat K^p), factor statistical fibres, or permit removing Dstat from exact source/path kernels.',
        radial_ground_projector='Continuous radial grounds tensor IDENTITY on ALL centres and flat/nonflat statistical sectors; ZERO on outside.',
        radial_preparation_Gram_lower=0.25,
        radial_preparation_Gram_proof='gphi,p>=(1-90V exp(-p))/(1+rout,p)>1/4 for p=Nt+1. This is ONLY the radial preparation filter, NOT a charged source/photon/toron Gram.',
        damped_cross_relative_Schatten_log10_upper=str(mp.nstr(cross_relative_log,40)),
        damped_cross_normalized_density_error_log10_upper=str(mp.nstr(cross_density_log,40)),
        damped_cross_boundary_purifier_error_log10_upper=str(mp.nstr(cross_purifier_log,40)),
        cross_proof='X0=epsilon Dstat, X=X0+epsilon Dbar. Duhamel ||expX-expX0||<=epsilon exp(2epsilon)exp(-Delta/2); expX0>=exp(-epsilon)I gives RELATIVE form z=epsilon exp(3epsilon)exp(-Delta/2). Congruence preserves this on the actual common Haar/Gauss range. Holder gives ||K-Kideal||p<=z||Kideal||p. Power telescope/reverse Schatten triangle give normalized density error<=8p z, including ALL sectors. BKS square-root plus Holder for Xi=Khalf Pphi K^m and gphi>1/4 gives purifier difference<=8sqrt(z) in the stated tiny regime. This is an operator relative/Schatten theorem, NOT scalar RN or trace-only angular inference.',
        microscopic_source_Gram_lower=None,photon_Gram_lower=None,
        source_admitted=False)


def outside_angular_bound(root_receipt: dict[str, Any], source_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full outside/PE operator theorem; microscopic source normalization UNPROVED."""
    with mp.workdps(100):
        root_sha = root_receipt['capsule']['source_sha256']
        inputs = root_receipt['parameters']
        canonical = json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()
        require(hashlib.sha256(canonical).hexdigest() == root_receipt['capsule']['input_sha256'],
                'Root parameter/input hash mismatch')
        family = root_receipt['renewed_family_bounds']
        photon = root_receipt['parameters']['photon']
        root = root_receipt['parameters']['root']
        volume = int(photon['spatial_size'])**3
        dimension = 60*volume
        naux = int(family['auxiliary_count'])
        require(naux == 10**(int(photon['beta_power'])+4*int(photon['kappa_power']))//16,
                'Root auxiliary count does not match explicit configuration')
        lam = mp.power(10, 4000)/volume
        load = 64*volume*(naux*mp.power(10, -int(photon['kappa_power']))+1)
        energy_unshifted = (mp.mpf(3)/(4*mp.power(2, mp.mpf(2)/3))
                            *mp.mpf(dimension)**(mp.mpf(4)/3)*(lam/2)**(mp.mpf(1)/3))
        shift = mp.mpf(3)/4*load**(mp.mpf(4)/3)/(2*lam)**(mp.mpf(1)/3)
        # Outward lower, not log(Eunshifted) renamed log(Eactual).
        require(shift < energy_unshifted/2, 'Actual full-Fock linear load destroys the outside gap')
        energy = energy_unshifted/2
        polynomial, colour_root, _ = scalar_heat_prefactor(dimension, lam, load, energy)
        colour_polynomial = 2*mp.sqrt(3*volume)*colour_root
        connection = volume*(naux+1)+1
        total_polynomial = polynomial+colour_polynomial+connection
        cap = int(family['full_B_F_log_transfer_norm_cap_exact'])
        trial = int(family['actual_full_chart_field_trial_upper'])
        delta = int(family['renewed_cycle_penalty_exact'])
        require(delta == 4*cap+4*trial+256, 'Exit penalty does not match selected Root ABI')
        require(int(root['normal_width_power']) > 0 and int(root['tube_radius_power']) > 0,
                'Root widths must be explicit positive powers')
        budget = root_receipt['actual_weighted_Phi_coupling']
        chart_root_log = max(budget['projective_step_colour_log10'],
                             budget['projective_step_normal_log10'])
        gap = math.exp(-.25)-math.exp(-1)
        epsilon = mp.mpf(1)/16
        # Cross norm /lambda0: e^(2cap+trial-Delta/2) epsilon e^(2epsilon).
        # Compute cancellation in INTEGERS before arbitrary precision conversion.
        cross_exponent_twice = 4*cap+2*trial-delta
        cross_log = (mp.mpf(cross_exponent_twice)/2+2*epsilon+mp.log(epsilon))/mp.log(10)
        chart_density_log = chart_root_log+math.log10(6)
        chart_Ldelta_log = chart_density_log+math.log10(4/math.e)
        chart_projector_log = chart_Ldelta_log-math.log10(gap)
        # Physical endpoint is decisive when bare E fails to dominate the cap.
        require(energy+mp.mpf(delta-4*(cap+trial+1)) > 0,
                'Actual outside endpoint fails neutral lambda0 separation')
        thermal = outside_thermal_budget(volume,naux,cap,trial,delta,energy,int(photon['time_slices']))
        require(mp.mpf(thermal['damped_cross_relative_Schatten_log10_upper'])+mp.log10(2)<chart_root_log,
                'Full cross perturbation exceeds the renewed chart/BKS budget')
        thermal['full_prepared_boundary_purifier_error_log10_upper'] = math.log10(32)+chart_root_log/2
        thermal['full_prepared_boundary_density_trace_error_log10_upper'] = math.log10(64)+chart_root_log/2
        thermal['full_normalized_thermal_density_error_log10_upper'] = math.log10(32*int(photon['time_slices']))+chart_root_log
        thermal['full_preparation_scope'] = 'Radial Xi only, identity on ALL centres/statistical sectors. Joint BKS uses epsilon_total<=8max(chart_colour,chart_normal), gphi>=1/4 and verified small p sqrt(epsilon). Charged halfsteps/source/photon clipping still require their OWN actual Gram; no admission.'
        # Cross LF/lambda0 <= e^-Delta Mout e^cap/2 *epsilon e^2eps /lambda0.
        # Endpoint outside factor gives e^-Delta/2; Dbar already e^-Delta/2.
        outside_cross_log = ((-energy/2 + mp.mpf(3*cap+2*trial-2*delta)/2
                              +2*epsilon+mp.log(epsilon)+mp.log(polynomial))/mp.log(10))
        # No commutation needed: n>=1 series <=2 Mout sqrt(q) ||F|| /b².
        outside_series_log = ((-energy + mp.mpf(2*cap+2*trial-delta)
                               +epsilon+mp.log(polynomial)+mp.log(2)+mp.mpf('.5'))/mp.log(10)+cross_log)
        outside_projector_log = max(outside_cross_log+mp.mpf('.25')/mp.log(10), outside_series_log)+mp.log10(2)
        # For whole-source angular norm, add chart cross perturbation to chart relative bound.
        cross_chart_Ldelta_log = cross_log+mp.log10(4/mp.e)
        total_chart_Ldelta_log = max(mp.mpf(chart_Ldelta_log),cross_chart_Ldelta_log)+mp.log10(2)
        total_projector_log = max(total_chart_Ldelta_log-mp.log10(gap),outside_projector_log)+mp.log10(2)
        source = source_receipt.get('macro', source_receipt) if source_receipt is not None else None
        source_metadata: dict[str, Any] = {}
        source_debt = dict(microscopic_post_insertion_PE_Gram_lower=None,
            nonzero_actual_PE_numerator=None,
            rejected_inference='Scalar Gaussian/action/reference comparison is NOT a microscopic inserted-Gauss/source Gram theorem; NF*kappa^4 variance is NOT a conditional Fourier/minor ratio.',
            owner_reported_native_kappa09_PE025_Gram=0.,
            native_scope='Field final signed native U1-Gauss ONLY: Pe6 Gram8.60e-5/Hmean5.0977, Pe=.25 numerator ZERO. Not macro SU4 and not a quarter-cut retag.',
            energy_support_statement='IF normalized Pe source exists, H<=.25 and highE=0. Nonzero numerator/normalization is not established.')
        if source is not None:
            exact = source['exact_energy_preparation']
            require(source['full_SU4_normal_Root']['source_sha256'] == root_sha,
                    'Actual retained source uses a different Root')
            require(int(source['auxiliary_count_exact']) == naux,
                    'Actual source does not retain the selected full finite Fock family')
            require(exact['common_neutral_lambda0'] is True and exact['energy_cut'] == .25,
                    'Actual retained source does not use SAME neutral lambda0 / PE=.25')
            require(source['full_SU4_normal_Root']['normal_width_parameter'] == f"1e-{root['normal_width_power']}"
                    and source['full_SU4_normal_Root']['tube_radius'] == f"1e-{root['tube_radius_power']}",
                    'Source width/radius does not match selected Root configuration')
            if root_sha != FROZEN_ROOT_SHA:
                require(source['full_SU4_normal_Root'].get('root_input_sha256') == root_receipt['capsule']['input_sha256'],
                        'New source has no matching exact Root input/configuration hash')
            source_metadata = dict(energy_cut=.25,common_neutral_lambda0=True,
                state_status='Conditional spectral preparation ONLY; actual microscopic Pe source Gram unproved.',
                conditional_high_energy_expectation_if_normalized=0.,
                actual_source_Gram_lower=None,
                authoritative_producer_admission=source.get('admission'),
                normalization='NOT ADMITTED: supplied reference/scalar Gram is not a microscopic Pe numerator lower bound.')
        def text(value: Any) -> str:
            return str(mp.nstr(value, 40))
        return dict(root_sha256=root_sha,root_input_sha256=root_receipt['capsule']['input_sha256'],
            configuration_ABI=root_receipt.get('configuration_ABI',dict(parameters=inputs)),
            full_dimension_exact=str(dimension),volume_exact=str(volume),
            normalized_full_sector_thermal_transfer=thermal,
            adverse_old_bare_gap_dominance=dict(holds=bool(energy>4*mp.mpf(cap+trial+1)),
                bare_energy_log10=text(mp.log10(energy)),full_cap_trial_log10=text(mp.log10(cap+trial)),
                status='Not used: actual declared endpoint exp(-Delta) repairs the kappa400 failure.'),
            lambda_eff_log10=text(mp.log10(lam)),linear_full_Fock_load_log10=text(mp.log10(load)),
            positive_auxiliary_energy_retained=True,all_compact_A_uniform=True,
            outward_outside_energy_lower_log10=text(mp.log10(energy)),
            linear_absorption_shift_log10=text(mp.log10(shift)),
            Cartesian_Lindex_heat_polynomial_prefactor_log10=text(mp.log10(polynomial)),
            colour_derivative_heat_polynomial_prefactor_log10=text(mp.log10(colour_polynomial)),
            full_Fock_connection_cap_log10=text(mp.log10(connection)),
            conservative_combined_heat_polynomial_prefactor_log10=text(mp.log10(total_polynomial)),
            outside_Lindex_bare_heat_norm_log10_upper=text(mp.log10(polynomial)-energy/(2*mp.log(10))),
            outside_Lindex_Tout_half_norm_log10_upper=text(mp.log10(polynomial)+(-energy-mp.mpf(delta)+epsilon)/(2*mp.log(10))),
            outside_Lindex_norm_proof='Weighted Friedrichs form and polar BBdag, no chart4/e, no TQ graph split, no entropy argument.',
            outside_gap_ratio_to_neutral_lambda0_log10_upper=text((-energy+mp.mpf(cap+trial-delta)+epsilon)/mp.log(10)),
            common_neutral_lambda0_lower_exact_exponent=str(-cap-trial),
            exit_Delta_exact=str(delta),inside_exponential_damping=True,
            crossing_deltaK_over_lambda0_log10_upper=text(cross_log),
            chart_relative_density_deltaK_over_lambda0_log10_upper=chart_density_log,
            chart_LdeltaK_over_lambda0_log10_upper=text(total_chart_Ldelta_log),
            chart_LdeltaK_proof='Use the renewed species-resolved V² coefficient sum, then Kchart=Nhalf R Kf Rdag Nhalf and explicit ||Lscalar Nhalf||<=4/e. Acting on the LEFT heat factor bounds the full graph difference, including coefficient dependence; this is NOT trace/norm proximity promoted to an unbounded-L estimate. Pphi annihilates scalar L and retains IDENTITY on every centre/statistical sector.',
            outside_LdeltaK_over_lambda0_log10_upper=text(outside_cross_log),
            outside_PE_Lindex_norm_log10_upper=text(outside_projector_log),
            whole_PE_Lscalar_norm_log10_upper=text(total_projector_log),
            scalar_operator_definition='Directsum chart covariant radial normal/colour derivative and outside world Phi-index derivative; chart Fock connection is separately paid, NOT included in this tiny scalar norm.',
            whole_PE_global_Lindex_norm_log10_upper=text(mp.log10(connection+mp.power(10,total_projector_log))),
            global_drive_definition='WORLD Lindex, hence chart scalar PLUS allmatter/Ref connection. Common-arm connection is bounded, not claimed tiny.',
            ordered_Sylvester_cutoff=.25,ordered_Sylvester_chart_gap=gap,
            actual_high_energy_weight_upper=None,
            outside_ordered_Sylvester_noncommuting_proof='X=-sum Tout^n F Bband^-n-1; ||LX||<=||LF||/b+M sqrt(q)||F||/[b(b-q)].',
            retained_source_bound_admitted=False,actual_retained_source=source_metadata,
            microscopic_source_admission_debt=source_debt,
            source_bound_scope='Full outside/Pe OPERATOR bound. Source consequence is conditional on a genuinely nonzero microscopic Pe numerator; no unconditional source/ground/sharp-pole/QED claim.')


def su4_basis() -> np.ndarray:
    """Tr(Ga Gb)/4=delta_ab, hence fundamental Casimir15."""
    basis = []
    for i in range(4):
        for j in range(i+1,4):
            a=np.zeros((4,4),complex);a[i,j]=a[j,i]=math.sqrt(2);basis.append(a)
            a=np.zeros((4,4),complex);a[i,j]=-1j*math.sqrt(2);a[j,i]=1j*math.sqrt(2);basis.append(a)
    for k in range(1,4):
        a=np.zeros((4,4),complex);a[np.arange(k),np.arange(k)]=1;a[k,k]=-k
        basis.append(a*math.sqrt(4/(k*(k+1))))
    return np.array(basis)


def native_full60_boundary(axis: int = 2) -> dict[str, Any]:
    """Actual 60 Cartesian components, all15 SU4 tangents, radial joint boundary."""
    sigma = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
    gamma = np.array([np.diag([1,1,-1,-1])]+[np.block([[np.zeros((2,2)),s],[s,np.zeros((2,2))]]) for s in sigma])
    phi = gamma*np.sqrt(np.array([.4187253908699568]+[.0632221974274214]*3))[:,None,None]
    basis = su4_basis()
    def real(x: np.ndarray) -> np.ndarray:
        return np.r_[np.real(x).reshape(-1),np.imag(x).reshape(-1)]
    cartesian=[]
    for mu in range(4):
        for g in basis:
            a=np.zeros_like(phi);a[mu]=g;cartesian.append(real(a))
    frame,_=np.linalg.qr(np.column_stack(cartesian))
    tangents=np.column_stack([real(1j*(g@phi-phi@g)) for g in basis])
    u,_,_=np.linalg.svd(frame.T@tangents,full_matrices=True)
    normal_frame=frame@u[:,15:]
    j,k=(axis+1)%3+1,(axis+2)%3+1
    theta=.37;spatial=np.eye(4)
    spatial[j,j]=spatial[k,k]=math.cos(theta)
    spatial[j,k]=-math.sin(theta);spatial[k,j]=math.sin(theta)
    spin=np.block([[sigma[axis]/2,np.zeros((2,2))],[np.zeros((2,2)),sigma[axis]/2]])
    U=expm(1j*theta*spin)
    rng=np.random.default_rng(222108);v=normal_frame@rng.normal(size=45)
    Y=(v[:64]+1j*v[64:]).reshape(4,4,4);Y/=np.linalg.norm(Y)
    Ynew=U.conj().T@np.einsum('jk,kab->jab',spatial,Y)@U
    X=sum((rng.normal()*g for g in basis),np.zeros((4,4),complex));X/=np.linalg.norm(X)
    Xnew=U.conj().T@X@U
    physical_old=phi+Y*.017
    physical_new=np.einsum('jk,kab->jab',spatial,physical_old)
    chart_reconstructed=U@(phi+Ynew*.017)@U.conj().T
    generator=np.zeros((4,4));generator[j,k]=-1;generator[k,j]=1
    world=np.einsum('jk,kab->jab',generator,Y)
    gauss_error=max(np.linalg.norm(np.einsum('jk,kab->jab',generator,1j*(g@Y-Y@g))-1j*(g@world-world@g)) for g in basis)
    dY=np.einsum('jk,kab->jab',generator,Y)-1j*(spin@Y-Y@spin)
    dX=-1j*(spin@X-X@spin)
    residual=np.linalg.norm(real(Ynew)-normal_frame@(normal_frame.T@real(Ynew)))
    result: dict[str, Any]=dict(axis=axis,full_cartesian_dimension=60,normal_dimension=45,all15_Gauss_commutator_error=float(gauss_error),
        normal_slice_invariance_error=float(residual),world_to_chart_reconstruction_error=float(np.linalg.norm(physical_new-chart_reconstructed)),
        normal_boundary_tangency_error=float(abs(np.vdot(Y,dY).real)),colour_boundary_tangency_error=float(abs(np.vdot(X,dX).real)),
        normal_radius_invariance_error=float(abs(np.linalg.norm(Ynew)-np.linalg.norm(Y))),
        colour_radius_invariance_error=float(abs(np.linalg.norm(Xnew)-np.linalg.norm(X))),
        centre_fixed_error=max(float(np.linalg.norm(U.conj().T@(z*np.eye(4))@U-z*np.eye(4))) for z in (1,1j,-1,-1j)),
        wrong_component_box_rotation_change=float(abs(np.max(abs(Ynew))-np.max(abs(Y)))),
        physical_links_fixed=True,primitive_matter_Ref_fixed=True,
        domain='Same joint radial chart complement; tangent BOTH normal and colour faces and their intersections. No component-box substitution.')
    require(max(result[k] for k in ('all15_Gauss_commutator_error','normal_slice_invariance_error','world_to_chart_reconstruction_error','normal_boundary_tangency_error','colour_boundary_tangency_error','centre_fixed_error'))<1e-10,
            'Actual full60 rotation does not preserve joint chart domain/Gauss')
    require(result['wrong_component_box_rotation_change']>1e-5,'Wrong component-box boundary control did not discriminate')
    return result


def native_outside_heat(radial_size: int = 22, angular_cut: int = 5) -> dict[str, Any]:
    """Full60 radial/zonal Galerkin, genuine Dirichlet exterior subdomain.

All16 occupations of four finite fermions and coefficient1 SU4 Casimirs
0,15,36 are evaluated, not a chart oscillator. The exterior radial subdomain
is a rotation-invariant part of the joint complement, NOT its replacement.
Spherical harmonics of S59 and radial Jacobian give the actual 60D kinetic.
L12 leaves the zonal sector; its exact derivative Gram l(l+58)/59 is used,
not the (zero) compressed angular generator. Finite cutoffs are falsifiers;
the untruncated bound is the weak-form proof above.
"""
    D=60;lam=.002;load=.15;R=1.2;outer=6.;dr=(outer-R)/(radial_size+1)
    r=R+dr*np.arange(1,radial_size+1)
    kinetic=(2*np.eye(radial_size)-np.diag(np.ones(radial_size-1),1)-np.diag(np.ones(radial_size-1),-1))/(2*dr**2)
    levels=np.arange(angular_cut+1);angular_size=len(levels)
    multiplication=np.zeros((angular_size,angular_size))
    for l in range(angular_cut):
        a=math.sqrt((l+1)*(l+D-2)/((2*l+D-2)*(2*l+D)))
        multiplication[l,l+1]=multiplication[l+1,l]=a
    radial=np.kron(np.eye(angular_size),np.diag(r))
    linear=np.kron(multiplication,np.diag(r))
    angular_gram=np.kron(np.diag(levels*(levels+D-2)/(D-1)),np.eye(radial_size))
    base=np.kron(np.eye(angular_size),kinetic+np.diag(lam*r**4))
    for l in levels:
        centrifugal=((D-1)*(D-3)/4+l*(l+D-2))/(2*r*r)
        sl=slice(l*radial_size,(l+1)*radial_size);base[sl,sl]+=np.diag(centrifugal)
    rows=[]
    max_moment=0.;uniform_bound=0.
    # Actual dGamma diag(-1,-1,+1,+1)/4. All16 Fock sectors retained.
    for phase in (0.,.43):
        for mask in range(16):
            occupied=[j for j in range(4) if mask&(1<<j)]
            matter=sum((-1 if j<2 else 1)/4 for j in occupied)*load*math.cos(phase)
            positive_aux=.7*len(occupied)
            for casimir in (0.,15.,36.):
                H=base+matter*linear+(positive_aux+casimir)*np.eye(base.shape[0])
                e,U=eigh(H);half=(U*np.exp(-e/2))@U.T
                moment=math.sqrt(max(0.,float(eigh(half@angular_gram@half,eigvals_only=True)[-1])))
                with mp.workdps(40):
                    pref,_,_=scalar_heat_prefactor(D,lam,load,float(e[0]))
                    bound=float(pref*mp.exp(-float(e[0])/2))
                require(moment<=bound*(1+1e-10),'Native Hout angular moment violates full Cartesian form estimate')
                max_moment=max(max_moment,moment);uniform_bound=max(uniform_bound,bound)
                rows.append(dict(compact_A=phase,Fock_occupation=mask,positive_Eaux_n=positive_aux,
                                 coefficient1_SU4_Casimir=casimir,actual_Hout_min=float(e[0]),
                                 actual_L12_heat_norm=moment,weighted_form_upper=bound))
    haar_rows=[]
    # Literal cycle Fourier-zero Haar, discretized uniformly; angular derivative
    # evaluated on sqrt(BBdag), not just on a fixed-A exponential.
    indices=np.array([l*radial_size+j for l in range(3) for j in range(min(10,radial_size))])
    hbase=base[np.ix_(indices,indices)];hlinear=linear[np.ix_(indices,indices)]
    gram=angular_gram[np.ix_(indices,indices)]
    for angles in (4,8):
        blocks=[]
        for A in 2*math.pi*np.arange(angles)/angles:
            H=hbase-load*math.cos(A)/4*hlinear+.7*np.eye(len(indices))
            e,U=eigh(H);blocks.append((U*np.exp(-e/2))@U.T/math.sqrt(angles))
        B=np.vstack(blocks)
        left,singular,_=np.linalg.svd(B,full_matrices=False)
        actual_Tout_half=(left*singular)@left.T
        derivative_gram_half=np.kron(np.eye(angles),np.diag(np.sqrt(np.diag(gram))))
        actual_moment=float(np.linalg.norm(derivative_gram_half@actual_Tout_half,2))
        polar_moment=float(np.linalg.norm(derivative_gram_half@B,2))
        require(abs(actual_moment-polar_moment)<=1e-10*polar_moment,
                'Actual Haar Tout square root angular norm violates the polar-factor identity')
        haar_rows.append(dict(cycle_angles=angles,actual_Haar_Tout_half_Lnorm=actual_moment,
                              actual_heat_Haar_factor_Lnorm=polar_moment,
                              retained_Haar_rank=int(np.count_nonzero(singular>1e-30))))
    # Norm proximity is not angular graph proximity: exact high angular modes.
    proximity=[]
    for m in (10,10000,10000000):
        delta=np.array([[0.,0.],[1e-6,0.]])
        L=np.diag([0.,float(m)])
        proximity.append(dict(m=m,norm_error=float(np.linalg.norm(delta,2)),
                              angular_derivative_error=float(np.linalg.norm(L@delta,2))))
    noquartic=[]
    for m in (10,100,1000):
        radius=float(m*m)
        centrifugal=((D-1)*(D-3)/4+m*(m+D-2))/(2*radius**2)
        trial_energy_upper=math.pi**2/2+centrifugal
        noquartic.append(dict(m=m,radial_support=[radius,radius+1],
            wrong_noquartic_trial_energy_upper=trial_energy_upper,
            wrong_noquartic_Lheat_norm_lower=m*math.exp(-trial_energy_upper/2),
            actual_quartic_trial_potential_lower=lam*radius**4))
    require(noquartic[-1]['wrong_noquartic_Lheat_norm_lower']>90*noquartic[0]['wrong_noquartic_Lheat_norm_lower'],
            'Removing quartic did not expose unbounded actual60D heat angular moment')
    # Actual nonzero off-block coupling; endpoint-only damping keeps chart loops.
    pin=np.diag([1.,0.]);pout=np.eye(2)-pin;cross=np.array([[0.,1.],[1.,0.]])
    penalty=pin+math.exp(-4/2)*pout
    wrong=pin@(expm(.07*cross)-np.eye(2))@pin
    correct=pin@(expm(.07*penalty@cross@penalty)-np.eye(2))@pin
    require(bool(np.linalg.norm(correct)<.03*np.linalg.norm(wrong)),'Inside exponential exit damping fails angular loop control')
    # Execute coefficientONE Casimir from all15 generators, not a guessed norm.
    generators=su4_basis();casimir_matrix=sum(g@g for g in generators)
    colour_generator=generators[-1]
    colour_norm=float(np.linalg.norm(colour_generator,2))
    colour_bound=math.sqrt(float(eigh(casimir_matrix,eigvals_only=True)[0]))
    require(bool(np.linalg.norm(casimir_matrix-15*np.eye(4))<1e-10),'SU4 fundamental Casimir normalization changed')
    require(colour_norm<=colour_bound,'SU4 colour derivative exceeds coefficientONE Casimir control')
    return dict(D=60,radial_size=radial_size,angular_cut=angular_cut,all16_Fock_sectors=True,
        rows=rows,maximum_native_angular_heat_norm=max_moment,maximum_form_bound=uniform_bound,
        derivative_is_uncompressed=True,angular_derivative_Gram='l(l+58)/59; actual L12 leaves zonal subspace',
        compact_A_phase_controls=[0.,.43],positive_Eaux_retained=True,
        actual_cycle_Haar_Tout_square_root_angular=haar_rows,
        full60_exterior_domain='r>1.2 with radial upper Dirichlet cut6, rotation-invariant subdomain of SAME joint outside; not full-domain proof by numerics.',
        wrong_norm_proximity_unbounded_moment=proximity,
        wrong_noquartic_unbounded_actual60D_heat_moment=noquartic,
        coefficientONE_SU4_generator_norm=colour_norm,coefficientONE_SU4_Casimir_bound=colour_bound,
        wrong_final_penalty_only_chart_loop=float(np.linalg.norm(wrong)),
        actual_inside_exponential_damped_chart_loop=float(np.linalg.norm(correct)),
        actual_finite_nonzero_transition=float(np.linalg.norm(pout@penalty@expm(.07*penalty@cross@penalty)@penalty@pin)))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-receipt',type=Path,default=ROOT/'covariant_root_transfer_capsule_bd4d5e1d.json')
    parser.add_argument('--source-receipt',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'covariant_outside_angular_capsule.json')
    args=parser.parse_args();started=time.monotonic()
    root_bytes=args.root_receipt.read_bytes();root_receipt=json.loads(root_bytes)
    root_sha=root_receipt['capsule']['source_sha256']
    root_source=root_receipt['capsule']['source_bytes_at_IMPORT']['covariant_root_transfer.py']
    require(hashlib.sha256(root_source.encode()).hexdigest()==root_sha,
            'Root receipt captured source/hash mismatch')
    source_bytes=args.source_receipt.read_bytes() if args.source_receipt else None
    source_receipt=json.loads(source_bytes) if source_bytes else None
    boundary=[native_full60_boundary(axis) for axis in range(3)]
    native=native_outside_heat();refined=native_outside_heat(30,7)
    bounds=outside_angular_bound(root_receipt,source_receipt)
    result: dict[str, Any]=dict(plan=PLAN,revision=REVISION,branch='NEW fullCartesian jointDirichlet outside angular supplier',
        native_domain_Gauss=boundary,native_Hout_angular=native,native_refined=refined,full_family=bounds,
        proof=__doc__,scientific_admission=False,campaign_closed=False,
        capsule=dict(executable_sha256=hashlib.sha256(SOURCE).hexdigest(),source_bytes=SOURCE.decode(),
                     root_sha256=root_sha,root_input_sha256=root_receipt['capsule']['input_sha256'],
                     root_receipt_sha256=hashlib.sha256(root_bytes).hexdigest(),
                     input_sha256=hashlib.sha256(json.dumps(dict(
                         root_receipt_sha256=hashlib.sha256(root_bytes).hexdigest(),
                         source_receipt_sha256=hashlib.sha256(source_bytes).hexdigest() if source_bytes else None),
                         sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                     source_receipt_path=str(args.source_receipt) if args.source_receipt else None,
                     source_receipt_sha256=hashlib.sha256(source_bytes).hexdigest() if source_bytes else None,
                     loaded_code_matches_captured_source=LOADED_CODE_MATCH,
                     python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,
                     runtime_executable=sys.executable,
                     runtime_executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()),
        elapsed_seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(natural_exit=0,output=str(args.output),source_sha256=result['capsule']['executable_sha256'],
        outside_Lheat_log10=bounds['outside_Lindex_Tout_half_norm_log10_upper'],
        whole_PE_Lscalar_log10=bounds['whole_PE_Lscalar_norm_log10_upper'],
        actual_source_admitted=bounds['retained_source_bound_admitted'],
        native_Lheat=refined['maximum_native_angular_heat_norm'],elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':
    main()
