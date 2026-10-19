"""Independent Gauss-tube form proof and exact adverse controls for the old law.

The wave normalizer is an unevaluated positive integral. This CLI evaluates
primitive bounds and finite controls, never a ground or a wave integral.
Historical receipts/capsules remain immutable and are explicitly cross-bound.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import joined_phi_u1_stationary_noncentral_quantum as quantum
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).read_bytes()
OLD_RECEIPT = 'joined_phi_u1_stationary_noncentral_quantum_actual_receipt_20261010.json'
OLD_CAPSULE = 'joined_phi_u1_stationary_noncentral_quantum_natural_exit_capsule_20261010.json'
OLD_ARCHIVE = 'joined_phi_u1_stationary_noncentral_quantum_actual_receipt_20261010_input_bytes.zip'
PINS = {
    **quantum.PINS,
    OLD_RECEIPT: 'c42058da1254206cb4c8d44bd015e0f909d2f3c3f99d115dbe100e479b15537b',
    OLD_ARCHIVE: '19b25df24e537ddb438f87407f53cd482219c08557f3812c8c87fedfda7a7c8e',
}
sys.set_int_max_str_digits(0)

PROOF = {
    'metric_and_orbit': (
        'M=R^3840_x x SU4^192, D=6720. Cart x and the left generators '
        'T_A=B_A/sqrt8 (Tr T_A T_B=delta_AB/2) are orthonormal for this '
        'auxiliary product metric; its derivative energy dominates the actual '
        'K/4chi because 3chi/20<=4chi. A bi-invariant SU4 metric has '
        'Ric(X,X)=1/4 sum_A ||[X,T_A]||^2>=0; the Euclidean factor has Ric=0. '
        'The compact SU4^64 action is isometric. Its orbit is an embedded '
        'homogeneous manifold G/H, not G: flat parallel transport determines '
        'each stabilizer element by its value at one site; the witness Pauli '
        'fields generate M2 on each of the two joint Wilson blocks. The '
        'centralizer is S(U1 x U1), dimension1 (possibly disconnected finite '
        'components do not alter dimension), so dim O=960-1=959, codim5761.'
    ),
    'normal_ray_proof': (
        'For (p,nu) in the unit normal bundle SN(O), let tau be its first '
        'minimizing normal cut time. For 0<t<tau, normal exp is nonsingular '
        '(a minimizing normal has no earlier focal point) and its volume '
        'Jacobian j relative to dVol_O dnu dt is positive. The 959 base Jacobi '
        'fields have J(0)=base tangent, Jprime(0)=-A_nu base tangent; the '
        '5760 normal angular Jacobi fields have J(0)=0, Jprime(0)=angular '
        'tangent. Therefore j(t)=t^5760(1+O(t)). The full transverse matrix '
        'S=Jprime J^-1 is symmetric by the Jacobi Wronskian, and has 6719 '
        'rows, not 5760 or 5761. Riccati gives tr Sprime=-tr S^2-Ric(nu,nu). '
        'For u=j^(1/6719), u_doubleprime/u=-(tr S^2-(tr S)^2/6719 '
        '+Ric(nu,nu))/6719<=0. Since u(0)=0, concavity on '
        '(epsilon,2t), followed by epsilon->0, yields u(2t)<=2u(t). '
        'Hence j(2t)<=2^6719 j(t) if 2t<tau. Define j_cut(t)=j(t) '
        'for t<tau and0 otherwise. The same scaling inequality holds in the '
        'crossing-cut and both-cut cases; no concavity of j_cut is asserted. '
        'For a compact smooth embedded submanifold the distance cut locus '
        'has ambient measure0; off it the nearest footpoint and its normal '
        'are unique, and the normal exponential integration has multiplicity1. '
        'Thus Vol(Tube(r))=integral_SN integral_0^r j_cut(t) dt, and '
        'Vol(Tube(2r))<=2^6720 Vol(Tube(r)). A qualitative local tubular '
        'neighbourhood follows from compact embeddedness; no quantitative '
        'reach, free action, flat normal bundle, or zero second fundamental '
        'form is required. Completeness and nonnegative ambient Ricci, '
        'smooth constant-dimensional embedded orbit, first-cut truncation '
        'and multiplicity-one a.e. are the actual premises.'
    ),
    'spectral_overlap': (
        'Use PLUS neutral h, physical Hneutral=-Lambda dGamma(h). At the '
        'rational flat witness each Pauli block has h(p)=(M+R) sigma3 '
        '+v1 sigma1+v2 sigma2, v1^2+v2^2=K2=2ci^2(b^2+c^2)>4. '
        'Pair p with p+pi(1,1,1): R and both transverse entries reverse '
        'sign. The even-site compression is (Pplus(p)+Pplus(p+pi))/2. '
        'Its determinant is (1-dot(nplus,nminus))/8, with dot '
        '=(M2-R2-K2)/sqrt((M2-R2-K2)^2+4M2K2). Its derivative '
        'with respect to R2 is -4M2K2/((M2-R2-K2)^2+4M2K2)^(3/2)<0. '
        'The maximum dot is at R2=0, yielding determinant>= '
        'eta=K2/[4(M2+K2)]. Compression eigenvalues are in[0,1], '
        'so each eigenvalue is at least that two-dimensional determinant. '
        'There are32 momentum pairs in each of2 blocks, hence128 '
        'compression eigenvalues and ||wedge^128 Pplus Omega_even||^2 '
        '=det(Pplus_even,even)>=eta^128>=2^-1024. The gap about0 '
        'is at least sqrt K2>2. No finite floating determinant is used '
        'as the analytic tube lower bound.'
    ),
    'global_section_and_form_domain': (
        'Omega_even fills all4 colours at each of32 even sites: each '
        'onsite exterior4 is exactly det(g_x)=1 under SU4. Thus '
        'v(F)=wedge^128 Pplus(F) Omega_even transforms equivariantly. '
        'The gauge-invariant scalar f=(1-d(F,O)/delta)_+, delta2^-2048, '
        'gives a global physical section without a gauge slice or choice '
        'of positive-band eigenvector phases. On the unit tube, total '
        'Phi HS radius<17. Each endpoint/path derivative has operator '
        'norm<=17 times its absolute coefficient weight; reverse paths '
        'and all256 matrix rows are dominated by CH=256*17*Kneutral. '
        'CH<2^20, CH delta<1, so Weyl gives the same128 positive bands '
        'and a gap>=1 throughout the tube. The off-diagonal projector '
        'derivative solves the positive/negative Sylvester equation; '
        'the convergent exponential integral bounds ||DP||<=2CH. '
        'Exterior differentiation gives128*2CH per unit tangent; '
        'the full6720 orthonormal derivative sum gives ||grad v||^2 '
        '<=6720*(256CH)^2<2^80. Integration along a shortest orbit-to-F '
        'geodesic gives ||v(F)-v(F_orbit)||<=256CH delta<2^-513. '
        'Consequently ||v||^2>=2^-1026 and ||v||<=1. The actual '
        'Z=integral_M f^2||v||^2 is finite and strictly positive, '
        'with Z>=2^-1026 integral_M f^2, not a selected normalization. '
        'f is Lipschitz, |grad f|<=1/delta a.e.; f>=1/2 on Tube(delta/2). '
        'The proved volume inequality bounds ||grad f||2^2/||f||2^2 '
        '<=4*2^6720/delta^2. The vector is H1, compact in Phi and full '
        'Haar-supported. Finite fixed Fock sectors and smooth bounded '
        'potentials on this support put it in the closed Friedrichs '
        'FORM domain. Extend fv by0 beyond the tube; its zero trace '
        'preserves H1. Invariant smoothing/compact Gauss projection '
        'approximates in form norm. No H2, D(H), or D(H^2) claim follows '
        'for this tent profile.'
    ),
    'primitive_costs': (
        'Actual coefficients remain Kphi=4chi sum p_x^2, '
        'Kcolour=3chi/20 sum E_A^2, Lambda=2^8192, chi=2^-4096 '
        'and Lambda^(5/4)=2^10240, a=1/64. Since ||v||<=1, '
        '||grad(fv)||2^2<=2||grad f||2^2+2*2^80||f||2^2. '
        'Dividing by Z bounds Phi/colour kinetic by '
        '32chi*2^1024*(4*2^6720/delta^2+2^80)<2^7752. '
        'For X=||Phi||HS, VGram/Lambda<=4X^4; its derivative '
        'is bounded by16*17^3 on this tube. KyFan128 contributes '
        'at most128 CH per unit distance. Together these are<2^60. '
        'At the exact centre alignment and plaquette vanish algebraically. '
        'Each of768 matrix alignment terms has endpoint/telescoping '
        'HS change<=34r; each of192 faces has '
        '4-ReTr Uface=||I-Uface||HS^2/2<=8r^2. '
        'Thus geometry/Lambda<= [768*34^2/40+192*8*7/10] '
        '*Lambda^(1/4)*r^2<2^20*2^2048*r^2. '
        'This prices exact original potentials, not tiny floating residuals.'
    ),
    'charged_Ref_Gauss_and_cover': (
        'Each of64 charged copies occupies4 colours on the32 even sites, '
        'particle number128 per copy. Its one-body density is the diagonal '
        'even-site identity; all offsite native one/two/AP terms have '
        'expectation0, and the onsite trace vanishes for arbitrary '
        'traceless Phi0. Its matter expectation is identically0, '
        'not a ground energy. FullRef4 is an onsite determinant singlet; '
        'its charge cancels the selected fixed origin background4. '
        'Subtracting the charged background2 per site per species '
        'gives rho_x=128(-1)^(x0+x1+x2), totalrho0. Incidence '
        'L=D D^T has Lrho=12rho, so HC=64*rho^T Lplus rho '
        '=64*(64*128^2)/12=2^24/3, without a self-energy subtraction. '
        'U1 longitudinal Gauss is solved by the actual reduced law; '
        'the neutral gauge-equivariant section, charged singlets and Ref '
        'satisfy SU4 Gauss. Total fixedN=128+64*128+4=8324. '
        'The beta0 periodic Theta state is constant and has HE=0. '
        'Choose a normalized beta envelope supported in a local chart '
        'with |A^-1 beta|<=1 and the same constant periodic representative. '
        'Inverse Floquet gives a normalized original-Hilbert L2 PACKET, '
        'with HE=64 integral |A^-1 beta|^2 |eta(beta)|^2<=64. '
        'There is no beta derivative in this decomposable electric '
        'form. No point-beta normalizable real-cover vacuum, minimum '
        'electric coset, or Borel point-evaluation claim is needed.'
    ),
    'central_static_and_old_T': (
        'For any uniform tracelessHermitian4 Phi and central twists, '
        'tracezero plus S128(H)<=Tr|H|/2 and the4x4 HS inequality give '
        'S128(H)<=2 sum64 sqrt(x^T G x). A Clifford realization '
        'attains this at every PSD G. Thus the exact central static '
        'floor is min_ell,G F_ell(G), '
        'F_ell=4096||G||F^2-2sum64 sqrt(x_ell^T G x_ell), '
        'which is8192-strongly convex. Each saved sector lower '
        'is F_ell(Gbar)_lower-||grad F_ell(Gbar)||_upper^2/16384; '
        'the CLI independently checks all64 convexity remainders and '
        'the global minimum binding, but does not regenerate archived '
        'radical intervals. This is a STATIC neutral-plus-Gram floor, '
        'not a full quantum central-sector ground floor. The new global '
        'physical form trial has E/Lambda<=enoncentral+2^-400. '
        'For every normalized old Tv in form domain, conditional on '
        'its root/SPD/gap/chart/normalizer suppliers, the actual b6 '
        'support is |r|<2^-20. The field_ray has an orthonormal adapted '
        'Q and a zero-mean fast field; distance to the old central '
        'orbit<=2|r|. prepared_radius is RMS, not support: '
        '||r f_A||2=sqrtTrCov, Holder and degree3 hypercontractivity '
        'give ||r P f_A||2<=3^(7/4)||P f_A||2 sqrtTrCov<9p sqrtTrCov, '
        'so normalization<=2 and b6<=1 give ||r Tv||<=prepared_radius. '
        'Cauchy bounds mean field distance by2prepared_radius. '
        'The support-wide neutral/quartic Lipschitz constant<2^60 '
        'therefore applies to expectations; charged fixed-sector '
        'Schur lower>=-2^30 holds on the genuine compact support, '
        'uniform allTheta (active129 included). All other form terms '
        'are nonnegative. Hence <Tv,H Tv>/Lambda>=central_floor '
        '-2^61 prepared_radius-2^30/Lambda. This excludes the whole '
        'old image conditionally, never every conceivable central '
        'quantum wave or the actual global root/ground.'
    ),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def primitive_exact(flat: dict[str, Any], old: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    cert = flat['exact_outward_certificate']
    profiles = flat['primitive_profiles']
    a, b, c = [F(cert['trial_parameters'][key]) for key in ('a', 'b', 'c')]
    neutral = profiles[0]
    rows = neutral['paths']
    assert len(rows) == 3 and all(row['length'] == 1 for row in rows)
    assert len({row['ci'] for row in rows}) == 1
    A, ci = F(neutral['onsite']), F(rows[0]['ci'])
    M2, K2 = (A*a)**2, 2*ci**2*(b*b+c*c)
    eta = K2/(4*(M2+K2))
    assert M2 > 0 and K2 > 4 and eta >= F(1, 256)
    overlap = eta**128
    assert overlap >= F(2)**-1024
    weight = lambda profile: abs(F(profile['onsite'])) + 2*sum(
        (abs(F(row['c0'])) + abs(F(row['ci'])) for row in profile['paths']), F(0))
    CH = 256*17*weight(neutral)
    charged_bound = 17*(129*weight(profiles[1])+8064*weight(profiles[2]))
    assert CH < F(2)**20 and charged_bound < F(2)**30
    R2 = 256*(a*a+b*b+c*c)
    pars = root['actual_stationary_root']['parameters']
    old_R2_upper = 256*(F(pars['g0']['upper'])+3*F(pars['gs']['upper']))
    assert R2 < 256 and old_R2_upper < 256
    D, orbit_dim = 64*4*15+192*15, 64*15-1
    assert D == 6720 and orbit_dim == 959
    delta, chi, lam = F(2)**-2048, F(2)**-4096, F(2)**8192
    c2 = F(2)**-1024
    assert CH*delta < 1 and 256*CH*delta < F(2)**-513
    gradient_squared = D*(256*CH)**2
    assert gradient_squared < F(2)**80
    lipschitz = 16*17**3+128*CH
    geometry = F(768*34**2, 40)+F(192*8*7, 10)
    assert lipschitz < F(2)**60 and geometry < F(2)**20
    cutoff_ratio = 4*F(2)**D/delta**2
    kinetic = 32*chi/c2*(cutoff_ratio+F(2)**80)
    assert kinetic < F(2)**7752
    variation = F(2)**60*delta+F(2)**20*F(2)**2048*delta**2
    HC, HE = F(64*64*128**2, 12), F(64)
    assert HC == F(2)**24/3
    correction = variation+(kinetic+HC+HE)/lam
    assert correction < F(2)**-400
    sectors = cert['central_sectors']
    assert len(sectors) == 64 and len({tuple(row['ell']) for row in sectors}) == 64
    for row in sectors:
        reconstructed = F(row['point_value']['lower'])-F(row['gradient_norm_squared']['upper'])/16384
        assert F(row['minimum_lower']['lower']) <= reconstructed
    central = min(F(row['minimum_lower']['lower']) for row in sectors)
    assert central == F(cert['central_all64_lower'])
    center_upper = F(cert['trial_static_energy_divided_Lambda']['upper'])
    unrounded_upper = F(cert['trial_quartic_exact'])-F(cert['trial_global128_sea']['lower'])
    assert 0 <= center_upper-unrounded_upper < F(2)**-100
    quantum_upper = center_upper+correction
    radius = F(old['uniform_intertwiner_price']['prepared_radius']['upper'])
    assert radius < F(2)**-3000
    old_lower = central-F(2)**61*radius-F(2)**30/lam
    static_gap, image_gap = central-quantum_upper, old_lower-quantum_upper
    assert static_gap > F(3139387, 10**7) and image_gap > F(3139387, 10**7)
    original = quantum.exact_budgets(flat, old)
    assert F(original['quantum_energy_divided_Lambda_upper']) == quantum_upper
    assert F(original['conditional_old_image_lower']) == old_lower
    return {
        'eta': str(eta), 'eta_power128_lower': str(overlap),
        'coarse_overlap_squared': '2^-1024', 'center_gap_squared': str(K2),
        'neutral_derivative_CH': str(CH), 'charged_absolute_lower_magnitude_upper': str(charged_bound),
        'field_HS_radius_squared': str(R2), 'old_center_HS_radius_squared_upper': str(old_R2_upper),
        'ambient_dimension': D, 'orbit_dimension': orbit_dim, 'codimension': D-orbit_dim,
        'normal_angular_dimension': D-orbit_dim-1, 'Riccati_dimension': D-1,
        'gradient_squared_upper': str(gradient_squared), 'primitive_lipschitz_upper': str(lipschitz),
        'geometry_quadratic_coefficient_upper': str(geometry),
        'kinetic_upper': str(kinetic), 'kinetic_upper_coarse': '2^7752',
        'HC_exact': str(HC), 'original_packet_HE_upper': str(HE),
        'correction_divided_Lambda': str(correction), 'correction_coarse': '<2^-400',
        'central_static_lower': str(central), 'quantum_upper_divided_Lambda': str(quantum_upper),
        'static_floor_separation_lower': str(static_gap), 'static_floor_separation_display': float(static_gap),
        'conditional_old_T_lower_divided_Lambda': str(old_lower),
        'conditional_old_T_separation_lower': str(image_gap), 'conditional_old_T_separation_display': float(image_gap),
        'old_T_RMS_radius_upper': str(radius), 'old_T_true_support_radius': '2^-20',
        'old_T_radius_is_not_support': True,
        'archived64_sector_radical_intervals': 'Input exact certificate; structural/convex remainder binding checked, no old producer rerun',
        'original_exact_budgets_agree': True,
    }


def exact_wrong_controls() -> dict[str, Any]:
    # Flat rays from a k-dimensional plane: j(t)=t^(D-k-1), u is concave.
    D, k, t = 7, 2, F(3, 5)
    codim = D-k
    j_ratio = (2*t)**(codim-1)/t**(codim-1)
    volume_ratio = F(2)**codim
    assert j_ratio == 16 and j_ratio <= F(2)**(D-1) and volume_ratio <= F(2)**D
    # First cut is not a focal time in general: its Jacobian need not vanish.
    cut = F(3, 4)
    good_cut_ratio = F(0) if 2*t >= cut else j_ratio
    assert good_cut_ratio == 0
    # Exact unit-circle tube at radius2: its inward rays stop at focal/cut1.
    circle_radius = F(2)
    outward_integral = circle_radius+circle_radius**2/2
    inward_cut_integral = F(1, 2)
    inward_uncut_absolute_integral = F(1, 2)+(circle_radius-1)**2/2
    circle_volume_over_pi = 2*(outward_integral+inward_cut_integral)
    wrong_circle_volume_over_pi = 2*(outward_integral+inward_uncut_absolute_integral)
    assert circle_volume_over_pi == 9 and wrong_circle_volume_over_pi == 10
    # Dropping Ric>=0 permits convex u=1+t^2, j=u^(D-1).
    bad_D = 3
    bad_ratio = F(5, 2)**(bad_D-1)
    assert bad_ratio > F(2)**(bad_D-1)
    hyperbolic_volume_ratio = (math.sinh(4)-4)/(math.sinh(2)-2)
    assert hyperbolic_volume_ratio > 8
    # The unit 1D tent is H1, but its derivative has jumps and its second
    # distributional derivative contains Dirac masses, not an L2 function.
    tent_norm, tent_gradient = F(2, 3), F(2)
    assert tent_gradient/tent_norm == 3
    tent_derivative_jumps = [F(1), F(-2), F(1)]
    assert any(jump != 0 for jump in tent_derivative_jumps)
    # An exact scalar distribution disproves RMS-as-support even without
    # underflow. 1/1024 probability at radius16 yields RMS1/2, support16.
    probability, far_radius, rms2 = F(1, 1024), F(16), F(1, 4)
    assert probability*far_radius**2 == rms2 and far_radius**2 > rms2
    # Actual overlap algebra on a Pauli pair: M=3, transverse=4, R=0.
    M2, K2 = F(9), F(16)
    dot = (M2-K2)/(M2+K2)
    determinant = (1-dot)/8
    eta = K2/(4*(M2+K2))
    assert determinant == eta == F(4, 25)
    # Do not substitute a single compression determinant for all128 factors.
    one_eigenvalue = F(1, 256)
    full_overlap = one_eigenvalue**128
    assert full_overlap == F(2)**-1024 and full_overlap < one_eigenvalue
    # Exact SU4 determinant singlet versus a plausible partial-colour fill.
    phases = [F(-1), F(-1), F(1), F(1)]
    assert math.prod(phases) == 1 and math.prod(phases[:2]) == 1
    wrong_partial_phase = phases[0]*phases[2]
    assert wrong_partial_phase != 1
    hc = F(2)**24/3
    assert hc > 0 and 64*128+128+4 == 8324
    return {
        'flat_normal_rays': {'D': D, 'orbit_dimension': k, 'jacobian_doubling': str(j_ratio), 'volume_doubling': str(volume_ratio)},
        'first_cut_crossing': {'tau': str(cut), 't': str(t), 'truncated_jacobian_ratio': str(good_cut_ratio), 'concavity_of_zero_extension_not_used': True},
        'wrong_uncut_normal_multiplicity': {'unit_circle_tube_radius': str(circle_radius), 'actual_volume_over_pi': str(circle_volume_over_pi), 'uncut_absolute_jacobian_volume_over_pi': str(wrong_circle_volume_over_pi), 'rejected': True},
        'wrong_omit_Ricci': {'convex_u_jacobian_ratio': str(bad_ratio), 'claimed_upper': '4', 'rejected': True, 'hyperbolic_H3_ball_volume_ratio_float': hyperbolic_volume_ratio, 'claimed_volume_upper': 8},
        'wrong_H2_tent': {'H1_norm_squared': str(tent_norm), 'gradient_norm_squared': str(tent_gradient), 'derivative_jump_masses': list(map(str, tent_derivative_jumps)), 'not_in_H2_or_DH2': True},
        'wrong_RMS_support': {'probability_at_far_radius': str(probability), 'far_radius': str(far_radius), 'RMS_squared': str(rms2), 'support_inference_rejected': True},
        'Pauli_overlap_identity': {'dot': str(dot), 'determinant': str(determinant), 'eta': str(eta)},
        'wrong_single_overlap_factor': {'single_eigenvalue': str(one_eigenvalue), 'actual128_product': str(full_overlap), 'rejected': True},
        'wrong_partial_colour_reference': {'determinant_four': str(math.prod(phases)), 'partial_02_multiplier': str(wrong_partial_phase), 'rejected': True, 'scope': 'Exact actual SU4 diagonal(-1,-1,1,1) determinant/partial-exterior control'},
        'wrong_omit_HC': {'missing_energy_exact': str(hc), 'rejected': True},
    }


def archive_binding(raws: dict[str, bytes]) -> dict[str, Any]:
    receipt = json.loads(raws[OLD_RECEIPT])
    capsule = json.loads(raws[OLD_CAPSULE])
    assert capsule['natural_exit'] is True and capsule['exit_code'] == 0
    assert capsule['source_before_sha256'] == receipt['own_source_sha256']
    assert capsule['science_receipt']['sha256'] == sha(raws[OLD_RECEIPT])
    assert capsule['science_input_archive'] is None
    assert receipt['immutable_input_archive']['path'] == OLD_ARCHIVE
    assert receipt['immutable_input_archive']['sha256'] == sha(raws[OLD_ARCHIVE])
    with zipfile.ZipFile(ROOT/OLD_ARCHIVE) as z:
        members = {name: z.read(name) for name in z.namelist()}
    for name, digest in {**receipt['source_sha256'], **receipt['input_sha256']}.items():
        assert sha(members[name]) == digest
    current_sha = sha(quantum.CAPTURED[Path(quantum.__file__).name])
    assert current_sha != receipt['own_source_sha256']
    return {
        'original_natural_exit': True, 'original_exit_code': 0,
        'original_receipt_sha256': sha(raws[OLD_RECEIPT]), 'original_capsule_sha256': sha(raws[OLD_CAPSULE]),
        'original_source_sha256': receipt['own_source_sha256'],
        'original_capsule_science_input_archive': None,
        'explicit_receipt_to_immutable_archive_binding': receipt['immutable_input_archive'],
        'archive_all6_member_hashes_agree': len(members) == 6,
        'current_source_sha256': current_sha,
        'source_epochs_distinct': True,
        'historical_capsule_not_backfilled_or_rebound': True,
    }


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    raws = {name: (ROOT/name).read_bytes() for name in PINS}
    assert all(sha(raw) == PINS[name] for name, raw in raws.items())
    raws[OLD_CAPSULE] = (ROOT/OLD_CAPSULE).read_bytes()
    flat, old = [json.loads(raws[name]) for name in quantum.PINS]
    binding = archive_binding(raws)
    data = quantum.field.load_mean(g_override=[1, 1, 1, 1], mean_descriptor={
        'scope': 'Exact original microconfig only, no adopted mean/root and no solver.'})
    for key in ('root_receipt', 'full_receipt'):
        descriptor = data['descriptor'][key]
        name = Path(descriptor['path']).name
        raws[name] = (ROOT/name).read_bytes()
        assert sha(raws[name]) == descriptor['sha256']
    root = json.loads(raws[Path(data['descriptor']['root_receipt']['path']).name])
    for actual, declared in zip(data['profiles'], flat['primitive_profiles'], strict=True):
        assert str(F(actual['onsite'])) == declared['onsite'] and actual['load'] == declared['load']
        for row, saved in zip(actual['paths'], declared['paths'], strict=True):
            assert all(str(F(row[key])) == saved[key] for key in ('c0', 'ci'))
            assert row['length'] == saved['length'] and row['axis'] == saved['axis']
    exact = primitive_exact(flat, old, root)
    controls = exact_wrong_controls()
    finite = quantum.finite_controls(data, F(exact['eta']))
    sources = {**quantum.field.CAPTURED, **quantum.CAPTURED, Path(__file__).name: SELF}
    assert all((ROOT/name).read_bytes() == raw for name, raw in {**sources, **raws}.items())
    archive = output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in {**sources, **raws}.items():
            z.writestr(name, raw)
    result = {
        'kind': 'INDEPENDENT_GLOBAL_GAUSS_FORM_VARIATIONAL_PROOF_STATIC_AND_CONDITIONAL_IMAGE_EXCLUSION',
        'plan': quantum.PLAN, 'proof': PROOF, 'exact_budgets': exact,
        'exact_good_wrong_controls': controls, 'actual_finite_N4_controls': finite,
        'findings': [
            'Tube doubling is valid with the stated complete Ric>=0 regular-normal first-cut/multiplicity-one hypotheses; no quantitative reach/free-orbit premise is missing.',
            'Corrected source proof: old prepared_radius is an L2/RMS bound, not pointwise support. The expectation exclusion survives by Cauchy and the true b6 support-wide Lipschitz bound.',
            'The tent is only claimed in the global physical form domain; the original D(H^2) nonclaim is essential.',
            'The central comparison is the all64 UNIFORM CENTRAL STATIC neutral+Gram floor. The full old-T image exclusion is separately supplier-conditional; no global full-quantum central-sector floor is asserted.',
        ],
        'historical_binding': binding,
        'source_sha256': {name: sha(raw) for name, raw in sources.items()},
        'input_sha256': {name: sha(raw) for name, raw in raws.items()},
        'sources_inputs_unchanged': True,
        'immutable_input_archive': {'path': archive.name, 'sha256': sha(archive.read_bytes())},
        'runtime': {'executable': sys.executable, 'python': sys.version},
        'elapsed_seconds': time.monotonic()-started,
        'not_measured': ['Z', 'normalized6720dim wave integral', 'actual Rayleigh integral'],
        'open': ['actual fullPhi ground characterization', 'ground identity or unique phase', 'beta* or minimizing electric coset', 'same-ground photon pole/dispersion/residue', 'common cone', 'oldSOURCE/CURRENT427 adoption', 'campaign closure'],
    }
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.output)
    print(json.dumps({
        'receipt': args.output.name, 'receipt_sha256': sha(args.output.read_bytes()),
        'static_separation': result['exact_budgets']['static_floor_separation_display'],
        'conditional_image_separation': result['exact_budgets']['conditional_old_T_separation_display'],
        'source_epoch': result['historical_binding']['current_source_sha256'],
        'archive': result['immutable_input_archive'], 'elapsed_seconds': result['elapsed_seconds'],
    }, indent=2))
