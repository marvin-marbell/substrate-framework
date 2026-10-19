"""Revision-bound static Jordan 3+1 review; no scientist import or root solve.

The analytical argument is recorded in REVIEW. Exact symbolic controls can reject
its load-bearing identities; native finite spectra are controls, not a proof of
an all-angle minimum or the full quantum ground. Only this file's prefix is owned.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import zipfile

import numpy as np
import sympy as sp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SOURCE = 'joined_phi_u1_jordan_31_odd_residual.py'
RECEIPT = 'joined_phi_u1_jordan_31_epoch1_actual_20261010.json'
CAPS = 'joined_phi_u1_jordan_31_epoch1_natural_exit_20261010.json'
CHECKER = 'joined_phi_u1_jordan_31_final_selected_checker_20261010.json'
SOURCE_PIN = '3f6e083870c544c893849601563e2a529712fa3b902697c00cdd62148f9311e1'
RECEIPT_PIN = 'edf1d10d58239c89e7fa5aea970dfd3d33fb352eb059836c0ed08644fcc3e3e4'
INCOMING = 'joined_phi_u1_matrix_jordan_flat_actual_20261010_receipt.json'
LSP = 'joined_phi_u1_jordan_31_mathreview_preedit_lsp_20261010.json'
REVIEW = {
    'verdict': 'VALID_DERIVED_LOWER_BOUND_AT_PINNED_STATIC_NEUTRAL_3PLUS1_SCOPE_ONLY',
    'binding': 'source3f6e0838 / receiptedf1d10d, not a provider approval. Native controls below are independently executed; original natural exit and checker are historical reported evidence only.',
    'domain_and_units': 'All four Phi_mu=A_mu directsum s_mu are Hermitian4 with only global tracezero; A_mu arbitrary Hermitian3, s_mu=-Tr A_mu real. Fields are constant in a common flat character frame; neutral physical -Lambda h at exactly global128 of256. Energies here are divided by Lambda, not charged fullH or total fullPhi energy. V=64 and a=1/64 give V/(16a)=256; summation is ordered mu,nu (off-diagonal pairs twice).',
    'odd_CH_and_collection': 'For any traceless Hermitian3 X, spectral theorem and b3=-b1-b2 give TrX^4=(TrX^2)^2/2, including zero and repeated eigenvalues. The t^2 coefficient for X+tY gives 4TrX^2Y^2+2TrXYXY=TrX^2TrY^2+2(TrXY)^2. With S=sum X_a^2 PSD and R_ab=TrX_aX_b, summing ordered pairs gives Q=(TrS)^2+2TrR^2-2TrS^2. HS triangle gives sqrt(TrS^2)<=sum sqrt(TrX_a^4)=TrS/sqrt2, hence Q>=2TrR^2. Subtracting trace part (4/3)TrR^2 gives residual >=(2/3)TrR^2. No commutativity or Clifford representation assumption is made.',
    'projection': 'For s nonzero P=I-ss^T/(s^Ts); otherwise ANY rank3 orthogonal P is legal. A real orthogonal field-index change makes P select three traceless linear combinations because TrA=-s. The full tensor F3_ab={A_a,A_b}-(2/3)T_ab I3 transforms in both indices; its squared HS norm is invariant. The selected 3x3 index subcollection is nonnegative and at most the full norm. Its Gram is the compression PTP and the odd theorem gives ||F3||^2>=(2/3)||PTP||^2. The fourth direction and scalar are retained in the other exact terms, never assumed traceless. s_0=0 alone is NOT s=0; P depends on the full vector.',
    'quartic': 'For T_ab=TrA_aA_b, orthogonal trace decomposition gives qJ=(1024/3)||T||F^2+1024(s^Ts)^2+256||F3||^2. Therefore qJ>=(1024/3)[||T||F^2+||PTP||F^2/2]+1024|s|^4. All coefficients and inequality signs follow from positive squared norms. Equivalently G=(T+ss^T)/4 gives exact base 4096[||G||^2+||G-ss^T||^2/3], plus 256||F3||^2; the alternative global residual uses PTP=4PGP and adds8192||PGP||^2/3. Rank-one approximation and PSD interlacing give qJ>=4096[g1^2+2sum_tail gi^2]; the discarded first-route coefficient4/3 gives a weaker floor, not a contradiction.',
    'dual_and_interlacing': 'On real symmetric4 matrices L:T->PTP is a self-adjoint orthogonal projection. M=I+L/2 has inverse I-L/3. Thus |TrCT|^2<=Q_P(T) D(P) with D(P)=||C||^2-||PCP||^2/3. C is the actual positive diagonal moment matrix. For sorted c1>=...>=c4>=0, the three compression eigenvalues obey lambda_i>=c_(i+1)>=0, so sum lambda_i^2>=c2^2+c3^2+c4^2. D(P)<=c1^2+(2/3)(c2^2+c3^2+c4^2)=D31. This statement would require repair for indefinite C; actual C is positive. No inverse T, inverse G, or field-rank division occurs.',
    'ALLangle_and_determinant': 'Neutral profile has nearest axial paths only: x0=A+2sum c0_j cos p_j, xj=-2ci_j sin p_j, p_j=(2pi k_j+pi+theta_bj)/4. For every continuous shift, four roots of unity annihilate Fourier modes1 and2. This proves zero means and sin*cos means and sin^2=cos^2=1/2 identically; independent axes prove mean64 xx^T=C in EACH character block. AP pi is retained. The actual neutral profile, not the two-link charged profiles, is used for this theorem. In3+1, e^(i theta1)=e^(-3i theta3); gcd(3,1)=1 gives a connected torus with no missed determinant sheets. theta3->theta3+2pi permutes k, while theta1 shifts -6pi. Coincident characters and central boundary points remain valid block-restricted controls, not an exclusion of arbitrary central fields.',
    'occupation_and_sign': 'Trace of the full256 h is64 A(TrA0+s0)=0, even though either block can have nonzero trace. The neutral -Lambda h ground with canonical128 occupies largest128 eigenvalues of +h: Esea/Lambda=-S128. Since tracezero, S128<=sum positive eigenvalues=Tr|h|/2, also at degeneracies and zero bands. For any Hermitian3 H, ||H||1<=sqrt(3TrH^2); Jensen over64 momenta gives halftrace3<=32sqrt(3TrCT). Scalar halftrace<=32sqrt(s^TCs). These are lower-bound relaxations, not blockwise canonical48/16 or fixed2 per momentum, and not spectral pairing.',
    'radial_and_boundaries': 'm3=TrCT>=0, m1=s^TCs>=0. Dual bound gives q3>=1024m3^2/(3D31), while m1<=cmax|s|^2 gives scalar q1>=1024m1^2/cmax^2. Subtract the pressure with the same sign: f3=alpha3 m3^2-32sqrt(3m3), f1=alpha1 m1^2-32sqrt(m1). Write r=sqrt(m)>=0. With r*=(beta/(4alpha))^(1/3)>0, alpha r^4-beta r-f(r*)=alpha(r-r*)^2(r^2+2rr*+3r*^2)>=0, including r=0. Hence min f3=-(9/4)(256D31)^(1/3) and min f1=-(3/4)(256cmax^2)^(1/3). Relaxing the trace/determinant coupling only decreases this infimum. Actual cmax,D31 are strictly positive; an all-zero C would be the separate trivial q>=0 case, not a division-by-zero extension. s=0, s0=0 with s!=0, rank0 through rank4, degenerate Gram, and arbitrarily small nonzero s cause no singular theorem step; P has norm1 even as its direction varies at s=0.',
    'interval_and_scope': 'An independent integer cube-root bisection below checks the irrational expression lies inside the receipted dyadic interval. It separately checks D31, improvement and central gap subtraction in exact rational arithmetic. The central upper is imported conditional incoming evidence, not independently re-proved by this review. The lower is -1454.7222761538771, improvement26.395961286853023, remaining gap22.668091361879316. No central dominance, fullquantum stationary ground, off-flat ground, same-ground photon/pole, source adoption or campaign closure follows.',
    'references': ['joined_phi_u1_jordan_31_odd_residual.py:29-42 (analytic proof), :88-103 (bound arithmetic), :106-147 (epoch binding)', 'joined_phi_u1_jordan_31_epoch1_actual_20261010.json:8-85 (proof and exact endpoints), :2510-2525 (archive/runtime/open)', 'joined_phi_u1_matrix_jordan_flat_bounds.py:39-54, :58-63 (law/scaling)', 'joined_phi_u1_stationary_flat_holonomy.py:30-33, :165-167, :190-212 (characters, Fourier/AP, global sign)', 'joined_phi_u1_stationary_phi_sea_envelope.py:29-38 (neutral +h convention/static limits)', 'induced_phi_quantum_fast_transfer.py:99-119; local_orbital_stability.py:32-42; rotation_band.py:80-92 (binary64 primitive coefficient owner)'],
    'minimal_repair': 'No load-bearing invalidity found at the stated scope. Preserve actual C positivity, neutral nearest-path scope, full s-perp projection and global filling; do not promote this lower bound beyond its static hypothesis.'
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact(x: Any) -> sp.Rational:
    value = sp.Rational(str(x))
    assert isinstance(value, sp.Rational)
    return value


def hs2(x: Any) -> Any:
    return sp.expand(sp.trace(x.conjugate().T*x))


def gram(matrices: list[Any]) -> Any:
    return sp.Matrix(len(matrices), len(matrices), lambda a, b: sp.trace(matrices[a]*matrices[b]))


def quartic(matrices: list[Any]) -> Any:
    return sp.expand(256*sum(hs2(a*b+b*a) for a in matrices for b in matrices))


def exact_algebra(c: list[F]) -> dict[str, Any]:
    u, v = sp.symbols('u v', real=True)
    eigen = sp.diag(u, v, -u-v)
    assert sp.expand(sp.trace(eigen**4)-sp.trace(eigen**2)**2/2) == 0
    # Generic traceless Hermitian3 has eight free real coordinates.
    a, b, d, e, f, g, h, j = sp.symbols('a b d e f g h j', real=True)
    x = sp.Matrix([[a, d+sp.I*e, f+sp.I*g], [d-sp.I*e, b, h+sp.I*j], [f-sp.I*g, h-sp.I*j, -a-b]])
    assert sp.expand(sp.trace(x**4)-sp.trace(x**2)**2/2) == 0
    # Distinct exact sharp parity witness, not the producer's embedded Pauli control.
    b0 = sp.diag(2, -1, -1)
    r2 = sp.trace(gram([b0])**2)
    anti = quartic([b0])/256
    wrong_parity = sp.Rational(8, 3)*r2
    assert anti == 2*r2 == 72 and wrong_parity-anti == 24
    z = sp.zeros(3)
    off1 = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    off2 = sp.Matrix([[0, 0, sp.I], [0, 0, 0], [-sp.I, 0, 0]])
    pauli2 = sp.Matrix([[0, -sp.I, 0], [sp.I, 0, 0], [0, 0, 0]])
    pauli3 = sp.diag(1, -1, 0)
    examples = {
        'rank0_s_zero': [z, z, z, z],
        'rank1_tracefree_s_zero': [b0, z, z, z],
        'rank1_traceful_onsite': [-sp.eye(3), z, z, z],
        'rank2_s0_zero_s_nonzero': [off1, sp.eye(3), z, z],
        'rank3_noncommuting_s_zero': [off1, pauli2, pauli3, z],
        'rank4_noncommuting_s_zero': [off1, pauli2, pauli3, off2],
        'rank4_traceful_general': [b0+sp.eye(3), off1+2*sp.eye(3), pauli2-sp.eye(3), off2+sp.diag(0, 0, 2)]
    }
    C = sp.diag(*[exact(v) for v in c])
    D = sum(v*v for v in c)-sp.Rational(1, 3)*sum(v*v for v in c[1:])
    rows = []
    for label, matrices in examples.items():
        s = -sp.Matrix([sp.trace(m) for m in matrices])
        T = gram(matrices)
        ss = s*s.T
        P = sp.eye(4)-ss/(s.dot(s)) if s.dot(s) else sp.diag(0, 1, 1, 1)
        assert P*P == P and P*s == sp.zeros(4, 1)
        Phi = [sp.diag(m, s[k]) for k, m in enumerate(matrices)]
        q = quartic(Phi)
        residual = sum(hs2(m*n+n*m-sp.Rational(2, 3)*T[k,l]*sp.eye(3)) for k, m in enumerate(matrices) for l, n in enumerate(matrices))
        projected = hs2(P*T*P)
        floor = sp.Rational(1024, 3)*(hs2(T)+projected/2)+1024*s.dot(s)**2
        assert sp.expand(q-sp.Rational(1024, 3)*hs2(T)-1024*s.dot(s)**2-256*residual) == 0
        assert residual-sp.Rational(2, 3)*projected >= 0 and q-floor >= 0
        G = (T+ss)/4
        assert sp.expand(q-4096*(hs2(G)+hs2(G-ss)/3)-256*residual) == 0
        Q = hs2(T)+projected/2
        d = hs2(C)-hs2(P*C*P)/3
        dual_slack = sp.factor(d*Q-sp.trace(C*T)**2)
        assert F(str(dual_slack)) >= 0 and D >= d
        rows.append({'case': label, 'field_Gram_rank': int(T.rank()), 's': [str(v) for v in s], 'qJ_exact': str(q), 'projected_floor_exact': str(floor), 'quartic_slack_exact': str(q-floor), 'odd_residual_slack_exact': str(residual-sp.Rational(2, 3)*projected), 'dual_slack_exact': str(dual_slack)})
    # Field-index tensor invariance is checked by a non-coordinate rational O.
    O = sp.eye(4)-2*sp.ones(4, 1)*sp.ones(1, 4)/4
    seed = examples['rank4_traceful_general']
    rotated = [sum((O[k,l]*seed[l] for l in range(4)), sp.zeros(3)) for k in range(4)]
    assert O.T*O == sp.eye(4)
    def full_phi(ms: list[Any]) -> list[Any]:
        return [sp.diag(m, -sp.trace(m)) for m in ms]
    assert quartic(full_phi(seed)) == quartic(full_phi(rotated))
    # Incorrectly treating s0=0 as s=0 permits P containing a traceful direction.
    wrong_P = sp.diag(1, 1, 1, 0)
    wrong_seed = [z, sp.eye(3), z, z]
    traceful = gram(wrong_seed)
    traceful_F = sum(hs2(m*n+n*m-sp.Rational(2, 3)*traceful[k,l]*sp.eye(3)) for k,m in enumerate(wrong_seed) for l,n in enumerate(wrong_seed))
    assert traceful_F == 0
    assert sp.Rational(2,3)*hs2(wrong_P*traceful*wrong_P) == 6
    # Omitting the compensating scalar changes a literal quartic by82944.
    assert quartic(full_phi(examples['rank1_traceful_onsite']))-quartic(examples['rank1_traceful_onsite']) == 82944
    # Dual equality fixes1/3, not1/2: this exact PSD T attains the dual norm.
    P = sp.diag(0, 1, 1, 1)
    T = C-(P*C*P)/3
    Q = hs2(T)+hs2(P*T*P)/2
    d = hs2(C)-hs2(P*C*P)/3
    wrong_d = hs2(C)-hs2(P*C*P)/2
    assert sp.factor(sp.trace(C*T)**2-d*Q) == 0
    rejection = sp.factor(sp.trace(C*T)**2-wrong_d*Q)
    assert F(str(rejection)) > 0
    alpha, r, t = sp.symbols('alpha r t', positive=True)
    assert sp.expand(alpha*r**4-4*alpha*t**3*r+3*alpha*t**4-alpha*(r-t)**2*(r**2+2*r*t+3*t**2)) == 0
    return {'generic_eight_real_variable_odd_CH_polynomial': 'identically zero', 'radial_global_factorization': 'identically zero', 'distinct_rank1_sharp_witness': {'anti': str(anti), '2Gram2': str(2*r2), 'wrong_8over3_Gram2_rejected_by': str(wrong_parity-anti)}, 'all_rank_and_trace_boundaries': rows, 'rational_orthogonal_field_rotation_invariance': True, 'wrong_s0_projection_residual_floor': '6 > actual selected residual0', 'wrong_scalar_omission_quartic_deficit': '82944', 'wrong_dual_half_instead_of_third_exact_rejection': str(rejection), 'dual_equality_P_excludes_Cmax': True}


def cube_bracket(x: F, bits: int = 150) -> tuple[F, F]:
    scale = 1 << bits
    low, high = 0, 1
    def below(n: int) -> bool:
        return n**3*x.denominator <= x.numerator*scale**3
    while below(high):
        high *= 2
    while high-low > 1:
        middle = (low+high)//2
        if below(middle):
            low = middle
        else:
            high = middle
    assert below(low) and not below(high)
    return F(low, scale), F(high, scale)


def interval_review(profile: dict[str, Any], receipt: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[F], dict[str, Any]]:
    A = F(profile['onsite'])
    rows = profile['paths']
    assert profile['scale'] == 'heavy' and profile['load'] == 1
    assert len(rows) == 3 and all(p['length'] == 1 for p in rows)
    # Literal source expression ordering, binary64 scalars preserved exactly.
    assert F.from_float(cfg['heavy_m']+3.*cfg['heavy_r']/cfg['a']) == A
    assert all(F.from_float(-cfg['heavy_r']/(2*cfg['a'])) == F(p['c0']) for p in rows)
    assert all(F.from_float(-cfg['heavy_v']/(2*cfg['a'])) == F(p['ci']) for p in rows)
    c = [A*A+2*sum(F(p['c0'])**2 for p in rows)]
    c += [2*F(next(p['ci'] for p in rows if p['axis'] == j))**2 for j in range(3)]
    assert all(v > 0 for v in c) and c[0] == max(c)
    cert = receipt['ALLangle_certificate']
    assert c == [F(v) for v in cert['C_exact']]
    d = c[0]**2+F(2,3)*sum(v*v for v in c[1:])
    assert d == F(cert['D31_exact'])
    rlo, rhi = cube_bracket(256*d)
    slo, shi = cube_bracket(256*c[0]**2)
    lower, upper = -F(9,4)*rhi-F(3,4)*shi, -F(9,4)*rlo-F(3,4)*slo
    row = cert['different_odd_block_pressure_floor']
    assert F(row['lower']) <= lower <= upper <= F(row['upper'])
    prev = F(cert['incoming_ALLangle_floor']['lower'])
    central = F(cert['central_trial_upper']['upper'])
    improvement = (lower-prev, upper-prev)
    gap = (central-upper, central-lower)
    assert improvement[0] > 0 and gap[0] > 0
    for key, pair in [('improvement_above_incoming_floor', improvement), ('remaining_gap_to_central_trial', gap)]:
        assert F(cert[key]['lower']) <= pair[0] <= pair[1] <= F(cert[key]['upper'])
    return c, {'coefficient_reconstructed_from_frozen_micro_config': True, 'C_exact': [str(v) for v in c], 'D31_exact': str(d), 'independent_150bit_lower': str(lower), 'independent_150bit_upper': str(upper), 'receipted_interval_contains_independent_bracket': True, 'improvement_display': float(improvement[0]), 'remaining_gap_display': float(gap[0]), 'central_upper_is_imported_not_reproved': True}


def angle_and_occupation(profile: dict[str, Any], c: list[F]) -> dict[str, Any]:
    z = sp.symbols('z', nonzero=True)
    waves = [z*sp.I**k for k in range(4)]
    co = [(w+1/w)/2 for w in waves]
    si = [(w-1/w)/(2*sp.I) for w in waves]
    checks = [sum(co)/4, sum(si)/4, sum(v*v for v in co)/4-sp.Rational(1,2), sum(v*v for v in si)/4-sp.Rational(1,2), sum(v*w for v,w in zip(co,si)) /4]
    assert all(sp.simplify(x) == 0 for x in checks)
    A = F(profile['onsite'])
    c0 = F(profile['paths'][0]['c0'])
    assert A > 0 and c0 < 0 and A*A > 18*c0*c0
    lattice = np.asarray(list(product(range(4), repeat=3)), dtype=float)
    def coefficient(theta: np.ndarray, AP: bool = True) -> np.ndarray:
        p = (2*np.pi*lattice+(np.pi if AP else 0)+theta)/4
        out = np.zeros((64,4))
        out[:,0] = float(A)
        for row in profile['paths']:
            axis = row['axis']
            out[:,0] += 2*float(F(row['c0']))*np.cos(p[:,axis])
            out[:,axis+1] -= 2*float(F(row['ci']))*np.sin(p[:,axis])
        return out
    # Real finite consumer path: literal64-block spectra, without scientist imports.
    controls = []
    B = np.diag([-1.,-1.,-1.,3.])
    for theta in (np.zeros(3), np.array([np.sqrt(2),np.pi/7,-np.sqrt(3)]), np.array([2*np.pi, -2*np.pi, 4*np.pi])):
        x3, x1 = coefficient(theta), coefficient(-3*theta)
        cov_error = max(float(np.max(abs(x.T@x/64-np.diag([float(v) for v in c])))) for x in (x3,x1))
        determinant_error = float(np.max(abs(np.exp(1j*(3*theta-3*theta))-1)))
        block3 = -x3[:,0,None]*np.ones((1,3))
        block1 = 3*x1[:,0,None]
        bands = np.concatenate([block3,block1],axis=1)
        sorted_bands = np.sort(bands.ravel())
        S = float(sorted_bands[-128:].sum())
        fixed = float(np.sort(block3.ravel())[-96:].sum()+np.sort(block1.ravel())[-32:].sum())
        halftrace = float(np.abs(bands).sum()/2)
        assert cov_error < 1e-10 and determinant_error < 1e-12
        assert abs(float(bands.sum())) < 1e-9 and S <= halftrace+1e-9
        assert abs(S-fixed) > 1
        controls.append({'theta3': theta.tolist(), 'theta1': (-3*theta).tolist(), 'covariance_max_error': cov_error, 'determinant_error': determinant_error, 'global_top128': S, 'half_absolute_trace': halftrace, 'wrong_fixed_block_half': fixed, 'wrong_fixed_top2_per_momentum': float(np.sort(bands,axis=1)[:,2:].sum())})
    # Exact AP occupation witness: x0=A+sqrt2*c0*z, z=3,1,-1,-3.
    # Multiplicities8,24,24,8. All64 scalar positive bands must be occupied,
    # then64 least-negative three-block bands:24 from z3 and40 from z1.
    global_exact = 128*exact(A)-112*sp.sqrt(2)*exact(c0)
    wrong_block = -288*sp.sqrt(2)*exact(c0)
    fixed_top2 = 128*exact(A)
    assert float(global_exact-fixed_top2) > 0 and float(global_exact-wrong_block) > 0
    assert abs(controls[0]['global_top128']-float(global_exact)) < 1e-9
    assert abs(controls[0]['wrong_fixed_block_half']-float(wrong_block)) < 1e-9
    # Matrix construction independent of shortcut:64 Hermitian4 eigenproblems.
    x = coefficient(np.zeros(3))
    native_bands = np.linalg.eigvalsh(x[:,0,None,None]*B)
    assert abs(float(np.sort(native_bands.ravel())[-128:].sum())-float(global_exact)) < 1e-9
    no_AP = coefficient(np.zeros(3),AP=False)
    AP_matrix_HS = float(np.linalg.norm((x[:,0]-no_AP[:,0])[:,None,None]*B))
    assert AP_matrix_HS > 1
    return {'symbolic_continuous_shift_fourpoint_Laurent_identities': ['0']*5, 'finite_native_controls_NOT_proof': controls, 'exact_global128': str(global_exact), 'exact_wrong_fixed_block_half': str(wrong_block), 'exact_wrong_fixed2': str(fixed_top2), 'exact_fixed2_rejection': str(global_exact-fixed_top2), 'exact_fixed_block_rejection': str(global_exact-wrong_block), 'AP_omission_literal_matrix_HS': AP_matrix_HS, 'canonical_physical_sea_sign': '-global_top128', 'canonical_population': 128, 'bands': 256}


def run(output: Path, contract: Path) -> dict[str, Any]:
    started = time.monotonic()
    names = [Path(__file__).name,SOURCE,RECEIPT,CAPS,CHECKER,LSP]
    snapshots = {name: (ROOT/name).read_bytes() for name in names}
    assert sha(snapshots[SOURCE]) == SOURCE_PIN and sha(snapshots[RECEIPT]) == RECEIPT_PIN
    receipt = json.loads(snapshots[RECEIPT])
    capsule = json.loads(snapshots[CAPS])
    checker = json.loads(snapshots[CHECKER])
    assert capsule['source_sha256'] == SOURCE_PIN and capsule['receipt_sha256'] == RECEIPT_PIN
    assert capsule['natural_exitcode'] == 0 and capsule['stderr'] == ''
    assert capsule['command'][1] == str(ROOT/SOURCE) and capsule['command'][3] == str(ROOT/RECEIPT)
    assert checker['source_sha256'] == SOURCE_PIN and checker['natural_exit_code'] == 0
    assert checker['selected_actual_project_files'] == [str(ROOT/SOURCE)]
    archive = Path(receipt['immutable_archive']['path'])
    snapshots[archive.name] = archive.read_bytes()
    assert sha(snapshots[archive.name]) == receipt['immutable_archive']['sha256']
    bindings = []
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == set(receipt['sources_inputs_sha256'])
        for name, pin in receipt['sources_inputs_sha256'].items():
            raw = zipped.read(name)
            assert sha(raw) == pin
            candidate = ROOT/name
            if name == 'governing_contract.json':
                live = contract.read_bytes()
            elif name == 'preedit_lsp.json':
                live = (ROOT/'joined_phi_u1_jordan_31_preedit_lsp.json').read_bytes()
            else:
                live = candidate.read_bytes()
            assert sha(live) == pin
            bindings.append({'name': name, 'sha256': pin, 'archive_and_live_match': True})
        assert zipped.read(SOURCE) == snapshots[SOURCE]
        incoming = json.loads(zipped.read(INCOMING))
        config = json.loads(zipped.read('induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json'))['input_descriptor']['original_micro_config_UNCHANGED']
    snapshots['governing_contract.json'] = contract.read_bytes()
    contract_data = json.loads(snapshots['governing_contract.json'])
    assert receipt['governing_revision'] == contract_data['governing_revision']
    assert receipt['pinned_template_revision'] == contract_data['pinned_template_revision']
    assert receipt['plan_id'] == contract_data['plan_id']
    profile = incoming['primitive_profiles'][0]
    c, arithmetic = interval_review(profile,receipt,config)
    controls = exact_algebra(c)
    occupation = angle_and_occupation(profile,c)
    # Only immutable owned epoch artifacts are created; source inputs untouched.
    owned_archive = output.with_name(output.stem+'_input_bytes.zip')
    if output.exists() or owned_archive.exists():
        raise FileExistsError('Review epoch already exists')
    with zipfile.ZipFile(owned_archive,'x',zipfile.ZIP_DEFLATED) as zipped:
        for name, raw in sorted(snapshots.items()):
            zipped.writestr(name,raw)
    assert all((ROOT/name).read_bytes() == snapshots[name] for name in names)
    result = {'kind': 'INDEPENDENT_REVISION_BOUND_SCIENTIFIC_MATHREVIEW_NOT_GROUND_ADMISSION', 'review': REVIEW, 'plan_id': receipt['plan_id'], 'governing_revision': receipt['governing_revision'], 'pinned_template_revision': receipt['pinned_template_revision'], 'parent_work_item': 'induced-photon', 'source_sha256': SOURCE_PIN, 'receipt_sha256': RECEIPT_PIN, 'live_archive_binding': bindings, 'independent_bound_arithmetic': arithmetic, 'independent_exact_algebra_and_adverse_controls': controls, 'independent_native_angles_occupation_and_AP': occupation, 'historical_capsule_status': {'natural_exitcode': capsule['natural_exitcode'], 'wall_seconds': capsule['wall_seconds'], 'rerun': False}, 'historical_checker_status': {'natural_exitcode': checker['natural_exit_code'], 'rerun': False}, 'sources_inputs_sha256': {name: sha(raw) for name,raw in snapshots.items()}, 'immutable_archive': {'path': str(owned_archive), 'sha256': sha(owned_archive.read_bytes())}, 'runtime': {'python': sys.executable, 'version': platform.python_version(), 'numpy': np.__version__, 'sympy': sp.__version__}, 'elapsed_seconds': time.monotonic()-started, 'OPEN': ['static31 central gap', 'off-flat/fullPhi quantum stationary ground', 'charged all64/fullHC/Pi/Gauss coupled ground', 'same-ground spatial photon pole/dispersion', 'originalSOURCE/cone/Compton/campaign']}
    with output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.output,args.contract)
    print(json.dumps({'receipt': str(args.output), 'verdict': result['review']['verdict'], 'bound': result['independent_bound_arithmetic'], 'elapsed_seconds': result['elapsed_seconds']},indent=2))


if __name__ == '__main__':
    main()
