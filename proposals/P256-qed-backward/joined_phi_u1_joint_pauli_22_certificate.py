"""Outward all-angle static-neutral NEW Jordan 2+2 certificate.

This permanent scientific consumer pays the rank-four scalar/Pauli penalty,
combines it with the protected frequency-one variance, and covers [0,1] by
closed rational cells. It does not run the old producer or a mean/root solver.
The source-bound input is the previously executed native neutral profile.
Only E/Lambda for the whole N4,V64 static neutral sector is certified; charged,
fullPhi quantum ground, photon/common cone and original-source adoption are open.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
import sympy as sp
from threadpoolctl import threadpool_limits

import joined_phi_u1_jordan_22_consumer as reviewed

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_joint_pauli_22_'
I = reviewed.I
SCALE = reviewed.SCALE
PINS = {
    'joined_phi_u1_jordan_22_consumer.py': '3bbe20dc7dd9203683eef16aa80cbb0735be91e398486c9b5fe7e0832911877e',
    'joined_phi_u1_jordan_22_typed_actual_20261010.json': 'c6d62aeadd356a4b5d04be0d34ea7129bdf60b774704695827be7bbae8d36edb',
    reviewed.INPUT: reviewed.INPUT_PIN,
}
PROOF = {
    'scope': 'NEW independent Jordan law; every static neutral flat2+2 field and ALL continuous angles, all eight determinant sheets. Not originalSOURCE/CURRENT427 or fullPhi quantum ground.',
    'scalar_algebra': 'Phi_mu=t_mu I2+v_mu.sigma, H=VV^T PSD rank<=3, s=t^Tt. Anticommutator=2[(t_mu t_nu+H_mu,nu)I2+(t_mu v_nu+t_nu v_mu).sigma]. Trace-square and sum give q_b=2048Q, Q=TrH^2+s^2+4tHt+2sTrH; coefficient256 comes from V64/(16a), a1/64. All scalar/cross terms are retained. Global tracezero sets t2=-t1, not t1=t2=0.',
    'spectral_identity': 'K=H+tt^T: Q=TrK^2+2sTrK+2tKt-4s^2. If rankK<=3, d=lambda_min(K)=0 and Q-TrK^2=2tHt+2sTrH>=0, including singularH/s0/t0/rank0. If rankK4, H=K-tt^T has rank3 and PSD, so determinant lemma gives t^TK^-1t=1. Set t=K^.5v, ||v||=1, w_i=|v_i|^2: s=sum w_i lambda_i. Penalty P=2sTrK+2sum w_i lambda_i^2-4s^2 is concave on the simplex, hence P>=sum w_i 2lambda_i(TrK-lambda_i)>=2d(TrK-d). Last inequality difference=(lambda_i-d)(TrK-lambda_i-d)>=0. No inverse is used in singular cases and rankK<=3 is never asserted globally.',
    'normalized_constraint': 'For r=sqrtQ>0 write normalized eigenvalues, d=lambda_min/r, T=TrK/r, u=K00/r. PSD and Q>=TrK^2 imply 0<=d<=u<=1 and d<=1/2 (even 10d^2<=1 by rankpenalty). Let mu_i=lambda_i+d for top3: sum mu_i^2=TrK^2+2dT<=1+2d^2. Thus T=sum mu_i-2d<=sqrt(3+6d^2)-2d<=sqrt3; final squaring follows 2d^2<=4sqrt3d for d<=1/2. These are exact derivations, not numerical rank tests.',
    'high_u': 'For u>=1/sqrt3, largest mu>=u+d and (u+d)^2>=(1+2d^2)/3, since 3u^2-1+6ud+d^2>=0. At fixed sum-square budget, z+sqrt[2(B-z^2)] decreases for z>=sqrt(B/3). Hence T<=u-d+sqrt2 sqrt(1+d^2-u^2-2ud)<=u+sqrt2 sqrt(1-u^2), using 0<=d<=u: radicand difference d(d-2u)<=0 and -d<=0. Feasible eigenvalues guarantee the radicand is nonnegative. Rankzero has pressure zero and is separate.',
    'joint_moment': 'Native exact C=diag(C0,Cs,Cs,Cs), C0>Cs>0. m/r=CsT+(C0-Cs)u <=Cs sqrt3+(C0-Cs)u for all u, and <=C0u+Cs sqrt2 sqrt(1-u^2) for u>=1/sqrt3. Intersect both with the already proved m/r<=sqrtD3. This is paid rank4, not a false rank3(K) relaxation.',
    'variance': reviewed.PROOF['protected_variance'],
    'defect': reviewed.PROOF['distinct_defect'],
    'support': reviewed.PROOF['support'],
    'closed_cells': 'Each closed [i/n,(i+1)/n] with n>=16384: increase positive linear u terms to hi, decrease sqrt(1-u^2) to its value at lo, decrease subtractive d_defect*u^2 to lo. If lo<3/5 use global trace envelope throughout (including straddling cell); otherwise high-u envelope is valid because (3/5)^2>1/3. Intersect with sqrtD3. Existing110bit integer outward Interval only. Max of every stored cell upper is a continuum certificate, not angle search/gridabsence.',
    'global_filling': 'Physical neutral=-Lambda h. Across all256 bands Trh=0 globally, so S128<=Tr|h|/2=sum_b,k max(|x.t_b|,sqrt(xH_bx))<=sum_b,k sqrt(xK_bx). No fixed block-half filling. For every SU4 determinant sheet theta2,j=pi*n_j-theta1,j, n inZ2^3, all continuous theta1 retained; relaxing this coupling cannot omit a component.',
    'radial_units': 'Whole static energy/Lambda=sum_b[2048Q_b-S_b]. r=sqrtQ, z=sqrt r gives each block >=2048z^4-64alpha*z. Minimum satisfies z^3=alpha/128, value=-48alpha*(alpha/128)^(1/3). Sum2=-96alpha*(alpha/128)^(1/3)=-3*(256alpha^4)^(1/3). This is whole V64 energy/Lambda, NOT energy perV and not fullH/Lambda. At Q0 pressure0. Lambda=2^8192 is positive; multiplying preserves the separation. chi2^-4096 and Lambda54=2^10240 are unchanged and not consumed by this static sector bound.',
    'debt': 'Charged1+63 canonical128/Ref4, HE=HC64, fullPhi kinetic/offflat/globalquantum ground characterization, stationary sameground spatial photon and commoncone remain OPEN; no adoption or bare Maxwell/covariance fit.',
}


def snapshot() -> dict[str, bytes]:
    captured = reviewed.snapshot()
    for name, pin in PINS.items():
        raw = (ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ArithmeticError('Pinned source/input changed: '+name)
        captured[name] = raw
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).parent == ROOT:
            captured[Path(path).name] = Path(path).read_bytes()
    return captured


def load_profile(raw: bytes) -> dict[str, Any]:
    profile = json.loads(raw)['primitive_profiles'][0]
    if profile['role'] != 'neutral heavy -Lambda' or profile['load'] != 1:
        raise ArithmeticError('Wrong native neutral role')
    profile['onsite'] = F(profile['onsite'])
    for row in profile['paths']:
        row['c0'], row['ci'] = F(row['c0']), F(row['ci'])
        if row['length'] != 1:
            raise ArithmeticError('Wrong neutral path length')
    if sorted(row['axis'] for row in profile['paths']) != [0, 1, 2]:
        raise ArithmeticError('Incomplete native axis population')
    # Reconstruct the archived nearest-path vectors for the native diagnostic.
    # Certificate arithmetic above/below retains the original exact fractions.
    profile['raw_paths'] = []
    for row in profile['paths']:
        displacement = np.zeros(3,dtype=int); displacement[row['axis']] = 1
        coefficient = np.zeros(4,dtype=complex); coefficient[0] = float(row['c0'])
        coefficient[row['axis']+1] = 1j*float(row['ci'])
        profile['raw_paths'].append((displacement,coefficient))
    return profile


def parameters(profile: dict[str, Any]) -> dict[str, Any]:
    a0 = profile['onsite']; c0 = [z['c0'] for z in profile['paths']]; ci = [z['ci'] for z in profile['paths']]
    if len(set(abs(z) for z in c0)) != 1 or len(set(abs(z) for z in ci)) != 1:
        raise ArithmeticError('Native isotropy required')
    c = [a0*a0+2*sum(z*z for z in c0), *[2*z*z for z in ci]]
    if not c[0] > c[1] > 0 or c[1:] != [c[1]]*3:
        raise ArithmeticError('Joint isotropic moment premise failed')
    d3 = c[0]**2+2*c[1]**2
    a, b = 2*abs(c0[0]), 4*ci[0]**2
    if b <= 3*a*a:
        raise ArithmeticError('Support completion not positive')
    support = a0*a0+3*b+3*(a0*a)**2/(b-3*a*a)
    variance = 8*a0*a0*sum(z*z for z in c0)
    fourth = I.of(d3).sqrt().sqrt()
    defect = I.of(variance)/(2*fourth*(I.of(support).sqrt()+fourth).square())
    return dict(C=c, D3=d3, support=support, variance=variance, defect=defect)


def cell_upper(p: dict[str, Any], lo: F, hi: F, use_variance: bool = True) -> I:
    c0, cs = p['C'][:2]
    if lo < F(3, 5):
        moment = I.of(cs)*I.of(3).sqrt()+I.of(c0-cs)*hi
    else:
        moment = I.of(c0)*hi+I.of(cs)*I.of(2).sqrt()*I.of(1-lo*lo).sqrt()
    capped = I(0, min(moment.hi, I.of(p['D3']).sqrt().hi))
    return capped.sqrt()-(p['defect']*(lo*lo) if use_variance else I.of(0))


def envelope(profile: dict[str, Any], cells: int) -> dict[str, Any]:
    if cells < 16384:
        raise ValueError('At least16384 complete rational cells required')
    p = parameters(profile); records = []; no_variance = []
    for index in range(cells):
        lo, hi = F(index, cells), F(index+1, cells)
        bound = cell_upper(p, lo, hi)
        records.append(dict(index=index, lo=str(lo), hi=str(hi), route='global_trace' if lo < F(3,5) else 'high_u', pressure_upper=str(F(bound.hi, SCALE))))
        no_variance.append(F(cell_upper(p, lo, hi, False).hi, SCALE))
    alpha = max(F(row['pressure_upper']) for row in records)
    radial = -3*(256*I.of(alpha).square().square()).cube_root()
    plain_alpha = max(no_variance)
    return dict(C_exact=[str(z) for z in p['C']], D3_exact=str(p['D3']), support_exact=str(p['support']), protected_variance_exact=str(p['variance']), defect=p['defect'].public(), switch_exact='3/5', closed_cells=cells, cell_bounds=records, alpha_upper=str(alpha), alpha_display=float(alpha), lower=radial.public(), without_variance_alpha=str(plain_alpha), without_variance_lower=(-3*(256*I.of(plain_alpha).square().square()).cube_root()).public())


def verify_cells(result: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    p = parameters(profile); rows = result['cell_bounds']; n = result['closed_cells']; previous = F(0); maximum = F(0)
    if len(rows) != n or n < 16384 or F(3,5)**2 <= F(1,3):
        raise ArithmeticError('Wrong continuum cover/switch')
    for i, row in enumerate(rows):
        lo, hi, upper = F(row['lo']), F(row['hi']), F(row['pressure_upper'])
        if row['index'] != i or lo != previous or lo != F(i,n) or hi != F(i+1,n):
            raise ArithmeticError('Rational cover hole/overlap')
        # Explicit independently formed majorant, not acceptance by producer flags.
        if lo < F(3,5):
            m = p['C'][1]*I.of(3).sqrt()+(p['C'][0]-p['C'][1])*hi
        else:
            m = p['C'][0]*I.of(hi)+p['C'][1]*I.of(2).sqrt()*I.of(1-lo*lo).sqrt()
        cap = I(0, min(m.hi, I.of(p['D3']).sqrt().hi))
        check = cap.sqrt()-p['defect']*(lo*lo)
        if upper < F(check.hi, SCALE):
            raise ArithmeticError('Stored pressure bound below independent majorant')
        maximum = max(maximum, upper); previous = hi
    if previous != 1 or maximum != F(result['alpha_upper']):
        raise ArithmeticError('Incomplete continuum/incorrect maximum')
    lower = -3*(256*I.of(maximum).square().square()).cube_root()
    if F(result['lower']['lower']) != F(lower.lo,SCALE):
        raise ArithmeticError('Wrong radial coefficient or units')
    return dict(exact_cover=True, independent_majorant_reconstruction=True, radial_coefficient_checked=True)


def spectral_case(h: Any, t: Any) -> dict[str, Any]:
    s = (t.T*t)[0]; k = h+t*t.T
    q = sp.trace(h*h)+s*s+4*(t.T*h*t)[0]+2*s*sp.trace(h)
    identity = sp.trace(k*k)+2*s*sp.trace(k)+2*(t.T*k*t)[0]-4*s*s
    eigenvalues = [value for value, count in k.eigenvals().items() for _ in range(count)]
    d = min(eigenvalues); penalty = 2*d*(sp.trace(k)-d)
    slack = sp.simplify(q-sp.trace(k*k)-penalty)
    if sp.simplify(q-identity) != 0 or slack < 0 or h.rank() > 3:
        raise ArithmeticError('Exact spectral premise failed')
    if q != 0 and sp.trace(k)**2 > 3*q:
        raise ArithmeticError('Paid trace moment failed')
    # Independent exact matrix anticommutators, not a copied polynomial result.
    sigma = (sp.Matrix([[0,1],[1,0]]), sp.Matrix([[0,-sp.I],[sp.I,0]]), sp.diag(1,-1))
    active = [j for j in range(4) if h[j,j] != 0]
    fields = []
    for j in range(4):
        vector = sp.zeros(2) if j not in active else sp.sqrt(h[j,j])*sigma[active.index(j)]
        fields.append(t[j]*sp.eye(2)+vector)
    direct = 256*sum(sp.trace((a*b+b*a)**2) for a in fields for b in fields)
    if sp.simplify(sp.sympify(direct-2048*q)) != 0:
        raise ArithmeticError('Independent scalar/Pauli anticommutator mismatch')
    high_slack = None
    if q != 0 and 3*k[0,0]**2 >= q:
        high_slack = sp.simplify(k[0,0]+sp.sqrt(2*(q-k[0,0]**2))-sp.trace(k))
        if high_slack < 0:
            raise ArithmeticError('Exact high-u trace envelope failed')
    return dict(rank_H=int(h.rank()), rank_K=int(k.rank()), Q=str(q), independent_direct_quartic=str(direct), trace_K=str(sp.trace(k)), trace_K_square=str(sp.trace(k*k)), d=str(d), penalty=str(penalty), spectral_slack=str(slack), high_u_trace_slack=None if high_slack is None else str(high_slack), dropped_4tHt_error=str(4*(t.T*h*t)[0]))


def exact_controls(profile: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for rank in range(4):
        h = sp.diag(*([1]*rank+[0]*(4-rank))); t = sp.zeros(4,1)
        cases.append(spectral_case(h,t))
    # Rank4 with singular H, exact endpoint saturation and an interior weight.
    h = sp.diag(0,2,3,4); t = sp.Matrix([1,0,0,0])
    cases.append(spectral_case(h,t))
    cases.append(spectral_case(sp.diag(0,1,1,1),sp.Matrix([1,1,0,0])))
    cases.append(spectral_case(sp.zeros(4),sp.Matrix([2,1,-1,3])))
    cases.append(spectral_case(sp.diag(0,1,1,1),sp.Matrix([3,0,0,0])))
    nondegenerate = reviewed.algebra_control()
    if nondegenerate['wrong_scalar_drop_error'] == 0:
        raise ArithmeticError('Scalar cross-term control is not discriminating')
    rank4 = cases[4]
    if rank4['rank_K'] != 4 or F(rank4['penalty']) <= 0 or F(rank4['spectral_slack']) != 0:
        raise ArithmeticError('Rank4 endpoint penalty not exercised')
    isotropic = spectral_case(sp.diag(0,1,1,1),sp.Matrix([1,0,0,0]))
    trace, naive_q, paid_q = F(isotropic['trace_K']), F(isotropic['trace_K_square']), F(isotropic['Q'])
    if not trace*trace > 3*naive_q or not trace*trace <= 3*paid_q:
        raise ArithmeticError('Missing-penalty/rank3 mutant is not rejected')
    a0 = profile['onsite']; c0 = [row['c0'] for row in profile['paths']]
    co = (F(1),F(0),F(-1),F(0))
    x0 = [a0+2*sum(c0[j]*co[k[j]] for j in range(3)) for k in product(range(4),repeat=3)]
    y = [x*x for x in x0]; mean = sum(y)/64; variance = sum((z-mean)**2 for z in y)/64
    protected = 8*a0*a0*sum(z*z for z in c0)
    if not protected > 0 or variance < protected or variance >= 2*protected:
        raise ArithmeticError('Exact wrong-variance control failed to discriminate')
    # Scalar field K=e00: exact all64 frequency-one projection, no float FFT.
    projection = []
    for j in range(3):
        coefficient = 2*sum((yy-mean)*co[k[j]] for yy,k in zip(y,product(range(4),repeat=3)))/64
        if coefficient != 4*a0*c0[j]:
            raise ArithmeticError('Wrong onsite frequency-one coefficient')
        projection.append(coefficient)
    if sum(z*z for z in projection)/2 != protected:
        raise ArithmeticError('Bessel normalization mismatch')
    alpha, z = sp.symbols('alpha z',positive=True)
    energy = 2048*z**4-64*alpha*z
    if sp.diff(energy,z) != 8192*z**3-64*alpha or F(96**3,128) != 27*256:
        raise ArithmeticError('Radial whole-volume normalization failed')
    for value in (F(0),F(1),F(9,16)):
        iv = I.of(value).sqrt(); exact = sp.sqrt(sp.Rational(value.numerator,value.denominator))
        if not sp.Rational(iv.lo,SCALE) <= exact <= sp.Rational(iv.hi,SCALE):
            raise ArithmeticError('Outward radical control failed')
    return dict(rank0_through4_and_scalar=cases, isotropic_rank4=isotropic, nondegenerate_exact_scalar_quartic=nondegenerate, mutants=dict(forced_rank3_rejected=rank4['rank_K']>3, missing_rankpenalty_trace_rejected=trace*trace>3*naive_q, wrong_doubled_variance_rejected=variance<2*protected, wrong_zero_frequency1_rejected=protected>0, wrong_dropped_scalar_rejected=nondegenerate['wrong_scalar_drop_error']!=0), exact_variance=dict(actual=str(variance), protected=str(protected), remainder=str(variance-protected), frequency1_coefficients=[str(z) for z in projection], wrong_double_deficit=str(2*protected-variance)), exact_radial_identity='96^3/128=27*256', zero_field='Q=0 => H=0,t=0,K=0, pressure=0; no division')


def global_filling_controls(profile: dict[str, Any]) -> list[dict[str, Any]]:
    # Scalar traceful Pauli field: independent exact global-fill calculation.
    a0 = profile['onsite']; width = sum(2*abs(z['c0']) for z in profile['paths'])
    minimum = a0-width
    if minimum <= 0:
        raise ArithmeticError('Native scalar filling witness lacks strict sign')
    t = np.array([1,0,0,0],float); zero = np.zeros((4,3)); phi = reviewed.pauli_fields(t,zero,zero)
    axes = np.pi*np.array([1/7,2/9,3/11]); out = []
    for sheet in product(range(2),repeat=3):
        angles = np.vstack([axes,np.pi*np.array(sheet)-axes]); bands = []
        for b in range(2):
            _, x = reviewed.flat.old.coefficients(profile,angles[b])
            block = phi[:,2*b:2*b+2,2*b:2*b+2]
            bands.append(np.linalg.eigvalsh(np.einsum('km,mij->kij',x,block)).ravel())
        global_fill = float(np.sort(np.concatenate(bands))[-128:].sum())
        half = sum(float(np.sort(b)[-64:].sum()) for b in bands)
        exact_global = 128*a0; exact_deficit_lower = 128*minimum
        if abs(global_fill-float(exact_global)) > 1e-9 or global_fill-half < float(exact_deficit_lower)-1e-9:
            raise ArithmeticError('Actual native global filling disagrees with independent scalar proof')
        out.append(dict(sheet=list(sheet), determinant_winding=list(sheet), arbitrary_axes_radians=angles.tolist(), global_top128=global_fill, exact_global_top128=str(exact_global), wrong_block_half=half, filling_deficit=global_fill-half, exact_deficit_lower=str(exact_deficit_lower), diagnostic_only_float=True))
    return out



def consumer_adverse_controls(certificate: dict[str, Any], profile: dict[str, Any]) -> dict[str, bool]:
    variants = {}
    optimistic = dict(certificate); optimistic['cell_bounds'] = [dict(row) for row in certificate['cell_bounds']]
    optimistic['cell_bounds'][0]['pressure_upper'] = str(F(optimistic['cell_bounds'][0]['pressure_upper'])-F(1,100))
    variants['optimistic_cell'] = optimistic
    hole = dict(certificate); hole['cell_bounds'] = [dict(row) for row in certificate['cell_bounds']]
    hole['cell_bounds'][0]['hi'] = str(F(1,2*certificate['closed_cells']))
    variants['rational_cover_hole'] = hole
    wrong_units = dict(certificate); wrong_units['lower'] = dict(certificate['lower'])
    wrong_units['lower']['lower'] = str(64*F(certificate['lower']['lower']))
    variants['wrong_whole_volume_radial_units'] = wrong_units
    rejected = {}
    for label, variant in variants.items():
        try:
            verify_cells(variant,profile)
        except ArithmeticError:
            rejected[label] = True
        else:
            raise ArithmeticError('Scientific consumer accepted plausible wrong '+label)
    return rejected

def run(output: Path, cells: int) -> dict[str, Any]:
    started = time.monotonic(); captured = snapshot()
    before = {name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()}
    archive = output.with_suffix('.input_source.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in captured.items(): z.writestr(name,raw)
        z.writestr(PREFIX+'inputs.json',json.dumps(dict(cells=cells, switch='3/5', partition=[2,2], all8=list(product(range(2),repeat=3)), input_pins=PINS)))
    profile = load_profile(captured[reviewed.INPUT]); certificate = envelope(profile,cells)
    checks = verify_cells(certificate,profile); controls = exact_controls(profile); fillings = global_filling_controls(profile)
    checks['adverse_consumer_controls'] = consumer_adverse_controls(certificate,profile)
    prior = json.loads(captured['joined_phi_u1_jordan_22_typed_actual_20261010.json'])
    source = json.loads(captured[reviewed.INPUT]); central = source['exact_central_convex_residual']
    upper = F(central['upper']); lower = F(certificate['lower']['lower']); separation = I.of(lower)-I.of(upper)
    without = F(certificate['without_variance_lower']['lower'])
    after = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in captured}
    if before != after: raise ArithmeticError('Source/input changed during scientific epoch')
    result = dict(kind='RIGOROUS_JOINT_TRACEFUL_PAULI22_ALLANGLE_STATIC_NEUTRAL_CERTIFICATE', proof=PROOF, governing_plan='issue-222-induced-photon-delta13-continuation-20261008', governing_revision='0bec1a8c2e8ddf0d21d0a9f2c4b4460abc88f7b481f50f0e2d9d0218d29ab42c', work_item='induced-photon/verification', data_stage='exact outward static sector certificate; controls exact except labelled native float diagnostics', exact_frozen_native_profile=source['primitive_profiles'][0], certificate=certificate, independent_cell_consumer=checks, exact_controls=controls, all8_global128_filling=fillings, central_trial=dict(upper=str(upper), upper_display=float(upper), source_input=reviewed.INPUT, exact_declared_rational_point=central['point_diagonal'], no_root_identity=True), separation_above_central_trial=separation.public(), excluded_static_neutral_flat22=separation.lo>0, decision='EXCLUDED: allangle static neutral flat2+2 is strictly above actual central trial' if separation.lo>0 else 'OPEN: outward bounds do not separate static neutral flat2+2', prior_gap_to_central_trial=prior['gap_to_central_trial'], no_variance_separation=(I.of(without)-I.of(upper)).public(), source_input_sha256_before=before, source_input_sha256_after=after, source_inputs_unchanged=before==after, immutable_input_archive=str(archive), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), runtime=dict(python=sys.executable, version=sys.version, numpy=np.__version__, sympy=sp.__version__), elapsed_seconds=time.monotonic()-started, next_construction='Combine separately proved static sector exclusions at parent; characterize off-flat/fullPhi and correlated all64 charged/electric ground before sameground spatial photon. This certificate pays neither quantum kinetic nor charged terms.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cells',type=int,default=16384)
    args = parser.parse_args()
    with threadpool_limits(limits=1): result = run(args.output,args.cells)
    print(json.dumps(dict(kind=result['kind'], alpha=result['certificate']['alpha_display'], lower=result['certificate']['lower']['lower_display'], separation=result['separation_above_central_trial']['lower_display'], excluded=result['excluded_static_neutral_flat22'], source_inputs_unchanged=result['source_inputs_unchanged']),indent=2))
