"""NEW Jordan 2+2 all-angle pressure envelope; not a fullPhi ground calculation."""
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

import joined_phi_u1_matrix_jordan_flat_bounds as flat

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_jordan_22_'
I = flat.I
SCALE = flat.old.SCALE
INPUT = 'joined_phi_u1_matrix_jordan_flat_actual_20261010_receipt.json'
INPUT_PIN = 'd3d0360739b6d1d12ac013bac65aafcb085004fa76f7d379445cbe8548d18e24'
PROOF = {
    'scope': 'NEW independent Jordan law, flat partition2+2 only; arbitrary traceful Hermitian2 blocks, global tracezero, global top128. No SOURCE/current427/root adoption.',
    'exact_scalar_vector': 'Phi_b,mu=t_b,mu I2+v_b,mu.sigma; t2=-t1 by global tracezero, H_b=V_b V_b^T PSD rank<=3. {Phi_mu,Phi_nu}=2[(t_mu t_nu+H_mu,nu)I+(t_mu v_nu+t_nu v_mu).sigma]. Summing its trace-square gives q_b=2048 Q_b, Q_b=||H_b||F^2+s_b^2+4t_b^T H_b t_b+2s_b TrH_b. The scalar-vector cross terms are retained: 4tHt is a squared norm, NOT zero by trace. There is no unpriced odd cubic term in this exact Jordan polynomial; the displayed anticommutator expansion proves this, rather than separately imposing block tracelessness.',
    'pressure': 'At each native AP momentum x, h_b=(x.t_b)I+(sum x_mu v_b,mu).sigma has bands w+-R, R^2=xH_bx. Globally Trh=64 A sum_b2t_b,0=0, hence S128<=Tr|h|/2=sum_b,k max(|w|,R)<=sum_b,k sqrt(xK_bx), K_b=H_b+t_b t_b^T. This is only a lower-energy relaxation; no block128/2 occupation is adopted. Separate H1,H2 remain throughout.',
    'quadratic_control': 'Let r_b=sqrt(Q_b). ||K_b||F^2=||H_b||F^2+s_b^2+2tHt<=Q_b, since Q_b-||K_b||F^2=2tHt+2s TrH>=0. C=mean64 xx^T is exactly angle independent. RankH<=3 and PSD imply m=TrCK<=sqrt(D3)r with D3=sum top3 eigenvalues(C)^2: (TrCH)^2<=D3||H||F^2, tCt<=Cmax s, TrCH<=Cmax TrH, Cmax^2<=D3. For u=K00/r in[0,1], m/r<=C00 u+sqrt(sum spatial Cii^2)sqrt(1-u^2). No common H, central SPD or rank4 assumption.',
    'protected_variance': 'Write x0=A+sum_j2c0_j cos p_j, xj=-2ci_j sin p_j and y=xKx. For ALL continuous shifts on four samples per axis, the orthogonal frequency1 projections are 4A c0_j K00 cos p_j and -4A ci_j K0j sin p_j. All degree2 trigonometric terms are orthogonal to these frequency1 axes, including the shifted frequency2 alias. Bessel therefore gives Var64(y)>=8A^2(sum c0_j^2)K00^2, without discarding the rest. Odd/cubic mixed contributions in the fourth moment are not asserted zero; they lie in the retained orthogonal remainder norm. Angles/determinant/AP do not alter the identity.',
    'failed_route': 'Pure rank-moment Jensen loses the protected onsite frequency1 variance and leaves the incoming41.65020178 gap. A curvature-at-support bound sqrt(y)<=sqrt(m)+(y-m)/(2sqrt(m))-(y-m)^2/(8M^(3/2)) tightens it only weakly. Both failed closure routes are retained numerically, not used as exclusion.',
    'distinct_defect': 'Materially different exact algebraic defect: sqrt(y)=sqrt(m)+(y-m)/(2sqrt(m))-(y-m)^2/[2sqrt(m)(sqrt(y)+sqrt(m))^2]. Average it and use y<=Lr and m<=sqrt(D3)r. Then mean sqrt(y)<=sqrt(r)[sqrt(m/r)-d u^2], d=8A^2sum c0^2/[2 D3^(1/4)(sqrt(L)+D3^(1/4))^2]. m=0 implies K=0 since C>0 and is handled directly; no division at zero.',
    'support': 'For equal native |c0_j| and |ci_j|, a=2|c0|, b=4ci^2, b>3a^2. ||x||^2=A^2+3b+2Aa sum z_j-z^T(bI-a^2 11^T)z, |z_j|<=1. Completing the positive quadratic square bounds it by L=A^2+3b+3(Aa)^2/(b-3a^2). This is an unconstrained upper bound, not a grid maximum.',
    'compact_certificate': 'Cover every u in[0,1] by closed rational cells. On[lo,hi], pressure coefficient<=sqrt(min(sqrt(D3),C00 hi+sqrt(sum Csp^2)sqrt(1-lo^2)))-d lo^2. All radicals use110-bit outward integer arithmetic. The maximum cell upper alpha is a rigorous continuum bound. Each block energy>=2048r^2-64alpha sqrt(r), minimized exactly; summing2 blocks gives E/Lambda>=-3(256alpha^4)^(1/3). Cell adjacency/coverage and all upper bounds are archived. This is not sampled angle or optimizer absence proof.',
    'determinant': 'theta2,j=pi n_j-theta1,j, n_j in{0,1}; theta1 arbitrary continuous on each axis. All Z2^3 components including101 remain. The proof relaxes determinant coupling but never suppresses a component. Joint character degeneracies and rank-deficient fields remain included.',
    'limits': 'Only bounded static neutral flat2+2 bracket/decision. Charged1+63/fullHC/Pi, off-flat/fullPhi quantum ground, same-ground photon pole/cone and NEW-law adoption remain OPEN. A negative lower-vs-central-upper difference is unresolved eligibility, not a lower counterexample.'
}


def snapshot() -> dict[str, bytes]:
    result = dict(flat.field.CAPTURED)
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).parent == ROOT:
            result[Path(path).name] = Path(path).read_bytes()
    result[INPUT] = (ROOT/INPUT).read_bytes()
    for name, pin in flat.sea.INPUTS.items():
        raw = (ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ArithmeticError('Frozen input changed: '+name)
        result[name] = raw
    if hashlib.sha256(result[INPUT]).hexdigest() != INPUT_PIN:
        raise ArithmeticError('Pinned prior exact receipt changed')
    return result


def envelope(profile: dict[str, Any], cells: int, route: str) -> dict[str, Any]:
    a0 = profile['onsite']
    paths = profile['paths']
    c0 = [z['c0'] for z in paths]
    ci = [z['ci'] for z in paths]
    if len(set(abs(z) for z in c0)) != 1 or len(set(abs(z) for z in ci)) != 1:
        raise ArithmeticError('Equal-axis support formula not applicable')
    c = [a0*a0+2*sum(z*z for z in c0), *[2*z*z for z in ci]]
    d3 = sum(z*z for z in sorted(c, reverse=True)[:3])
    spatial = sum(z*z for z in c[1:])
    a, b = 2*abs(c0[0]), 4*ci[0]**2
    if b <= 3*a*a:
        raise ArithmeticError('Support quadratic not positive')
    support = a0*a0+3*b+3*(a0*a)**2/(b-3*a*a)
    crude = (abs(a0)+sum(2*abs(z) for z in c0))**2+sum(4*z*z for z in ci)
    variance = 8*a0*a0*sum(z*z for z in c0)
    fourth = I.of(d3).sqrt().sqrt()
    if route == 'moment':
        defect = I.of(0)
    elif route == 'curvature':
        defect = I.of(variance)/(8*I.of(crude)*I.of(crude).sqrt())
    else:
        defect = I.of(variance)/(2*fourth*(I.of(support).sqrt()+fourth).square())
    records = []
    for index in range(cells):
        lo, hi = F(index, cells), F(index+1, cells)
        upper_m = I.of(c[0])*hi+I.of(spatial).sqrt()*I.of(1-lo*lo).sqrt()
        capped = I(0, min(upper_m.hi, I.of(d3).sqrt().hi))
        upper = capped.sqrt()-defect*(lo*lo)
        records.append(dict(index=index, lo=str(lo), hi=str(hi), pressure_upper=str(F(upper.hi, SCALE))))
    alpha = max(F(z['pressure_upper']) for z in records)
    radial = -3*(256*I.of(alpha).square().square()).cube_root()
    return dict(route=route, C_exact=[str(z) for z in c], D3_exact=str(d3), support_exact=str(support), crude_support_exact=str(crude), protected_variance_exact=str(variance), defect=defect.public(), alpha_upper=str(alpha), lower=radial.public(), closed_cells=cells, complete_cover='index0 lo0, last hi1, adjacent exact equal, all closed', cell_bounds=records)


def pauli_fields(t: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    sigma = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]], complex)
    out = np.zeros((4,4,4), complex)
    out[:,:2,:2] = t[:,None,None]*np.eye(2)+np.einsum('ma,aij->mij',v1,sigma)
    out[:,2:,2:] = -t[:,None,None]*np.eye(2)+np.einsum('ma,aij->mij',v2,sigma)
    return out


def algebra_control() -> dict[str, Any]:
    t = np.array([3, -2, 1, 2],float)/16
    v1 = np.array([[4,1,-2],[1,3,1],[2,-1,3],[-1,2,1]],float)/16
    v2 = np.array([[-2,3,1],[3,1,-1],[1,2,4],[2,-2,1]],float)/16
    phi = pauli_fields(t,v1,v2)
    matrices = [sp.Matrix([[sp.Rational(float(z.real))+sp.I*sp.Rational(float(z.imag)) for z in row] for row in m]) for m in phi]
    exact = 256*sum(sp.trace((m*n+n*m)**2).expand() for m in matrices for n in matrices)
    formula = sp.Rational(0)
    contributions = []
    for v in (v1,v2):
        vs=sp.Matrix([[sp.Rational(float(z)) for z in row] for row in v]);ts=sp.Matrix([sp.Rational(float(z)) for z in t]);h=vs*vs.T;s=(ts.T*ts)[0]
        mixed=4*(ts.T*h*ts)[0]
        q=2048*(sp.trace(h*h)+s*s+mixed+2*s*sp.trace(h));formula+=q
        contributions.append(dict(rank=int(h.rank()), scalar_square=str(s*s), cross_4tHt=str(mixed), scalar_trace_cross=str(2*s*sp.trace(h)), quartic=str(q)))
    if sp.simplify(exact-formula) != 0:
        raise ArithmeticError('Exact scalar/Pauli quartic mismatch')
    stripped=pauli_fields(np.zeros(4),v1,v2)
    wrong=flat.quartics(stripped)[0]
    return dict(phi_real=np.real(phi).tolist(),phi_imag=np.imag(phi).tolist(), exact_quartic=str(exact),formula_exact=str(formula),blocks=contributions,wrong_separately_traceless_quartic=wrong,wrong_scalar_drop_error=float(exact)-wrong,oldGram_wrong_quartic=flat.quartics(phi)[1],exact_global_trace=[str(sp.trace(m)) for m in matrices])


def run(output: Path, cells: int) -> dict[str, Any]:
    started=time.monotonic(); captured=snapshot()
    before={name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()}
    archive=output.with_suffix('.input_source.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in captured.items():z.writestr(name,raw)
        z.writestr(PREFIX+'run_input.json',json.dumps(dict(cells=cells,route='exact_defect',partition=[2,2],all_components=[list(z) for z in product(range(2),repeat=3)],arbitrary_continuous_axes=True)))
    data=flat.field.load_mean(g_override=[1,1,1,1],mean_descriptor={'scope':'NEW Jordan22 unit control frame, no mean/root solver'})
    profile=data['profiles'][0];native=data['native']
    routes=[envelope(profile,cells,route) for route in ('moment','curvature','exact_defect')]
    source=json.loads(captured[INPUT]); central=source['exact_central_convex_residual']
    prior=next(z for z in source['exact_rational_trial_certificates'] if z['partition']==[2,2])
    phi=np.asarray(prior['field_real'])+1j*np.asarray(prior['field_imag'])
    trial=flat.certify_trial(profile,(2,2),phi,[F(z) for z in prior['angles_pi'][0]],tuple(prior['determinant_winding']))
    algebra=algebra_control();adverse=np.asarray(algebra['phi_real'])+1j*np.asarray(algebra['phi_imag'])
    # Arbitrary independent scalar/Pauli fields and nonidentical irrational-looking axes.
    axes=np.pi*np.array([1/7,2/9,3/11]);components=[]
    for sheet in product(range(2),repeat=3):
        angles=np.vstack([axes,np.pi*np.array(sheet)-axes]);bands=[];moments=[]
        for b,theta in enumerate(angles):
            _,x=flat.old.coefficients(profile,theta)
            block=adverse[:,2*b:2*b+2,2*b:2*b+2]
            eig=np.linalg.eigvalsh(np.einsum('km,mij->kij',x,block));bands.append(eig.ravel())
            t=np.trace(block,axis1=1,axis2=2).real/2;vv=block-t[:,None,None]*np.eye(2)
            h=np.einsum('mij,nji->mn',vv,vv).real/2;k=h+np.outer(t,t)
            y=np.einsum('km,mn,kn->k',x,k,x);variance=float(np.var(y));protected=float(F(routes[-1]['protected_variance_exact']))*k[0,0]**2
            if variance+1e-9<protected:raise ArithmeticError('Protected harmonic variance control failed')
            moments.append(dict(variance=variance,protected=protected,remainder=variance-protected,wrong_zero_odd_onsite_projection=0.,nonzero_onsite_projection_norm=protected))
        allbands=np.concatenate(bands);global128=float(np.sort(allbands)[-128:].sum());fixed=sum(float(np.sort(z)[-64:].sum()) for z in bands)
        components.append(dict(sheet=list(sheet),angles_radians=angles.tolist(),determinant_winding=(angles.sum(axis=0)/np.pi).tolist(),global_top128=global128,wrong_block_half=fixed,filling_deficit=global128-fixed,moments=moments))
    angles=np.vstack([axes,np.pi*np.array([1,0,1])-axes])
    native_controls=[flat.native_control(p,native,adverse,(2,2),angles) for p in data['profiles']]
    if max(z['filling_deficit'] for z in components)<=1e-6:
        raise ArithmeticError('Wrong block-half filling control not discriminating')
    # Rank3 saturation is admissible despite noncentral characters; never use SPD Schur to exclude it.
    rankv=np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]],float)/4
    rankphi=pauli_fields(np.zeros(4),rankv,rankv)
    rj,rg,re=flat.quartics(rankphi)
    if abs(re)>1e-12:raise ArithmeticError('Rank3 Jordan saturation control failed')
    lower=routes[-1]['lower']; upper=trial['new_static_energy_divided_Lambda'];central_upper=F(central['upper'])
    improved=I.of(F(lower['lower']))-I.of(F(routes[0]['lower']['lower']))
    separation=I.of(F(lower['lower']))-I.of(central_upper)
    after={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in captured}
    if before!=after:raise ArithmeticError('Source/input changed within scientific epoch')
    result=dict(kind='RIGOROUS_NEW_JORDAN_22_ALLANGLE_VARIANCE_PRESSURE_TIGHTENING_NOT_GROUND',proof=PROOF,governing_plan='issue-222-induced-photon-delta13-continuation-20261008',governing_revision='55352a0aabbcc1da539641239f48f2807ec1e2a609b5d533ca0f186d8df27b6e',work_item='slice22',routes=routes,sector_bracket=dict(lower=lower,upper=upper,width=(I.of(F(upper['upper']))-I.of(F(lower['lower']))).public()),lower_improvement=improved.public(),gap_to_central_trial=(-separation).public(),excluded=separation.lo>0,decision='2+2 remains eligible: materially tighter all-angle lower, NOT an outward lower counterexample',rational_trial=trial,exact_scalar_quartic_control=algebra,all8_continuous_angle_controls=components,actual_native_all3_roles=native_controls,rank3_noncentral_saturation=dict(rank=3,quartic_Jordan=rj,quartic_Gram=rg,residual=re,commuting_noncentral_holonomies_allowed=True),source_input_sha256_before=before,source_input_sha256_after=after,source_inputs_unchanged=before==after,immutable_archive=str(archive),immutable_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),python=sys.executable,elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cells',type=int,default=16384);args=parser.parse_args()
    if args.cells<1:raise ValueError('Positive exact-cover cell count required')
    with threadpool_limits(limits=1):result=run(args.output,args.cells)
    print(json.dumps(dict(kind=result['kind'],lower=result['sector_bracket']['lower']['lower_display'],gap=result['gap_to_central_trial']['upper_display'],improvement=result['lower_improvement']['lower_display'],excluded=result['excluded'],source_inputs_unchanged=result['source_inputs_unchanged'],elapsed_seconds=result['elapsed_seconds']),indent=2))
