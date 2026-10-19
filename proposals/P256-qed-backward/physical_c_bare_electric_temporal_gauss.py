#!/usr/bin/env python3
"""Literal centered endpoint Pe, temporal links and full-occupation projection.

The link endpoint gauge shift is essential.  An unshifted S Pe S path is kept
as an adverse control, not silently replaced by Pe T.  This proves a pre-Gauss
conditional field-path identity, not commutation of the solved Coulomb R with
SF, and does not claim a conditioned normalizer/source pole.
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
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import mpmath as mp
import numpy as np
from scipy.linalg import eigh
import physical_c_bare_electric_closedwalk as closed
loops=closed.loops;alpha=closed.alpha
ROOT=Path(__file__).resolve().parent;SOURCE_BYTES=Path(__file__).read_bytes()
DEPENDENCY_BYTES=closed.SOURCE_BYTES


def temporal_hopping(theta: np.ndarray,a0: np.ndarray,targets: np.ndarray) -> np.ndarray:
    nt,v=len(theta),len(targets);H=closed.hopping(theta,targets,nt);eye=np.eye(4);beta=alpha.gamma_frame()[0]
    # Transfer Q_t follows T_t.  Temporal forward row t -> t+1 has U0=Q_t†.
    for t in range(nt):
        seam=-1 if t==nt-1 else 1
        for x in range(v):
            sx=slice(4*(t*v+x),4*(t*v+x)+4);sy=slice(4*(((t+1)%nt)*v+x),4*(((t+1)%nt)*v+x)+4)
            u=np.exp(-1j*a0[t,x])
            H[sx,sy]+=seam*(u-1)*(eye-beta)
            H[sy,sx]+=seam*(u.conjugate()-1)*(eye+beta)
    return H


def state(law,b,theta: np.ndarray) -> dict:
    p=law.potential(theta,b['phi'],b['links']);e,r=eigh(p['h'])
    return dict(T=(r*np.exp(e))@r.conj().T,S=(r*np.exp(e/2))@r.conj().T,pref=p['pref'])


def determinant(P: np.ndarray) -> tuple[complex,float]:
    sign,log=np.linalg.slogdet(np.eye(len(P))+P);return complex(sign),float(log)


def endpoint_probe(b: dict,nt: int) -> dict:
    rng=np.random.default_rng(222900+nt);v=len(b['targets']);law=loops.make_law(.06,b)
    theta=rng.normal(0,.25,(nt,v,3));a0=rng.normal(0,.35,(nt,v))
    P=np.eye(4*v,dtype=complex);endpoint=P.copy();unshifted=P.copy();pref=0.;epref=0.;rows=[]
    for t in range(nt):
        s=state(law,b,theta[t]);q=np.diag(np.repeat(np.exp(1j*a0[t]),4))
        gradient=a0[t,:,None]-a0[t,b['targets']]
        shifted=state(law,b,theta[t]+gradient)
        endpoint_vertex=shifted['S']@q@s['S'];target=q@s['T']
        rows.append(dict(half_gauge_covariance_residual=float(np.linalg.norm(shifted['S']-q@s['S']@q.conj().T)),
            literal_endpoint_residual=float(np.linalg.norm(endpoint_vertex-target)),
            scalar_gauge_difference=abs(shifted['pref']-s['pref']),
            Pe_T_commutator=float(np.linalg.norm(q@s['T']-s['T']@q))))
        P=target@P;endpoint=endpoint_vertex@endpoint;unshifted=(s['S']@q@s['S'])@unshifted
        pref+=s['pref'];epref+=(s['pref']+shifted['pref'])/2
    phase=np.exp(-2j*float(np.sum(a0)))
    sign,log=determinant(P);esign,elog=determinant(endpoint);usign,ulog=determinant(unshifted)
    H=temporal_hopping(theta,a0,b['targets']);dsign,dlog=np.linalg.slogdet(np.eye(len(H))-.06*H)
    wrongH=temporal_hopping(theta,-a0,b['targets']);_,wronglog=np.linalg.slogdet(np.eye(len(H))-.06*wrongH)
    scaled=pref+log+4*v*nt*math.log(2*law.kappa)
    return dict(nt=nt,theta=theta.tolist(),A0=a0.tolist(),vertices=rows,
        actual_fullscalar_centered_Pe_logdet=scaled,Wilson_D_logdet=float(dlog),logdet_equality_error=abs(scaled-float(dlog)),
        centered_character_phase=[float((phase*sign).real),float((phase*sign).imag)],
        Wilson_D_phase=[float(dsign.real),float(dsign.imag)],
        endpoint_log_character_error=abs(epref+elog-pref-log),endpoint_phase_error=abs(esign-sign),
        unshifted_half_Pe_adverse_log_error=abs(ulog-log),unshifted_half_Pe_adverse_phase_error=abs(usign-sign),
        dropped_background_offset_phase=[float(sign.real),float(sign.imag)],
        wrong_temporal_link_orientation_log_error=abs(float(wronglog)-float(dlog)))


def compact_projection(b: dict) -> dict:
    law=loops.make_law(.06,b);rng=np.random.default_rng(222920);theta=rng.normal(0,.2,b['a'].shape)
    s=state(law,b,theta);T=s['T'];v=len(b['targets']);sign0,log0=determinant(T)
    e,r=eigh(T);occupation=e/(1+e)
    coefficients=np.array([1.])
    for p in occupation:coefficients=np.convolve(coefficients,[1-p,p])
    global_exact=float(coefficients[2*v])
    C=(r*occupation)@r.conj().T;local_probs=eigh(C[:4,:4],eigvals_only=True)
    local_coefficients=np.array([1.])
    for p in local_probs:local_coefficients=np.convolve(local_coefficients,[1-p,p])
    local_exact=float(local_coefficients[2])
    global_rows=[];local_rows=[]
    for count in (4*v+1,4*v+3):
        total=0j
        for a in 2*np.pi*np.arange(count)/count:
            sign,log=determinant(np.exp(1j*a)*T)
            total+=np.exp(-2j*v*a)*sign/sign0*np.exp(log-log0)
        total/=count;global_rows.append(dict(order=count,neutral_projection=[float(total.real),float(total.imag)],
            exact_alloccupation_grade_error=abs(total-global_exact)))
    for count in (5,7):
        total=0j
        for a in 2*np.pi*np.arange(count)/count:
            q=np.eye(4*v,dtype=complex);q[:4,:4]*=np.exp(1j*a)
            sign,log=determinant(q@T)
            total+=np.exp(-2j*a)*sign/sign0*np.exp(log-log0)
        total/=count;local_rows.append(dict(order=count,site0_charge_zero_projection=[float(total.real),float(total.imag)],
            exact_alloccupation_local_distribution_error=abs(total-local_exact)))
    return dict(global_projection=global_rows,local_projection=local_rows,exact_global_grade2V=global_exact,exact_site0_occupation2=local_exact,
        population='All original grades0..4V included in determinant before either Fourier projection. No early N restriction.',
        scope='Exact compact global charge and one-site Pe Fourier projector in fixed-background one-species fullFock. NOT real electric-divergence Gauss integral or conditioned source normalization.')


def centre_probe(b: dict) -> list[dict]:
    law=loops.make_law(.06,b);rng=np.random.default_rng(222930);theta=rng.normal(0,.15,b['a'].shape);rows=[]
    for centre in range(4):
        colour=b['links'].astype(complex);colour[0,0]*=np.exp(1j*np.pi*centre/2)
        actual=law.potential(theta,b['phi'],colour)
        shifted=theta.copy();shifted[0,0]+=np.pi*centre/2
        scalar=law.potential(shifted,b['phi'],b['links'])
        rows.append(dict(centre=centre,actual_SU4_centre_vs_Abelian_h_residual=float(np.linalg.norm(actual['h']-scalar['h'])),
            pref_residual=abs(actual['pref']-scalar['pref'])))
    return rows


def coordinate_contract() -> dict:
    """Keep the declared macro theta law distinct from native canonical A."""
    family=loops.core.full_family_definition()
    with mp.workdps(220):
        beta=16*mp.mpf('6.25e1148')*mp.mpf('1e-250')**4
        utheta=mp.mpf('.02');uA=beta*utheta
        bare_cEA=1/(4*uA);betaEA=1+1/(2*uA)
        cone=1/betaEA
        fixed_uA=mp.mpf('.02');fixed_betaEA=1+1/(2*fixed_uA)
        return dict(captured_core_family_contract=family,
            coordinate_map='A=sqrt(beta_theta)*theta, Pi_theta=sqrt(beta_theta)*Pi_A, e0=Q/sqrt(beta_theta); UA=beta_theta*Utheta. Auxiliary cB_theta=cE_theta=beta_theta/2 becomes cB_A=cE_A=1/2.',
            independently_derived_beta_theta=mp.nstr(beta,80),
            declared_macro_theta_case=dict(Utheta='0.02',UA=mp.nstr(uA,80),
                bare_cE_A=mp.nstr(bare_cEA,80),auxiliary_cE_A='0.5',betaB_A='1',
                betaE_A_exact='1+2.5e-149',cone_squared_exact='1/(1+2.5e-149)',
                cone_squared=mp.nstr(cone,190),one_minus_cone_squared=mp.nstr(1-cone,70),
                status='Declared6a macro family chart_EM: UE=.02 in physicaltheta units; this is NOT the native finite regulator input interpreted as a macro simulation.'),
            distinct_fixed_macro_canonical_case=dict(UA='0.02',Utheta=mp.nstr(fixed_uA/beta,80),
                bare_cE_A='12.5',auxiliary_cE_A='0.5',betaB_A='1',betaE_A='26',
                cone_squared_exact='1/26',cone_squared=mp.nstr(1/fixed_betaEA,80),
                status='DIFFERENT macro electric input, not declared6a physicaltheta law.'),
            native_toy_case=dict(native_UE='0.02',coordinate='native realcover/canonical A',
                status='NF0 effectivecharge finite regulator; no literal macroNaux/beta simulation and no inferred Utheta=.02 or Utheta=1.',
                source_contract=family['native_scaling']),
            temporal_Pe_coordinate='A0 in endpoint probes is the dimensionless local charge Fourier phase; it is not silently identified with the canonical photon amplitude A.',
            cone_scope='Quadratic pureAbelian fullFock preprojection response plus the DECLARED bare electric coefficient. Not an observed fully projected interacting source pole.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--capsule',required=True);args=parser.parse_args();start=time.time()
    b=loops.background((4,4,2));endpoints=[endpoint_probe(b,nt) for nt in (1,2,4)];projection=compact_projection(b);centres=centre_probe(b)
    bound=closed.macro_bound()
    proof=dict(centered_Pe='Pe_t=exp[i sum_x(N_x-2)A0_tx]; fullscalar trace=exp(sum pref_t)*exp[-2i sum_tx A0_tx]*det[I+prod_t(Q_t T_t)], Q_t=diag_x exp(iA0_tx)I4.',
        literal_endpoint='Gauge endpoint shift dA0_xy=A0_x-A0_y; S(theta+dA0)=Q S(theta) Q†. Thus S(theta+dA0) Q S(theta)=Q T(theta), including scalar halves. Unshifted S(theta)Q S(theta) is NOT substituted.',
        temporal_block_elimination='D upper equation B psi+_t+F psi-_t-Q_(t-1) psi+_(t-1)=0; lower -F†psi+_t+Bpsi-_t-Q_t†psi-_(t+1)=0. With chi_t=[psi+_t,Q_t†psi-_(t+1)], chi_t=T_t Q_(t-1) chi_(t-1). Cyclic rotation gives prod(Q_t T_t). Determinant elimination supplies det(Q)^(-1/2)=exp(-2i sum A0).',
        Wilson_links='Actual gamma Wilson D=I-kH with temporal forward U0(t,x)=exp(-iA0(t,x)), spatial Uj=exp(i theta_t,x,j), APBC at time seam.',
        macro_extension='The actual Pe/D identity extends closedwalk action and Hessian remainder to arbitrary four-dimensional U1 links before projection. Each spin-projector directional hopping norm is2, independent of temporal phases; ||H||<=8. Gamma5 Hermiticity plus8k<1 implies positive real determinant by continuity from k=0.',
        preprojection_remainder='For all theta,A0 and scalar centre holonomies, log(detD(A)/detD(0)) times Naux differs from16Naux*k^4 sum(cosF-1) by <=delta=Naux*V4*(4/3)*(8k)^6/(1-(8k)^2). No occupation grade is omitted.',
        integration_bound='For any normalized compact Fourier measure with unit-modulus charge character, absolute projected normalized-weight error<=exp(delta)-1: leading plaquette weight<=1. A finite coherent centre sum adds sum(abs(weights)); no centre/occupation restriction is taken before character/projection.',
        real_Gauss_limit='Pointwise uniform bound survives before realGauss projection. The noncompact real Fourier delta limit requires its actual electric heat measure/dominating integral. No finite total-variation norm, positive conditioned denominator, relative projected error or source pole is inferred.',
        scalar_R_scope='At a conditional field path the unprojected electric kernel is a scalar Fock weight. The already solved core Coulomb diagonal is charge-dependent and need not commute with SF; this diagnostic does NOT commute it or equate its compressed native matrix to an unprojected determinant.',
        centre_scope='All4 literal scalar SU4 centre labels tested against Abelian phase absorption, with full C/pref. Noncentral SU4 and Phi remain separate debts.')
    passed=all(x['logdet_equality_error']<1e-7 and x['endpoint_log_character_error']<1e-7 and abs(complex(*x['centered_character_phase'])-1)<1e-7 for x in endpoints)
    passed=passed and all(row['exact_alloccupation_grade_error']<1e-9 for row in projection['global_projection'])
    passed=passed and all(row['exact_alloccupation_local_distribution_error']<1e-9 for row in projection['local_projection'])
    passed=passed and all(row['actual_SU4_centre_vs_Abelian_h_residual']<1e-8 and row['pref_residual']<1e-8 for row in centres)
    report=dict(proof=proof,endpoint_probes=endpoints,compact_projection=projection,centre_labels=centres,macro_bound_reused=bound,
        coordinate_contract=coordinate_contract(),
        extended_scope='Pure Abelian arbitrary spatial AND temporal link phases, scalar centre holonomies, fullFock before local/global projection. Finite compact charge projections explicitly exercised; actual noncompact realGauss normalization/source NOT proved.',checks_passed=passed)
    execution=dict(source_sha256=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        closedwalk_sha256=hashlib.sha256(DEPENDENCY_BYTES).hexdigest(),closedwalk_bytes_at_IMPORT=DEPENDENCY_BYTES.decode(),
        loops_bytes_at_IMPORT=loops.SOURCE_BYTES.decode(),core_sha256=hashlib.sha256(loops.CORE_BYTES).hexdigest(),core_bytes_at_IMPORT=loops.CORE_BYTES.decode(),
        frozen_dependencies=loops.IMPORT_EPOCH,argv=sys.argv,python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
        thread_environment={name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},elapsed_seconds=time.time()-start)
    Path(args.capsule).write_text(json.dumps(dict(report=report,execution=execution),indent=2)+'\n')
    print(json.dumps(dict(capsule=args.capsule,passed=passed,endpoints=endpoints,projection=projection,centres=centres),indent=2),flush=True)
    if not passed:raise ArithmeticError('Literal centered Pe/temporal D or projection map failed; macro extension not admitted')


if __name__=='__main__':main()
