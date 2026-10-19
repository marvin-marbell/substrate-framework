"""NEW N4 mixed ROW receiver moments and finite-clock Sobolev transport.

Exact identities and rational upper coefficients are conditional on the pinned
meanPhi/Clifford body. Floating Slater/Fourier signals are NOT interval lowers.
No ground, pole, cone, original SOURCE or free-band propagation is claimed.
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

import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as current
import joined_phi_u1_finite_current_kernel as prices
import joined_phi_u1_nonlocal_receiver_electric_control as electric
from joined_phi_u1_nonlocal_body_transport_controls import dense_control

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-induced-photon-delta13-continuation-20261008'
REVISION = '027e45d965834c9861190f672dd346fc3387dd8d4a6a73e0b5f9346e3de817f9'
G = F(64)
DERIVATION = {
 'kernel': 'C_x0(t)=<Ex UB(t)psi,US(t)E0psi>, Kx=HSEx-ExHB, Lx=HSKx-KxHB; Cprime=-i<Kx UBpsi,US E0psi>, Cdoubleprime=-<Lx UBpsi,US E0psi>. Raw E0 norm squared .5, background norm1. No normalization change.',
 'receiver': 'Ex=exp(iq.Theta)*.5 sum4 c_xalpha dagger R0alpha. Ref4 to Ref3, delta=ex-e0=Dk (ROW minus COL), q=Q^T k. Unreduced full Wilson is qTheta+delta lambda_C; lambda_C=Lplus D A minus ANY anchor. Neutral delta removes anchor. Reduced CAR already contains longitudinal dressing. Ex norm <=1 on Ref4 background, since ExdagEx=(4-n_x)/4.',
 'initial': 'B_x0=.25 sum_alpha c_xalpha c_0alpha dagger, b=.25Tr(1-P)_x0. m1=<Kxpsi,E0psi>. Pi term: exp(-iqTheta)[q²-i qTheta/S] b integrates to0 by Gaussian IBP, S2. This is cancellation, not omission of Pi dynamics. Matter term is .25 E exp(-iqTheta) Tr sum_j h_xj(Theta)(1-P)_j0 (actual ROW x).',
 'coulomb': 'KHC=Ex Vdelta, Vdelta=128 deltaLrho+64deltaLdelta; rho is background all64 charged plus static Ref4 offsets. <rho_v B_x0>=meanrho_v b-.25Tr(P_xv(1-P)_v0). Thus mHC=64 exp(-S q²/2)[(2deltaLmeanrho+deltaLdelta)b-.5 sum_v(deltaL)_v Tr(P_xv(1-P)_v0)]. Auxiliary connected contractions vanish ONLY for this initial active transition; their full mean and future HC/Sobolev effects remain. Not the 59/122 energy increment or64/128 bare selfcoefficient.',
 'blind_exact': 'Fixed translational mean h(k)=sum_mu real d_mu(k) gamma_mu, h(k)^2 scalar, signh(k)=sum real d_mu gamma_mu/|d|. Orthogonal Clifford traces vanish. Hence TrP_x0=0 offsite, C0=0 exactly. In Tr(P_xv(1-P)_v0), nonzero offsite products of two gamma vectors have trace symmetric under v->x-v; deltaL changes sign under that reflection, while endpoint single-gamma traces vanish. Thus full initial HC contraction is0 exactly (aux meanrho0). Pi and HC initial omission controls are blind and must remain labeled blind. They are discriminated at finite time in the separate actual same-H dense CAR/Hermite consumer, not inferred from m1.',
 'blind_row_column': 'The N4 ROW formula is correct even where the wrong column formula happens to agree. Translation/AP covariance makes the Gaussian weighted h a twisted convolution; spin trace of the convolution commutator [h_weighted,1-P]_x0 vanishes termwise by Tr(gamma_mu gamma_nu)=Tr(gamma_nu gamma_mu). Thus this initial scalar trace is exactly blind to row versus column. Removing static Ref shifts meanrho bydelta0 times C0=0 and is likewise initially blind. The random nontranslation dense SAME-H control discriminates both; no claim of discrimination on blind N4 moments.',
 'domain': 'On realR129, finite fixed charged sectors: HC positive bounded matrix, M smooth bounded Fourier multiplier. H=64Pi²+HC+M selfadjoint on D(Pi²); Gaussian and Ex-inserted states lie on Schwartz core. Neutral heavy commutes Ex,Pi,HC and factors out EXACTLY. No neutral Lambda in conserved body energies.',
 'conservation': 'E=64Pi²+HC; E terms commute and are positive. E1=<E>0+2m norm², E2=||64Pi²psi||+||HCpsi||+2m norm bound <E>t and ||E UBpsi||. Therefore P=||grad||<=sqrt(E1/64), D=||Delta||<=E2/64, ||HC^.5 grad||<=E2/(2sqrt64). These are conserved state-weighted estimates, NOT operatornormPi or future Wick/Slater propagation.',
 'second_price': 'Lx decomposes TT,2TC,CC,TM+MT,CM+MC,MM. TT<=g²(q4+4q³P)+4gq²E2. CC<=4gdE2+4(gd)^1.5 sqrtE1+(gd)². 2TC<=2g[q²v1(d)+2|q|(sqrt(d)E2+gdP)], v1(d)=2sqrt(gdE1)+gd. MM<=hmax_active² by same-flavour creation commutator, NOT extensive M². Incoming active creation chains j,x have coefficient b, f=q+q_jx, delta_j=e_j-e0; TM+MT chain price b*g[(q²+|f|²)+2|2q+q_jx|P], CM+MC chain b[v1(d_x)+v1(d_j)]. Contact products from [M,Pi] and [M,rho] sum genuinely ALL64 charged hop coefficients: 2g sum_h 4b_h load |q dot q_h| and 2g sum_h 4b_h load |deltaLdelta_h|. Onsite active chain included, aux HC never dropped.',
 'remainder': '|C(t)-C0+it m1|<=t²/2 Lstar ||E0psi||. Uniform bound on Lx UB(t)psi uses the same conserved background E1,E2, not old K0. Kx price independently retained. Any finiteclock lower relative to observed m1 uses explicit1percent floating allowance; no interval signal certification.',
 'transfer': 'If actual NEW uniform source receipt establishes ||HT-T(Hb+Ephi)||<=epsilon and receiver endpoint defect<=eta, ordinary bounded-receiver Duhamel gives |Cfull-Cbody|<=eta*sqrt(.5)+2|t|epsilon*sqrt(.5) (Ex norm1, E0 norm sqrt(.5)); endpoint eta assumed on evolved body vectors, not merely initial state. This lane does not bind an absent receipt or use unbounded-current graph proof.'}


def sqrt_upper(x: F, bits: int = 40) -> F:
    """Outward dyadic rational square root, checked by integer arithmetic."""
    if x < 0:
        raise ArithmeticError('Negative squared norm')
    scale = 1 << bits
    n = math.isqrt((x.numerator*scale*scale)//x.denominator)
    if F(n*n, scale*scale) < x:
        n += 1
    answer = F(n, scale)
    if answer*answer < x:
        raise ArithmeticError('Outward rational root failed')
    return answer


def rational_energies(geom: dict[str, Any], species: list[body.Species]) -> dict[str, F]:
    tr, tr2, diag = (F(geom[key]) for key in ('exact_L_trace','exact_L2_trace','exact_L_diagonal'))
    particles = 8192
    # Initial fixed-number cumulant bound from the full charged density proof;
    # background meanrho=0, so third-cumulant/mean terms are absent.
    fourth = 3*((128*tr)**2+2*128**2*tr2)+78*particles*diag**2
    hcmean = G*128*tr
    hcnorm = G*sqrt_upper(fourth/3)
    matter = 128*species[0].hmax_exact+8064*species[1].hmax_exact
    e1 = G*F(129,8)+hcmean+2*matter
    e2 = G*sqrt_upper(F(129*131))/8+hcnorm+2*matter
    return dict(E1=e1,E2=e2,P=sqrt_upper(e1/G),rootE1=sqrt_upper(e1),matter=matter,
                HCmean=hcmean,HCnorm=hcnorm,charged_particles=F(particles))


def rational_caps(r: int, geo: dict[str, Any], geom: dict[str, Any], sp: body.Species,
                  species: list[body.Species], energy: dict[str,F]) -> dict[str,F]:
    den = int(geom['denominator']); lint = geom['L_integer']
    k = geo['k'].astype(np.int64); delta = geo['delta'].astype(np.int64)
    dr = delta[r]; kr = k[r]; x = int(geo['x'][r])
    def density(a: np.ndarray, b: np.ndarray) -> F:
        return F(int(a@lint@b),den)
    def gram(a: np.ndarray, b: np.ndarray, da: np.ndarray, db: np.ndarray) -> F:
        return F(int(a@b))-density(da,db)
    q2 = gram(kr,kr,dr,dr); d = density(dr,dr); q = sqrt_upper(q2)
    p,e2,re = energy['P'],energy['E2'],energy['rootE1']
    def v1(dd: F) -> F:
        return 2*sqrt_upper(G*dd)*re+G*dd
    terms = dict(TT=G**2*(q2**2+4*q*q2*p)+4*G*q2*e2,
                 TC=2*G*(q2*v1(d)+2*q*(sqrt_upper(d)*e2+G*d*p)),
                 CC=4*G*d*e2+4*G*d*sqrt_upper(G*d)*re+(G*d)**2,
                 TM=F(0),CM=F(0),MM=sp.hmax_exact**2)
    # Incoming active h_jx paths, which create at j. Geometry is not col0.
    lookup = {(int(z['axis']),int(z['length'])):abs(F(z['c0']))+abs(F(z['ci'])) for z in sp.profile['paths']}
    native_paths = body.joined.old.native_paths(sp.profile, _NATIVE)
    op = [lookup[(int(z['axis']),int(z['length']))] for z in native_paths]
    op = op+op
    for h in np.flatnonzero(geo['y']==x):
        dd = np.eye(64,dtype=np.int64)[int(geo['x'][h])]-np.eye(64,dtype=np.int64)[0]
        f2 = gram(kr+k[h],kr+k[h],dr+delta[h],dr+delta[h])
        fdouble = gram(2*kr+k[h],2*kr+k[h],2*dr+delta[h],2*dr+delta[h])
        terms['TM'] += op[h]*G*(q2+f2+2*sqrt_upper(fdouble)*p)
        terms['CM'] += op[h]*(v1(d)+v1(density(dd,dd)))
    onsite = abs(F(sp.profile['onsite']))
    terms['TM'] += onsite*G*(2*q2+4*q*p)
    terms['CM'] += 2*onsite*v1(d)
    for flavour,load in zip(species,(1,63)):
        lookupf = {(int(z['axis']),int(z['length'])):abs(F(z['c0']))+abs(F(z['ci'])) for z in flavour.profile['paths']}
        paths = body.joined.old.native_paths(flavour.profile,_NATIVE)
        coeff = [lookupf[(int(z['axis']),int(z['length']))] for z in paths]*2
        for h,b in enumerate(coeff):
            terms['TM'] += 2*G*4*b*load*abs(gram(kr,k[h],dr,delta[h]))
            terms['CM'] += 2*G*4*b*load*abs(density(dr,delta[h]))
    km = sp.hmax_exact
    terms['K_Pi'] = G*(q2+2*q*p)
    terms['K_HC'] = v1(d)
    terms['K_matter'] = km
    terms['Kstar'] = terms['K_Pi']+terms['K_HC']+km
    terms['Lstar'] = sum((terms[z] for z in ('TT','TC','CC','TM','CM','MM')),F(0))
    terms['quadratic_cap'] = terms['Lstar']*sqrt_upper(F(1,2))/2
    return terms


def moments(r: int, geo: dict[str,Any], sp: body.Species, aux: body.Species,
            native: dict[str,Any]) -> dict[str,Any]:
    x = int(geo['x'][r]); q = geo['q'][r]; delta = geo['delta'][r]
    empty = np.eye(256)-sp.p
    pb = sp.p.reshape(64,4,64,4).transpose(0,2,1,3)
    eb = empty.reshape(64,4,64,4).transpose(0,2,1,3)
    b = .25*np.trace(eb[x,0]); phase = np.exp(-float(q@q))
    density = np.einsum('xixi->x',sp.p.reshape(64,4,64,4)).real
    density += 63*np.einsum('xixi->x',aux.p.reshape(64,4,64,4)).real-128
    cov = np.einsum('vij,vji->v',pb[x],eb[:,0])
    dl = delta@native['Lplus']; d = float(dl@delta)
    hcself = 64*phase*d*b
    hccross = 64*phase*(2*(dl@density)*b-.5*(dl@cov))
    matter = 0j; wrongcol = 0j; omitsecond = 0j
    blocks = current.blocks(sp)
    for h in np.flatnonzero(geo['x']==x):
        j = int(geo['y'][h]); attenuation = np.exp(-np.sum((geo['q'][h]-q)**2))
        value = .25*attenuation*np.trace(blocks[h]@eb[j,0])
        matter += value
        if np.dot(geo['k'][h],geo['k'][h])==1:omitsecond += value
    onsite = .25*phase*np.trace(sp.onsite@eb[x,0]); matter += onsite;omitsecond += onsite
    for h in np.flatnonzero(geo['y']==0):
        j=int(geo['x'][h]);wrongcol += .25*np.exp(-np.sum((geo['q'][h]-q)**2))*np.trace(eb[x,j]@blocks[h])
    wrongcol += .25*phase*np.trace(eb[x,0]@sp.onsite)
    # Bra Pi acts on the initial Gaussian: the two nonzero constituents cancel.
    pibare = 64*float(q@q)*phase*b;pi_ibp = -pibare
    m1 = matter+hcself+hccross+pibare+pi_ibp
    return dict(path=int(r),row=x,col=0,length=int(geo['k'][r]@geo['k'][r]),
                C0=complex(phase*b),Pi_self=complex(pibare),Pi_IBP=complex(pi_ibp),Pi_total=0j,
                HC_self=complex(hcself),HC_covariance_and_allcharged_mean=complex(hccross),
                HC_total=complex(hcself+hccross),matter=complex(matter),m1=complex(m1),
                wrong_old_column_formula=complex(wrongcol),row_column_observed_contrast=float(abs(matter-wrongcol)),
                omitted_second_link_matter=complex(omitsecond),second_link_observed_contrast=float(abs(matter-omitsecond)),
                initial_C0_Pi_HC_omissions_exactly_blind=True,
                initial_adverse_controls=dict(
                    no_Pi=dict(m1=complex(m1),discriminates=False,reason='Exact initial Gaussian IBP'),
                    no_HC=dict(m1=complex(matter),discriminates=False,reason='Exact initial Clifford/reflection spintrace cancellation'),
                    bare_Coulomb_self_shortcut=dict(m1=complex(matter+hcself),discriminates=False,reason='C0=0 makes self-only shortcut blind; full covariance still computed'),
                    omitted_Ref=dict(discriminates=False,reason='Initial extra meanrho times C0=0; finite dense control discriminates'),
                    wrong_column=dict(m1=complex(wrongcol),discriminates=False,reason='Initial translation/AP and Clifford scalar trace; independent random dense control discriminates'),
                    omitted_second_link=dict(m1=complex(omitsecond),discriminates=True,observed_contrast=float(abs(matter-omitsecond)))),
                background_allcharged_mean_defect=float(np.max(abs(density))))


_NATIVE: dict[str,Any] = {}


def run(source_receipt: Path | None = None) -> dict[str,Any]:
    global _NATIVE
    start=time.monotonic();own=Path(__file__).read_bytes()
    data,species,inputs,_=body.full_inputs();_NATIVE=data['native']
    geo=current.geometry(data,species[0]);geom=prices.exact_geometry(_NATIVE,geo)
    energy=rational_energies(geom,species)
    rows=[];caps=[]
    for r in np.flatnonzero(geo['y']==0):
        rows.append(moments(int(r),geo,species[0],species[1],_NATIVE))
        caps.append(rational_caps(int(r),geo,geom,species[0],species,energy))
    if len(rows)!=12 or {z['length'] for z in rows}!={1,2}:raise ArithmeticError('Actual12 receivers absent')
    for z in rows:
        body.require(abs(z['C0']),1e-10);body.require(abs(z['HC_total']),1e-8)
        if abs(z['m1'].real)<1e-5:raise ArithmeticError('Mixed ROW signal vanished')
    colocal=complex(species[0].mean_source_derivative(2))
    allowance=F(99,100)
    # Explicit dyadic clock selected only from measured first moments and exact
    # rational caps; no state/receiver/H retuning. Finite signal remains floating.
    allowed=min(.125*min(abs(z['m1'].real),abs(z['m1'].real-colocal.real))/float(c['quadratic_cap']) for z,c in zip(rows,caps))
    power=max(1,math.ceil(-math.log2(allowed)));t=F(1,1<<power)
    for z,c in zip(rows,caps):
        remainder=t*t*c['quadratic_cap']; signal=float(t)*float(allowance)*abs(z['m1'].real)
        # Co-local body only has active KM: K0=c(h col0); double commutator
        # needs its own bound, not reused as this receiver's remainder.
        z['clock']=dict(t_exact=str(t),power=-power,t=float(t),linear=complex(z['C0']-1j*float(t)*z['m1']),
                        observed_imaginary_signal_with_one_percent_allowance=signal,
                        rational_remainder_upper=str(remainder),remainder_upper=float(remainder),
                        observed_imaginary_signal_minus_remainder=signal-float(remainder),
                        differs_from_C0_with_declared_float_allowance=signal>float(remainder),
                        # C0 coloc=.5, so real separation persists regardless of
                        # co-local first derivative; bounded K0 allows first-order bound.
                        co_local_C0=.5,co_local_change_upper=float(t)*float(species[0].hmax_exact)*math.sqrt(.5),
                        nonlocal_absolute_upper=abs(z['C0'])+float(t)*abs(z['m1'])+float(remainder))
        z['rational_caps']={name:str(v) for name,v in c.items()}
        if signal<=float(remainder):raise ArithmeticError('Useful actual12 clock not obtained')
        if .5-z['clock']['co_local_change_upper']<=z['clock']['nonlocal_absolute_upper']:
            raise ArithmeticError('Co-local finiteclock separation not obtained')
    oracle=dense_control()
    gauge=electric.run()
    captured={**body.CAPTURED,**inputs,Path(current.__file__).name:Path(current.__file__).read_bytes(),
              Path(prices.__file__).name:Path(prices.__file__).read_bytes(),
              Path(electric.__file__).name:Path(electric.__file__).read_bytes(),Path(__file__).name:own,
              'joined_phi_u1_nonlocal_body_transport_controls.py':(ROOT/'joined_phi_u1_nonlocal_body_transport_controls.py').read_bytes()}
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in captured.items()}
    for name,raw in captured.items():
        if (ROOT/name).exists() and (ROOT/name).read_bytes()!=raw:raise ArithmeticError('Input changed: '+name)
    binding=None
    if source_receipt is not None:
        raw=source_receipt.read_bytes();binding=dict(path=str(source_receipt),sha256=hashlib.sha256(raw).hexdigest(),receipt=json.loads(raw),
            status='Parent supplies source-action receipt; no full-source acceptance inferred by this bounded body lane')
    return dict(status='ACTUAL12_BODY_FINITE_TRANSPORT',derivation=DERIVATION,receiver_rows=rows,
                conserved_background_rational_prices={k:str(v) for k,v in energy.items()},
                co_local_actual_matter_m1=colocal,dense_same_H_CAR_Hermite=oracle,
                gauge_complete_receiver_controls={k:gauge[k] for k in ('all_orthogonal_Gauss_split_defect','full_unreduced_Wilson_lift_defect','arbitrary_local_U1_receiver_covariance_defect','wrong_unreduced_CAR_without_long_dressing_defect')},
                exact_rational_remainder_coefficients=True,signal_arithmetic='FLOATING, NOT INTERVAL; 1percent allowance is declared, not proven lower',
                source_action_binding=binding,full_current_transfer_claim=False,
                conditional=['Selected exact meanPhi root, Clifford and orthogonalQ preparation','Conditional analytic body proof; observed Slater matrices and first moments floating','Finite Hermite same-H control is a cutoff observation, NOT convergence'],
                excluded=['ground','charged pole','common cone','original SOURCE','future Slater/Wick propagation','fullN4 measured finite evolution'],
                governing_plan=PLAN,captured_parent_revision=REVISION,template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
                input_hashes=hashes,own_source_sha256=hashes[Path(__file__).name],own_source_utf8=own.decode(),
                runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'joined_phi_u1_nonlocal_body_transport_receipt.json')
    parser.add_argument('--source-receipt',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.source_receipt)
    args.output.write_text(json.dumps(body.packed(result),indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],rows=len(result['receiver_rows']),clock=result['receiver_rows'][0]['clock'],
                          elapsed_seconds=result['elapsed_seconds']),default=body.packed),flush=True)
