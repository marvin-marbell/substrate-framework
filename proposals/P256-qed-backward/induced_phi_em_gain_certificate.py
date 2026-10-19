#!/usr/bin/env python3
"""NEW actual-root gain enclosure for the tiled quantum PH/momentum source.

Full electric and Coulomb energies cancel in the gain by exact operator
commutation. N2 midpoint arithmetic is enclosed independently, not re-pinned.
No finite-box-to-macro, vacuum or stationary-pole promotion follows.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import mpmath as mp
from mpmath.libmp import to_str
import numpy as np
from induced_phi_em_gaussian_variational import build_native
from induced_phi_em_momentum_dressing import SOURCE_BYTES as DRESS_BYTES
from induced_phi_backreaction import PhiConfig, SELECTED, models, scalar_paths
from induced_phi_ir_well_transfer import rational, sqrt_bounds

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**DRESS_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
IV: Any=mp.iv


def su(value: F) -> F:
    return sqrt_bounds(value)[1]


def sl(value: F) -> F:
    return sqrt_bounds(value)[0]


def encoded(value: F) -> dict[str,str]:
    return dict(numerator=str(value.numerator),denominator=str(value.denominator),decimal_display=str(float(value)))


def exp_upper(value: F) -> F:
    if value<0:raise ValueError('Positive exponential endpoint required')
    term=F(1);total=term
    for n in range(1,129):term*=value/n;total+=term
    if value>=130:raise ArithmeticError('Actual determinant exponent exceeds positive series radius')
    return total+term*value/129/(1-value/130)


def centre(mean: dict[str,Any],refiner: dict[str,Any]) -> tuple[list[F],list[F]]:
    bounds=mean['centre_enclosure'];delta=rational(refiner['refined_all64_Gram_distance_to_exact000_upper'])
    mids=[];errors=[]
    for name in ('g0','gs'):
        lo=rational(bounds[name+'_lower']);hi=rational(bounds[name+'_upper'])
        mids.append((lo+hi)/2);errors.append((hi-lo)/2+delta)
    return mids,errors


def holomorphic_gain_price(mean: dict[str,Any],refiner: dict[str,Any],kappa: F) -> dict[str,Any]:
    mids,errors=centre(mean,refiner);radius=F(1,20)
    if any(g<=radius for g in mids):raise ArithmeticError('Complex mean disk crosses a scalar square-root cut')
    ds=[radius/(sl(g-radius)+sl(g)) for g in mids]
    relative_h=su(sum((d*d/(g-radius) for d,g in zip(ds,mids)),F(0)))
    relative_R=su(max(g/(g-radius) for g in mids))-1
    pnorm=1+(relative_h+relative_R)/2
    epsilon=abs(kappa)*su(F(2))*pnorm
    if epsilon>=F(1,4):raise ArithmeticError('Actual FCS resolvent disk is not certified')
    logdet=2*kappa*kappa+4*pnorm*pnorm*kappa*kappa/(1-epsilon)
    cfg=PhiConfig(Lambda=1.,**mean['microscopic_config']);charged=models(cfg)[1:]
    loads=sum(int(load) for load,_,_ in charged);detcap=exp_upper(loads*logdet)
    scales=[su(g+radius) for g in mids]
    dressed=F(0);seed=F(0);rows=[]
    for load,model,role in charged:
        onsite,paths=scalar_paths(model);onsitecap=abs(F(float(onsite)))*scales[0];pathcap=F(0)
        for _,scalars in paths:
            for mu,scalar in enumerate(scalars):
                z=complex(scalar)
                if z.real and z.imag:raise ArithmeticError('Original primitive scalar is not purely real or imaginary')
                pathcap+=abs(F(z.real if z.real else z.imag))*scales[0 if mu==0 else 1]
        d=int(load)*(2*pnorm*onsitecap+8*detcap*pnorm/(1-epsilon)*pathcap)
        s=int(load)*2*pnorm*(onsitecap+2*pathcap)
        dressed+=d;seed+=s;rows.append(dict(role=role,load=int(load),onsite_cap=encoded(onsitecap),path_sum_cap=encoded(pathcap)))
    cap=dressed+seed;price=cap*sum((error/(radius-error) for error in errors),F(0))
    return dict(radius=encoded(radius),centre=[encoded(g) for g in mids],root_distance_to_midpoint=[encoded(e) for e in errors],
        complex_projector_norm_upper=encoded(pnorm),FCS_resolvent_perturbation_upper=encoded(epsilon),
        one_profile_logdet_abs_upper=encoded(logdet),all64_FCS_determinant_abs_upper=encoded(detcap),
        dressed_CAR_holomorphic_cap_per_V=encoded(dressed),seed_CAR_holomorphic_cap_per_V=encoded(seed),
        gain_holomorphic_cap_per_V=encoded(cap),actual_root_gain_variation_per_V_upper=encoded(price),profiles=rows,
        proof='For real Fourier scalar coefficients, h(z)^2=R(z)^2 I4 with ReR²>0 on the bidisk. The displayed perturbation gives ||P(z)||<=pnorm. Local TrPxx=2 and TrGamma_j Pxx=0 by exact antiperiodic momentum parity. For pathell, v=(PTell)_j, ||v||²<=||ell||²<=2. K=(Rphase-I)P obeys ||K||<=epsilon and ||K||HS²<=8pnorm²kappa². TrK=2sum(cos(kappa v)-1), magnitude<=2kappa². Trace-log tail<=||K||HS²/[2(1-epsilon)], so |chi|^64<=detcap independentV. Transition resolvent is bounded by1/(1-epsilon), not a source overlap normalization. Gaussian endpoint dephasing is an operator-norm contraction. Local hop trace<=4pnorm/(1-epsilon)||F||; onsite rank2V gives2Vpnorm||onsite||. Actual polynomial scalar sums and64loads yield the displayed holomorphic CAR caps. Cauchy derivative on the remaining disk gives cap*sum(delta_g/(radius-delta_g)). FULL electric and Coulomb cancel exactly in gain for everymean; no irrelevant Coulomb-cap amplification.',
        scope='Actual uniform finiteN and exactimplicit222 mean variation. This does not bound finiteN-to-macro change of the prepared current or its dynamics.')


def iv(value: F | int) -> Any:
    value=F(value);return IV.mpf(value.numerator)/value.denominator


def ivc(value: complex) -> Any:
    return IV.mpc(iv(F(value.real)),iv(F(value.imag)))


def endpoints(value: Any) -> tuple[F,F]:
    lo=to_str(value._mpi_[0],75,strip_zeros=False);hi=to_str(value._mpi_[1],75,strip_zeros=False)
    if not isinstance(lo,str) or not isinstance(hi,str):raise TypeError('Real interval endpoint serialization failed')
    return F(lo)-F(1,10**70),F(hi)+F(1,10**70)


def inverse_rational(matrix: list[list[F]]) -> list[list[F]]:
    n=len(matrix);a=[row.copy()+[F(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for col in range(n):
        pivot=next(i for i in range(col,n) if a[i][col]);a[col],a[pivot]=a[pivot],a[col]
        scale=a[col][col];a[col]=[x/scale for x in a[col]]
        for i in range(n):
            if i!=col:
                factor=a[i][col];a[i]=[x-factor*y for x,y in zip(a[i],a[col])]
    return [row[n:] for row in a]


def exact_PT(native: dict[str,Any]) -> list[list[F]]:
    D=native['D'];V=native['V'];E=3*V
    if not np.array_equal(D,np.rint(D)):raise ArithmeticError('Physical incidence matrix lost integrality')
    d=[[F(int(D[i,j])) for j in range(E)] for i in range(V)]
    lap=[[sum((d[i][e]*d[j][e] for e in range(E)),F(0))+F(1,V) for j in range(V)] for i in range(V)]
    inv=inverse_rational(lap);L=[[inv[i][j]-F(1,V) for j in range(V)] for i in range(V)]
    result=[[F(i==j)-sum((d[x][i]*L[x][y]*d[y][j] for x in range(V) for y in range(V)),F(0)) for j in range(E)] for i in range(E)]
    if sum((result[i][i] for i in range(E)),F(0))!=2*V+1:raise ArithmeticError('Exact real quotient omitted harmonic coordinates')
    return result


def determinant(matrix: Any) -> Any:
    a=matrix.copy();n=a.rows;result=IV.mpc(1)
    for col in range(n):
        pivot=a[col,col]
        if 0 in pivot.real and 0 in pivot.imag:raise ArithmeticError('Interval determinant pivot meets zero')
        result*=pivot
        for i in range(col+1,n):
            factor=a[i,col]/pivot
            for j in range(col+1,n):a[i,j]-=factor*a[col,j]
    return result


def midpoint_N2_gain(mean: dict[str,Any],refiner: dict[str,Any],kappa: F,variance: F,axis: int) -> dict[str,Any]:
    IV.dps=65;native=build_native(mean,2);PT=exact_PT(native);mids,_=centre(mean,refiner)
    V=native['V'];E=3*V;n=4*V;I=IV.eye(n);I4=IV.eye(4)
    gamma=[IV.matrix([[ivc(complex(value)) for value in row] for row in g]) for g in SELECTED]
    scales=[IV.sqrt(iv(mids[0])),*([IV.sqrt(iv(mids[1]))]*3)]
    phi=[scales[mu]*gamma[mu] for mu in range(4)];G=gamma[axis+1]
    projectors={r:(I4+r*G)/2 for r in (-1,1)};profiles=[]
    for raw in native['Pprofiles']:
        onsite,primitive=scalar_paths(raw['model']);h=IV.zeros(n);onsiteM=iv(F(float(onsite)))*phi[0]
        for x in range(V):
            for a in range(4):
                for b in range(4):h[4*x+a,4*x+b]+=onsiteM[a,b]
        paths=[]
        for p in raw['paths']:
            hop=IV.zeros(4)
            for mu,z in enumerate(p['scalars']):hop+=ivc(complex(z))*phi[mu]
            hop*=p['seam'];x=p['source'];y=p['target']
            for a in range(4):
                for b in range(4):
                    h[4*x+a,4*y+b]+=hop[a,b];h[4*y+b,4*x+a]+=IV.mpc(hop[a,b].real,-hop[a,b].imag)
            if x==0:paths.append(dict(p,interval_hop=hop))
        two=[s for disp,s in primitive if sum(int(v) for v in disp)==2]
        x0=F(float(onsite))-2*sum((F(complex(s[0]).real) for s in two),F(0))
        one=[s for disp,s in primitive if sum(int(v) for v in disp)==1]
        speeds=[2*abs(F(complex(s[j+1]).imag)) for j,s in enumerate(one)]
        R=IV.sqrt(iv(mids[0]*x0*x0+mids[1]*sum((w*w for w in speeds),F(0))))
        P=(I-h/R)/2
        defect=P*P-P
        if max(float(endpoints(abs(defect[i,j]))[1]) for i in range(n) for j in range(n))>1e-50:
            raise ArithmeticError('Actual N2 projector identity failed; no fitted spectral formula permitted')
        profiles.append(dict(load=raw['load'],P=P,onsite=onsiteM,paths=paths))
    seed=IV.mpf(0);dressed=IV.mpf(0);tail=F(0)
    for profile in profiles:
        P=profile['P'];A=profile['onsite'];dA=IV.zeros(4)
        for r in (-1,1):
            for s in (-1,1):
                delta2=(r-s)**2*PT[axis][axis]
                dA+=(projectors[r]*A*projectors[s])*IV.exp(-iv(kappa*kappa*delta2/(8*variance)))
        seed+=V*profile['load']*sum((A[a,b]*P[b,a] for a in range(4) for b in range(4)),IV.mpc(0)).real
        dressed+=V*profile['load']*sum((dA[a,b]*P[b,a] for a in range(4) for b in range(4)),IV.mpc(0)).real
    for number,template in enumerate(profiles[0]['paths']):
        ell=[F(0)]*E
        for e in template['edges']:ell[e]+=1
        b=[sum((PT[i][j]*ell[j] for j in range(E)),F(0)) for i in range(E)]
        v=b[axis::3];ell2=sum((ell[i]*b[i] for i in range(E)),F(0))
        Rphase=IV.zeros(n)
        for x in range(V):
            block=IV.cos(iv(kappa*v[x]))*I4+IV.mpc(0,1)*IV.sin(iv(kappa*v[x]))*G
            for a in range(4):
                for c in range(4):Rphase[4*x+a,4*x+c]=block[a,c]
        data=[]
        for profile in profiles:
            P=profile['P'];M=I-P+Rphase*P;K=I-M;inverse=I.copy();term=I.copy()
            for _ in range(12):term=term*K;inverse+=term
            chi=determinant(M);transition=Rphase*P*inverse
            data.append(dict(chi=chi,transition=transition))
        determinant_product=IV.mpc(1)
        for profile,ctx in zip(profiles,data):determinant_product*=ctx['chi']**profile['load']
        for profile,ctx in zip(profiles,data):
            path=profile['paths'][number];y=path['target'];A=path['interval_hop'];dA=IV.zeros(4)
            for r in (-1,1):
                for s in (-1,1):
                    delta2=PT[axis][axis]+PT[3*y+axis][3*y+axis]-2*r*s*PT[axis][3*y+axis]
                    factor=IV.exp(-iv(kappa*kappa*delta2/(8*variance))+IV.mpc(0,1)*iv(kappa*(r*v[0]-s*v[y])/2))
                    dA+=factor*(projectors[r]*A*projectors[s])
            characteristic=IV.exp(-iv(variance*ell2/2));T=ctx['transition'];P=profile['P']
            seed+=2*V*profile['load']*characteristic*sum((A[a,c]*P[4*y+c,a] for a in range(4) for c in range(4)),IV.mpc(0)).real
            dressed+=2*V*profile['load']*characteristic*(determinant_product*sum((dA[a,c]*T[4*y+c,a] for a in range(4) for c in range(4)),IV.mpc(0))).real
            eps=abs(kappa)*su(sum((value*value for value in v),F(0)))
            if eps>=F(1,4):raise ArithmeticError('Actual Neumann evaluation lacks uniform convergence')
            scalar_cap=sum((abs(F(complex(z).real))+abs(F(complex(z).imag)) for z in path['scalars']),F(0))*max(su(g) for g in mids)
            detcap=exp_upper(64*(2*kappa*kappa+4*kappa*kappa/(1-abs(kappa)*su(F(2)))))
            tail+=8*V*profile['load']*scalar_cap*detcap*eps**13/(1-eps)
    lo,hi=endpoints(seed-dressed)
    return dict(N=2,V=V,M=2*V+1,seed_CAR_interval=[encoded(v) for v in endpoints(seed)],
        dressed_CAR_interval=[encoded(v) for v in endpoints(dressed)],midpoint_gain_lower=encoded(lo-tail),midpoint_gain_upper=encoded(hi+tail),
        Neumann_observable_tail_upper=encoded(tail),interval_precision_decimal=65,
        implementation='Independent originalbasis32x32 MPInterval projector/FCS calculation; exactrational physicalPT includesall17 modes. No numericaleigh/SVD/inverse/fullseaoverlap normalization. Nearidentity observable resolvent is summed12times withpricedtail. Source remains defined byunitaryconditionaltranslations, independent of this evaluation.')


def run(mean_path: Path,refiner_path: Path,source_path: Path) -> dict[str,Any]:
    start=time.monotonic();inputs={}
    for role,path in (('mean',mean_path),('refiner',refiner_path),('source',source_path)):
        raw=path.read_bytes();inputs[role]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),capsule=json.loads(raw))
    mean=inputs['mean']['capsule'];refiner=inputs['refiner']['capsule'];source=inputs['source']['capsule']
    state=source['descriptor'];kappa=F(float(state['kappa']));variance=F(float(state['s_density_variance']))
    if state['N']!=2:raise ValueError('Actual rigorous midpoint solver is scopedN2, notN6 ormacro')
    cap=holomorphic_gain_price(mean,refiner,kappa);point=midpoint_N2_gain(mean,refiner,kappa,variance,int(state['axis']))
    lower=rational(point['midpoint_gain_lower'])/8-rational(cap['actual_root_gain_variation_per_V_upper'])
    descriptor=dict(inputs={role:{key:entry[key] for key in ('path','sha256')} for role,entry in inputs.items()},
        N=2,charged_counts=[1,63],kappa=str(kappa),variance=str(variance),physical_mean='EXACTimplicitmacro222, enclosedbyoriginal000interval+refinedrootdistance',
        plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='85975947857ac7816cf0d8674f69229861014434b125a07a6996af0ecf72c4eb',work_item='joined-explanatory-campaign')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        holomorphic_mean_gain=cap,actual_midpoint_interval=point,actual_root_gain_per_V_lower=encoded(lower),actual_root_gain_positive=lower>0,
        cancellations='Both source andseed haveSAMEoriginal64CAR/realGaussHref. U_kappa commutes everyrho_x andeveryPi_a, soentireCoulomb+electricexpectations cancel identically forALLmeans, not justdiagonalrhofit. Only actualnonadiabaticPeierls/onsiteCAR gain isenclosed.',
        limitations='Actual N2 exactimplicitmean preparedstate improvement ONLY. No N2/N6-to-macro reduction, stationarity, fullcausalpropagator, photonvacuum/commoncone/pole/residue or QEDpromotion. Completeabsoluteenergymeanimageprice remainsasitsownoldbound; itdoesnotprice thiscancelledgain.',
        source_at_import={name:dict(sha256=hashlib.sha256(raw).hexdigest(),bytes_utf8=raw.decode()) for name,raw in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==raw for name,raw in SOURCE_BYTES.items()},
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,mpmath=mp.__version__,platform=platform.platform()),elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for role in ('mean','refiner','source','output'):parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args();result=run(args.mean,args.refiner,args.source)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:result[name] for name in ('input_sha256','actual_midpoint_interval','actual_root_gain_per_V_lower','actual_root_gain_positive','elapsed_seconds')},indent=2))
