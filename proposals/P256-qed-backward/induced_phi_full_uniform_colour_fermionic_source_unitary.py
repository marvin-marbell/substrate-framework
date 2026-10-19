"""NEW actual uniform105 Phi/SU4-colour finite-CAR globalGauss source proof.

The body has90 coupled normal coordinates, exact mixed Riccati Gaussian and
an explicit fast-Hermite stiffness corrector at the NEW colour-BO feedback mean.
Original .3chi colour electric, .2Lambda^5/4 alignment and .7Lambda^5/4
plaquette laws remain. Full105 Haar/cometric/Noether/volume are priced.
No nonuniform/localGauss/Theta/Pi/Coulomb/originalSOURCE/QED admission.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES:dict[str,bytes]={Path(__file__).name:Path(__file__).read_bytes()}


class CapturedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname:str)->Any:
        raw=Path(self.path).read_bytes();SOURCE_BYTES[Path(self.path).name]=raw
        return self.source_to_code(raw,self.path)


class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname:str,path:Any=None,target:Any=None)->Any:
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.origin is not None:
            source=Path(spec.origin)
            if source.parent==ROOT and source.suffix=='.py':
                spec.loader=CapturedLoader(fullname,str(source));return spec
        return None


finder=CapturedFinder();sys.meta_path.insert(0,finder)
import induced_phi_uniform_colour_stiffness_resolvent as resolvent
import induced_phi_uniform_relative_colour_quantum as colour
import induced_phi_colour_uniform_global_gauss as gauss
sys.meta_path.remove(finder)
SOURCE_BYTES.update(resolvent.SOURCE_BYTES);SOURCE_BYTES.update(colour.CAPTURED)
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

joined=resolvent.joined
iv:Any=joined.iv
exact,cap,pack,endpoint,interval=joined.exact,joined.cap,joined.pack,joined.endpoint,joined.interval
midpoint,enclosure=joined.midpoint,joined.enclosure
RESOLVENT_SHA='00ac73b3e00e77bfb73a6b5024778ccf8e09c5a74304e0260a80b1a89b122c6f'
COLOUR_SHA='8f50cdc7ab6f5860fde58a0d0607ced4afd6549c6b4207d916b1b14839faf01b'
GAUSS_SHA='c65fb72c037bfbf3892aecf989d923fbfbb32c4bc19c6554dbc9ea545550eb9d'
Poly=dict[tuple[int,int],Any]


def term(value:Any,p:int=0,c:int=0)->Poly:
    return {(p,c):cap(value)}


def add(*polynomials:Poly)->Poly:
    result:Poly={}
    for polynomial in polynomials:
        for key,value in polynomial.items():
            result[key]=result[key]+value if key in result else value
    return result


def multiply(left:Poly,right:Poly)->Poly:
    result:Poly={}
    for (p,c),x in left.items():
        for (q,d),y in right.items():
            key=(p+q,c+d);value=x*y
            result[key]=result[key]+value if key in result else value
    return result


def scale(polynomial:Poly,value:Any)->Poly:
    return {key:cap(coefficient*value) for key,coefficient in polynomial.items()}


def power(polynomial:Poly,n:int)->Poly:
    result=term(exact(1))
    for _ in range(n):result=multiply(result,polynomial)
    return result


class GaussianRadialBounds:
    """Exact joint Gaussian: Phi and Ctilde=C+M Phi are independent marginals."""
    def __init__(self,phi_covariance:Any,z_covariance:Any):
        self.phi=phi_covariance;self.z=z_covariance;self.cache:dict[tuple[int,int,int],Any]={}

    def moment(self,which:int,degree:int,p:int)->Any:
        key=(which,degree,p)
        if key not in self.cache:
            covariance=self.phi if which==0 else self.z
            n=p*degree//2;factor=1
            for j in range(n):factor*=45+2*j
            self.cache[key]=cap((exact(factor)*covariance**n)**exact(F(1,p)))
        return self.cache[key]

    def norm(self,polynomial:Poly,p:int=2)->Any:
        return cap(sum(value*self.moment(0,a,p)*self.moment(1,b,p) for (a,b),value in polynomial.items()))


def packet_prices(root:dict[str,Any],quantum:dict[str,Any],profiles:list[dict[str,Any]],points:list[list[dict[str,Any]]],algebra:dict[str,Any])->dict[str,Any]:
    Lambda=exact(F(2)**8192);Lambda54=exact(F(2)**10240);fast=exact(F(2)**3072)
    chi=exact(F(2)**-4096);Gp=8*chi;Gc=exact(F(3,10))*chi;R=F(1,8192);Rv=exact(R)
    g0,gs=interval(root['parameters']['g0']),interval(root['parameters']['gs']);gmax=joined.maximum([g0,gs])
    cphi=exact(F(quantum['actual_joint_Phi_covariance_operator_upper']['upper']))
    M=exact(F(quantum['actual_joint_conditional_mean_M_operator_upper']['upper']))
    eps=exact(F(quantum['actual_joint_ACC_relative_primitive_alpha0_operator_upper']['upper']))
    eta=exact(F(quantum['reference_relative_stiffness_offdiagonal_norm_upper']['upper']))
    if endpoint(eps,1)>=F(1,2) or endpoint(eta,1)>=F(1,2):
        raise ArithmeticError('Actual joint90 Gaussian conditional comparison is not certified')
    zcov=cap(1/(1-eps));moments=GaussianRadialBounds(cphi,zcov)
    sigma_values=[iv.sqrt(exact(F(2)**-7168)*x) for x in algebra['covariance']]
    sigmax=joined.maximum(sigma_values);sigmin=exact(min(endpoint(x,0) for x in sigma_values))
    alpha0max=joined.maximum([fast*w/Gc for w in algebra['omega']]);accmax=cap((1+eps)*alpha0max)
    kappapmax=cap(96*gmax)  # Primitive Phi Hessian bound; actual Phi sea Hessian is negative semidefinite.
    aphimax=cap(iv.sqrt(1+eta)*iv.sqrt(Lambda*kappapmax/Gp))
    amax=cap(aphimax+(1+M)**2*accmax)
    qcovmax=cap(cphi+(1+M)**2*sigmax**2*zcov);qcovmin=exact(endpoint(1/(2*amax),0))
    ph=term(exact(1),1);z=term(exact(1),0,1)
    physicalC=add(scale(ph,M),scale(z,sigmax));radius=add(scale(ph,1+M),scale(z,sigmax))
    weighted=sum(x*x for matrix in algebra['Y'] for row in matrix for x in row)
    A=cap(iv.sqrt(weighted));B=cap(3*iv.sqrt(exact(15))*A)
    P=add(term(A,1,2),term(B,1,0));oneP=add(term(exact(1)),P)
    P4=moments.norm(P,4);P2=moments.norm(P)
    if endpoint(P4,1)>=F(1,100):raise ArithmeticError('Actual polynomial-corrected preparation is not controlled')
    gradPhiP=add(term(A,0,2),term(B),term(2*A*M/sigmin,1,1))
    gradP=add(term(A,0,2),term(B),term(2*A*(1+M)/sigmin,1,1))
    hessPhiP=add(term(4*A*M/sigmin,0,1),term(2*A*M*M/(sigmin*sigmin),1,0))
    hessP=add(term(4*A*(1+M)/sigmin,0,1),term(2*A*(1+M)**2/(sigmin*sigmin),1,0))
    gradPhiLog=add(scale(ph,aphimax),scale(z,M*accmax*sigmax))
    gradLog=add(scale(ph,aphimax),scale(z,(1+M)*accmax*sigmax))
    slow=moments.norm(scale(add(multiply(gradPhiLog,gradPhiP),scale(hessPhiP,iv.sqrt(exact(45))/2)),Gp))
    acc_error=moments.norm(term(2*Gc*eps*alpha0max*sigmax*A/sigmin,1,2))
    dweighted=sum((algebra['D'][i][a][b]*iv.sqrt(algebra['covariance'][a]*algebra['covariance'][b]))**2 for i in range(45) for a in range(15) for b in range(15))
    Aq=cap(fast*iv.sqrt(dweighted));v=scale(ph,M/sigmin)
    normal_mean_change=moments.norm(scale(multiply(ph,add(scale(multiply(z,v),exact(2)),power(v,2))),Aq))
    qC=scale(multiply(ph,add(power(add(z,v),2),term(3*iv.sqrt(exact(15))))),Aq)
    qP=moments.norm(multiply(qC,P))
    nonlinear:Poly=add(term(Lambda*iv.sqrt(gmax)/(2*exact(F(1,64))*iv.sqrt(exact(512))),3),term(Lambda/exact(64),4))
    nonlinear=add(nonlinear,scale(multiply(power(ph,2),power(physicalC,2)),Lambda54/exact(2560)))
    framebound=add(term(2*iv.sqrt(g0+3*gs)),term(1/iv.sqrt(exact(128)),1))
    nonlinear=add(nonlinear,scale(multiply(power(framebound,2),power(physicalC,4)),Lambda54/exact(7680)),scale(power(physicalC,4),Lambda54*exact(F(7,10))*64*3/exact(8192)))
    C1=exact(0);C2=exact(0);species=[];active=None
    for profile,rows in zip(profiles,points,strict=True):
        onsite=abs(exact(F(profile['onsite'])));Dphi=(onsite+2*sum(abs(exact(F(p['c0'])))+abs(exact(F(p['ci']))) for p in profile['paths']))/(8*iv.sqrt(exact(2)))
        Dc,bpc,bcc,bpcc,bccc=[exact(0) for _ in range(5)]
        for path in profile['paths']:
            c0,ci=abs(exact(F(path['c0']))),abs(exact(F(path['ci'])));l=exact(path['length'])/(8*iv.sqrt(exact(2)))
            mean=c0*iv.sqrt(g0)+ci*iv.sqrt(gs);derivative=(c0+ci)/(8*iv.sqrt(exact(2)))
            Dc+=2*l*mean;bpc+=2*l*derivative;bcc+=l*l*mean;bpcc+=l*l*derivative;bccc+=l**3*mean/3
        Emin=exact(min(endpoint(r['E'],0) for r in rows));Emax=joined.maximum([r['E'] for r in rows]);delta=Emin/2
        D1=cap(Dphi+Dc+bpc*Rv);H2=cap(2*bpc+2*bcc+2*bpcc*Rv)
        if endpoint(2*Rv*D1,1)>=endpoint(Emin/4,0):raise ArithmeticError('Actual nonlinear90 all-species compact gap ball failed')
        d=cap(D1/delta);e=cap(H2/delta)
        first=cap(64*20*d);second=cap(64*(400*d*d+80*e));C1+=profile['load']*first;C2+=profile['load']*second
        h1=add(scale(ph,Dphi),scale(physicalC,Dc))
        h2=add(scale(multiply(ph,physicalC),bpc),scale(power(physicalC,2),bcc))
        h3=add(scale(multiply(ph,power(physicalC,2)),bpcc),scale(power(physicalC,3),bccc))
        rest=add(h2,h3);allh=add(h1,rest)
        sea=add(scale(h3,exact(4)),scale(add(scale(multiply(h1,rest),exact(2)),power(rest,2)),2/delta),scale(power(allh,3),exact(F(4,3))/delta**2))
        physical_scale=Lambda if profile['scale']=='heavy' else exact(1)
        nonlinear=add(nonlinear,scale(sea,64*profile['load']*physical_scale))
        if profile['role']=='active filled charged':active=dict(first=first,lip=cap(D1+4*Emax*D1/delta))
        species.append(dict(role=profile['role'],load=profile['load'],physical_scale=profile['scale'],gap_ball_distance_to_zero_lower=pack(delta),joint90_first_derivative_upper=pack(D1),nonlinear90_second_derivative_upper=pack(H2),CAR_first_per_coordinate=pack(first),CAR_second_per_coordinate=pack(second)))
    if active is None:raise ArithmeticError('Actual active CAR source profile absent')
    Zlower=exact(1)-cap(12*moments.norm(radius)**2/(Rv*Rv))
    if endpoint(Zlower,0)<=F(1,2):raise ArithmeticError('Actual corrected compact packet normalizer price failed')
    normalization=cap(1/iv.sqrt(Zlower))
    bumpgrad=scale(radius,12/(Rv*Rv))
    bumphess=add(term(12*iv.sqrt(exact(90))/(Rv*Rv)),scale(power(radius,2),exact(120)/(Rv**4)))
    bump=cap(normalization*moments.norm(scale(add(multiply(bumpgrad,add(multiply(oneP,gradLog),gradP)),scale(multiply(bumphess,oneP),iv.sqrt(exact(90))/2)),Gp)))
    gradient_gaussian_L4=cap((exact(90*92)*(amax/2)**2)**exact(F(1,4)))
    gradient_scalar=cap(normalization*((1+P4)*gradient_gaussian_L4+moments.norm(gradP)+moments.norm(multiply(oneP,bumpgrad))))
    kato_first=cap(Gp*iv.sqrt(exact(90))*C1*gradient_scalar)
    kato_second=cap(Gp*90*(C2+C1*C1)/2)
    nonlinear_action=cap(normalization*moments.norm(multiply(nonlinear,oneP)))
    root_force=cap(4*normalization*sum(exact(F(quantum[key]['upper'])) for key in ('actual_root_box_force_times_Phi_coordinate_norm_upper','actual_omitted_static_colour_PH_ZPE_force_times_Phi_coordinate_norm_upper','actual_omitted_Phi_ZPE_force_times_Phi_coordinate_norm_upper')))
    canonical=cap(normalization*(slow+acc_error+normal_mean_change+qP)+bump+kato_first+kato_second+nonlinear_action+root_force)
    # Exact corrected BODY weighted derivatives for full105 Gauss/metric prices.
    b1poly=multiply(radius,add(multiply(oneP,add(gradLog,bumpgrad,term(iv.sqrt(exact(90))*C1))),gradP))
    Afrob=cap(iv.sqrt(exact(45))*(aphimax+(1+M)**2*accmax))
    basehess=add(power(gradLog,2),term(Afrob),scale(multiply(bumpgrad,gradLog),exact(2)),bumphess,
        scale(add(gradLog,bumpgrad),2*iv.sqrt(exact(90))*C1),term(90*(C2+C1*C1)))
    b2poly=multiply(power(radius,2),add(multiply(oneP,basehess),scale(multiply(gradP,add(gradLog,bumpgrad,term(iv.sqrt(exact(90))*C1))),exact(2)),hessP))
    B1=cap(normalization*moments.norm(b1poly));B2=cap(normalization*moments.norm(b2poly))
    params=root['parameters'];metric=gauss.full_uniform_gauss_prices(F(params['g0']['lower']),F(params['gs']['lower']),F(params['g0']['upper']),F(params['gs']['upper']),64,8324,R,F(pack(B1)['upper']),F(pack(B2)['upper']),F(3,10))
    metric_residual=exact(F(2)**-4096*metric['total_reference_packet_Noether_and_metric_residual_divided_chi_upper'])
    complete=cap(canonical+metric_residual)
    endpoint_price=cap(normalization*4*iv.sqrt(exact(90))*active['first']*moments.norm(multiply(radius,oneP)))
    charged=cap(normalization*active['lip']*moments.norm(multiply(radius,oneP)))
    times=[];rootp=iv.sqrt(exact(F(1,2)))
    for t in (F(0),F(1,4096),F(4096)):
        re=sum(iv.cos(row['E']*exact(t)) for row in points[1])/128;im=-sum(iv.sin(row['E']*exact(t)) for row in points[1])/128
        vac=cap(exact(t)*canonical);src=cap(endpoint_price+exact(t)*rootp*(canonical+charged))
        base_error=cap(2*rootp*vac+2*src+rootp*endpoint_price)
        additional=cap(exact(8*abs(t))*metric_residual);error=cap(base_error+additional)
        if not t:error=exact(0)
        modulus=iv.sqrt(re*re+im*im);lower_K=max(F(0),endpoint(modulus-error,0))
        times.append(dict(time=str(t),raw_reference_real=pack(re),raw_reference_imag=pack(im),canonical_twoleg_complex_error_upper=pack(base_error),additional_full105_Haar_Noether_metric_error_upper=pack(additional),total_same_physical_H_complex_error_upper=pack(error),
            raw_actual_same_H_source_real=pack(enclosure(endpoint(re,0)-endpoint(error,1),endpoint(re,1)+endpoint(error,1))),
            raw_actual_same_H_source_imag=pack(enclosure(endpoint(im,0)-endpoint(error,1),endpoint(im,1)+endpoint(error,1))),
            nonzero_modulus_lower=pack(exact(lower_K)),raw_compound_return_probability=pack(enclosure(lower_K**2,min(F(1,2),endpoint((modulus+error)**2,1)))),vacuum_leg_norm_price=pack(vac),source_leg_norm_price=pack(src)))
    if F(times[-1]['nonzero_modulus_lower']['lower'])<=0:raise ArithmeticError('Actual full105 raw physical source nonzero4096 failed')
    tail=exact(F(2)**-32768);tailmargin=Rv*Rv/(8*qcovmax)-exact(F(32768)+F(90,4))*iv.ln(exact(2))
    if endpoint(tailmargin,0)<=0:raise ArithmeticError('Positive all90 Gaussian tail price not established')
    primitive_centre=Lambda*4096*(g0*g0+3*gs*gs)
    sea_centre=sum(-2*profile['load']*(Lambda if profile['scale']=='heavy' else exact(1))*sum(row['E'] for row in rows) for profile,rows in zip(profiles,points,strict=True))
    phi_min=exact(F(quantum['certified_Phi_eigenvalue_lower_divided_Lambda']['lower']));colour_min=exact(F(quantum['certified_colour_eigenvalue_lower_divided_Lambda5over4']['lower']))
    colour_max=cap((g0+3*gs)/10+exact(F(2)**-2048)*exact(F(quantum['certified_colour_sea_norm_divided_Lambda']['upper'])))
    fieldenergy=enclosure(endpoint(45*iv.sqrt(1-eta)*(iv.sqrt(Gp*Lambda*phi_min)+iv.sqrt(Gc*Lambda54*colour_min))/2,0),endpoint(45*iv.sqrt(1+eta)*(iv.sqrt(Gp*Lambda*kappapmax)+iv.sqrt(Gc*Lambda54*colour_max))/2,1))
    energy=primitive_centre+sea_centre+fieldenergy
    return dict(radius_exact=str(R),bump_power=6,normal_dimensions=90,physical_uniform_dimensions=105,
        exact_joint_wave_ABI=quantum['exact_joint_wave_ABI'],covariance_operator_lower=pack(qcovmin),covariance_operator_upper=pack(qcovmax),conditional_Phi_covariance_upper=pack(cphi),conditional_colour_relative_covariance_upper=pack(zcov),
        exact_fast_corrector='P=sum_i Phi_i sum_axis[(C_axis+M_axis Phi)^T Xi_i(C_axis+M_axis Phi)-Tr(Cov0 Xi_i)], M=A_CC^-1 A_CP EXACT JOINT90 function, Xi=primitiveDi/(OmegaA+OmegaB). Body=Z^-1/2(1-P)b6 f_A. Not product90Gaussians or a formal unexecuted resolvent.',
        corrector_L2_norm_upper=pack(P2),corrector_L4_norm_upper=pack(P4),compact_corrected_normalizer_lower=pack(Zlower),positive_all90_Gaussian_tail_L2_factor_upper=pack(tail),tail_log_margin=pack(tailmargin),
        actual_all65_gap_and_Kato_prices=species,global_CAR_first_per_coordinate_upper=pack(cap(C1)),global_CAR_second_per_coordinate_upper=pack(cap(C2)),
        weighted_body_r_gradient_norm_upper=pack(B1),weighted_body_r2_Hessian_norm_upper=pack(B2),
        nonlinear_and_corrector_prices=dict(actual_slow_action_on_fast_polynomial=pack(slow),actual_ACC_difference_action=pack(acc_error),conditional_Cmean_shift_in_primitive_cubic=pack(normal_mean_change),actual_primitive_cubic_times_corrector=pack(qP),true_primitive_and_sea_Taylor_remainder_on_corrected_packet=pack(nonlinear_action),compact_bump_action=pack(bump),CAR_Kato_first=pack(kato_first),CAR_Kato_second_PLUS_square=pack(kato_second),explicit_mean_box_and_omitted_ZPE_force_prices=pack(root_force)),
        canonical_reference_energy_residual_upper=pack(canonical),full105_reference_energy_residual_upper=pack(complete),actual_prepared_Rayleigh_residual_upper=pack(2*complete),
        actual_full105_metric_prices={key:pack(exact(value)) for key,value in metric.items()},additional_full105_Haar_Noether_metric_reference_residual_upper=pack(metric_residual),
        raw_primitive_centre_energy=pack(primitive_centre),raw_all65_occupied_centre_energy=pack(sea_centre),raw_exact_joint90_harmonic_energy_enclosure=pack(fieldenergy),
        raw_same_H_reference_energy_enclosure=pack(energy),actual_prepared_Rayleigh_energy_enclosure=pack(energy+enclosure(-endpoint(complete,1),endpoint(complete,1))),
        raw_source_times=times,
        physical_comparison='BODY Href=Ecentre+E90_exact+dGamma(hmean) fixes THIS corrected compact scalar field body. Physical comparator is U_CAR W_Kato Href W_Kato^dag U_CAR^dag on the chart, with scalar sqrtJ unitary conjugation. Physical E0 becomes Etilde=W_Kato^dag E0 W_Kato; mean E0 used ONLY as endpoint-priced reference, never hmean acting directly on the physical Kato Slater. Bothlegs rawcommonphase cancels only in paired Heisenberg inner product.',
        primitive_linear_force='Bare class+sea Phi force at NEW BO mean is NONZERO, exactly minus primitivefast colour ZPE gradient. Kept together with actualPhiC² cubic. FastHermiteP cancels its mean-subtracted excited component. Only BO effectiveforce0; no bare-force suppression or actualquantumground claim.',
        domain='b6=(1-|q|²/R²)^6 insideball, zerooutside. C5/H4, polynomialP, smooth gap-ball Kato and regular exact105chart imply D(H²). A smooth unitary globalextension is obtained by principal-log W on2R withC6 collar toidentity before3R, identity outside small SU4 principal chart; no offball gap or compact harmonic propagation assumption. Comparison fieldI keeps packet fixed, physicalH evolves unrestricted.',
        mathematical_prices='All90 exact mixed K retained in RiccatiA. ConditionalPhi andCtilde are independent exactGaussian marginals; radialmoment Minkowski/Cauchy bound actualpolynomial corrector and its derivatives. Staticsea Taylor includes nonlinear hPhiC,hCC,hPhiCC,hCCC and exactcontact subtraction, not linear90 fiction. Original alignmentevenC givesexplicitPhi²C² andC4 remainders; uniformplaquette Hessian0 but .7Lambda54 commutatorquartic retained. Full105 cometric differsdiag(8,.3) byO(r²), all90drift/cross/Noether/volume and completeSU4Haar priced using OWN actualB1/B2.')


def literal_nonlinear90(root:dict[str,Any],fullcap:dict[str,Any],profiles:list[dict[str,Any]],R:F)->dict[str,Any]:
    native,geom=joined.literal_geometry(root,fullcap);T=geom['M']/np.sqrt(8)
    theta=(2*np.pi*np.asarray(list(product(range(4),repeat=3)))+np.pi)/4
    def hblocks(profile:dict[str,Any],q:np.ndarray)->np.ndarray:
        frame=native['frame']+np.einsum('i,imab->mab',q[:45],geom['deltas'])/8
        X=np.einsum('xa,aij->xij',q[45:].reshape(3,15),T)/8
        h=np.repeat((float(F(profile['onsite']))*frame[0])[None],64,axis=0)
        for path in profile['paths']:
            axis,l=path['axis'],path['length'];hop=float(F(path['c0']))*frame[0]+1j*float(F(path['ci']))*frame[axis+1]
            U=expm(1j*l*X[axis]);coefficient=(hop@U+U@hop)/2;phase=np.exp(1j*l*theta[:,axis])
            h+=phase[:,None,None]*coefficient+phase.conj()[:,None,None]*coefficient.conj().T
        return -h if profile['scale']=='heavy' else h
    rng=np.random.default_rng(222105);directions=[np.eye(90)[0],np.eye(90)[45],np.eye(90)[45+13],rng.normal(size=90)]
    directions=[d/np.linalg.norm(d) for d in directions];coordinates=[np.zeros(90),*[.8*float(R)*d for d in directions]]
    species=[];source=[];ODE=[]
    for pindex,profile in enumerate(profiles):
        ranks=[];gaps=[]
        for q in coordinates:
            hs=hblocks(profile,q);energies,U=np.linalg.eigh(hs);negative=energies<0
            if not np.all(np.sum(negative,axis=1)==2):raise ArithmeticError('Actual nonlinear90 uniform occupied rank changed')
            ranks.append(np.sum(negative,axis=1).tolist());gaps.append(float(np.min(energies[:,2]-energies[:,1])))
            if pindex==1:
                W=U[:,:,:2];P=np.mean(W@W.conj().transpose(0,2,1),axis=0)
                source.append(dict(coordinate_norm=float(np.linalg.norm(q)),literal_source=joined.scalar.literal_car_source(P)))
        q=coordinates[-1];step=2**-15
        def rhs(t:float,u:np.ndarray)->np.ndarray:
            h=hblocks(profile,t*q)[0]
            derivative=(hblocks(profile,(t+step)*q)[0]-hblocks(profile,(t-step)*q)[0])/(2*step)
            _,connection=joined.projector_connection(h,derivative)
            return (connection@u.reshape(4,4)).reshape(-1)
        sol=solve_ivp(rhs,(0.,1.),np.eye(4,dtype=complex).reshape(-1),rtol=2e-11,atol=2e-13,method='DOP853')
        if not sol.success:raise ArithmeticError(sol.message)
        U=sol.y[:,-1].reshape(4,4);P0,_=joined.projector_connection(hblocks(profile,np.zeros(90))[0],np.zeros((4,4),complex));P1,_=joined.projector_connection(hblocks(profile,q)[0],np.zeros((4,4),complex))
        errors=dict(unitarity=float(np.linalg.norm(U.conj().T@U-np.eye(4))),occupied_subspace_transport=float(np.linalg.norm(U@P0@U.conj().T-P1)))
        if max(errors.values())>5e-10:raise ArithmeticError('Actual nonlinear90 Kato transport failed')
        ODE.append(errors);species.append(dict(role=profile['role'],load=profile['load'],all64_occupied_ranks=ranks,minimum_BOTHband_gap=min(gaps)))
    chart=gauss.literal_chart(native,geom['Qcart'],R,F(3,10))
    return dict(all65_species=species,actual_physical_CAR_source=source,actual_nonlinear90_radial_Kato_ODE=ODE,actual_full105_original_Haar_metric_chart=chart,
        exact_source_probability='Every supported uniform nonlinearPhi/SU4colour h(k) has negative rank2 by certifiedgap. Translation givesTrQ_site=2; physicalco-localE0CAR/Ref4->3 yieldsnorm²1/2 for ANY normalizedcorrectedfieldbody/globalorientation. No chiral or productwave assumption.')


def run(supplier_path:Path,full_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=supplier_path.read_bytes();fraw=full_path.read_bytes();supplier=json.loads(raw);fullcap=json.loads(fraw)
    for module,digest in ((resolvent,RESOLVENT_SHA),(colour,COLOUR_SHA),(gauss,GAUSS_SHA)):
        if hashlib.sha256(SOURCE_BYTES[module.__name__+'.py']).hexdigest()!=digest:raise ArithmeticError('Completed frozen scientific dependency epoch required')
    if hashlib.sha256(fraw).hexdigest()!=joined.FULL_RECEIPT_SHA:raise ArithmeticError('Frozen f63 configuration/normal-basis controls required')
    if not all(supplier['source_unchanged_since_import'].values()) or supplier['source_at_import'][Path(colour.__file__).name]['sha256']!=COLOUR_SHA:
        raise ArithmeticError('Actual full90 supplier source capsule mismatch')
    if supplier['actual_joint90_quantum'] is None:raise ArithmeticError('Mixing discriminator is not a completed quantum supplier')
    iv.dps=3000;mp.mp.dps=80
    root=supplier['actual_NEW_colour_BO_feedback_mean'];profiles=supplier['input_descriptor']['species_coefficients']
    if [p['load'] for p in profiles]!=[1,1,63]:raise ArithmeticError('Original all65 matter loads changed')
    native,geom=joined.literal_geometry(root,fullcap)
    if geom['labels']!=supplier['Phi_labels']:raise ArithmeticError('Actual90 normal coordinate ordering mismatch')
    g0,gs=interval(root['parameters']['g0']),interval(root['parameters']['gs']);points=joined.bloch_points(profiles,g0,gs)
    algebra=resolvent.clifford_primitive_resolvent(geom,[g0,gs,gs,gs]);proof=packet_prices(root,supplier['actual_joint90_quantum'],profiles,points,algebra)
    literal=literal_nonlinear90(root,fullcap,profiles,F(proof['radius_exact']))
    literal['actual_NEW_mean_fast_Hermite_differential_and_original_alignment']=resolvent.literal_primitive(algebra,geom,native)
    coefficients=[]
    for i in range(45):
        for a in range(15):
            for b in range(a,15):
                value=algebra['D'][i][a][b]
                if endpoint(value,0)!=0 or endpoint(value,1)!=0:
                    coefficients.append(dict(Phi_coordinate=i,colour_grade_A=a,colour_grade_B=b,
                        primitive_cubic_divided_Lambda54=pack(value),
                        fast_resolvent_Xi_divided_Lambda78=pack(algebra['X'][i][a][b]),
                        normalized_fast_Hermite_coefficient=pack(algebra['Y'][i][a][b])))
    proof['actual_fast_Hermite_coefficients_at_NEW_feedback_mean']=coefficients
    proof['actual_exact_Clifford_Hermite_identity_enclosure_residual_upper']=pack(algebra['coefficient_residual'])
    descriptor=dict(law='NEW_ORIGINALCOLOUR_UNIFORM105_REAL_PHI_EXACT_FINITE_CAR_GLOBAL_SU4_GAUSS',plan='issue-222-induced-photon-delta13-continuation-20261008',assigned_scientific_revision=joined.scalar.uniform.ASSIGNED_REVISION,
        parameters=dict(N=4,V=64,a='1/64',UEphys=1,u=64,Lambda='2^8192',chi='2^-4096',Gphi='8chi',Gc='3chi/10',charged_active=1,charged_auxiliary=63,neutral_heavy=1,staticRef=4,total_particles=8324,finite_CAR_modes=16644),
        source='E0=.5 sum4 c_active0alpha^dag Ralpha; physical raw CAR Ref4->3, no whitening',
        supplier_input=dict(path=str(supplier_path),sha256=hashlib.sha256(raw).hexdigest(),input_sha256=supplier['input_sha256']),full45_configuration_controls=dict(path=str(full_path),sha256=hashlib.sha256(fraw).hexdigest()),
        exact_NEW_BO_feedback_mean=root,exact_joint90_wave_ABI=proof['exact_joint_wave_ABI'],original_colour_law=supplier['input_descriptor']['physical_colour_law'],
        actual_corrector_supplier_sha256=RESOLVENT_SHA,actual_full105_Haar_Noether_supplier_sha256=GAUSS_SHA,omitted_terms=['nonuniform fields','local/nonuniform Gauss','Theta/Pi','Coulomb'],original_SOURCE_frozen_separate=True)
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_full105_corrected_joint90_quantum_CAR_source_proof=proof,
        actual_original_nonlinear90_CAR_Kato_Haar_consumers=literal,
        source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},
        input_unchanged=dict(supplier=supplier_path.read_bytes()==raw,full45=full_path.read_bytes()==fraw),runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__),elapsed_seconds=time.monotonic()-started,
        adverse_facts=['NEWuniform105/globalGaussbody90 physicalsource only, notnonuniform/localGauss/fullEM/originalSOURCE/eR/QED/commoncone/scattering.','ActualPhi-Cstaticmix isNONZERO andretainedinexactRiccatiGaussian; no90scalarproductfiction.','BOfeedbackmean includesleadingcolourZPE ONLY; omittedPhiZPE/colourPHforcespriced, no truequantumground/selfconsistentquantummean admission.','BareclassicalPhi forceNONZERO; primitivePhiC² excitedcomponent explicitlycancelledbyactualfastHermitepolynomial, remainingactionpriced.','Whole105 cometric/SU4Haar/Noe/volume retained withactualbodyB1/B2, notolderab8Phi45Gaussreuse.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--supplier',type=Path,required=True);parser.add_argument('--full45',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=run(args.supplier,args.full45)
    if not all(result['source_unchanged_since_import'].values()) or not all(result['input_unchanged'].values()):raise ArithmeticError('Actual scientific source/input changed during full105 execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n');proof=result['actual_full105_corrected_joint90_quantum_CAR_source_proof']
    print(json.dumps(dict(input_sha256=result['input_sha256'],Rayleigh_residual=proof['actual_prepared_Rayleigh_residual_upper'],raw_source_times=proof['raw_source_times'],elapsed_seconds=result['elapsed_seconds']),indent=2))
