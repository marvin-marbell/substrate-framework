"""NEW same-source parity/connected density closure at 2^-16; no clock adoption.

Exact identities and rational outward envelopes are distinct from conditional
floating initial contractions. No interacting-time Wick assertion is used.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import joined_phi_u1_body_word as body
import joined_phi_u1_current_moments as current
import joined_phi_u1_electric_hop_transport as history
import joined_phi_u1_finite_current_kernel as prices

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_parity_density_closure'
CLOCK = F(1, 65536)
REVISION = '9a2fd8299a741b12cbc9effe2c6a264ce87e2deef1bf73787109d7604849ce49'
PROOF = {
 'domain': 'On fixed finite CAR sectors HC is a bounded constant matrix and matter is a bounded smooth finite Fourier multiplier. H is selfadjoint on D(Pi²), and the Gaussian times finite CAR initial vectors lie in the Schwartz core. Smooth bounded derivatives preserve the required finite-time momentum domains. Connected commutator formulas are first on this common core and extend by the supplied positive moment estimates; no unbounded Pi operator norm is taken.',
 'antiunitary': 'A=Gamma(Gamma1 Gamma3) K combined with Theta parity. A i A^-1=-i, Theta odd, Pi even. U Gamma0bar Udag=Gamma0 and U Gammaibar Udag=-Gammai. Real positive frame rescalings preserve these identities. Every literal onsite and native c0Gamma0+i ciGammai hop, its adjoint, real AP seam and ordered scalar phase exp(iqTheta) is invariant. HE=64Pi², HC=64rhoLplusrho and all Ref/site numbers are invariant separately. Neutral charge0 factor may be arbitrary for charged observables because it factors exactly.',
 'state': 'AP labels j map to 3-j componentwise, so k maps to -k modulo2pi; all64 labels close. At Theta0 the spectral negative projector is invariant under sitewise U K. Its occupied determinant line is invariant up to phase, not necessarily each orbital. Tensor64 fixed-mean charged seas and real even Gaussian densityvariance2 give invariant background density matrix. Ref4 full occupied determinant transforms by detU (unit modulus), hence its density matrix is invariant. E0=.5 sum_alpha c0alpha dagger Ralpha is a spin singlet under the SAME U on active and Ref; its transformed coefficient sum uses U Udag=I. The Ref3 holes are orthogonal and tracing them gives the ACTUAL four unequal raw Slater weights; their mixture, not each channel, is invariant. Raw mass=.5 is not normalized.',
 'parity': 'A H A^-1=H implies A U(t) A^-1=U(-t). For invariant initial density matrix and Hermitian A-even n,Pi,C, expectations are even(t), while J is odd(t). Antiunitary conjugation also flips the explicit i: D=i<[Theta(t),Theta]> and mixed=i<[J(t),Theta]> are odd(t), RJJ=-i<[J(t),J]> odd(t). This needs invariance of the actual source mixture; no stationarity is assumed. Odd density Taylor coefficients vanish exactly, in particular cubic.',
 'second': 'nsecond(0)=-<[H,[H,n]]>. Since [E+K,n]=0, inner word is sum_r wxr O_r, wxr=1[col=x]-1[row=x]. E word retains HE and HC separately. On initial real Gaussian, <exp(iqTheta)qPi>=-q²/2 exp(-q²), so HE term cancels exactly; HC is64 exp(-q²)[2(<O>meanrho_v+Tr(A(I-P)V P))+deltaLdelta<O>]. meanrho includes active+63aux-128 and Ref3 hole -e0. Onsite is exp(-q²)Tr([K,A]P). Hop² uses actual shared-endpoint CAR commutator and Gaussian exp(-|q_i+q_j|²). Wick is ONLY at t0.',
 'fourth_words': 'Right normal order every connected term as W prod_l s_rl, with W a product of m literal signed hopping factors and degree d in commuting native s_r=q_rPi+v_rrho. E commutes all s and [E,W]=W64(2 sum_word s_r+|sum_word k|²). Since native |k|²<=2, positive coefficient transitions are (m,d+1):128m and (m,d):128m². Onsite preserves each rho and changes hopping blocks with nuclear price2m||Ksite||. Matter-CAR step is (m+1,d):m Dcar, Dcar=2max_r sum_sameSpecies_sharedEndpoint b_j. Polynomial commutation is a finite difference in s with shifts k_r.k_j. Multinomial Holder yields (m+1,d-l):binom(d,l) max_r L_l(r), l1..d, where L_l=sum_ALL64charged b_j|k_r.k_j|^l. This retains cross-species Coulomb/Gauss terms. Start Bx=sum_incidentActive b_r at(m1,d0), apply three transitions to bound ad_H^4 n. It is a connected-word fourth bound, not a global H or extensive M norm.',
 'moments': 'All s_r commute, so spectral Holder bounds ||prod_l s_rl psi_t||<=max_r ||s_r^d psi_t|| for totaldegree d. Full-H moment hierarchy a_kprime<=sum_j<k binom(k,j)L_(k-j)a_j follows shifted powers and unitarity. The initial Slater tilted projector proof Lsecond=Tr(Pt V²)-Tr(Pt V Pt V)=.5||[Pt,V]||HS²<=.5TrV² gives variance parameter2sum_site v² per species, hence128sumv² for64; GaussianPi addsq²/8. Raw source mean<=2max|v| including Refhole. SubGaussian tail integration E|Xcenter|^(2k)<=2(2sigma²)^k k!; Minkowski adds mean. Only initial determinant semantics are used; positive triangular polynomial bounds apply to evolved vectors.',
 'remainder': 'Even density f(t) obeys |f(t)-f(0)-t²fsecond(0)/2|<=t^4 sup_|u|<=t |ffourth(u)|/24. For raw source mass.5 plus half-background mass.5 total multiplier1, use normalized channel moment maxima and preserve weights in the coefficient. Negative time follows the same absolute hierarchy. There is no panel quadrature or zero-time secondword replacement debt.',
 'physical': 'Actual physical local n_x has norm4. Local Haar spin rotations commute n_x but nonlocal Kato W does not. Rank4 Px gives ||Wdag Px W-Px||trace<=8||W-I||, hence eta_n<=8b1 packet_radius. E0eta0=b1 packet_radius, actual source price4(2eta0+eta0²). Fixed Tsrc evolution price8t epsilon; both t and0 density endpoints cost2eta_n and source prices cost twice when bounding density CHANGE.',
 'scope': 'Same N4/all129/AP/1+2link/active+63aux/Ref4to3/source law and parameters. Exact rational envelopes are outward without fitted constants; initial root-centre coefficients remain conditional floating arithmetic. A failed interval cannot support a sign claim or free photon pole. Original issue222 remains OPEN.'}


def bounds_for_paths(data: dict[str, Any], sp: list[Any], geo: dict[str, Any], geom: dict[str, Any]) -> tuple[list[F], list[F], F, list[F]]:
    paths = body.joined.old.native_paths(sp[0].profile, data['native'])
    looks = [{(p['axis'],p['length']):4*(abs(F(p['c0']))+abs(F(p['ci']))) for p in s.profile['paths']} for s in sp]
    species_b = [[look[p['axis'],p['length']] for p in paths]*2 for look in looks]
    b = species_b[0]
    dots = np.rint(geo['k']@geo['k'].T).astype(int)
    shared = (geo['y'][:,None]==geo['x'][None,:])|(geo['x'][:,None]==geo['y'][None,:])
    dcar = max(2*sum((bb[j] for j in np.flatnonzero(shared[i])),F(0)) for bb in species_b for i in range(len(b)))
    allb = [species_b[0][j]+63*species_b[1][j] for j in range(len(b))]
    force = [F(0)]+[max(sum((allb[j]*abs(int(dots[i,j]))**p for j in np.flatnonzero(dots[i])),F(0)) for i in range(len(b))) for p in range(1,4)]
    den=geom['denominator']; lint=geom['L_integer']; delta=geo['delta'].astype(int)
    vnum=delta@lint
    variance=max(F(int(geom['denominator']*np.sum(geo['k'][i]**2)-delta[i]@lint@delta[i]),8*den)+128*sum((F(int(z)**2,den**2) for z in vnum[i]),F(0)) for i in range(len(b)))
    mean=max(2*max(abs(F(int(z),den)) for z in row) for row in vnum)
    # Rational square-root uppers, and integer ceilings for subsequent roots,
    # are deliberately loose but genuinely outward, not float cushions.
    initial=[F(1)]
    for k in range(1,4):
        if k==1:
            squared=variance+mean*mean
            scale=10**18; n=math.isqrt((squared.numerator*scale*scale)//squared.denominator)+1
            initial.append(F(n,scale))
        else:
            moment=2*(2*variance)**k*math.factorial(k)
            scale=10**18
            scaled=moment*scale**(2*k)
            ceil=(scaled.numerator+scaled.denominator-1)//scaled.denominator
            initial.append((F(next_integer_root(ceil,2*k),scale)+mean)**k)
    # a_k(0)=sqrt E s^(2k); the previous expression uses an outward L2k norm
    # raised to k, exactly matching Minkowski and the desired vector norm.
    horizon=initial.copy()
    polys=[[initial[k]] for k in range(4)]
    for k in range(1,4):
        poly=[initial[k]]+[F(0)]*k
        for j in range(k):
            for n,c in enumerate(polys[j]):poly[n+1]+=math.comb(k,j)*force[k-j]*c/F(n+1)
        polys[k]=poly
        horizon[k]=sum((c*CLOCK**n for n,c in enumerate(poly)),F(0))
    return b,force,dcar,horizon


def next_integer_root(value: int, order: int) -> int:
    low=0; high=1
    while high**order<value:high*=2
    while low+1<high:
        mid=(low+high)//2
        if mid**order>=value:high=mid
        else:low=mid
    return high


def fourth_budget(B: F, dcar: F, onsite: F, force: list[F], moments: list[F]) -> tuple[F, dict[str,str]]:
    terms={(1,0):B}
    for _ in range(3):
        out=defaultdict(F)
        for (m,d),c in terms.items():
            out[m,d+1]+=c*128*m
            out[m,d]+=c*(128*m*m+2*m*onsite)
            out[m+1,d]+=c*m*dcar
            for l in range(1,d+1):out[m+1,d-l]+=c*math.comb(d,l)*force[l]
        terms=dict(out)
    return sum((c*moments[d] for (m,d),c in terms.items()),F(0)),{f'{m},{d}':str(c) for (m,d),c in sorted(terms.items())}


def parity_controls(data: dict[str,Any], sp: list[Any], captured: dict[str,bytes]) -> dict[str,Any]:
    receipt_name='joined_phi_u1_nonlocal_signal_interval_certificate_typed_verified_receipt_20261010.json'
    raw=(ROOT/receipt_name).read_bytes();r=json.loads(raw); archive=ROOT/r['immutable_inputs']['path']
    with zipfile.ZipFile(archive) as z:gdata=json.loads(z.read('actual_full_Q_AP_Coulomb_geometry.json'))
    gamma=np.array(gdata['Clifford_real'])+1j*np.array(gdata['Clifford_imaginary'])
    u=gamma[1]@gamma[3]; fullu=np.kron(np.eye(64),u)
    for i in range(4):
        if not np.array_equal(u@gamma[i].conj()@u.conj().T,(1 if i==0 else -1)*gamma[i]):raise ArithmeticError('Exact Clifford parity')
    if not np.array_equal(u@u.conj(),-np.eye(4)):raise ArithmeticError('Kramers sign')
    theta=np.linspace(-.37,.41,129)
    rows=[]
    for s in sp:
        hdef=float(np.max(abs(fullu@s.h(-theta).toarray().conj()@fullu.conj().T-s.h(theta).toarray())))
        pdef=float(np.max(abs(fullu@s.p.conj()@fullu.conj().T-s.p)))
        body.require(hdef);body.require(pdef)
        terms=body.joined.old.path_terms(s.profile,data['native'],data['geom'])
        bloch=body.joined.old.bloch_h(s.profile,data['native'],terms)
        pairing=np.arange(64).reshape(4,4,4)[::-1,::-1,::-1].ravel()
        blochdef=float(np.max(abs(u@bloch[pairing].conj()@u.conj().T-bloch)))
        body.require(blochdef)
        rows.append(dict(role=s.profile['role'],all_path_onsite_parity_defect=hdef,occupied_projector_parity_defect=pdef,all64_AP_Bloch_pairing_defect=blochdef))
    q=sp[0].empty_source;excess=q@q.conj().T/4
    mixdef=float(np.max(abs(fullu@excess.conj()@fullu.conj().T-excess)));body.require(mixdef)
    wrongspin=float(np.max(abs(sp[0].h(-theta).toarray().conj()-sp[0].h(theta).toarray())))
    # A generic spin-polarized channel need not be invariant; this adversary
    # rejects promoting mixture symmetry to an individual raw channel.
    wrongstate=max(float(np.max(abs(fullu@np.outer(q[:,a],q[:,a].conj()).conj()@fullu.conj().T-np.outer(q[:,a],q[:,a].conj())))) for a in range(4))
    if wrongspin<.1 or wrongstate<.01:raise ArithmeticError('Wrong symmetry/state controls blind')
    ap=body.joined.old.QINTEGERS
    if not np.array_equal((ap+(3-ap)+1)%4,np.zeros_like(ap)):raise ArithmeticError('AP inversion labels')
    captured[receipt_name]=raw
    captured['parity_integer_clifford_geometry.json']=json.dumps(gdata,sort_keys=True).encode()
    return dict(exact_spin_gaussian_integer=True,Kramers_square_exact=-1,AP_pairing='j ->3-j componentwise, all64',native_body=rows,raw_mixture_projector_defect=mixdef,wrong_spin_K_only_defect=wrongspin,wrong_single_channel_state_defect=wrongstate,Ref4_singlet_detU=body.packed(complex(np.linalg.det(u))),E0_spin_singlet_defect=float(np.max(abs(u@u.conj().T-np.eye(4)))))


def second_coefficients(x: int, sp: list[Any], geo: dict[str,Any], channels: list[dict[str,Any]], lp: np.ndarray) -> dict[str,float]:
    blocks=current.blocks(sp[0]);incident=np.flatnonzero((geo['x']==x)|(geo['y']==x))
    electric=0j; onsite=0j
    auxmean,_=history.density_stats(sp[1].p)
    states=[(-.5,sp[0].p,False)]+[(c['weight'],c['p'],True) for c in channels]
    for weight,p,hole in states:
        mean,_=history.density_stats(p);rho=mean+63*auxmean-128
        if hole:rho[0]-=1
        pb=p.reshape(64,4,64,4).transpose(0,2,1,3)
        for i in incident:
            row,col=int(geo['x'][i]),int(geo['y'][i]);a=blocks[i];v=geo['delta'][i]@lp
            covblock=v[col]*pb[col,row]-(p[4*col:4*col+4]*np.repeat(v,4)[None,:])@p[:,4*row:4*row+4]
            value=np.trace(a@pb[col,row]);connected=np.trace(a@covblock)
            w=int(col==x)-int(row==x);damping=math.exp(-float(geo['q'][i]@geo['q'][i]))
            electric+=weight*w*64*damping*(2*(value*(v@rho)+connected)+(v@geo['delta'][i])*value)
            onsite+=weight*w*damping*np.trace((sp[0].onsite@a-a@sp[0].onsite)@pb[col,row])
    excess=sum((c['weight']*c['p'] for c in channels),np.zeros_like(sp[0].p))-.5*sp[0].p
    hop,absolute=history.second_word(x,geo,blocks,excess)
    body.require(abs(electric.imag),1e-7);body.require(abs(onsite.imag),1e-7)
    return dict(HE_second_exact=0.,HC_second=-float(electric.real),onsite_hop_second=-float(onsite.real),hop_hop_second=-hop,complete_second=-float(electric.real+onsite.real)-hop,hop_hop_absolute_sum=absolute)


def initial_coefficient_oracle() -> dict[str,float]:
    """Independent full finite-CAR second derivative, only an initial oracle."""
    creators=[]
    for i in range(4):
        c=np.zeros((16,16))
        for mask in range(16):
            if not (mask>>i)&1:c[mask|(1<<i),mask]=(-1)**((mask&((1<<i)-1)).bit_count())
        creators.append(c)
    def dg(a: np.ndarray) -> np.ndarray:
        return sum((a[i,j]*creators[i]@creators[j].T for i in range(4) for j in range(4)),np.zeros((16,16),complex))
    w,_=np.linalg.qr(np.array([[1.,2.],[2.,-1.],[1.,1.],[-2.,1.]]));p=w@w.T
    psi=np.zeros(16)
    for i in range(4):
        for j in range(i+1,4):psi[(1<<i)|(1<<j)]=np.linalg.det(w[[i,j]])
    a=np.array([[.2,1.,-.7,.4],[1.,-.3,.8,.1],[-.7,.8,.5,-.9],[.4,.1,-.9,.6]])
    v=np.diag([.6,-.4,.2,-.1]);n=np.diag([1.,0.,0.,0.])
    hc=64*dg(v)@dg(v);hm=dg(a);h=hc+hm;number=dg(n)
    comm=h@number-number@h
    direct=-complex(psi.conj()@(h@comm-comm@h)@psi)
    coul=0j;wrong=0j
    for i in range(4):
        for j in range(4):
            o=np.zeros((4,4));o[i,j]=a[i,j]
            weight=int(j==0)-int(i==0);shift=v[i,i]-v[j,j]
            mean=np.trace(o@p);joint=mean*np.trace(v@p)+np.trace(o@(np.eye(4)-p)@v@p)
            plus=mean*np.trace(v@p)+np.trace(o@(np.eye(4)+p)@v@p)
            coul+=64*weight*(2*shift*joint+shift*shift*mean)
            wrong+=64*weight*(2*shift*plus+shift*shift*mean)
    inner=a@n-n@a;matter=np.trace((a@inner-inner@a)@p)
    predicted=-coul-matter;defect=abs(direct-predicted)
    body.require(defect,1e-10)
    if abs(direct-(-wrong-matter))<.01:raise ArithmeticError('Wrong Wick-plus density coefficient control blind')
    return dict(full_finite_CAR_second_defect=defect,full_second=float(direct.real),HC_second=float(-coul.real),matter_squared_second=float(-matter.real),wrong_Wick_plus_defect=abs(direct+wrong+matter))


def run(output: Path) -> dict[str,Any]:
    started=time.monotonic();data,sp,inputs,_=body.full_inputs();native=data['native'];geo=current.geometry(data,sp[0]);geom=prices.exact_geometry(native,geo)
    captured={**body.CAPTURED,**inputs};channels=history.initial_channels(sp[0])
    parity=parity_controls(data,sp,captured);cross=history.controls(native,geo,geom)
    parity['independent_initial_second_CAR_oracle']=initial_coefficient_oracle()
    b,force,dcar,moments=bounds_for_paths(data,sp,geo,geom)
    source_name='joined_phi_u1_actual_compact_source_action_certificate_verified_receipt_20261010.json'
    transfer_name='joined_phi_u1_nonlocal_body_transport_verified_source_transfer_receipt_20261010.json'
    for name in (source_name,transfer_name):captured[name]=(ROOT/name).read_bytes()
    source=json.loads(captured[source_name]);transfer=json.loads(captured[transfer_name])
    epsilon=F(source['action_input']['epsilon']['upper']);radius=F(source['source_weights']['prepared_radius_L2']['upper']);b1=F(transfer['fundamental_active_W1_upper_exact'])
    eta0=b1*radius;etan=8*eta0;transfer_error=8*CLOCK*epsilon+2*etan+8*(2*eta0+eta0*eta0)
    rows=[]
    for i in np.flatnonzero(geo['y']==0):
        x=int(geo['x'][i]);incident=np.flatnonzero((geo['x']==x)|(geo['y']==x))
        onsite_upper=max(abs(F(s.profile['onsite'])) for s in sp)
        B=sum((b[j] for j in incident),F(0));fourth,terms=fourth_budget(B,dcar,onsite_upper,force,moments)
        error=CLOCK**4*fourth/24;c=second_coefficients(x,sp,geo,channels,native['Lplus']);signal=float(CLOCK**2)*c['complete_second']/2
        # No invented roundoff enclosure; intervals about this floating centre
        # are conditional until an initial contraction outward supplier lands.
        rows.append(dict(path=int(i),row=x,col=0,length=int(geo['k'][i]@geo['k'][i]),second_coefficients_conditional=c,density_change_second_centre_conditional=signal,fourth_derivative_upper_exact=str(fourth),connected_fourth_error_upper_exact=str(error),connected_word_positive_terms=terms,physical_change_transfer_error_exact=str(transfer_error),density_endpoint_eta_exact=str(etan),source_E0_eta0_exact=str(eta0),receiver_Ex_eta_exact=str((b1+F(int(geo['k'][i]@geo['k'][i])))*radius),physical_margin_conditional=abs(signal)-float(error+transfer_error),physical_controlled_conditional=abs(signal)>float(error+transfer_error),nohop_density_change_exact='0',body_error_to_signal_conditional=float(error)/abs(signal) if signal else None))
    if len(rows)!=12:raise ArithmeticError('Actual12 density scope')
    onsite_body=np.kron(np.eye(64),sp[0].onsite)
    fullh=sp[0].h(np.zeros(129)).toarray()
    nohop_defect=0.;wrong_hop_min=math.inf
    for row in rows:
        px=np.zeros((256,256));x=row['row'];px[4*x:4*x+4,4*x:4*x+4]=np.eye(4)
        nohop_defect=max(nohop_defect,float(np.max(abs(onsite_body@px-px@onsite_body))))
        wrong_hop_min=min(wrong_hop_min,float(np.max(abs(fullh@px-px@fullh))))
    if nohop_defect!=0 or wrong_hop_min<.1:raise ArithmeticError('Physical nohop density assertion not reached')
    parity['actual12_nohop_onsite_commutator_exact_defect']=nohop_defect
    parity['wrong_hop_density_commutator_minimum']=wrong_hop_min
    numerical=output.with_suffix('.npz');np.savez_compressed(numerical,signed_k=geo['k'],signed_q=geo['q'],signed_delta=geo['delta'],active_initial_projector=sp[0].p,auxiliary_initial_projector=sp[1].p,raw_channel_weights=[c['weight'] for c in channels],raw_channel_projectors=[c['p'] for c in channels])
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':captured.setdefault(Path(file).name,Path(file).read_bytes())
    for name,raw in captured.items():
        if name=='parity_integer_clifford_geometry.json':continue
        if (ROOT/name).read_bytes()!=raw:raise ArithmeticError('Input/source mutation '+name)
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(captured.items()):z.writestr(name,raw)
    result=dict(status='PARITY_DENSITY_FOURTH_BOUND_CONDITIONAL_CONTROLLED' if all(r['physical_controlled_conditional'] for r in rows) else 'PARITY_DENSITY_FOURTH_BOUND_NOT_CONTROLLED',clock_exact=str(CLOCK),clock_adopted=False,plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision=REVISION,proof=PROOF,receiver_rows=rows,controls=dict(parity=parity,Coulomb=cross),raw_mass_exact='1/2',raw_fourRef_weights=[c['weight'] for c in channels],all63_auxiliaries=True,all129=True,positive_moment_horizon_upper_exact=[str(v) for v in moments],force_Lp_max_exact=[str(v) for v in force],Dcar_exact=str(dcar),input_bytes=dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files={name:hashlib.sha256(raw).hexdigest() for name,raw in sorted(captured.items())}),numerical_capsule=dict(path=numerical.name,sha256=hashlib.sha256(numerical.read_bytes()).hexdigest()),elapsed_seconds=time.monotonic()-started,open='Initial second contractions floating conditional; connected fourth bound failure is a closure failure at NEW2^-16, not a no-go or sign/pole claim.')
    output.write_text(json.dumps(result,indent=2,default=history.encode)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):result=run(args.output)
    print(json.dumps(dict(status=result['status'],actual12=len(result['receiver_rows']),max_error_to_signal=max(r['body_error_to_signal_conditional'] for r in result['receiver_rows']),elapsed_seconds=result['elapsed_seconds'])),flush=True)
