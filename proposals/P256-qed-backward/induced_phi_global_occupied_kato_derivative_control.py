"""Real radial occupied-Kato envelopes; not whole-source or ground admission."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import mpmath as mp
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import induced_phi_nonuniform_fermionic_source_unitary as source
import joined_phi_u1_finite_car as bridge

ROOT = Path(__file__).resolve().parent
OLD = ROOT/'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'
OLD_SHA = 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6'
JETS = ROOT/'induced_phi_global_normal_h_jet_control_receipt.json'
JETS_SHA = 'e48a53776184f9eda572d5f06ef061f880cfb13935b41f3a707b25d1994e58b8'
R = F(1,1 << 20)
sys.set_int_max_str_digits(0)
DERIVATION = {
    'domain': 'Actual source.field_ray: Phi=Phi0+Q0 slow/8+Q(slow) fast, zero-mean fast, ordered SU4 colour links. Real full5760 body ball R and3R collar. Polar NORMAL Q satisfies norm<9/8,Q1<512,Q2<2^20,Q3<2^33 on collar by exact Cauchy comparison. This is not the normal polar frame itself and not a theta-following occupied frame.',
    'h3': 'F1=17/8+512rho,F2=1024+2^20rho,F3=3*2^20+2^33rho. Bphi=8vp. Bc=16vc+pc F1rho. D1=Bphi F1+Bc; D2=Bphi F2+2pc F1+sum L²(hop+c1 F1rho/sqrt2). D3=Bphi F3+6pc F2+6(sum L²c1)F1+2sum L³(hop+c1 F1rho). Ordered unitary-path k-jets are <=L^k product||T deltaC||, with ||T deltaC||<=||deltaC||/sqrt2; both endpoint Phi and both path orientations retained. Three mixed secondPhi/colour and three firstPhi/secondcolour placements are covered by the conservative6 coefficients. No frozen pointwise per-q normalizer.',
    'spectral': 'Unscaled Hermitian h has distance d>=gap_mean/2-2D1rho>0. Neutral physical -Lambda h swaps P with I-P; [d(I-P),I-P]=[dP,P], hence radial transport and prices unchanged and Lambda cancels. Imaginary-axis Riesz derivative integrals, ||(h-iomega)^-1||<=(d²+omega²)^-1/2, give P1<=D1/(2d),P2<=(2/3)D1²/d²+D2/(2d),P3<=(3/2)D1³/d³+2D1D2/d²+D3/(2d), using pi>3. These are all mixed real multilinear derivatives, not a complex tube claim.',
    'radial_W': 'Define K(t,r)=[DP(tr)[r],P(tr)] and d_t W=K W,W(0)=I, exactly the nonlinear radial source construction, not affine h interpolation. Real K is anti-Hermitian. For unit u,v integrate derivative bounds: A=2P1+rho(P2+P1²) bounds integral||DK[u]||. B=(2/3)rho P3+2P2+2rho P1P2+2P1² bounds integral||D²K[u,v]|| (last term is harmless even where commutators cancel). Differentiated unitary Duhamel gives ||DW||<=A,||D²W||<=B+A²: two ordered insertion simplexes sum to the full squared integral, no exponential factor.',
    'consumer': 'Exact rational upper A and B+A² lie below the LOWER endpoint of the repaired archived b1/b2, even on3R collar. Thus those conservative real radial frame envelopes can be retained. Full Fock lift in fixed counts128,129,8064: directional first<=sum N A, second<=sum N(B+A²)+(sum N A)². Ambient gradient requires ceil(sqrt5760)=76; Hessian Frobenius requires5760 times directional bound. No per-q or directional-to-tensor substitution.',
    'control': 'Actual wholeN4 colour ray, all one/two/AP paths. Independent analytic ordered H1/H2 versus native h_jet and finite nonlinear projector differences; projector second derivative solves idempotence and [h,P] equations with ONLY occupied/unoccupied gap division. W1=[P1,P],W2=[P2,P]+W1² at zero. Omitting actual second Hamiltonian curvature is a wrong control. Finite ray test outside packetR is labelled a numerical control, not interval proof or mean/source adoption.',
    'limits': 'Conditional exact mean root/gap and profile hypotheses remain. This certificate proves only occupied radial Kato derivative envelopes and their dimension-correct lifts. No whole supplier action, physical SOURCE, ground, current, pole, common cone or interacting body propagation admission.'
}


def upper(x: dict[str, Any]) -> F:
    return F(x['upper'])


def lower(x: dict[str, Any]) -> F:
    return F(x['lower'])


def record(x: F) -> dict[str, Any]:
    return {'exact':str(x),'decimal':mp.nstr(mp.mpf(x.numerator)/x.denominator,25)}


def ray_jets(profile: dict[str, Any], native: dict[str, Any], fields: np.ndarray,
             generators: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    result = [np.zeros((256,256),complex) for _ in range(3)]
    for x in range(64):
        result[0][4*x:4*x+4,4*x:4*x+4] = float(profile['onsite'])*fields[x,0]
    for path in field.native_paths(profile,native):
        x,y = int(path['source']),int(path['target'])
        gs = generators[list(path['edges'])]
        u1 = 1j*gs.sum(axis=0)
        u2 = -sum((g@g for g in gs),np.zeros((4,4),complex))
        for j in range(len(gs)):
            for k in range(j+1,len(gs)):
                u2 -= 2*gs[j]@gs[k]
        fx = np.einsum('m,mij->ij',path['coefficients'],fields[x])
        fy = np.einsum('m,mij->ij',path['coefficients'],fields[y])
        for h,u in zip(result,(np.eye(4),u1,u2),strict=True):
            hop = path['seam']*(fx@u+u@fy)/2
            h[4*x:4*x+4,4*y:4*y+4] += hop
            h[4*y:4*y+4,4*x:4*x+4] += hop.conj().T
    sign = -1 if profile['scale']=='heavy' else 1
    return sign*result[0],sign*result[1],sign*result[2]


def projection_jets(h: np.ndarray,h1: np.ndarray,h2: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    w,u = np.linalg.eigh(h)
    if not (w[127]<0<w[128]):
        raise ArithmeticError('Actual negative projector is not rank128')
    a,b = u.conj().T@h1@u,u.conj().T@h2@u
    diff = w[:128,None]-w[None,128:]
    p = np.diag(np.r_[np.ones(128),np.zeros(128)]).astype(complex)
    p1 = np.zeros_like(p)
    p1[:128,128:] = a[:128,128:]/diff
    p1[128:,:128] = p1[:128,128:].conj().T
    p2 = np.zeros_like(p)
    sq = p1@p1
    p2[:128,:128] = -2*sq[:128,:128]
    p2[128:,128:] = 2*sq[128:,128:]
    comm = a@p1-p1@a
    p2[:128,128:] = (b[:128,128:]-2*comm[:128,128:])/diff
    p2[128:,:128] = p2[:128,128:].conj().T
    return u@p@u.conj().T,u@p1@u.conj().T,u@p2@u.conj().T


def finite_controls(data: dict[str, Any]) -> list[dict[str, Any]]:
    native,geom = data['native'],data['geom']
    fields = np.broadcast_to(native['frame'],(64,4,4,4)).copy()
    rng = np.random.default_rng(2221009)
    c = rng.normal(size=(192,15));c /= np.linalg.norm(c)
    generators = np.einsum('ea,aij->eij',c,geom['M']/np.sqrt(8))
    links = np.broadcast_to(np.eye(4),(192,4,4)).copy().astype(complex)
    delta = 2.**-8
    rows = []
    for profile in data['profiles']:
        h,h1,h2 = ray_jets(profile,native,fields,generators)
        native_h,native_h1 = source.h_jet(profile,native,fields,np.zeros_like(fields),links,1j*generators)
        literal_error = max(float(np.linalg.norm(h-native_h)),float(np.linalg.norm(h1-native_h1)))
        p,p1,p2 = projection_jets(h,h1,h2)
        points = []
        for t in (-delta,delta):
            actual_links = np.asarray([expm(1j*t*g) for g in generators])
            ht = bridge.assemble(profile,native,fields,actual_links,np.zeros(192))
            wt,ut = np.linalg.eigh(ht)
            points.append(ut[:,wt<0]@ut[:,wt<0].conj().T)
        first_error = float(np.linalg.norm((points[1]-points[0])/(2*delta)-p1))
        second_error = float(np.linalg.norm((points[1]+points[0]-2*p)/(delta*delta)-p2))
        _,_,wrongp2 = projection_jets(h,h1,np.zeros_like(h2))
        k = p1@p-p@p1
        second = p2@p-p@p2+k@k
        wrongsecond = wrongp2@p-p@wrongp2+k@k
        wrong = float(np.linalg.norm(second-wrongsecond))
        if literal_error>1e-10 or first_error>2e-6 or second_error>2e-6 or wrong<1e-7:
            raise ArithmeticError(f'Actual nonlinear occupied-ray control failed: {profile["role"]}, {literal_error}, {first_error}, {second_error}, {wrong}')
        rows.append({'role':profile['role'],'all64_sites_all_one_two_AP_paths':True,
                     'ray_colour_coordinate_norm':float(np.linalg.norm(c)),'finite_difference_TEST_step_outside_source_R':delta,
                     'native_h_and_first_jet_error':literal_error,'projector_first_FD_error':first_error,
                     'projector_second_FD_error':second_error,'W_first_norm':float(np.linalg.norm(k,2)),
                     'W_second_norm':float(np.linalg.norm(second,2)),
                     'wrong_omit_h_second_W_second_error':wrong})
    return rows


def run() -> dict[str, Any]:
    start=time.monotonic()
    own=Path(__file__).read_bytes();raw=OLD.read_bytes();jraw=JETS.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=OLD_SHA or hashlib.sha256(jraw).hexdigest()!=JETS_SHA:
        raise ArithmeticError('Pinned source or repaired jet input changed')
    old,jets=json.loads(raw),json.loads(jraw)
    mean=old['input_descriptor']['exact_NEW_allq_BO_mean']
    centres=[F(x) for x in mean['centre_exact_rational']]
    field.full.iv.dps=100;mp.mp.dps=100
    data=field.load_mean(g_override=[centres[0],*([centres[1]]*3)],mean_descriptor={'law':'Saved mean display for NEW occupied derivative control; not an adopted root'})
    params=mean['parameters']
    intervals=[field.full.enclosure(F(params[k]['lower']),F(params[k]['upper'])) for k in ('g0','gs')]
    data['g']=[intervals[0],*([intervals[1]]*3)]
    ex,iv=field.full.exact,field.full.iv
    cases=[]
    for label,rho in (('support',R),('collar',3*R)):
        qbounds=[F(n**n)*F(9,8)/(F(1,256)-rho)**n for n in (1,2,3)]
        if not all(x<y for x,y in zip(qbounds,(512,1 << 20,1 << 33),strict=True)):
            raise ArithmeticError('Polar Q derivative collar extension fails')
        f1=ex(F(17,8)+512*rho);f2=ex(F(1024)+(1 << 20)*rho);f3=ex(F(3*(1 << 20))+(1 << 33)*rho)
        first=F(0);second=F(0);rows=[]
        for profile,archived in zip(data['profiles'],jets['profiles'],strict=True):
            if profile['role']!=archived['role']:
                raise ArithmeticError('Profile ordering changed')
            bounds=field.exact_operator_bounds(profile,data['g'])
            vp,vc=ex(upper(bounds['Phi_vertex_operator_upper'])),ex(upper(bounds['colour_vertex_operator_upper']))
            pc=sum(p['length']*(abs(p['c0'])+abs(p['ci'])) for p in profile['paths'])
            pc2=sum(p['length']**2*(abs(p['c0'])+abs(p['ci'])) for p in profile['paths'])
            cc,ccc=ex(0),ex(0)
            for pth in profile['paths']:
                c1=ex(abs(pth['c0'])+abs(pth['ci']))
                hop=abs(ex(pth['c0']))*iv.sqrt(data['g'][0])+abs(ex(pth['ci']))*iv.sqrt(data['g'][pth['axis']+1])
                cc+=ex(pth['length']**2)*(hop+c1*f1*ex(rho)/iv.sqrt(ex(2)))
                ccc+=ex(pth['length']**3)*(hop+c1*f1*ex(rho))
            d1=8*vp*f1+16*vc+ex(pc)*f1*ex(rho)
            d2=8*vp*f2+2*ex(pc)*f1+cc
            d3=8*vp*f3+6*ex(pc)*f2+6*ex(pc2)*f1+2*ccc
            D1,D2,D3=(upper(field.full.bounds(x)) for x in (d1,d2,d3))
            distance=lower(bounds['minimum_gap_unscaled'])/2-2*D1*rho
            if distance<=0:
                raise ArithmeticError('Real radial gap is not positive')
            P1=D1/(2*distance)
            P2=F(2,3)*D1*D1/(distance*distance)+D2/(2*distance)
            P3=F(3,2)*D1**3/distance**3+2*D1*D2/distance**2+D3/(2*distance)
            A=2*P1+rho*(P2+P1*P1)
            B=F(2,3)*rho*P3+2*P2+2*rho*P1*P2+2*P1*P1
            W2=B+A*A
            a1,a2=lower(archived['directional_Kato_first']),lower(archived['directional_Kato_second'])
            if not (A<a1 and W2<a2):
                raise ArithmeticError('Archived repaired frame envelope cannot be retained')
            count=archived['fixed_bulk_particle_count']
            first+=count*A;second+=count*W2
            rows.append({'role':profile['role'],'count':count,'D1':record(D1),'D2':record(D2),'D3':record(D3),
                         'real_gap_lower':record(distance),'projector_first_upper':record(P1),'projector_second_upper':record(P2),
                         'projector_third_upper':record(P3),'W_first_upper':record(A),'W_second_upper':record(W2),
                         'archived_b1_lower':record(a1),'archived_b2_lower':record(a2),'archived_envelopes_proven_conservative':True})
        cases.append({'domain':label,'radius':str(rho),'polar_Cauchy_upper':[str(x) for x in qbounds],'profiles':rows,
                      'Fock_directional_first_upper':record(first),'Fock_directional_second_upper':record(second+first*first),
                      'full5760_gradient_upper':record(76*first),'full5760_Hessian_Frobenius_upper':record(5760*(second+first*first))})
    with threadpool_limits(limits=1):
        controls=finite_controls(data)
    captures={**field.CAPTURED,**source.SOURCE_BYTES,Path(__file__).name:own,Path(bridge.__file__).name:Path(bridge.__file__).read_bytes()}
    stable={name:{'sha256':hashlib.sha256(b).hexdigest(),'unchanged':b==(ROOT/name).read_bytes()} for name,b in captures.items()}
    if not all(x['unchanged'] for x in stable.values()) or OLD.read_bytes()!=raw or JETS.read_bytes()!=jraw:
        raise ArithmeticError('Source or input changed during execution')
    return {'derivation':DERIVATION,'conditional_cases':cases,'finite_actual_controls':controls,
            'inputs_sha256':{str(OLD):OLD_SHA,str(JETS):JETS_SHA},'source_at_execution':stable,'own_source_utf8':own.decode(),
            'governing_plan':'issue-222-induced-photon-delta13-continuation-20261008','governing_revision':'f4e8ad04d46c6e0812cf213dd46e856c682e142a69df13092b04c75ffca413bc',
            'template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
            'runtime':{'executable':sys.executable,'python':sys.version,'numpy':np.__version__},'elapsed_seconds':time.monotonic()-start}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'cases':[{'domain':x['domain'],'profiles':[{'role':p['role'],'W1':p['W_first_upper']['decimal'],'W2':p['W_second_upper']['decimal']} for p in x['profiles']]} for x in result['conditional_cases']],
                      'finite_controls':result['finite_actual_controls'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
