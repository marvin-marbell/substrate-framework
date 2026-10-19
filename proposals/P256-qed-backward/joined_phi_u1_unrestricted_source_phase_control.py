"""Unrestricted N4 Pi/Coulomb source phase, not a dressed pole or vacuum."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from itertools import product
from pathlib import Path
import sys
import time
from typing import Any

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import joined_phi_u1_finite_car as bridge

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'
OLD_SHA='da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6'
G=F(64);S=F(2);M=129
DERIVATION={
    'law_and_state':'NEW body H=UE/a*(sum129PiQ²+rho_allcharged Lplus rho_allcharged)+dGamma_active(h(meanPhi,Theta))+63dGamma_aux(h(meanPhi,Theta)); UE/a=64 exactly, no1/2 convention. Theta real unwrapped coexact+harmonic coordinates. Background active128+aux63x128 andRef4, Gaussian densityvariance2 times fixed actual negative-mean Slaters. Source E=.5sum4 a_active0alpha^dag Ralpha mapsRef4->Ref3/active129; Ref is static except its full Coulomb charge. Neutral identical factor cancels only in same-H pair. This is a NEW preparation, not originalSOURCE or theta-following ground.',
    'operator':'H has bounded smooth finiteCAR matter/Coulomb matrix potential on realR129, so selfadjoint on Gaussian H² domain. E has norm<=1 onRef4 and E†E=(4-n_active0)/4. Fullrho obeys rho_src E=E rho_bg, hence bothHE and fullHC intertwine co-localE. K=HsrcE-EHbg=.5sum4 a†(h_active e0alpha)Ralpha and K†K=.25sum a(h ealpha)a†(h ealpha)<=Hmax². [Hbg,E†E]=E†K-K†E, NOT a plus sign.',
    'paired_derivatives':'A(t)=<E Ubg(t)Psi,Usrc(t)E Psi>. A(0)=.5, Aprime=-i<UbgPsi,K†UsrcE Psi>, Asecond=-<UbgPsi,L†UsrcE Psi>, L=HsrcK-KHbg. Matter component is creation h²-column, norm<=Hmax², NOT full8193-particle energy timesK. Pi commutator differentiates hTheta and fullCoulomb term is retained below.',
    'coulomb':'For creation atx withRefhole0, delta=delta_x-delta_0. Fullrho normvector<=129sqrt64, ||Lplus||<=1/2, ||delta||<=sqrt2. HC commutator on that column is g[2delta Lplus rho+delta Lplus delta]. Its norm<=g(2sqrt2*(1/2)*129*8+1)<99136 using sqrt2<3/2. Onsite x0 commutator vanishes. Sum all offsite block norms yields ||L_C||<=99136 Hhop_active. Countsall64, notoneflavour.',
    'pi':'Each path incidence q=Q†inc has ||q||<=sqrt2. FourRef creation-column Hilbert-Schmidt/CAR bound gives joint||gradTheta K||<=sqrt2 Hhop<=3Hhop/2 and ||DeltaTheta K||<=2Hhop. Therefore ||L_E†xi||<=g(2Hhop||xi||+3Hhop||Pi xi||). For source evolution xi=Usrc E Psi, ||xi||<=1 and ||Pi xi(0)||<=sqrt(129/8)<5. Heisenberg force is onlygradTheta matter (Coulomb/Pi commutePi); jointforce<=sqrt(2M)sumN Hhop<17sumN Hhop=Fforce. Duhamel gives ||Pi xi(t)||<=5+tFforce; unboundedPi is NOT replaced by a global bounded norm.',
    'remainder':'Hence |A(t)-(.5-i t m1)|<=.5t²[Hmax²+(99136+17g)Hhop]+(g Hhop Fforce/2)t³. The cubic coefficient is integral(t-s)*3gHhop Fforce*s ds= gHhop Fforce*t³/2. Simultaneously |A(t)-.5|<=3tHmax is a coarse all-state channel disk. These are actual unrestricted-H estimates, NOT finiteword convergence or a Trotter certificate.',
    'first_moment':'m1=(1/2V)sum_APk sum_mu gmu dmu(k)dbar_mu(k)/sqrt(sum_mu gmu dmu(k)^2). APk=(2n+1)pi/4 has exact radical sin/cos; dbar uses realGaussian Peierlscharacter exp(-s||Q†inc||²/2). Periodic electricL has exacteigenvalues0,2,4,2 peraxis; exactpathvariances are43/64 length1 and129/80 length2. Pi/fullCoulomb initialcommutator is zero, not their dynamics. Interval formula proves initialm1 only under supplied exactRoot/Clifford hypotheses.',
    'scope':'Real same-H source phase and survival for NEW fixed meanSlater Gaussian body with ALLN4/129electric/64charged/4Ref/fullCoulomb. No bodyground, physicaldressedmass/pole, commoncone/current/Compton or originalSOURCE admission. Embedding in compactPhi quantum source requires separatelyaccepted global action/root/supplier hypotheses; it is not performed by this body-only proof.'
}


def bounds(x: Any) -> dict[str, Any]:
    return field.full.bounds(x)


def upper(x: Any) -> F:
    return F(bounds(x)['upper'])


def lower(x: Any) -> F:
    return F(bounds(x)['lower'])


def record(x: F) -> dict[str, Any]:
    return {'exact':str(x),'decimal':mp.nstr(mp.mpf(x.numerator)/x.denominator,25)}


def path_variance(length: int) -> F:
    electric=(0,2,4,2)
    cosperiodic=(1,0,-1,0)
    correction=F(0)
    for ns in product(range(4),repeat=3):
        lam=sum(electric[n] for n in ns)
        if lam:
            correction+=F(2*(1-cosperiodic[(length*ns[0])%4]),64*lam)
    return F(length)-correction


def moment(profile: dict[str, Any], g: list[Any], chars: dict[int, Any]) -> Any:
    ex,iv=field.full.exact,field.full.iv
    half=iv.sqrt(ex(F(1,2)))
    cos1=(half,-half,-half,half);sin1=(half,half,-half,-half)
    sin2=(ex(1),ex(-1),ex(1),ex(-1))
    total=ex(0)
    for ns in product(range(4),repeat=3):
        d=[ex(profile['onsite']),ex(0),ex(0),ex(0)]
        db=d.copy()
        for p in profile['paths']:
            n=ns[p['axis']];L=p['length']
            if L not in (1,2):
                raise ArithmeticError('Actual path ABI not one/two')
            co=cos1[n] if L==1 else ex(0)
            si=sin1[n] if L==1 else sin2[n]
            d[0]+=2*ex(p['c0'])*co
            d[p['axis']+1]-=2*ex(p['ci'])*si
            db[0]+=2*ex(p['c0'])*chars[L]*co
            db[p['axis']+1]-=2*ex(p['ci'])*chars[L]*si
        energy=iv.sqrt(sum(gv*dv*dv for gv,dv in zip(g,d,strict=True)))
        if field.full.endpoint(energy,0)<=0:
            raise ArithmeticError('Actual AP mean band is not gapped')
        total+=sum(gv*dv*bv for gv,dv,bv in zip(g,d,db,strict=True))/energy
    return total/128


def finite_initial_control(data: dict[str, Any], qnorms: dict[int,F], m1: Any) -> dict[str, Any]:
    native=data['native'];profile=data['profiles'][1]
    fields=np.broadcast_to(native['frame'],(64,4,4,4)).copy()
    links=np.broadcast_to(np.eye(4),(192,4,4)).copy().astype(complex)
    h0=bridge.assemble(profile,native,fields,links,np.zeros(192))
    w,u=np.linalg.eigh(h0);p=u[:,w<0]@u[:,w<0].conj().T
    hbar=np.zeros_like(h0);wrong=np.zeros_like(h0);errors=[]
    for x in range(64):
        hbar[4*x:4*x+4,4*x:4*x+4]=wrong[4*x:4*x+4,4*x:4*x+4]=float(profile['onsite'])*fields[x,0]
    for path in field.native_paths(profile,native):
        x,y=int(path['source']),int(path['target']);L=len(path['edges'])
        inc=np.zeros(192);inc[list(path['edges'])]=1
        qnorm=float(np.linalg.norm(native['Q'].T@inc)**2)
        errors.append(abs(qnorm-float(qnorms[L])))
        fx=np.einsum('m,mij->ij',path['coefficients'],fields[x]);fy=np.einsum('m,mij->ij',path['coefficients'],fields[y])
        b=path['seam']*(fx+fy)/2
        for h,char in ((hbar,np.exp(-float(S)*qnorm/2)),(wrong,np.exp(-float(S)*L*float(qnorms[1])/2))):
            hop=char*b;h[4*x:4*x+4,4*y:4*y+4]+=hop;h[4*y:4*y+4,4*x:4*x+4]+=hop.conj().T
    numeric=np.trace((hbar@(np.eye(256)-p))[:4,:4])/4
    omitted=np.trace((wrong@(np.eye(256)-p))[:4,:4])/4
    expected=float((field.full.endpoint(m1,0)+field.full.endpoint(m1,1))/2)
    error=abs(numeric-expected)
    difference=abs(numeric-omitted)
    raw=.25*(4-np.trace(p[:4,:4]).real)
    frozen=np.trace((h0@(np.eye(256)-p))[:4,:4])/4
    frozen_difference=abs(numeric-frozen)
    character_difference=abs(np.exp(-float(S*qnorms[2])/2)-np.exp(-float(S*2*qnorms[1])/2))
    if max(errors)>1e-12 or error>1e-10 or abs(raw-.5)>1e-12 or frozen_difference<1e-5 or character_difference<.01:
        raise ArithmeticError(f'Actual initial source control: Q_error={max(errors)}, m1_numeric={numeric}, m1_interval={expected}, raw={raw}, frozen_difference={frozen_difference}, character_difference={character_difference}')
    return {'raw_source_norm_squared':float(raw),'initial_m1_numeric_real':float(numeric.real),'initial_m1_numeric_imag':float(numeric.imag),
            'interval_formula_numeric_error':float(error),'exact_periodic_path_variance_native_Q_error':max(errors),
            'wrong_drop_two_link_cross_character_m1_real':float(omitted.real),'wrong_character_difference':float(difference),
            'two_link_source_projection_observed_blind':bool(difference<1e-12),
            'wrong_two_link_primitive_character_difference':float(character_difference),
            'wrong_frozen_Theta_m1_real':float(frozen.real),'wrong_frozen_Theta_difference':float(frozen_difference),
            'status':'Actual N4 native ordered AP paths; numeric control, not a floating interval certificate'}


def run() -> dict[str, Any]:
    start=time.monotonic();own=Path(__file__).read_bytes();raw=OLD.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=OLD_SHA:
        raise ArithmeticError('Pinned source mean changed')
    cap=json.loads(raw);mean=cap['input_descriptor']['exact_NEW_allq_BO_mean'];centres=[F(x) for x in mean['centre_exact_rational']]
    field.full.iv.dps=100;mp.mp.dps=100
    data=field.load_mean(g_override=[centres[0],*([centres[1]]*3)],mean_descriptor={'law':'NEW unrestricted-body source phase at saved mean; not adopted root'})
    params=mean['parameters'];g0,gs=[field.full.enclosure(F(params[k]['lower']),F(params[k]['upper'])) for k in ('g0','gs')]
    g=[g0,gs,gs,gs];ex,iv=field.full.exact,field.full.iv
    qnorms={L:path_variance(L) for L in (1,2)}
    if qnorms!={1:F(43,64),2:F(129,80)}:
        raise ArithmeticError('Exact periodic physical129-mode projection mismatch')
    chars={L:iv.exp(-ex(S*qnorms[L]/2)) for L in (1,2)}
    m1=moment(data['profiles'][1],g,chars)
    mlo=lower(m1)
    if mlo<=0:
        raise ArithmeticError('Initial source phase is not strictly positive')
    profile_bounds=[];force=F(0);Hmax=F(0);Hhop=F(0)
    for profile in data['profiles'][1:]:
        hop=2*sum(abs(ex(p['c0']))*iv.sqrt(g0)+abs(ex(p['ci']))*iv.sqrt(g[p['axis']+1]) for p in profile['paths'])
        maxh=abs(ex(profile['onsite']))*iv.sqrt(g0)+hop
        hup,hmax=upper(hop),upper(maxh)
        count=128*profile['load']+(1 if profile['load']==1 else 0)
        force+=17*count*hup
        if profile['load']==1:
            Hmax,Hhop=hmax,hup
        profile_bounds.append({'role':profile['role'],'count_source':count,'all_real_Theta_Hhop_upper':record(hup),'all_real_Theta_Hmax_upper':record(hmax),'charged_physical_scale':1})
    L0=Hmax*Hmax+(F(99136)+17*G)*Hhop
    clocks=[]
    for t in (F(1,1 << 22),F(1,1 << 21),F(1,1 << 20),F(1,1 << 18)):
        remainder=L0*t*t/2+G*Hhop*force*t**3/2
        negative_imag=t*mlo-remainder
        real_lower=F(1,2)-remainder
        coarse_lower=F(1,2)-3*t*Hmax
        if real_lower<=0 or coarse_lower<=0:
            raise ArithmeticError('Selected actual source clock bound is vacuous')
        clocks.append({'time_exact':str(t),'first_moment_phase_lower':record(t*mlo),'quadratic_cubic_error_upper':record(remainder),
                       'actual_negative_imag_lower':record(negative_imag),'nonzero_phase_proven':negative_imag>0,
                       'actual_real_lower':record(real_lower),'raw_probability_lower':record(real_lower*real_lower),
                       'coarse_survival_disk_radius':record(3*t*Hmax)})
    if not clocks[2]['nonzero_phase_proven']:
        raise ArithmeticError('Target2^-20 unrestricted source phase not discriminated')
    with threadpool_limits(limits=1):
        controls=finite_initial_control(data,qnorms,m1)
    captures={**field.CAPTURED,Path(__file__).name:own,Path(bridge.__file__).name:Path(bridge.__file__).read_bytes()}
    stable={name:{'sha256':hashlib.sha256(b).hexdigest(),'unchanged':b==(ROOT/name).read_bytes()} for name,b in captures.items()}
    if OLD.read_bytes()!=raw or not all(x['unchanged'] for x in stable.values()):
        raise ArithmeticError('Source or selected input changed')
    return {'derivation':DERIVATION,'exact_selected_body':{'N':4,'V':64,'real_photon_modes':129,'charged_flavours':64,'source_particles_active':129,'source_particles_auxiliary':8064,'Ref_before':4,'Ref_after':3,'HE_coefficient':'64','HC_coefficient':'64','Gaussian_density_variance':'2'},
            'initial_m1_interval':bounds(m1),'exact_path_variances':{str(k):str(v) for k,v in qnorms.items()},'path_character_intervals':{str(k):bounds(v) for k,v in chars.items()},
            'profile_bounds':profile_bounds,'joint_force_upper':record(force),'second_commutator_constant_upper':record(L0),'unrestricted_source_clocks':clocks,
            'finite_actual_control':controls,'input_sha256':{str(OLD):OLD_SHA},'source_at_execution':stable,'own_source_utf8':own.decode(),
            'governing_plan':'issue-222-induced-photon-delta13-continuation-20261008','governing_revision':'b5f1c36f7d60458c9318ffcab64cb99f087072ac35dc173323bcbd0763e0df00','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
            'runtime':{'executable':sys.executable,'python':sys.version,'numpy':np.__version__},'elapsed_seconds':time.monotonic()-start}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'m1_interval':result['initial_m1_interval'],
                     'clocks':[{'time':x['time_exact'],'phase_lower':x['actual_negative_imag_lower']['decimal'],'error':x['quadratic_cubic_error_upper']['decimal'],'phase_proven':x['nonzero_phase_proven'],'raw_probability_lower':x['raw_probability_lower']['decimal']} for x in result['unrestricted_source_clocks']],
                     'actual_control':result['finite_actual_control'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
