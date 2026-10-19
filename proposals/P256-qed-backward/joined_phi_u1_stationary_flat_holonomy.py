"""Exact flat SU4 classification and certified noncentral neutral-static competitor.

No mean/root solve. The static comparison does not admit a quantum ground/photon.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from math import isqrt
from pathlib import Path
import sys
import time
from typing import Any
import zipfile
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import joined_phi_u1_stationary_phi_sea_envelope as sea

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_stationary_flat_holonomy'
BITS = 110
SCALE = 1 << BITS
KS = np.array(list(product(range(4), repeat=3)))
PINS = dict(sea.INPUTS)
PROOF = {
 'classification': 'For Aalign=P=0 every unitary face has ReTr=4, hence is I; parallel transport is path-independent on the universal cover. A spanning-tree gauge makes Phi constant and leaves three torus holonomies Wj. Contractible commutators imply [Wj,Wk]=0, and parallel Phi implies [Wj,Phi_mu]=0. Simultaneously diagonalize the commuting normal Wj; equal JOINT characters define blocks d_b. Phi_mu are arbitrary Hermitian d_b matrices with SUM_b TrPhi_mu,b=0, NOT separately traceless. The only partitions are 4,3+1,2+2,2+1+1,1+1+1+1. Distinct joint characters, not distinct characters on each axis, define the partition. Gauge equivalence is simultaneous block conjugation and block permutation with character relabelling. Degenerate characters merge blocks.',
 'determinant_AP': 'Wj=directsum_b exp(i theta_bj) I_db, theta modulo2pi, sum_b d_b theta_bj=2pi n_j. Choosing the first B-1 angles arbitrarily leaves theta_last,j=(2pi n_j-sum_before d_b theta_bj)/d_last, n_j=0,...,d_last-1. These discrete winding sheets MUST remain: for2+2 n=1 cannot be removed by theta2=-theta1. Partition4 forces theta_j=pi ell_j/2. SU4 seam gauge puts Wj on positive wrap edges and I elsewhere, each determinant1. AP remains the separate native scalar -1 and is NOT included in SU4 determinant constraints. The allowed Fourier momenta in block b are p_bj=(2pi k_j+pi+theta_bj)/4. A common fourth root need not have determinant1; seam gauge avoids that error.',
 'centralizer_topology': 'The determinant-one character torus {z_b in U1^B: product z_b^d_b=1} has gcd(d_b) connected components: its induced pi1 map Z^B to Z has image gcd(d_b)Z. Winding sum d_b theta_b/(2pi) modulo gcd labels these components. Thus block4 has Z4,2+2 has Z2, while3+1,2+1+1,1^4 are connected. In three directions there are independent component labels. The witness occupies m=(1,0,1) in Z2^3; n=0-only coverage is incomplete. Pairwise character coincidences are lower-partition boundary strata and do not remove the determinant component.',
 'native_block_Fourier': 'Because parallel Phi commutes with Wilson products, literal endpoint average equals Phi times the ordered path. For every primitive displacement d and coefficient c_d, h_b(k)=sum_mu x_mu(p_b) Phi_mu,b, x_mu=A delta_mu0+2Re sum_d c_d,mu exp(ip_b.d). Native h is PLUS heavy sheet, physical neutral is -Lambda dGamma(h). All64 momenta and all colours occur. Neutral primitive has nearest paths only; charged two-link paths remain separate native controls, never imported into neutral energy. The block plane waves exp(ip_b.x)/8 are unitary and diagonalize actual seam/AP assembly.',
 'filling_Gram_quartic': 'Globally TrH=64 A sum_b TrPhi0,b=0. Exact neutral128 minimum is -Lambda S128(H), sum of the GLOBAL largest128 eigenvalues of all64*d_b bands. No block-half or top2-per-momentum rule is valid in general, even if some blocks have nonzero traces. G_mu,nu=(1/4)sum_b ReTr(Phi_mu,b Phi_nu,b); original quartic/Lambda=4096 TrG^2. At fixed G and scalar central twists only, Clifford gives exact envelope F_ell(G)=4096 TrG^2-2sum64 sqrt(xGx). Noncentral shifts are colour-dependent, so replacing them by central x is invalid.',
 'central_minimum_certificate': 'F_ell on PSD4 is 8192-strongly convex. At ANY SPD rational point Gbar, global minimum >=F_ell(Gbar)-||8192Gbar-sum_p xx^T/sqrt(xGbarx)||F^2/16384. Saved matching g0,gs enclosures provide only a fixed point (the exact midpoint), not identification of the coupled root with the neutral-static minimum. All64 sectors are evaluated by outward dyadic arithmetic and exact nested radicals sin/cos(n pi/8). Take min of64 lower bounds and use F_0(Gbar) as central upper. No root solve is used.',
 'counterexample': 'Take joint blocks2+2, W=(diag(I2,-I2),I4,diag(-I2,I2)). Characters theta_first=(0,0,pi), theta_second=(pi,0,0); determinant winding n=(1,0,1). Set Phi0=a diag(sigma3,sigma3), Phi1=diag(b sigma1,0), Phi2=c diag(sigma2,sigma2), Phi3=diag(0,b sigma1), a=537/1000,b=458/1000,c=324/1000. All Phi are traceless, holonomies commute with them, and alignment/plaquette are exactly ZERO (float residuals are not multiplied by2^2048). Gram=diag(a^2,b^2/2,c^2,b^2/2). Each2-block has eigenvalues +/-r. On first block sin^2(px)=sin^2(py)=1/2, missing z field has toron pi; on second missing x has toron pi. Thus r=sqrt(a^2 x0^2+2 ci^2(b^2+c^2)), and x0 distributions are identical by axis permutation. There are exactly128 positive eigenvalues, so GLOBAL128 sea=2sum64 r. The Pauli spectral identity permits rigorous radical intervals with no numerical eigensolver premise. This real noncentral competitor is strictly BELOW the all64 central Clifford minimum; it refutes central global-neutral-well dominance.',
 'all_angle_bracket': 'For arbitrary commuting holonomies and compatible Phi, four equally spaced samples have average sin=cos=sin*cos=0 and sin^2=cos^2=1/2. Hence TrH^2=256 Tr(C G), C=diag(A^2+2sum c0_j^2,2ci_1^2,2ci_2^2,2ci_3^2), independent of all continuous toron angles AND determinant sheets. Tracezero gives S128<=Tr|H|/2<=8||H||HS=128sqrt(TrCG)<=128sqrt(||C||F||G||F). Analytic radial minimization gives E/Lambda >=-3(256 TrC^2)^(1/3). This is an exact ALL-ANGLE lower bound; the counterexample supplies an exact upper bracket. The coarse bound is NOT alone a ground characterization: central exclusion is supplied independently by the strict counterexample certificate.',
 'remaining': 'New noncentral flat branch must be retained in the stationary design. Its minimizing partition/angles/Phi and possible off-flat competitors, fullPhi/SU4 kinetic/Gauss/64charged species, Pi, HC, quantum ground, same-ground spectral photon/commoncone remain OPEN. No beta, current, source, clock, parameter, or Maxwell admission. Floating representatives/search cannot certify all-angle dominance. Exact classification, counterexample and bracket are independent of floating controls.'
}


def ceildiv(a: int, b: int) -> int:
    return -((-a)//b)


@dataclass(frozen=True)
class Interval:
    """Outward fixed-dyadic arithmetic, using integer operations exclusively."""
    lo: int
    hi: int

    @classmethod
    def of(cls, v: Any) -> Interval:
        if isinstance(v, cls):
            return v
        q=F(v)
        return cls(q.numerator*SCALE//q.denominator,ceildiv(q.numerator*SCALE,q.denominator))

    def __add__(self, other: Any) -> Interval:
        q=Interval.of(other)
        return Interval(self.lo+q.lo,self.hi+q.hi)

    __radd__=__add__

    def __neg__(self) -> Interval:
        return Interval(-self.hi,-self.lo)

    def __sub__(self, other: Any) -> Interval:
        return self+-Interval.of(other)

    def __rsub__(self, other: Any) -> Interval:
        return Interval.of(other)+-self

    def __mul__(self, other: Any) -> Interval:
        q=Interval.of(other)
        vals=[self.lo*q.lo,self.lo*q.hi,self.hi*q.lo,self.hi*q.hi]
        return Interval(min(vals)//SCALE,ceildiv(max(vals),SCALE))

    __rmul__=__mul__

    def __truediv__(self, other: Any) -> Interval:
        q=Interval.of(other)
        if q.lo<=0<=q.hi:
            raise ArithmeticError('Interval division across zero')
        inv=Interval(SCALE*SCALE//q.hi,ceildiv(SCALE*SCALE,q.lo))
        return self*inv

    def square(self) -> Interval:
        lo=0 if self.lo<=0<=self.hi else min(self.lo*self.lo,self.hi*self.hi)
        hi=max(self.lo*self.lo,self.hi*self.hi)
        return Interval(lo//SCALE,ceildiv(hi,SCALE))

    def sqrt(self) -> Interval:
        if self.lo<0:
            raise ArithmeticError('Negative radical interval')
        lo=isqrt(self.lo*SCALE);hi=isqrt(self.hi*SCALE)
        return Interval(lo,hi+(hi*hi!=self.hi*SCALE))

    def cube_root(self) -> Interval:
        if self.lo<0:
            raise ArithmeticError('Negative cube-root interval')
        def floor_root(n: int) -> int:
            a,b=0,1 << ((n.bit_length()+2)//3)
            while a+1<b:
                m=(a+b)//2
                if m*m*m<=n:a=m
                else:b=m
            return a
        lo=floor_root(self.lo*SCALE*SCALE);n=self.hi*SCALE*SCALE;hi=floor_root(n)
        return Interval(lo,hi+(hi**3!=n))

    def public(self) -> dict[str, Any]:
        return dict(lower=str(F(self.lo,SCALE)),upper=str(F(self.hi,SCALE)),lower_display=float(F(self.lo,SCALE)),upper_display=float(F(self.hi,SCALE)),outward_bits=BITS)


def trig(n: int) -> tuple[Interval,Interval]:
    r2=Interval.of(2).sqrt();h=r2/2
    c=(2+r2).sqrt()/2;s=(2-r2).sqrt()/2
    values=[Interval.of(0),s,h,c,Interval.of(1),c,h,s,Interval.of(0),-s,-h,-c,Interval.of(-1),-c,-h,-s]
    return values[(n+4)%16],values[n%16]


def exact_x(profile: dict[str,Any], k: tuple[int,...], ell: tuple[int,...]) -> list[Interval]:
    x=[Interval.of(profile['onsite']),*[Interval.of(0) for _ in range(3)]]
    for row in profile['paths']:
        if row['length']!=1:raise ArithmeticError('Neutral nearest law required')
        j=row['axis'];co,si=trig(4*k[j]+2+ell[j])
        x[0]+=2*row['c0']*co;x[j+1]-=2*row['ci']*si
    return x


def exact_certificate(profile: dict[str,Any], root: dict[str,Any]) -> dict[str,Any]:
    pars=root['actual_stationary_root']['parameters']
    point=[(F(pars[name]['lower'])+F(pars[name]['upper']))/2 for name in ('g0','gs')]
    g=[Interval.of(point[0]),*[Interval.of(point[1]) for _ in range(3)]]
    sectors=[]
    for ell in product(range(4),repeat=3):
        sums=Interval.of(0);m=[[Interval.of(0) for _ in range(4)] for _ in range(4)]
        for k in product(range(4),repeat=3):
            x=exact_x(profile,k,ell);r=sum((g[j]*x[j].square() for j in range(4)),Interval.of(0)).sqrt();sums+=r
            for j in range(4):
                for l in range(4):m[j][l]+=x[j]*x[l]/r
        f=4096*sum((v.square() for v in g),Interval.of(0))-2*sums
        grad=[[8192*g[j]-m[j][l] if j==l else -m[j][l] for l in range(4)] for j in range(4)]
        norm=sum((v.square() for row in grad for v in row),Interval.of(0))
        lower=f-norm/16384
        sectors.append(dict(ell=list(ell),point_value=f.public(),gradient_norm_squared=norm.public(),minimum_lower=lower.public()))
    a,b,c=map(F,('537/1000','458/1000','324/1000'))
    q=4096*(a**4+b**4/2+c**4)
    ci=next(row['ci'] for row in profile['paths'] if row['axis']==0)
    sea_sum=Interval.of(0)
    for k in product(range(4),repeat=3):
        x=exact_x(profile,k,(0,0,2));r=(a*a*x[0].square()+2*ci*ci*(b*b+c*c)).sqrt();sea_sum+=2*r
    trial=Interval.of(q)-sea_sum
    central_lower=min(F(row['minimum_lower']['lower']) for row in sectors)
    central_upper=F(sectors[0]['point_value']['upper'])
    separation=Interval.of(central_lower)-trial
    if separation.lo<=0:raise ArithmeticError('Noncentral strict exclusion not certified')
    C=[profile['onsite']**2+2*sum(row['c0']**2 for row in profile['paths']),*[2*next(row['ci'] for row in profile['paths'] if row['axis']==j)**2 for j in range(3)]]
    universal=-3*Interval.of(256*sum(v*v for v in C)).cube_root()
    return dict(saved_matching_point='exact midpoint of saved current-coupled g0,gs; used as point only, not neutral minimum identity',central_all64_lower=str(central_lower),central_all64_upper=str(central_upper),central_lower_display=float(central_lower),central_upper_display=float(central_upper),central_sectors=sectors,trial_parameters={'a':str(a),'b':str(b),'c':str(c)},trial_quartic_exact=str(q),trial_global128_sea=sea_sum.public(),trial_static_energy_divided_Lambda=trial.public(),strict_separation=separation.public(),flat_ALL_angle_global_lower=universal.public(),corrected_flat_bracket={'lower':str(F(universal.lo,SCALE)),'upper':str(F(trial.hi,SCALE))},no_numeric_root_solver=True)


def coefficients(profile: dict[str,Any], angles: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    p=(2*np.pi*KS+np.pi+angles)/4;x=np.zeros((64,4));x[:,0]=float(profile['onsite'])
    for disp,c in profile['raw_paths']:x+=2*np.real(np.exp(1j*p@disp)[:,None]*c)
    return p,x


def seam_links(native: dict[str,Any], dimensions: list[int], angles: np.ndarray) -> np.ndarray:
    chars=np.repeat(angles,dimensions,axis=0);links=np.broadcast_to(np.eye(4,dtype=complex),(192,4,4)).copy()
    for s,site in enumerate(native['sites']):
        for j in range(3):
            if site[j]==3:links[3*s+j]=np.diag(np.exp(1j*chars[:,j]))
    return links


def block_fourier(profile: dict[str,Any], native: dict[str,Any], phi: np.ndarray, dimensions: list[int], angles: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    v=np.zeros((256,256),complex);direct=np.zeros_like(v);offset=0
    for db,theta in zip(dimensions,angles,strict=True):
        p,x=coefficients(profile,theta);bloch=np.einsum('pm,mij->pij',x,phi[:,offset:offset+db,offset:offset+db])
        for k in range(64):
            for a in range(db):v[np.arange(64)*4+offset+a,k*4+offset+a]=np.exp(1j*native['sites']@p[k])/8
            sl=slice(k*4+offset,k*4+offset+db);direct[sl,sl]=bloch[k]
        offset+=db
    return v,direct


def control(profile: dict[str,Any], native: dict[str,Any], phi: np.ndarray, dimensions: list[int], angles: np.ndarray) -> dict[str,Any]:
    links=seam_links(native,dimensions,angles);fields=np.broadcast_to(phi,(64,4,4,4))
    sign=-1 if profile['scale']=='heavy' else 1;h=sign*field.assemble(profile,native,fields,links)
    v,direct=block_fourier(profile,native,phi,dimensions,angles);actual=np.linalg.eigvalsh(h);bands=np.array([np.linalg.eigvalsh(direct[4*k:4*k+4,4*k:4*k+4]) for k in range(64)])
    fixed=0.;offset=0
    for db in dimensions:
        ev=np.concatenate([np.linalg.eigvalsh(direct[4*k+offset:4*k+offset+db,4*k+offset:4*k+offset+db]) for k in range(64)])
        fixed+=float(np.sort(ev)[-32*db:].sum());offset+=db
    align,plaquette=sea.geometry_cost(native,fields,links)
    rng=np.random.default_rng(442);generators=[]
    for _ in range(64):
        m=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));m=(m+m.conj().T)/2;m-=np.trace(m)*np.eye(4)/4;generators.append(expm(1j*m/13))
    gauges=np.array(generators);gfields=gauges[:,None]@fields@gauges.conj().transpose(0,2,1)[:,None];glinks=links.copy();G=np.zeros_like(h)
    for s,site in enumerate(native['sites']):
        G[4*s:4*s+4,4*s:4*s+4]=gauges[s]
        for j in range(3):
            t=int(np.ravel_multi_index(tuple((site+np.eye(3,dtype=int)[j])%4),(4,4,4)));glinks[3*s+j]=gauges[s]@links[3*s+j]@gauges[t].conj().T
    hg=sign*field.assemble(profile,native,gfields,glinks);ga,gp=sea.geometry_cost(native,gfields,glinks)
    _,wrong=block_fourier(profile,native,phi,[4],np.zeros((1,3)))
    no_AP_native=dict(native);no_AP_native['Pprofiles']=[dict(p,paths=[dict(row,seam=1) for row in p['paths']]) for p in native['Pprofiles']]
    no_ap=sign*field.assemble(profile,no_AP_native,fields,links)
    g=sea.gram(phi);s=float(actual[-128:].sum());q=4096*float(np.sum(g*g))
    errors={'Fourier_literal_HS':float(np.linalg.norm(v.conj().T@h@v-direct)),'Fourier_unitarity_HS':float(np.linalg.norm(v.conj().T@v-np.eye(256))),'global128_spectral':abs(s-float(np.sort(bands.ravel())[-128:].sum())),'gauge_covariance_HS':float(np.linalg.norm(hg-G@h@G.conj().T)),'link_determinant':float(max(abs(np.linalg.det(u)-1) for u in links))}
    law=field.original.OriginalBosonicLaw(4,profile['model'],1,1,1)
    original_quartic=64*float(law.field_potential_at(phi,[]))
    errors['original_microscopic_quartic']=abs(original_quartic-q)
    path=next(row for row in field.native_paths(profile,native) if row['length']==1)
    source,target=path['source'],path['target'];u=links[path['edges'][0]]
    owner=field.original.original_endpoint(phi,phi,u,path['axis'],profile['model'])
    c=path['coefficients'];hop=np.einsum('m,mij->ij',c,phi)
    errors['original_endpoint_owner']=float(np.linalg.norm(owner-(hop@u+u@hop)/2))
    if max(errors.values())>2e-9:raise ArithmeticError(json.dumps(errors))
    return dict(partition=dimensions,role=profile['role'],angles_radians=angles.tolist(),weighted_determinant_winding=(np.array(dimensions)@angles/(2*np.pi)).tolist(),Wilson_spectra=[[[float(z.real),float(z.imag)] for z in np.repeat(np.exp(1j*angles[:,j]),dimensions)] for j in range(3)],errors=errors,geometry_exact='0 by constant block Phi and commuting holonomies, also after gauge; residuals floating ONLY',geometry_float={'alignment':align,'plaquette':plaquette,'gauge_alignment':ga,'gauge_plaquette':gp},Gram=g.tolist(),quartic=q,global_largest128=s,energy_static_divided_Lambda=q-s,wrong_fixed_block_half=fixed,wrong_top2_per_momentum=float(bands[:,2:].sum()),wrong_central_flattening_matrix_HS=float(np.linalg.norm(direct-wrong)),wrong_AP_omission_matrix_HS=float(np.linalg.norm(h-no_ap)),full_global_bands=actual.tolist())


def witness_phi() -> np.ndarray:
    s1=np.array([[0,1],[1,0]]);s2=np.array([[0,-1j],[1j,0]]);s3=np.diag([1,-1]);phi=np.zeros((4,4,4),complex)
    phi[0,:2,:2]=.537*s3;phi[0,2:,2:]=.537*s3;phi[1,:2,:2]=.458*s1;phi[2,:2,:2]=.324*s2;phi[2,2:,2:]=.324*s2;phi[3,2:,2:]=.458*s1
    return phi


def run(output: Path) -> dict[str,Any]:
    started=time.monotonic();snapshots=dict(field.CAPTURED)
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).suffix=='.py' and Path(path).parent==ROOT:snapshots.setdefault(Path(path).name,Path(path).read_bytes())
    snapshots[Path(__file__).name]=Path(__file__).read_bytes()
    for name,pin in PINS.items():
        snapshots[name]=(ROOT/name).read_bytes()
        if hashlib.sha256(snapshots[name]).hexdigest()!=pin:raise ArithmeticError('Pinned input changed: '+name)
    data=field.load_mean(g_override=[1,1,1,1],mean_descriptor={'scope':'fresh exact unit Clifford frame; coefficients only, no root solve'})
    profiles=data['profiles'];native=data['native'];certificate=exact_certificate(profiles[0],json.loads(snapshots[next(iter(PINS))]))
    rng=np.random.default_rng(22210);controls=[]
    for dims in ([4],[3,1],[2,2],[2,1,1],[1,1,1,1]):
        angles=np.zeros((len(dims),3))
        for b in range(len(dims)-1):angles[b]=np.pi*np.array([F(2*b+1,7),F(b+1,5),F(3*b+1,11)],dtype=float)
        angles[-1]=-np.array(dims[:-1])@angles[:-1]/dims[-1] if len(dims)>1 else np.pi*np.array([1,2,3])/2
        phi=np.zeros((4,4,4),complex);offset=0
        for db in dims:
            for mu in range(4):
                m=rng.integers(-5,6,size=(db,db))+1j*rng.integers(-5,6,size=(db,db));phi[mu,offset:offset+db,offset:offset+db]=(m+m.conj().T)/40
            offset+=db
        phi-=np.trace(phi,axis1=1,axis2=2)[:,None,None]*np.eye(4)/4
        for profile in profiles:controls.append(control(profile,native,phi,dims,angles))
    angles=np.pi*np.array([[0,0,1],[1,0,0]]);witness=[control(p,native,witness_phi(),[2,2],angles) for p in profiles]
    # Independent adverse filling case with unequal block trace/occupation.
    adverse=np.zeros((4,4,4),complex);adverse[0]=np.diag([3,-1,-1,-1])/5
    adverse_angles=np.pi*np.array([[F(1,7),F(2,9),F(3,11)],[-F(1,21),-F(2,27),-F(1,11)]],dtype=float)
    filling=control(profiles[0],native,adverse,[1,3],adverse_angles)
    if abs(filling['global_largest128']-filling['wrong_fixed_block_half'])<1 or abs(filling['global_largest128']-filling['wrong_top2_per_momentum'])<1:raise ArithmeticError('Wrong filling was not discriminated')
    if abs(witness[0]['energy_static_divided_Lambda']-certificate['trial_static_energy_divided_Lambda']['lower_display'])>1e-8:raise ArithmeticError('Literal witness does not reach certified energy')
    archive=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in sorted(snapshots.items()):z.writestr(name,raw)
    after={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in snapshots};before={name:hashlib.sha256(raw).hexdigest() for name,raw in snapshots.items()}
    if before!=after:raise ArithmeticError('Source/input bytes changed during execution')
    result=dict(schema=1,scope='STATIC neutral flat noncentral counterexample, NOT quantum ground or photon',proof=PROOF,exact_outward_certificate=certificate,all_partition_native_controls=controls,certified_witness_native_controls=witness,adverse_global_filling=filling,primitive_profiles=[dict(role=p['role'],scale=p['scale'],load=p['load'],onsite=str(p['onsite']),paths=[{k:str(v) if isinstance(v,F) else v for k,v in row.items()} for row in p['paths']]) for p in profiles],source_input_sha256_before=before,source_input_sha256_after=after,immutable_input_archive=str(archive),runtime={'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'threads':1},elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with threadpool_limits(1):result=run(args.output)
    cert=result['exact_outward_certificate']
    print(json.dumps({'receipt':str(args.output),'central_minimum_lower':cert['central_lower_display'],'noncentral_trial_upper':cert['trial_static_energy_divided_Lambda']['upper_display'],'strict_separation_lower':cert['strict_separation']['lower_display'],'corrected_flat_bracket':cert['corrected_flat_bracket'],'quantum_ground_photon':'NOT ADMITTED'},indent=2))


if __name__=='__main__':main()
