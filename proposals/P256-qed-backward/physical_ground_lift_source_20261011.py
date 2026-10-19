"""Same-law compact full-field equivariant lift of immutable selected local g.

Lazy canonical source: exact harmonic extraction isometry followed by native
exterior powers. No ground solver, historical consumer, or frozen field kinetic.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, cast
import numpy as np
from scipy.linalg import expm
import bulk_electric_phase_consumer as primitive
import harmonic_car_bridge_scientist_20261011 as bridge

ROOT = Path(__file__).resolve().parent
STATE = 'local_so5_resum_actual04_20261011_state.npz'
STATE_SHA = 'dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2'
R = Q(1, 2**24)

class Jet:
    """Literal value/first/second directional jet with complete Leibniz cross."""
    def __init__(self, v: Any, d: Any = None, dd: Any = None):
        self.v: Any = np.asarray(v, complex)
        self.d: Any = np.zeros_like(self.v) if d is None else np.asarray(d, complex)
        self.dd: Any = np.zeros_like(self.v) if dd is None else np.asarray(dd, complex)
    def __add__(self, b: Any) -> Jet:
        b = b if isinstance(b, Jet) else Jet(b)
        return Jet(self.v+b.v, self.d+b.d, self.dd+b.dd)
    def __neg__(self) -> Jet: return Jet(-self.v, -self.d, -self.dd)
    def __sub__(self, b: Any) -> Jet: return self+-b if isinstance(b, Jet) else self+Jet(-b)
    def __mul__(self, b: Any) -> Jet:
        b = b if isinstance(b, Jet) else Jet(b)
        return Jet(self.v*b.v, self.d*b.v+self.v*b.d, self.dd*b.v+2*self.d*b.d+self.v*b.dd)
    def __matmul__(self, b: Jet) -> Jet:
        return Jet(self.v@b.v, self.d@b.v+self.v@b.d, self.dd@b.v+2*self.d@b.d+self.v@b.dd)
    def adj(self) -> Jet: return Jet(self.v.conj().T, self.d.conj().T, self.dd.conj().T)
    def conj(self) -> Jet: return Jet(self.v.conj(), self.d.conj(), self.dd.conj())
    def power(self, p: float) -> Jet:
        return Jet(self.v**p, p*self.v**(p-1)*self.d, p*self.v**(p-1)*self.dd+p*(p-1)*self.v**(p-2)*self.d**2)
    def item(self, i: int, j: int) -> Jet: return Jet(self.v[i,j],self.d[i,j],self.dd[i,j])


def polynomial_pair(f: list[Jet]) -> tuple[Jet, Jet]:
    a = f[0]+(f[2]@f[3]-f[3]@f[2])*2j
    b = f[1]+f[2]+f[3]+(f[0]@f[2]-f[2]@f[0])*1j
    return a,b


def spectral_jet(a: Jet) -> tuple[np.ndarray, Jet]:
    """Parallel-gauge eigenvector jets, simple eigenvalues only."""
    e,w = np.linalg.eigh(a.v)
    if np.min(np.diff(e))<=1/16:raise ArithmeticError('outside simple spectral chart')
    h = w.conj().T@a.d@w
    first = np.zeros_like(w); second = np.zeros_like(w)
    for j in range(4):
        for k in range(4):
            if k != j: first[:,j] += w[:,k]*h[k,j]/(e[j]-e[k])
    for j in range(4):
        z = a.dd@w[:,j]+2*a.d@first[:,j]-2*h[j,j]*first[:,j]
        second[:,j] = -cast(complex, np.vdot(first[:,j],first[:,j]))*w[:,j]
        for k in range(4):
            if k != j: second[:,j] += w[:,k]*np.vdot(w[:,k],z)/(e[j]-e[k])
    return e,Jet(w,first,second)


def frame_jet(f: list[Jet]) -> tuple[Jet, dict[str, Any]]:
    a,b = polynomial_pair(f);e,w = spectral_jet(a)
    body = w.adj()@b@w
    phases = [Jet(1.) for _ in range(4)]
    for parent,child in [(0,2),(2,1),(0,3)]:
        z = body.item(parent,child)
        if abs(z.v)<=1/16:raise ArithmeticError('outside nonzero phase-tree chart')
        phases[child] = phases[parent]*z.conj()*(z*z.conj()).power(-.5)
    p = Jet(np.diag([x.v.item() for x in phases]),np.diag([x.d.item() for x in phases]),np.diag([x.dd.item() for x in phases]))
    v = w@p
    # SU4 determinant correction, local fourth-root branch. Different branches
    # are Z4 and disappear on the full 256-particle matching-block source.
    det = np.linalg.det(v.v); inv = np.linalg.inv(v.v)
    ld = np.trace(inv@v.d)
    ldd = np.trace(inv@v.dd-inv@v.d@inv@v.d)
    root = det**(-.25)
    v = v*Jet(root,-.25*ld*root,(-.25*ldd+.0625*ld*ld)*root)
    return v,{'eigenvalues':e.tolist(),'minimum_gap':float(np.min(np.diff(e))),'tree_edge_min':float(min(abs(body.v[i,j]) for i,j in [(0,2),(2,1),(0,3)]))}


def exterior(m: np.ndarray, k: int) -> np.ndarray:
    indices = list(itertools.combinations(range(m.shape[1]),k))
    rows = list(itertools.combinations(range(m.shape[0]),k))
    return np.array([[np.linalg.det(m[np.ix_(r,c)]) for c in indices] for r in rows])


def exterior_jet(m: Jet, k: int) -> Jet:
    rows=list(itertools.combinations(range(m.v.shape[0]),k));cols=list(itertools.combinations(range(m.v.shape[1]),k))
    out=[np.zeros((len(rows),len(cols)),complex) for _ in range(3)]
    for i,r in enumerate(rows):
        for j,c in enumerate(cols):
            a=m.v[np.ix_(r,c)];da=m.d[np.ix_(r,c)];dda=m.dd[np.ix_(r,c)]
            out[0][i,j]=np.linalg.det(a)
            for t in range(k):
                b=a.copy();b[:,t]=da[:,t];out[1][i,j]+=np.linalg.det(b)
                b=a.copy();b[:,t]=dda[:,t];out[2][i,j]+=np.linalg.det(b)
                for s in range(k):
                    if s != t:
                        b=a.copy();b[:,t]=da[:,t];b[:,s]=da[:,s];out[2][i,j]+=np.linalg.det(b)
    return Jet(*out)


def native_slot_map() -> np.ndarray:
    j,_,_,_,_,v = bridge.car_exact()
    # Serialized actual04 is REAL representation: complex slot phase is applied
    # before the exact native J1, not confused with the oneparticle source phase.
    return np.array((v*j).evalf(),complex)*np.array([1.,-1j,1.,1.,1j])[None,:]


def harmonic_amplitude(n: int, slots: list[np.ndarray], coefficients: np.ndarray) -> complex:
    """Executable J_n via normalized extraction, any requested native component.

    No 5^126 dense tensor is allocated. Each slot contraction lowers n by1;
    extraction matrices include all harmonic l,p,q and trace subtraction.
    """
    if len(slots) != n: raise ValueError('one native slot covector per replica required')
    x=np.asarray(coefficients,complex)
    for k,row in zip(range(n,0,-1),slots):
        es=bridge.extraction(k)
        x=sum((row[a]*(es[a]@x) for a in range(5)),np.zeros(bridge.dim(k-1),complex))
    return complex(x[0])


class LiftedGround:
    """Actual g coefficients + native fullCAR exterior-power evaluation recipe."""
    def __init__(self, coefficients: np.ndarray, centre: np.ndarray):
        self.g=coefficients.reshape(14,690880)/np.linalg.norm(coefficients)
        self.centre=centre
        self.wstar,_=frame_jet([Jet(x) for x in centre])
        gamma=centre[1]/math.sqrt(float(np.trace(centre[1]@centre[1]).real/4))
        self.spin=np.column_stack([np.eye(4)[:,0],gamma@np.eye(4)[:,0],np.eye(4)[:,1],gamma@np.eye(4)[:,1]])
    def transport(self, fields: np.ndarray, u: Jet, theta: Jet) -> tuple[Jet,Jet]:
        w,_=frame_jet([Jet(x) for x in fields])
        return self.transport_jet(w,u,theta)
    def transport_jet(self,w: Jet,u: Jet,theta: Jet) -> tuple[Jet,Jet]:
        left=w@self.wstar.adj()@Jet(self.spin)
        plus=Jet(np.exp(.5j*theta.v),.5j*theta.d*np.exp(.5j*theta.v),(.5j*theta.dd-.25*theta.d**2)*np.exp(.5j*theta.v))
        minus=Jet(np.exp(-.5j*theta.v),-.5j*theta.d*np.exp(-.5j*theta.v),(-.5j*theta.dd-.25*theta.d**2)*np.exp(-.5j*theta.v))
        return left*plus,(u.adj()@left)*minus
    def component(self, left: Jet, right: Jet, occupations: list[tuple[int,...]]) -> complex:
        """Native canonical full8-mode wedge4 component for every actual species.

        Each species has two wedge2 spin replicas. A 70x36 exterior map mixes
        them through actual colour frames; recursive contractions retain exact g.
        This exact finite expression can be costly for generic spin mixing, but
        never replaces g or claims its arbitrary component was cheaply evaluated.
        """
        if len(occupations)!=64: raise ValueError('actual1+63 species required')
        maps=[]
        phase=np.diag([1.,-1.,-1.,1.])
        for occ in occupations:
            f=np.zeros((8,8),complex)
            for spin in range(2):
                ix=[2*spin,2*spin+1]
                for site,m in enumerate([left.v,right.v]):
                    f[4*site:4*site+4,4*spin+2*site:4*spin+2*site+2]=m[:,ix]
            f=f@np.kron(np.eye(2),phase)
            combinations=list(itertools.combinations(range(4),2))
            native=native_slot_map()[[3,5,9,6,10,12]]
            z=np.zeros((5,5),complex)
            for ia,a in enumerate(combinations):
                for ib,b in enumerate(combinations):
                    cols=list(a)+[x+4 for x in b]
                    z += np.linalg.det(f[np.ix_(occ,cols)])*np.outer(native[ia],native[ib])
            maps.append(z)
        # An explicit exact contraction over five native slots per replica.
        # In diagonal spin channels it factorizes; arbitrary channels recurse.
        def descend(species: int, x: np.ndarray, na: int, nb: int) -> complex:
            if species==64:return complex(x[0,0])
            n=na if species==0 else nb
            es=bridge.extraction(n);es2=bridge.extraction(n-1)
            total=0j
            for a in range(5):
                for b in range(5):
                    z=maps[species][a,b]
                    if z==0:continue
                    if species==0:
                        y=es2[b]@(es[a]@x)
                        total+=z*descend(1,y,0,nb)
                    else:
                        y=(es2[b]@(es[a]@x.T)).T
                        total+=z*descend(species+1,y,0,nb-2)
            return total
        return descend(0,self.g,2,126)


def source_shape(fields: np.ndarray,u: np.ndarray,trial: LiftedGround,radius: float) -> float:
    try:
        w,_=frame_jet([Jet(x) for x in fields[0]])
    except ArithmeticError:
        return 0.
    ref=np.array([trial.wstar.v.conj().T@x@trial.wstar.v for x in trial.centre])
    body=np.array([[w.v.conj().T@x@w.v for x in fields[0]], [w.v.conj().T@u@x@u.conj().T@w.v for x in fields[1]]])
    s=float(np.sum(abs(body-ref[None])**2)/radius**2)
    return max(1-s,0.)**4

def source_shape_jet(fields: list[list[Jet]],u: Jet,trial: LiftedGround,radius: float) -> Jet:
    """Actual compact amplitude jets, with Phi/Haar/frame and cross derivatives."""
    try:
        w,_=frame_jet(fields[0])
    except ArithmeticError:
        return Jet(0.)
    s=Jet(0.)
    for mu in range(4):
        ref=Jet(trial.wstar.v.conj().T@trial.centre[mu]@trial.wstar.v)
        for body in [w.adj()@fields[0][mu]@w,w.adj()@u@fields[1][mu]@u.adj()@w]:
            z=body-ref;zz=z*z.conj()
            s=s+Jet(np.sum(zz.v),np.sum(zz.d),np.sum(zz.dd))*(1/radius**2)
    if s.v.real>=1:return Jet(0.)
    return (Jet(1.)-s).power(4)

def block_jet(left: Jet,right: Jet) -> Jet:
    z=np.zeros((4,4),complex)
    return Jet(*[np.block([[a,z],[z,b]]) for a,b in [(left.v,right.v),(left.d,right.d),(left.dd,right.dd)]])

def joined_jet_controls(trial: LiftedGround,centre: np.ndarray,basis: list[np.ndarray]) -> dict[str,Any]:
    point=np.broadcast_to(centre,(2,4,4,4)).copy()
    point[0,0]+=float(R)/8*basis[0]
    direction=np.zeros_like(point);direction[0,0]=basis[0]
    fields=[[Jet(point[x,mu],direction[x,mu]) for mu in range(4)] for x in range(2)]
    shape=source_shape_jet(fields,Jet(np.eye(4)),trial,float(R))
    w,_=frame_jet(fields[0]);left,right=trial.transport_jet(w,Jet(np.eye(4)),Jet(0.))
    fibre=exterior_jet(block_jet(left,right),4);whole=fibre*shape
    omission=float(np.linalg.norm(whole.dd-fibre.v*shape.dd))
    cross=float(np.linalg.norm(2*fibre.d*shape.d))
    hh=2**-36
    plus=source_shape(point+hh*direction,np.eye(4),trial,float(R))
    minus=source_shape(point-hh*direction,np.eye(4),trial,float(R))
    fd1=abs((plus-minus)/(2*hh)-shape.d.item())/max(1.,abs(shape.d.item()))
    fd2=abs((plus-2*shape.v.item()+minus)/hh**2-shape.dd.item())/max(1.,abs(shape.dd.item()))
    # Native70 physical U1 covariance/Gauss jet, including its background.
    left,right=trial.transport_jet(trial.wstar,Jet(np.eye(4)),Jet(0.,1.,0.))
    em=exterior_jet(block_jet(left,right),4)
    masks=list(itertools.combinations(range(8),4))
    rho=np.array([sum(j<4 for j in row)-2 for row in masks])
    gauss=float(np.linalg.norm(em.d-1j*rho[:,None]*em.v))
    second=float(np.linalg.norm(em.dd+rho[:,None]**2*em.v))
    # Haar scalar-frame join at the exact same physical width.
    uj=Jet(np.eye(4),1j*basis[4],-basis[4]@basis[4])
    for mu in range(4):
        point[1,mu]+=float(R)/8*1j*(basis[4]@centre[mu]-centre[mu]@basis[4])
    haar_shape=source_shape_jet([[Jet(x) for x in side] for side in point],uj,trial,float(R))
    wh,_=frame_jet([Jet(x) for x in point[0]])
    l,r=trial.transport_jet(wh,uj,Jet(0.))
    haar_fibre=exterior_jet(block_jet(l,r),4)
    haar_whole=haar_fibre*haar_shape
    haar_omission=float(np.linalg.norm(haar_whole.dd-haar_shape.dd*haar_fibre.v))
    haar_cross=float(np.linalg.norm(2*haar_shape.d*haar_fibre.d))
    result={'native70_complete_Phi_Pi2_unscaled_action_Frobenius':float(np.linalg.norm(whole.dd)),
            'native70_wrong_omit_Phi_fibre_action_gap':omission,'native70_Phi_scalar_frame_Berry_cross_norm':cross,
            'actual_width_scalar_first_FD_relative_error':float(fd1),'actual_width_scalar_second_FD_relative_error':float(fd2),
            'native70_U1_Gauss_first_error':gauss,'native70_U1_Gauss_second_error':second,
            'native70_complete_Haar_unscaled_second_action_Frobenius':float(np.linalg.norm(haar_whole.dd)),
            'native70_Haar_scalar_frame_cross_norm':haar_cross,
            'native70_wrong_omit_Haar_fibre_action_gap':haar_omission,
            'physical_kinetic_coefficients':'Phi4chi, Haar3chi/20, chi2^-4096; displayed diagnostic actions omit only this named overall prefactor, not any derivatives',
            'not_full_H_residual':True}
    if gauss>1e-10 or second>1e-10 or fd1>1e-2 or fd2>1e-2 or min(omission,cross,haar_cross,haar_omission)<=0:raise ArithmeticError(result)
    return result


def run(epoch: str) -> None:
    if hashlib.sha256((ROOT/STATE).read_bytes()).hexdigest()!=STATE_SHA:raise ValueError('immutable g changed')
    fr,scale=primitive.frozen.exact_frame((ROOT/primitive.INPUTS[0]).read_bytes())
    centre=np.array([[[complex(float(z[0]),float(z[1])) for z in row] for row in m] for m in fr])
    cfg=json.loads((ROOT/primitive.INPUTS[1]).read_bytes())['input_descriptor']['original_micro_config_UNCHANGED']
    profiles=primitive.frozen.profiles(cfg,fr,scale)
    with np.load(ROOT/STATE,allow_pickle=False) as data:g=data['state']
    trial=LiftedGround(g,centre)
    sf=Q(float(-centre[0,0,0].real));rf=Q(float(centre[1,0,3].real))
    exact_gap=min(2*(sf-4*rf**2),8*rf**2)
    if exact_gap<=Q(1,8) or rf<=Q(1,8):raise ArithmeticError('frame chart centre gap')
    rhoa=np.array([p-q for l,p,q in bridge.labels(2)])
    rhob=np.array([p-q for l,p,q in bridge.labels(126)])
    rho=rhoa[:,None]+rhob[None,:]
    first=1j*rho*trial.g;second=-rho**2*trial.g
    rho_mom=[float(np.sum(rho**k*trial.g**2)) for k in [1,2,4]]
    basis=[]
    for i in range(4):
        for j in range(i+1,4):
            z=np.zeros((4,4),complex);z[i,j]=z[j,i]=.5;basis.append(z)
            z=np.zeros((4,4),complex);z[i,j]=-.5j;z[j,i]=.5j;basis.append(z)
    for k in range(1,4):basis.append(np.diag([1.]*k+[-float(k)]+[0.]*(3-k))/math.sqrt(2*k*(k+1)))
    df=np.zeros_like(centre);df[0]=basis[0]
    w,frame_info=frame_jet([Jet(x,y) for x,y in zip(centre,df)])
    left,right=trial.transport_jet(w,Jet(np.eye(4)),Jet(0.))
    # Full native canonical70 exterior4, not a selected two-level role surrogate.
    f=Jet(np.block([[left.v,np.zeros((4,4))],[np.zeros((4,4)),right.v]]),np.block([[left.d,np.zeros((4,4))],[np.zeros((4,4)),right.d]]),np.block([[left.dd,np.zeros((4,4))],[np.zeros((4,4)),right.dd]]))
    ej=exterior_jet(f,4)
    h=2**-13
    vals=[]
    for t in [-h,0.,h]:
        wt,_=frame_jet([Jet(x+t*y) for x,y in zip(centre,df)])
        lt,rt=trial.transport_jet(wt,Jet(np.eye(4)),Jet(0.))
        vals.append(exterior(np.block([[lt.v,np.zeros((4,4))],[np.zeros((4,4)),rt.v]]),4))
    # Centre branches of SU4 fourth-root cause only a scalar; align physical
    # wedge4 matrices by their scalar overlap before native jet comparisons.
    for i in [0,2]:
        phase=np.vdot(vals[1],vals[i]);vals[i]*=np.conj(phase)/abs(phase)
    fd1=float(np.linalg.norm((vals[2]-vals[0])/(2*h)-ej.d))
    fd2=float(np.linalg.norm((vals[2]-2*vals[1]+vals[0])/h**2-ej.dd))
    rng=np.random.default_rng(2221011)
    gr=expm(1j*sum(rng.normal()*b for b in basis));gc=expm(1j*sum(rng.normal()*b for b in basis))
    transformed=np.array([gr@x@gr.conj().T for x in centre])
    lt,rt=trial.transport(transformed,Jet(gr@gc.conj().T),Jet(0.))
    expected=np.block([[gr@left.v,np.zeros((4,4))],[np.zeros((4,4)),gc@right.v]])
    actual=np.block([[lt.v,np.zeros((4,4))],[np.zeros((4,4)),rt.v]])
    cov=float(np.linalg.norm(exterior(actual,4)-exterior(expected,4)))
    frozen_wrong=float(np.linalg.norm(exterior(np.eye(8),4)-exterior(np.block([[gr,np.zeros((4,4))],[np.zeros((4,4)),gc]]),4)))
    # Actual field/Haar jets from the same transport, including U'' and cross.
    gen=basis[4];uj=Jet(np.eye(4),1j*gen,-gen@gen)
    lc,rc=trial.transport_jet(trial.wstar,uj,Jet(0.))
    colour_second=float(np.linalg.norm(rc.dd))
    fpoint=np.broadcast_to(centre,(2,4,4,4)).copy()
    shape_centre=source_shape(fpoint,np.eye(4),trial,float(R))
    fpoint[1,3]+=float(R)/4*basis[0]
    shape_fullphi=source_shape(fpoint,np.eye(4),trial,float(R))
    fpoint[1,3]+=4*float(R)*basis[0]
    shape_exterior=source_shape(fpoint,np.eye(4),trial,float(R))
    small=bridge.dim(2);slots=[np.array([0.,1.,0.,0.,0.])]*2
    coefficients=np.arange(1.,small+1);coefficients/=np.linalg.norm(coefficients)
    harmonic_component=harmonic_amplitude(2,slots,coefficients)
    small_expected=coefficients[bridge.idx(2,0,0)]
    if abs(harmonic_component-small_expected)>1e-12:raise ArithmeticError('native extraction amplitude mismatch')
    # Native role omission: nonzero difference on the accepted all128 g.
    aa=bridge.native_matrix(2,1.,0.,0.)[0]
    massdiff=float(Q(profiles[0]['onsite'])-Q(profiles[1]['onsite']))
    wrong_role=float(np.linalg.norm(massdiff*(aa@trial.g)))
    controls={'immutable_g_dimension':int(g.size),'immutable_g_norm':float(np.linalg.norm(g)), 'normalized_lift_g_norm':float(np.linalg.norm(trial.g)), 'g_rho_moments_1_2_4':rho_mom,'g_U1_first_jet_norm':float(np.linalg.norm(first)),'g_complete_E64_action_norm':64*float(np.linalg.norm(second)),'wrong_omit_Pi_action_gap':64*float(np.linalg.norm(second)), 'native70_Phi_first_FD_error':fd1,'native70_Phi_second_FD_error':fd2,'native70_full_Phi_first_norm':float(np.linalg.norm(ej.d)), 'native70_full_Phi_second_norm':float(np.linalg.norm(ej.dd)), 'native70_wrong_frozen_frame_gap':frozen_wrong,'native70_SU4_covariance_error':cov,'actual_Haar_second_transport_norm':colour_second,'wrong_actual1_aux63_role_action_gap':wrong_role,'shape_at_centre':shape_centre,'shape_after_Phi_c3_variation':shape_fullphi,'frame':frame_info}
    controls.update({'shape_exterior_zero':shape_exterior,'actual_source_radius':str(R),'native_recursive_J2_component_error':float(abs(harmonic_component-small_expected)),'wrong_drop_active_aux_totalrho_cross_gap':64*float(np.linalg.norm(2*rhoa[:,None]*rhob[None,:]*trial.g))})
    controls.update(joined_jet_controls(trial,centre,basis))
    if cov>1e-8 or fd1>1e-4 or fd2>1e-3 or min(wrong_role,colour_second,rho_mom[1],frozen_wrong)<=0:raise ArithmeticError(controls)
    proof=proof_record(profiles,centre,frame_info,controls)
    out=ROOT/f'physical_ground_lift_{epoch}_20261011'
    out.with_name(out.name+'_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    out.with_name(out.name+'_receipt.json').write_text(json.dumps(controls,indent=2)+'\n')
    print(json.dumps(controls,indent=2),flush=True)


def proof_record(profiles: Any,centre: np.ndarray,frame_info: Any,controls: Any) -> dict[str,Any]:
    v=64;m=v//2;d=60*v;e=3*v
    # Complete finite-N action upper. Scalar shape derivatives are absolute,
    # never pointwise log derivatives at the compact boundary.
    zexp=16384*v
    terms={'Phi_complete_scalar_frame_Berry':-4096+math.ceil(math.log2(4*d))+zexp+240,
           'Haar_complete_transport_scalar_cross':-4096+math.ceil(math.log2(15*e))+zexp+240,
           'fullE64_longitudinal_and_transverse':math.ceil(math.log2(64*e))+zexp+240,
           'neutral_all_canonical2V_onsite_paths':8192+math.ceil(math.log2(v))+30,
           'charged_actual1_aux63_all_onsite_one_two_paths':math.ceil(math.log2(v))+30,
           'Jordan_full_support':8192+math.ceil(math.log2(v))+30,
           'alignment_full_support':10240+math.ceil(math.log2(v))+30,
           'colour_plaquette_full_Haar_support':10240+math.ceil(math.log2(v))+30}
    bound=max(terms.values())+math.ceil(math.log2(len(terms)))
    sf=Q(float(-centre[0,0,0].real));rf=Q(float(centre[1,0,3].real))
    exact_gap=min(2*(sf-4*rf**2),8*rf**2)
    return {'kind':'NEW_EXACT_CANONICAL_LOCAL_G_EQUIVARIANT_QUANTUM_TRIAL_COMPLETE_D_H_NOT_GROUND',
      'configuration_binding':{'centre':'exact Phi_binary_section bytes from pinned f8a0793 NPZ, natural U=I, theta=0; no mean/root/parameter adoption','Phi_binary_components':[[[[float(z.real),float(z.imag)] for z in row] for row in p] for p in centre], 'charged_roles':[{'role':p['role'],'count':p['count'],'onsite_fullPhi_exact':str(p['onsite']/Q(1210903848802971,2251799813685248)),'frozen_onsite_exact':str(p['onsite'])} for p in profiles], 'source_oneparticle_phase':'diag(1,-1,-1,1) on each spin replica; spin basis (e0,gamma1e0,e1,gamma1e1), natural physical U identity; complex-to-real harmonic slot phase separately applied by native_slot_map', 'frozen_h':'onsite*m Phi0/scale and matching Jordan endpoint hopping give exactly m A+c0 B+d C; c0,d exact corrected freeframe fractions; not an old equal-role singlet'},
      'trial':{'radius_exact':str(R),'shape':'product_matching (1-s_X)^4_+, s_X=||W_r†Phi_r W_r-Phi_body*||HS²/R²+||W_r†U_X Phi_c U_X† W_r-Phi_body*||HS²/R²; scalar Z^(-1/2) from actual Cartesian x Haar integral; 0<Z<infinity','full_dimensional':'120 Cartesian coordinates/block with15 orbit orientations+105 body/relative coordinates; allSU4 links Haar, no frozenPhi or colour links','section':'A=Phi0+2i[Phi2,Phi3], B=Phi1+Phi2+Phi3+i[Phi0,Phi2]; ordered A eigenvectors; phase tree(0,2),(2,1),(0,3); SU4 determinant fourthroot. W_c=U_X†W_r. Pointwise source Gamma(exp(i theta/2)W_r Wstar†spin, exp(-i theta/2)U_X†W_r Wstar†spin) J_2 tensor J_126 g. CommonZ4 ambiguity acts as z^256=1 per matchingblock. Halfphase yields exp(i theta totalrho_r), not exp(-i theta n_c) with missing background.','exact_g_vs_serialization':'immutable normalized binary x is executed, NOT literally exactg. Exact selected local g=P0x/||P0x|| has squared infidelity cap274877906944/2204801412260314559041; vector error epsg<=sqrt(2)*sqrt(cap). The identical isometric section lifts exactg. For M=V/2 blocks, telescope ||Psi_g-Psi_x||<=M epsg; the displayed complete source-to-action operator cap gives ||H(Psi_g-Psi_x)||<=2^bound M epsg, including unbounded neutralLambda through explicit compact support/action. This is a very poor but honest graph-weighted transfer, not a small fullH residual.','normalization':'J_n extraction identity sum E_a†E_a=I; recursive native tensor isometry; native exterior-power unitary; matching product norm1 pointwise. Scalar normalizer is actual positive integral, not chosen1. Neutral parity wedge4/0 canonical2V, Ref4 singlet unchanged. Each charged species4 particles/block=>2V/species, actual1+63.'},
      'Gauss':['SU4 equivariance of A,B and phase tree yields W(gPhi)=gW(Phi)z; partner U† transports exact endpoint transformation. Z4 cancels on256 block particles, not on each fluctuating vertex independently. Neutral parity and Ref are localSU4 singlets. Shape uses invariant body distances.','U1 exp(i theta rho_r) with totalrho=sum64(n_s,r-2) yields divE=rho; totalrho squared and cross species retained. No per-copy rho² replacement. HC is once-paid longitudinal part of full64E², not added twice.','Original noncompact cover: retain normalized even compact smooth beta packet on all d=2V+1 cycles, |beta_j|<w=1/(64dTrG), G=Bcycle†Bcycle, DBcycle=0; fullE=E0+Bcycle beta. Complete64E² includes128(Bbeta)E0 and64|Bbeta|². Scalar packet derivatives are not theta derivatives; real-cover envelope is Schwartz. ActualN4 TrG738 gives w=1/6092928.'],
      'analytic_jets':['For polynomial A and simple eigenvalues: W1_j=sum_k!=j Wk <k,A1j>/(ej-ek). W2_j=sum_k!=j Wk<k,A2j+2A1W1j-2ej1W1j>/(ej-ek)-||W1j||²Wj. Phase z/|z| and determinant root jets use complete product rules, implemented Jet and spectral_jet. Native exterior_jet includes every distinct second-column insertion.',f'Exact binary centre s={sf},r={rf}; A spectrum +/-s+/-4r², exact minimum gap={exact_gap}>1/8; B tree edge magnitude>=r>1/8. On body tube R2^-24, polynomial perturbations<=16R and Weyl estimates preserve gap/edge>1/16. For one unit Cart coordinate A1,B1<16,A2,B2<16; spectral denominators<16 imply eigenvector first<2^12,second<2^30. Three normalized tree phases and determinant correction give W1<2^30,W2<2^70. Haar U jets<=1. Full256particle exterior first<2^40,second<2^90, with collective cross terms.','Body distance first<2^40,second<2^90. Absolute derivatives of (1-s)^4_+ throughsecond<2^240, including R^-2. Boundary s=1 has zero value and first/second jets: extend byzero outside the regular chart, no boundary delta. C2 compactPhi source belongs D(H) under the smooth-core realization, NOT D(H²).','Normalizer lower: Cartesian coefficient and matching SU4 exponential boxes side delta/2^10,delta=R/2^40, lie s<1/4. Normalized SU4 Haar exponential density lower>2^-100 there (generators Tr(TaTb)=delta/2; compact SU4 volume bound). Scalar factors>=1/4. Phi support cube[-32,32] in60V dimensions gives volume upper2^(360V); Haar volume1. Lower boxes dimension135/block,squared factors>=1/16,Haar density>=2^-100/block yield sqrt(volume_upper/Z)<2^(16384V), thus every scalar first/second L2 ratio<2^(16384V+240). No pointwise boundary log-derivative bound.'],
      'complete_first_H':['H Psi=-4chi sum60V [f_ii U+2f_i U_i+f U_ii]g -(3chi/20)sum15x3V [X²f U+2Xf XU+f X²U]g +64sum3V[-f dtheta²U+2(Bbeta)f(-i dtheta U)+|Bbeta|²fU]g +[VJordan+F Walign+F Wcolour+sum_actual64 dGamma(h_s)-Lambda dGamma(h_neutral)]Psi. U denotes complete endpoint frame product and totalrho phase. Neutral parity and Ref retained; intercell matter action is NOT discarded when its expectation vanishes.','On compact support ||Phi||<4, every scalar profile coefficient and onsite<64, finite-CAR number<=4V/species, six one/two paths all actual64 plus heavy neutral3 nearest paths. Full matter triangle<=2^30V charged and2^30Lambda V neutral. Ordered two-link Wilsons retain actual intermediate links and endpointPhi. Jordan and alignment polynomials bounded respectively2^30Lambda V and2^30FV, colour2^30FV. No ground energy or norm bound is called variance.','Outside support Psi and first/second jets are zero, including boundary vanishing. No exterior Hamiltonian term unpaid: full action maps via actual finite CAR intercell hops outside harmonic/spin/matching subspace but not outside coordinate support. Caps act in all number sectors reached by each hop; full first-H upper includes them.'],
      'N4_complete_action_upper':{'term_log2_upper':terms,'norm_H_Psi_upper':'2^'+str(bound),'generic_residual_upper':'|| (H-E)Psi || <=2^'+str(bound)+'+|E| for any declared E; NOT measured variance or stationarity','N_uniform':False},
      'stationarity_failure_scope':'Trial not admitted stationary. Exposed first-H directions: all60V Phi radial/body/orbit Berry, all Haar link Laplacians, unmatched one/two hopping (native canonical occupation defects), neutral heavy action, totalrho longitudinal and transverse cycle action. Exact selectedg is ground only in frozen one-edge sector; none of these directions is canceled by that theorem. Analytic norm bound, actual Pi action and native controls do not certify fullphysical ground or phase.',
      'supplier_assumptions':['certified all-n harmonic extraction/isometry and corrected source phase from immutable harmonic_car_bridge derivation','accepted selected local gap and infidelity under standardbinary64 guard; exactg hypothesis only used for exactg identification, binary trial construction independent','same-law finite lattice smooth-core/selfadjoint Hamiltonian realization; original-cover cycle direct integral supplier'],
      'controls_scope':'native70 complete colour canonical4 exterior field jets andSU4 controls; actual9.6M g totalrho Pi jets and role omission; no fullN4 numerical H variance or field integral', 'controls':controls,'campaign_complete':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);run(parser.parse_args().epoch)
