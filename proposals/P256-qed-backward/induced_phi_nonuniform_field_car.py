"""Actual N4 all-q normal-Phi45 / relative-SU4-colour45 finite-CAR supplier.

NEW originalFullForm branch: physical negative bands, endpoint Fourier fields,
ordered native links, complex contacts and BOTH q/-q PH processes. This is not
an admission of the frozen original source or an interacting quantum vacuum.
Numerical matrices observe exact finite operators; they do not define the wave.
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

ROOT = Path(__file__).resolve().parent
CAPTURED: dict[str, bytes] = {Path(__file__).name: Path(__file__).read_bytes()}


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes()
        CAPTURED[Path(self.path).name] = raw
        return self.source_to_code(raw, self.path)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.origin is not None:
            source = Path(spec.origin)
            if source.parent == ROOT and source.suffix == '.py':
                spec.loader = Loader(fullname, str(source))
                return spec
        return None


finder = Finder()
sys.meta_path.insert(0, finder)
import induced_phi_uniform_selfconsistent_full_normal_quantum as stationary
import induced_phi_uniform_relative_colour_quantum as uniform
import bosonic_original_law_join as original
import induced_phi_nonuniform_gauss_quantum as gauss
sys.meta_path.remove(finder)
CAPTURED.update(stationary.CAPTURED)
CAPTURED.update(uniform.CAPTURED)
CAPTURED.update(gauss.SOURCE_BYTES)
import mpmath as mp
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

full = stationary.full
UNIFORM_SHA = '8f50cdc7ab6f5860fde58a0d0607ced4afd6549c6b4207d916b1b14839faf01b'
GAUSS_SHA = '2bd70176910108fd695c5490b31ce490da70c860a9a43693758479ec2ccc2c06'
ROOT_SHA = '4691ffaa9cd81902942cbc228fc47fafff14518be6b5df2e84d08c29c7da4756'
FULL_SHA = '5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354'
QINTEGERS = np.asarray(list(product(range(4), repeat=3)), dtype=int)
THETA = (2*np.pi*QINTEGERS+np.pi)/4
LITERAL_Q = ((0,0,0), (1,0,0), (1,2,0), (2,0,0))
sys.set_int_max_str_digits(0)


def encode(value: np.ndarray) -> dict[str, Any]:
    return dict(real=np.real(value).tolist(), imag=np.imag(value).tolist())


def load_mean(root_path: Path = ROOT/'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json', full_path: Path = ROOT/'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json', g_override: Any = None, mean_descriptor: Any = None) -> dict[str, Any]:
    """Build actual microscopic profiles/native geometry; override is a display."""
    rraw, fraw = root_path.read_bytes(), full_path.read_bytes()
    if hashlib.sha256(rraw).hexdigest() != ROOT_SHA or hashlib.sha256(fraw).hexdigest() != FULL_SHA:
        raise ArithmeticError('Actual f63 root/full receipts required')
    if hashlib.sha256(CAPTURED[Path(uniform.__file__).name]).hexdigest() != UNIFORM_SHA:
        raise ArithmeticError('Frozen uniform8f50 source required for q0 comparison')
    rc, fc = json.loads(rraw), json.loads(fraw)
    pars = rc['actual_stationary_root']['parameters']
    intervals = [full.enclosure(F(pars[k]['lower']), F(pars[k]['upper'])) for k in ('g0','gs')]
    centres = [stationary.midpoint(v) for v in intervals]
    if g_override is not None:
        values = list(g_override)
        if len(values) != 4 or not all(float(x)>0 for x in values) or values[1:] != [values[1]]*3:
            raise ValueError('Actual positive cubic g0,gs,gs,gs display required')
        centres = [F(str(values[0])), F(str(values[1]))]
        intervals = [full.exact(v) for v in centres]
    mean = stationary.input_mean(dict(microscopic_config=fc['input_descriptor']['original_micro_config_UNCHANGED']), centres)
    g = [intervals[0], *[intervals[1]]*3]
    native = full.supplier.quantum.build_native(mean, 4)
    geom = full.normal_geometry(native, g)
    profiles = full.scalar.fast.primitive_profiles(mean)
    desc = dict(root_receipt=dict(path=str(root_path),sha256=hashlib.sha256(rraw).hexdigest()), full_receipt=dict(path=str(full_path),sha256=hashlib.sha256(fraw).hexdigest()), physical_mean=mean_descriptor if mean_descriptor is not None else dict(law='f63 ALL65 classical stationary mean, NOT quantum ground', exact_root=pars), numerical_display_g=[float(v) for v in (centres[0], centres[1], centres[1], centres[1])], original_micro_config_UNCHANGED=mean['microscopic_config'], coordinate_order=dict(Phi=geom['labels'],colour='axis-major axis0 T0..14, axis1 T0..14, axis2 T0..14; native edge=3site+axis'), parameters=dict(N=4,V=64,a='1/64',Lambda='2^8192',chi='2^-4096',Gphi='8chi',Gc='3chi/10',fermion_momentum='(2pi k+pi)/4',boson_momentum='2pi q/4',Fourier='exp(iq.x)/sqrt64',physical_species_loads=[1,1,63]), exact_profiles=rc['input_descriptor']['exact_profile_paths'], originalFullForm=uniform.original_contract())
    return dict(mean=mean,native=native,geom=geom,profiles=profiles,g=g,descriptor=desc)

def load_quantum_mean(root_path: Path = ROOT/'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json', full_path: Path = ROOT/'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json') -> dict[str, Any]:
    """Exact NEW all-q feedback mean, native45 geometry and local-Gauss epoch."""
    if hashlib.sha256(CAPTURED[Path(gauss.__file__).name]).hexdigest()!=GAUSS_SHA:
        raise ArithmeticError('Frozen actual nonuniform Gauss source required')
    data=load_mean(root_path,full_path)
    quantum=gauss.solve_mean(root_path,full_path)
    for key in ('mean','native','geom','profiles','g'):
        data[key]=quantum[key]
    data['descriptor']['physical_mean']=dict(law='NEW all64q fast-ZPE feedback implicit root; NOT full quantum ground',exact_root=quantum['root'])
    data['descriptor']['numerical_display_g']=[float(stationary.number(v,mp.mp)) for v in (quantum['centres'][0],*[quantum['centres'][1]]*3)]
    data['descriptor']['local_Gauss_source_at_import_sha256']=GAUSS_SHA
    data['descriptor']['parameters']['Gc_originalFullForm']=data['descriptor']['parameters'].pop('Gc')
    data['descriptor']['parameters']['G_local_Gauss']='chi diag(8I45, .3I45+(d ddagger) tensor Horbit^-1), d_axis=1-exp(iq_axis)'
    return data



def path_terms(profile: dict[str, Any], native: dict[str, Any], geom: dict[str, Any]) -> list[dict[str, Any]]:
    terms = []
    for displacement, coefficients in profile['raw_paths']:
        axis = int(np.flatnonzero(displacement)[0]); length = int(displacement[axis])
        if length not in (1,2):
            raise ArithmeticError('Actual one/two-link ordered path required')
        terms.append(dict(displacement=displacement,axis=axis,length=length,hop=np.einsum('m,mij->ij',coefficients,native['frame']),dPhi=np.einsum('m,amij->aij',coefficients,geom['deltas']),coefficients=coefficients))
    return terms


def bloch_h(profile: dict[str, Any], native: dict[str, Any], terms: list[dict[str, Any]]) -> np.ndarray:
    h = np.broadcast_to(float(profile['onsite'])*native['frame'][0], (64,4,4)).copy().astype(complex)
    for term in terms:
        phase = np.exp(1j*THETA@term['displacement'])
        hop = term['hop']
        h += phase[:,None,None]*hop+phase.conj()[:,None,None]*hop.conj().T
    return (-1 if profile['scale']=='heavy' else 1)*h


def vertices_q(profile: dict[str, Any], native: dict[str, Any], geom: dict[str, Any], terms: list[dict[str, Any]], q: np.ndarray) -> np.ndarray:
    """F(q) outgoing k+q, incoming k; reverse NEVER conjugates field q."""
    T = geom['M']/np.sqrt(8)
    v = np.zeros((64,90,4,4),complex)
    v[:,:45] = float(profile['onsite'])*geom['deltas'][:,0][None]/8
    for term in terms:
        d, axis, length = term['displacement'], term['axis'], term['length']
        phase = np.exp(1j*THETA@d); eq = np.exp(1j*q@d)
        sf = sum(np.exp(1j*q[axis]*j) for j in range(length))/8
        dh, hop = term['dPhi'], term['hop']
        v[:,:45] += phase[:,None,None,None]*(1+eq)*dh[None]/16
        v[:,:45] += (phase.conj()/eq)[:,None,None,None]*(1+eq)*dh.conj().transpose(0,2,1)[None]/16
        cv = .5j*(hop@T+T@hop)*sf
        cr = -.5j*(hop.conj().T@T+T@hop.conj().T)*sf
        sl = slice(45+15*axis,60+15*axis)
        v[:,sl] += phase[:,None,None,None]*cv[None]+(phase.conj()/eq)[:,None,None,None]*cr[None]
    return (-1 if profile['scale']=='heavy' else 1)*v


def contact_q(profile: dict[str, Any], geom: dict[str, Any], terms: list[dict[str, Any]], q: np.ndarray, projector: np.ndarray) -> np.ndarray:
    """D²h[z_a(-q),z_b(q)] from endpoint and ordered exp(iC T)."""
    T = geom['M']/np.sqrt(8); anti = T[:,None]@T[None]+T[None]@T[:,None]
    result = np.zeros((90,90),complex)
    sign = -1 if profile['scale']=='heavy' else 1
    for term in terms:
        axis,length,d = term['axis'],term['length'],term['displacement']
        dh,hop = term['dPhi'],term['hop']; phase=np.exp(1j*THETA@d)
        eqminus = np.exp(-1j*q@d); sf=sum(np.exp(1j*q[axis]*j) for j in range(length))
        pc = .5j/64*(dh[:,None]@T[None]+eqminus*T[None]@dh[:,None])*sf
        pcr = -.5j/64*(T[None]@dh.conj().transpose(0,2,1)[:,None]+eqminus*dh.conj().transpose(0,2,1)[:,None]@T[None])*sf
        pair = np.einsum('kij,k,abji->ab',projector,phase,pc)+np.einsum('kij,k,abji->ab',projector,phase.conj(),pcr)
        sl=slice(45+15*axis,60+15*axis)
        result[:45,sl] += sign*pair
        d2=-length*anti/(2*64); d2r=d2.copy()
        for j in range(length):
            for l in range(j+1,length):
                e=np.exp(1j*q[axis]*(l-j))
                d2 -= (e*T[:,None]@T[None]+e.conjugate()*T[None]@T[:,None])/64
                d2r -= (e*T[None]@T[:,None]+e.conjugate()*T[:,None]@T[None])/64
        cc=.5*(hop@d2+d2@hop); ccr=.5*(d2r@hop.conj().T+hop.conj().T@d2r)
        result[sl,sl] += sign*(np.einsum('kij,k,abji->ab',projector,phase,cc)+np.einsum('kij,k,abji->ab',projector,phase.conj(),ccr))
    result[45:,:45]=result[:45,45:].conj().T
    return result


def response(row: dict[str, Any], frequency: float) -> np.ndarray:
    """Unscaled Euclidean bubble; physical omega = |scale|*frequency."""
    p,m=row['PH_plus'],row['PH_minus']; dp,dm=row['gap_plus'],row['gap_minus']
    return np.einsum('kaij,kbij,kij->ab',p.conj(),p,1/(dp-1j*frequency))+np.einsum('kaij,kbij,kij->ab',m.conj(),m,1/(dm+1j*frequency))


def build_species(profile: dict[str, Any], native: dict[str, Any], geom: dict[str, Any], q_integer: Any) -> dict[str, Any]:
    """Complete per-copy finite-CAR90 operator, both physical bands and ±q."""
    qi=np.asarray(q_integer,dtype=int)%4
    if qi.shape!=(3,):
        raise ValueError('Three periodic integer momentum components required')
    q=2*np.pi*qi/4; terms=path_terms(profile,native,geom)
    h=bloch_h(profile,native,terms); bands,U=np.linalg.eigh(h)
    if not np.all(bands[:,1]<0) or not np.all(bands[:,2]>0):
        raise ArithmeticError('ALL64 occupied negative physical BOTH bands required')
    kout=np.asarray([int(np.ravel_multi_index(tuple((k+qi)%4),(4,4,4))) for k in QINTEGERS],dtype=int)
    vertices=vertices_q(profile,native,geom,terms,q)
    W,Z=U[:,:,:2],U[:,:,2:]; Wo,Zo=W[kout],Z[kout]
    plus=Zo.conj().transpose(0,2,1)[:,None]@vertices@W[:,None]
    minus=Wo.conj().transpose(0,2,1)[:,None]@vertices@Z[:,None]
    dp=bands[kout,2:,None]-bands[:,None,:2]
    dm=bands[:,None,2:]-bands[kout,:2,None]
    row: dict[str, Any]=dict(profile=profile,sign=-1 if profile['scale']=='heavy' else 1,q_integer=qi,q=q,h=h,bands=bands,U=U,vertices=vertices,PH_plus=plus,PH_minus=minus,gap_plus=dp,gap_minus=dm,terms=terms,kout=kout)
    row['B']=response(row,0.)
    row['contact']=contact_q(profile,geom,terms,q,W@W.conj().transpose(0,2,1))
    row['sea']=row['contact']-row['B']
    row['minimum_gap']=float(min(dp.min(),dm.min()))
    row['hermiticity']=dict(B=float(np.max(abs(row['B']-row['B'].conj().T))),contact=float(np.max(abs(row['contact']-row['contact'].conj().T))),sea=float(np.max(abs(row['sea']-row['sea'].conj().T))))
    if max(row['hermiticity'].values())>5e-11:
        raise ArithmeticError('Complex Hermitian finite-q static operator failed')
    return row


def build_all_q(data: dict[str, Any]) -> list[dict[str, Any]]:
    result=[]
    for qi in QINTEGERS:
        rows=[build_species(p,data['native'],data['geom'],qi) for p in data['profiles']]
        result.append(dict(q_integer=qi,species=rows,sea_heavy=rows[0]['sea'],sea_charged=sum(p['load']*r['sea'] for p,r in zip(data['profiles'][1:],rows[1:],strict=True))))
    return result


def native_paths(profile: dict[str, Any], native: dict[str, Any]) -> list[dict[str, Any]]:
    result=[]
    for displacement,c in profile['raw_paths']:
        support=[s for s in native['Pprofiles'][0]['paths'] if np.array_equal(s['displacement'],displacement)]
        if len(support)!=64:
            raise ArithmeticError('ALL64 native path translates and AP seams required')
        for s in support:
            if len(s['edges'])!=int(np.sum(displacement)) or any(e%3!=s['axis'] for e in s['edges']):
                raise ArithmeticError('Native ordered edge convention changed')
            result.append(dict(s,coefficients=c))
    return result


def assemble(profile: dict[str, Any], native: dict[str, Any], fields: np.ndarray, links: np.ndarray) -> np.ndarray:
    """Independent literal original hxy=(Fx Uxy+Uxy Fy)/2 on native paths."""
    h=np.zeros((256,256),complex)
    for x in range(64):
        h[4*x:4*x+4,4*x:4*x+4]=float(profile['onsite'])*fields[x,0]
    for path in native_paths(profile,native):
        x,y=int(path['source']),int(path['target']); c=path['coefficients']
        U=np.eye(4,dtype=complex)
        for edge in path['edges']:
            U=U@links[edge]
        Fx,Fy=np.einsum('m,mij->ij',c,fields[x]),np.einsum('m,mij->ij',c,fields[y])
        hop=path['seam']*(Fx@U+U@Fy)/2
        h[4*x:4*x+4,4*y:4*y+4]+=hop
        h[4*y:4*y+4,4*x:4*x+4]+=hop.conj().T
    return (-1 if profile['scale']=='heavy' else 1)*h


def literal_vertices(profile: dict[str, Any], native: dict[str, Any], geom: dict[str, Any], qi: np.ndarray) -> np.ndarray:
    """Differentiate actual endpointfields/edge exponentials BEFORE Fourier."""
    q=2*np.pi*qi/4; fs=np.exp(1j*native['sites']@q)/8; T=geom['M']/np.sqrt(8)
    v=np.zeros((90,256,256),complex)
    for x in range(64):
        v[:45,4*x:4*x+4,4*x:4*x+4]=float(profile['onsite'])*geom['deltas'][:,0]*fs[x]
    for path in native_paths(profile,native):
        x,y=int(path['source']),int(path['target']); c=path['coefficients']; ax=path['axis']
        hop=np.einsum('m,mij->ij',c,native['frame']); dh=np.einsum('m,amij->aij',c,geom['deltas'])
        dU=np.zeros((15,4,4),complex)
        dUr=np.zeros_like(dU)
        for edge in path['edges']:
            dU += 1j*T*fs[edge//3]
            dUr += -1j*T*fs[edge//3]
        forward=np.concatenate((dh*(fs[x]+fs[y])/2,(hop@dU+dU@hop)/2))
        reverse=np.concatenate((dh.conj().transpose(0,2,1)*(fs[x]+fs[y])/2,(dUr@hop.conj().T+hop.conj().T@dUr)/2))
        slots=np.r_[np.arange(45),np.arange(45+15*ax,60+15*ax)]
        sl,tl=slice(4*x,4*x+4),slice(4*y,4*y+4)
        v[slots,sl,tl]+=path['seam']*forward
        v[slots,tl,sl]+=path['seam']*reverse
    return (-1 if profile['scale']=='heavy' else 1)*v


def literal_contact(profile: dict[str, Any], native: dict[str, Any], geom: dict[str, Any], qi: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Differentiate ordered one/two-link matrices on actual native edges."""
    q=2*np.pi*qi/4; fs=np.exp(1j*native['sites']@q)/8; T=geom['M']/np.sqrt(8)
    anti=T[:,None]@T[None]+T[None]@T[:,None]
    result=np.zeros((90,90),complex)
    for path in native_paths(profile,native):
        x,y=int(path['source']),int(path['target']); ax=path['axis']; c=path['coefficients']
        hop=np.einsum('m,mij->ij',c,native['frame']); dh=np.einsum('m,amij->aij',c,geom['deltas'])
        # Separate derivative directions have momenta -q,+q. Reverse adjoints
        # reverse matrix products, not their Fourier coefficient directions.
        dU=sum((1j*T*fs[e//3] for e in path['edges']),np.zeros((15,4,4),complex))
        dUr=sum((-1j*T*fs[e//3] for e in path['edges']),np.zeros((15,4,4),complex))
        pc=(dh[:,None]*fs[x].conjugate()@dU[None]+dU[None]@dh[:,None]*fs[y].conjugate())/2
        dhr=dh.conj().transpose(0,2,1)
        pcr=(dUr[None]@dhr[:,None]*fs[x].conjugate()+dhr[:,None]*fs[y].conjugate()@dUr[None])/2
        d2=np.zeros((15,15,4,4),complex); d2r=np.zeros_like(d2)
        edges=path['edges']
        for j,e in enumerate(edges):
            d2 -= anti*fs[e//3].conjugate()*fs[e//3]/2
            d2r -= anti*fs[e//3].conjugate()*fs[e//3]/2
            for f in edges[j+1:]:
                a,b=fs[e//3].conjugate()*fs[f//3],fs[e//3]*fs[f//3].conjugate()
                d2 -= T[:,None]@T[None]*a+T[None]@T[:,None]*b
                d2r -= T[None]@T[:,None]*a+T[:,None]@T[None]*b
        cc=(hop@d2+d2@hop)/2; ccr=(d2r@hop.conj().T+hop.conj().T@d2r)/2
        block=P[4*y:4*y+4,4*x:4*x+4]; back=P[4*x:4*x+4,4*y:4*y+4]
        factor=path['seam']*(-1 if profile['scale']=='heavy' else 1)
        sl=slice(45+15*ax,60+15*ax)
        result[:45,sl]+=factor*(np.einsum('ij,abji->ab',block,pc)+np.einsum('ij,abji->ab',back,pcr))
        result[sl,sl]+=factor*(np.einsum('ij,abji->ab',block,cc)+np.einsum('ij,abji->ab',back,ccr))
    result[45:,:45]=result[:45,45:].conj().T
    return result


def literal_check(row: dict[str, Any], native: dict[str, Any], geom: dict[str, Any]) -> dict[str, Any]:
    p,qi=row['profile'],row['q_integer']
    fields=np.broadcast_to(native['frame'],(64,4,4,4)); links=np.broadcast_to(np.eye(4),(192,4,4))
    h=assemble(p,native,fields,links); v=literal_vertices(p,native,geom,qi)
    vm=literal_vertices(p,native,geom,(-qi)%4)
    fourier=np.kron(np.exp(1j*native['sites']@THETA.T)/8,np.eye(4))
    expectedh=np.zeros_like(h); expectedv=np.zeros_like(v)
    for k,out in enumerate(row['kout']):
        expectedh[4*k:4*k+4,4*k:4*k+4]=row['h'][k]
        expectedv[:,4*out:4*out+4,4*k:4*k+4]=row['vertices'][k]
    ev,U=np.linalg.eigh(h); W,Z=U[:,:128],U[:,128:]
    if not ev[127]<0<ev[128]:
        raise ArithmeticError('Literal256 physical BOTH-band filling failed')
    a=Z.conj().T@v@W; b=W.conj().T@v@Z
    dp=ev[128:,None]-ev[None,:128]; dm=ev[None,128:]-ev[:128,None]
    B=np.einsum('aij,bij,ij->ab',b.conj(),b,1/dm)+np.einsum('aij,bij,ij->ab',a.conj(),a,1/dp)
    contact=literal_contact(p,native,geom,qi,W@W.conj().T)
    errors=dict(full256_original_h=float(np.max(abs(fourier.conj().T@h@fourier-expectedh))),ALL90_full256_currents=float(np.max(abs(fourier.conj().T@v@fourier-expectedv))),q_minusq_reverse=float(np.max(abs(vm-v.conj().transpose(0,2,1)))),ALL90_fullPH=float(np.max(abs(B-row['B']))),ALL_PC_CC_contacts=float(np.max(abs(contact-row['contact']))),ALL90_sea=float(np.max(abs(contact-B-row['sea']))))
    dynamic=[]
    for om in (0.,1.,10.,100.):
        measured=np.einsum('aij,bij,ij->ab',a.conj(),a,1/(dp-1j*om))+np.einsum('aij,bij,ij->ab',b.conj(),b,1/(dm+1j*om))
        predicted=response(row,om); err=float(np.max(abs(measured-predicted)))
        errors['complex_dynamic_'+str(om)]=err
        dynamic.append(dict(unscaled_frequency=om,absolute_error=err,imaginary_norm=float(np.linalg.norm(np.imag(predicted))),odd_frequency_norm=float(np.linalg.norm((predicted-response(row,-om))/2))))
    result=dict(role=p['role'],load=p['load'],q_integer=qi.tolist(),literal_current_shape=list(v.shape),literal_PH_shapes=[list(a.shape),list(b.shape)],all256_physical_bands=ev.tolist(),errors=errors,dynamic=dynamic)
    if max(errors.values())>5e-10:
        raise ArithmeticError(json.dumps(dict(failed_literal=result)))
    return result


def gauge_jet(profile: dict[str, Any], native: dict[str, Any], generators: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Actual first/second h jet along Phi'=gPhi gdag,U'=gx gydag."""
    phi=native['frame']; dphi=1j*(generators[:,None]@phi[None]-phi[None]@generators[:,None])
    ddphi=-(generators[:,None]@generators[:,None]@phi[None]-2*generators[:,None]@phi[None]@generators[:,None]+phi[None]@generators[:,None]@generators[:,None])
    first=np.zeros((256,256),complex); second=np.zeros_like(first)
    for x in range(64):
        first[4*x:4*x+4,4*x:4*x+4]=float(profile['onsite'])*dphi[x,0]
        second[4*x:4*x+4,4*x:4*x+4]=float(profile['onsite'])*ddphi[x,0]
    for path in native_paths(profile,native):
        x,y=int(path['source']),int(path['target']); c=path['coefficients']
        hop=np.einsum('m,mij->ij',c,phi)
        dx,dy=np.einsum('m,mij->ij',c,dphi[x]),np.einsum('m,mij->ij',c,dphi[y])
        ddx,ddy=np.einsum('m,mij->ij',c,ddphi[x]),np.einsum('m,mij->ij',c,ddphi[y])
        dU=np.zeros((4,4),complex); ddU=np.zeros_like(dU)
        for edge in path['edges']:
            source=edge//3; axis=edge%3
            target=int(np.ravel_multi_index(tuple((native['sites'][source]+np.eye(3,dtype=int)[axis])%4),(4,4,4)))
            gx,gy=generators[source],generators[target]
            du=1j*(gx-gy); ddu=-gx@gx+2*gx@gy-gy@gy
            ddU += ddu+2*dU@du
            dU += du
        f=(dx+dy+hop@dU+dU@hop)/2
        s=(ddx+ddy+2*dx@dU+2*dU@dy+hop@ddU+ddU@hop)/2
        sl,tl=slice(4*x,4*x+4),slice(4*y,4*y+4)
        first[sl,tl]+=path['seam']*f; first[tl,sl]+=path['seam']*f.conj().T
        second[sl,tl]+=path['seam']*s; second[tl,sl]+=path['seam']*s.conj().T
    sign=-1 if profile['scale']=='heavy' else 1
    return sign*first,sign*second


def gauge_ward(profile: dict[str, Any], native: dict[str, Any], geom: dict[str, Any], qi: tuple[int,int,int]) -> dict[str, Any]:
    """Finite noncommuting local SU4 and first/second Ward, including AP seams."""
    q=2*np.pi*np.asarray(qi)/4; T=geom['M']/np.sqrt(8)
    phase=native['sites']@q
    gx=np.cos(phase)[:,None,None]*T[1]+.37*np.sin(phase)[:,None,None]*T[4]
    h=assemble(profile,native,np.broadcast_to(native['frame'],(64,4,4,4)),np.broadcast_to(np.eye(4),(192,4,4)))
    G=np.zeros((256,256),complex)
    for x in range(64):
        G[4*x:4*x+4,4*x:4*x+4]=gx[x]
    first,second=gauge_jet(profile,native,gx)
    comm=1j*(G@h-h@G); dd=-(G@G@h-2*G@h@G+h@G@G)
    errors=dict(infinitesimal_angularPhi_linkgradient=float(np.max(abs(first-comm))),second_PC_CC_ordered_link_Ward=float(np.max(abs(second-dd))))
    finite=[]
    for eps in (.13,.047):
        gauges=np.asarray([expm(1j*eps*g) for g in gx]); fields=gauges[:,None]@native['frame'][None]@gauges.conj().transpose(0,2,1)[:,None]
        links=[]
        for x in range(64):
            for axis in range(3):
                y=int(np.ravel_multi_index(tuple((native['sites'][x]+np.eye(3,dtype=int)[axis])%4),(4,4,4)))
                links.append(gauges[x]@gauges[y].conj().T)
        actual=assemble(profile,native,fields,np.asarray(links))
        transformed=np.zeros_like(h)
        for x in range(64):
            for y in range(64):
                transformed[4*x:4*x+4,4*y:4*y+4]=gauges[x]@h[4*x:4*x+4,4*y:4*y+4]@gauges[y].conj().T
        err=float(np.max(abs(actual-transformed)))
        finite.append(dict(epsilon=eps,full256_gauge_covariance_error=err,negative_sea_energy_error=float(abs(np.linalg.eigvalsh(actual)[:128].sum()-np.linalg.eigvalsh(h)[:128].sum()))))
        errors['finite_'+str(eps)]=err
    # Exercise the named original endpoint itself, not only the equivalent loop.
    p0=next(s for s in native_paths(profile,native) if s['length']==1 and s['source']==0)
    target=int(p0['target']); axis=int(p0['axis'])
    U=gauges[0]@gauges[target].conj().T
    endpoint_actual=original.original_endpoint(fields[0],fields[target],U,axis,profile['model'])
    c=p0['coefficients']; endpoint_literal=(np.einsum('m,mij->ij',c,fields[0])@U+U@np.einsum('m,mij->ij',c,fields[target]))/2
    errors['original_endpoint_owner']=float(np.max(abs(endpoint_actual-endpoint_literal)))
    result=dict(role=profile['role'],q_integer=list(qi),noncommuting_generator_commutator=float(np.linalg.norm(T[1]@T[4]-T[4]@T[1])),errors=errors,finite=finite,scope='Angular Phi + link-gradient Ward only; numerical PH90 remains true physical normal Phi45, not angular retag.')
    if max(errors.values())>5e-10:
        raise ArithmeticError(json.dumps(dict(failed_Ward=result)))
    return result


def bare_coefficients(data: dict[str, Any], qi: np.ndarray) -> dict[str, Any]:
    """Consume frozen GaussMean physical K and full reduced cometric."""
    geometry=gauss.build_geometry(data['native'],data['geom'],qi)
    K1=np.zeros((90,90),complex); K1[:45,:45]=geometry['Kprim45']
    return dict(Kprimitive=K1,Kfast=geometry['Kfast90'],Gdivchi=geometry['Gdiag'],Horbit=geometry['H'],laplacian=geometry['qhat2'])


def exact_operator_description() -> dict[str, Any]:
    return dict(coordinate='Phi_x=Phi*(g*)+sum_q exp(iq.x)/8 sum_a z_Phi,a(q) deltaPhi_a; U_x,axis=exp(i sum_q exp(iq.x)/8 sum_A z_C,axis,A(q) T_A). z(-q)=z(q)*. 64 momenta x90 =5760 REAL coordinates, selfinverse q counted once; q/-q pairs are real cosine/sine, never duplicate modes.', originalFullForm_normalization=dict(TrTT='delta/2',Casimir='15/8',G='unreduced chi diag(8I45,3I45/10); localGauss reduction G(q) supplied by GaussMean',alignment='.2Lambda^5/4 Frobenius²/8',plaquette='.7Lambda^5/4(4-ReTrUface)',primitive='Lambda exact45 Gram Hessian; no bare EM curl'), operator='K(q)=Lambda^5/4 Kfast(q)+Lambda [Kprimitive(q)+sea_heavy(q)]+sea_active(q)+63sea_aux(q). sea_s(q)=Tr Pocc D²h_s[-q,+q]−sum BOTH particle/hole q,-q matrix quotients. Negative physical heavy occupies upper original h BOTH bands. Physical |scale| rescales gap/current/contact and dynamic omega; load counts independent CAR copies.', exact_static_wave='A(q)=G(q)^-1/2 (G(q)^1/2 K(q) G(q)^1/2)^1/2 G(q)^-1/2; A G A=K, Cov(q)=.5 A(q)^-1. Psi over all5760 real coordinates is normalized det(A/pi)^1/4 exp(-z^T A z/2). Exact finite-CAR projectors, original ordered SU4 derivatives, implicit-root Clifford Q and algebraic AP/q phases define K; rounded arrays/eigenvectors NEVER define coefficients.', dynamics='B_ab(q,iomega)=sum PHplus_a* PHplus_b/(Dplus-iomega)+PHminus_a* PHminus_b/(Dminus+iomega), with physical scale restored. Complex gyro and q/-q processes retained. Static-PH Gaussian is a selected preparation, NOT full interacting/PH-dynamical/nonlinear vacuum.', rigorous_bounds='For exact implicit positive g box and exact binary microscopic coefficients, gap>=2 min_k sqrt(sum_mu g_mu x_s,mu(k)^2); AP phases belong Q(sqrt2,i). Outward exact_operator_bounds below evaluate gaps and allq vertex/contact/bubble/sea operator bounds without rounded matrices. Filled rank128 gives |contact_ab|<=128||D²h_ab||. Norm prices retain load and physical scale separately.', limits='No photon speed/common cone/QED/eR/scattering/anomaly/vacuum/original SOURCE claim. Full source-prepared consumer is next owner; this branch is not frozen427 promotion.')


def exact_operator_bounds(profile: dict[str, Any], g: list[Any]) -> dict[str, Any]:
    """Outward algebraic ALL-q bounds, independent of numerical PH matrices."""
    ex,iv,bounds=full.exact,full.iv,full.bounds
    halfroot=iv.sqrt(ex(F(2)))/2
    cos1,sin1,sin2=(1,-1,-1,1),(1,1,-1,-1),(1,-1,1,-1)
    energies=[]
    for k in QINTEGERS:
        x=[ex(profile['onsite']),ex(0),ex(0),ex(0)]
        for p in profile['paths']:
            axis,L=p['axis'],p['length']
            co=cos1[k[axis]]*halfroot if L==1 else ex(0)
            si=sin1[k[axis]]*halfroot if L==1 else ex(sin2[k[axis]])
            x[0]+=2*ex(p['c0'])*co; x[axis+1]-=2*ex(p['ci'])*si
        energies.append(iv.sqrt(sum(g[m]*x[m]**2 for m in range(4))))
    minimum=min(full.endpoint(e,0) for e in energies)
    gap=ex(2*minimum)
    if minimum<=0:
        raise ArithmeticError('Exact physical spectral gap bound failed')
    roots=[iv.sqrt(v) for v in g]; l1=F(0); cweighted=ex(0); pc=ex(0); cc=ex(0)
    for p in profile['paths']:
        coeff=abs(p['c0'])+abs(p['ci']); L=p['length']
        hop=abs(ex(p['c0']))*roots[0]+abs(ex(p['ci']))*roots[p['axis']+1]
        l1+=coeff; cweighted+=L*hop; pc+=L*ex(coeff); cc+=ex(F(L*L,2))*hop
    vp=(abs(ex(profile['onsite']))+2*ex(l1))/(8*iv.sqrt(ex(2)))
    vc=cweighted/(8*iv.sqrt(ex(2)))
    contact=45*iv.sqrt(2*pc**2+cc**2)
    bubble=256*45*(vp**2+vc**2)/gap
    return dict(role=profile['role'],load=profile['load'],physical_scale='Lambda' if profile['scale']=='heavy' else '1',minimum_gap_unscaled=bounds(gap),Phi_vertex_operator_upper=bounds(vp),colour_vertex_operator_upper=bounds(vc),PC_contact_entry_upper=bounds(pc),CC_contact_entry_upper=bounds(cc),contact90_operator_upper=bounds(contact),static_and_ALLreal_Euclidean_bubble90_operator_upper=bounds(bubble),sea90_operator_upper=bounds(contact+bubble),proof='For unit normal Phi, each deltaPhi component opnorm<=1/sqrt2; ||T_A||=1/sqrt8. Native row bounds count each forward and reverse one/two-link path separately. PC entry<=sum L(|c0|+|ci|), CC entry<=.5 sum L² hopnorm after occupied rank128. Contact Frobenius bound=45sqrt(2 PC²+CC²). Each ±q PH process HS²<=128 sum90||F_a||²; both processes give bubble<=256*45(vPhi²+vC²)/gap. Physical scale/load multiply these unscaled bounds; no array residual is promoted to an interval certificate.')


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    p=row['profile']; controls=[]
    for om in (0.,1.,10.,100.):
        r=response(row,om)
        controls.append(dict(unscaled_frequency=om,PH=encode(r),odd_frequency_norm=float(np.linalg.norm((r-response(row,-om))/2)),imaginary_norm=float(np.linalg.norm(np.imag(r)))))
    return dict(role=p['role'],load=p['load'],physical_scale='Lambda' if p['scale']=='heavy' else '1',occupancy='upper h BOTH bands' if p['scale']=='heavy' else 'negative h BOTH bands',B=encode(row['B']),contact=encode(row['contact']),sea=encode(row['sea']),bands_unscaled=row['bands'].tolist(),gaps_plus_unscaled=row['gap_plus'].tolist(),gaps_minus_unscaled=row['gap_minus'].tolist(),PH_plus=encode(row['PH_plus']),PH_minus=encode(row['PH_minus']),minimum_gap=row['minimum_gap'],hermiticity=row['hermiticity'],complex_dynamic_controls=controls,observed_operator_norms=dict(B=float(np.linalg.norm(row['B'],2)),contact=float(np.linalg.norm(row['contact'],2)),sea=float(np.linalg.norm(row['sea'],2)),vertices_max_component=float(np.max(np.linalg.norm(row['vertices'],ord=2,axis=(2,3)))),vertices_ALL90_HS_squared=float(np.sum(abs(row['vertices'])**2))))


def spectral_observation(data: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
    bare=bare_coefficients(data,block['q_integer']); K1=bare['Kprimitive']+block['sea_heavy']; K0=block['sea_charged']; Kf=bare['Kfast']
    gv,gu=np.linalg.eigh(bare['Gdivchi'])
    if gv[0]<=0:
        raise ArithmeticError('Actual local-Gauss cometric not positive')
    sqrtG=(gu*np.sqrt(gv))@gu.conj().T
    scaled=sqrtG@Kf@sqrtG
    values,U=np.linalg.eigh(scaled); error=float(np.linalg.norm(scaled-(U*values)@U.conj().T,2)); norm1=float(np.linalg.norm(sqrtG@K1@sqrtG,2)); norm0=float(np.linalg.norm(sqrtG@K0@sqrtG,2))
    result=dict(K_Lambda5over4=encode(Kf),K_Lambda=encode(K1),K_unit=encode(K0),complex_Hermitian_K_coefficients=True,leading_frequency_squared_coefficient_units_chi_Lambda5over4=values.tolist(),fast_eigen_reconstruction_error=error,exact_slow_perturbation_bound=dict(expression='2^-2048 ||sqrt(G/chi)K1sqrt(G/chi)|| + 2^-10240 ||sqrt(G/chi)K0sqrt(G/chi)||',coefficient_Lambda=norm1,coefficient_unit=norm0),no_slow_terms_dropped='Exact K functional retains all three coefficient matrices; floating leading spectra are observations with explicit nonzero dyadic perturbation bounds.',physical_mode_count=90,spectrum_scope='For nonzero q, all90 have positive leading fast coefficients. At q0, Kfast has45 null slowPhi directions, NOT physical zero-frequency modes; the45 positive slowPhi stiffness eigenvalues are listed separately. These two scales together count90, not135 modes.')
    result['G_divided_chi']=encode(bare['Gdivchi'])
    result['actual_local_Gauss_cometric_eigenvalues_divided_chi']=gv.tolist()
    if np.any(block['q_integer']):
        if values[0]<=0:
            raise ArithmeticError('Nonuniform originalFullForm fast coefficient not positive')
        result['positive_static_operator_argument']='lambda_min(Kfast)>displayed lower−residual; exact norm-priced 2^-2048 perturbation and 2^-10240 charged term are smaller. All90 physical eigenbranches, no mode projection.'
    else:
        phi=K1[:45,:45]; pv=np.linalg.eigvalsh(phi)
        if pv[0]<=0 or np.linalg.eigvalsh(Kf[45:,45:])[0]<=0:
            raise ArithmeticError('Uniform slow/fast diagonal blocks not positive')
        result.update(uniform_slow_Phi_eigenvalues_divided_Lambda=pv.tolist(),uniform_cross_norm_divided_Lambda=float(np.linalg.norm(K1[:45,45:],2)),positive_static_operator_argument='Phi block Lambda-positive and colour block Lambda5/4-positive; Schur complement cross correction is O(Lambda^-1/4) relative to slow block, explicitly norm-priced. Charged matrix remains in exact definition.')
    return result


def run(root_path: Path, full_path: Path) -> dict[str, Any]:
    started=time.monotonic(); full.iv.dps=80; mp.mp.dps=80
    data=load_quantum_mean(root_path,full_path); blocks=build_all_q(data)
    literal=[]; wards=[]; q0=[]; public=[]
    colour=uniform.colour_geometry(data['native'],data['geom'])
    for block in blocks:
        qi=tuple(int(v) for v in block['q_integer'])
        species=block['species']
        if qi in LITERAL_Q:
            for row in species:
                literal.append(literal_check(row,data['native'],data['geom']))
        if qi==(0,0,0):
            for row in species:
                source=uniform.bloch_species(row['profile'],data['native'],data['geom'],colour)
                errors={key:float(np.max(abs(row[key]-source[key]))) for key in ('B','contact','sea')}
                if max(errors.values())>5e-10:
                    raise ArithmeticError(json.dumps(dict(failed_q0_ABI=errors,role=row['profile']['role'])))
                q0.append(dict(role=row['profile']['role'],errors=errors,interpretation='Observable comparison only; q0 is executed by nonuniform endpoint Fourier code, not projected/copied from uniform.'))
        public.append(dict(q_integer=list(qi),species=[public_row(r) for r in species],actual_static_spectral_supplier=spectral_observation(data,block)))
    for qi in ((1,0,0),(1,2,0),(2,0,0)):
        for p in data['profiles']:
            wards.append(gauge_ward(p,data['native'],data['geom'],qi))
    reality=[]
    for block in blocks:
        qi=block['q_integer']; neg=int(np.ravel_multi_index(tuple((-qi)%4),(4,4,4))); other=blocks[neg]
        errors=[float(np.max(abs(a['sea'].conj()-b['sea']))) for a,b in zip(block['species'],other['species'],strict=True)]
        if max(errors)>5e-10:
            raise ArithmeticError(json.dumps(dict(failed_reality=qi.tolist(),errors=errors)))
        reality.append(dict(q_integer=qi.tolist(),q_minusq_complex_conjugation_errors=errors))
    descriptor=dict(data['descriptor'],law='NEW all64q5760 originalFullForm static-PH supplier at actual specified mean; not SOURCE promotion',literal_q=[list(q) for q in LITERAL_Q],q0_comparator_source_sha256=UNIFORM_SHA,exact_operator=exact_operator_description())
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_number_periodic_q=64,actual_number_real_physical_coordinates=5760,actual_ALL64_q=public,exact_ALLq_operator_bounds=[exact_operator_bounds(p,data['g']) for p in data['profiles']],actual_independent_literal256_ALL90=literal,actual_original_local_SU4_Ward=wards,actual_q0_observable_ABI=q0,actual_complex_q_reality=reality,source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(raw).hexdigest(),bytes_utf8=raw.decode()) for name,raw in CAPTURED.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==raw for name,raw in CAPTURED.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-input',type=Path,default=ROOT/'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json')
    parser.add_argument('--full-input',type=Path,default=ROOT/'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        result=run(args.root_input,args.full_input)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Scientific source epoch changed during execution')
    with args.output.open('w') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(input_sha256=result['input_sha256'],actual_q=result['actual_number_periodic_q'],actual_real_coordinates=result['actual_number_real_physical_coordinates'],literal_checks=len(result['actual_independent_literal256_ALL90']),local_Ward_checks=len(result['actual_original_local_SU4_Ward']),elapsed_seconds=result['elapsed_seconds']),indent=2))
