#!/usr/bin/env python3
"""Actual FULL4V light law/current and untruncated EM Weyl factorization.

No active8 CAR clipping. Field projection is explicitly the shared9 chart,
not full90V; the full-Fock light operator is supplied as exact bilinear factors.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, cast
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from scipy.linalg import expm
from prepared_guard_colour_chart import Jet2,constant,onebody_coordinate_jets
from prepared_neutral_spatial_gauss_well import PreparedSpatialGaussQuantumSlice
from sea_selection import SELECTED
from original_charged_principal import portable
from improved_phi_band import C


def shared_coordinate_fields(model:Any,ip:int,ic:int) -> tuple[np.ndarray,np.ndarray]:
    """Exactly the existing owner's Phi/colour rule before CAR projection."""
    p=model.points[ip]*model.scales[0];c=model.points[ic]*model.scales[1]
    phi=np.tile(model.frame,(model.law.volume,1,1,1)).copy()
    phi[:,0]+=p[0]*SELECTED[1]+p[1]*SELECTED[2]
    links=np.tile(np.eye(4,dtype=complex),(model.law.volume,3,1,1))
    links[:,2]=expm(1j*(c[0]*SELECTED[1]+c[1]*SELECTED[2])/math.sqrt(8))
    return phi,links


def full_coordinate_light_operators(law:Any,phi:np.ndarray,links:np.ndarray,
                                    a:np.ndarray,first:np.ndarray,
                                    second:np.ndarray|None=None) -> dict[str,np.ndarray]:
    """Complex-linear J and bilinear Hessian C, actual real-field primitive.

    Guard's physical jet takes REAL tangent fields. Split complex travelling
    profiles explicitly, rather than silently discarding their imaginary part.
    No photon coefficient is applied here; a includes explicit theta/N.
    """
    a=np.asarray(a)
    if np.any(cast(Any,a).imag):raise ValueError('Base EM coordinate is a real physical field')
    first=np.asarray(first,complex);second=first if second is None else np.asarray(second,complex)
    Phi=constant(np.asarray(phi));U=constant(np.asarray(links));cache={}
    def jet(tangent):
        tangent=np.asarray(tangent,float);key=tangent.tobytes()
        if key not in cache:cache[key]=onebody_coordinate_jets(law,Phi,U,Jet2(cast(Any,a).real,tangent,np.zeros_like(tangent)))
        return cache[key]
    def cross(u,v):
        return (jet(u+v).second-jet(u).second-jet(v).second)/2
    ar,ai=cast(Any,first).real,cast(Any,first).imag;br,bi=cast(Any,second).real,cast(Any,second).imag
    current=jet(ar).first+1j*jet(ai).first
    contact=cross(ar,br)-cross(ai,bi)+1j*(cross(ar,bi)+cross(ai,br))
    return dict(H=jet(ar).value,J=current,C=contact)


def coexact_plane_profile(law:Any,mode:tuple[int,int,int],polarization:np.ndarray) -> tuple[np.ndarray,float]:
    pol=np.array(polarization,dtype=complex,copy=True)
    if np.linalg.norm(pol)==0:raise ValueError('Zero polarization')
    pol/=np.linalg.norm(pol)
    wrapped=[int(q)%law.size for q in mode]
    if not any(wrapped):raise ValueError('Harmonic zero mode is not a coexact photon')
    phase=np.array([2*np.pi*q/law.size for q in wrapped]);gradient=np.exp(1j*phase)-1
    # Actual curl†curl projector, not continuum k.dot(e)=0 substitution.
    pol-=gradient*np.vdot(gradient,pol)/np.vdot(gradient,gradient).real
    norm=np.linalg.norm(pol)
    if norm<1e-12:raise ValueError('Polarization has no discrete coexact component')
    pol/=norm
    profile=np.exp(1j*(law.sites@phase))[:,None]*pol[None,:]/math.sqrt(law.volume)
    return profile,float(C)*float(np.sqrt(np.vdot(gradient,gradient).real))


def photon_current_legs(law:Any,phi:np.ndarray,links:np.ndarray,a:np.ndarray,
                        first_profile:np.ndarray,first_omega:float,
                        second_profile:np.ndarray,second_omega:float,ue:float) -> dict[str,np.ndarray]:
    raw=full_coordinate_light_operators(law,phi,links,a,first_profile,second_profile)
    return dict(H=raw['H'],J=math.sqrt(ue/first_omega)*raw['J'],
                C=ue/math.sqrt(first_omega*second_omega)*raw['C'])


def full_light_field_factorization(model:Any) -> dict[str,Any]:
    """Exact dGamma108 factors on the SAME shared9 Phi/colour chart.

    Every directed hopping carries its actual ALL55 coordinate Weyl word.
    The52 coexact oscillators and3 free harmonic coordinates are UNTRUNCATED.
    The returned auxiliary loop factor is kept, not replaced by a number.
    """
    law=model.law;nodes=[shared_coordinate_fields(model,ip,ic) for ip,ic in model.pc_points]
    Q=model.Q2
    def project(blocks):return np.einsum('qa,qij,qb->abij',Q,np.asarray(blocks),Q)
    onsite=[];hopping=[]
    for x in range(law.volume):
        onsite.append(dict(site=x,block=project([law.onsite*phi[x,0] for phi,_ in nodes])))
        for displacement,scalar in law.paths:
            axis=int(np.flatnonzero(displacement)[0]);length=int(displacement[axis]);samples=[];y=x;word=np.zeros(55)
            for _ in range(length):word+=model.em_maps[:,y,axis];y=int(law.target[y,axis])
            for phi,links in nodes:
                z=x;tree=np.eye(4,dtype=complex)
                for _ in range(length):tree=tree@links[z,axis];z=int(law.target[z,axis])
                left=np.einsum('m,mij->ij',scalar,phi[x]);right=np.einsum('m,mij->ij',scalar,phi[y])
                samples.append((left@tree+tree@right)/2)
            hopping.append(dict(source=x,target=y,axis=axis,length=length,block=project(samples),word=word))
    # The reference is the SAME native static bank: no invented onsite mass.
    # Its full4 colour/Gauss action and actual Coulomb charge stay present.
    frozen_light_onsite=(law.volume-2)*np.einsum('ij,abji->ab',(np.eye(4)-SELECTED[0])/2,model.onsite_coeff)
    rest_scalar=model.scalar_matrix-frozen_light_onsite
    phi_kinetic=(model.chi/(2*model.scales[0]**2)*np.kron(model.pair_p2,np.eye(3))
                 +.3*model.chi/model.scales[1]**2*np.kron(np.eye(3),model.pair_p2))
    return dict(onsite=onsite,hopping=hopping,
                static_reference_no_invented_onsite=True,reference_modes=4,
                auxiliary_hopping_loop=model.loop,field_dimension=Q.shape[1],oneparticle_dimension=4*law.volume,
                coexact_omega=model.coexact_omega,EM_coordinate_map=model.em_maps,
                EM_law='sum_m omega_m(a†a+1/2) + (UE/N) ptheta²; no harmonic oscillator potential',
                Coulomb_law='UE sum_xy (n_x-2+qref delta_xR) Green_xy (n_y-2+qref delta_yR), BEFORE square/projection',
                Coulomb_Green=model.coulomb_green,reference_site=model.reference_site,
                reference_charge='Nref-4 for BOTH physical channels; coherentSeaRef4/SeaPlus4positiveRef0 successor has same totalN2V+4',
                scalar_without_frozen_light_onsite=rest_scalar,
                native_Phi_colour_kinetic=phi_kinetic,native_heavy_matrix=model.heavy_matrix,
                native_auxiliary_Root=model.aux_penalty*(np.eye(2)-model.loop)/2+model.beta*model.defect,
                frozen_light_hopping_traces_NOT_added=True,
                scope='actual full108 light/reference bilinears and ALLmodeWeyl onshared9 quantumfield chart; NOTfull90V or a oneparticle replacement for shared-field manybody dynamics')


def field_matrix(factors:dict[str,Any],EM_coordinates:np.ndarray) -> np.ndarray:
    n=factors['oneparticle_dimension'];d=factors['field_dimension'];out=np.zeros((d,d,n,n),complex)
    for term in factors['onsite']:
        x=term['site'];out[:,:,4*x:4*x+4,4*x:4*x+4]+=term['block']
    for term in factors['hopping']:
        x,y=term['source'],term['target'];block=np.exp(1j*np.dot(term['word'],EM_coordinates))*term['block']
        out[:,:,4*x:4*x+4,4*y:4*y+4]+=block
        out[:,:,4*y:4*y+4,4*x:4*x+4]+=block.conj().swapaxes(0,1).swapaxes(2,3)
    return out


def native_full_current(model:Any) -> dict[str,Any]:
    law=model.law;V=law.volume;signature=model.operator_signature();n=4*V
    phi=np.tile(model.frame,(V,1,1,1));links=np.tile(np.eye(4,dtype=complex),(V,3,1,1))
    a=np.tile(np.array([.21,-.13,.09])/law.size,(V,1));g,w=coexact_plane_profile(law,(1,0,0),np.array([0,0,1.]))
    h,v=coexact_plane_profile(law,(0,1,0),np.array([0,0,1.]))
    raw=full_coordinate_light_operators(law,phi,links,a,g,h.conj())
    errors=[];eps=1e-4
    for tangent in (cast(Any,g).real,cast(Any,g).imag):
        jet=full_coordinate_light_operators(law,phi,links,a,tangent)
        plus=law.onebody(phi,links,a+eps*tangent);minus=law.onebody(phi,links,a-eps*tangent)
        errors.append([float(np.linalg.norm((plus-minus)/(2*eps)-jet['J'])),float(np.linalg.norm((plus-2*jet['H']+minus)/eps**2-jet['C']))])
    conj=full_coordinate_light_operators(law,phi,links,a,g.conj(),h)
    complex_adjoint=float(np.linalg.norm(raw['J'].conj().T-conj['J']))
    spectral=float(np.linalg.norm((model.curl.T@model.curl)@g.reshape(-1)-(w/float(C))**2*g.reshape(-1)))
    profile_coeff=model.coexact_basis@g.reshape(-1);profile_norm=float(np.linalg.norm(profile_coeff))
    mixed_pol=np.array([1.,1j,.3],complex);mixed_copy=mixed_pol.copy()
    mixed,mixed_w=coexact_plane_profile(law,(1,1,0),mixed_pol)
    mixed_spectral=float(np.linalg.norm((model.curl.T@model.curl)@mixed.reshape(-1)-(mixed_w/float(C))**2*mixed.reshape(-1)))
    if mixed_spectral>1e-12 or np.linalg.norm(mixed_pol-mixed_copy)!=0:
        raise AssertionError('General discrete coexact projection or input mutation')
    P=np.kron(model.orbitals@model.orbitals.conj().T,np.eye(4))
    outside_current=float(np.linalg.norm(raw['J']-P@raw['J']@P))
    if max(x[0] for x in errors)>1e-7 or max(x[1] for x in errors)>2e-6 or complex_adjoint>1e-12 or spectral>1e-12:
        raise AssertionError('Actual FULL108/current/contact/physical leg spectral relation failed')
    factors=full_light_field_factorization(model);coords=np.zeros(55);coords[0]=.13;coords[52:]=[.21,-.13,.09]
    actual=field_matrix(factors,coords);samples=[]
    for ip,ic in model.pc_points:
        Phi,U=shared_coordinate_fields(model,ip,ic);A=np.einsum('m,mxe->xe',coords,model.em_maps)
        samples.append(law.onebody(Phi,U,A))
    primitive=np.einsum('qa,qij,qb->abij',model.Q2,np.asarray(samples),model.Q2)
    factor_error=float(np.linalg.norm(actual-primitive))
    if factor_error>1e-11:raise AssertionError('Actual untruncated all55 Weyl factorization mismatch')
    # A bona fide fullCAR norm bound on omitted quantum-field feedback. It is
    # NOT silently calledsmall, and the source108 state is not evolved byold8H.
    mean=law.onebody(phi,links,np.einsum('m,mxe->xe',coords,model.em_maps))
    delta=actual-np.einsum('ab,ij->abij',np.eye(9),mean)
    feedback=55*sum(float(np.linalg.norm(delta[p,q],2)) for p in range(9) for q in range(9))
    return dict(owner_signature=signature,owner_signature_unchanged=model.operator_signature()==signature,state_reprepared=False,
        full_oneparticle_dimension=n,field_dimension=9,coexact_modes=52,harmonic_coordinates=3,
        actual_Hermitian_current_contact_difference_errors=errors,complex_current_adjoint_error=complex_adjoint,
        actual_coexact_spectral_relation_error=spectral,hard_mode_omega=w,hard_profile_coeff_norm=profile_norm,
        full_current_norm=float(np.linalg.norm(raw['J'])),outside_old8_current_norm=outside_current,
        actual_fullPhi_all55_Weyl_factorization_error=factor_error,
        actual_mixed_complex_coexact_spectral_error=mixed_spectral,
        native_rest_dimensions=dict(Phi_colour=9,heavy=54,auxiliary_Root=2,reference=4),
        static_reference_no_invented_onsite_mass=factors['static_reference_no_invented_onsite'],
        frozen_light_scalar_and_hopping_NOT_double_counted=True,
        fullCAR_quantum_field_feedback_operator_upper=feedback,T6_naive_frozen_field_shadow_norm_error_upper=min(2.,6*feedback),
        frozen_field_shadow_paid=False,full108_source_state_NOT_evolved_by_old8_H=True,
        input_embedding='same cached NPZ operator input; historical full54/55/53 light control is NOT the coherent58-count SAMEGauss successor supplied by coherent_bank API',
        leg_law='[H_EM,a_g†]=omega_g a_g†; [Hlight,a_g†]=sqrt(UE/omega_g) J_g, gnormalizedfullV; no double1/sqrtV',
        scope='actual fullcoordinateH/J/contact/all55Weyl and exactshared9 field factorization; not fullFockT6 evolution, legalMinSet or Born probability')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state-capsule',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent
    loaded={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest()
            for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
            and Path(cast(str,m.__file__)).parent==root and Path(cast(str,m.__file__)).suffix=='.py'}
    model=PreparedSpatialGaussQuantumSlice(state_capsule=a.state_capsule)
    names=('prepared_neutral_full_onebody.py','prepared_neutral_spatial_gauss_well.py','prepared_neutral_spatial_em_well.py','prepared_guard_colour_chart.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    result:dict[str,Any]=dict(native=native_full_current(model),source_sha256=hashes,input_state_sha256=hashlib.sha256(a.state_capsule.read_bytes()).hexdigest())
    result['loaded_scientific_source_sha256']=loaded
    result['loaded_sources_unchanged']=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==h for name,h in loaded.items())
    result['source_unchanged']=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==h for name,h in hashes.items())
    print(json.dumps(portable(result),indent=2,sort_keys=True))

if __name__=='__main__':main()
