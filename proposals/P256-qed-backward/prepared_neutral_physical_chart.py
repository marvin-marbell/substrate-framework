#!/usr/bin/env python3
"""Actual joint-Gauss Noether integral chart with the entire EM coordinate space.

This supplies physical coordinate/operator coefficients, not a flat kinetic
replacement or a completed finite quantum closed-loop propagator.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
from typing import Any, cast
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.integrate import solve_ivp
from prepared_neutral_full_onebody import shared_coordinate_fields,full_coordinate_light_operators
from prepared_neutral_coherent_bank import canonical_bank_coordinate_marks,IDENTITY
from prepared_neutral_full_field import BETA
from soft_spatial_closed_rotation import native_joint_noether_vector
from original_charged_principal import portable


class PhysicalNoetherCoordinateChart:
    """One nonlinear Phi/U flow coordinate plus ALL coexact/harmonic coordinates.

    The Phi/U vector and its SU4 Gauss projection are independent of A.  The
    EM vector is retained as a vector in the entire52+3 coordinate space, not
    forced into the new Phi/U direction.  No tangent-plane extrapolation is used.
    """
    def __init__(self,model:Any,alpha:np.ndarray,velocity:np.ndarray,
                 ip:int=0,ic:int=0):
        self.model=model;self.law=model.law;self.alpha=np.asarray(alpha,float)
        self.velocity=np.asarray(velocity,float)
        if self.alpha.shape!=(self.law.volume,) or self.velocity.shape!=(self.law.volume,3):
            raise ValueError('Physical profile must specify every native lattice site')
        self.fields,self.links=shared_coordinate_fields(model,ip,ic)
        self.shapes=(self.fields.shape,self.links.shape)
        self.cut=self.fields.size
        self.initial=np.r_[self.fields.ravel(),self.links.ravel()]
        self.EM_map=np.asarray(model.em_maps).reshape(55,-1).T
        if np.linalg.matrix_rank(self.EM_map)!=55:
            raise ValueError('Native chart must retain ALL52coexact and3harmonic coordinates')
        self.EM_inverse=np.linalg.pinv(self.EM_map)
        self.solutions:dict[int,Any]={}
        self.coverage:dict[int,float]={}

    def unpack(self,value:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        return value[:self.cut].reshape(self.shapes[0]),value[self.cut:].reshape(self.shapes[1])

    def vector(self,fields:np.ndarray,links:np.ndarray,a:np.ndarray) -> dict[str,Any]:
        return native_joint_noether_vector(self.law,fields,links,a,self.alpha,
            self.velocity,self.model.reference_site)

    def tangent(self,info:dict[str,Any],links:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        coefficients=info['horizontal_vector']
        phi=np.einsum('xma,aij->xmij',coefficients[:,:60].reshape(self.law.volume,4,15),self.law.basis)
        generators=np.einsum('xja,aik->xjik',coefficients[:,60:].reshape(self.law.volume,3,15),self.law.generators)
        return phi,1j*np.einsum('xjab,xjbc->xjac',generators,links)

    def field_at(self,q:float) -> tuple[np.ndarray,np.ndarray]:
        if q==0:return self.fields.copy(),self.links.copy()
        sign=1 if q>0 else -1
        if sign not in self.solutions or abs(q)>self.coverage[sign]:
            def rhs(t:float,value:np.ndarray) -> np.ndarray:
                phi,links=self.unpack(value)
                info=self.vector(phi,links,np.zeros((self.law.volume,3)))
                dphi,dlinks=self.tangent(info,links)
                return np.r_[dphi.ravel(),dlinks.ravel()]
            solution=solve_ivp(rhs,(0,q),self.initial,method='DOP853',rtol=2e-10,
                atol=2e-12,dense_output=True)
            if not solution.success:raise RuntimeError(solution.message)
            self.solutions[sign]=solution;self.coverage[sign]=abs(q)
        return self.unpack(self.solutions[sign].sol(q))

    def at(self,q:float,EM_coordinates:np.ndarray) -> dict[str,Any]:
        em=np.asarray(EM_coordinates,float)
        if em.shape!=(55,):raise ValueError('ALL55 real EM coordinates required')
        fields,links=self.field_at(q);a=(self.EM_map@em).reshape(self.law.volume,3)
        info=self.vector(fields,links,a)
        em_vector=self.EM_inverse@info['horizontal_EM_vector'].ravel()
        em_remainder=info['horizontal_EM_vector'].ravel()-self.EM_map@em_vector
        metric=float(np.sum(info['horizontal_vector']**2*info['mass_diagonal']))
        marks=canonical_bank_coordinate_marks(self.law,fields,links,np.mean(a,axis=0)*self.law.size,
            self.model.reference_site,self.model.orbitals[:,0],BETA,a)
        return dict(identity=IDENTITY,fields=fields,links=links,EM=a,
            onebody=self.law.onebody(fields,links,a),full_SOURCE_marks=marks,
            full_current=full_coordinate_light_operators,Noether=info,
            Noether_EM_coordinates=em_vector,EM_tangent_remainder_norm=float(np.linalg.norm(em_remainder)),
            curve_mass_metric=metric,EM_modes=55,coexact_modes=52,harmonic_modes=3,
            physical_coordinate_scope='nonlinear full105 Phi/U integralcurve plus entire untruncated55EM; exact full108 operator coefficients BEFORE CAR clipping',
            kinetic_scope='original ambient Phi/colour kinetic and Haar/covariant symmetric momenta REQUIRED; curve metric is not a flat or completed Galerkin Hamiltonian')


def native_chart_control(model:Any) -> dict[str,Any]:
    started=time.perf_counter();law=model.law;V=law.volume
    alpha=np.ones(V);alpha[model.reference_site]=0
    wrapped=(law.sites-law.size//2+law.size//2)%law.size-law.size//2
    velocity=np.c_[-wrapped[:,1],wrapped[:,0],np.zeros(V)]
    zero_index=int(np.argmin(np.linalg.norm(model.points,axis=1)))
    chart=PhysicalNoetherCoordinateChart(model,alpha,velocity,zero_index,zero_index)
    em=np.linspace(-.17,.23,55);base=chart.at(0,em);eps=2e-4
    plus=chart.at(eps,em);minus=chart.at(-eps,em)
    dphi,dlinks=chart.tangent(base['Noether'],base['links'])
    first_error=float(np.linalg.norm((plus['fields']-minus['fields'])/(2*eps)-dphi)
        +np.linalg.norm((plus['links']-minus['links'])/(2*eps)-dlinks))
    vector_plus=plus['Noether']['horizontal_vector'];vector_minus=minus['Noether']['horizontal_vector']
    acceleration=(vector_plus-vector_minus)/(2*eps)
    curvature=float(np.linalg.norm(acceleration))
    unitary=float(np.linalg.norm(plus['links']@plus['links'].conj().swapaxes(-1,-2)-np.eye(4)))
    hermitian=float(np.linalg.norm(plus['fields']-plus['fields'].conj().swapaxes(-1,-2)))
    other=chart.at(0,-em)
    A_independence=float(np.linalg.norm(base['Noether']['horizontal_vector']-other['Noether']['horizontal_vector']))
    current=full_coordinate_light_operators(law,base['fields'],base['links'],base['EM'],
        base['Noether']['horizontal_EM_vector'])
    stress=float(np.linalg.norm(current['J']))
    result=dict(identity=IDENTITY,full_coordinates=105,EM_modes=55,oneparticle_dimension=4*V,
        Gauss_orthogonality=base['Noether']['gauss_orthogonality_defect'],
        EM_tangent_remainder=base['EM_tangent_remainder_norm'],
        nonlinear_curve_first_tangent_error=first_error,actual_projected_vector_acceleration_norm=curvature,
        false_stationary_tangent_adverse_error=curvature,curve_links_unitarity_error=unitary,
        curve_Phi_Hermiticity_error=hermitian,Phi_colour_vector_EM_independence_error=A_independence,
        full108_EM_Noether_current_norm=stress,curve_mass_metric=base['curve_mass_metric'],
        actual_reference_internal_norm=float(np.linalg.norm(base['Noether']['derived_Ref_internal'])),
        harmonic_EM_vector=base['Noether']['harmonic_EM_vector'],lambda_C_FAR=float(base['Noether']['lambda_C'][model.reference_site]),
        SOURCE_scope='full uncompressed108 canonical original maps; active12 input basis not reused for chart clipping',
        field_scope=base['physical_coordinate_scope'],kinetic_scope=base['kinetic_scope'],
        preparation_or_Hamiltonian_changed=False,spatial_closed_quantum_loop_proved=False,
        wall_seconds=time.perf_counter()-started)
    if first_error>1e-5 or unitary>1e-8 or hermitian>1e-8 or curvature<1e-3 or A_independence>1e-12 or base['EM_tangent_remainder_norm']>1e-10:
        raise AssertionError(result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--state-capsule',type=Path,required=True);args=parser.parse_args()
    from prepared_neutral_spatial_gauss_well import PreparedSpatialGaussQuantumSlice
    root=Path(__file__).resolve().parent
    loaded={Path(cast(str,m.__file__)).name:hashlib.sha256(Path(cast(str,m.__file__)).read_bytes()).hexdigest()
        for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(cast(str,m.__file__)).parent==root and Path(cast(str,m.__file__)).suffix=='.py'}
    model=PreparedSpatialGaussQuantumSlice(state_capsule=args.state_capsule)
    print(json.dumps(portable(dict(native=native_chart_control(model),loaded_source_sha256=loaded,
        input_state_sha256=hashlib.sha256(args.state_capsule.read_bytes()).hexdigest(),
        loaded_sources_unchanged=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in loaded.items()))),indent=2,sort_keys=True))
