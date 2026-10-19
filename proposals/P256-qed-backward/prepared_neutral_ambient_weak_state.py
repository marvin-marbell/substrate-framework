#!/usr/bin/env python3
"""Full-normal/Haar native weak H and SOURCE-created functions.

The native finite quadrature/Galerkin state is not the legal full-H Xi.  The
same original coefficients, fixed sea/R4 banks and unshifted C=0 Hamiltonian
are used. Root is explicitly compressed to a fixed flat-code chart; beta Q,
particle-hole/multipair and quadrature complements are not declared closed.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'

import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve

from improved_phi_band import ImprovedPhiLaw, selected_band
from original_charged_principal import portable
from prepared_neutral_ambient_metric import AmbientFieldChart
from prepared_neutral_coherent_bank import IDENTITY
from prepared_neutral_slater_fibre import FullSlaterFibre, lattice_green, slater_expectation
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE, local_potential

PLAN='issue-222-fullfield-closedrotation-continuation-20261006'
REVISION='e1e9ca716d3015f1a91e5e143b5d564ce8d188f2c074811a890c91d3d12fe5d6'
ROOT_AUTHORITY='ed25add3f7d678d8bde74a104f2b5e850577e172aaf00ae2cc2bb8ae424e6ae4'


def neutral_selector(values: np.ndarray) -> np.ndarray:
    """Smooth compact preparation: support[-1,2], exact plateau[0,1]."""
    out=np.zeros_like(values,dtype=float)
    for index,x in np.ndenumerate(values):
        if 0<=x<=1:out[index]=1.
        elif -1<x<0 or 1<x<2:
            t=x+1 if x<0 else 2-x
            a=math.exp(-1/t);b=math.exp(-1/(1-t));out[index]=a/(a+b)
    return out


def prepare_native_galerkin(H: np.ndarray,Gram: np.ndarray,raw: np.ndarray) -> dict[str,Any]:
    """ONE actual spectral filter on the shared native H; not legal e_global/L."""
    energies,vectors=eigh(H,Gram)
    width=max(1.,float(np.ptp(energies)))
    floor=float(energies[0])-width/8;window=3*width/4
    selector=neutral_selector((energies-floor)/window)
    coefficients=vectors@(selector*(vectors.conj().T@Gram@raw))
    norm=float(np.vdot(coefficients,Gram@coefficients).real)
    if norm<=0:raise AssertionError('Actual native global neutral filter vanished')
    return dict(coefficients=coefficients/math.sqrt(norm),unnormalized_norm_squared=norm,
        eigenvalues=energies,selector_values=selector,filter_floor=floor,filter_window=window,
        bank_weights=[float(np.vdot(coefficients[:7],Gram[:7,:7]@coefficients[:7]).real/norm),
                     float(np.vdot(coefficients[7:],Gram[7:,7:]@coefficients[7:]).real/norm)],
        no_charged_filter=True,one_global_normalization=True,
        scope='native ONE-filter Galerkin analogue with explicit spectrum-defined diagnostic floor/window; NEVER substituted for legal e_global/L/Xi')


def coexact_coordinates(law: Any, ue: float, c: float) -> dict[str,Any]:
    """All 2(V-1) real coexact modes; the three harmonics are separate."""
    v=law.volume;gradient=np.zeros((3*v,v))
    for x in range(v):
        for j in range(3):
            gradient[3*x+j,int(law.target[x,j])]+=1
            gradient[3*x+j,x]-=1
    lap=gradient.T@gradient
    values,modes=eigh(lap);positive=values>1e-10
    longitudinal=(gradient@modes[:,positive])/np.sqrt(values[positive])
    harmonic=np.tile(np.eye(3),(v,1))/math.sqrt(v)
    projector=np.eye(3*v)-longitudinal@longitudinal.T-harmonic@harmonic.T
    scalar=np.kron(lap,np.eye(3));full=projector@scalar@projector
    lam,vec=eigh(full);keep=lam>1e-9
    if keep.sum()!=2*(v-1):raise AssertionError('Incomplete coexact mode bank')
    omega=c*np.sqrt(lam[keep]);vectors=vec[:,keep]
    legs=vectors*np.sqrt(2*ue/omega)[None,:]
    return dict(omega=omega,vectors=vectors,legs=legs,harmonic=harmonic,
                dimension=len(omega)+3,coexact_projector=projector)


def compact_radial(q: np.ndarray, radius: float) -> tuple[float,np.ndarray]:
    """Literal smooth cutoff; no unused coordinate is frozen."""
    r2=float(q@q)/radius**2
    if r2<=.25:return 1.,np.zeros_like(q)
    if r2>=1:return 0.,np.zeros_like(q)
    a=r2-.25;b=1-r2
    x=math.exp(-1/a);y=math.exp(-1/b);value=y/(x+y)
    derivative=(-y*x/(x+y)**2)*(1/a**2+1/b**2)
    return value,derivative*2*q/radius**2


class AmbientWeakState:
    """A shared, full-measure basis and original positive-form H interface.

    The Gaussian precision is an explicit native variational input, not a
    substituted Riccati covariance or a claim of the actual ground state.
    Basis columns are field polynomials times each fixed neutral bank. The
    charged columns are their ACTUAL B(q,theta) images, not polynomial fits.
    """
    def __init__(self, law: Any, fields: np.ndarray, links: np.ndarray,
                 reference_site: int, profile: np.ndarray, root_lemma: dict[str,Any],
                 Lambda: float=2.**24, ue: float=2.**-20, c: float=.125,
                 sigma: float=2.**-13, radius: float=.012,
                 directions: np.ndarray | None=None):
        if root_lemma['authority']['sha256']!=ROOT_AUTHORITY:
            raise ValueError('Immutable original106 authority required')
        if root_lemma['periodic_identity']!='each beta spatialedge anticommutes a contractible face-cycle; P Z_e P=0 and P Vbeta P=0':
            raise ValueError('Actual flat-P beta compression identity required')
        self.law=law;self.chart=AmbientFieldChart(law,fields,links)
        self.fibre=FullSlaterFibre(law,fields,links,reference_site,profile)
        self.Lambda=float(Lambda);self.chi=Lambda**-.5;self.ue=float(ue);self.c=float(c)
        self.K=c*c/(2*ue);self.K4=.7;self.sigma=float(sigma);self.radius=float(radius)
        self.reference_site=int(reference_site);self.root_lemma=root_lemma
        self.em=coexact_coordinates(law,ue,c);self.green=lattice_green(law)
        self.HC={s:self.fibre.coulomb(self.green,ue,s) for s in ('neutral','charged')}
        self.heavy_law=ImprovedPhiLaw(law.size,RotationBand(**BARE))
        self.seed_fibre=self.fibre
        h0=self.massive_onebody(fields,links);values,modes=eigh(h0)
        self.heavy_orbitals=modes[:,values>0]
        self.heavy_projector=self.heavy_orbitals@self.heavy_orbitals.conj().T
        n=self.chart.normal_dim
        if directions is None:
            candidates=np.eye(n,6)
        else:
            candidates=np.asarray(directions,float)
            if candidates.shape!=(n,6):raise ValueError('Six FULL-normal directions required')
        self.directions=np.linalg.qr(candidates,mode='reduced')[0]
        if np.linalg.matrix_rank(candidates)!=6:raise ValueError('Directions are dependent')
        self.polynomial_dimension=7;self.dimension=14
        self.edges=[(x,int(law.target[x,j])) for x in range(law.volume) for j in range(3)]
        self.reference_frame=np.eye(4,dtype=complex)

    def global_centre_control(self) -> dict[str,Any]:
        """Actual heavy+light+Ref nality, with the flat-P onsite representation."""
        count=self.heavy_orbitals.shape[1];centre=1j*np.eye(self.fibre.n)
        heavy_phase=np.linalg.det(self.heavy_orbitals.conj().T@centre@self.heavy_orbitals)
        errors={}
        for sector in ('neutral','charged'):
            number=self.fibre.onebody(np.eye(self.fibre.n),sector)+self.fibre.reference(np.eye(4),sector)
            phase=np.exp(.5j*math.pi*np.diag(number))*heavy_phase
            errors[sector]=float(np.max(abs(phase-1)))
        moved=self.in_mean_interaction_frame(centre,1j*np.eye(4))
        a=np.zeros((self.law.volume,3));theta=np.zeros(3)
        source=self.source_at(self.fibre.mean_fields,self.fibre.mean_links,theta,a)
        transformed=moved.source_at(self.fibre.mean_fields,self.fibre.mean_links,theta,a)
        source_error=float(np.linalg.norm(source['source_matrix']-transformed['source_matrix']))
        total=count+2*self.law.volume+4
        if total%4 or max(errors.values())>1e-10 or source_error>1e-10:
            raise AssertionError('Global SU4 centre obstructs the actual native state')
        return dict(heavy_count=count,light_plus_reference_count=2*self.law.volume+4,
            actual_total_count=total,total_nality_mod4=total%4,
            heavy_centre_phase=heavy_phase,light_reference_centre_phase=-1 if (2*self.law.volume+4)%4==2 else 1,
            actual_fibre_phase_errors=errors,actual_SOURCE_centre_covariance_error=source_error,
            Root_representation='flat-P onsite isometry: T=diag(pi/2,pi/2,pi/2,-3pi/2), exp(i dGamma(T))=i^N since n4=(1-B4)/2 is a projector; holonomy labels add no nality',
            Root_full_aux_centre_NOT_claimed_scalar=True,
            legal_total_count='4V+4, divisible4 for every integer V',
            scope='actual selected heavy/light/Ref flat-P representation; necessary global-stabilizer consistency, not alone full group-averaging proof')

    def in_mean_interaction_frame(self,light: np.ndarray,reference: np.ndarray) -> AmbientWeakState:
        """Transport the ONCE prepared orbitals; no h(q) ground replacement.

        B supplies the actual full108/Ref4 mean unitaries and subtracts their
        time connection in its weak generator. All field derivatives remain
        unchanged because these mean unitaries are coordinate independent.
        """
        light=np.asarray(light,complex);reference=np.asarray(reference,complex)
        if light.shape!=(self.fibre.n,self.fibre.n) or reference.shape!=(4,4):
            raise ValueError('Actual full108 and Ref4 mean unitaries required')
        if np.linalg.norm(light.conj().T@light-np.eye(self.fibre.n))>1e-9 or np.linalg.norm(reference.conj().T@reference-np.eye(4))>1e-9:
            raise ValueError('Mean frame is not unitary')
        if abs(np.linalg.det(reference)-1)>1e-9:
            raise ValueError('Anchored traceless SU4 reference frame required')
        moved=copy(self);moved.fibre=copy(self.fibre)
        moved.fibre.orbitals=light@self.fibre.orbitals
        moved.fibre.empty=light@self.fibre.empty
        moved.fibre.occupied={q:light@u for q,u in self.fibre.occupied.items()}
        moved.fibre.projectors={q:light@p@light.conj().T for q,p in self.fibre.projectors.items()}
        moved.reference_frame=reference@self.reference_frame
        moved.HC={s:moved.fibre.coulomb(self.green,self.ue,s) for s in ('neutral','charged')}
        return moved

    def massive_onebody(self,fields: np.ndarray,links: np.ndarray) -> np.ndarray:
        """Original bare h_F(Phi,Ucolour), EM-neutral; never borrowed h_light."""
        return self.heavy_law.onebody(fields,links,np.zeros((self.law.volume,3)))

    def microscopic_operators(self,fields: np.ndarray,links: np.ndarray,a: np.ndarray) -> dict[str,np.ndarray]:
        return dict(massive=self.massive_onebody(fields,links),
                    light=self.law.onebody(fields,links,a))

    def current_binding_control(self) -> dict[str,Any]:
        fields,links=self.fibre.mean_fields,self.fibre.mean_links
        direction=self.em['vectors'][:,0].reshape(self.law.volume,3);eps=2.**-18
        plus=self.microscopic_operators(fields,links,eps*direction)
        minus=self.microscopic_operators(fields,links,-eps*direction)
        currents={key:(plus[key]-minus[key])/(2*eps) for key in plus}
        return dict(heavy_bare_parameters=BARE,heavy_U1_charge=0,
            massive_EM_current_norm=float(np.linalg.norm(currents['massive'])),
            light_EM_current_norm=float(np.linalg.norm(currents['light'])),
            bare_heavy_vs_improved_light_onebody_defect=float(np.linalg.norm(plus['massive']-self.fibre.h0)),
            SU4_charge='massive+light+Ref fundamental; Root flat-code onsite representation',
            electromagnetic_density='light n_x-2 plus SAME Ref (Nref-4); massive contributes0',
            C0_unshifted=True,heavy_holonomy_kappa_NOT_EM_theta=True)

    def basis_source_at(self,fields: np.ndarray,links: np.ndarray,theta: np.ndarray,a: np.ndarray) -> dict[str,Any]:
        """Fixed INITIAL SOURCE functions, including all their field dependence."""
        return self.seed_fibre.source_vectors_at(fields,links,theta,a)

    def current_source_function_residual(self,sample: dict[str,Any],coefficients: np.ndarray) -> dict[str,Any]:
        physical=self.source_at(sample['point']['phi'],sample['point']['links'],sample['theta'],sample['a'])['source_matrix']@sample['neutral']
        target=sample['charged']
        return dict(full448_physical_SOURCE=physical@coefficients,
            transported_initial_SOURCE=target@coefficients,
            full448_difference_norm=float(np.linalg.norm((physical-target)@coefficients)),
            SOURCE_basis_NOT_recaptured=True,
            scope='full448 diagnostic; finite field-function projection debt must be paid separately')

    def source_at(self,fields: np.ndarray,links: np.ndarray,theta: np.ndarray,a: np.ndarray) -> dict[str,Any]:
        source=self.fibre.source_vectors_at(fields,links,theta,a)
        marks=source['full_SOURCE_marks']
        plus=marks[1].reshape(self.fibre.n,4)@self.reference_frame
        minus=marks[-1].transpose(1,0,2).reshape(4,self.fibre.n)
        minus=self.reference_frame.conj().T@minus
        return self.fibre.source_vectors_from_marks({1:plus.reshape(self.law.volume,4,4),
            -1:minus.reshape(4,self.law.volume,4).transpose(1,0,2)})

    def reference_operator(self,reference: np.ndarray,sector: str) -> np.ndarray:
        return self.fibre.reference(self.reference_frame.conj().T@reference@self.reference_frame,sector)

    def electromagnetic_field(self, em: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        em=np.asarray(em,float);n=len(self.em['omega'])
        if em.shape!=(n+3,):raise ValueError('ALL coexact plus harmonic coordinates required')
        theta=em[n:];a=(self.em['legs']@em[:n]).reshape(self.law.volume,3)+theta/self.law.size
        return a,theta

    def polynomial(self,q: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        values=np.r_[1.,math.sqrt(2)*self.directions.T@q/self.sigma]
        derivative=np.column_stack((np.zeros(self.chart.normal_dim),math.sqrt(2)*self.directions/self.sigma))
        return values,derivative

    def basis_at(self,q: np.ndarray,em: np.ndarray,point: dict[str,Any] | None=None) -> dict[str,Any]:
        q=np.asarray(q,float)
        if q.shape!=(self.chart.normal_dim,):raise ValueError('No six-coordinate shortcut')
        point=self.chart.evaluate(q) if point is None else point
        a,theta=self.electromagnetic_field(em);p,dp=self.polynomial(q)
        neutral=np.zeros((2,self.dimension),complex);neutral[0,:7]=p;neutral[1,7:]=p
        source=self.basis_source_at(point['phi'],point['links'],theta,a)
        # source_matrix contains the literal operator, without a point normalizer.
        charged=source['source_matrix']@neutral
        cutoff,dcutoff=compact_radial(q,self.radius)
        return dict(q=q,em=np.asarray(em,float),point=point,a=a,theta=theta,
                    polynomial=p,polynomial_derivative=dp,neutral=neutral,charged=charged,
                    source=source,cutoff=cutoff,cutoff_gradient=dcutoff,
                    importance_weight=cutoff**2,
                    field_measure='J dq Haar, wave has J^(-1/2); ONE density cancellation',
                    gaussian_all_normal_coordinates=self.chart.normal_dim)

    def directional_basis(self,sample: dict[str,Any],dq: np.ndarray,
                          dem: np.ndarray | None=None,step: float=2.**-19) -> dict[str,np.ndarray]:
        """Physical wave derivatives divided by the common Gaussian/Jhalf factor.

        Haar logarithmic derivative is exact. Literal SOURCE directional
        derivative is central FD; its finite-difference remainder is a named
        native debt, not an exact canonical q2 proof.
        """
        dq=np.asarray(dq,float);dem=np.zeros(self.em['dimension']) if dem is None else np.asarray(dem,float)
        q,em,point=sample['q'],sample['em'],sample['point']
        log_density_derivative=self.chart.directional_jet(q,dq,point)['log_density_first'] if np.any(dq) else 0.
        logder=-float(q@dq)/self.sigma**2-float(em@dem)-.5*log_density_derivative
        cutoff=sample['cutoff']
        if cutoff==0:raise ValueError('Derivative requested outside compact support')
        logder+=float(sample['cutoff_gradient']@dq)/cutoff
        dp=sample['polynomial_derivative'].T@dq
        nd=np.zeros_like(sample['neutral']);nd[0,:7]=dp;nd[1,7:]=dp
        nd+=logder*sample['neutral']
        scale=max(1.,float(np.linalg.norm(dq)),float(np.linalg.norm(dem)));eps=step/scale
        def source_at(sign: float) -> np.ndarray:
            qq=q+sign*eps*dq;ee=em+sign*eps*dem
            pp=self.chart.embedding(qq);aa,tt=self.electromagnetic_field(ee)
            return self.basis_source_at(pp['phi'],pp['links'],tt,aa)['source_matrix']
        dsource=(source_at(1)-source_at(-1))/(2*eps)
        cd=dsource@sample['neutral']+sample['source']['source_matrix']@nd
        return dict(neutral=nd,charged=cd)

    def gauge_operators(self,coefficients: np.ndarray,sector: str) -> dict[str,Any]:
        coeff=np.asarray(coefficients,float).reshape(self.law.volume,15)
        matrices=np.einsum('xa,aij->xij',coeff,self.law.generators)
        body=np.zeros((4*self.law.volume,4*self.law.volume),complex)
        for x,m in enumerate(matrices):body[4*x:4*x+4,4*x:4*x+4]=m
        ref=self.reference_frame.conj().T@matrices[self.reference_site]@self.reference_frame
        q=self.fibre.onebody(body,sector)+self.fibre.reference(ref,sector)
        q2=self.fibre.bilinear_product(body,body,sector,ref,ref)
        heavy=slater_expectation(self.heavy_projector,body).real
        heavy2=slater_expectation(self.heavy_projector,body,body).real
        identity=self.fibre.overlap(sector)
        return dict(first=q+heavy*identity,
                    second=q2+2*heavy*q+heavy2*identity,
                    outside_second=q2-q@q+(heavy2-heavy**2)*identity,
                    body=body,reference=ref)

    def first_order_weak(self,sample: dict[str,Any],dq: np.ndarray,dem: np.ndarray,
                         gauge_velocity: np.ndarray,body: np.ndarray,
                         reference: np.ndarray,sector: str='charged',
                         heavy_body: np.ndarray | None=None) -> np.ndarray:
        """B-owned Noether integration hook: literal symmetric weak transport.

        Pass the actual pullback Y and body/Ref ONCE. This method does not
        manufacture Noether coefficients or replace G by a spin matrix.
        """
        value=sample[sector];d=self.directional_basis(sample,dq,dem)[sector]
        gauge=self.gauge_operators(gauge_velocity,sector)['first']
        d=d+1j*gauge@value
        matter=self.fibre.onebody(body,sector)+self.reference_operator(reference,sector)
        if heavy_body is not None:
            matter+=slater_expectation(self.heavy_projector,heavy_body)*self.fibre.overlap(sector)
        return sample['importance_weight']*(-.5j*(value.conj().T@d-d.conj().T@value)+value.conj().T@matter@value)

    def classical_potential(self,point: dict[str,Any]) -> dict[str,float]:
        phi,links=point['phi'],point['links'];law=self.law
        elastic,alignment=local_potential(phi,self.edges,[links[x,j] for x in range(law.volume) for j in range(3)])
        magnetic=0.
        for x in range(law.volume):
            for j in range(3):
                for k in range(j+1,3):
                    y,z=int(law.target[x,j]),int(law.target[x,k])
                    loop=links[x,j]@links[y,k]@links[z,j].conj().T@links[x,k].conj().T
                    magnetic+=self.K4/2*float(np.linalg.norm(loop-np.eye(4))**2)
        heavy_h=self.massive_onebody(phi,links)
        massive=-self.Lambda*slater_expectation(self.heavy_projector,heavy_h).real
        return dict(quartic=self.Lambda*float(elastic),
                    alignment=self.Lambda**1.25*float(alignment),
                    colour_magnetic=self.Lambda**1.25*magnetic,massive_sea=massive)

    def positive_direction(self,sample: dict[str,Any],direction: np.ndarray) -> dict[str,np.ndarray]:
        """Unbiased full-cometric kinetic sample, Wick BEFORE compression."""
        n=self.chart.normal_dim;dq=direction[:n];gauge=direction[n:]
        derivatives=self.directional_basis(sample,dq)
        forms={}
        for sector in ('neutral','charged'):
            value=sample[sector];d=derivatives[sector];op=self.gauge_operators(gauge,sector)
            momentum=-1j*d+op['first']@value
            # The first weak matrix misses particle-hole/multipair directions;
            # the exact preprojection second product puts their norm BACK.
            form=momentum.conj().T@momentum+value.conj().T@op['outside_second']@value
            forms[sector]=self.chi/2*sample['importance_weight']*(form+form.conj().T)/2
        return forms

    def compile(self,nodes: int=14,kinetic_directions: int=4,seed: int=222) -> dict[str,Any]:
        if nodes<14 or nodes%2:raise ValueError('Seven field polynomials require at least seven antithetic pairs')
        rng=np.random.default_rng(seed);n=self.chart.normal_dim
        grams={s:np.zeros((self.dimension,self.dimension),complex) for s in ('neutral','charged')}
        hamiltonians={s:np.zeros_like(grams[s]) for s in grams};records=[];kinetic_variances=[]
        qnodes=rng.standard_normal((nodes//2,n))*self.sigma/math.sqrt(2)
        enodes=rng.standard_normal((nodes//2,self.em['dimension']))/math.sqrt(2)
        for index,(q0,e0) in enumerate(zip(qnodes,enodes)):
            for sign in (-1,1):
                q,em=sign*q0,sign*e0;sample=self.basis_at(q,em);point=sample['point']
                if sample['cutoff']==0:raise AssertionError('Native quadrature missed support')
                scalar=self.classical_potential(point);total_scalar=sum(scalar.values())
                h=self.law.onebody(point['phi'],point['links'],sample['a'])
                # Full coexact vacuum positive kinetic+potential, not another
                # separately added zero point. Hharm has kinetic ONLY.
                coe=float(np.sum(self.em['omega']*em[:len(self.em['omega'])]**2))
                for sector in grams:
                    value=sample[sector];gram=sample['importance_weight']*value.conj().T@value
                    grams[sector]+=gram/nodes
                    weak=self.fibre.onebody(h,sector)+self.HC[sector]
                    hamiltonians[sector]+=(sample['importance_weight']*value.conj().T@weak@value+(total_scalar+coe)*gram)/nodes
                factor=lu_factor(np.sqrt(self.chart.mass)[:,None]*point['jacobian'])
                forms=[]
                for _ in range(kinetic_directions):
                    white=rng.standard_normal(self.chart.raw_dim)
                    direction=lu_solve(factor,white)
                    forms.append(self.positive_direction(sample,direction))
                for sector in grams:
                    hamiltonians[sector]+=sum(f[sector] for f in forms)/(nodes*kinetic_directions)
                    kinetic_variances.append(float(np.std([np.trace(f[sector]).real for f in forms])))
                # Harmonic ptheta differentiates SOURCE itself and Gaussian.
                for j in range(3):
                    dem=np.zeros(self.em['dimension']);dem[-3+j]=1
                    derivative=self.directional_basis(sample,np.zeros(n),dem)
                    for sector in grams:
                        d=derivative[sector]
                        hamiltonians[sector]+=self.ue/self.law.size*sample['importance_weight']*(d.conj().T@d)/nodes
                records.append(dict(node=index,sign=sign,normal_norm=float(np.linalg.norm(q)),
                    nonzero_normal_coordinates=int(np.count_nonzero(q)),
                    coexact_norm=float(np.linalg.norm(em[:-3])),theta=sample['theta'],
                    full_density=point['relative_density'],normal_gauge_cross_norm=float(np.linalg.norm(point['normal_gauge_metric'])),
                    SOURCE_function_norm=float(np.linalg.norm(sample['charged'])),potential=scalar))
        raw=np.zeros(self.dimension,complex);raw[0]=raw[7]=1/math.sqrt(2)
        for sector in grams:
            hamiltonians[sector]=(hamiltonians[sector]+hamiltonians[sector].conj().T)/2
            if np.min(eigh(grams[sector],eigvals_only=True))<=1e-10:raise AssertionError('Dependent sampled weak basis')
        prepared=prepare_native_galerkin(hamiltonians['neutral'],grams['neutral'],raw)
        neutral=prepared['coefficients'];norm=prepared['unnormalized_norm_squared']
        source_norm=float(np.vdot(neutral,grams['charged']@neutral).real)
        if source_norm<=0:raise AssertionError('Actual SOURCE-function norm vanished')
        return dict(Gram=grams,H=hamiltonians,raw_neutral_coefficients=raw,
                    normalized_neutral_coefficients=neutral,actual_neutral_norm_squared=norm,
                    once_prepared_native_filter=prepared,global_centre_control=self.global_centre_control(),
                    microscopic_current_binding=self.current_binding_control(),
                    actual_SOURCE_norm_squared=source_norm,
                    normalized_SOURCE_coefficients=neutral/math.sqrt(source_norm),
                    SOURCE_map_in_created_function_basis=np.eye(self.dimension),
                    SOURCE_capture_scope='EXACT FUNCTIONS B(q,theta)psi_i(q); identity labels are definition, not pointwise polynomial capture',
                    nodes=records,kinetic_trace_sample_standard_deviations=kinetic_variances,
                    quadrature=dict(nodes=nodes,kinetic_directions=kinetic_directions,seed=seed,
                        positive_full_dimension_importance_sampling=True,normal_dimension=n,
                        electromagnetic_dimension=self.em['dimension'],normal_covariance=self.sigma**2/2,
                        finite_quadrature_and_stochastic_trace_debt=True),
                    Root_projection=dict(authority=ROOT_AUTHORITY,P_beta_P=0,beta_total=1/16,
                        unrepresented_Root_Q=True,all8_s_in_legal_model=True,native_fixed_s=0),
                    unshifted_H=True,C=0,no_field_or_perbank_SOURCE_whitening=True,
                    scope='native projected variational ONE-filter state; not actual legal Xi, full Fock, exact quadrature, full auxiliary Gauss-kernel evolution, ground state or closed physical rotation')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-lemma',type=Path,required=True)
    parser.add_argument('--output-npz',type=Path,required=True)
    parser.add_argument('--nodes',type=int,default=14);parser.add_argument('--kinetic-directions',type=int,default=4)
    args=parser.parse_args();started=time.perf_counter()
    source_root=Path(__file__).resolve().parent
    source_bytes={Path(cast(str,m.__file__)).name:Path(cast(str,m.__file__)).read_bytes() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(cast(str,m.__file__)).parent==source_root and Path(cast(str,m.__file__)).suffix=='.py'}
    band,frame,_=selected_band();law=ImprovedPhiLaw(3,band)
    fields=np.tile(frame,(law.volume,1,1,1));links=np.tile(np.eye(4,dtype=complex),(law.volume,3,1,1))
    profile=np.exp(-np.sum((law.sites-1)**2,axis=1)/(.65**2));profile=profile/np.linalg.norm(profile)
    root_lemma=json.loads(args.root_lemma.read_bytes())
    model=AmbientWeakState(law,fields,links,law.index[(0,1,1)],profile,root_lemma)
    result=model.compile(args.nodes,args.kinetic_directions)
    np.savez_compressed(args.output_npz,neutral_Gram=result['Gram']['neutral'],charged_Gram=result['Gram']['charged'],
        neutral_H=result['H']['neutral'],charged_H=result['H']['charged'],
        neutral_state=result['normalized_neutral_coefficients'],source_state=result['normalized_SOURCE_coefficients'])
    result['source_sha256']={name:hashlib.sha256(body).hexdigest() for name,body in sorted(source_bytes.items())}
    result['loaded_sources_unchanged']=all((source_root/name).read_bytes()==body for name,body in source_bytes.items())
    result['output_npz_sha256']=hashlib.sha256(args.output_npz.read_bytes()).hexdigest()
    result['root_lemma_sha256']=hashlib.sha256(args.root_lemma.read_bytes()).hexdigest()
    result['plan']=PLAN;result['revision']=REVISION;result['identity']=IDENTITY
    result['wall_seconds']=time.perf_counter()-started
    print(json.dumps(portable(result),sort_keys=True));


if __name__=='__main__':main()
