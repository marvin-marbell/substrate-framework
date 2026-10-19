#!/usr/bin/env python3
"""Original primitive-quaternion/per-edge SOURCE on one cached native state.

This changes SOURCE, not the historical Hamiltonian/state. It is not MinSet,
full Haar, a physical rotation, or a legal-N scattering proof.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, cast
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator
from noncompact_screened_gauss_dressing import screened_marks
from improved_physical_compton import decode
from original_charged_principal import portable
from prepared_neutral_full_field import BETA,require
from prepared_neutral_spatial_charge_transfer import PreparedSpatialChargeTransfer,GAMMA5
from prepared_neutral_spatial_gauss_well import PreparedSpatialGaussQuantumSlice
from prepared_neutral_spatial_em_well import gaussian_weyl
from sea_selection import SELECTED

REGIME='original-screened-peredge-signed-path-spatial-quaternion-raw-global2'
LABELS=np.array([np.eye(4),*(SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2)))])/2


def canonical_coordinate_marks(model,quartet):
    """Consume the immutable original producer, including all interior frames."""
    rows=[]
    for ip,ic in model.pc_points:
        p=model.points[ip]*model.scales[0];c=model.points[ic]*model.scales[1]
        phi=np.tile(model.frame,(model.law.volume,1,1,1)).copy()
        phi[:,0]+=p[0]*SELECTED[1]+p[1]*SELECTED[2]
        links=np.tile(np.eye(4,dtype=complex),(model.law.volume,3,1,1))
        links[:,2]=expm(1j*(c[0]*SELECTED[1]+c[1]*SELECTED[2])/math.sqrt(8))
        mark=model.source_coordinate_mark(phi,links,np.zeros(3),quartet=quartet)
        rows.append(mark)
    return np.asarray(rows)


def literal_car_transfers(model,charge):
    """Full four reference holes/particles; exact12-mode CAR, no spin tracing."""
    incoming=model.bulk_masks[4]
    ref4=sum(1<<r for r in range(8,12))
    out=([m|(ref4^(1<<r)) for r in range(8,12) for m in model.bulk_masks[5]] if charge==1
         else [m|(1<<r) for r in range(8,12) for m in model.bulk_masks[3]])
    index={m:i for i,m in enumerate(out)};creators={}
    for i,r in itertools.product(range(8),range(4)):
        rows=[];cols=[];signs=[];ref=8+r
        for col,b in enumerate(incoming):
            mask=b|(ref4 if charge==1 else 0)
            first,second=(ref,i) if charge==1 else (i,ref)
            if not(mask>>first&1):continue
            sign=(-1)**((mask&((1<<first)-1)).bit_count());mask^=1<<first
            if mask>>second&1:continue
            sign*=(-1)**((mask&((1<<second)-1)).bit_count());mask|=1<<second
            rows.append(index[mask]);cols.append(col);signs.append(sign)
        creators[i,r]=sparse.csr_matrix((signs,(rows,cols)),shape=(224,70))
    return creators


def literal_coordinate_mark(law,charge,fields,links,theta,reference_site,
                            profile=None,quartet=None,a=None):
    """Original coefficient producer; charge selects a map, NEVER a new H."""
    V=law.volume
    fields=np.asarray(fields,complex);links=np.asarray(links,complex)
    theta=np.asarray(theta,float)
    require(fields.shape==(V,4,4,4) and links.shape==(V,3,4,4),'full original field/link source arrays')
    require(theta.shape==(3,) and charge in (-1,1),'theta=N A_h and signed SOURCE coefficient map')
    profile=np.ones(V,complex) if profile is None else np.asarray(profile,complex)
    quartet=np.asarray(BETA if quartet is None else quartet,complex)
    a=np.zeros((V,3)) if a is None else np.asarray(a,float)
    require(profile.shape==(V,) and quartet.shape==(4,) and a.shape==(V,3),'original SOURCE coefficient and longitudinal arrays')
    if charge==-1:
        fields=-GAMMA5@fields@GAMMA5
        links=GAMMA5@links@GAMMA5
    primitives=screened_marks(law,fields,links,a,profile,reference_site,18.,harmonic=theta/law.size)
    mark=np.einsum('a,axr->xr',quartet,primitives).reshape(V,4,4)/2
    if charge==-1:
        mark=-GAMMA5@mark.conj().swapaxes(-1,-2)@GAMMA5
    return mark


class CanonicalPreparedSpatialChargeTransfer(PreparedSpatialChargeTransfer):
    """New SOURCE view on the owner's SAME already-prepared H/state."""
    def __init__(self,prepared,charge):
        super().__init__(prepared,charge)
        if charge==1:
            self.B=self.literal_source_operator()
            self.source_state=self.B@self.xi
            self.source_norm=float(np.linalg.norm(self.source_state))
            require(self.source_norm>1e-8,'canonical native creator norm')
            self.charged_state=self.source_state/self.source_norm

    def source_coordinate_mark(self,fields,links,theta,profile=None,quartet=None,a=None):
        """Whole ORIGINAL producer; interior fields replace the old tree-only API."""
        return literal_coordinate_mark(self.law,self.charge,fields,links,theta,
                                       self.reference_site,profile,quartet,a)

    def literal_source_operator(self,profile=None,quartet=None):
        profile=np.asarray(self.orbitals[:,0] if profile is None else profile,complex)
        quartet=np.asarray(BETA if quartet is None else quartet,complex)
        require(profile.shape==(self.law.volume,),'canonical native profile')
        require(quartet.shape==(4,),'literal raw four coefficients')
        nodes=canonical_coordinate_marks(self,quartet)
        creators=literal_car_transfers(self,self.charge)
        terms=[];reference=self.law.sites[self.reference_site]
        # Group equal signed z-displacements only: the original path is NOT
        # a modulo-N word when its harmonic displacement is unwrapped.
        signed_z=sorted({int(x[2]-reference[2]) for x in self.law.sites})
        for orbital,dz in itertools.product(range(2),signed_z):
            sites=[x for x in range(self.law.volume) if int(self.law.sites[x,2]-reference[2])==dz]
            Fnodes=nodes[:,sites[0]]
            require(all(np.linalg.norm(nodes[:,x]-Fnodes)<1e-12 for x in sites),'uniform native transverse colour restriction')
            F=np.einsum('qa,qij,qb->abij',self.Q2,Fnodes,self.Q2)
            E=np.zeros((56,56),complex)
            for x in sites:
                d=self.law.sites[x]-reference;w=np.zeros(55);w[52:]=-self.charge*d/self.law.size
                eta=profile[x] if self.charge==1 else profile[x].conjugate()
                E+=eta*self.orbitals[x,orbital]*gaussian_weyl(w)
            for i,r in itertools.product(range(4),repeat=2):
                A=F[:,:,i,r] if self.charge==1 else F[:,:,r,i]
                if np.linalg.norm(A)>1e-13 and np.linalg.norm(E)>1e-13:
                    terms.append((A,sparse.csr_matrix(E),creators[4*orbital+i,r]))
        def apply(value,adjoint=False):
            si=self.shape_charged if adjoint else self.shape_neutral
            so=self.shape_neutral if adjoint else self.shape_charged
            v=np.asarray(value).reshape(si);y=np.zeros(so,complex)
            seed=not adjoint and np.count_nonzero(v[1:])==0 and np.count_nonzero(v[0,1:])==0
            for F,E,M in terms:
                if adjoint:F,E,M=F.conj().T,E.conj().T,M.conj().T
                if seed:
                    root=(M@v[0,0].reshape(12,70).T).T.reshape(6,2,224)
                    y+=np.einsum('a,b,hri->abhri',F[:,0],E.getcol(0).toarray().ravel(),root,optimize=True)
                else:
                    z=np.einsum('ab,bpqri->apqri',F,v,optimize=True)
                    moved=np.moveaxis(z,1,0);dims=moved.shape
                    z=np.moveaxis(np.asarray(E@moved.reshape(56,-1)).reshape(dims),0,1)
                    y+=np.asarray((M@z.reshape(-1,si[-1]).T).T).reshape(so)
            return y.reshape(-1)
        B=cast(Any,LinearOperator)((math.prod(self.shape_charged),math.prod(self.shape_neutral)),matvec=apply,rmatvec=lambda x:apply(x,True),dtype=complex)
        B.source_terms=terms
        return B

    def source_fourier_decomposition(self,profile=None):
        profile=np.asarray(self.orbitals[:,0] if profile is None else profile,complex)
        reference=self.law.sites[self.reference_site]
        return [dict(site=x,profile=eta,orbital_coefficients=self.orbitals[x].copy(),
                     signed_colour_z_power=int(reference[2]-self.law.sites[x,2]),
                     harmonic_word=-self.charge*(self.law.sites[x]-reference)/self.law.size,
                     coexact_word=np.zeros(52),source_regime=REGIME)
                for x,eta in enumerate(profile)]

    def creator_gram(self,profile=None):
        vectors=[self.literal_source_operator(profile,np.eye(4)[k])@self.xi for k in range(4)]
        G=np.array([[np.vdot(a,b) for b in vectors] for a in vectors])
        c=BETA if self.charge==1 else BETA.conj()
        raw=sum((c[k]*vectors[k] for k in range(4)),np.zeros_like(vectors[0]))
        actual=self.source_state if profile is None else self.literal_source_operator(profile)@self.xi
        err=float(np.linalg.norm(raw-actual));require(err<1e-11,'canonical raw quartet linearity')
        return dict(G=G,eigenvalues=np.linalg.eigvalsh(G),raw_coefficients=c,raw_squared_norm=float(np.vdot(c,G@c).real),actual_squared_norm=float(np.linalg.norm(actual)**2),linearity_error=err,not_MinSet=True,input_whitened=False)

    def metadata(self):
        data=super().metadata()
        data.update(source_regime=REGIME,source_formula='immutable screened_marks primitive stack contracted with raw quartet/2; signed Birth→Ref per-edge product, all interior frames, harmonic opposite displacement; minus explicit PH',historical_source_retained_only_as_control=True,state_reprepared=False,full_MinSet_transfer=False)
        return data


def chiral_centre_gram(model):
    """Actual4colour Clifford trace, not a Spin3=>scalar multiplicity claim."""
    h=model.law.onsite*SELECTED[0]+.13*SELECTED[1]-.21*SELECTED[2]+.07*SELECTED[3]
    eigenvalues,U=np.linalg.eigh(h);P=U[:,eigenvalues>0]@U[:,eigenvalues>0].conj().T
    G=np.array([[np.trace(A.conj().T@P@B) for B in LABELS] for A in LABELS])
    return dict(G=G,scalar_half_error=float(np.linalg.norm(G-np.eye(4)/2)),chiral_defect=float(np.linalg.norm(GAMMA5@h@GAMMA5+h)),raw_centre_squared=float(np.vdot(BETA,G@BETA).real),scope='centred Clifford fullRef trace ONLY; actual filtered light comparison still unpaid')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--state-capsule',type=Path,required=True)
    parser.add_argument('--descriptor',type=Path)
    a=parser.parse_args();prepared=PreparedSpatialGaussQuantumSlice(state_capsule=a.state_capsule)
    descriptor=None if a.descriptor is None else decode(json.loads(a.descriptor.read_bytes()))
    original_signature=prepared.operator_signature();results=[]
    for charge in (1,-1):
        m=CanonicalPreparedSpatialChargeTransfer(prepared,charge)
        profile=None;projection=None;source=m.source_state
        if descriptor is not None:
            operator,projection=m.source_from_descriptor(descriptor)
            source=operator@m.xi
            profile=np.zeros(prepared.law.volume,complex)
            for offset,eta in zip(descriptor['compact_displacement'],descriptor['source_eta']):
                profile[prepared.law.index[tuple((np.asarray(offset,dtype=int)+1)%3)]]+=eta
        norm=float(np.linalg.norm(source));charged=source/norm
        mode=np.zeros(55);mode[0]=1
        creator=m.photon_creation_parts(mode,source,charged=True)
        # Exercise the new FULL-coordinate public producer; no old tree-only
        # fallback or instance-method replacement is installed.
        phi=np.tile(prepared.frame,(prepared.law.volume,1,1,1))
        links=np.tile(np.eye(4,dtype=complex),(prepared.law.volume,3,1,1))
        coordinate=m.source_coordinate_mark(phi,links,np.array([.21,-.13,.09]),profile=profile)
        results.append(dict(metadata=m.metadata(),creator_gram=m.creator_gram(profile),source_norm=norm,current_norm=float(np.linalg.norm(m.Jc@charged)),contact_norm=float(np.linalg.norm(m.Kc@charged)),true_photon_creator_squared=creator['full_squared_norm'],missing_two_excitation_squared=creator['outside_squared_norm'],coexact_occupation=creator['occupation'],descriptor_projection=projection,full_coordinate_source_shape=list(coordinate.shape),full_coordinate_source_HS_squared=float(np.linalg.norm(coordinate)**2)))
    require(prepared.operator_signature()==original_signature,'owner historical H/state unchanged')
    print(json.dumps(portable(dict(regime=REGIME,charge_views=results,chiral_centre=chiral_centre_gram(prepared),state_sha256=hashlib.sha256(a.state_capsule.read_bytes()).hexdigest(),owner_signature=original_signature,state_reprepared=False,scope='NEW canonical source on historical native H/state; not MinSet/physicalNoether/legalN Born transfer')),indent=2))

if __name__=='__main__':main()
