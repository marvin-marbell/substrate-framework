#!/usr/bin/env python3
"""Actual constant-Phi endpoint current/contact comparison for v3W transfer.

This is a derived comparison functional, not a replacement microscopic H.
Exact legal Fourier labels remain orthogonal even when float angles coincide.
"""
from __future__ import annotations
import argparse,json,itertools,math
import numpy as np
from improved_phi_band import ImprovedPhiLaw,selected_band
from local_orbital_stability import scalar_paths
from prepared_neutral_full_field import require


class PreparedCentredCurrent:
    def __init__(self,frame=None,band=None):
        selected,self.frame,self.metric=selected_band()
        if frame is not None:self.frame=np.asarray(frame,dtype=complex)
        self.band=selected if band is None else band
        self.onsite,self.paths=scalar_paths(self.band)
        self.coefficients=[(np.asarray(d,dtype=int),np.einsum('m,mij->ij',s,self.frame)) for d,s in self.paths]
        self.state_identity='PNFF-v3-localwell-smooth-window-correlated-Hessian'
        self.comparison_scope='actualcentredA0 wholelight negativecluster; requires v3W source/occupation/Root/Gauss/state transfer, NOToldGammaoroldP'

    def onebody(self,p):
        p=np.asarray(p,float);h=self.onsite*self.frame[0].copy()
        for d,F in self.coefficients:
            phase=np.exp(1j*(d@p));h+=phase*F+phase.conjugate()*F.conj().T
        return h

    @staticmethod
    def path_factor(length,q):return sum(np.exp(1j*j*q) for j in range(length))

    def current(self,pout,pin,polarization):
        pout=np.asarray(pout,float);pin=np.asarray(pin,float);q=pout-pin
        result=np.zeros((4,4),complex)
        for d,F in self.coefficients:
            axis=int(np.flatnonzero(d)[0]);length=int(d[axis])
            s=self.path_factor(length,q[axis]);e=polarization[axis]
            result+=1j*e*s*(F*np.exp(1j*d@pin)-F.conj().T*np.exp(-1j*d@pout))
        return result

    def contact(self,pin,k1,k2,polarization1,polarization2):
        pin=np.asarray(pin,float);k1=np.asarray(k1,float);k2=np.asarray(k2,float);pout=pin+k1+k2
        result=np.zeros((4,4),complex)
        for d,F in self.coefficients:
            a=int(np.flatnonzero(d)[0]);length=int(d[a])
            factor=-polarization1[a]*polarization2[a]*self.path_factor(length,k1[a])*self.path_factor(length,k2[a])
            result+=factor*(F*np.exp(1j*d@pin)+F.conj().T*np.exp(-1j*d@pout))
        return result

    def negative_projector(self,p):
        values,vectors=np.linalg.eigh(self.onebody(p));require(values[1]<0<values[2],'actualwholelightgap atsample')
        return vectors[:,:2]@vectors[:,:2].conj().T

    @staticmethod
    def exact_mode_phase(mode,site,size):
        residue=sum(int(k)*int(x) for k,x in zip(mode,site))%int(size)
        return np.exp(2j*math.pi*float(residue/size))

    def source_fourier(self,mode,offsets,eta,size):
        # Orthogonality is the exact mode identity, NEVER a float-angle test.
        return sum(complex(a)*self.exact_mode_phase(tuple(-int(k) for k in mode),d,size) for d,a in zip(offsets,eta))

    def metadata(self):
        return dict(state_identity=self.state_identity,scope=self.comparison_scope,
            law='same improvedlight literalPhi endpoints/range2 Peierls products beforeanymatrixcompression',
            exact_distinct_modes='tuple(integerlabels), independently of float momenta',
            metric=self.metric,actual_ground_claimed=False,new_leading_coefficient_not_assumed=True)


def native():
    model=PreparedCentredCurrent();law=ImprovedPhiLaw(3,model.band)
    phi=np.tile(model.frame,(law.volume,1,1,1));u=np.tile(np.eye(4),(law.volume,3,1,1));a=np.zeros((law.volume,3))
    pin=2*math.pi*np.array([0,1,0])/3;k1=2*math.pi*np.array([1,0,0])/3;k2=2*math.pi*np.array([0,1,0])/3
    p1=pin+k1;p2=pin+k1+k2;eps1=np.array([0.,0.,1.]);eps2=np.array([0.,0.,1.])
    def fold(p):return np.exp(1j*(law.sites@p))[:,None,None]*np.eye(4)[None]/math.sqrt(law.volume)
    fi=fold(pin).reshape(4*law.volume,4);fo1=fold(p1).reshape(4*law.volume,4);fo2=fold(p2).reshape(4*law.volume,4)
    c1=np.cos(law.sites@k1)[:,None]*eps1;s1=np.sin(law.sites@k1)[:,None]*eps1
    c2=np.cos(law.sites@k2)[:,None]*eps2;s2=np.sin(law.sites@k2)[:,None]*eps2
    H=fi.conj().T@law.onebody(phi,u,a)@fi
    J=fo1.conj().T@(law.onebody(phi,u,a,c1)+1j*law.onebody(phi,u,a,s1))@fi
    Q=fo2.conj().T@(law.onebody(phi,u,a,c1,c2)-law.onebody(phi,u,a,s1,s2)+1j*(law.onebody(phi,u,a,c1,s2)+law.onebody(phi,u,a,s1,c2)))@fi
    errors=dict(H=float(np.linalg.norm(H-model.onebody(pin))),J=float(np.linalg.norm(J-model.current(p1,pin,eps1))),contact=float(np.linalg.norm(Q-model.contact(pin,k1,k2,eps1,eps2))))
    for key,error in errors.items():require(error<1e-10,'literalcentred'+key)
    wrong=model.current(p1,pin,eps1)@model.current(p2,p1,eps2)
    adverse=float(np.linalg.norm(wrong-model.contact(pin,k1,k2,eps1,eps2)))
    require(adverse>1e-6,'contact isnotfirstcurrentsquared')
    return dict(metadata=model.metadata(),full108_literal_errors=errors,wrong_first_squared_contact_defect=adverse)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args();r=native()
    from original_charged_principal import portable
    print(json.dumps(portable(r),indent=2))
if __name__=='__main__':main()
