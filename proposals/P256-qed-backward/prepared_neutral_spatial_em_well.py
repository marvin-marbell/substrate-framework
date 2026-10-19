#!/usr/bin/env python3
"""All N3 spatial EM modes on the SAME actual-window quantum chart.

Exact Gaussian matrix elements precede compression. Noncompact H has K curl²;
physical Noether sine hardware is constructed separately by the angular owner.
"""
from __future__ import annotations
import argparse,json,math,os,time
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from prepared_neutral_spatial_well import PreparedSpatialQuantumSlice,TensorHamiltonian
from prepared_neutral_full_field import require
from improved_phi_band import C
IDENTITY='PNFF-v3W-shared-spatial-complete52coexact3harmonic'


def gaussian_weyl(word: np.ndarray) -> np.ndarray:
    """P_{totaldegree<=1} exp(i word.q) P, NOT exp(i PqP)."""
    e=math.exp(-float(word@word)/4);n=len(word)
    M=np.zeros((n+1,n+1),complex);M[0,0]=e
    M[0,1:]=M[1:,0]=1j*word/math.sqrt(2)*e
    M[1:,1:]=(np.eye(n)-np.outer(word,word)/2)*e
    return M


class PreparedSpatialEMQuantumSlice(PreparedSpatialQuantumSlice):
    """PC3x3, EM totaldegree<=1 dim56, heavy6, exact Ref/CAR Root blocks."""
    def __init__(self):
        super().__init__()
        self.shape_n=(3,3,56,6,140);self.shape_c=(3,3,56,6,448)
        self.boson_shape=(3,3,56)
        self.em_labels=['vacuum']+[f'coexact{k}' for k in range(52)]+['harmonic_x','harmonic_y','harmonic_z']
        self.xi=self.state

    def _project_actual_coordinates(self):
        super()._project_actual_coordinates()
        law=self.law;V=law.volume
        curl=np.zeros((3*V,3*V))
        for x in range(V):
            for a,b,c in ((0,1,2),(1,2,0),(2,0,1)):
                curl[3*x+a,3*int(law.target[x,b])+c]+=1
                curl[3*x+a,3*x+c]-=1
                curl[3*x+a,3*int(law.target[x,c])+b]-=1
                curl[3*x+a,3*x+b]+=1
        eigen,vector=np.linalg.eigh(curl.T@curl)
        pos=eigen>1e-9;require(int(np.count_nonzero(pos))==2*(V-1),'completecoexactrank')
        coexact=vector[:,pos].T;omega=float(C)*np.sqrt(eigen[pos])
        self.em_maps=np.zeros((55,V,3))
        self.em_maps[:52]=coexact.reshape(52,V,3)*np.sqrt(2*self.ue/omega)[:,None,None]
        for a in range(3):self.em_maps[52+a,:,a]=1/law.size
        self.coexact_basis=coexact;self.coexact_omega=omega;self.curl=curl
        self.shape_neutral=(9,56,6,2,70);self.shape_charged=(9,56,6,2,224)
        self.W={}
        for axis,length,_ in self._endpoint_data[0]:
            for derivative in (0,1,2):self.W[axis,length,derivative]=self._em_word_projection(axis,length,derivative,self.photon_profiles[0],self.photon_profiles[0])
        # Actual coordinate curl potential, compressed analytically. Harmonic
        # curl is zero, not a claimed harmonic oscillator confinement.
        metric=self.K/2*(self.em_maps.reshape(55,-1)@curl.T)@(self.em_maps.reshape(55,-1)@curl.T).T
        self.photon_scalar=np.zeros((56,56),complex)
        self.photon_scalar[0,0]=np.trace(metric)/2
        self.photon_scalar[1:,1:]=metric+np.trace(metric)/2*np.eye(55)
        kinetic=np.concatenate([omega/2,np.repeat(self.ue/law.size,3)])
        self.em_kinetic=np.diag(np.concatenate([[kinetic.sum()/2],kinetic.sum()/2+kinetic]))
        self.em_free_energy=self.photon_scalar+self.em_kinetic
        require(np.linalg.norm(self.em_free_energy-np.diag(self.em_free_energy.diagonal()))<1e-10,'coexactactualquadraticdiagonal')

    def _em_word_projection(self,axis,length,derivative=0,first=None,second=None):
        V=self.law.volume;ys=np.arange(V);word=np.zeros((V,55));d1=np.zeros(V);d2=np.zeros(V)
        for _ in range(length):
            word+=self.em_maps[:,ys,axis].T
            if first is not None:d1+=first[ys,axis]
            if second is not None:d2+=second[ys,axis]
            ys=self.law.target[ys,axis]
        result=np.zeros((56,56,2,2),complex)
        for x in range(V):
            weight=1 if derivative==0 else 1j*d1[x] if derivative==1 else -d1[x]*d2[x]
            E=gaussian_weyl(word[x])*weight
            result+=E[:,:,None,None]*np.outer(self.orbitals[x],self.orbitals[ys[x]])
        return result

    def hamiltonian(self,sector='neutral',derivative=0):
        H=super().hamiltonian(sector,derivative)
        if derivative==0:
            # Remove ONLY the inherited two-pair p² matrices. Both inherited
            # original scalar and actual56-dimensional curl potential remain.
            H.terms=[(c,fs) for c,fs in H.terms if not(len(fs)==1 and fs[0][0]==1 and fs[0][1].shape==(9,9))]
            H.terms.append((1,[(1,self.em_kinetic)]))
        return H

    def current(self,sector,first,second=None):
        order=1 if second is None else 2
        old=self.W
        self.W=dict(old)
        try:
            for axis,length,_ in self._endpoint_data[0]:self.W[axis,length,order]=self._em_word_projection(axis,length,order,first,second)
            return self.hamiltonian(sector,order)
        finally:self.W=old

    def source_from_spatial_coefficients(self,coefficients):
        """Actual retained orbital projection; no per-column normalization."""
        coefficients=np.asarray(coefficients,dtype=complex)
        require(coefficients.shape==(2,),'retained physical source orbitalcoefficients')
        original=self.source_coeff
        mark=original[:,:,:4,:].copy()
        self.source_coeff=np.concatenate([coefficients[0]*mark,coefficients[1]*mark],axis=2)
        try:return self.source_operator()
        finally:self.source_coeff=original

    def metadata(self):
        d=super().metadata()
        d.update(identity=IDENTITY,em_modes=55,coexact_modes=52,harmonic_modes=3,
            em_H='canonicalLAW05 UEp²+(K/2)curl²; allN3coexact+harmonic spatialmodes, totalHermiteorder<=1',
            em_projection='exact55DGaussianWeyl beforecompression; notexp(compressedA)',
            native_v3W_scope='actualfullHwindow Galerkin CONTROL; notlegalN Dirichlet/gamma/truecentres',
            canonical_B='nativechart PURECOULOMB source; actual19complexprofile mustproject explicitly, defaultfx primitive NOTSC19',
            full_spatial_modes_retained=True,full_photon_Fock_retained=False)
        return d


def native(T=.02):
    start=time.perf_counter();m=PreparedSpatialEMQuantumSlice();v=m.state
    z=m.Hn@v
    normH=float(np.linalg.norm(z));variance=float(np.vdot(z,z).real-np.vdot(v,z).real**2)
    evolved=expm_multiply(-1j*T*m.Hn,v,traceA=-1j*T*m.Hn.trace_value())
    charged=expm_multiply(-1j*T*m.Hc,m.charged_state,traceA=-1j*T*m.Hc.trace_value())
    normerror=abs(np.linalg.norm(evolved)-1);require(normerror<1e-7,'actualfullEMneutralflightnorm')
    # Literal ordered current/contact are derivatives of the same physicalH.
    return dict(metadata=m.metadata(),time=T,neutral_norm_error=normerror,Hstate_norm=normH,Hvariance=variance,
        actual_source_neutral_right_difference=float(np.linalg.norm(charged-m.B@evolved/m.source_norm)),
        actual_current_norm=float(np.linalg.norm(m.Jc@m.charged_state)),actual_contact_norm=float(np.linalg.norm(m.Kc@m.charged_state)),
        elapsed_seconds=time.perf_counter()-start,physical_Noether_owner='ClosedSpatialRotation',SC19_consumer_owner='PhysicalCarrierTransport')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--time',type=float,default=.02);a=p.parse_args()
    print(json.dumps(native(a.time),indent=2))
if __name__=='__main__':main()
