#!/usr/bin/env python3
"""Shared complete-EM Galerkin with actual Coulomb/complement/reference source.

Frozen omitted orbitals specify a native comparison embedding, not the actual
legal-N whole light sea. Its density products are compressed BEFORE squaring.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os,time
from pathlib import Path
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator,expm_multiply
from prepared_neutral_spatial_em_well import PreparedSpatialEMQuantumSlice,gaussian_weyl
from prepared_neutral_spatial_well import creation_number_matrix
from prepared_neutral_local_well import actual_window
from prepared_neutral_full_field import require,BETA
from noncompact_screened_gauss_dressing import frame_chart,centre_scalar
from sea_selection import SELECTED
IDENTITY='PNFF-v3W-shared-completeEM-Coulomb-complement-staticRef'


class PreparedSpatialGaussQuantumSlice(PreparedSpatialEMQuantumSlice):
    def __init__(self,state_capsule:Path|None=None,save_state:Path|None=None):
        self.state_capsule=state_capsule;self.save_state=save_state
        super().__init__()
        self.prepared_once=True

    def _project_actual_coordinates(self):
        super()._project_actual_coordinates()
        V=self.law.volume;orb=self.orbitals;R=orb@orb.T;S=np.eye(V)-R
        self.reference_site=self.law.index[(0,1,1)]
        self.retained_density_diagonal=np.diag(R)
        lap=np.zeros((V,V))
        for x in range(V):
            for a in range(3):
                y=int(self.law.target[x,a]);lap[x,x]+=1;lap[y,y]+=1;lap[x,y]-=1;lap[y,x]-=1
        e,U=np.linalg.eigh(lap);positive=e>1e-9
        self.coulomb_green=(U[:,positive]/e[positive])@U[:,positive].T
        M=orb.T@(self.coulomb_green*S)@orb
        self.complement_charge_matrix=M
        self.coulomb_blocks={}
        for number in (4,5):
            masks=self.bulk_masks[number];n=len(masks);I=np.eye(n)
            rho=[]
            for x in range(V):
                A=np.kron(np.outer(orb[x],orb[x]),np.eye(4));mat=np.zeros((n,n),complex)
                for i,j in itertools.product(range(8),repeat=2):mat+=A[i,j]*self.light_operators[number][i][j].toarray()
                mat-=2*R[x,x]*I
                if number==5 and x==self.reference_site:mat-=I
                rho.append(mat)
            green_e,green_U=np.linalg.eigh(self.coulomb_green)
            modes=np.einsum('xm,xij->mij',green_U[:,green_e>1e-9]*np.sqrt(green_e[green_e>1e-9]),np.array(rho))
            Cmat=sum((q@q for q in modes),np.zeros((n,n),complex))
            correction=np.kron(M,SELECTED[0]);onebody=np.zeros((n,n),complex)
            for i,j in itertools.product(range(8),repeat=2):onebody+=correction[i,j]*self.light_operators[number][i][j].toarray()
            Cmat=self.ue*(Cmat+onebody+2*np.trace(M)*I)
            require(np.linalg.norm(Cmat-Cmat.conj().T)<1e-10,'actualCoulombGalerkinHermiticity')
            require(float(np.linalg.eigvalsh(Cmat)[0])>-1e-9,'actualpositiveCoulombbeforecompression')
            self.coulomb_blocks[number]=sparse.csr_matrix(Cmat) if number==4 else sparse.kron(sparse.eye(4),sparse.csr_matrix(Cmat),format='csr')
        # Omitted light orbitals: negative Gamma0, not an invented scalar sea.
        Pminus=(np.eye(4)-SELECTED[0])/2;Pplus=(np.eye(4)+SELECTED[0])/2
        light_onsite=(V-2)*np.einsum('ij,abji->ab',Pminus,self.onsite_coeff)
        self.scalar_matrix=self.scalar_matrix.astype(complex)+light_onsite
        self.outside_light=[np.einsum('ij,abji->ab',Pminus,F) for F in self.endpoint_coeff]
        PoutH=np.kron(np.eye(V)-np.ones((V,V))/V,Pplus)
        heavy_out=[]
        for ip,ic in self.pc_points:
            p=self.points[ip]*self.scales[0];c=self.points[ic]*self.scales[1]
            phi=np.tile(self.frame,(V,1,1,1));phi[:,0]+=p[0]*SELECTED[1]+p[1]*SELECTED[2]
            links=np.tile(np.eye(4,dtype=complex),(V,3,1,1));links[:,2]=expm(1j*(c[0]*SELECTED[1]+c[1]*SELECTED[2])/math.sqrt(8))
            heavy_out.append(-self.Lambda*np.trace(PoutH@self.heavy.onebody(phi,links,np.zeros((V,3)))).real)
        self.scalar_matrix+=self.Q2.T@(np.array(heavy_out)[:,None]*self.Q2)
        self.complement_scalar_nodes=np.array(heavy_out)+(V-2)*np.trace(Pminus@(self.law.onsite*self.frame[0])).real
        self._mark_by_z={}
        for length in range(3):
            values=[]
            for ip,ic in self.pc_points:
                p=self.points[ip]*self.scales[0];c=self.points[ic]*self.scales[1]
                phi=self.frame.copy();phi[0]+=p[0]*SELECTED[1]+p[1]*SELECTED[2]
                U=expm(1j*(c[0]*SELECTED[1]+c[1]*SELECTED[2])/math.sqrt(8))
                word=np.linalg.matrix_power(U,length)
                values.append(self.source_coordinate_mark(phi,word,phi,np.zeros(3),np.zeros(3)))
            self._mark_by_z[length]=np.einsum('qa,qij,qb->abij',self.Q2,np.array(values),self.Q2)

    def gauss_background(self,parameter):
        return 2*np.dot(np.asarray(parameter),self.retained_density_diagonal)

    def hamiltonian(self,sector='neutral',derivative=0):
        H=super().hamiltonian(sector,derivative)
        for k,(axis,length,_) in enumerate(self._endpoint_data[0]):
            F=self.outside_light[k];W=sum(self.W[axis,length,derivative][:,:,a,a] for a in range(2))
            H.terms.append((-1,[(0,F),(1,W),(3,self.loop)]))
            H.terms.append((-1,[(0,F.conj().T),(1,W.conj().T),(3,self.loop)]))
        if derivative==0:
            if sector=='charged':H.terms=[(c,fs) for c,fs in H.terms if not(len(fs)==0 and c==self.ue)]
            H.terms.append((1,[(4,self.coulomb_blocks[4 if sector=='neutral' else 5])]))
        return H

    def form_floor(self):
        # True projected Coulomb is positive. Complement scalar node minima
        # and hopping l1 account for the otherwise missing ordinary offset.
        original=super().form_floor()
        hopping=2*sum(np.linalg.norm(A,2) for A in self.outside_light)
        return original+float(self.complement_scalar_nodes.min())-hopping-1e-7

    def source_coordinate_mark(self,phi_x,tree,phi_ref,theta,displacement):
        """Literal SOURCE before compression; both60Phi endpoint jets exist."""
        Fx=frame_chart(np.asarray(phi_x))[0];Fr=frame_chart(np.asarray(phi_ref))[0]
        labels=np.array([np.eye(4),*(SELECTED[0]@SELECTED[j] for j in range(1,4))])
        Q=np.einsum('a,aij->ij',BETA,labels)/2
        scalar=centre_scalar(Fx.conj().T@tree@Fr,18.,self.law.generators)[0]
        phase=np.exp(-1j*np.dot(theta,displacement)/self.law.size)
        return scalar*phase*Fx@Q@Fr.conj().T

    def source_fourier_decomposition(self,profile=None):
        """Exact finite harmonic Fourier words; no coexact dependence."""
        if profile is None:profile=self.orbitals[:,0]
        reference=self.law.sites[self.reference_site];terms=[]
        for x,eta in enumerate(profile):
            displacement=self.law.sites[x]-reference
            terms.append(dict(site=x,profile=complex(eta),orbital_coefficients=self.orbitals[x].copy(),
                colour_tree_z_length=int(displacement[2]%3),harmonic_word=-displacement/self.law.size,
                coexact_word=np.zeros(52),Phi_x_and_Phi_ref_independent_before_projection=True))
        return terms

    def source_operator(self,profile=None):
        V=self.law.volume
        if profile is None:profile=self.orbitals[:,0].astype(complex)
        profile=np.asarray(profile,complex);require(profile.shape==(V,),'actualspatialsourceprofile')
        self.canonical_profile=profile.copy()
        index={m:k for k,m in enumerate(self.charged_masks)}
        creators={}
        for i,r in itertools.product(range(8),range(4)):
            rows=[];cols=[];vals=[];j=8+r
            for k,m in enumerate(self.neutral_masks):
                m1=m^(1<<j)
                if m1>>i&1:continue
                sign=(-1)**((m&((1<<j)-1)).bit_count()+(m1&((1<<i)-1)).bit_count())
                rows.append(index[m1|(1<<i)]);cols.append(k);vals.append(sign)
            creators[i,r]=sparse.csr_matrix((vals,(rows,cols)),shape=(224,70))
        grouped={}
        ref=self.law.sites[self.reference_site]
        for orbital in range(2):
            for length in range(3):
                E=np.zeros((56,56),complex)
                for x in range(V):
                    dz=int((self.law.sites[x,2]-ref[2])%3)
                    if dz!=length:continue
                    word=np.zeros(55);word[52:]=-(self.law.sites[x]-ref)/3
                    E+=profile[x]*self.orbitals[x,orbital]*gaussian_weyl(word)
                if np.linalg.norm(E)>1e-14:grouped[orbital,length]=sparse.csr_matrix(E)
        terms=[]
        for (orbital,length),E in grouped.items():
            for i,r in itertools.product(range(4),range(4)):
                F=self._mark_by_z[length][:,:,i,r]
                if np.linalg.norm(F)>1e-13:terms.append((F,E,creators[4*orbital+i,r]))
        def apply(value,adjoint=False):
            shape_in=self.shape_charged if adjoint else self.shape_neutral
            shape_out=self.shape_neutral if adjoint else self.shape_charged
            x=np.asarray(value).reshape(shape_in);y=np.zeros(shape_out,complex)
            for F,E,M in terms:
                if adjoint:F,E,M=F.conj().T,E.conj().T,M.conj().T
                z=np.einsum('ab,bpqri->apqri',F,x,optimize=True)
                moved=np.moveaxis(z,1,0);dims=moved.shape
                z=np.moveaxis(np.asarray(E@moved.reshape(56,-1)).reshape(dims),0,1)
                z=np.asarray((M@z.reshape(-1,shape_in[-1]).T).T).reshape(shape_out)
                y+=z
            return y.reshape(-1)
        B=LinearOperator((math.prod(self.shape_charged),math.prod(self.shape_neutral)),matvec=apply,rmatvec=lambda x:apply(x,True),dtype=complex)
        B.source_terms=terms
        return B

    def source_from_descriptor(self,descriptor):
        displacements=np.asarray(descriptor['compact_displacement'],dtype=int)
        values=np.asarray(descriptor['source_eta'],dtype=complex)
        require(displacements.shape==(len(values),3),'actualsource19compactdisplacements')
        profile=np.zeros(27,complex)
        for d,eta in zip(displacements,values):
            native_site=tuple((d+1)%3)
            profile[self.law.index[native_site]]+=eta
        projection=self.orbitals.conj().T@profile
        residue=float(np.linalg.norm(profile-self.orbitals@projection))
        return self.source_operator(profile),dict(scope='actual19complexsource onN3body, nativeFAR/harmonicdistance scaled; notliterallegalNscattering',profile_norm=float(np.linalg.norm(profile)),retained_projection=projection,spatial_projection_residue=residue,carrier_retained=True,normalized_per_column=False)

    def operator_signature(self):
        h=hashlib.sha256()
        for H in (self.Hn,self.Hc):
            h.update(np.asarray(H.tensor_shape,dtype=np.int64).tobytes());h.update(np.array([H.offset],float).tobytes())
            for c,fs in H.terms:
                h.update(np.array([c],complex).tobytes())
                for axis,A in fs:
                    h.update(np.array([axis],np.int64).tobytes());h.update(np.asarray(A.toarray() if sparse.issparse(A) else A).tobytes())
        for F,E,M in self.B.source_terms:
            for A in (F,E,M):h.update(np.asarray(A.toarray() if sparse.issparse(A) else A).tobytes())
        h.update(self.em_maps.tobytes());return h.hexdigest()

    def prepare_window(self):
        signature=self.operator_signature()
        if self.state_capsule is not None:
            with np.load(self.state_capsule,allow_pickle=False) as data:
                require(str(data['operator_signature'])==signature,'preparedcapsule actualoperatoridentity')
                state=data['state'];norm=float(data['window_norm'])
                require(state.shape==self.seed.shape and abs(np.linalg.norm(state)-1)<1e-10,'preparedcapsule normalizedsame-state')
        else:
            upper=-self.Hn.offset
            for coefficient,factors in self.Hn.terms:
                bound=abs(coefficient)
                for _,A in factors:
                    B=abs(A)
                    column=float(np.asarray(B.sum(axis=0)).max())
                    row=float(np.asarray(B.sum(axis=1)).max())
                    bound*=math.sqrt(column*row)
                upper+=bound
            self.window_native_upper=upper
            if upper-self.floor<=self.window_width:
                # Functional calculus, not an approximate/no-op filter:
                # the entire finite spectrum is PROVED in f's flat plateau.
                state=self.seed.copy();norm=1.
                self.window_native_plateau_proved=True
            else:
                state,norm=actual_window(self.Hn,self.seed,self.floor,self.window_width,60)
                self.window_native_plateau_proved=False
        if self.save_state is not None:
            np.savez_compressed(self.save_state,state=state,window_norm=norm,floor=self.floor,window_width=self.window_width,operator_signature=signature)
        return state,norm

    def photon_creation_parts(self,coefficients,state,charged=False):
        g=np.asarray(coefficients,complex);require(g.shape==(55,) and np.linalg.norm(g[52:])<1e-12,'physicalcoexactmodecoefficients')
        require(abs(np.linalg.norm(g)-1)<1e-10,'unitcanonicalphotonmode')
        shape=self.shape_charged if charged else self.shape_neutral
        v=np.asarray(state).reshape(shape);ann=np.einsum('i,aibcd->abcd',g.conj(),v[:,1:],optimize=True)
        clipped=np.zeros_like(v);clipped[:,1:]=g[None,:,None,None,None]*v[:,0,None]
        outside=float(np.linalg.norm(v[:,1:])**2+np.linalg.norm(ann)**2)
        total=float(np.linalg.norm(clipped)**2+outside)
        require(abs(total-(np.linalg.norm(state)**2+np.linalg.norm(ann)**2))<1e-8,'actualCCRwithtwoexcitationremainder')
        return dict(clipped=clipped.reshape(-1),two_excitation_factors=(g,v[:,1:]),outside_squared_norm=outside,full_squared_norm=total,occupation=float(np.linalg.norm(ann)**2),native_H_does_not_evolve_two_excitation_sector=True)

    def metadata(self):
        d=super().metadata();d.update(identity=IDENTITY,
            Coulomb='canonicalUE P rho Green rho P; retquartic+2trM+dGamma(M tensorGamma0) BEFOREcompression',
            complement='50lightGamma0negative and52heavyGamma0positive frozencomparisonorbitals; wholelight54/heavy54, NOTactualwholefull108negativecluster',
            source_reference='actualuniformPhi F_R(q) retained; staticRefposition is NOTfrozenfieldframe; treeUz^dz andphysicalharmonicphase beforeprojection; all4Ref3holes',
            reference_site=self.reference_site,gauss_background='2 sumlambda Rxx',state_capsule_loaded=self.state_capsule is not None,
            source_coexact_scope='fullcanonicaloperatorcommutescoexact; totaldegree1compression canbreakmixedcoexact/harmonicCCR, remainderretained',
            ordinary_complement_offsets_retained=True,field_kinetic_scope='flat tangent twoPhi/twoSU4 chart CONTROL, notexactfullHaar/GaussLaplace',
            native_window_entire_spectrum_plateau_proved=getattr(self,'window_native_plateau_proved',None),native_window_upper=getattr(self,'window_native_upper',None))
        return d


def native(T,state_capsule=None,save_state=None):
    start=time.perf_counter();m=PreparedSpatialGaussQuantumSlice(state_capsule,save_state)
    evolved=m.state.copy() if T==0 else expm_multiply(-1j*T*m.Hn,m.state,traceA=-1j*T*m.Hn.trace_value())
    current=float(np.linalg.norm(m.Jc@m.charged_state));contact=float(np.linalg.norm(m.Kc@m.charged_state))
    g=np.zeros(55,complex);g[0]=1;creation=m.photon_creation_parts(g,m.source_state,True)
    return dict(metadata=m.metadata(),operator_signature=m.operator_signature(),time=T,neutral_flight_executed=T!=0,norm_error=float(abs(np.linalg.norm(evolved)-1)),current_norm=current,contact_norm=contact,
        creation_full_norm_squared=creation['full_squared_norm'],creation_missing_two_quantum_norm_squared=creation['outside_squared_norm'],source_weighted_occupation=creation['occupation']/m.source_norm**2,
        elapsed_seconds=time.perf_counter()-start,state_capsule=str(save_state) if save_state is not None else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--time',type=float,default=.01);p.add_argument('--state-capsule',type=Path);p.add_argument('--save-state',type=Path);a=p.parse_args()
    from original_charged_principal import portable
    print(json.dumps(portable(native(a.time,a.state_capsule,a.save_state)),indent=2))
if __name__=='__main__':main()
