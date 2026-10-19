#!/usr/bin/env python3
"""Shared spatial full-field Galerkin Hamiltonian and actual-window state.

The continuum Lie chart is closed under SO2. The lattice Hamiltonian need not
be: its physical lattice remainder is retained, not symmetrized away.
"""
from __future__ import annotations
import argparse,itertools,json,math,os,time
from pathlib import Path
from typing import Any
if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator,expm_multiply
from numpy.polynomial.hermite import hermgauss
from improved_phi_band import ImprovedPhiLaw,selected_band,exterior,C
from rotation_band import RotationBand
from unrestricted_field_minimum import BARE,local_potential
from sea_selection import SELECTED
from noncompact_screened_gauss_dressing import frame_chart,centre_scalar
from prepared_neutral_local_well import actual_window
from prepared_neutral_full_field import BETA,require

IDENTITY='PNFF-v3W-spatial-twoorbital-four-pair-full-H-window'


def pair_basis(order:int=3):
    x,w=hermgauss(order);w=w/math.sqrt(math.pi)
    points=np.array(list(itertools.product(x,x)))
    weight=np.array([a*b for a,b in itertools.product(w,w)])
    values=np.stack([np.ones(len(points)),math.sqrt(2)*points[:,0],math.sqrt(2)*points[:,1]],axis=1)
    Q=np.sqrt(weight)[:,None]*values
    require(np.linalg.norm(Q.T@Q-np.eye(3))<1e-12,'joint two-dimensional Hermite quadrature')
    p2=np.diag([.5,1.5,.5])+np.diag([.5,.5,1.5])
    L=np.array([[0,0,0],[0,0,-1j],[0,1j,0]])
    return points,Q,p2,L


def creation_number_matrix(masks:list[int],i:int,j:int):
    index={m:k for k,m in enumerate(masks)};rows=[];cols=[];vals=[]
    for k,m in enumerate(masks):
        if not m>>j&1:continue
        m1=m^(1<<j);sgn=(-1)**((m&((1<<j)-1)).bit_count())
        if m1>>i&1:continue
        sgn*=(-1)**((m1&((1<<i)-1)).bit_count());m2=m1|(1<<i)
        if m2 in index:rows.append(index[m2]);cols.append(k);vals.append(sgn)
    return sparse.csr_matrix((vals,(rows,cols)),shape=(len(masks),len(masks)),dtype=complex)


class TensorHamiltonian(LinearOperator):
    """Explicit sum of physical tensor terms; no enormous dense allocation."""
    def __init__(self,shape,terms,offset=0.):
        self.tensor_shape=tuple(shape);self.terms=terms;self.offset=offset
        n=math.prod(shape);super().__init__(dtype=np.dtype(complex),shape=(n,n))
    def _matvec(self,value):
        x=np.asarray(value).reshape(self.tensor_shape);y=-self.offset*x
        for coefficient,factors in self.terms:
            z=x
            for axis,A in factors:
                flat=np.moveaxis(z,axis,0);dims=flat.shape
                flat=A@flat.reshape(dims[0],-1)
                z=np.moveaxis(np.asarray(flat).reshape(dims),0,axis)
            y=y+coefficient*z
        return y.reshape(-1)
    def _rmatvec(self,value):return self._matvec(value)
    def _matmat(self,value):return np.column_stack([self._matvec(value[:,j]) for j in range(value.shape[1])])
    def trace_value(self):
        n=math.prod(self.tensor_shape);trace=-self.offset*n
        for coefficient,factors in self.terms:
            value=coefficient*n
            for axis,A in factors:value*=np.asarray(A.diagonal()).sum()/self.tensor_shape[axis]
            trace+=value
        return complex(trace)


class PreparedSpatialQuantumSlice:
    """Actual invariant Nref blocks, not an occupation-truncated light sea.

    Tensor axes: Phi/color product9, coexact/harmonic product9, heavy6,
    auxiliary2, and exact bulk/ref CAR block (70 neutral,224 charged).
    Every pair basis is (00,10,01), lexicographic outer tensor order.
    """
    def __init__(self,quadrature_order=3):
        self.band,self.frame,_=selected_band();self.law=ImprovedPhiLaw(3,self.band)
        self.heavy=ImprovedPhiLaw(3,RotationBand(**BARE))
        self.Lambda=4.;self.chi=self.Lambda**-.5;self.ue=.0025
        self.K=float(C)**2/(2*self.ue);self.beta=.13;self.aux_penalty=20.
        self.scales=np.array([.10,.14,math.sqrt(2*self.ue/(float(C)*math.sqrt(3))),.12])
        points,Q,p2,L=pair_basis(quadrature_order)
        self.points,self.Q,self.pair_p2,self.pair_L=points,Q,p2,L
        self.pair_labels=[(0,0),(1,0),(0,1)];self.boson_shape=(3,3,3,3)
        pc=np.array(list(itertools.product(range(len(points)),repeat=2)))
        self.pc_points=pc;Q2=np.kron(Q,Q);self.Q2=Q2
        xyz=self.law.sites.astype(float)-1
        eta=np.exp(-np.sum(xyz*xyz,axis=1)/(2*.65**2))
        orbitals=np.stack([xyz[:,0]*eta,xyz[:,1]*eta],axis=1)
        orbitals/=np.linalg.norm(orbitals,axis=0)
        require(np.linalg.norm(orbitals.T@orbitals-np.eye(2))<1e-12,'two real physical spatial orbitals')
        self.orbitals=orbitals;self.fold=np.kron(orbitals,np.eye(4))
        self.heavy_fold=np.tile(np.eye(4),(self.law.volume,1))/math.sqrt(self.law.volume)
        self.Sz=-.5j*SELECTED[1]@SELECTED[2]
        self.photon_profiles=np.zeros((2,self.law.volume,3))
        for a in range(2):self.photon_profiles[a,:,a]=np.cos(2*math.pi*self.law.sites[:,2]/3)
        self.photon_profiles/=np.linalg.norm(self.photon_profiles,axis=(1,2))[:,None,None]
        self.harmonic_profiles=np.zeros_like(self.photon_profiles)
        for a in range(2):self.harmonic_profiles[a,:,a]=1/3
        self.bulk_masks={q:[sum(1<<i for i in inds) for inds in itertools.combinations(range(8),q)] for q in (4,5)}
        self.neutral_masks=[m|sum(1<<i for i in range(8,12)) for m in self.bulk_masks[4]]
        self.charged_masks=[m|(sum(1<<i for i in range(8,12))^(1<<r)) for r in range(8,12) for m in self.bulk_masks[5]]
        self.shape_neutral=(9,9,6,2,70);self.shape_charged=(9,9,6,2,224)
        self.light_operators={q:[[creation_number_matrix(self.bulk_masks[q],i,j) for j in range(8)] for i in range(8)] for q in (4,5)}
        self.loop=np.diag([1.,-1.]);self.defect=np.array([[0,1],[1,0]],complex)
        self._endpoint_data=[];self._heavy=[];self._scalar=[];self._source=[]
        self._project_actual_coordinates()
        self.Hn=self.hamiltonian('neutral');self.Hc=self.hamiltonian('charged')
        self.Jn=self.hamiltonian('neutral',derivative=1);self.Jc=self.hamiltonian('charged',derivative=1)
        self.Cn=self.hamiltonian('neutral',derivative=2);self.Cc=self.hamiltonian('charged',derivative=2)
        self.B=self.source_operator()
        h0=self.fold.conj().T@self.law.onebody(np.tile(self.frame,(27,1,1,1)),np.tile(np.eye(4),(27,3,1,1)),np.zeros((27,3)))@self.fold
        hh0=self.heavy_fold.conj().T@self.heavy.onebody(np.tile(self.frame,(27,1,1,1)),np.tile(np.eye(4),(27,3,1,1)),np.zeros((27,3)))@self.heavy_fold
        _,light=np.linalg.eigh(h0);_,heavy=np.linalg.eigh(hh0)
        self.light_seed=np.array([np.linalg.det(light[[i for i in range(8) if m>>i&1],:4]) for m in self.bulk_masks[4]])
        _,self.heavy_masks=exterior(np.zeros((4,4)),2)
        self.heavy_seed=np.array([np.linalg.det(heavy[[i for i in range(4) if m>>i&1],2:]) for m in self.heavy_masks])
        self.seed=np.zeros(self.shape_neutral,complex);self.seed[0,0,:,0,:]=np.outer(self.heavy_seed,self.light_seed)
        self.seed=self.seed.reshape(-1);self.seed/=np.linalg.norm(self.seed)
        self.energy=float(np.vdot(self.seed,self.Hn@self.seed).real)
        # One SAME scalar shift, not separate neutral/charged minima.
        self.Hn.offset=self.energy;self.Hc.offset=self.energy
        self.floor=self.form_floor()-self.energy
        excess=-self.floor;require(excess>0,'lawful native positive trial excess')
        self.window_width=4*excess
        self.state,self.window_norm=self.prepare_window()
        self.source_state=self.B@self.state;self.source_norm=float(np.linalg.norm(self.source_state))
        require(self.source_norm>1e-5,'actual new normalized source nonzero')
        self.charged_state=self.source_state/self.source_norm
        self.xi=self.state;self.trial_n=self.seed
        self.shape_n=(3,3,3,3,6,140);self.shape_c=(3,3,3,3,6,448)
        self.heavy_spin6=exterior(self.Sz,2)[0]
        self.light_spin8=np.kron(np.eye(2),self.Sz)
        self.orbital8=np.kron(np.array([[0,-1j],[1j,0]]),np.eye(4))
        self.Kn=self.Cn;self.Kc=self.Cc
        gs=float(np.trace(self.frame[1]@self.frame[1]).real/4)
        self.rest_spin_current=-1j*(self.frame[1]@self.frame[2]-self.frame[2]@self.frame[1])/(4*gs)
        require(np.linalg.norm(self.rest_spin_current-self.Sz)<1e-12,'rest spin from actual Phi current commutator')

    def prepare_window(self):
        return actual_window(self.Hn,self.seed,self.floor,self.window_width,60)

    def lift_root(self,onebody8,charged=False):
        sector='charged' if charged else 'neutral'
        size=224 if charged else 70;M=sparse.csr_matrix((size,size),dtype=complex)
        for i,j in itertools.product(range(8),repeat=2):
            if abs(onebody8[i,j])>1e-14:M+=onebody8[i,j]*self._car(sector,i,j)
        return sparse.kron(sparse.eye(2),M,format='csr')

    def axis_operator(self,axis,matrix,charged=False):
        shape=self.shape_c if charged else self.shape_n
        return TensorHamiltonian(shape,[(1,[(axis,matrix)])])

    def _project_actual_coordinates(self):
        Q=self.Q2;n=len(self.points);V=self.law.volume
        labels=np.array([np.eye(4),*(SELECTED[0]@SELECTED[j] for j in range(1,4))])
        quartet=np.einsum('a,aij->ij',BETA,labels)/2
        # The whole range-two link product is computed before projection.
        for ip,ic in self.pc_points:
            p=self.points[ip]*self.scales[0];c=self.points[ic]*self.scales[1]
            phi=np.tile(self.frame,(V,1,1,1));phi[:,0]+=p[0]*SELECTED[1]+p[1]*SELECTED[2]
            links=np.tile(np.eye(4,dtype=complex),(V,3,1,1))
            U=expm(1j*(c[0]*SELECTED[1]+c[1]*SELECTED[2])/math.sqrt(8));links[:,2]=U
            data=[]
            for disp,scalar in self.law.paths:
                axis=int(np.flatnonzero(disp)[0]);length=int(disp[axis]);u=np.linalg.matrix_power(U,length) if axis==2 else np.eye(4)
                f=np.einsum('m,mij->ij',scalar,phi[0]);data.append((axis,length,(f@u+u@f)/2))
            self._endpoint_data.append(data)
            hh=self.heavy_fold.conj().T@self.heavy.onebody(phi,links,np.zeros((V,3)))@self.heavy_fold
            self._heavy.append(exterior(-self.Lambda*hh,2)[0].toarray())
            edges=[(x,int(self.law.target[x,k])) for x in range(V) for k in range(3)]
            elastic,alignment=local_potential(phi,edges,[links[x,k] for x in range(V) for k in range(3)])
            self._scalar.append(self.Lambda*float(elastic)+self.Lambda**1.25*float(alignment))
            frame=frame_chart(phi[0])[0];word=U@U
            mark=centre_scalar(frame.conj().T@word@frame,18.,self.law.generators)[0]*frame@quartet@frame.conj().T
            self._source.append(np.vstack([mark,np.zeros_like(mark)]))
        self.heavy_matrix=np.einsum('qa,qij,qb->aibj',Q,np.array(self._heavy),Q).reshape(54,54)
        self.scalar_matrix=Q.T@(np.array(self._scalar)[:,None]*Q)
        self.source_coeff=np.einsum('qa,qij,qb->abij',Q,np.array(self._source),Q)
        self.onsite_coeff=np.zeros((9,9,4,4),complex)
        for k,(ip,ic) in enumerate(self.pc_points):
            p=self.points[ip]*self.scales[0]
            onsite=self.law.onsite*(self.frame[0]+p[0]*SELECTED[1]+p[1]*SELECTED[2])
            self.onsite_coeff+=np.outer(Q[k],Q[k])[:,:,None,None]*onsite
        # Project actual Peierls factors and their ordered first/contact jets.
        self.W={}
        for axis,length,_ in self._endpoint_data[0]:
            ys=np.arange(V)
            for _ in range(length):ys=self.law.target[ys,axis]
            for derivative in (0,1,2):
                vals=[]
                for iph,ih in self.pc_points:
                    em=np.einsum('a,axj->xj',self.points[iph]*self.scales[2],self.photon_profiles)+np.einsum('a,axj->xj',self.points[ih]*self.scales[3],self.harmonic_profiles)
                    phase=np.zeros(V);insert=np.zeros(V);z=np.arange(V)
                    for _ in range(length):phase+=em[z,axis];insert+=self.photon_profiles[0,z,axis];z=self.law.target[z,axis]
                    weights=np.exp(1j*phase)*(1j*insert)**derivative
                    vals.append(np.einsum('xa,x,xb->ab',self.orbitals,weights,self.orbitals[ys]))
                self.W[axis,length,derivative]=np.einsum('qa,qij,qb->abij',Q,np.array(vals),Q)
        self.endpoint_coeff=[np.einsum('qa,qij,qb->abij',Q,np.array([d[k][2] for d in self._endpoint_data]),Q) for k in range(len(self._endpoint_data[0]))]
        self.photon_scalar=np.zeros((9,9),complex)
        vals=[]
        for iph,ih in self.pc_points:
            em=np.einsum('a,axj->xj',self.points[iph]*self.scales[2],self.photon_profiles)
            curl=np.zeros((V,3))
            for a,b,c in ((0,1,2),(1,2,0),(2,0,1)):curl[:,a]=em[self.law.target[:,b],c]-em[:,c]-em[self.law.target[:,c],b]+em[:,b]
            vals.append(self.K/2*float(np.sum(curl*curl)))
        self.photon_scalar=Q.T@(np.array(vals)[:,None]*Q)

    def _car(self,sector,i,j):
        M=self.light_operators[4 if sector=='neutral' else 5][i][j]
        return M if sector=='neutral' else sparse.kron(sparse.eye(4),M,format='csr')

    def hamiltonian(self,sector='neutral',derivative=0):
        terms=[];shape=self.shape_neutral if sector=='neutral' else self.shape_charged
        def add(C,F):
            if abs(C)>1e-14:terms.append((C,F))
        if derivative==0:
            for orbital in range(2):
                for i,j in itertools.product(range(4),repeat=2):
                    A=self.onsite_coeff[:,:,i,j]
                    if np.linalg.norm(A)>1e-13:add(1,[(0,A),(4,self._car(sector,4*orbital+i,4*orbital+j))])
        for k,(axis,length,_) in enumerate(self._endpoint_data[0]):
            F=self.endpoint_coeff[k];W=self.W[axis,length,derivative]
            for a,b in itertools.product(range(2),repeat=2):
                Wab=W[:,:,a,b]
                if np.linalg.norm(Wab)<1e-13:continue
                for i,j in itertools.product(range(4),repeat=2):
                    Fij=F[:,:,i,j]
                    if np.linalg.norm(Fij)<1e-13:continue
                    op=self._car(sector,4*a+i,4*b+j)
                    add(1,[(0,Fij),(1,Wab),(3,self.loop),(4,op)])
                    add(1,[(0,Fij.conj().T),(1,Wab.conj().T),(3,self.loop),(4,op.conj().T)])
        if derivative==0:
            # heavy matrix spans axes PC9 and heavy6, implemented as its exact
            # elementary heavy blocks, not multiplied after compression.
            HM=self.heavy_matrix.reshape(9,6,9,6)
            for i,j in itertools.product(range(6),repeat=2):
                A=HM[:,i,:,j]
                if np.linalg.norm(A)>1e-13:
                    e=np.zeros((6,6));e[i,j]=1;add(1,[(0,A),(2,e)])
            add(1,[(0,self.scalar_matrix)]);add(1,[(1,self.photon_scalar)])
            for axis,coeff in ((0,self.chi/(2*self.scales[0]**2)),(0,.3*self.chi/self.scales[1]**2),(1,self.ue/self.scales[2]**2),(1,self.ue/(3*self.scales[3]**2))):
                inner=0 if coeff in (self.chi/(2*self.scales[0]**2),self.ue/self.scales[2]**2) else 1
                A=np.kron(self.pair_p2,np.eye(3)) if inner==0 else np.kron(np.eye(3),self.pair_p2)
                add(coeff,[(axis,A)])
            add(1,[(3,self.aux_penalty*(np.eye(2)-self.loop)/2+self.beta*self.defect)])
            if sector=='charged':add(self.ue,[])
        return TensorHamiltonian(shape,terms)

    def source_operator(self):
        index={m:k for k,m in enumerate(self.charged_masks)};matrices={}
        for i,r in itertools.product(range(8),range(4)):
            rows=[];cols=[];vals=[];j=8+r
            for k,m in enumerate(self.neutral_masks):
                m1=m^(1<<j)
                if m1>>i&1:continue
                s=(-1)**((m&((1<<j)-1)).bit_count()+(m1&((1<<i)-1)).bit_count())
                rows.append(index[m1|(1<<i)]);cols.append(k);vals.append(s)
            matrices[i,r]=sparse.csr_matrix((vals,(rows,cols)),shape=(224,70))
        terms=[(self.source_coeff[:,:,i,r],matrices[i,r]) for i,r in matrices if np.linalg.norm(self.source_coeff[:,:,i,r])>1e-13]
        def forward(value):
            x=np.asarray(value).reshape(self.shape_neutral);y=np.zeros(self.shape_charged,complex)
            for A,M in terms:
                z=np.einsum('ab,bpqri->apqri',A,x,optimize=True)
                y+=np.asarray((M@z.reshape(-1,70).T).T).reshape(self.shape_charged)
            return y.reshape(-1)
        def adjoint(value):
            x=np.asarray(value).reshape(self.shape_charged);y=np.zeros(self.shape_neutral,complex)
            for A,M in terms:
                z=np.einsum('ab,bpqri->apqri',A.conj().T,x,optimize=True)
                y+=np.asarray((M.conj().T@z.reshape(-1,224).T).T).reshape(self.shape_neutral)
            return y.reshape(-1)
        return LinearOperator((math.prod(self.shape_charged),math.prod(self.shape_neutral)),matvec=forward,rmatvec=adjoint,dtype=complex)

    def form_floor(self):
        # Direct finite-form lower bound: minimum scalar+heavy block at every
        # cubature node, kinetic positive, light norm bounded by endpoint l1.
        floor=min(float(s+np.linalg.eigvalsh(h)[0]) for s,h in zip(self._scalar,self._heavy))
        light=8*(np.linalg.norm(self.onsite_coeff.reshape(81,16),axis=1).sum()+2*sum(np.linalg.norm(F.reshape(81,16),axis=1).sum() for F in self.endpoint_coeff))
        return floor-light-abs(self.beta)-1e-7

    def metadata(self):
        return dict(identity=IDENTITY,N=3,neutral_dimension=math.prod(self.shape_neutral),charged_dimension=math.prod(self.shape_charged),tensor_shape_neutral=self.shape_neutral,tensor_shape_charged=self.shape_charged,pair_labels=self.pair_labels,pair_axes=['Phi','colour','coexact','harmonic'],orbital_profile='x exp(-r²/(2*.65²)),y exp(-r²/(2*.65²))',neutral_bulk=4,neutral_ref=4,charged_bulk=5,charged_ref=3,all_ref_holes=4,auxiliary_states=2,heavy_number=2,heavy_modes=4,native_root='actual loop+defect two-sector Galerkin; not full12edge Pauli Hilbert space',quadrature='joint2D each pair; original range2 product and nonlinearSU4/Peierls BEFORE projection',source='pureCoulomb original rawquartet Phi/SU4 chart mark; native chart approximant not legalN fullHaar evaluation',actual_window_norm=self.window_norm,source_norm=self.source_norm,trial_energy=self.energy,lawful_native_floor_shifted=self.floor,window_width=self.window_width,full_field_ground_claim=False)


def native(time_value=.2):
    start=time.perf_counter();m=PreparedSpatialQuantumSlice();v=m.state
    rng=np.random.default_rng(222);u=rng.normal(size=len(v))+1j*rng.normal(size=len(v));u/=np.linalg.norm(u)
    herm=abs(np.vdot(u,m.Hn@v)-np.vdot(m.Hn@u,v));require(herm<1e-7,'actual spatialH Hermiticity')
    evolved=expm_multiply(-1j*time_value*m.Hn,v,traceA=-1j*time_value*m.Hn.trace_value())
    charged=expm_multiply(-1j*time_value*m.Hc,m.charged_state,traceA=-1j*time_value*m.Hc.trace_value())
    norm=abs(np.linalg.norm(evolved)-1);require(norm<1e-7,'actual spatial newstate flight norm')
    relative=float(np.linalg.norm(charged-m.B@evolved/m.source_norm))
    current=float(np.linalg.norm(m.Jc@m.charged_state));contact=float(np.linalg.norm(m.Cc@m.charged_state))
    return dict(metadata=m.metadata(),time=time_value,hermiticity_error=herm,norm_error=norm,actual_neutral_right_source_difference=relative,current_norm=current,contact_norm=contact,actual_seconds=time.perf_counter()-start,spatial_closed_loop_owned_by='ClosedSpatialRotation',legalN_proof_not_inferred=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--time',type=float,default=.2);a=p.parse_args()
    print(json.dumps(native(a.time),indent=2))
if __name__=='__main__':main()
