#!/usr/bin/env python3
"""Literal Ref4->3 / Ref0->1 CAR views on one prepared spatial field state."""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,os,time
from pathlib import Path
if __name__=='__main__':
 for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import expm
from prepared_neutral_spatial_gauss_well import PreparedSpatialGaussQuantumSlice
from prepared_neutral_spatial_em_well import gaussian_weyl
from prepared_neutral_spatial_well import creation_number_matrix
from prepared_neutral_full_field import BETA,require
from noncompact_screened_gauss_dressing import frame_chart
from sea_selection import SELECTED
GAMMA5=SELECTED[0]@SELECTED[1]@SELECTED[2]@SELECTED[3]
LABELS=np.array([np.eye(4),*(SELECTED[0]@SELECTED[j] for j in range(1,4))])/2


def hodge(masks,n):
 index={m:i for i,m in enumerate(masks)};out=np.zeros((len(masks),len(masks)),complex)
 for col,mask in enumerate(masks):
  sign=1
  for bit in reversed(range(n)):
   sign*=(-1)**((mask&((1<<bit)-1)).bit_count());mask^=1<<bit
  out[index[mask],col]=sign
 return out


def exterior_unitary(U,masks):
 indices=[[i for i in range(len(U)) if mask>>i&1] for mask in masks]
 return np.array([[np.linalg.det(U[np.ix_(row,col)]) for col in indices] for row in indices])


class PreparedSpatialChargeTransfer(PreparedSpatialGaussQuantumSlice):
 """A charge view, never a new Hamiltonian/state preparation or plus-P echo."""
 def __init__(self,prepared:PreparedSpatialGaussQuantumSlice,charge:int):
  require(charge in (-1,1),'literalchargechoice')
  self.__dict__=prepared.__dict__.copy();self.charge=charge;self.preparation_owner=prepared
  self.gamma5=GAMMA5
  if charge==1:return
  # Actual full PH sends Ref4 to Ref0. Normal Phi modes are Gamma5-odd,
  # colour modes reverse; both quantum CAR seas are transformed, not relabelled.
  parity=np.kron(np.eye(3),np.diag([1,-1,-1]))
  light=exterior_unitary(np.kron(np.eye(2),GAMMA5),self.bulk_masks[4])@hodge(self.bulk_masks[4],8)
  heavy_masks=prepared.heavy_masks
  heavy=exterior_unitary(GAMMA5,heavy_masks)@hodge(heavy_masks,4)
  v=prepared.xi.conj().reshape(self.shape_neutral)
  for axis,A in ((0,parity),(2,heavy),(4,light)):
   move=np.moveaxis(v,axis,0);shape=move.shape
   v=np.moveaxis((A@move.reshape(shape[0],-1)).reshape(shape),0,axis)
  self.state=v.reshape(-1);self.xi=self.state
  overlap=np.vdot(prepared.xi,self.xi)
  self.neutral_C_overlap=float(abs(overlap));require(abs(np.linalg.norm(self.xi)-1)<1e-10,'literalholepreparednorm')
  # Direct bulk3/ref1 CAR and actual qref=Nref, not old Ref3 reused.
  self.bulk_masks=dict(prepared.bulk_masks);self.bulk_masks[3]=[sum(1<<i for i in c) for c in itertools.combinations(range(8),3)]
  self.light_operators=dict(prepared.light_operators)
  self.light_operators[3]=[[creation_number_matrix(self.bulk_masks[3],i,j) for j in range(8)] for i in range(8)]
  self.coulomb_blocks=dict(prepared.coulomb_blocks)
  rho=[];I=np.eye(56)
  for x in range(27):
   A=np.kron(np.outer(self.orbitals[x],self.orbitals[x]),np.eye(4));r=np.zeros((56,56),complex)
   for i,j in itertools.product(range(8),repeat=2):r+=A[i,j]*self.light_operators[3][i][j].toarray()
   r-=2*self.retained_density_diagonal[x]*I
   if x==self.reference_site:r+=I
   rho.append(r)
  e,U=np.linalg.eigh(self.coulomb_green);positive=e>1e-9
  modes=np.einsum('xm,xij->mij',U[:,positive]*np.sqrt(e[positive]),np.array(rho))
  C=sum((r@r for r in modes),np.zeros((56,56),complex))
  M=self.complement_charge_matrix;A=np.kron(M,SELECTED[0]);correction=2*np.trace(M)*I
  for i,j in itertools.product(range(8),repeat=2):correction=correction+A[i,j]*self.light_operators[3][i][j].toarray()
  C=self.ue*(C+correction)
  require(float(np.linalg.eigvalsh(C)[0])>-1e-9,'literalholepositiveCoulomb')
  self.coulomb_blocks[5]=sparse.kron(sparse.eye(4),sparse.csr_matrix(C),format='csr')
  self.Hc=self.hamiltonian('charged');self.Hc.offset=prepared.Hn.offset
  self.Jc=self.hamiltonian('charged',1);self.Cc=self.hamiltonian('charged',2);self.Kc=self.Cc
  self.B=self.literal_source_operator();self.source_state=self.B@self.xi
  self.source_norm=float(np.linalg.norm(self.source_state));require(self.source_norm>1e-8,'actualholecreatornorm')
  self.charged_state=self.source_state/self.source_norm

 def _car(self,sector,i,j):
  if self.charge==-1 and sector!='neutral':return sparse.kron(sparse.eye(4),self.light_operators[3][i][j],format='csr')
  return super()._car(sector,i,j)

 def literal_source_operator(self,profile=None,quartet=None):
  if profile is None:profile=self.orbitals[:,0]
  profile=np.asarray(profile,complex);quartet=BETA if quartet is None else np.asarray(quartet,complex)
  Qraw=np.einsum('a,aij->ij',BETA,LABELS);Qtarget=np.einsum('a,aij->ij',quartet,LABELS)
  require(quartet.shape==(4,),'rawquartetdimension')
  native_phi=self.frame.copy();by_length={}
  for length in range(3):
   values=[]
   for ip,ic in self.pc_points:
    phi=native_phi.copy();p=self.points[ip]*self.scales[0];c=self.points[ic]*self.scales[1]
    phi[0]+=p[0]*SELECTED[1]+p[1]*SELECTED[2]
    U=expm(1j*(c[0]*SELECTED[1]+c[1]*SELECTED[2])/math.sqrt(8));word=np.linalg.matrix_power(U,length)
    if self.charge==-1:
     phi=-np.einsum('ij,ajk,kl->ail',GAMMA5,phi,GAMMA5);word=GAMMA5@word@GAMMA5
    mark=self.source_coordinate_mark(phi,word,phi,np.zeros(3),np.zeros(3));Fr=frame_chart(phi)[0]
    # Exact linearity through the existing literal source, no second mark law.
    mark=mark@Fr@np.linalg.solve(Qraw,Qtarget)@Fr.conj().T
    if self.charge==-1:mark=-GAMMA5@mark.conj().T@GAMMA5
    values.append(mark)
   by_length[length]=np.einsum('qa,qij,qb->abij',self.Q2,np.array(values),self.Q2)
  incoming=self.bulk_masks[4]
  out=[m|(sum(1<<r for r in range(8,12))^(1<<r)) for r in range(8,12) for m in self.bulk_masks[5]] if self.charge==1 else [m|(1<<r) for r in range(8,12) for m in self.bulk_masks[3]]
  index={m:i for i,m in enumerate(out)};creators={}
  for i,r in itertools.product(range(8),range(4)):
   rows=[];cols=[];signs=[];ref=8+r
   for col,b in enumerate(incoming):
    mask=b|(sum(1<<k for k in range(8,12)) if self.charge==1 else 0)
    first,second=(ref,i) if self.charge==1 else (i,ref)
    if not(mask>>first&1):continue
    sign=(-1)**((mask&((1<<first)-1)).bit_count());mask^=1<<first
    if mask>>second&1:continue
    sign*=(-1)**((mask&((1<<second)-1)).bit_count());mask|=1<<second
    rows.append(index[mask]);cols.append(col);signs.append(sign)
   creators[i,r]=sparse.csr_matrix((signs,(rows,cols)),shape=(224,70))
  terms=[];ref=self.law.sites[self.reference_site]
  for orbital,length in itertools.product(range(2),range(3)):
   E=np.zeros((56,56),complex)
   for x in range(27):
    d=self.law.sites[x]-ref
    if int(d[2]%3)!=length:continue
    w=np.zeros(55);w[52:]=-self.charge*d/3
    eta=profile[x] if self.charge==1 else profile[x].conjugate()
    E+=eta*self.orbitals[x,orbital]*gaussian_weyl(w)
   for i,r in itertools.product(range(4),repeat=2):
    F=by_length[length][:,:,i,r] if self.charge==1 else by_length[length][:,:,r,i]
    if np.linalg.norm(F)>1e-13 and np.linalg.norm(E)>1e-13:terms.append((F,sparse.csr_matrix(E),creators[4*orbital+i,r]))
  def apply(value,adjoint=False):
   si=self.shape_charged if adjoint else self.shape_neutral;so=self.shape_neutral if adjoint else self.shape_charged
   v=np.asarray(value).reshape(si);y=np.zeros(so,complex)
   sparse_seed_support=not adjoint and np.count_nonzero(v[1:])==0 and np.count_nonzero(v[0,1:])==0
   for F,E,M in terms:
    if adjoint:F,E,M=F.conj().T,E.conj().T,M.conj().T
    # Exact sparse-support path for this once-prepared plateau state.
    if sparse_seed_support:
     root=(M@v[0,0].reshape(12,70).T).T.reshape(6,2,224)
     y+=np.einsum('a,b,hri->abhri',F[:,0],E.getcol(0).toarray().ravel(),root,optimize=True)
    else:
     z=np.einsum('ab,bpqri->apqri',F,v,optimize=True);moved=np.moveaxis(z,1,0);dims=moved.shape
     z=np.moveaxis(np.asarray(E@moved.reshape(56,-1)).reshape(dims),0,1)
     y+=np.asarray((M@z.reshape(-1,si[-1]).T).T).reshape(so)
   return y.reshape(-1)
  B=LinearOperator((math.prod(self.shape_charged),math.prod(self.shape_neutral)),matvec=apply,rmatvec=lambda x:apply(x,True),dtype=complex);B.source_terms=terms
  return B

 def creator_gram(self,profile=None):
  vectors=[self.literal_source_operator(profile,quartet=np.eye(4)[k])@self.xi for k in range(4)]
  G=np.array([[np.vdot(a,b) for b in vectors] for a in vectors])
  c=BETA if self.charge==1 else BETA.conj();raw=sum((c[k]*vectors[k] for k in range(4)),np.zeros_like(vectors[0]))
  actual=(self.preparation_owner.source_state if self.charge==1 else self.source_state) if profile is None else self.literal_source_operator(profile)@self.xi
  err=float(np.linalg.norm(raw-actual));require(err<1e-11,'actualraw-sourceGramlinearity')
  return dict(G=G,eigenvalues=np.linalg.eigvalsh(G),raw_coefficients=c,raw_squared_norm=float(np.vdot(c,G@c).real),actual_squared_norm=float(np.linalg.norm(actual)**2),linearity_error=err,not_neutral_Markov=True,input_whitened=False)

 def source_operator(self,profile=None):
  return self.literal_source_operator(profile)

 def metadata(self):
  d=super().metadata()
  d.update(charge=self.charge,neutral_ref=4 if self.charge==1 else 0,charged_ref=3 if self.charge==1 else 1,
   charged_bulk=5 if self.charge==1 else 3,charge_view_not_new_preparation=True,
   source_minus='literalCAR cRefdagger*c_bulk with -Gamma5 M(sigmaPhi,sigmaU)dagger Gamma5; oppositeharmonicword/profileconjugate; rawcoeffconjugation specified, notconjugateplusprobability')
  return d


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state-capsule',type=Path,required=True);a=p.parse_args();start=time.perf_counter()
 m=PreparedSpatialGaussQuantumSlice(state_capsule=a.state_capsule);results=[]
 for charge in (1,-1):
  view=PreparedSpatialChargeTransfer(m,charge);G=view.creator_gram()
  results.append(dict(charge=charge,neutral_ref=4 if charge==1 else 0,charged_ref=3 if charge==1 else 1,charged_bulk=5 if charge==1 else 3,reference_deviation=-1 if charge==1 else 1,all_Ref=4,creator=G,source_norm=view.source_norm,current_norm=float(np.linalg.norm(view.Jc@view.charged_state)),contact_norm=float(np.linalg.norm(view.Kc@view.charged_state)),neutral_C_overlap=getattr(view,'neutral_C_overlap',None),neutral_prepared_again=False))
 from original_charged_principal import portable
 print(json.dumps(portable(dict(state_sha256=hashlib.sha256(a.state_capsule.read_bytes()).hexdigest(),charge_views=results,elapsed_seconds=time.perf_counter()-start,scope='literalnative CAR3/5+Ref0/4 onSAMEoncepreparedfields; notlegalN creatorGram proof orminusBornconjugation')),indent=2))
if __name__=='__main__':main()
