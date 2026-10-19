"""Exact finite global J=L_left+S representation, not a spin energy penalty."""
from __future__ import annotations
from functools import lru_cache
from math import sqrt
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan
from rich_composite_geometry_20261015_model import add,inner


def spin_multiplets(basis,primitive):
    n=len(basis.vectors);rows=[];columns=[];values=[];raw_closure=0.;max_raw_closure=0.
    for j,packet in enumerate(basis.vectors):
        image=add((1,primitive.spin(packet,0)),(1j,primitive.spin(packet,1)))
        coefficients=basis.coordinates(image)
        missing=float(inner(image,image).real-sum(abs(v)**2 for v in coefficients.values()))
        raw_closure+=missing;max_raw_closure=max(max_raw_closure,abs(missing))
        for i,value in coefficients.items():rows.append(i);columns.append(j);values.append(value)
    raising=sparse.csr_matrix((values,(rows,columns)),shape=(n,n));lowering=raising.conj().T
    multiplets=[];members_count=0;max_highest=0.;max_lower=0.
    for lab,indices in basis.blocks.items():
        m2=lab[-1]
        if m2<0:continue
        upper=basis.blocks.get(lab[:-1]+(m2+2,),[])
        block=raising[upper][:,indices].toarray()
        gram=block.conj().T@block
        eigenvalues,eigenvectors=eigh(gram)
        kernel=eigenvectors[:,eigenvalues<1e-9]
        for k in range(kernel.shape[1]):
            v=sparse.csc_matrix((kernel[:,k],(indices,np.zeros(len(indices),int))),shape=(n,1))
            max_highest=max(max_highest,float(sparse.linalg.norm(raising@v)))
            members={m2:v};s=m2/2
            for current in range(m2,-m2,-2):
                m=current/2;v=(lowering@v/sqrt((s+m)*(s-m+1))).tocsc()
                members[current-2]=v
                max_lower=max(max_lower,abs(float((v.conj().T@v).toarray()[0,0].real)-1))
            multiplets.append({'twice_S':m2,'counts':lab[:-1],'members':members});members_count+=m2+1
    sz=sparse.diags([lab[-1]/2 for lab in basis.labels],format='csr')
    receipt={'body_dimension':n,'reconstructed_spin_member_count':members_count,
        'multiplet_count':len(multiplets),'raw_raising_omitted_norm2_signed_sum':raw_closure,
        'max_raw_raising_omitted_norm2_absolute':max_raw_closure,'max_highest_weight_error':max_highest,
        'max_lowering_norm_error':max_lower,'SU2_error':float(sparse.linalg.norm(raising@lowering-lowering@raising-2*sz)),
        'scope':'Highest-weight kernels per exact physical-count/Sz block, all lowered members; no fitted S2 energy or deletion of charged spinhalf/threehalf.'}
    if members_count!=n:raise ArithmeticError(f'Incomplete actual spin closure: {receipt}')
    return multiplets,receipt


@lru_cache(None)
def cg(s2,m2,l,mr,j2,M2):
    return float(clebsch_gordan(Rational(s2,2),l,Rational(j2,2),Rational(m2,2),mr,Rational(M2,2)))


def coupled(basis,multiplets,twice_J,twice_M=None):
    M2=twice_J if twice_M is None else twice_M
    # Spherical lab vector eigenbasis, phases fixed by L+|m>=sqrt(2)|m+1>.
    spherical={1:np.array([-1,-1j,0])/sqrt(2),0:np.array([0,0,1.]),-1:np.array([1,-1j,0])/sqrt(2)}
    rows=[[] for _ in range(10)];cols=[[] for _ in range(10)];values=[[] for _ in range(10)]
    labels=[];column=0
    for alpha,multiplet in enumerate(multiplets):
        s2=multiplet['twice_S']
        for l in (0,1):
            if not abs(s2-2*l)<=twice_J<=s2+2*l:continue
            for a in (range(3) if l else range(1)):
                for m2,member in multiplet['members'].items():
                    difference=M2-m2
                    if difference%2:continue
                    mr=difference//2
                    if abs(mr)>l:continue
                    coefficient=cg(s2,m2,l,mr,twice_J,M2)
                    if abs(coefficient)<1e-14:continue
                    rr=[(0,1.)] if not l else [(1+3*i+a,spherical[mr][i]) for i in range(3) if spherical[mr][i]!=0]
                    indices=member.tocoo()
                    for r,angular in rr:
                        rows[r].extend(indices.row);cols[r].extend([column]*len(indices.data));values[r].extend(coefficient*angular*indices.data)
                labels.append({'counts':multiplet['counts'],'twice_S':s2,'l':l,'right_axis':a,'multiplet':alpha});column+=1
    u=[sparse.csr_matrix((values[r],(rows[r],cols[r])),shape=(len(basis.vectors),column)) for r in range(10)]
    gram=sum((v.conj().T@v for v in u),sparse.csr_matrix((column,column),dtype=complex))
    return {'twice_J':twice_J,'twice_M':M2,'U':u,'labels':labels,'dimension':column,
        'isometry_error':float(sparse.linalg.norm(gram-sparse.eye(column))),
        'meaning':'Actual Clebsch-Gordan isometry into complete physical S x SO3(j0+j1); other M reconstructed by same multiplet, not copied energies from another law.'}


def project_product(left,body,rotation,right=None):
    right=left if right is None else right
    out=sparse.csr_matrix((left['dimension'],right['dimension']),dtype=complex)
    rr=sparse.coo_matrix(rotation)
    for i,j,value in zip(rr.row,rr.col,rr.data):
        if left['U'][i].nnz and right['U'][j].nnz:
            out+=value*(left['U'][i].conj().T@(body@right['U'][j]))
    out.eliminate_zeros()
    return out


def right_angular(rotor):
    result=[]
    for axis in range(3):
        small=np.zeros((10,10),complex)
        from rich_colour_composite_20261014_model import epsilon
        generator=np.array([[-1j*epsilon(axis,i,j) for j in range(3)] for i in range(3)])
        small[1:,1:]=-np.kron(np.eye(3),generator)
        result.append(sparse.csr_matrix(small))
    return result
