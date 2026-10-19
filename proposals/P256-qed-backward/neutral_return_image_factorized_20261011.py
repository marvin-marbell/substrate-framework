"""Exact labelled CAR Gram API for the complete64-copy neutral image.
Density inputs are normalized selected-slot reduced densities, not product
approximations. Functions are on-demand and never allocate a two-cell space.
"""
from __future__ import annotations
from collections import Counter
from itertools import product
import sympy as s
from typing import cast

# A feature is None (identity) or ((physical_spin_slot,Dirac),(...)).
# A density callback receives the ordered distinct physical slot labels; it
# must return rho1 (5x5) or rho2 (25x25), including actual role correlations.
def feature_word(feature, dagger=False):
    if feature is None:
        return []
    i,k=feature
    return [(k,True),(i,False)] if dagger else [(i,True),(k,False)]

def local_moment(word, ann, W5, density):
    """Exact word expectation using at most TWO selected physical slots."""
    if not word:
        return s.Integer(1)
    balance=Counter()
    for (slot,_),creation in word:
        balance[slot]+=1 if creation else -1
    if any(balance.values()):
        return s.Integer(0)
    slots=sorted(balance)
    if len(slots)>2:
        raise ValueError('This interface accepts bilinear Gram words, at most two balanced slots')
    # Reordering odd CAR factors into slot blocks retains the contact sign.
    inversions=sum(word[i][0][0]>word[j][0][0] for i in range(len(word)) for j in range(i+1,len(word)))
    block=[]
    for slot in slots:
        O=s.eye(16)
        for (owner,p),creation in word:
            if owner==slot:
                O=O*(ann[p].T if creation else ann[p])
        block.append(W5.conjugate().T*O*W5)
    Op=cast(s.MatrixBase, block[0] if len(block)==1 else s.kronecker_product(*block))
    rho=s.Matrix(density(tuple(slots)))
    if rho.shape!=Op.shape:
        raise ValueError('Selected-slot density shape does not match exact CAR word')
    return s.simplify((-1)**inversions*s.trace(Op*rho))

def local_gram(left, right, ann, W5, density):
    return local_moment(feature_word(left,True)+feature_word(right),ann,W5,density)

def factorized_gram_entry(left_term, right_term, ann, W5, densityA, densityB):
    """One coherent term-pair contribution; summation belongs to consumer.

    A term is (coefficient, featureA, featureB). No representation directsum
    is implied by these features, and all cross-history entries are retained.
    """
    c,A,B=left_term;d,C,D=right_term
    return s.conjugate(c)*d*local_gram(A,C,ann,W5,densityA)*local_gram(B,D,ann,W5,densityB)

def return_terms(M, copies=64):
    """Stream EXACT C=TTdag+TdagT contact+exchange on all labelled copies.

    M is the actual4x4 physical-spin/endpoint-Dirac matrix in the shared
    source frame, not a receiver path. Default64=active1+aux63. Physical
    role of slot s is active iff s<2, otherwise auxiliary. Never evaluate
    the returned terms as independently orthogonal branches.
    """
    M=s.Matrix(M)
    if M.shape!=(4,4):
        raise ValueError('Actual spin2 x endpointDirac2 matrix must be4x4')
    AA=M*M.conjugate().T;BB=M.conjugate().T*M
    def amode(f,i):return (2*f+i//2,2+i%2)
    def bmode(f,i):return (2*f+i//2,i%2)
    for f in range(copies):
        for i,k in product(range(4),repeat=2):
            if AA[i,k]!=0:yield (AA[i,k],(amode(f,i),amode(f,k)),None)
            if BB[i,k]!=0:yield (BB[i,k],None,(bmode(f,i),bmode(f,k)))
    entries=[(i,j,M[i,j]) for i,j in product(range(4),repeat=2) if M[i,j]!=0]
    for f,g in product(range(copies),repeat=2):
        for i,j,z in entries:
            for k,l,w in entries:
                yield (-2*z*s.conjugate(w),(amode(f,i),amode(g,k)),(bmode(g,l),bmode(f,j)))

def matter_commutator(feature, profile_h):
    """Exact full4Dirac mixed-role free commutator, no rho shortcut."""
    if feature is None:
        return
    (si,p),(sk,q)=feature
    hi=s.Matrix(profile_h('active' if si<2 else 'auxiliary'))
    hk=s.Matrix(profile_h('active' if sk<2 else 'auxiliary'))
    for r in range(4):
        if hi[r,p]!=0:yield (hi[r,p],((si,r),(sk,q)))
        if hk[q,r]!=0:yield (-cast(s.Expr,hk[q,r]),((si,p),(sk,r)))
