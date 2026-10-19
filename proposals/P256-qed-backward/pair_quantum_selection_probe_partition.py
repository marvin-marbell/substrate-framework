#!/usr/bin/env python3
"""Integer Pauli traces with exact actual-native rational source coefficients."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import numpy as np
import sympy as sp
import pair_quantum_selection_probe_symbolic as a
P=[np.eye(2,dtype=complex),np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1,-1]).astype(complex)]
def split(x): return np.rint(x.real).astype(np.int64),np.rint(x.imag).astype(np.int64)
def mul(x,y): return x[0]@y[0]-x[1]@y[1],x[0]@y[1]+x[1]@y[0]
basis=[np.kron(P[int(label[0])],P[int(label[1])]) for label in a.labels]
changes=np.zeros((3,60,8,8),complex)
for n,name in enumerate(a.names):
    B=basis[a.labels.index(name[2:])]
    if name[1]=='0':
        changes[0,n]=np.kron(P[0] if name[0]=='s' else P[3],B)
        if name[0]=='s': changes[1,n]=-np.kron(P[1],B)
    elif name[0]=='s': changes[2,n]=np.kron(P[2],B)
D=split(changes)
projections={}
for i in range(2):
    mask4=np.diag([int(j//2==i) for j in range(4)])
    mask=np.kron(P[0],mask4)
    W=np.kron(P[1],np.diag([(-1)**j*int(j//2==i) for j in range(4)]))
    A=np.kron(P[0],np.diag([(-1)**j*int(j//2==i) for j in range(4)]))
    Y=np.kron(P[2],np.kron(np.diag([int(j==i) for j in range(2)]),P[1]))
    for sign in (-1,1):
        for shift in (-1,1):
            projections[i,sign,shift]=[split(mask+shift*W),split(sign*A@(mask+shift*W)),split(sign*Y@(mask+shift*W))]
# Inputs have magnitude<=2 and1, so each product entry is bounded by16;
# a trace contraction has absolute value<=2*64*16^2, far below int64.
terms={}
for i in range(2):
 for j in range(2):
  for re in (-1,1):
   for ro in (-1,1):
    pe,po=projections[j,-1,re],projections[i,1,ro]
    for u in range(3):
     left=mul(pe[u],D)
     for v in range(3):
      right=mul(po[v],D)
      for p in range(3):
       for q in range(p,3):
        weight=np.einsum('aij,bji->ab',left[0][p],right[0][q])-np.einsum('aij,bji->ab',left[1][p],right[1][q])
        if p!=q:
            weight+=np.einsum('aij,bji->ab',left[0][q],right[0][p])-np.einsum('aij,bji->ab',left[1][q],right[1][p])
        if np.any(weight): terms[i,j,re,ro,u,v,p,q]=weight
frames=[{label:sp.Integer(0) for label in a.labels} for _ in range(2)]
frames[0]['03']=(a.a1+a.a2)*a.R/(2*a.c);frames[0]['33']=(a.a1-a.a2)*a.R/(2*a.c)
frames[1]['01']=(a.y1+a.y2)*a.R/(2*a.t);frames[1]['31']=(a.y1-a.y2)*a.R/(2*a.t)
parameters=(a.c,a.d,a.t)
def raw_entry(n,m):
    x,y=a.names[n],a.names[m]
    mu,nu=int(x[1]),int(y[1]);label,other=x[2:],y[2:]
    elastic=0
    if x[0]==y[0]:
        if mu==nu and label==other:
            elastic+=2*(a.c*a.c if mu==0 else a.t*a.t)/a.R
            if x[0]=='a': elastic+=2*a.J
        if mu==nu: elastic+=2*a.k*sum(f[label]*f[other] for f in frames)
        elastic+=2*a.k*frames[mu][other]*frames[nu][label]
    sea=0
    for (i,j,re,ro,u,v,p,q),weight in terms.items():
        w=int(weight[n,m])
        if not w: continue
        f=(1,(a.a1,a.a2)[j],(a.y1,a.y2)[j])[u]
        g=(1,(a.a1,a.a2)[i],(a.y1,a.y2)[i])[v]
        gap=a.R*(2+a.d/a.c*(re*(a.a1,a.a2)[j]-ro*(a.a1,a.a2)[i]))
        sea+=sp.Rational(w,16)*parameters[p]*parameters[q]*f*g/gap
    return elastic-sea

def exact_partition():
    groups=list(a.groups)
    groups += [['s020','s112','a022'],['s021','s123','a013'],['s022','s110','a020'],['s023','s121','a011']]
    assigned={name for group in groups for name in group}
    groups += [[name] for name in a.names if name not in assigned]
    ownership={name:i for i,g in enumerate(groups) for name in g}
    if len(ownership)!=60: raise AssertionError('Partition omits/repeats an active coordinate')
    checked=0;nontrivial=0
    for n in range(60):
      for m in range(n+1,60):
       if ownership[a.names[n]]==ownership[a.names[m]]: continue
       expr=raw_entry(n,m);checked+=1
       if expr!=0:
           nontrivial+=1
           if a.norm(expr)!=0: raise AssertionError(f'Off-block entry does not vanish: {a.names[n]} {a.names[m]}')
    for first,second in zip(a.groups[2:6],groups[11:15]):
        h=sp.Matrix([[a.norm(raw_entry(a.names.index(n),a.names.index(m))) for m in second] for n in second])
        other=sp.Matrix([[a.entry(n,m) for m in first] for n in first])
        for degree in (1,2,3):
            if a.norm(sp.trace(h**degree)-sp.trace(other**degree))!=0:
                raise AssertionError('Multiplicity-two invariant mismatch')
    constants=[]
    for group in groups:
        if len(group)!=1: continue
        name=group[0];n=a.names.index(name);value=a.norm(raw_entry(n,n))
        constants.append((name,str(value)))
        if name=='a030': expected=a.entry(name,name)
        elif name in ('s002','s032','s102','s132'): expected=0
        elif name=='s130': expected=2*a.t**2/a.R
        else: expected=2*a.t**2/a.R+2*a.J
        if a.norm(value-expected)!=0: raise AssertionError(f'Singleton has unexpected value: {name}')
    return dict(exact_integer_Pauli_off_block_pairs=checked,nontrivial_cancelled_pairs=nontrivial,
        entire_active60_partition_verified=True,multiplicity_two_characteristics_verified=True,
        integer_trace_overflow_upper=2*64*16**2,singletons=constants)
if __name__=='__main__': print(exact_partition())
