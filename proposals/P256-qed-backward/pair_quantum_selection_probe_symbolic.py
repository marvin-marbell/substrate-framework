#!/usr/bin/env python3
import os
from typing import cast
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import sympy as s
from full_band_gauss_section import FullBandGaussSection
from pair_global_field_bound import coefficients
from rotation_band import RotationBand
from unrestricted_field_minimum import ALIGNMENT, BARE, STIFFNESS
P=[s.eye(2),s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),s.diag(1,-1)]
a1,a2,y1,y2,R=s.symbols('a1 a2 y1 y2 R',real=True)
_native=FullBandGaussSection(RotationBand(**BARE),heavy_scale=1.)
# Rational(float) captures the actual binary coefficient, not rounded text.
c,d,t=map(s.Rational,coefficients(_native))
k,J=map(s.Rational,(STIFFNESS,ALIGNMENT))
del _native
delta=s.symbols('delta',real=True)
alpha=(c**4+t**4)/k
q=t**4/(c**4+t**4)
square={a1:1-q+delta,a2:1-q-delta,y1:q-delta,y2:q+delta}
def reduce_poly(expr):
    poly=s.Poly(s.expand(expr),a1,a2,y1,y2,R)
    result=0
    for powers,coeff in poly.terms():
        term=coeff
        for variable,power in zip((a1,a2,y1,y2),powers[:4]):
            term*=square[variable]**(power//2)*variable**(power%2)
        term*=alpha**(powers[4]//3)*R**(powers[4]%3)
        result+=term
    return s.expand(result)
def norm(expr):
    # Reduce in the finite radical quotient BEFORE multivariate polynomial
    # cancellation. Native dyadics make cancel-then-reduce needlessly costly.
    num,den=s.fraction(s.together(expr))
    num,den=reduce_poly(num),reduce_poly(den)
    for variable in (a1,a2,y1,y2):
        if den.has(variable):
            conjugate=den.xreplace({variable:-variable})
            num,den=reduce_poly(num*conjugate),reduce_poly(den*conjugate)
    return s.cancel(num/den)
A=s.diag(a1,-a1,a2,-a2);Y=s.diag(y1*P[1],y2*P[1]);Q=s.kronecker_product(P[0],A)+s.kronecker_product(P[2],Y)
labels=[str(i)+str(j) for i in range(4) for j in range(4) if i+j]
basis={str(i)+str(j):s.kronecker_product(P[i],P[j]) for i in range(4) for j in range(4) if i+j}
names=[site+str(mu)+a for site in ('s','a') for mu in range(2) for a in labels]
D={}
V={}
for name in names:
    site,mu,label=name[0],int(name[1]),name[2:]
    b=basis[label];factor=s.sqrt(2)/2
    v=[b*factor,b*factor*(1 if site=='s' else -1)]
    V[name]=v
    dh=s.zeros(8)
    if mu==0:
        dh[:4,:4]=c*v[0];dh[4:,4:]=c*v[1]
        hop=-d*(v[0]+v[1])/2
    else: hop=-s.I*t*(v[0]+v[1])/2
    dh[:4,4:]=hop;dh[4:,:4]=cast(s.MatrixBase,hop).conjugate().T
    D[name]=dh
proj={}
for i,(aa,yy) in enumerate(((a1,y1),(a2,y2))):
    ii=s.diag(*[int(j//2==i) for j in range(4)])
    mask=s.kronecker_product(P[0],ii)
    W=s.kronecker_product(P[1],s.diag(*[(-1)**j*int(j//2==i) for j in range(4)]))
    for sign in (-1,1):
        for shift in (-1,1): proj[i,sign,shift]=s.expand((mask+sign*Q*mask)*(mask+shift*W)/4)

def entry(n,m):
    un,um=int(n[1]),int(m[1]);B,C=V[n],V[m]
    frames=[R/c*A,R/t*Y]
    elastic=0
    for site in range(2):
        inner=lambda x,z:s.trace(x*z)/4
        if un==um:
            elastic+=2*k*(sum(inner(B[site],v)*inner(C[site],v) for v in frames)+inner(B[site],frames[un])*inner(C[site],frames[un])+inner(frames[un],frames[un])*inner(B[site],C[site]))
        else:
            elastic+=2*k*inner(B[site],frames[um])*inner(C[site],frames[un])
    if n[0]=='a' and m[0]=='a' and un==um:
        elastic+=2*J*s.trace(basis[n[2:]]*basis[m[2:]])/4
    sea=0
    for i in range(2):
      for j in range(2):
       for re in (-1,1):
        for ro in (-1,1):
         pe,po=proj[j,-1,re],proj[i,1,ro]
         weight=s.expand(s.re(s.trace(pe*D[n]*po*D[m])))
         if weight:
          gap=R*(2+d/c*(re*(a1,a2)[j]-ro*(a1,a2)[i]))
          sea+=2*weight/gap
    return norm(elastic-sea)
groups=[['s001','s031','s103','s133'],['s003','s033','s101','s131'],['s010','s122','a012'],['s011','s113','a023'],['s012','s120','a010'],['s013','s111','a021'],['s030','a002','a032'],['a001','a031','a103','a133'],['a003','a033'],['a030'],['a101','a131']]
if __name__=='__main__':
 for group in groups:
    print('group',group,flush=True)
    h=s.Matrix([[entry(n,m) for m in group] for n in group])
    print('H',h,flush=True)
    power=s.eye(len(group));traces=[];es=[s.Integer(1)]
    for degree in range(1,len(group)+1):
        power=power*h
        power=power.applyfunc(norm)
        traces.append(norm(s.trace(power)))
        es.append(norm(sum((-1)**(i-1)*es[degree-i]*traces[i-1] for i in range(1,degree+1))/degree))
    print('E',es,flush=True)
