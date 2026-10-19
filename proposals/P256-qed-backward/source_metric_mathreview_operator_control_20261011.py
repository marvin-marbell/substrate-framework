"""Exact small-domain CAR falsifiers and rational operation-budget inequalities."""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import sympy as sp

def car(mask,p,create):
 if bool(mask&(1<<p))==create:return None
 return mask^(1<<p),(-1)**((mask&((1<<p)-1)).bit_count())

def transfer(state,f,target,source,M):
 out={}
 for mask,z in state.items():
  for i in range(2):
   for j in range(2):
    a=car(mask,4*source+2*f+j,False)
    if a is None:continue
    b=car(a[0],4*target+2*f+i,True)
    if b is not None:out[b[0]]=out.get(b[0],0)+z*M[i][j]*a[1]*b[1]
 return {m:z for m,z in out.items() if z}

def add(states):
 out={}
 for state in states:
  for m,z in state.items():out[m]=out.get(m,0)+z
 return {m:z for m,z in out.items() if z}

def inner(a,b):return sum(z*b.get(m,0) for m,z in a.items())

def main():
 M=[[1,2],[3,-1]];N=[[2,-1],[1,1]]
 state={sum(1<<(4*c+2*f) for c in range(3) for f in range(2)):1}
 ab={};ba={}
 for f in range(2):
  for g in range(2):
   ab[f,g]=transfer(transfer(state,g,1,2,N),f,0,1,M)
   ba[f,g]=transfer(transfer(state,f,0,1,M),g,1,2,N)
 for owner in (ab,ba):
  for k in owner:
   for h in owner:
    if k!=h:assert inner(owner[k],owner[h])==0
 AB=add(ab.values());BA=add(ba.values())
 assert inner(AB,AB)==sum(inner(v,v) for v in ab.values())
 assert inner(BA,BA)==sum(inner(v,v) for v in ba.values())
 sa=2*2*sum(z*z for row in M for z in row);sb=2*2*sum(z*z for row in N for z in row)
 assert inner(AB,AB)<=sa*sb and inner(BA,BA)<=sa*sb
 assert inner(ab[0,0],ab[0,0])>0 and inner(ba[0,0],ba[0,0])>0
 contact=add([AB,{m:-z for m,z in BA.items()}]);contactnorm=inner(contact,contact)
 assert contactnorm>0
 coherent=Q(inner(AB,AB)+inner(BA,BA)+2*inner(AB,BA),4)
 wrong=Q(inner(AB,AB)+inner(BA,BA),2)
 assert wrong-coherent==Q(contactnorm,4)>0
 gamma=lambda n:Q(n,2**53-n)
 # Dense real/imaginary dependency paths <=256, frame entry path sum <=2a.
 # ||frame-error||F<=8a gamma256; trace(frame^2)/4 relative perturbation.
 q=8*gamma(256)+16*gamma(256)**2+gamma(128)*(1+4*gamma(256))**2
 assert q<2*gamma(2048)-gamma(2048)**2
 # sqrt relative perturbation <= gamma2048, plus sqrt rounding remains covered.
 assert (q+2*Q(1,2**53))<2*gamma(2048)-gamma(2048)**2
 gamma_entry=(2*gamma(256)+gamma(2048)+Q(1,2**53))/(1-gamma(2048))
 assert gamma_entry<gamma(4096)
 # Induced entrywise-l1 coefficient error from two S multiplications, M conversion/reduction.
 coeff=(1+4*gamma(4096))**2*(1+gamma(256))-1
 assert coeff<64*gamma(8192)
 x,y,a,b,c=sp.symbols('x y a b c',positive=True)
 F=1+x*x*a+y*y*b+x*x*y*y*c
 mixed=sp.diff(sp.log(F),x,y)
 assert sp.simplify(mixed-4*x*y*(c-a*b)/F**2)==0
 pAB=x*x*y*y*c/F;pA=(x*x*a+x*x*y*y*c)/F;pB=(y*y*b+x*x*y*y*c)/F
 assert sp.simplify(pAB-pA*pB-x*x*y*y*(c-a*b)/F**2)==0
 t=Q(1,2);aa=Q(2);bb=Q(3);cc=Q(7);FF=1+t*t*(aa+bb)+t**4*cc
 diagnostic=t**4*(cc-aa*bb)/FF;probability=diagnostic/FF
 assert diagnostic!=probability
 out={'kind':'INDEPENDENT_EXACT_FULL_CAR_OPERATOR_CONTROL','natural_main_finished':True,
 'domain':'three cells, two copies, two spin modes/copy/cell; fixed number1 per cell/copy, finite structural analogue only',
 'ordered_copy_sector_orthogonality':True,'same_copy_full_CAR_kept':True,
 'AB_norm_square':inner(AB,AB),'BA_norm_square':inner(BA,BA),'ordered_overlap':inner(AB,BA),
 'Sa':sa,'Sb':sb,'commutator_norm_square':contactnorm,'coherent_norm_square':str(coherent),
 'wrong_incoherent_norm_square':str(wrong),'coefficient_budget_inequalities_exact':True,
 'mixed_log_derivative_symbolic_identity':True,'probability_covariance_symbolic_identity':True,
 'covariance_label_control':{'source_connected_divided_F':str(diagnostic),'probability_covariance':str(probability)},
 'own_source_SHA':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 Path(__file__).with_name('source_metric_mathreview_operator_control_actual1_20261011_receipt.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out),flush=True)

if __name__=='__main__':main()
