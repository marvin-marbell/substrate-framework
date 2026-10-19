from __future__ import annotations
import hashlib, json, math, time
from fractions import Fraction
from pathlib import Path
from typing import Any
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parent
INPUT='local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json'
PIN=hashlib.sha256((ROOT/INPUT).read_bytes()).hexdigest()
OWN=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BITS=100;Q=1<<BITS
# Every endpoint is an integer multiple of 2^-100; integer division is outward.
def iv(x):
 f=Fraction(x);a=f.numerator*Q;b=f.denominator
 return (a//b,-((-a)//b))
def add(a,b):return (a[0]+b[0],a[1]+b[1])
def neg(a):return (-a[1],-a[0])
def sub(a,b):return add(a,neg(b))
def mul(a,b):
 p=[a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]]
 return (min(p)//Q,-((-max(p))//Q))
def inv(a):
 assert a[0]>0 or a[1]<0
 return (Q*Q//a[1],-((-Q*Q)//a[0]))
def sqrt(a):
 assert a[0]>=0
 lo=math.isqrt(a[0]*Q);hi=math.isqrt(a[1]*Q)
 return (lo,hi+(hi*hi<a[1]*Q))
def scale(a,n):return (a[0]*n,a[1]*n) if n>=0 else (a[1]*n,a[0]*n)
def mid(a):return (a[0]+a[1])/(2*Q)
def square_upper(a):return (max(abs(a[0]),abs(a[1]))**2+Q-1)//Q
ZERO=iv(0);ONE=iv(1)
p=json.loads((ROOT/INPUT).read_bytes());roles=[]
RITZ='local_so5_resum_actual04_20261011_receipt.json'
RITZ_PIN=hashlib.sha256((ROOT/RITZ).read_bytes()).hexdigest()
p['current_Ritz_mu']=json.loads((ROOT/RITZ).read_bytes())['zero_u']['energy']
assert p['corrected_source_unitarity_H_density_identities'] and p['parent_pair_reusable_after_new_input_binding']
for item in p['profiles']:
 m=iv(Fraction(item['m_exact']));d=iv(Fraction(item['d_exact']));c=iv(Fraction(item['c0_exact']));e=sqrt(iv(Fraction(item['e_squared_exact'])))
 assert Fraction(item['e_squared_exact'])==Fraction(item['m_exact'])**2+Fraction(item['d_exact'])**2
 energy=[scale(e,-2),scale(c,2),ZERO,scale(c,-2),scale(e,2)]
 gap=[sub(x,energy[0]) for x in energy]
 r={}
 for b in [1,3]:
  for a,v in [(0,mul(d,inv(scale(e,2)))),(2,mul(m,inv(mul(sqrt(iv(2)),e)))),(4,neg(mul(d,inv(scale(e,2)))) )]:r[a,b]=r[b,a]=v
 roles.append((int(item['spin_replicas']),energy,gap,r))
base=ZERO
for n,energy,_,_ in roles:base=add(base,scale(energy[0],n))
def occupations(n,gap,budget):
 out=[]
 def visit(j,remaining,cost,word):
  if j==5:out.append(((remaining,*word),cost));return
  # Nonnegative gaps and lower endpoints certify every excluded continuation.
  limit=min(remaining,max(0,(budget*Q-cost[0])//gap[j][0]))
  for k in range(limit+1):visit(j+1,remaining-k,add(cost,scale(gap[j],k)),word+[k])
 visit(1,n,ZERO,[]);return out
start=time.monotonic();support_budget=322;output_budget=240;z=iv(-5123)
aa=occupations(roles[0][0],roles[0][2],support_budget);bb=occupations(roles[1][0],roles[1][2],support_budget)
states=[(a,b,add(ea,eb)) for a,ea in aa for b,eb in bb if add(ea,eb)[0]<=support_budget*Q]
index={(a,b):i for i,(a,b,_) in enumerate(states)}
outputs=[i for i,(_,_,cost) in enumerate(states) if cost[1]<=output_budget*Q]
row_index={s:j for j,s in enumerate(outputs)}
maxstep=max(sub(energy[4],energy[0])[1] for _,energy,_,_ in roles)
assert output_budget*Q+maxstep<support_budget*Q
# Check the entire density image of every retained OUTPUT row, not PHP inputs.
for i in outputs:
 a,b,_=states[i]
 for role,(_,_,_,r) in enumerate(roles):
  word=a if role==0 else b
  for dest,src in r:
   if word[src]:
    nw=list(word);nw[src]-=1;nw[dest]+=1
    key=(tuple(nw),b) if role==0 else (a,tuple(nw))
    assert key in index
columns=[[] for _ in states]
rr=[];cc=[];vv=[]
for j,(a,b,_) in enumerate(states):
 for role,(_,_,_,r) in enumerate(roles):
  word=a if role==0 else b
  for (dest,src),value in r.items():
   if not word[src]:continue
   nw=list(word);nw[src]-=1;nw[dest]+=1
   key=(tuple(nw),b) if role==0 else (a,tuple(nw))
   target=index.get(key)
   if target in row_index:
    row=row_index[target];value=mul(value,sqrt(iv(word[src]*(word[dest]+1))))
    columns[j].append((row,value));rr.append(row);cc.append(j);vv.append(mid(value))
B=coo_matrix((vv,(rr,cc)),shape=(len(outputs),len(states))).tocsr()
D=[sub(add(base,cost),z) for _,_,cost in states]
L=[j for j,v in enumerate(D) if v[1]<0];positive=[j for j,v in enumerate(D) if v[0]>0]
assert len(L)+len(positive)==len(states)
assert base[0]+support_budget*Q>z[1]
BL=B[:,L].toarray();BQ=B[:,positive]
dq=np.array([mid(D[j]) for j in positive])
K=np.eye(len(outputs))+64*((BQ.multiply(1/dq))@BQ.T).toarray()
Y=cho_solve(cho_factor(K,lower=True),BL)
YI=[[iv(float(x)) for x in row] for row in Y]
k=len(L);m=len(outputs)
# Exact dyadic enclosures of all expressions evaluated at the binary witness Y.
def column_product(j):
 result=[ZERO]*k
 for row,b in columns[j]:
  result=[add(result[t],mul(b,YI[row][t])) for t in range(k)]
 return result
T={j:column_product(j) for j in positive}
G=[[ZERO]*k for _ in range(k)]
for t,j in enumerate(L):
 v=column_product(j)
 for u in range(k):G[t][u]=add(G[t][u],v[u]);G[u][t]=add(G[u][t],v[u])
for row in YI:
 for t in range(k):
  for u in range(t,k):
   v=mul(row[t],row[u]);G[t][u]=sub(G[t][u],v)
   if u!=t:G[u][t]=sub(G[u][t],v)
RY=[[neg(x) for x in row] for row in YI]
for t,j in enumerate(L):
 for row,b in columns[j]:RY[row][t]=add(RY[row][t],b)
for j in positive:
 v=T[j];dinv=inv(D[j]);weighted=[scale(mul(x,dinv),64) for x in v]
 for row,b in columns[j]:
  for t in range(k):RY[row][t]=sub(RY[row][t],mul(b,weighted[t]))
 for t in range(k):
  for u in range(t,k):
   val=mul(v[t],weighted[u]);G[t][u]=sub(G[t][u],val)
   if u!=t:G[u][t]=sub(G[u][t],val)
residual_squared_upper=sum(square_upper(x) for row in RY for x in row)
S=[[scale(G[t][u],64) for u in range(k)] for t in range(k)]
for t,j in enumerate(L):S[t][t]=add(S[t][t],D[j])
center=[[(x[0]+x[1])//2 for x in row] for row in S]
assert all(center[t][u]==center[u][t] for t in range(k) for u in range(k))
radius=max(sum(max(center[t][u]-S[t][u][0],S[t][u][1]-center[t][u]) for u in range(k)) for t in range(k))
# K>=I gives G(Y)<=BL* K^-1 BL<=G(Y)+(BL-KY)*(BL-KY).
# Entrywise interval uncertainty is enclosed in radius I; residual Frobenius cap encloses its Gram.
def inertia(matrix):
 a=[row[:] for row in matrix];previous=1;signs=[];minors=[]
 for t in range(len(a)):
  pivot=a[t][t];assert pivot!=0
  signs.append(1 if pivot*previous>0 else -1);minors.append(pivot)
  for i in range(t+1,len(a)):
   for j in range(t+1,len(a)):
    numerator=a[i][j]*pivot-a[i][t]*a[t][j]
    assert numerator%previous==0
    a[i][j]=numerator//previous
  previous=pivot
 return dict(negative=signs.count(-1),positive=signs.count(1),leading_minor_signs=[1 if x>0 else -1 for x in minors])
low=[row[:] for row in center];high=[row[:] for row in center]
for t in range(k):low[t][t]-=radius;high[t][t]+=radius+64*residual_squared_upper
lower_inertia=inertia(low);upper_inertia=inertia(high)
assert lower_inertia['negative']==upper_inertia['negative']==1
result: dict[str, Any]
result=dict(kind='EXACT_OUTWARD_DYADIC_POSITIVE_OUTPUT_ROW_SCHUR_INERTIA',bits=BITS,integer_rounding_outward=True,selected_local_containing_space='Sym^2(C5) x Sym^126(C5), containing actual harmonic14x690880; NOT other canonical spin representations/Phi/lattice',source_rows=len(outputs),input_support=len(states),negative_free_columns=k,free_ground_interval=[base[0]/Q,base[1]/Q],outside_free_floor_lower=(base[0]+support_budget*Q)/Q,positive_output_row_budget=output_budget,input_support_budget=support_budget,exact_second_eigenvalue_lower=-5123,lower_inertia=lower_inertia,upper_inertia=upper_inertia,interval_matrix_error_upper=radius/Q,Schur_witness_residual_Frobenius_squared_upper=residual_squared_upper/Q,second_eigenvalue_certified=True,proof='Hlo=D+64 R* P_rows R<=H; positive output rows, NOT input PHP cropping. Entire output density image checked. Complement D outside support exceeds z. Positive free Schur Q block. Fixed binary Y variational Gram enclosed exactly by outward dyadic operations, K>=I residual bound. Integer Bareiss inertia for both Loewner enclosures is exactly one negative; hence Hlo has one eigenvalue below -5123 and lambda2(H)>=-5123.',existing_Ritz_mu=p['current_Ritz_mu'],Ritz_roundoff_interval_certified=False,local_ground_state_certified=False,fullPhi_ground=False,bulk_stiffness=False,phase=False,pole=False,input_sha256={INPUT:PIN},own_source_sha256=OWN,input_unchanged=hashlib.sha256((ROOT/INPUT).read_bytes()).hexdigest()==PIN,source_unchanged=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==OWN,elapsed_seconds=time.monotonic()-start)
result['input_sha256'][RITZ]=RITZ_PIN
result['input_unchanged']=result['input_unchanged'] and hashlib.sha256((ROOT/RITZ).read_bytes()).hexdigest()==RITZ_PIN
result['native_phase_binding']='NEW corrected free5 pair; frozen same-source U statement withdrawn. Actual source oneparticle phase and F preserve rho/G/harmonic sectors.'
(ROOT/'local_so5_positive_output_row_inertia_certificate_20261011_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
