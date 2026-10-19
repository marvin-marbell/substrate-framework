"""Independent exact/rational charged spectral audit; never imports target modules.
Consumes immutable raw observations under explicit operation-model premises.
"""
from __future__ import annotations
import argparse
from typing import cast
from fractions import Fraction as F
from pathlib import Path
import hashlib
import io
import itertools
import json
import math
import sympy as sy
import numpy as np
import zipfile

ROOT = Path(__file__).resolve().parent
PINS = {
 'charged_stieltjes_scientist_20261011.py':'adaa0b8064e5744071a90e66defc806ae5cf8603cb3a5f4816fc1575d76d551a',
 'charged_stieltjes_proof_20261011.json':'182d714cd5f5a6a1bedf13121d0c3d2443a6f8f5476f7f68a2cc2788305a1d36',
 'charged_stieltjes_actual1_20261011_receipt.json':'dad7bbbf79e3298bc598eca59507d90f1d9f4b4e8f10065241d28e9ffd6a359f',
 'charged_stieltjes_actual1_20261011_executed.zip':'cb158c776ddc1b7e962d987343c9116a216e2547aa3712cc64f3da644c78ffe6'}

def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def load(name):
 return json.loads((ROOT/name).read_text())

class Interval:
 """Rational endpoints without target fixed-point implementation."""
 def __init__(self, lo: F | int | str = 0, hi: F | int | str | None = None):
  self.lo=F(lo);self.hi=F(lo if hi is None else hi)
 def __add__(self, other):
  o=asiv(other);return Interval(self.lo+o.lo,self.hi+o.hi)
 __radd__=__add__
 def __neg__(self):return Interval(-self.hi,-self.lo)
 def __sub__(self, other):return self+-asiv(other)
 def __rsub__(self, other):return asiv(other)+-self
 def __mul__(self, other):
  o=asiv(other);v=[a*b for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
  return Interval(min(v),max(v))
 __rmul__=__mul__
 def __truediv__(self, other):
  o=asiv(other);assert o.lo>0 or o.hi<0
  return self*Interval(1/o.hi,1/o.lo)
 def __pow__(self,n):
  out=Interval(1)
  for _ in range(n):out=out*self
  return out
 def sqrt(self):
  assert self.lo>=0
  q=1<<120
  a=math.isqrt(self.lo.numerator*q*q//self.lo.denominator)
  b=math.isqrt(self.hi.numerator*q*q//self.hi.denominator)
  return Interval(F(a,q),F(b+(F(b*b,q*q)<self.hi),q))
 def data(self):return [str(self.lo),str(self.hi)]

def asiv(x):return x if isinstance(x,Interval) else Interval(x)

def expr(x):
 x=sy.simplify(x)
 if x.is_Rational:return Interval(F(int(x.p),int(x.q)))
 if x.is_Add:return sum((expr(a) for a in x.args),Interval())
 if x.is_Mul:
  out=Interval(1)
  for a in x.args:out=out*expr(a)
  return out
 if x.is_Pow and x.exp==sy.Rational(1,2):return expr(x.base).sqrt()
 raise ValueError(x)

def wedge_annihilators():
 words=[tuple(p for p in range(4) if mask & (1<<p)) for mask in range(16)]
 index={w:i for i,w in enumerate(words)};ans=[]
 for p in range(4):
  a=sy.zeros(16)
  for j,w in enumerate(words):
   if p in w:a[index[tuple(q for q in w if q!=p)],j]=(-1)**w.index(p)
  ans.append(a)
 return words,ans

def lift(h,ann):
 return sum((h[p,q]*ann[p].T*ann[q] for p,q in itertools.product(range(4),repeat=2)),sy.zeros(16))

def tensor_coeff(SA,SB,CA,CB,M,spinA,spinB):
 y={};v={}
 for k,l,a,b in itertools.product(range(4),range(4),range(5),range(5)):
  yy=sy.Integer(0);vv=sy.Integer(0)
  for p,q in itertools.product(range(2),repeat=2):
   m=M[2*spinA+p,2*spinB+q]
   yy+=m*SA[p][k,a]*SB[q][l,b]
   vv+=m*(CA[p][k,a]*SB[q][l,b]+SA[p][k,a]*CB[q][l,b]+64*SA[p][k,a]*SB[q][l,b])
  yy=sy.simplify(yy);vv=sy.simplify(vv)
  if yy:y[k,l,a,b]=yy
  if vv:v[k,l,a,b]=vv
 return y,v

def tensor_inner(y,v,K):
 # Direct tensor vector inner product; no T/U or target contraction routine.
 polys={}
 for (k,l,a,b),c in y.items():
  for (kk,ll,d,e),t in v.items():
   if (k,l)==(kk,ll):
    key=(a,d,b,e);polys[key]=polys.get(key,0)+sy.conjugate(c)*t
 result=Interval()
 for (a,d,b,e),c in polys.items():
  c=sy.simplify(sy.re(c))
  if c:result+=expr(c)*K[a][d]*K[b][e]
 return result

def controls():
 x,a,G=sy.symbols('x a G',positive=True)
 q1=1/a-(x-a)/a**2+(x-a)**2/(G*a**2)
 q2=1/a**2-2*(x-a)/a**3+(x-a)**2*(2*G+a)/(G**2*a**3)
 f1=sy.factor(q1-1/x);f2=sy.factor(q2-1/x**2)
 assert sy.simplify(f1-(x-a)**2*(x-G)/(G*a*a*x))==0
 assert sy.simplify(f2-(x-a)**2*(x-G)*(2*G*x+a*(x+G))/(G*G*a**3*x*x))==0
 assert sy.simplify(1/x-1/a+(x-a)/a**2-(x-a)**2/(a*a*x))==0
 assert sy.simplify(1/x**2-1/a**2+2*(x-a)/a**3-(x-a)**2*(2*x+a)/(a**3*x*x))==0
 measure=[(F(2),F(1,2)),(F(1000000),F(1,2))]
 m=[sum(w*t**j for t,w in measure) for j in range(3)]
 K=sum(w/t for t,w in measure);J=sum(w/t**2 for t,w in measure)
 assert K>m[0]**2/m[1] and J>m[0]**3/m[1]**2
 tailwrong=F(1,5)-(F(1000000)-5)/25
 assert tailwrong<F(1,1000000)
 # General complex T, not an eigenstate or real-only test: C=iS and D=iS.
 cross_good=2*sy.re(sy.conjugate(sy.I)*sy.I)
 cross_wrong=2*sy.re(sy.I*sy.I)
 assert 4==1+1+cross_good and 4!=1+1+cross_wrong and 4!=1+1
 # Exact residual state, H=diag(0,1), s=(3/5,4/5), B=I.
 Es=F(16,25);r2=F(144,625)
 true_first=Es;true_second=Es
 observed=F('0.00000005679');cap=F(1,65536)
 # A legitimate residual below accepted cap but above the old observation:
 # rational unit s=((1-t²)/(1+t²),2t/(1+t²)), H=diag(0,1).
 t=F(1,262144);source_a=(1-t*t)/(1+t*t);source_b=2*t/(1+t*t)
 small_r=source_a*source_b
 assert observed<small_r<cap
 # Replacing residual by observation excludes this exact second moment.
 assert small_r**2>observed**2
 omitted_shift_first=F(0)
 assert true_first**2>r2 and omitted_shift_first==0
 # Omit external64: spectral functional changes, even with exact matter energy.
 exact=F(1,74);wrong=F(1,10)
 assert exact!=wrong and F(1,74**2)!=F(1,10**2)
 # Infidelity is squared: phase-aligned rational unit vectors.
 overlap=F(4,5);I=1-overlap**2;distance2=2*(1-overlap)
 assert distance2<=2*I and distance2>2*I**2
 return {'global_inverse_upper_difference':str(f1),'global_square_upper_difference':str(f2),
  'all_x_proof':'Every factor is nonnegative for x>=G>0,a>0; no finite grid argument.',
  'Jensen_wrong_upper_rejected':{'K':str(K),'J':str(J),'wrong_K':str(m[0]**2/m[1]),'wrong_J':str(m[0]**3/m[1]**2)},
  'missing_tail_rejected':str(tailwrong),'missing_m2_cross_rejected':True,'wrong_T_not_dagger_rejected':True,
  'omitted_external64_rejected':{'correct_K':str(exact),'wrong_K':str(wrong)},
  'observed_residual_substitution_rejected':{'accepted_cap':str(cap),'observation_not_bound':str(observed),'legitimate_exact_residual':str(small_r),'wrong_second_moment_upper':str(observed**2),'true_second_moment':str(small_r**2)},
  'omitted_ground_shift_rejected':{'true_m1':str(true_first),'true_m2':str(true_second),'residual_squared':str(r2)},
  'squared_infidelity_without_sqrt_rejected':True}

def run():
 parser=argparse.ArgumentParser()
 parser.add_argument('--output',required=True)
 args=parser.parse_args()
 observed={n:sha(ROOT/n) for n in PINS};assert observed==PINS
 receipt=load('charged_stieltjes_actual1_20261011_receipt.json')
 bindings=receipt['input_sha256'];assert all(sha(ROOT/n)==v for n,v in bindings.items())
 archive_verified={}
 with zipfile.ZipFile(ROOT/'charged_stieltjes_actual1_20261011_executed.zip') as z:
  pre_name='charged_stieltjes_actual1_20261011_preexecution.zip'
  prior_exit=load('charged_stieltjes_actual1_20261011_exit.json')
  pre_bytes=z.read(pre_name)
  assert hashlib.sha256(pre_bytes).hexdigest()==prior_exit['preexecution_sha256']
  assert prior_exit['natural_exit'] and prior_exit['returncode']==0
  with zipfile.ZipFile(io.BytesIO(pre_bytes)) as pre:
   for n,v in {**bindings,**{k:v for k,v in PINS.items() if not k.endswith('.zip')}}.items():
    provider=z if n in z.namelist() else pre
    matches=[m for m in provider.namelist() if m==n or m.endswith('/'+n)]
    assert len(matches)==1,(n,matches)
    got=hashlib.sha256(provider.read(matches[0])).hexdigest();assert got==v
    archive_verified[n]=got
 result=controls()
 words,ann=wedge_annihilators()
 for p,q in itertools.product(range(4),repeat=2):
  assert ann[p]*ann[q].T+ann[q].T*ann[p]==sy.eye(16)*int(p==q)
 R=sy.diag(sy.Rational(1,2),sy.Rational(1,2),-sy.Rational(1,2),-sy.Rational(1,2));RF=lift(R,ann)
 hbase=[sy.diag(-1,1,-1,1),sy.Matrix([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]]),sy.Matrix([[0,0,0,sy.I],[0,0,sy.I,0],[0,-sy.I,0,0],[-sy.I,0,0,0]])]
 V=sy.zeros(16,5)
 for j,mask in enumerate((5,3,12,10)):V[mask,j]=1
 V[9,4]=V[6,4]=1/sy.sqrt(2)
 J=sy.Matrix(load('harmonic_car_bridge_typed_attempt3_20261011_receipt.json')['controls']['joint_n1_native_embedding'])
 W=V*J*sy.diag(1,-sy.I,1,1,sy.I)
 assert sy.simplify(W.conjugate().T*W)==sy.eye(5)
 M=sy.Matrix(receipt['source_endpoint']['joint_native_matrix_exact'])
 assert all(sy.im(v)==0 for v in M)
 with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',allow_pickle=False) as data:
  phi=sy.Matrix([[sy.Rational(F(float(complex(v).real)))+sy.I*sy.Rational(F(float(complex(v).imag))) for v in row] for row in data['Phi_binary_section'][1]])
 had=sy.Matrix([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])/sy.sqrt(2)
 direction=had.T*phi*had
 direction=sy.simplify(direction/sy.sqrt(sy.trace(direction**2)/4))
 frame=sy.Matrix.hstack(sy.eye(4)[:,0],direction[:,0],sy.eye(4)[:,1],direction[:,1])
 path=next(p for p in load('charged_first_shell_actual04_20261011_receipt.json')['all64_boundary']['roles'][0]['paths'] if p['axis']==1 and p['length']==1)
 physical=sy.Matrix([[sy.Rational(a)+sy.I*sy.Rational(b) for a,b in row] for row in path['matrix_exact']])
 phase=sy.diag(1,-1,1,-1)
 assert sy.simplify(phase*frame.conjugate().T*physical*frame*phase-M)==sy.zeros(4)
 assert sy.simplify(frame.conjugate().T*frame)==sy.eye(4)
 geometry=load('joined_phi_u1_harmonic_transverse_affine_boundary_control_20261011_receipt.json')
 for qa in (-1,1):
  geo=next(r for r in geometry['rows'] if r['axis']==1 and r['length']==1 and r['sites']==[0,0] and r['qa']==qa)
  assert list(map(F,geo['affine_u']))==[-F(qa,2),F(qa,2)]
 L=sum((abs(v) for row in M.tolist() for v in row),sy.Integer(0))
 assert F(str(L))==F(receipt['source_operator_L1_bound']['exact'][1])
 gamma=F(64)-10240-2*F(receipt['ground_energy_upper_exact'])
 assert [str(gamma)]*2==receipt['gamma_exact'] and gamma>0
 profiles=load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json')['profiles']
 N=F(receipt['source_norm_squared_exact'])
 moment_checks=[];commutators=0
 for prof in profiles:
  h=cast(sy.MutableDenseMatrix,sum((sy.Rational(prof[key])*hb for key,hb in zip(('m_exact','c0_exact','d_exact'),hbase)),sy.zeros(4)));D=lift(h,ann)
  # Complete 16-dimensional fermion domain, both orientations, all4 modes,
  # and a nonzero spectator density. Not merely the five-dimensional input.
  spectator=sy.Rational(7,3)
  neutral=D+64*(RF+spectator*sy.eye(16))**2
  for eps,u,p in itertools.product((-1,1),(-sy.Rational(1,2),sy.Rational(1,2)),range(4)):
   O=ann[p].T if eps==1 else ann[p]
   charged=D+64*(RF+(spectator+u)*sy.eye(16))**2
   delta=eps*R[p,p]+u
   dc=sum(((cast(sy.Expr,h[q,p])*ann[q].T if eps==1 else -cast(sy.Expr,h[p,q])*ann[q]) for q in range(4)),sy.zeros(16))
   rhs=dc+64*delta**2*O+128*delta*O*(RF+spectator*sy.eye(16))
   assert sy.simplify(charged*O-O*neutral-rhs)==sy.zeros(16);commutators+=1
  raw=receipt['new_raw_grams'][prof['role']]
  K=[[Interval(F(float(raw[i][j]))/N-F(1,1<<24),F(float(raw[i][j]))/N+F(1,1<<24)) for j in range(5)] for i in range(5)]
  Kcenter=[[Interval(F(float(raw[i][j]))/N) for j in range(5)] for i in range(5)]
  for qa in (-1,1):
   sa=[];sb=[];ca=[];cb=[]
   for eps,S,C in ((qa,sa,ca),(-qa,sb,cb)):
    selected=[i for i,w in enumerate(words) if len(w)==2+eps]
    for p in range(2):
     O=ann[p].T if eps==1 else ann[p]
     # Actual offsets u=-eps/2 cancel delta for physical site0.
     dc=D*O-O*D
     S.append((O*W).extract(selected,range(5)))
     C.append((dc*W).extract(selected,range(5)))
   m=[Interval(),Interval(),Interval()]
   mc=[Interval(),Interval(),Interval()]
   for spinA,spinB in itertools.product(range(2),repeat=2):
    y,v=tensor_coeff(sa,sb,ca,cb,M,spinA,spinB)
    m[0]+=tensor_inner(y,y,K);m[1]+=tensor_inner(y,v,K);m[2]+=tensor_inner(v,v,K)
    mc[0]+=tensor_inner(y,y,Kcenter);mc[1]+=tensor_inner(y,v,Kcenter);mc[2]+=tensor_inner(v,v,Kcenter)
   row=next(r for r in receipt['rows'] if r['role']==prof['role'] and r['qA']==qa)
   delta=2*asiv(F(str(L)))*(F(1,65536)+F(receipt['Rayleigh_ground_shift_cap_exact']))
   target_delta=Interval(*row['full_action_residual_shift_norm_price']['exact'])
   assert target_delta.lo<=delta.lo<=delta.hi<=target_delta.hi
   prices=[Interval(),m[0].sqrt()*delta,2*m[2].sqrt()*delta+delta**2]
   centered=[Interval(t.lo-p.hi,t.hi+p.hi) for t,p in zip(m,prices)]
   # Different algebraic groupings need not contain one another. Record
   # each independently valid envelope and their exact endpoint differences;
   # separately require target to contain raw-center action/error arithmetic.
   center_priced=[Interval(t.lo-p.hi,t.hi+p.hi) for t,p in zip(mc,prices)]
   discrepancies=[]
   for independent,central,target in zip(centered,center_priced,row['literal_stored_source_moments']):
    lo,hi=map(F,target['exact'])
    assert independent.lo<=hi and lo<=independent.hi
    assert lo<=central.lo and central.hi<=hi
    discrepancies.append({'independent_minus_target_lower':str(independent.lo-lo),'target_minus_independent_upper':str(hi-independent.hi)})
   moment_checks.append({'role':prof['role'],'qA':qa,'independent_direct_tensor_moments':[t.data() for t in centered],'independent_raw_center_priced':[t.data() for t in center_priced],'raw_center_priced_contained_in_target':True,'envelopes_overlap':True,'endpoint_grouping_differences':discrepancies,'actual_density_commutator_delta':'0','envelope_interpretation':'Overlap is a consistency control, not a substitute for outward derivation. Both enclosures follow valid independent grouping; neither must contain the other.'})
 u=F(1,1<<53)
 def rounding(k):return k*u/(1-k*u)
 # Conservative rational norm bound |E| <= sqrt(2) <=2 for all slots
 # simultaneously; exact extraction isometry is an inherited domain premise.
 e=2*rounding(12)
 formation=2*e+e*e+rounding(65536)*(1+e)**2
 formation=formation+rounding(300)*(1+formation)
 assert formation<F(1,1<<24)
 block_counts={}
 for n,other in ((2,690880),(126,14)):
  d=n*(n+1)*(2*n+1)//6;blocks=(d*other+32767)//32768
  assert blocks<=300;block_counts[str(n)]=blocks
 I=asiv(F(274877906944,2204801412260314559041));eta=(2*I).sqrt()
 targeteta=Interval(*receipt['ground_norm_distance_price']['exact'])
 assert targeteta.lo<=eta.lo<=eta.hi<=targeteta.hi
 aggregate_checks=[]
 for z,agg in receipt['aggregate'].items():
  for key,target in agg.items():
   endpoint_sums=[sum(F(r['physical_copy_count'])*F(next(k for k in r['kernels'] if k['z_exact']==z)[key]['exact'][j]) for r in receipt['rows']) for j in (0,1)]
   assert list(map(F,target['exact']))==endpoint_sums
  for row in receipt['rows']:
   ker=next(k for k in row['kernels'] if k['z_exact']==z)
   for power,key in ((1,'source_ground_kernel_price'),(2,'source_ground_square_price')):
    price=4*F(str(L))**2*eta/(gamma+F(z))**power
    targetprice=Interval(*ker[key]['exact']);assert targetprice.lo<=price.lo<=price.hi<=targetprice.hi
   # Independent rational interval evaluation of spectral bounds on actual
   # stored moments, plus the separately audited exact-ground transfer.
   m0,m1,m2=[Interval(*v['exact']) for v in row['literal_stored_source_moments']]
   zz=F(z);gg=gamma+zz;aa=F(ker['a_exact'])
   n1=m1+zz*m0;n2=m2+2*zz*m1+zz**2*m0
   variance=n2-2*aa*n1+aa**2*m0
   first=m0/aa-(n1-aa*m0)/aa**2+variance/(gg*aa**2)
   second=m0/aa**2-2*(n1-aa*m0)/aa**3+variance*(2*gg+aa)/(gg**2*aa**3)
   for power,key,lower,upper in ((1,'inverse',m0**2/n1,first),(2,'inverse_square',m0**3/n1**2,second)):
    price=4*F(str(L))**2*eta/gg**power
    independent=Interval(max(F(0),lower.lo-price.hi),min(upper.hi,(m0/gg**power).hi)+price.hi)
    target=Interval(*ker[key]['exact'])
    assert target.lo<=independent.lo and independent.hi<=target.hi,(key,z,independent.data(),target.data())
  aggregate_checks.append({'z':z,'exact_labelled_1_plus_63_sum':True})
 rooted=2*F(receipt['aggregate']['0']['inverse_square']['exact'][1])
 assert rooted>1
 out={'verdict':'CONDITIONAL','algebraic_scope':'VALID global scalar inequalities, full CAR commutators and independent moment/price arithmetic under inherited physical source and machine premises',
  'plan_provenance':{'plan_id':'issue-222-induced-photon-delta13-continuation-20261008','captured_governor':'2f78f9876289c8d9411f62e04f8a4f295cb829cca71d92c6aa87a6d44b19723d','accepted_scoped_science':'6f65d328398ea35d60e7e9b64d8c1631832a3a7746b0d22f797b8cafc6f88693','latest_owner_bookkeeping_only':'8115649c915772d522c604be80732ccebdb58383b34cfa1738e0466466d429d9','work_item_id':'induced-photon','plan_owner':'Main; no child writes'},
  'target_sha256':observed,'all_input_sha256':bindings,'immutable_old_zip_members_verified':archive_verified,
  'controls':result,'direct_tensor_moment_checks':moment_checks,'complete_commutator_cases':commutators,
  'formation':{'u':str(u),'conservative_normalized_entry_error':str(formation),'target_budget':'1/16777216 times FiFj','Fi':[1,128],'block_counts':block_counts,'isometry':'inherited exact harmonic/physical extraction domain; no numerical isometry assumption','cancellation_paid':'combined extraction perturbation <=2 gamma12, not merely a relative output error'},
  'ground_transfer':{'infidelity_squared':str(I.lo),'eta_interval':eta.data(),'pair_distance':'<=2eta','quadratic_difference':'<=(||Bs||+||Bg||)||B(s2-g2)|| ||f(X)|| <=4L^2 eta ||f(X)||','stationary_ground':'canonical local frozen g only; s is a variational source, never stationary','Rayleigh_shift':'paid before stored-source moments; same X uses true E0'},
  'aggregate_checks':aggregate_checks,'single_edge_allocation2_inverse_square_upper_exact':str(rooted),
  'root_interpretation':'above1 fails this sufficient guard only; neither instability nor an all-path rooted sum',
  'load_bearing_findings':[],
  'machine_operation_premise':['Binary64 round-to-nearest coefficient conversion, sqrt, CSR products/additions, Gram products/additions and block accumulation satisfy the stated gamma error model.','No underflow/overflow, FTZ/DAZ, nonfinite arithmetic, reduced-precision products or approximate BLAS; this is conditional and not attested by Python/BLAS success.','At most2 CSR terms per row, <=32768 products per block and <=300 accumulations; exact rho integers with absolute value<=128; exact dyadic source and exact normalization N.'],
  'inherited_premises':['Reviewed physical harmonic embedding including all labelled Nj=2 replicas, corrected phase/Dirac profile and frozen local fullcanonical ground.','Charged full containing-space lower floor-5120 for both affine signs; accepted residual including formation1/262144 within cap1/65536.','Accepted squared infidelity and Rayleigh-ground shift; no old solver or certificate producer rerun.'],
  'nonclaims':['No unconditional machine certification or independent recertification of old ground/floor proof.','No physicalPhi or lattice stationary ground, coupling/law adoption, lambda1 continuation, bulk stiffness, IR atom, massless pole, commoncone, originalSOURCE or campaign closure.','One unit axis1 edge only; no all-path/all-order connected rooted expansion.','No finite source moment proxy is relabelled a spectral functional; high-energy tail covered by all-x identities.'],
  'old_cli_executed':False,'target_imported':False,'source_and_inputs_unchanged':all(sha(ROOT/n)==v for n,v in {**PINS,**bindings}.items()),'own_source_sha256':sha(Path(__file__))}
 assert out['source_and_inputs_unchanged']
 path=Path(args.output);path.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'verdict':out['verdict'],'receipt':path.name,'receipt_sha256':sha(path),'commutator_cases':commutators,'independent_rows':len(moment_checks),'root_allocation2_upper':float(rooted)}),flush=True)

if __name__=='__main__':run()
