from __future__ import annotations
import hashlib, json, math, time
from fractions import Fraction
from pathlib import Path
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parent
NAMES=['local_so5_resum_actual04_20261011_receipt.json','local_so5_resum_actual04_20261011_state.npz','local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json','local_so5_positive_output_row_inertia_certificate_20261011_receipt.json']
PINS={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in NAMES}
OWN=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
r=json.loads((ROOT/NAMES[0]).read_bytes());frame=json.loads((ROOT/NAMES[2]).read_bytes());lower=json.loads((ROOT/NAMES[3]).read_bytes())
assert PINS[NAMES[1]]==r['npz_sha256'] and lower['exact_second_eigenvalue_lower']==-5123
assert lower['input_sha256'][NAMES[2]]==PINS[NAMES[2]]
@njit(cache=False)
def idx(l,p,q):return l*(l+1)*(2*l+1)//6+p*(l+1)+q
@njit(cache=False)
def neighbors(n,m,c,d):
 size=(n+1)*(n+2)*(2*n+3)//6
 target=np.full((size,8),-1,np.int32);value=np.zeros((size,8));diag=np.zeros(size);rho=np.zeros(size)
 for l in range(n+1):
  for p in range(l+1):
   for q in range(l+1):
    j=idx(l,p,q);diag[j]=2*m*(p+q-l);rho[j]=p-q;k=0
    if p<l:target[j,k]=idx(l,p+1,q);value[j,k]=d*math.sqrt((l-p)*(p+1));k+=1
    if p>0:target[j,k]=idx(l,p-1,q);value[j,k]=d*math.sqrt((l-p+1)*p);k+=1
    if q<l:target[j,k]=idx(l,p,q+1);value[j,k]=d*math.sqrt((l-q)*(q+1));k+=1
    if q>0:target[j,k]=idx(l,p,q-1);value[j,k]=d*math.sqrt((l-q+1)*q);k+=1
    if l<n:
     f=math.sqrt((n-l)*(n+l+3)*(l+1)/(2*(l+2)))
     target[j,k]=idx(l+1,p+1,q);value[j,k]=c*f*math.sqrt((p+1)/(l+1))*math.sqrt((l-q+1)/(l+1))*math.sqrt(2);k+=1
     target[j,k]=idx(l+1,p,q+1);value[j,k]=-c*f*math.sqrt((l-p+1)/(l+1))*math.sqrt((q+1)/(l+1))*math.sqrt(2);k+=1
    if l>0:
     f=math.sqrt((n-l+1)*(n+l+2)*l/(2*(l+1)))
     if p>0 and q<l:target[j,k]=idx(l-1,p-1,q);value[j,k]=c*f*math.sqrt(p/l)*math.sqrt((l-q)/l)*math.sqrt(2);k+=1
     if q>0 and p<l:target[j,k]=idx(l-1,p,q-1);value[j,k]=-c*f*math.sqrt((l-p)/l)*math.sqrt(q/l)*math.sqrt(2);k+=1
 return target,value,diag,rho
@njit(cache=False)
def action(x,at,av,ad,ar,bt,bv,bd,br):
 nb=len(bd);out=np.empty(len(x))
 for a in range(len(ad)):
  for b in range(nb):
   j=a*nb+b;v=(ad[a]+bd[b]+64*(ar[a]+br[b])**2)*x[j]
   for k in range(8):
    if at[a,k]>=0:v+=av[a,k]*x[at[a,k]*nb+b]
    if bt[b,k]>=0:v+=bv[b,k]*x[a*nb+bt[b,k]]
   out[j]=v
 return out
start=time.monotonic();params=[]
for item,exact in zip(r['parameters'],frame['profiles']):
 diffs=[abs(Fraction(item['onsite_float'])-Fraction(exact['m_exact'])),abs(Fraction(item['c0'])-Fraction(exact['c0_exact'])),abs(Fraction(item['ci'])-Fraction(exact['d_exact']))]
 assert max(diffs)<Fraction(1,2**48)
 params.append(neighbors(item['spin_replicas'],item['onsite_float'],item['c0'],item['ci']))
with np.load(ROOT/NAMES[1],allow_pickle=False) as data:state=np.array(data['state'],dtype=np.float64,copy=True)
assert state.shape==(9672320,) and np.all(np.isfinite(state))
hv=action(state,*params[0],*params[1]);normal=float(state@state);mu=float((state@hv)/normal)
residual=float(np.linalg.norm(hv-mu*state)/math.sqrt(normal))
assert abs(normal-1)<1e-10 and residual<1e-6
# Standard binary64 arithmetic model, no unobserved solver convergence assumption.
# Every CSR/neighbor row has <=8 offdiagonal entries per role and one diagonal.
# Exact m,c,d differ from the captured binary coefficients by <2^-48.
# For n<=126 each radical expression uses <=32 elementary positive operations;
# product/division/sqrt error <=gamma64, all integer radicands <2^53.
# Both maximum absolute row and column sums of full H are <2^22:
# 64*128^2 +2*20*128 +16*(2*6*128) <2^22.
# This encloses coefficient formation and <=20 term action accumulation by2^-18||v||.
# Residual subtraction and its full9672320 norm use gamma(2D+8)<2^-27.
# All bounds include much larger margins than subnormal absolute error.
dimension=len(state);u=Fraction(1,2**53);gamma=(2*dimension+8)*u/(1-(2*dimension+8)*u)
assert gamma<Fraction(1,2**27)
rowcap=64*128**2+2*20*128+16*(2*6*128)
assert rowcap<2**22
formation=(64*u/(1-64*u))*2**22+Fraction(128*6,2**48)
accumulation=(40*u/(1-40*u))*2**22
subtraction=4*u*(2**22+abs(Fraction(mu)))
# Squared-norm underflow propagates through sqrt; both norm paths need a cap.
underflow=Fraction(100*dimension,2**1074)+Fraction(2,2**490)
assert formation+accumulation+subtraction+underflow<Fraction(1,2**18)
computed=Fraction(residual)
assert computed<Fraction(1,2**20)
# Guard applies to the normalized exact binary source, not a rounded Ritz spacing.
sigma=Fraction(1,2**16);muq=Fraction(mu)
# Both residual and source norms enter the ratio; four gamma covers both.
assert computed/(1-4*gamma)+Fraction(1,2**18)<sigma
rayleigh_lower=muq-sigma;rayleigh_upper=muq+sigma;second=Fraction(-5123)
assert rayleigh_upper<second
separation=second-rayleigh_upper
energy_lower=rayleigh_lower-sigma**2/separation
fidelity_error=sigma**2/separation**2
result=dict(kind='INDEPENDENT_COMPLETE_SERIALIZED_STATE_ACTION_AND_ANALYTIC_ROUNDOFF_TEMPLE_COMPOSITION',source_dimension=dimension,norm_float=normal,energy_float=mu,independent_full_H_residual_float=residual,coefficient_and_action_error_upper_exact='1/262144',normalized_exact_source_residual_upper_exact=str(sigma),Rayleigh_interval_exact=[str(rayleigh_lower),str(rayleigh_upper)],ground_energy_interval_exact=[str(energy_lower),str(rayleigh_upper)],selected_ground_gap_lower_exact=str(separation),selected_ground_infidelity_upper_exact=str(fidelity_error),selected_local_ground_certified_under_binary64_model=True,standard_binary64_correct_rounding_assumed=True,proof='Independent native neighbor full H action on immutable actual04 state (not solver rerun). Exact phase-bound representation and outward lower-form certificate give lambda2>=-5123. Analytic coefficient/action/norm rounding cap bounds true source residual and Rayleigh interval. Temple plus minmax identifies unique selected-local ground and SQUARED source projection error (infidelity), not projector norm. No other spin representations/Phi/lattice ground.',whole_rho_zero_projector_is_spectral_ground_Q0=False,bulk_stiffness=False,fullPhi_ground=False,phase=False,pole=False,own_source_sha256=OWN,input_sha256=PINS,input_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in PINS.items()),source_unchanged=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==OWN,elapsed_seconds=time.monotonic()-start)
(ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
