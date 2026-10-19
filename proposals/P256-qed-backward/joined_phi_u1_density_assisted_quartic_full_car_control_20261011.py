from __future__ import annotations
import hashlib,itertools,json,time
from pathlib import Path
from typing import Any, cast
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix,diags,eye,kron,bmat
from scipy.sparse.linalg import expm_multiply
import bulk_electric_phase_consumer as native
ROOT=Path(__file__).resolve().parent
START=time.monotonic()
INPUT_NAMES=[native.INPUTS[0],native.INPUTS[1],'bulk_electric_phase_typed_20261010.json','joined_phi_u1_full_gauss_quartic_local_so5_assignment_packet_20261011.json']
PINS={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in INPUT_NAMES}
OWN=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
frame,scale=native.frozen.exact_frame((ROOT/INPUT_NAMES[0]).read_bytes())
cfg=json.loads((ROOT/INPUT_NAMES[1]).read_bytes())['input_descriptor']['original_micro_config_UNCHANGED']
profiles=native.frozen.profiles(cfg,frame,scale)
def cx(z):return complex(float(z[0]),float(z[1]))
phi=[np.array([[cx(z) for z in row] for row in a]) for a in frame]
masks=[sum(1<<j for j in z) for z in itertools.combinations(range(8),4)]
index={m:i for i,m in enumerate(masks)}
D=len(masks)
def car_matrix(h):
 rows=[];cols=[];values=[]
 for j,mask in enumerate(masks):
  for p,q in zip(*np.nonzero(h)):
   step=native.car(mask,int(p),int(q))
   if step is not None:
    rows.append(index[step[0]]);cols.append(j);values.append(step[1]*h[p,q])
 return csr_matrix((values,(rows,cols)),shape=(D,D),dtype=np.complex128)
local=list(itertools.combinations(range(4),2))
def epsilon_sign(a):
 b=tuple(i for i in range(4) if i not in a);word=a+b
 return -1 if sum(word[i]>word[j] for i in range(4) for j in range(i+1,4))%2 else 1
seed=np.zeros((D,D),dtype=np.complex128)
for row in local:
 for col in local:
  ma=sum(1<<i for i in row)+sum(1<<(i+4) for i in col)
  seed[index[ma],index[255^ma]]=epsilon_sign(row)*epsilon_sign(col)/6
psi0=seed.reshape(-1)
assert abs(np.vdot(psi0,psi0)-1)<1e-14
ra=np.array([(m&15).bit_count()-2 for m in masks])
rho=(ra[:,None]+ra[None,:]).reshape(-1)
assert np.linalg.norm(rho*psi0)==0
ident=eye(D,format='csr',dtype=cast(Any,np.complex128))
ons=[];hops=[];hop_der=[]
for p in profiles:
 path=next(x for x in p['paths'] if x['axis']==0 and x['length']==1)
 t=np.array([[cx(z) for z in row] for row in path['matrix']])
 # Actual native Phi1 uniform-scale direction; no arbitrary endpoint extension.
 dt=cx(path['coefficients'][1])*phi[1]
 oh=np.zeros((8,8),dtype=np.complex128)
 oh[:4,:4]=oh[4:,4:]=float(p['onsite']/scale)*phi[0]
 hh=np.zeros((8,8),dtype=np.complex128);hh[:4,4:]=t;hh[4:,:4]=t.conj().T
 dh=np.zeros_like(hh);dh[:4,4:]=dt;dh[4:,:4]=dt.conj().T
 ons.append(car_matrix(oh));hops.append(car_matrix(hh));hop_der.append(car_matrix(dh))
hop=kron(hops[0],ident,format='csr')+kron(ident,hops[1],format='csr')
hons=kron(ons[0],ident,format='csr')+kron(ident,ons[1],format='csr')
hd=kron(hop_der[0],ident,format='csr')+kron(ident,hop_der[1],format='csr')
density2=diags(2*rho*rho,format='csr',dtype=np.float64)
C=density2@hop-hop@density2
Cd=density2@hd-hd@density2
H=hons+hop+diags(64*rho*rho,format='csr',dtype=np.float64)
assert np.linalg.norm((C+C.conj().T).data)<1e-12
chi=hop@psi0
assert np.linalg.norm(C@psi0-2*chi)<1e-12
first=-2*np.vdot(H@psi0,C@psi0).real
expected=-4*np.vdot(chi,chi).real
assert abs(first-expected)<1e-10
rho_a=np.repeat(ra,D);rho_b=np.tile(ra,D)
Dind=diags(2*(rho_a*rho_a+rho_b*rho_b),format='csr',dtype=np.float64)
Cind=Dind@hop-hop@Dind
site=diags([1]*4+[0]*4,format='csr',dtype=np.float64).toarray()
qh=[]
for p in profiles:
 path=next(x for x in p['paths'] if x['axis']==0 and x['length']==1)
 t=np.array([[cx(z) for z in row] for row in path['matrix']])
 hh=np.zeros((8,8),dtype=np.complex128);hh[:4,4:]=t;hh[4:,:4]=t.conj().T
 qh.append(car_matrix(hh@site-site@hh))
Q=kron(qh[0],ident,format='csr')+kron(ident,qh[1],format='csr')
def measure(v):
 hv=H@v;e=float(np.vdot(v,hv).real)
 return dict(norm=float(np.vdot(v,v).real),energy=e,matter_energy=float(np.vdot(v,(hons+hop)@v).real),electric_energy=float(64*np.vdot(rho*v,rho*v).real),full_H_action_norm_squared=float(np.vdot(hv,hv).real),full_variance=float(np.vdot(hv,hv).real-e*e),charge_norm=float(np.linalg.norm(rho*v)))
cases=[]
for tau in [1/128,1/64]:
 v=np.asarray(expm_multiply(-tau*C,psi0,traceA=0))
 independent=np.asarray(expm_multiply(-tau*Cind,psi0,traceA=0))
 quadratic=np.asarray(expm_multiply(-tau*Q,psi0,traceA=0))
 m=measure(v);mq=measure(quadratic)
 print(json.dumps(dict(tau=tau,quartic=m,quadratic=mq)),flush=True)
 assert abs(mq['matter_energy'])<1e-10 and mq['electric_energy']>0
 # The larger finite step overshoots: retain it as a rejected source, not descent.
 assert (m['energy']<0) if tau==1/128 else (m['energy']>0)
 assert np.linalg.norm(v-independent)>1e-7
 cropped=v.copy();cropped[abs(rho)>1]=0;cropped/=np.linalg.norm(cropped)
 hc=H@cropped;exterior=hc.copy();exterior[abs(rho)<=1]=0
 tail=float(np.vdot(exterior,exterior).real);assert tail>1e-5
 cases.append(dict(tau=tau,full_nonGaussian= m,wrong_independent_density=measure(independent),wrong_quadratic= mq,common_vs_independent_state_defect=float(np.linalg.norm(v-independent)),charge_crop_changed_source=True,same_cropped_source_unpaid_PHP_squared_exterior=tail,nonclaim='Neither fullH variance is stationary. Conditional TWOcopies/one-edge tree, not all64/N4/wholePhi.',v=v))
v=cases[0].pop('v');cases[1].pop('v');tau=cases[0]['tau']
# Same literal state, native angle/Gauss and fieldscore boundaries.
theta=.37;epsilon=1e-5
vt=np.exp(1j*theta*rho)*v
fd=(np.exp(1j*(theta+epsilon)*rho)*v-np.exp(1j*(theta-epsilon)*rho)*v)/(2*epsilon)
angle_defect=float(np.linalg.norm(fd-1j*rho*vt));assert angle_defect<1e-8
wrong_orientation_defect=float(2*np.linalg.norm(rho*vt));assert wrong_orientation_defect>1e-3
aug=bmat([[-tau*C,-tau*Cd],[None,-tau*C]],format='csr')
jet=np.asarray(expm_multiply(aug,np.concatenate([np.zeros_like(psi0),psi0]),traceA=0))[:len(v)]
fdphi=(np.asarray(expm_multiply(-tau*(C+epsilon*Cd),psi0,traceA=0))-np.asarray(expm_multiply(-tau*(C-epsilon*Cd),psi0,traceA=0)))/(2*epsilon)
score_defect=float(np.linalg.norm(jet-fdphi));score_norm=float(np.linalg.norm(jet))
assert score_defect<1e-8 and score_norm>1e-3
# Independent actual SU4 covariance: transform onebody and recompute exponential.
rng=np.random.default_rng(20261011);gs=[]
for _ in range(2):
 z=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));z=z+z.conj().T;z-=np.trace(z)/4*np.eye(4)
 gs.append(np.asarray(expm(1j*z)))
u=np.zeros((8,8),dtype=np.complex128);u[:4,:4]=gs[0];u[4:,4:]=gs[1]
wd=np.array([[np.linalg.det(u[np.ix_(tuple(i for i in range(8) if a>>i&1),tuple(j for j in range(8) if b>>j&1))]) for b in masks] for a in masks])
# Rebuild the transformed native hop, not a transformed C echo.
transformed_hops=[]
for p in profiles:
 path=next(x for x in p['paths'] if x['axis']==0 and x['length']==1);t=np.array([[cx(z) for z in row] for row in path['matrix']])
 t=gs[0]@t@gs[1].conj().T
 hh=np.zeros((8,8),dtype=np.complex128);hh[:4,4:]=t;hh[4:,:4]=t.conj().T
 transformed_hops.append(car_matrix(hh))
hg=kron(transformed_hops[0],ident,format='csr')+kron(ident,transformed_hops[1],format='csr');cg=density2@hg-hg@density2
vg=np.asarray(expm_multiply(-tau*cg,psi0,traceA=0))
expected_g=(wd@v.reshape(D,D)@wd.T).reshape(-1)
gauge_defect=float(np.linalg.norm(vg-expected_g));assert gauge_defect<1e-12
result: dict[str, Any]
result=dict(kind='FULL_CANONICAL_TWO_ACTUAL_DIRAC_COPY_QUARTIC_METHOD_CONTROL_NOT64_OR_N4_ADMISSION',local_sites=2,colours=4,canonical_particles_eachcopy=4,Dirac_copies='active1 +ONEauxiliary; all63 price/exterior not computed here',basis_percopy=D,fulltensor_dimension=D*D,source='existing localSU4 wedge2xwedge2 epsilon singlet at eachvertex; genuine nonGaussian; pointnorm1',Gauss='NativeONEedge tree: integerE=rho, HE+HC reduced64rho^2; no ambientN4Lplus or 129Theta replacement',native_onsites=[str(p['onsite']) for p in profiles],derivative=first,exact_CAR_derivative_expected=expected,cases=cases,angle_Gauss_defect=angle_defect,wrong_angle_orientation_defect=wrong_orientation_defect,Phi1_scale_direction_score_norm=score_norm,Phi1_scale_direction_Frechet_FD_defect=score_defect,wrong_frozen_Phi_score_price=score_norm**2,fullSU4_covariance_defect=gauge_defect,complete_action=True,variance_stationary=False,source64_exponential_computed=False,fullPhi_source_computed=False,ground=False,phase=False,pole=False,input_sha256=PINS,own_source_sha256=OWN,input_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in PINS.items()),own_source_unchanged=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==OWN,elapsed_seconds=time.monotonic()-START)
result['baseline']=measure(psi0)
result['selected_tau']=tau
result['overshoot_rejected']={'tau':cases[1]['tau'],'positive_same_H_energy':cases[1]['full_nonGaussian']['energy'],'scope':'Finite source step failure, not monotonicity or a Hamiltonian change.'}
(ROOT/'joined_phi_u1_density_assisted_quartic_full_car_control_20261011_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
