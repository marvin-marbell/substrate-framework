from __future__ import annotations
import hashlib, json, math, time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import cg
import local_so5_resum_source_20261011 as native
ROOT=Path(__file__).resolve().parent
NAMES=['local_so5_resum_source_20261011.py','local_so5_resum_actual04_20261011_receipt.json','local_so5_resum_actual04_20261011_state.npz','local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json','local_so5_positive_output_row_inertia_certificate_20261011_receipt.json']
PINS={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in NAMES}
OWN=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();START=time.monotonic()
r=json.loads((ROOT/NAMES[1]).read_bytes());frame=json.loads((ROOT/NAMES[3]).read_bytes());lower=json.loads((ROOT/NAMES[4]).read_bytes())
assert PINS[NAMES[0]]==r['source_and_input_sha256'][NAMES[0]] and PINS[NAMES[2]]==r['npz_sha256']
parts=[]
for item,exact in zip(r['parameters'],frame['profiles']):
 assert max(abs(F(item[k])-F(exact[j])) for k,j in [('onsite_float','m_exact'),('c0','c0_exact'),('ci','d_exact')])<F(1,2**48)
 h,rho,labels=native.representation(item['spin_replicas'],item['onsite_float'],item['c0'],item['ci']);parts.append((h,rho))
block=native.Block(parts[0][0],parts[1][0],parts[0][1],parts[1][1],0.)
with np.load(ROOT/NAMES[2],allow_pickle=False) as data:v=np.array(data['state'],copy=True)
v/=np.linalg.norm(v);rho=block.rho.reshape(-1);hv=block.action(v);mu=float(v@hv);residual=float(np.linalg.norm(hv-mu*v))
source_norm_squared_float=float(v@v)
mean=float(v@(rho*v));b=rho*v-mean*v;weight=float(b@b)
def lifted_action(x):
 x=np.asarray(x).reshape(-1)
 return block.action(x)-mu*x+v*float(v@x)
A=native.RealOperator(len(v),lifted_action)
# Positive Jacobi solve preconditioner; no Hamiltonian, source or tolerance change.
preconditioner_diagonal=block.diagonal.reshape(-1)-mu+v*v
assert float(np.min(preconditioner_diagonal))>0
M=native.RealOperator(len(v),lambda x:np.asarray(x).reshape(-1)/preconditioner_diagonal)
iterations=0
def callback(x):
 global iterations
 iterations+=1
 if iterations%25==0:print(json.dumps(dict(iteration=iterations,matvecs=block.calls)),flush=True)
x,info=cg(A,b,M=M,rtol=1e-10,atol=0.,maxiter=400,callback=callback)
ax=lifted_action(x);solve_residual=float(np.linalg.norm(b-ax))
print(json.dumps(dict(cg_info=int(info),complete_solve_residual=solve_residual,iterations=iterations)),flush=True)
assert info==0
variational=float(2*b@x-x@ax);xnorm=float(np.linalg.norm(x))
# New complete source moments and exact contact ordering, not finite-u Ritz difference.
jv=block.action(rho*v)-rho*hv
current_weight=float(jv@jv)
hop_v=hv-(parts[0][0].diagonal()[:,None]+parts[1][0].diagonal()[None,:]+64*block.rho**2).reshape(-1)*v
first_moment=float(-.5*v@hop_v)
# Tighter analytic rounding cap is needed for a useful response, not a solver tolerance change.
D=len(v);unit=F(1,2**53);gamma=(2*D+8)*unit/(1-(2*D+8)*unit)
formation=64*unit/(1-64*unit)*2**22+F(128*6,2**48)
accumulation=40*unit/(1-40*unit)*2**22
subtraction=4*unit*(2**22+abs(F(mu)));underflow=F(100*D,2**1074)+F(2,2**490)
action_error=formation+accumulation+subtraction+underflow
assert action_error<F(1,2**24)
sigma=F(1,2**22)
assert (F(residual)+action_error)/(1-4*gamma)<sigma
muq=F(mu);second=F(lower['exact_second_eigenvalue_lower']);gap=second-(muq+sigma);assert gap>1
energy_error=sigma+sigma*sigma/gap
projector_error=sigma/gap
# Whole rank-one ground block is removed; no density-zero or mean-only substitute.
norm_defect=abs(F(source_norm_squared_float)-1)+gamma*abs(F(source_norm_squared_float))/(1-gamma)+underflow
assert norm_defect<F(1,1000)
deltaA=energy_error+2*projector_error+norm_defect
assert deltaA<F(1,1000)
# Source formation is not exact normalization: price both the rank-one norm defect
# and the rounded mean contraction, besides the actual ground projection.
delta_b=384*projector_error+1024*norm_defect+512*gamma
# Rational enclosure of the binary approximate b norm.
bnorm=F(1)
xnorm_upper=F(xnorm)/(1-4*gamma)+F(2,2**490)
assert F(weight)/(1-4*gamma)<1 and xnorm_upper<1
Amin=1-deltaA
lifted_action_error=action_error+16*gamma
solve_residual_upper=F(solve_residual)/(1-4*gamma)+lifted_action_error*xnorm_upper+unit*(1+2**23*xnorm_upper)+underflow
solve_error=solve_residual_upper**2/Amin
# Attempt2's assumed 1e-7 dot-product budget was rejected. Derive an outward
# budget from the actual solve vector; do not weaken a tolerance or omit it.
float_form_error=gamma*(2+2**23*xnorm_upper**2)+lifted_action_error*xnorm_upper**2+underflow
# Resolvent identity and insertion perturbation; bounds apply to exact selected ground.
state_response_error=(2*delta_b*(bnorm+delta_b)/Amin)+(bnorm+delta_b)**2*deltaA/Amin
alpha_lo=F(variational)-float_form_error-state_response_error
alpha_hi=F(variational)+float_form_error+solve_error+state_response_error
curvature_lo=F(128)-F(32768)*alpha_hi;curvature_hi=F(128)-F(32768)*alpha_lo
assert curvature_lo>0
result=dict(kind='ACTUAL_GROUND_PROJECTED_SELECTED_LOCAL_ELECTRIC_MEMORY_NOT_BULK_PHASE',physical_hopping=1,selected_representation='active2 x auxiliary126 harmonic, frozen binaryPhi two-site tree; NOT fullPhi/lattice/other sectors',complete_dimension=D,mu_float=mu,full_source_residual_float=residual,source_residual_upper_exact=str(sigma),gap_lower_exact=str(gap),rho_mean_float=mean,rho_connected_weight_float=weight,current_commutator_weight_float=current_weight,double_commutator_first_moment_float=first_moment,cg_iterations=iterations,cg_info=int(info),complete_projected_resolvent_residual_float=solve_residual,source_resolvent_quadratic_float=variational,source_resolvent_interval_exact=[str(alpha_lo),str(alpha_hi)],incoming_electric_ground_curvature_interval_exact=[str(curvature_lo),str(curvature_hi)],curvature_interval_float=[float(curvature_lo),float(curvature_hi)],bare_contact=128,source_coupling=128,formula='Esecond(0)=128-32768 <Q0 rho Omega,(H-E0)^-1 Q0 rho Omega>; whole selected ground block Q0, NOT P_rho0 or mean-only stationary proxy',rounding_model='binary64 correctly rounded elementary sqrt/operations, no fastmath; positive operator lift H-E0+P0 and interval native parameter/action caps',standard_binary64_model_assumed=True,proof_status='PARENT NEW COMPOSITION PENDING INDEPENDENT REVIEW; no automatic admission from asserted runtime bound',ground_below_gap_weight_probe_not_run='tautological zero under certified selected gap, not a phase discriminator',rho_zero_is_not_spectral_ground=True,local_positive_response_NOT_bulk_stiffness=True,fullPhi_ground=False,N_uniform_linked_stability=False,phase=False,IRatom=False,pole=False,campaign_complete=False,input_sha256=PINS,own_source_sha256=OWN,inputs_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in PINS.items()),source_unchanged=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==OWN,elapsed_seconds=time.monotonic()-START)
result['derived_response_error_bounds']={k:str(z) for k,z in dict(norm_defect=norm_defect,rank_one_ground_lift_error=deltaA,source_insertion_error=delta_b,solve_residual_upper=solve_residual_upper,variational_form_error=float_form_error,state_response_error=state_response_error).items()}
(ROOT/'local_so5_ground_electric_memory_consumer_20261011_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
