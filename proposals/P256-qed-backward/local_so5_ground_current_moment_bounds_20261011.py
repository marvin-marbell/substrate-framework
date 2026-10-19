from __future__ import annotations
import hashlib, json, time
from fractions import Fraction as F
from pathlib import Path
from typing import Any
import numpy as np
import local_so5_resum_source_20261011 as native

ROOT=Path(__file__).resolve().parent
NAMES=['local_so5_ground_electric_memory_consumer_20261011_receipt.json','local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json','local_so5_resum_source_20261011.py']
PINS={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in NAMES}
OWN=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();START=time.monotonic()
r=json.loads((ROOT/NAMES[0]).read_bytes());frame=json.loads((ROOT/NAMES[1]).read_bytes())
assert r['inputs_unchanged'] and r['source_unchanged'] and r['standard_binary64_model_assumed']
assert PINS[NAMES[1]]==r['input_sha256'][NAMES[1]] and PINS[NAMES[2]]==r['input_sha256'][NAMES[2]]
# Four within-l and four between-l neighbors per role. All have |Delta rho|=1.
# sqrt((l-k)(k+1)) <= (n+1)/2; each radial CG amplitude <= n+3.
# Hence both the hopping and i[H,rho] operator norms have this row/column bound.
B=sum((2*abs(F(p['d_exact']))*(int(p['spin_replicas'])+1)+4*abs(F(p['c0_exact']))*(int(p['spin_replicas'])+3) for p in frame['profiles']),F(0))
assert B<2**12
D=int(r['complete_dimension']);unit=F(1,2**53);gamma=(2*D+8)*unit/(1-(2*D+8)*unit)
underflow=F(100*D,2**1074)+F(2,2**490)
sigma=F(r['source_residual_upper_exact']);gap=F(r['gap_lower_exact'])
guards={k:F(v) for k,v in r['derived_response_error_bounds'].items()}
e=guards['norm_defect'];source_distance=2*sigma/gap+e
formation=64*unit/(1-64*unit)*2**22+F(128*6,2**48)
accumulation=40*unit/(1-40*unit)*2**22
subtraction=4*unit*(2**22+abs(F(r['mu_float'])))
action_error=formation+accumulation+subtraction+underflow
# A fresh physical current is NOT inferred by combining Pi and matter probes.
# This is the single-tree continuity current i[H,rho] in the certified sector.
m0=F(r['rho_connected_weight_float']);m1=F(r['double_commutator_first_moment_float']);m2=F(r['current_commutator_weight_float'])
db=guards['source_insertion_error']
m0_error=2*db+db**2+gamma*abs(m0)/(1-gamma)+underflow
# Hv-Dv subtraction, diagonal formation and dot-product rounding are retained.
off_action_error=2*action_error+32*unit*2**22
m1_error=B*(2+e)*source_distance/2+(gamma*(B+off_action_error)+off_action_error)*(1+e)**2/2+underflow
# Main current used H(rho v)-rho Hv, not an exact diagonal cancellation in float.
j_round_error=512*action_error+1024*unit*2**22+underflow
j_error=B*source_distance+j_round_error
assert m2/(1-4*gamma)<151**2
m2_error=2*151*j_error+j_error**2+gamma*abs(m2)/(1-gamma)+underflow
intervals=[(m0-m0_error,m0+m0_error),(m1-m1_error,m1+m1_error),(m2-m2_error,m2+m2_error)]
assert all(lo>0 for lo,hi in intervals)
# Strict positive energy variance rejects SINGLE-energy saturation, not any atom.
variance_lower=intervals[2][0]/intervals[0][1]-(intervals[1][1]/intervals[0][0])**2
assert variance_lower>0
alpha_moment_lower=intervals[0][0]**2/intervals[1][1]
alpha_lo,alpha_hi=map(F,r['source_resolvent_interval_exact'])
assert alpha_moment_lower<alpha_hi

controls=[]
for p in frame['profiles']:
    m,c,d=(float(F(p[k])) for k in ['m_exact','c0_exact','d_exact'])
    h,rho,labels=native.representation(2,m,c,d);dense=h.toarray();R=np.diag(rho)
    hop=dense-np.diag(dense.diagonal());current=dense@R-R@dense
    halfcomm=(R@current-current@R)/2
    shift_defect=max(abs(rho[i]-rho[j])-1 for i,j in zip(*np.nonzero(hop)))
    assert shift_defect==0 and np.linalg.norm(halfcomm+hop/2)<1e-12
    H=dense+64*R@R;energies,vectors=np.linalg.eigh(H)
    g=vectors[:,0];highest=vectors[:,-1]
    ground_m1=float(g@halfcomm@g);excited_m1=float(highest@halfcomm@highest)
    assert ground_m1>0 and excited_m1<0
    # A stationary excited eigenvector has negative spectral first moment.
    # Stationary alone cannot replace GROUND in positive-measure arguments.
    omitted,_,_=native.representation(2,m,0.,d)
    wrong_comm=(R@(omitted.toarray()@R-R@omitted.toarray())-(omitted.toarray()@R-R@omitted.toarray())@R)/2
    missing_contact_defect=float(np.linalg.norm(halfcomm-wrong_comm))
    assert missing_contact_defect>1
    controls.append(dict(n=2,ground_first_moment=ground_m1,stationary_excited_first_moment=excited_m1,omitted_c0_contact_defect=missing_contact_defect,actual_offdiagonal_charge_shift=1,scope='independent finite operator control only, not all64 state evolution'))
result:dict[str,Any]=dict(kind='SELECTED_GROUND_DENSITY_CONTINUITY_CURRENT_MOMENT_INTERVAL_COMPOSITION',source_dimension=D,selected_local_only=True,physical_hopping=1,hop_and_current_norm_upper_exact=str(B),source_distance_upper_exact=str(source_distance),moment_units=['charge_squared','energy_times_charge_squared','energy_squared_times_charge_squared'],moment_intervals_exact=[[str(lo),str(hi)] for lo,hi in intervals],moment_intervals_float=[[float(lo),float(hi)] for lo,hi in intervals],positive_energy_variance_lower_exact=str(variance_lower),positive_energy_variance_lower_float=float(variance_lower),inverse_density_response_moment_lower_exact=str(alpha_moment_lower),inverse_density_response_moment_lower_float=float(alpha_moment_lower),not_single_energy_saturation=True,not_absence_of_spectral_atoms=True,current='i[H,rho], single-tree native continuity current; NOT combined 3D/Pi/matter vector probe',controls=controls,proof_status='PARENT DERIVED PRICES PENDING INDEPENDENT REVIEW',fullPhi_ground=False,bulk_stiffness=False,N_uniform_phase=False,sharp_photon=False,campaign_complete=False,input_sha256=PINS,own_source_sha256=OWN,inputs_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha for n,sha in PINS.items()),source_unchanged=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==OWN,elapsed_seconds=time.monotonic()-START)
(ROOT/'local_so5_ground_current_moment_bounds_20261011_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
