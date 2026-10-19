from __future__ import annotations
import argparse, hashlib, json, time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--review-bindings',required=True);args=parser.parse_args()
contract=json.loads((ROOT/args.review_bindings).read_bytes())
OWN=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();START=time.monotonic()
for name,sha in contract['input_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
load=lambda name:json.loads((ROOT/name).read_bytes())
physical=load('physical_singlet_circuit_epoch4_proof_20261011.json')
frame=load('local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json')
inertia=load('local_so5_positive_output_row_inertia_certificate_20261011_receipt.json')
upper=load('local_so5_ground_upper_complete_state_consumer_20261011_receipt.json')
quartic=load('joined_phi_u1_density_assisted_quartic_full_car_control_20261011_receipt.json')
assert contract['parent_review_complete'] and contract['energy_valid'] and contract['domain_valid']
assert contract['local_phase_pair_valid'] and contract['positive_row_valid'] and contract['upper_roundoff_valid']
assert frame['corrected_source_unitarity_H_density_identities'] and frame['parent_pair_reusable_after_new_input_binding']
# Compose per-site original physical Hamiltonian prices, never the frozen local body energies.
eta=F(1,2**4200);chi=F(1,2**4096);lam=F(2**8192);bigf=F(2**10240);alpha=F(1,2**20);t=F(2**7168)
m2=F(physical['uniform_prices']['neutral_M2_exact']);m4=F(2**24)
neutral=(-2*alpha*m2+F(4,3)*alpha**3*m4)*lam/F(2**2100)
positive=60*chi/eta+3000*lam*eta**2+F(9,2)*bigf*eta+F(27,4)*chi*t+F(7,5)*bigf/t*(450+48*7168+24)+2**30*chi/eta+2**30*chi+2**30+F(2**24,2**2100)
assert neutral+positive<-F(2**6070)
assert -neutral+positive>=-F(2**6070)  # wrong physical neutral sign fails the outcome
# Restricted local ground composition: second eigenvalue must exceed a paid source upper.
assert inertia['lower_inertia']['negative']==inertia['upper_inertia']['negative']==1
second=F(inertia['exact_second_eigenvalue_lower']);lo,hi=map(F,upper['Rayleigh_interval_exact']);sigma=F(upper['normalized_exact_source_residual_upper_exact'])
gap=second-hi;assert gap>1
emin=lo-sigma**2/gap;assert emin==F(upper['ground_energy_interval_exact'][0])
projector_error=sigma**2/gap**2
assert projector_error==F(upper['selected_ground_infidelity_upper_exact'])
assert upper['whole_rho_zero_projector_is_spectral_ground_Q0'] is False
# A plausible wrong input crop is NOT a lower electric form; output projection is.
R=np.array([[1.,1.],[1.,1.]]);P=np.diag([1.,0.]);full=R.T@R
output_lower=R.T@P@R;input_crop=P@full@P
assert np.linalg.eigvalsh(full-output_lower)[0]>=-1e-14
wrong_input_crop_min=float(np.linalg.eigvalsh(full-input_crop)[0]);assert wrong_input_crop_min<-1
# No inference of stationarity from energetic descent or a dropped exterior.
selected=quartic['cases'][0]['full_nonGaussian'];baseline=quartic['baseline']
assert selected['energy']<baseline['energy'] and selected['full_variance']>baseline['full_variance']
assert quartic['overshoot_rejected']['positive_same_H_energy']>0
assert quartic['cases'][0]['same_cropped_source_unpaid_PHP_squared_exterior']>1
result=dict(kind='COMPOSED_PARENT_PHYSICAL_SOURCE_AND_RESTRICTED_LOCAL_GROUND_CONSUMER',physical_source=dict(exact_energy_upper_per_site='less than -2^6070',finite_even_N_ge4=True,physical_Gauss=True,domain='D(H), not D(H^2)',actual_full_H_variance_computed=False,neutral_driven_not_charged_only=True,source_shape_not_coupling=True,normalized_Clifford_ground_identification=False),restricted_local_ground=dict(energy_interval_exact=[str(emin),str(hi)],selected_gap_lower_exact=str(gap),ground_infidelity_upper_exact=str(projector_error),scope='fixed canonical spin/top harmonic active2 x auxiliary126 frozen binaryPhi one-edge tree, standard binary64 guard; not other sectors/fullPhi/lattice ground',rho_zero_is_not_ground_Q0=True,incoming_u_not_bulk_stiffness=True),controls=dict(wrong_input_PHP_lower_form_min_eigenvalue=wrong_input_crop_min,quartic_two_copy_descent=selected['energy'],quartic_full_variance=selected['full_variance'],baseline_full_variance=baseline['full_variance'],larger_step_rejected=quartic['overshoot_rejected']['positive_same_H_energy']),joined_physicalGauss_stationary_ground=False,fullPhi_ground=False,N_uniform_phase=False,IRatom=False,sharp_photon=False,commoncone=False,originalSOURCE=False,campaign_complete=False,input_sha256=contract['input_sha256'],own_source_sha256=OWN,inputs_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in contract['input_sha256'].items()),source_unchanged=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==OWN,elapsed_seconds=time.monotonic()-START)
(ROOT/'joined_phi_u1_full_physical_source_local_ground_parent_consumer_20261011_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
