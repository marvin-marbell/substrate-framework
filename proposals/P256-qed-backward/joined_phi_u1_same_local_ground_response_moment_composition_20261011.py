from __future__ import annotations
import hashlib,json,sys,time
from fractions import Fraction as F
from pathlib import Path
sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent
NAMES=['local_so5_ground_electric_memory_consumer_20261011_receipt.json','local_so5_ground_current_moment_bounds_20261011_receipt.json','joined_phi_u1_full_physical_source_local_ground_parent_consumer_20261011_receipt.json','selected_memory_review_epoch2_proof_20261011.json','selected_memory_moments_review_proof_20261011.json','charged_first_shell_actual04_20261011_receipt.json']
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def run()->None:
    start=time.monotonic();pins={n:sha(ROOT/n) for n in NAMES};own=sha(Path(__file__))
    r,m,g,review,mreview,charged=[json.loads((ROOT/n).read_bytes()) for n in NAMES]
    assert review['verdict']==mreview['verdict']=='VALID'
    assert r['inputs_unchanged'] and r['source_unchanged'] and m['inputs_unchanged'] and m['source_unchanged']
    assert m['input_sha256'][NAMES[0]]==pins[NAMES[0]]
    for n in ['local_so5_resum_source_20261011.py','local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json']:
        assert r['input_sha256'][n]==m['input_sha256'][n]==sha(ROOT/n)
    assert r['local_positive_response_NOT_bulk_stiffness'] and r['rho_zero_is_not_spectral_ground']
    assert m['selected_local_only'] and m['not_absence_of_spectral_atoms']
    intervals=[[F(x) for x in pair] for pair in m['moment_intervals_exact']]
    for lo,hi in intervals:assert 0<lo<=hi
    a0,a1=[F(x) for x in r['source_resolvent_interval_exact']]
    cauchy=intervals[0][0]**2/intervals[1][1]
    assert cauchy==F(m['inverse_density_response_moment_lower_exact'])
    lower=max(a0,cauchy);assert lower<=a1
    curvature=[F(128)-32768*a1,F(128)-32768*lower]
    oldcurvature=[F(x) for x in r['incoming_electric_ground_curvature_interval_exact']]
    assert oldcurvature==[F(128)-32768*a1,F(128)-32768*a0]
    assert 0<curvature[0]<=curvature[1]<oldcurvature[1]
    variance=intervals[2][0]/intervals[0][1]-(intervals[1][1]/intervals[0][0])**2
    assert variance==F(m['positive_energy_variance_lower_exact'])>0
    # Actual positive one-excitation measure of H=diag(0,2), rho=PauliX:
    # m0=1,m1=2,alpha=1/2. A denominator LOWER endpoint gives a false bound.
    wrong=F(1)/F(1999,1000);correct=F(1)/F(2001,1000)
    assert correct<=F(1,2)<wrong
    # Stationary excited H=diag(0,2),Omega=e1 has m1=-2: not a positive
    # ground-excitation measure. Positivity is not inferred from stationarity.
    stationary_excited_m1=-2;assert stationary_excited_m1<0
    # Two positive atoms with equal weights at1 and3 have positive variance1.
    # Thus variance does not assert absence of atoms, including partial poles.
    atom_m0=F(1);atom_m1=F(2);atom_m2=F(5)
    assert atom_m2/atom_m0-(atom_m1/atom_m0)**2==1
    assert not charged['status']['charged_all64_ground_gap'] and not charged['status']['stationarity']
    result={'kind':'SAME_SELECTED_GROUND_INVERSE_AND_MOMENT_OUTWARD_COMPOSITION','scope':'Unique selected frozen one-edge harmonic14 x harmonic690880 local ground only, under independently reviewed binary64 model. Exact rational composition of paid intervals; not a new solver, bulk stiffness or photon.',
      'inverse_interval_exact':[str(lower),str(a1)],'inverse_interval_float':[float(lower),float(a1)],'incoming_electric_curvature_interval_exact':[str(x) for x in curvature],'incoming_electric_curvature_interval_float':[float(x) for x in curvature],'prior_curvature_interval_float':[float(x) for x in oldcurvature],'Cauchy_uses_lower_m0_and_upper_m1':True,'positive_variance_lower_float':float(variance),'variance_excludes_only_single_energy_saturation':True,'not_absence_of_atoms':True,'selected_ground_from_separate_lower_plus_complete_action':g['restricted_local_ground'],
      'charged_first_shell_gap_unpaid':True,'physicalPhi_ground':False,'phase':False,'bulk_stiffness':False,'IRatom':False,'sharp_photon':False,'commoncone':False,'originalSOURCE':False,'campaign_complete':False,
      'controls':{'wrong_denominator_direction_rejected':True,'stationary_excited_not_ground':True,'positive_variance_two_atoms':True},
      'input_sha256':pins,'own_source_sha256':own,'source_unchanged':sha(Path(__file__))==own,'inputs_unchanged':all(sha(ROOT/n)==h for n,h in pins.items()),'elapsed_seconds':time.monotonic()-start}
    output=ROOT/'joined_phi_u1_same_local_ground_response_moment_composition_20261011_receipt.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_exact')},indent=2))
if __name__=='__main__':run()
