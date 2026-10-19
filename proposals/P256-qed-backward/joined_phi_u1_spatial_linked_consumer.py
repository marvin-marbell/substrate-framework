"""Consume immutable extensive-source observations without rerunning suppliers."""
from __future__ import annotations
import argparse
import ast
from fractions import Fraction
from math import cos, sqrt
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
EXPECTED='c8d6e49783078701a1fa8b129e93f0a942c9552bb04d673c0ec65cffc50913ee'

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def capsule(path: Path, table: dict[str, bytes]) -> dict[str, str]:
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(table.items()):z.writestr(n,b)
    return {'path':str(path),'sha256':sha(path.read_bytes())}

def consume(receipt: Path, output: Path) -> dict:
    raw=receipt.read_bytes();assert sha(raw)==EXPECTED
    # Read exact executed packets, not whichever source currently occupies disk.
    r=json.loads(raw);table={receipt.name:raw,Path(__file__).name:Path(__file__).read_bytes()}
    for key in ('preexecution','executed'):
        path=Path(r[key]['path']);b=path.read_bytes();assert sha(b)==r[key]['sha256'];table[path.name]=b
        with zipfile.ZipFile(path) as z:
            assert {n:sha(z.read(n)) for n in z.namelist()}==r[key]['members']
    prepath=Path(r['preexecution']['path']);expath=Path(r['executed']['path'])
    with zipfile.ZipFile(prepath) as z,zipfile.ZipFile(expath) as zz:
        assert z.namelist()==zz.namelist()
        assert all(z.read(n)==zz.read(n) for n in z.namelist())
        science_source=zz.read('joined_phi_u1_spatial_linked_source.py')
    base=receipt.name.removesuffix('_receipt.json')
    for suffix in ('_exit.json','_stdout.txt','_stderr.txt'):
        p=ROOT/(base+suffix);table[p.name]=p.read_bytes()
    exitinfo=json.loads(table[base+'_exit.json']);assert exitinfo['exit']==0 and exitinfo['natural_exit']
    assert sha(table[base+'_stdout.txt'])==exitinfo['stdout_sha256']
    assert sha(table[base+'_stderr.txt'])==exitinfo['stderr_sha256']
    first='joined_phi_u1_spatial_linked_actual_20261010'
    for suffix in ('_exit.json','_stdout.txt','_stderr.txt','_receipt_preexecution.zip'):
        p=ROOT/(first+suffix);table[p.name]=p.read_bytes()
    initial=json.loads(table[first+'_exit.json']);assert initial['exit']==1 and initial['natural_exit']
    pre=capsule(output.with_name(output.stem+'_preexecution.zip'),table)
    tree=ast.parse(science_source)
    symbols={n.name:n.lineno for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    rows=[]
    assert [c['N'] for c in r['cases']]==[4,6]
    density=r['cases'][0]['PH_density_per_site']
    for c in r['cases']:
        V=c['N']**3;assert c['V']==V and c['cells']==V//4
        assert c['particles_per_species']==2*V and c['charged_species']==64 and c['neutral_particles']==2*V and c['Ref_particles']==4
        assert c['total_particles']==130*V+4
        assert abs(c['PH_density_per_site']-density)<1e-15 and density>0
        assert abs(c['shared_scalar_norm_observed']-1)<1e-12
        assert abs(c['loop_probability_character_observed']-float(Fraction('8/33')))<1e-12
        assert c['contractible_character']['winding_numerator']==[0,0,0]
        assert c['contractible_character']['divergence_max']==0 and c['contractible_character']['harmonic_norm_squared']<1e-25
        assert c['winding_control']['winding']==[1,0,0] and c['winding_control']['norm_squared']==c['N']
        for k in ('norm_projector_density_defect','gauge_H_defect','gauge_source_defect','Berry_max'):
            assert c[k]<1e-8
        assert abs(c['HE_energy']+c['HC_energy']-c['HEHC_energy'])<1e-9 and c['HC_energy']>0
        assert c['wrong_replica_shared_scalar_norm_per_cell']>1e6 and abs(c['wrong_independent_scalar_norm_per_cell']-1)<1e-10
        roles = c['charged_role_observations']
        assert [r['load'] for r in roles] == [1,63]
        onsite = [Fraction(r['onsite_exact']) for r in roles]
        assert onsite[0] != onsite[1]
        s0 = sqrt(c['frozen_G_display']['g0'])
        assert abs(c['reference_charged_energy']+2*V*s0*float(onsite[0]+63*onsite[1])) < 1e-8
        weighted_matter = sum(r['load']*r['matter_energy'] for r in roles)
        assert abs(c['matter_energy']-weighted_matter) < 1e-8
        assert abs(c['matter_energy']-64*roles[0]['matter_energy']) > 1
        weight = [(1+cos(phi)/4)**2/(33/32) for phi in c['raw_cell_phi']]
        # Same hopping paths leave ONLY the independently specified onsite delta.
        for wa,ph,ea,eu in zip(weight,c['raw_cell_PH_weighted_integrand'],
                              roles[0]['cell_energy_weighted_integrand'],
                              roles[1]['cell_energy_weighted_integrand']):
            expected = float(onsite[1]-onsite[0])*s0*(-8*wa+2*ph)
            assert abs((eu-ea)-expected) < 1e-9
        overlap = c['conditional_shared_overlap']
        assert overlap['grid'] > overlap['degree_upper']
        assert abs(overlap['constant_overlap_weight_once']-1) < 1e-12
        assert overlap['wrong_constant_overlap_repeated_weight'] > 1
        assert 0 <= overlap['shared_mean_weight_once'] <= 1+1e-12
        # No overlap-correlation threshold: this observable is numerically blind.
        covariance_lower = (6/11)*2*c['same_H_A_equals_1_control']['PH_loop_character_real']**2
        assert c['shared_PH_pair_connected_covariance_per_cell'] >= covariance_lower > 0
        assert c['variance_decomposition_defect']<1e-6 and c['omitted_exterior_variance_per_copy']>1
        assert c['complete_H_variance_upper_bound']>c['complete_H_variance_lower_bound']>0
        assert c['operator_path_counts']['forward_ordered_paths']==6*V and c['operator_path_counts']['two_link_forward']==3*V and c['operator_path_counts']['AP_seam_paths']>0
        neutral=c['neutral_full_Fourier_exterior'];assert neutral['rank_exact']==2*V and abs(neutral['rank_observed']-2*V)<1e-8 and neutral['exact_variance']==0
        assert neutral['neutral_U1_independence_defect']==0 and neutral['projector_defect']<1e-8
        cheaper=c['same_H_A_equals_1_control']
        assert cheaper['PH_loop_character_real']>1e-12 and abs(cheaper['PH_loop_character_imag'])<1e-12
        assert c['imposed_loop_delta_same_H']>0 and c['retained_cheaper_trial']=='A=1'
        rows.append({
            'N':c['N'],'charged_density':2,
            'PH_density_per_site_A1':cheaper['PH_density_per_site'],
            'contractible_PH_character_A1':cheaper['PH_loop_character_real'],
            'A1_energy_relative_neutral_eigenenergy':cheaper['complete_body_energy_relative_neutral_eigenenergy'],
            'imposed_A_energy_relative_neutral_eigenenergy':c['complete_body_energy_relative_neutral_eigenenergy'],
            'imposed_A_same_H_cost':c['imposed_loop_delta_same_H'],
            'complete_H_variance_lower_bound':c['complete_H_variance_lower_bound'],
            'complete_H_variance_upper_bound_A1':cheaper['complete_H_variance_upper_bound'],
            'complete_H_variance_upper_bound_imposed_A':c['complete_H_variance_upper_bound'],
            'HC_energy_imposed_A':c['HC_energy'],
            'charged_roles':[{k:r[k] for k in ('role','load','onsite_exact','matter_energy','matter_action_norm_upper')} for r in roles],
            'pointwise_matter_variance_by_role':c['pointwise_matter_variance_by_role'],
            'omitted_exterior_variance_per_copy':c['omitted_exterior_variance_per_copy'],
            'conditional_shared_overlap':overlap,
            'overlap_correlation_verdict':'NO DISCRIMINATION from the observed float difference; not claimed exactly zero.',
            'shared_PH_pair_connected_covariance_per_cell':c['shared_PH_pair_connected_covariance_per_cell'],
            'PH_covariance_contrary_independent_fields':0,
            'PH_covariance_Fourier_lower_float_evaluation':covariance_lower,
            'stationarity':'FAIL for both actual normalized candidates','retained_trial':'A=1',
        })
    executed=capsule(output.with_name(output.stem+'_executed.zip'),table)
    result={'kind':'EXECUTED_EXTENSIVE_SPATIAL_LINKED_SOURCE_CONSUMER_CONDITIONAL_BODY_ONLY','source_receipt':{'path':str(receipt),'sha256':sha(raw)},'source_bytes_sha256':sha(science_source),'source_ast_symbols':symbols,'source_natural_exit':0,'first_attempt_natural_exit':1,'first_attempt_correction':'Initial combined PH-loop/overlap-variation assertion failed. Do NOT attribute that failure to absent loop: executed A=1 comparison observes a nonzero1.489964543157e-8 contractible PH coefficient. The imposed A amendment is a separate actual variational shape, not an observed ground mechanism; it raises energy. Retain cheaper A=1, which also fails stationarity.','cases':rows,'preexecution':pre,'executed':executed,'claims':{'real_normalized_full_exterior_state':True,'extensive_fixed_density':True,'shared_field_before_63_products':True,'complete_body_action_upper_and_orthogonal_lower_bound':True,'old_T_or_NEW_Jordan_lift_adopted':False,'full_ground':False,'induced_stiffness_or_pole':False,'imposed_loop_is_ground_ordering_evidence':False,'interval_enclosures':False,'campaign_complete':False},'unpaid':['NEW Jordan fullPhi radial and SU4/Haar lift including their complete action','full-beta minimization','stationary GROUND+D(L), wholeQ0/nonorthogonal source Gram before spectral/pole consumer'],'material_next_decision':'Use the cheaper A=1 finite-density contractible circuit as a local ingredient, not a stationary source. Replace fixed8-per-cell superselection by overlapping intercell number-conserving covariant gates or whole-exterior Slater response. The positive different-cell-number-sector residual cannot be cured by scalar loop amplitude, electric coefficient adjustment or bias polishing.','runtime':{'python':sys.executable,'version':sys.version}}
    result['source_correction_scope'] = {
        'frozen_original_source':'816cffc7','frozen_original_receipt':'ce765963',
        'old_review_verdict':'INVALID for role-collapsed energies/action upper and replicated-weight overlap diagnostic',
        'current':'NEW actual active1+aux63 role-weighted Hamiltonian epoch; one common overlap weight, centered PH correlation',
        'old_epochs_rebound':False,
        'correlation_not_entanglement':True,
        'pointwise_role_variances_not_integrated_full_H_variance':True,
    }
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--receipt',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    r=consume(args.receipt,args.output)
    print(json.dumps({'receipt':str(args.output),'sha256':sha(args.output.read_bytes()),'cases':r['cases'],'claims':r['claims']},indent=2))
