"""Independent exact LDL and canonical spectral-cover integration.

Consumes immutable NEW certificates; never reruns a ground/certificate solver.
Exact frozen canonical H0 scope is separate from physicalPhi/lattice ground.
Neutral source/moment numbers remain FLOAT. Review verdicts gate publication,
not numerical truth: their explicit proof hypotheses remain recorded.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
from typing import Any

ROOT=Path(__file__).resolve().parent


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def ldl_inertia(matrix: list[list[int]]) -> dict[str,Any]:
    # Fraction LDL is independent of the producer's fraction-free Bareiss.
    a=[[F(x) for x in row] for row in matrix]
    assert all(a[i][j]==a[j][i] for i in range(len(a)) for j in range(len(a)))
    signs=[]
    for k in range(len(a)):
        pivot=a[k][k];assert pivot!=0
        signs.append(1 if pivot>0 else -1)
        for i in range(k+1,len(a)):
            for j in range(i,len(a)):
                value=a[i][j]-a[i][k]*a[j][k]/pivot
                a[i][j]=a[j][i]=value
    return {'negative':signs.count(-1),'positive':signs.count(1),'pivot_signs':signs}


def consume(manifest_path: Path, receipt: Path) -> None:
    started=time.monotonic();own=sha(Path(__file__))
    manifest=json.loads(manifest_path.read_text())
    members=manifest['members']
    assert all(sha(ROOT/n)==h for n,h in members.items())
    def data(key: str) -> Any:
        name=manifest['sources'][key];assert name in members
        return json.loads((ROOT/name).read_text())
    audited=[]
    def audit_family(family: dict[str,Any], expected: int) -> None:
        blocks=family['parity_blocks']
        negative=0
        for block in blocks:
            name=block['matrix_path'];assert members[name]==block['matrix_sha256']
            witness=block['witness_path'];assert members[witness]==block['witness_sha256']
            matrix=json.loads((ROOT/name).read_text())
            low=ldl_inertia(matrix['lower_integer']);high=ldl_inertia(matrix['upper_integer'])
            assert low==block['lower_inertia'] and high==block['upper_inertia']
            radius=matrix['radius_integer'];residual=matrix['residual_squared_upper_integer']
            center=matrix['center_integer']
            assert matrix['lower_integer']==[[x-(radius if i==j else 0) for j,x in enumerate(row)] for i,row in enumerate(center)]
            assert matrix['upper_integer']==[[x+(radius+64*residual if i==j else 0) for j,x in enumerate(row)] for i,row in enumerate(center)]
            assert low['negative']==high['negative']
            negative+=low['negative']
            audited.append({'matrix':name,'dimension':len(center),'inertia':low,'witness_hash_checked':True})
        assert negative==expected

    boundary=data('image_lower')
    for family in boundary['families']:
        assert members[family['receipt']]==family['receipt_sha256']
        d=json.loads((ROOT/family['receipt']).read_text())
        assert d['floor_exact']=='-5123' and F(d['omitted_spectrum_lower_exact'])>F(-5123)
        audit_family(d,1 if family['kind']=='vector' else 0)
    quartet=data('quartet')
    assert {tuple(f['roles']) for f in quartet['families']}=={(1,1,1,1),(0,1,1,1),(0,0,1,1)}
    for family in quartet['families']:
        assert family['entire_density_output_image_checked']
        assert family['floor_exact']=='-5123' and F(family['omitted_spectrum_lower_exact'])>F(-5123)
        assert all(x['canonical_number']==256 for x in family['physical_number_allocations'])
        audit_family(family,0)
    mixed=data('mixed')
    assert {tuple(f['roles_vector_number1_number3']) for f in mixed['role_assignments']}=={(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)}
    for family in mixed['role_assignments']:
        assert family['output_budget']==240 and family['support_budget']==322
        assert family['floor_exact']=='-5123' and F(family['omitted_free_H_minus_z_lower_exact'])>0
        assert family['certified_lambda1_above_z']
        audit_family(family,0)
    failed=data('mixed_original_failed')
    assert all(f['negative_lower']==3 and f['negative_upper']==3 for f in failed['role_assignments'])
    # Failure of an output lower form is not an upper bound on the full H.

    cover=data('cover');proof=cover['proof'];enumeration=cover['enumeration']
    assert enumeration['neutral_negative_histograms']==111
    records=enumeration['records']
    assert all(r['covered']['with_scalar_transfer'] for r in records)
    assert any(not r['covered']['literal_new_domains'] for r in records)
    assert any(not r['covered']['old_shortcut'] for r in records)
    assert sum(int(r['labelled_free_basis_multiplicity']) for r in records)==332468225
    assert F(proof['orthogonal_complement']['D_minus_minus5123_lower_interval'][0])>0
    # Check the sharp all-slot free cut using actual outward radicals; no N4
    # source-harmonic or permutation-symmetric assumption enters this arithmetic.
    profile=data('profiles')['profiles']
    def radical_interval(x: str, bits: int=160) -> tuple[F,F]:
        import math
        value=F(x);q=1<<bits;n=math.isqrt(value.numerator*q*q//value.denominator)
        assert F(n,q)**2<=value<=F(n+1,q)**2
        return F(n,q),F(n+1,q)
    e=[radical_interval(p['e_squared_exact']) for p in profile]
    c=F(profile[0]['c0_exact']);assert c==F(profile[1]['c0_exact']) and c<0
    assert e[1][0]>e[0][1] and e[0][0]+3*c>0
    budget_upper=-5123+4*e[0][1]+252*e[1][1]
    assert budget_upper<4*e[0][0]
    assert 2*e[0][0]+4*(e[0][0]+c)>4*e[0][1]
    assert 6*(e[0][0]+c)>4*e[0][1]
    # Number0+3+3 and number4+1+1 are canonical256, but merely replacing
    # scalar2 by number0/4 in a one1/one3 triple is NOT canonical.
    assert 0+3+3==4+1+1==2+1+3==6
    assert 0+1+3!=6 and 4+1+3!=6

    caps=data('caps')
    ground_upper=F(caps['ground_energy_interval_exact'][1])
    gap=-5123-ground_upper
    assert gap==F(boundary['denominator_lower_exact']) and gap>0
    I=F(caps['selected_ground_infidelity_upper_exact'])
    assert 2*I<F(1,60000)**2
    assert F(caps['normalized_exact_source_residual_upper_exact'])==F(1,65536)
    image=data('all64')
    assert image['inputs_unchanged'] and image['source_unchanged']
    assert image['accepted_eta_exact']=='1/65536'
    reference_shift=F(caps['Rayleigh_interval_exact'][1])-F(caps['ground_energy_interval_exact'][0])
    assert F(image['Rayleigh_to_ground_reference_shift_upper_exact'])==reference_shift and reference_shift>0
    assert F(image['missing_source_residual_action_norm_upper_exact'])==2*(F(1,65536)+reference_shift)*F(image['all64_C_norm_upper_exact'])
    assert F(image['source_to_true_ground_Q_return_vector_norm_error_upper_exact'])==F(image['all64_C_norm_upper_exact'])/30000
    assert image['complete_H_variance_claim'] is False and image['ground_or_phase_claim'] is False
    root=data('root_review')
    assert 'INVALID' in root['overall_verdict'] and root['finding']['verdict'].startswith('INVALID')
    physical=data('physical_harmonic_parent')
    assert physical['strict_half_gain'] and physical['negative_baseline_per_site_upper_less_than_minus2pow6070']
    assert F(physical['charged_per_site_increment_upper_exact'])<0
    assert physical['selected_local_source_IS_NOT_fullphysical_source']
    assert physical['firstshell_guard_FAILED_NOT_allorder_or_global_no_go']
    assert physical['heat_and_Feshbach_FLOAT_NOT_INTERVAL']

    reviews={key:data(key) for key in manifest.get('reviews',[])}
    result={
        'kind':'CANONICAL_H0_NEUTRAL_ROOT_CORRECTION_PARENT_CONSUMER_NOT_FULLPHI_OR_PHASE',
        'own_source_sha256':own,'manifest_sha256':sha(manifest_path),
        'task_governor':manifest['task_governor'],'accepted_science_before_packet':manifest['accepted_science'],
        'verified_member_sha256':members,'independent_Fraction_LDL':audited,
        'new_exact_lowfree_cut_and_scalar_transfer_arithmetic':True,
        'ground_composition_scope':'Conditioned on exact profile/joint CAR/positive-output proofs and accepted binary64 source caps: all labelled invariant domain images cover the full negative free subspace. Their spectral projections below -5123 are SAME span(g) on pure-vector images and zero on mixed/odd/fourodd/scalar-transferred images. Coherent invariant union and free-positive orthogonal complement yield ONE canonical local H0 ground, not duplicated image grounds. Independent proof review remains explicit below.',
        'frozen_canonical_ground_claim_pending_independent_proof':'canonical_review' not in reviews,
        'canonical_local_gap_lower_exact':str(gap),
        'whole_P0_is_not_source_P_or_rho_zero':True,
        'old_freecut_shortcut_withdrawn':True,'mixed_failed_lower_form_preserved_then_output_strengthened':True,
        'root_observed_residual_certification_withdrawn':True,
        'source_residual_cap_and_reference_shift_retained':True,
        'all64_neutral_FLOAT':{k:image[k] for k in ('all64_unprojected_neutral_squared_norm_FLOAT','source_product_projector_removed_norm_FLOAT_NOT_exact_P0','all64_commutator_first_FLOAT_conditional_ground_equation','all64_commutator_second_FLOAT_conditional_ground_equation')},
        'physical_charged_trial_increment_exact_negative':True,
        'physical_source_selected_frozen_ground_DISTINCT':True,
        'reviews':reviews,
        'unpaid':['outward all64 density/moment/action enclosures','physicalPhi ground/lattice lift','whole charged higher exterior','normalized N-uniform linked stability','phase/stiffness/IRatom/sharp photon','commoncone/originalSOURCE'],
        'elapsed_seconds':time.monotonic()-started,'source_inputs_unchanged':sha(Path(__file__))==own and all(sha(ROOT/n)==h for n,h in members.items()),
        'fullPhi_ground':False,'lattice_stationarity':False,'bulk_phase':False,'sharp_photon':False,'campaign_complete':False,
    }
    assert result['source_inputs_unchanged'] and not receipt.exists()
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'receipt':receipt.name,'elapsed_seconds':result['elapsed_seconds'],'matrix_blocks':len(audited),'canonical_gap':str(gap),'source_inputs_unchanged':True},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();consume(args.manifest,args.receipt)
