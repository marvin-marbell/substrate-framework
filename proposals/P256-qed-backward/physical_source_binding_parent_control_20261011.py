"""Exact frozen coefficients as a trial configuration, never root/ground adoption."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import joined_phi_u1_bulk_flux_ring_consumer as primitive

ROOT=Path(__file__).resolve().parent


def matches_native_spin_frame(matrix: list[list[primitive.C]], block: list[list[primitive.C]]) -> bool:
    expected = [[primitive.ZERO for _ in range(4)] for _ in range(4)]
    for indices in [(0, 3), (1, 2)]:
        for i, row in enumerate(indices):
            for j, col in enumerate(indices):
                expected[row][col] = block[i][j]
    return matrix == expected


def main() -> None:
    sys.set_int_max_str_digits(0)
    p=argparse.ArgumentParser();p.add_argument('--request',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    r=json.loads(a.request.read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in r['pins'].items())
    raw=(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz').read_bytes()
    frame,s=primitive.exact_frame(raw)
    cfg=json.loads((ROOT/'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json').read_text())['input_descriptor']['original_micro_config_UNCHANGED']
    profiles=primitive.profiles(cfg,frame,s)
    frozen=json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles']
    gram=[]
    for mu in range(4):
        for nu in range(mu,4):
            anticom=[[primitive.sum_c([primitive.add(primitive.mul(frame[mu][i][k],frame[nu][k][j]),primitive.mul(frame[nu][i][k],frame[mu][k][j])) for k in range(4)]) for j in range(4)]for i in range(4)]
            scalar=anticom[0][0][0]
            assert all(anticom[i][j]==(scalar if i==j else F(0),F(0))for i in range(4)for j in range(4))
            if mu==nu:gram.append(scalar/2)
            else:assert scalar==0
    assert min(gram)/2>F(1,20)
    rows=[]
    for profile, old in zip(profiles,frozen,strict=True):
        assert profile['role']==old['role'] and profile['count']==old['count']
        assert profile['onsite']==F(old['m_exact'])
        t=next(x['matrix'] for x in profile['paths']if x['axis']==0 and x['length']==1)
        expected=[[( -F(old['c0_exact']),F(0)),(F(0),F(old['d_exact']))],[(F(0),F(old['d_exact'])),(F(old['c0_exact']),F(0))]]
        assert matches_native_spin_frame(t, expected)
        wrong_cross_spin = [row.copy() for row in t]
        wrong_cross_spin[0][1] = primitive.add(wrong_cross_spin[0][1], primitive.ONE)
        assert all([[wrong_cross_spin[i][j] for j in indices] for i in indices] == expected for indices in [(0, 3), (1, 2)])
        assert not matches_native_spin_frame(wrong_cross_spin, expected)
        assert 2*profile['onsite']!=F(old['m_exact'])
        rows.append({'role':profile['role'],'count':profile['count'],'onsite_exact':str(profile['onsite']),'two_native_spin_blocks_exact':True,'entire_four_by_four_spin_frame_exact':True,'wrong_cross_spin_coupling_rejected':True,'wrong_double_Phi_rejected':True})
    result={'kind':'EXACT_CURRENT_NATIVE_FULLPHI_CONFIGURATION_TO_FROZEN_G_BINDING_NOT_GROUND','Phi_frame_square_exact':[str(x)for x in gram],'h_min_exact':str(min(gram)/2),'roles':rows,'same_operator_literal_field_not_mean_substitution':True,'source_centre_not_root_or_coupling_adoption':True,'actual_nonzero_centre_needs_equivariant_Gauss_lift':True,'selected_canonical_g_not_fullphysical_ground':True,'old_main_solver_root_checker_executed':False,'campaign_complete':False}
    with a.output.open('x')as out:json.dump(result,out,indent=2);out.write('\n')
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in r['pins'].items())
    print(json.dumps({'roles':[x['role']for x in rows],'exact_native_spin_blocks':4,'ground_admitted':False}))


if __name__=='__main__':main()
