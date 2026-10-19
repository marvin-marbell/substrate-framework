"""Independent bounded native source/frame/Gauss controls; no ground solver."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, cast
from scipy.sparse import csr_matrix
import numpy as np
from scipy.linalg import expm
import sympy as sp

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def norm(a):
    return float(np.linalg.norm(a))


def dgamma(h, occupations):
    """Signed canonical number-preserving CAR, independently from target."""
    masks = [sum(1 << j for j in row) for row in occupations]
    lookup = {m: i for i, m in enumerate(masks)}
    out = np.zeros((len(masks), len(masks)), complex)
    for j, original in enumerate(masks):
        for q in range(h.shape[0]):
            if not original >> q & 1:
                continue
            s1 = (-1) ** ((original & ((1 << q) - 1)).bit_count())
            removed = original ^ (1 << q)
            for p in range(h.shape[0]):
                if removed >> p & 1:
                    continue
                s2 = (-1) ** ((removed & ((1 << p) - 1)).bit_count())
                out[lookup[removed | (1 << p)], j] += s1 * s2 * h[p, q]
    return out


def minors(matrix, rows, columns):
    return np.array([[np.linalg.det(matrix[np.ix_(r, c)]) for c in columns] for r in rows])


def replica_map(slot, physical):
    pairs = list(itertools.combinations(range(4), 2))
    fours = list(itertools.combinations(range(8), 4))
    columns = [list(a) + [v + 4 for v in b] for a in pairs for b in pairs]
    return minors(physical, fours, columns) @ np.kron(slot, slot)


def j_tensor(n, bridge):
    if n == 0:
        return np.ones((1, 1))
    previous = j_tensor(n - 1, bridge)
    return np.vstack([previous @ e.toarray() for e in bridge.extraction(n)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, expected in request['pins'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    import physical_ground_lift_source_20261011 as source
    bridge = source.bridge
    primitive = source.primitive.frozen
    exact, scale = primitive.exact_frame((ROOT / source.primitive.INPUTS[0]).read_bytes())
    centre = np.array([[[complex(float(z[0]), float(z[1])) for z in row] for row in m] for m in exact])
    exact_phi = [sp.Matrix([[sp.Rational(z[0].numerator, z[0].denominator) + sp.I * sp.Rational(z[1].numerator, z[1].denominator) for z in row] for row in m]) for m in exact]
    cfg = json.loads((ROOT / source.primitive.INPUTS[1]).read_text())['input_descriptor']['original_micro_config_UNCHANGED']
    profiles = primitive.profiles(cfg, exact, scale)
    accepted = json.loads((ROOT / 'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles']
    pairs = list(itertools.combinations(range(4), 2))
    fours = list(itertools.combinations(range(8), 4))
    rows = [sum(1 << k for k in row) for row in pairs]
    slot = source.native_slot_map()[rows]
    J, _, h4, _, _, V = bridge.car_exact()
    slot_complex = np.array((V * J).evalf(), complex)[rows]
    gamma = exact_phi[1] / sp.Rational(float(centre[1, 0, 3].real))
    spin = sp.Matrix.hstack(sp.eye(4)[:, 0], gamma[:, 0], sp.eye(4)[:, 1], gamma[:, 1])
    assert spin.H * spin == sp.eye(4)
    phase = sp.diag(1, -1, -1, 1)
    exact_f = sp.zeros(8)
    for s in range(2):
        for site in range(2):
            exact_f[4*site:4*site+4, 4*s+2*site:4*s+2*site+2] = spin[:, 2*s:2*s+2]
    exact_f *= sp.diag(phase, phase)
    physical = np.array(exact_f, complex)
    L = replica_map(slot, physical)
    density = np.diag([sum(j < 4 for j in row) - 2 for row in fours])
    rho1 = bridge.native_matrix(1, 0., 0., 0.)[1]
    expected_density = np.diag((rho1[:, None] + rho1[None, :]).ravel())
    controls: dict[str, Any] = {'slot_isometry_error': norm(slot.conj().T @ slot - np.eye(5)), 'labelled_two_spin_isometry_error': norm(L.conj().T @ L - np.eye(25)), 'labelled_two_spin_density_intertwining_error': norm(density @ L - L @ expected_density)}
    JT = j_tensor(2, bridge)
    LJ = L @ JT
    controls['native70_J2_isometry_error'] = norm(LJ.conj().T @ LJ - np.eye(14))
    controls['native70_J2_Gauss_intertwining_error'] = norm(density @ LJ - LJ @ np.diag([p-q for l,p,q in bridge.labels(2)]))
    role_rows = []
    for profile, frozen in zip(profiles, accepted, strict=True):
        m, c, d = [sp.Rational(frozen[key]) for key in ['m_exact', 'c0_exact', 'd_exact']]
        onsite = m * exact_phi[0] / sp.Rational(scale.numerator, scale.denominator)
        path = next(z['matrix'] for z in profile['paths'] if z['axis'] == 0 and z['length'] == 1)
        hopping = sp.Matrix([[sp.Rational(z[0].numerator,z[0].denominator)+sp.I*sp.Rational(z[1].numerator,z[1].denominator) for z in row] for row in path])
        native8 = sp.Matrix.vstack(sp.Matrix.hstack(onsite, hopping), sp.Matrix.hstack(hopping.H, onsite))
        expected4 = m*h4[0] + c*h4[1] + d*h4[2]
        assert exact_f.H * native8 * exact_f == sp.diag(expected4, expected4)
        assert exact_f.H * sp.diag(*([sp.Rational(1,2)]*4+[-sp.Rational(1,2)]*4)) * exact_f == sp.diag(h4[3],h4[3])
        H = dgamma(np.array(native8, complex), fours)
        harmonic = bridge.native_matrix(2, float(m), float(c), float(d))[0].toarray()
        correct_error = norm(H @ LJ - LJ @ harmonic)
        no_phase = replica_map(slot, physical @ np.array(sp.diag(phase, phase), complex)) @ JT
        wrong_real_phase = replica_map(slot_complex, physical) @ JT
        wrong_orientation8 = sp.Matrix.vstack(sp.Matrix.hstack(onsite,hopping.H),sp.Matrix.hstack(hopping,onsite))
        wrong_orientation = norm(dgamma(np.array(wrong_orientation8,complex),fours) @ LJ - LJ @ harmonic)
        role_rows.append({'role':profile['role'],'count':profile['count'],'exact_8x8_same_h_and_density':True,'native70_same_h_intertwining_error':correct_error,'wrong_omit_oneparticle_phase_gap':norm(H@no_phase-no_phase@harmonic),'wrong_omit_real_harmonic_slot_phase_gap':norm(H@wrong_real_phase-wrong_real_phase@harmonic),'wrong_reverse_hopping_orientation_gap':wrong_orientation})
        assert correct_error < 1e-10
        assert min(role_rows[-1][k] for k in ['wrong_omit_oneparticle_phase_gap','wrong_omit_real_harmonic_slot_phase_gap','wrong_reverse_hopping_orientation_gap']) > 1.
    controls['roles'] = role_rows
    for n in [1, 2, 3, 4]:
        es = bridge.extraction(n)
        gram = cast(csr_matrix, sum((e.conj().T @ e for e in es)))
        tensor = j_tensor(n, bridge)
        error = norm(gram.toarray()-np.eye(bridge.dim(n)))
        assert error < 1e-12
        controls[f'J{n}_normalized_extraction_gram_error'] = error
        controls[f'J{n}_recursive_tensor_isometry_error'] = norm(tensor.conj().T @ tensor - np.eye(bridge.dim(n)))
    labels126 = bridge.labels(126)
    assert bridge.dim(2) * bridge.dim(126) == 9672320
    with np.load(ROOT / source.STATE, allow_pickle=False) as data:
        x = data['state'].reshape(14, 690880)
    x = x / np.linalg.norm(x)
    trial = source.LiftedGround(x, centre)
    rho_a = np.array([p-q for l,p,q in bridge.labels(2)])
    rho_b = np.array([p-q for l,p,q in labels126])
    totalrho = rho_a[:,None] + rho_b[None,:]
    first = norm(totalrho*x)
    controls['immutable_x_dimension'] = int(x.size)
    controls['immutable_x_normalized_norm'] = norm(x)
    controls['actual_totalrho_mean'] = float(np.sum(totalrho*abs(x)**2))
    controls['actual_totalrho_first_action_norm'] = first
    controls['actual_totalrho_squared_action_norm'] = norm(totalrho**2*x)
    controls['wrong_cancel_only_mean_Gauss_gap'] = first
    controls['wrong_omit_active_aux_rho_cross_gap'] = norm(2*rho_a[:,None]*rho_b[None,:]*x)
    controls['wrong_replace_actual1_mass_with_auxiliary_mass_gap'] = norm(float(profiles[0]['onsite']-profiles[1]['onsite']) * (bridge.native_matrix(2,1.,0.,0.)[0] @ x))
    controls['wrong_unsubtracted_background128_Gauss_gap'] = norm(128*x)
    assert first > .1 and controls['wrong_omit_active_aux_rho_cross_gap'] > .1
    basis = []
    for i in range(4):
        for j in range(i+1,4):
            a=np.zeros((4,4),complex);a[i,j]=a[j,i]=.5;basis.append(a)
            a=np.zeros((4,4),complex);a[i,j]=-.5j;a[j,i]=.5j;basis.append(a)
    for k in range(1,4):
        basis.append(np.diag([1.]*k+[-float(k)]+[0.]*(3-k))/math.sqrt(2*k*(k+1)))
    rng=np.random.default_rng(222101103)
    gr=expm(1j*sum(rng.normal()*b for b in basis));gc=expm(1j*sum(rng.normal()*b for b in basis))
    transformed=np.array([gr@z@gr.conj().T for z in centre])
    theta=.37
    left,right=trial.transport(transformed,source.Jet(gr@gc.conj().T),source.Jet(theta))
    transformed_f=np.zeros((8,8),complex)
    for s in range(2):
        for site,endpoint in enumerate([left.v,right.v]):
            transformed_f[4*site:4*site+4,4*s+2*site:4*s+2*site+2]=endpoint[:,2*s:2*s+2]
    transformed_f=transformed_f@np.array(sp.diag(phase,phase),complex)
    actual=replica_map(slot,transformed_f)@JT
    expected=minors(np.block([[gr,np.zeros((4,4))],[np.zeros((4,4)),gc]]),fours,fours)@LJ
    expected=np.exp(1j*theta*np.diag(density))[:,None]*expected
    controls['native70_complete_source_SU4_U1_covariance_error']=norm(actual-expected)
    wrong_partner=np.block([[left.v,np.zeros((4,4))],[np.zeros((4,4)),(gr@gc.conj().T)@left.v*np.exp(-1j*theta)]])
    controls['wrong_partner_U_instead_of_Udagger_covariance_gap']=norm(source.exterior(wrong_partner,4)-source.exterior(np.block([[left.v,np.zeros((4,4))],[np.zeros((4,4)),right.v]]),4))
    controls['wrong_frozen_Phi_frame_covariance_gap']=norm(source.exterior(np.eye(8),4)-source.exterior(np.block([[gr,np.zeros((4,4))],[np.zeros((4,4)),gc]]),4))
    controls['Z4_full_block_descent_error']=norm(source.exterior(1j*np.eye(8),4)-np.eye(70))
    controls['wrong_independent_vertex_Z4_descent_gap']=norm(source.exterior(np.diag([1j]*4+[1.]*4),4)-np.eye(70))
    controls['wrong_particle_sector3_Z4_descent_gap']=norm(source.exterior(1j*np.eye(8),3)-np.eye(56))
    stabilizer=np.zeros((16,4),complex)
    a,b=source.polynomial_pair([source.Jet(z) for z in centre])
    ev,vectors=np.linalg.eigh(a.v)
    body=vectors.conj().T@b.v@vectors
    edges=[(0,2),(2,1),(0,3)]
    incidence=np.zeros((3,4))
    for k,(p,q) in enumerate(edges):incidence[k,p]=1;incidence[k,q]=-1
    controls['phase_tree_incidence_rank']=int(np.linalg.matrix_rank(incidence))
    controls['phase_tree_edge_magnitudes']=[float(abs(body[p,q])) for p,q in edges]
    orbit=np.column_stack([np.concatenate([(1j*(gen@phi-phi@gen)).ravel() for phi in centre]) for gen in basis])
    controls['SU4_orbit_tangent_rank']=int(np.linalg.matrix_rank(orbit))
    fields=np.broadcast_to(centre,(2,4,4,4)).copy()
    shape_at_centre=source.source_shape(fields,np.eye(4),trial,float(source.R))
    positive=[]
    for endpoint in range(2):
        for mu in range(4):
            for gen in basis:
                point=fields.copy();point[endpoint,mu]+=float(source.R)/4096*gen
                positive.append(source.source_shape(point,np.eye(4),trial,float(source.R)))
    controls['full120_positive_coordinate_probes']=len(positive)
    controls['full120_min_shape']=min(positive)
    controls['shape_at_centre']=shape_at_centre
    controls['shape_on_arbitrary_gauge_orbit']=source.source_shape(np.array([transformed,np.array([gc@z@gc.conj().T for z in centre])]),gr@gc.conj().T,trial,float(source.R))
    controls['integer_character_2pi_error']=float(np.max(abs(np.exp(2j*np.pi*totalrho)-1)))
    theta_j=source.Jet(0.,1.,0.)
    le,re=trial.transport_jet(trial.wstar,source.Jet(np.eye(4)),theta_j)
    jet=source.exterior_jet(source.block_jet(le,re),4)
    rho=np.diag(density)
    controls['native70_Gauss_first_error']=norm(jet.d-1j*rho[:,None]*jet.v)
    controls['native70_Gauss_second_error']=norm(jet.dd+rho[:,None]**2*jet.v)
    controls['wrong_one_sided_phase_Gauss_background2_gap']=norm(2*jet.v)
    controls['wrong_Gauss_orientation_gap']=norm(jet.d+1j*rho[:,None]*jet.v)
    # Actual N4 integer incidence and full electric action; no spectral solver.
    box=source.primitive.IntegerBox(4)
    D=np.zeros((box.volume,3*box.volume),dtype=np.int64)
    for edge,(tail,head) in enumerate(zip(box.tail,box.head)):
        D[tail,edge]=1;D[head,edge]=-1
    k=np.zeros(3*box.volume,dtype=np.int64)
    matching=[i for i,site in enumerate(box.sites) if sum(site)%2==0]
    for i,charge in zip(matching,[2,-1,1,-2]+[0]*28):
        k[3*i]=charge
    charge=D@k
    B,chords=box.cycle_basis()
    assert B.shape==(192,129) and np.array_equal(D@B,np.zeros((64,129),dtype=np.int64))
    lap=D@D.T
    potential=np.r_[0.,np.linalg.solve(lap[1:,1:],charge[1:])]
    longitudinal=D.T@potential
    transverse=k-longitudinal
    beta=np.zeros(129);beta[0]=1/6092928/2
    full=k+B@beta
    controls['N4_integer_matching_Gauss_error']=norm(D@k-charge)
    controls['N4_integer_cycle_DB_exact_zero']=not bool(np.any(D@B))
    controls['N4_cycle_dimension']=int(B.shape[1])
    controls['N4_cycle_Gram_trace']=int(np.sum(B.astype(np.int64)**2))
    controls['N4_original_cover_twisted_Gauss_error']=norm(D@full-charge)
    controls['N4_fullE64_Hodge_split_error']=abs(64*norm(full)**2-64*norm(longitudinal)**2-64*norm(transverse+B@beta)**2)
    controls['N4_wrong_omit_transverse_fullE64_energy_gap']=64*norm(transverse)**2
    controls['N4_wrong_pay_HC_twice_energy_gap']=64*norm(longitudinal)**2
    controls['N4_wrong_reverse_matching_orientation_Gauss_gap']=norm(-D@k-charge)
    controls['N4_twist_complete_cross_plus_square_identity_error']=abs(norm(full)**2-norm(k)**2-2*float(k@(B@beta))-norm(B@beta)**2)
    assert controls['N4_cycle_Gram_trace']==738
    assert controls['N4_fullE64_Hodge_split_error']<1e-9
    assert controls['N4_original_cover_twisted_Gauss_error']<1e-10
    assert min(controls[key] for key in ['N4_wrong_omit_transverse_fullE64_energy_gap','N4_wrong_pay_HC_twice_energy_gap','N4_wrong_reverse_matching_orientation_Gauss_gap'])>1.
    assert controls['native70_complete_source_SU4_U1_covariance_error'] < 1e-10
    assert controls['phase_tree_incidence_rank']==3 and controls['SU4_orbit_tangent_rank']==15
    assert controls['full120_min_shape'] > .9 and controls['shape_on_arbitrary_gauge_orbit'] > .999
    assert controls['native70_Gauss_first_error'] < 1e-10 and controls['native70_Gauss_second_error'] < 1e-10
    assert min(controls[k] for k in ['wrong_partner_U_instead_of_Udagger_covariance_gap','wrong_independent_vertex_Z4_descent_gap','wrong_particle_sector3_Z4_descent_gap','wrong_Gauss_orientation_gap']) > .1
    for name, expected_hash in request['pins'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected_hash,name
    result={'kind':'NEW_INDEPENDENT_BOUNDED_NATIVE_SOURCE_FRAME_GAUSS_CONTROLS_NOT_GROUND','source_inputs_unchanged':True,'controls':controls,'limits':['bounded native70/full8 modes and harmonic n<=4 tensor materializations only','immutable actual9672320 binary x read for charge/role actions, not materialized physical full-field Fock wave','all-n extraction and exactg=P0x identity proved conditionally in formal review, not numerically re-solved','no historical main/root/eigensolver/checker or full ground variance run']}
    with args.output.open('x') as output:
        json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
