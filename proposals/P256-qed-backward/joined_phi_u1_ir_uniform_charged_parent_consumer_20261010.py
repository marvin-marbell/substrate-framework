"""Consume scoped IR/uniform/charged epochs; exercise NEW fullE boost and Gram boundaries.

Finite matrix controls are not interacting-ground or infinite-volume measurements.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
from scipy.linalg import expm
import sympy as sp
from threadpoolctl import threadpool_limits

import joined_phi_u1_native_ir_mechanism_consumer as native

ROOT = Path(__file__).resolve().parent
sys.set_int_max_str_digits(0)
PINS = {
    'joined_phi_u1_uniform_electric_current_bridge_proof_20261010.json': 'b68accf76f67d368bf2e747ac633a899b020963b5742bb152dc8c11c6bef6a0e',
    'joined_phi_u1_uniform_electric_current_bridge_integration_addendum_20261010.json': 'c9393c1aa3cd339958e96ae61bb213d87e7b2d829ab885481e8a820eb2d68c71',
    'joined_phi_u1_uniform_electric_current_bridge_mathreview_ElectricGraphLimitReview_20261010.json': '125ba4bd50d220f3283eee21ce731aaf8bcb80efcf9bbcb1facba77bb685fb58',
    'joined_phi_u1_charged_exterior_resummation_mathreview_SharedCharacterActionReview_20261010.json': '78149c8012a24fa5356431e3c44d13141b8401dc8b62a0a03d5b1ed9cc43f213',
    'joined_phi_u1_uniform_gauss_limit_consumer.py': '37086f9ed5faa3b14dbb3fa9061393c527d61226b5288349bc6be04d07cbfafd',
    'joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual.json': '3661b8e5f6420da11313ae5d282acdaeba37fb45f9c555f9f6f6484a739f964d',
    'joined_phi_u1_uniform_gauss_limit_epoch4_20261010_natural_exit.json': '3a1dca1cac324bfde9b4033f19b8d201f23625f852732aa255538d92c893bb47',
    'joined_phi_u1_uniform_gauss_limit_mathreview_UniformLimitProofReview_epoch4_20261010.json': 'e66f1c082e13fb1c965d8f535936ac29a4bb06ac0f450627cd208969e47259aa',
    'joined_phi_u1_native_ir_mechanism_consumer.py': '98ecb55fc17663612e855723e0d642effeb0bafff8226d6990ea556b89f0a6d8',
    'joined_phi_u1_native_ir_mechanism_proof_corrected_20261010.json': '0f38e399c179b79f7df920b0d975d0dc377348bfe14dab7867005e7b98bcf06f',
    'joined_phi_u1_native_ir_mechanism_corrected2_20261010_actual.json': '35f3c586a52cf818d57f3643c35e1609a2b48e6f5a1ad3c2e7c6a8e6e3f37635',
    'joined_phi_u1_native_ir_mechanism_corrected2_20261010_natural_exit.json': '7d469a7058dd9a4f46516f8bf4b620defcaf957d69429b02d58e8e1416acf340',
    'joined_phi_u1_native_ir_mechanism_mathreview_NativeSchurProofReview_20261010.json': 'f3b06ba592158734ca992193a9821af295d9a0fa2724d8a894d64f10b93b30a7',
    'joined_phi_u1_charged_exterior_resummation_consumer.py': 'e5d00ea90d48c3c4a8f807f22ec62da878c525c8fd8cf32905face7e39f5c483',
    'joined_phi_u1_charged_exterior_resummation_final_20261010.json': '1e12c4f23214969496dce0a27689f93fe44c4aaa985e9a0b65a68d94a8931b35',
    'joined_phi_u1_charged_exterior_resummation_delivery_receipt_20261010.json': '5dc1e082f127e5a8c81ca7fb8488cbed957427a5a3df19573ec746ab7a17e3f2',
    'joined_phi_u1_charged_exterior_resummation_unbiased_observable.py': 'cb22ede41809ea875b72d34544c9bad151f79438cfcda0e8df5ede94da4e5aaa',
    'joined_phi_u1_charged_exterior_resummation_unbiased_observable_checked_20261010.json': 'ae547ed29b5db408d5f0f4c810e2c6a6ab01916f7be26e7b7f7745027a87f39b',
    'joined_phi_u1_charged_exterior_resummation_unbiased_observable_delivery_receipt_20261010.json': '46279f6484c8294b7460d047e93a7a45f8240766d4347d10d5edfc0c8bed22d5',
    'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz': 'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
    'joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json': '1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94',
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def matrix(rows: list[list[list[float]]]) -> np.ndarray:
    return np.asarray([[complex(*value) for value in row] for row in rows])


def uniform_prices(actual: dict[str, Any]) -> dict[str, Any]:
    lam, chi, fast = F(1 << 8192), F(1, 1 << 4096), F(1 << 10240)
    diagonal, offdiagonal = F(435, 4), F(105, 4)
    ordered = 4*diagonal + 12*offdiagonal
    assert ordered == 750 and str(ordered) == actual['exact_constants']['Jordan_ordered16_Wick']
    jordan = ordered/(16*F(1, 64))
    assert jordan == 3000 and jordan != 4*diagonal/(16*F(1, 64))
    alignment, colour = F(3*2*30, 40), F(7*3*4, 10)
    assert alignment+colour == F(129, 10)
    electric = 64*64**2
    assert electric == 262144 and electric != 64*64
    trial = 60*chi+jordan*lam+(alignment+colour)*fast+electric
    budget = trial+8192*lam
    assert budget == F(actual['exact_constants']['R'])
    assert 2*budget/lam < (1 << 2053) and budget/fast < 13
    current = F('15475017073321605/1099511627776')
    assert current < (1 << 14)
    # Squared CAR/Phi bound uses <sum_S s>, not a unit Gaussian variance.
    radial_second = 2*budget/lam
    assert current**4 * 16 * radial_second < (1 << 2114)
    assert current**2 * 16 * radial_second < (1 << 2085)
    return dict(ordered_Jordan_Wick=str(ordered), Jordan_price_lambda=str(jordan),
                alignment_fast=str(alignment), colour_fast=str(colour), electric_price=electric,
                Phi_fourth_bound='<2^2053', geometry_mean_bound='<13',
                edge_current_squared_bound='<2^1057', squared_cutoff_error='<2^2085/M',
                scope='Exact rational microscopic prices/form budgets; no sampled ground moment.')


def boost_controls(profiles: list[dict[str, Any]], geometry: dict[str, np.ndarray]) -> dict[str, Any]:
    rng = np.random.default_rng(904712)
    raw = rng.normal(size=(64, 4, 4, 4))+1j*rng.normal(size=(64, 4, 4, 4))
    phi = .035*(raw+raw.conj().swapaxes(-1, -2))/2
    phi -= np.trace(phi, axis1=-2, axis2=-1)[..., None, None]*np.eye(4)/4
    raw_link = rng.normal(size=(192, 4, 4))+1j*rng.normal(size=(192, 4, 4))
    generator = .07*(raw_link+raw_link.conj().swapaxes(-1, -2))/2
    generator -= np.trace(generator, axis1=-2, axis2=-1)[:, None, None]*np.eye(4)/4
    links = np.asarray([expm(1j*value) for value in generator])
    theta = rng.normal(scale=.3, size=192)
    sites = geometry['sites']
    seam_site = next(x for x, site in enumerate(sites) if tuple(map(int, site)) == (3, 1, 2))
    edge = 3*seam_site
    v = np.zeros(192); v[edge] = 1
    d, q = geometry['D'], geometry['Q']
    lp = np.linalg.pinv(d@d.T, hermitian=True)
    transverse = q@(q.T@v)
    density_angle = lp@(d@v)
    assert norm(transverse+d.T@density_angle-v) < 1e-12
    rows = []
    for role, profile in enumerate(profiles):
        h, j, k = native.native_jets(profile, sites, phi, links, theta, v)
        plus = (-k-1j*j)/2
        minus = (-k+1j*j)/2
        hermiticity = max(norm(h-h.conj().T), norm(j-j.conj().T), norm(k-k.conj().T))
        assert hermiticity < 1e-12 and norm(minus-plus.conj().T) < 1e-12
        shifted_rows = []
        for s in (1/8, 1/64, 1/512):
            shifted, _, _ = native.native_jets(profile, sites, phi, links, theta-s*v, v)
            delta = shifted-h
            expected = (np.exp(-1j*s)-1)*plus+(np.exp(1j*s)-1)*minus
            shift_error = norm(delta-expected)
            assert shift_error < 1e-11
            if profile['scale'] == 'heavy':
                assert norm(delta) == 0 and norm(j) == 0
                continue
            envelope = s*s*(norm(plus)+norm(minus))/2
            taylor_error = norm(delta+s*j)
            assert 0 < taylor_error <= envelope*(1+1e-8)
            wrong_sign_error = norm(delta-s*j)
            assert wrong_sign_error > 20*taylor_error
            pi_only, _, _ = native.native_jets(profile, sites, phi, links, theta-s*transverse, v)
            density_phase = np.repeat(np.exp(-1j*s*density_angle), 4)
            full_reduced = density_phase[:, None]*pi_only*density_phase.conj()[None, :]
            fullE_error = norm(full_reduced-shifted)
            omitted_density_error = norm(pi_only-shifted)
            assert fullE_error < 1e-11 and omitted_density_error > 1e-5
            no_two, _, _ = native.native_jets(profile, sites, phi, links, theta, v, omit_two=True)
            no_two_shift, _, _ = native.native_jets(profile, sites, phi, links, theta-s*v, v, omit_two=True)
            omitted_two_error = norm(delta-(no_two_shift-no_two))
            assert omitted_two_error > 1e-5
            shifted_rows.append(dict(s=s, exact_literal_shift_error=shift_error,
                                     graph_Taylor_remainder=taylor_error,
                                     analytic_matrix_remainder_envelope=envelope,
                                     wrong_current_sign_error=wrong_sign_error,
                                     fullE_reduced_density_phase_error=fullE_error,
                                     wrong_Pi_only_error=omitted_density_error,
                                     omitted_native_two_link_error=omitted_two_error))
        rows.append(dict(role=role, scale=profile['scale'], hermiticity=hermiticity,
                         neutral_phase_exact_zero=profile['scale'] == 'heavy', shifts=shifted_rows))
    return dict(edge=edge, crosses_AP_seam=True, seed=904712, rows=rows,
                scope='NEW allnative onebody fullPhi/orderedSU4 structural probe and physical Gauss fullE conjugation; not interacting ground or interval.')


def source_and_ground_controls() -> dict[str, Any]:
    r = sp.Rational
    ell = sp.Matrix([[2,r(1,3),r(1,4),0], [r(1,3),4,0,r(1,5)],
                     [r(1,4),0,6,r(1,7)], [0,r(1,5),r(1,7),9]])
    assert all(ell.extract(list(range(n)),list(range(n))).det() > 0 for n in range(1, 5))
    z = sp.Symbol('z')
    a = ell.extract([0,1],[0,1])
    b = ell.extract([0,1],[2,3])
    bath = ell.extract([2,3],[2,3])
    c = sp.Matrix([[1,1], [0,2]])
    source = c.col_join(sp.zeros(2))
    gram = source.T*source
    sigma = b*(z*sp.eye(2)-bath).inv()*b.T
    denominator = z*sp.eye(2)-a-sigma
    point = -2
    actual = source.T*(point*sp.eye(4)-ell).inv()*source
    schur = c.T*denominator.subs(z,point).inv()*c
    assert actual == schur
    wrong_unit_Gram = denominator.subs(z,point).inv()
    assert actual != wrong_unit_Gram and gram.det() > 0
    expected_derivative = -b*(point*sp.eye(2)-bath).inv()**2*b.T
    derivative = sigma.diff(z).subs(z,point)
    assert derivative == expected_derivative
    metric = sp.eye(2)-derivative
    assert metric[0,0] > 0 and metric.det() > 0
    force = sp.zeros(2).col_join(b.T)
    reducible = force.T*(point*sp.eye(4)-ell).inv()*force
    assert reducible != sigma.subs(z,point)
    electric = sp.Matrix([1,2,0,0])
    physical_current = -sp.I*ell*electric
    contact = 2*(electric.T*ell*electric)[0]
    full_static_JJ = (physical_current.conjugate().T*ell.inv()*physical_current)[0]
    assert contact > 0 and contact-2*full_static_JJ == 0
    frequency = 2
    response_kernel = 2*ell*(ell**2+frequency**2*sp.eye(4)).inv()
    response = contact-(physical_current.conjugate().T*response_kernel*physical_current)[0]
    electric_response = (frequency**2*electric.T*response_kernel*electric)[0]
    assert response == electric_response and response > 0
    # Degenerate ground: mean subtraction is NOT the full zero-energy projection.
    h = sp.diag(0,0,3,5)
    omega = sp.Matrix([1,0,0,0])
    o = sp.Matrix([[0,1,1,0],[1,0,0,0],[1,0,0,0],[0,0,0,0]])
    mean = (omega.T*o*omega)[0]
    connected = (o-mean*sp.eye(4))*omega
    excitation = sp.diag(0,0,1,1)*connected
    assert (connected.T*connected)[0] == 2 and (excitation.T*excitation)[0] == 1
    excited_omega = sp.Matrix([0,0,1,0])
    l_excited = h-3*sp.eye(4)
    bad_source = o*excited_omega
    assert (bad_source.T*l_excited*bad_source)[0] == -3
    return dict(scope='EXACT_RATIONAL_METHOD_AND_GROUND_BLOCK_CONTROLS_NOT_N4_SPECTRAL_EVALUATION',
                source_Gram=str(gram), full_source_Schur_exact=True,
                wrong_unit_Gram_error=float((actual-wrong_unit_Gram).norm()),
                reducible_selfenergy_error=float((reducible-sigma.subs(z,point)).norm()),
                resolvent_derivative_negative_exact=True, residue_denominator_positive_exact=True,
                mean_connected_mass=2, whole_ground_projected_mass=1,
                stationary_excited_source_first_energy=-3,
                external_phase_contact=str(contact), full_static_response_exact_zero=True,
                nonzero_frequency_response=str(response), electric_current_response_identity_exact=True)


def charged_controls(actual: dict[str, Any], observable: dict[str, Any], d: np.ndarray) -> dict[str, Any]:
    matrices = {name: matrix(rows) for name, rows in actual['complete_shared_character_branch_matrices'].items()}
    coefficient = np.ones(2)
    overlap = float(np.vdot(coefficient, matrices['S']@coefficient).real)
    energy = float(np.vdot(coefficient, matrices['H']@coefficient).real/overlap)
    action2 = float(np.vdot(coefficient, matrices['H2']@coefficient).real/overlap)
    variance = action2-energy**2
    tail = sum(float(np.vdot(coefficient, matrices[name]@coefficient).real/overlap) for name in ('occupation_tail','Fourier_tail'))
    assert abs(variance-actual['chosen']['variance']) < 1e-5 and tail > 700000 and variance > 800000
    assert not actual['chosen']['stationary']
    unbiased = actual['unbiased_material_fork_control']
    assert unbiased['energy_shift'] < energy and unbiased['variance'] < variance and not unbiased['stationary']
    values: dict[tuple[tuple[int,int],...], complex] = {}
    for row in observable['all_Laurent_coefficients']:
        key = tuple((int(e),int(n)) for e,n in row['integer192_character'])
        assert key not in values
        values[key] = complex(*row['amplitude'])
        flow = np.zeros(192,dtype=int)
        for edge,count in key: flow[edge] = count
        assert np.array_equal(d.astype(int)@flow,np.zeros(64,dtype=int))
    covariance = F(0)
    convolution_real, convolution_imag = F(0),F(0)
    for key,value in values.items():
        if not key: continue
        opposite = tuple((edge,-count) for edge,count in key)
        reverse = values[opposite]
        assert reverse == value.conjugate()
        vr,vi,rr,ri = F(value.real),F(value.imag),F(reverse.real),F(reverse.imag)
        covariance += vr*vr+vi*vi
        convolution_real += vr*rr-vi*ri
        convolution_imag += vr*ri+vi*rr
    assert covariance == convolution_real > 0 and convolution_imag == 0
    assert abs(float(covariance)-observable['observable']['distinct_aux_shared_field_connected_covariance']) < 1e-16
    assert unbiased['distinct_aux_density_covariance'] == 0
    return dict(complete_biased_variance=variance, paid_biased_exterior=tail,
                retained_unbiased_energy_shift=unbiased['energy_shift'], retained_unbiased_variance=unbiased['variance'],
                density_covariance_zero_is_not_independence=True,
                all63_connected_matter_energy_covariance_exact_serialized=str(covariance),
                all63_connected_matter_energy_covariance=float(covariance),
                nonconstant_Laurent_modes=len(values)-1, Gauss_divergence_exact=True,
                stationarity_rejected=True,
                scope='Immutable fullH Gram contraction and exact serialized-Laurent covariance, not independent rederivation of allH images or outward root/ground certificate.')


def run(output: Path) -> dict[str, Any]:
    started = time.monotonic()
    own = Path(__file__).read_bytes()
    inputs = {name:(ROOT/name).read_bytes() for name in PINS}
    assert all(sha(raw) == PINS[name] for name,raw in inputs.items())
    uniform = json.loads(inputs['joined_phi_u1_uniform_gauss_limit_epoch4_20261010_actual.json'])
    uniform_cap = json.loads(inputs['joined_phi_u1_uniform_gauss_limit_epoch4_20261010_natural_exit.json'])
    assert uniform_cap['scientific']['exit_code'] == 0 and uniform_cap['scientific']['natural_exit'] and not uniform_cap['scientific']['stderr']
    assert uniform_cap['receipt']['sha256'] == PINS[uniform_cap['receipt']['path']]
    native_cap = json.loads(inputs['joined_phi_u1_native_ir_mechanism_corrected2_20261010_natural_exit.json'])
    assert native_cap['natural_exit'] == 0 and native_cap['source_inputs_unchanged'] and native_cap['stderr_empty']
    assert native_cap['actual_receipt_sha256'] == PINS['joined_phi_u1_native_ir_mechanism_corrected2_20261010_actual.json']
    profiles = json.loads(inputs['joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json'])['primitive_profiles']
    with np.load(io.BytesIO(inputs['joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz'])) as archive:
        geometry = {key:archive[key] for key in ('D','Q','sites')}
    proof = json.loads(inputs['joined_phi_u1_uniform_electric_current_bridge_proof_20261010.json'])
    addendum = json.loads(inputs['joined_phi_u1_uniform_electric_current_bridge_integration_addendum_20261010.json'])
    prices = uniform_prices(uniform)
    boosts = boost_controls(profiles,geometry)
    method = source_and_ground_controls()
    charged = charged_controls(json.loads(inputs['joined_phi_u1_charged_exterior_resummation_final_20261010.json']),
                               json.loads(inputs['joined_phi_u1_charged_exterior_resummation_unbiased_observable_checked_20261010.json']),geometry['D'])
    assert all((ROOT/name).read_bytes() == raw for name,raw in inputs.items()) and Path(__file__).read_bytes() == own
    result = dict(kind='NEW_PARENT_FULL_E_BOOST_AND_SCOPED_UNIFORM_IR_CHARGED_INTEGRATION_NOT_POLE',
                  governor=proof['governor'], input_sha256=PINS, own_source_sha256=sha(own), inputs_unchanged=True,
                  uniform_exact_prices=prices, actual_native_fullE_boost=boosts,
                  source_Gram_and_ground_block_controls=method, immutable_charged_contraction=charged,
                  proof_status='Analytic b68accf7 bridge independently reviewed VALID at selected-source domain scope; final parent science integration pending. Finite matrices do not prove infinite-volume transitions.',
                  scope_corrections=proof['scope_corrections'],
                  integration_addendum=addendum,
                  no_claims=proof['not_proved'], elapsed_seconds=time.monotonic()-started)
    output.write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('source/'+Path(__file__).name,own)
        for name,raw in inputs.items(): archive.writestr('inputs/'+name,raw)
        archive.writestr('receipt/'+output.name,output.read_bytes())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        receipt = run(args.output)
    print(json.dumps({key:receipt[key] for key in ('kind','uniform_exact_prices','source_Gram_and_ground_block_controls','immutable_charged_contraction','elapsed_seconds')},indent=2))
