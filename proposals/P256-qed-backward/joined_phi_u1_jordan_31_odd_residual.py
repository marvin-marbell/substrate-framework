"""NEW Jordan 3+1: exact odd-block residual and ALL-angle pressure certificate.

The theorem covers arbitrary Hermitian3 blocks with compensating scalar fields.
Native finite controls and a rational trial do not characterize the quantum ground.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import zipfile

import numpy as np
import sympy as sp
from threadpoolctl import threadpool_limits

import induced_phi_nonuniform_field_car as field
import joined_phi_u1_matrix_jordan_flat_bounds as flat

ROOT = Path(__file__).resolve().parent
I = flat.I
GOVERNOR = '55352a0aabbcc1da539641239f48f2807ec1e2a609b5d533ca0f186d8df27b6e'
PROOF = {
    'scope': 'NEW independently labelled matrix-Jordan law; currentN4 flat STATIC neutral128 only, ALL continuous determinant-one SU4 toron characters. No SOURCE/CURRENT427 adoption.',
    'fields': 'A_mu arbitrary Hermitian3; s_mu=-Tr A_mu is the compensating scalar. Phi_mu=A_mu directsum s_mu, mu=0..3. No individual tracefree block restriction. T_mu,nu=Tr A_mu A_nu; G=(T+ss^T)/4. All four field directions remain.',
    'odd_CH': 'For traceless Hermitian3 B with eigenvalues b1+b2+b3=0, sum b_i^4=(sum b_i^2)^2/2 by expanding b3=-b1-b2. Polarizing this polynomial gives 4Tr(X^2Y^2)+2Tr(XYXY)=Tr(X^2)Tr(Y^2)+2(Tr XY)^2.',
    'odd_collection': 'For arbitrary traceless Hermitian3 X_a, let R_ab=Tr X_a X_b and S=sum_a X_a^2. Summed polarization gives sum_ab Tr({X_a,X_b}^2)=(Tr S)^2+2||R||F^2-2Tr S^2. HS triangle and odd_CH give ||S||HS<=sum_a ||X_a^2||HS=Tr S/sqrt(2). Consequently the anticommutator sum is >=2||R||F^2. Equivalently sum_ab ||{X_a,X_b}-(2/3)R_ab I3||HS^2 >=(2/3)||R||F^2. This is a quantitative trace/parity residual, not a representation-name exclusion.',
    'traceful_projection': 'If s!=0 let P project onto s-perp in real field-index R4, rank3; if s=0 choose any rank3 P. Rotate field indices orthogonally so P selects three coordinates. Their A combinations are traceless because Tr A=-s. The sum of squared traceless anticommutators over those three directions is a subset of the full orthogonally invariant nonnegative sum. Thus full residual >=(2/3)||PTP||F^2. Scalars and the fourth arbitrary Hermitian3 field were NOT discarded.',
    'quartic': 'The exact flat quartic is qJ=256 sum_mu,nu Tr({Phi_mu,Phi_nu}^2). Trace decomposition on the3 block plus the scalar gives qJ=(1024/3)||T||F^2+1024||ss^T||F^2+256||F3||F^2, F3_mu,nu={A_mu,A_nu}-(2/3)T_mu,nu I3. Hence qJ>=(1024/3)[||T||F^2+(1/2)||PTP||F^2]+1024|s|^4.',
    'first_route': 'The scalar-block rank-one residual alone gives qJ>=4096[||G||F^2+(1/3)||G-ss^T||F^2]. Optimizing its rank-one residual yields qJ>=4096[g1^2+(4/3)sum_i>1 gi^2]. The resulting global-moment bound is weaker than the incoming floor. It is retained as a failed route; the odd_CH collection inequality is a materially different traceful3-block route.',
    'alternative_global': 'Combining the first-route exact decomposition with odd_collection adds (8192/3)||PGP||F^2. Rank-one approximation/interlacing give ||G-ss^T||F^2>=sum_i>1 gi^2 and ||PGP||F^2>=sum_i>1 gi^2, so qJ>=4096[g1^2+2sum_i>1 gi^2]. Weighted Cauchy yields the independent global-moment lower bound reported below.',
    'dual_pressure': 'For Q_P(T)=||T||F^2+(1/2)||PTP||F^2, the positive operator T->PTP is an orthogonal projection on the real symmetric matrix space. Inverting I+(1/2)projection gives C-dual norm squared D(P)=||C||F^2-(1/3)||PCP||F^2. Weighted Cauchy proves (Tr CT)^2<=D(P) Q_P(T). Eigenvalue interlacing for compression to any3-plane gives ||PCP||F^2>=sum of the smallest3 eigenvalues(C)^2; hence D(P)<=D31=cmax^2+(2/3)sum_remaining c_i^2. This is uniform in s orientation and rank drops.',
    'all_angles': 'Native Fourier coefficients x_b(k) satisfy mean64 xx^T=C independently in EACH block for every continuous theta: all four samples per axis have zero sin/cos and sincos averages and sin^2=cos^2=1/2. Determinant is theta1=-3theta3 mod2pi; the3+1 character torus is connected. No angle minimizer, grid or centrality assumption is used.',
    'occupation': 'Tr H=64 onsite(Tr A0+s0)=0 globally only. For256 native bands exact physical neutral128 is -sum GLOBAL top128. KyFan gives S128<=Tr|H|/2. Per3 block, trace-norm Cauchy and momentum Jensen give half sea<=32sqrt(3 Tr CT); scalar half sea<=32sqrt(s^T C s). This lower bound does not assert blockwise half occupation.',
    'minimum': 'Put m3=Tr CT>=0, m1=s^T C s>=0. qJ-S128>=(1024/(3D31))m3^2-32sqrt(3m3)+(1024/cmax^2)m1^2-32sqrt(m1). Exact radial minimization gives L31=(3/4)[-3(256D31)^(1/3)]+(1/4)[-3(256cmax^2)^(1/3)]. Dropping further trace/character coupling only relaxes a lower bound. All roots are enclosed by outward integer dyadic arithmetic.',
    'limits': 'No central dominance, native counterexample, fullPhi/off-flat quantum ground or same-ground photon claim. FullPhi/SU4 kinetic/Gauss, charged active1+aux63/fullHC/Pi, spatial photon pole/dispersion, cone/originalSOURCE/Compton remain OPEN.'
}


def residual_controls() -> dict[str, Any]:
    z = sp.zeros(3)
    xs = [sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
          sp.Matrix([[0, -sp.I, 0], [sp.I, 0, 0], [0, 0, 0]]),
          sp.diag(1, -1, 0)]
    r = sp.Matrix(3, 3, lambda a, b: sp.trace(xs[a]*xs[b]))
    anti = sum(sp.trace((a*b+b*a)**2) for a in xs for b in xs)
    gram2 = sp.trace(r*r)
    assert anti == 2*gram2 == 24
    # Independent adverse control: a larger purported parity coefficient fails
    # on exact embedded Pauli matrices, including rank3 (not only rank1).
    wrong = sp.Rational(8, 3)*gram2
    assert wrong > anti
    arbitrary = [sp.Matrix([[2, 1+sp.I, -2], [1-sp.I, -1, sp.I], [-2, -sp.I, 3]]),
                 sp.Matrix([[1, 2, sp.I], [2, -2, 1-sp.I], [-sp.I, 1+sp.I, 0]])]
    examples = []
    for matrices in (xs, [a-sp.trace(a)*sp.eye(3)/3 for a in arbitrary]):
        t = sp.Matrix(len(matrices), len(matrices), lambda a, b: sp.trace(matrices[a]*matrices[b]))
        s = sum((a*a for a in matrices), z)
        q = sum(sp.trace((a*b+b*a)**2).expand() for a in matrices for b in matrices)
        polarization = sp.expand(sp.trace(s)**2+2*sp.trace(t*t)-2*sp.trace(s*s))
        assert sp.simplify(q-polarization) == 0
        assert sp.expand(q-2*sp.trace(t*t)) >= 0
        examples.append({'anticommutator_exact': str(q), 'polarization_exact': str(polarization), 'residual_above_odd_bound_exact': str(sp.expand(q-2*sp.trace(t*t)))})
    rng = np.random.default_rng(310222)
    traceful = []
    for _ in range(12):
        a = rng.integers(-4, 5, (4, 3, 3))+1j*rng.integers(-4, 5, (4, 3, 3))
        a = (a+a.conj().transpose(0, 2, 1))/16
        s = -np.trace(a, axis1=1, axis2=2).real
        t = np.einsum('aij,bji->ab', a, a).real
        p = np.eye(4)-np.outer(s, s)/np.dot(s, s) if np.dot(s, s) else np.diag([1., 1., 1., 0.])
        phi = np.zeros((4, 4, 4), complex)
        phi[:, :3, :3] = a
        phi[:, 3, 3] = s
        q, _, _ = flat.quartics(phi)
        rhs = 1024/3*(np.sum(t*t)+.5*np.sum((p@t@p)**2))+1024*np.dot(s, s)**2
        assert q+1e-8 >= rhs
        traceful.append({'quartic': q, 'traceful_projected_floor': float(rhs), 'slack': float(q-rhs), 'scalar': s.tolist()})
    return {'exact_positive_saturation': {'Q': str(anti), '2Gram2': str(2*gram2)}, 'independent_wrong_stronger_coefficient': {'wrong_floor': str(wrong), 'actual': str(anti), 'strict_rejection': str(wrong-anti)}, 'exact_noncommuting_odd_identities': examples, 'finite_traceful_projection_controls_not_proof': traceful}


def allangle_bound(profile: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    c = [profile['onsite']**2+2*sum(p['c0']**2 for p in profile['paths'])]
    c += [2*next(p['ci'] for p in profile['paths'] if p['axis'] == j)**2 for j in range(3)]
    ordered = sorted(c, reverse=True)
    head, tail = ordered[0]**2, sum(v*v for v in ordered[1:])
    d31 = head+F(2, 3)*tail
    scalar = -3*I.of(256*head).cube_root()
    odd = -3*I.of(256*d31).cube_root()
    lower = F(3, 4)*odd+F(1, 4)*scalar
    first = -3*I.of(256*(head+F(3, 4)*tail)).cube_root()
    alternative = -3*I.of(256*(head+F(1, 2)*tail)).cube_root()
    oldrow = next(p for p in incoming['exact_ALLangle_sector_lower_bounds']['partitions'] if p['partition'] == [3, 1])
    previous = I.of(F(oldrow['ALL_continuous_angles_lower']['lower']))
    upper = I.of(F(incoming['exact_central_convex_residual']['upper']))
    assert lower.lo > previous.hi
    return {'C_exact': [str(v) for v in c], 'D31_exact': str(d31), 'incoming_ALLangle_floor': previous.public(), 'failed_scalar_rankone_route_floor': first.public(), 'first_route_improves_incoming': first.lo > previous.hi, 'different_odd_block_pressure_floor': lower.public(), 'alternative_global_odd_residual_floor': alternative.public(), 'improvement_above_incoming_floor': (lower-previous).public(), 'central_trial_upper': upper.public(), 'remaining_gap_to_central_trial': (upper-lower).public(), 'central_dominance_certified': lower.lo >= upper.hi, 'all_continuous_angles': True, 'all_arbitrary_traceful_blocks': True, 'global_top128_occupation': True}


def run(output: Path, contract: Path) -> dict[str, Any]:
    start = time.monotonic()
    snapshots = dict(field.CAPTURED)
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).parent == ROOT:
            snapshots.setdefault(Path(path).name, Path(path).read_bytes())
    for name, pin in flat.sea.INPUTS.items():
        snapshots[name] = (ROOT/name).read_bytes()
        assert hashlib.sha256(snapshots[name]).hexdigest() == pin
    receipt_name = 'joined_phi_u1_matrix_jordan_flat_actual_20261010_receipt.json'
    snapshots[receipt_name] = (ROOT/receipt_name).read_bytes()
    snapshots['governing_contract.json'] = contract.read_bytes()
    snapshots['preedit_lsp.json'] = (ROOT/'joined_phi_u1_jordan_31_preedit_lsp.json').read_bytes()
    incoming = json.loads(snapshots[receipt_name])
    archive = output.with_name(output.stem+'_input_bytes.zip')
    if output.exists() or archive.exists():
        raise FileExistsError('Immutable epoch path already exists')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as zipped:
        for name, raw in sorted(snapshots.items()):
            zipped.writestr(name, raw)
    # This is the existing root-free unit-frame input loader, NOT a mean solver.
    data = field.load_mean(g_override=[1, 1, 1, 1], mean_descriptor={'scope': 'NEW odd3 exact inequality/native controls; no root solve'})
    profile, native = data['profiles'][0], data['native']
    bound = allangle_bound(profile, incoming)
    algebra = residual_controls()
    oldtrial = next(p for p in incoming['exact_rational_trial_certificates'] if p['partition'] == [3, 1])
    phi = np.asarray(oldtrial['field_real'])+1j*np.asarray(oldtrial['field_imag'])
    trial = flat.certify_trial(profile, (3, 1), phi, [F(v) for v in oldtrial['angles_pi'][0]], (0, 0, 0))
    native_controls = [flat.native_control(p, native, phi, (3, 1), np.asarray(trial['angles_native_radians'])) for p in data['profiles']]
    adverse = np.zeros((4, 4, 4), complex)
    adverse[0] = np.diag([-1, -1, -1, 3])/5
    theta = np.pi*np.asarray([[F(1, 7), F(2, 9), F(3, 11)], [-F(3, 7), -F(2, 3), -F(9, 11)]], float)
    occupation = flat.native_control(profile, native, adverse, (3, 1), theta)
    occ = occupation['native_oldlaw_control']
    assert abs(occ['global_largest128']-occ['wrong_fixed_block_half']) > 1
    assert abs(occ['global_largest128']-occ['wrong_top2_per_momentum']) > 1
    assert occ['wrong_AP_omission_matrix_HS'] > 1
    before = {name: hashlib.sha256(raw).hexdigest() for name, raw in snapshots.items()}
    unchanged = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == pin for name, pin in before.items() if (ROOT/name).exists()}
    assert all(unchanged.values())
    result = {'kind': 'NEW_JORDAN_ODD3_ALLANGLE_QUANTITATIVE_RESIDUAL_NOT_GROUND_ADMISSION', 'plan_id': 'issue-222-induced-photon-delta13-continuation-20261008', 'governing_revision': GOVERNOR, 'pinned_template_revision': '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe', 'parent_work_item': 'induced-photon', 'slice': 'slice31', 'proof': PROOF, 'ALLangle_certificate': bound, 'controls': algebra, 'selected_exact_rational_sector_trial': trial, 'all_three_primitive_role_native_controls': native_controls, 'independent_wrong_occupation_and_AP_control': occupation, 'sources_inputs_sha256': before, 'sources_unchanged': unchanged, 'immutable_archive': {'path': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, 'runtime': {'python': sys.executable, 'version': platform.python_version(), 'numpy': np.__version__}, 'elapsed_seconds': time.monotonic()-start, 'OPEN': ['remaining static31 gap to central trial', 'off-flat/fullPhi quantum ground', 'same-ground physical spatial photon pole/dispersion', 'cone/originalSOURCE/Compton/campaign']}
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.output, args.contract)
    print(json.dumps({'receipt': str(args.output), 'ALLangle_certificate': result['ALLangle_certificate'], 'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
