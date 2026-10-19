"""Counterexample and repaired global-normal h-jet envelope, not source admission.

The archived source's per-q derivative number is not a full-normal bound.
This CLI tests the actual native operator and derives a support-wide envelope.
Frozen scientific suppliers and their historical receipts remain unchanged.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import induced_phi_nonuniform_fermionic_source_unitary as source

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json'
INPUT_SHA = 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6'
R = F(1, 1 << 20)
DERIVATION = {
    'exact_counterexample': (
        'Let v be any unit eigenvector of Gamma0. X=(vv*-I4/4)/sqrt(3/2) is traceless '
        'Hermitian, commutes with Phi0, and has Cartesian norm sqrt(2TrX^2)=1. '
        'Set only deltaPhi0(x)=l_x X, l=(delta_x0-1/64)/sqrt(63/64). This is a '
        'unit global fast normal direction, has zero site mean and zero orbit pairing. '
        'A one-particle vector supported at site0 has |<v,dh v>|=|onsite|sqrt(63/64)sqrt(3/8); '
        'all one/two-link hops leave site0 and contribute zero to this Rayleigh value. '
        'The rational lower sqrt(63/64)>=63/64 and sqrt(3/8)>=3/5 already contradicts '
        'the stored full-normal D1 for all three profiles. No root rounding is needed for the witness.'),
    'global_field_bound': (
        'Cartesian norm uses Tr(Ta Tb)=delta_ab/2, hence ||sum xa Ta||op<=||x||/sqrt2. '
        'Native block-row Schur bound counts forward AND reverse hops. '
        'For arbitrary global Phi tangent, Bphi=(|onsite|+2sum_path |c|1)/sqrt2=8*vp. '
        'The 8 is sqrt(V), undoing the single-q Fourier 1/8 normalization. '
        'For arbitrary link C tangent the generator norm is <=||C||/sqrt2, not the '
        'single signed-Clifford generator norm 1/sqrt8. At the mean Bc=16*vc. '
        'On the actual support, hopnorm<=hop_mean+|c|1*F1*R/sqrt2, so '
        'Bc=16*vc+sum L|c|1*F1*R.'),
    'adapted_composition': (
        'Admitted actual polar-Q envelopes imply F1<=1+9/8+512R and F2<=1025. '
        'Thus Dh<=Bphi F1+Bc. Ordered exponential insertions give '
        'h_Phi,C<=sum L|c|1 and h_C,C<=sum L^2(hop_mean+|c|1 F1R/sqrt2). '
        'The complete second derivative is bounded by Bphi F2+2(sum L|c|1)F1+Bcc. '
        'Both endpoints, all ordered one/two-link derivatives and noncommuting matrices retained. '
        'These bounds are also uniform in added scalar Peierls phases of modulus one, '
        'but do not define a theta-following occupied frame.'),
    'Kato_and_gradient_boundary': (
        'At supported distance>=gap/2-2D1R, use b1=8D1/distance, '
        'b2=32D1^2/distance^2+8D2/distance. These are directional one-particle bounds. '
        'For a fixed-particle Fock sector, a safe full ambient gradient envelope includes '
        'ceil(sqrt5760)=76 times sum Nspecies*b1. Sum Nspecies*b2 is only directional '
        'second-frame data; Hessian/kinetic traces must also retain dimension and first-square terms.'),
    'not_certified': (
        'Corrected h/Kato envelopes alone do NOT reprice or certify the whole source. '
        'Old Rayleigh/time/probability prices are historical conditional arithmetic with a '
        'refuted literal all-normal derivative premise. Global chart/tube/root, other supplier '
        'inequalities, source kinetic/Haar propagation, U1Pi/Coulomb and physical poles remain separate.')
}


def run() -> dict[str, Any]:
    started = time.monotonic()
    raw = INPUT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise ArithmeticError('Historical source-price input changed')
    cap = json.loads(raw)
    proof = cap['actual_full5760_local960_corrected_quantum_CAR_source_proof']
    centres = [F(x) for x in cap['input_descriptor']['exact_NEW_allq_BO_mean']['centre_exact_rational']]
    field.full.iv.dps = 100
    mp.mp.dps = 100
    data = field.load_mean(g_override=[centres[0], centres[1], centres[1], centres[1]],
                           mean_descriptor={'law': 'NEW derivative control at saved mean display; NOT adopted root'})
    parameters = cap['input_descriptor']['exact_NEW_allq_BO_mean']['parameters']
    root_intervals = [field.full.enclosure(F(parameters[k]['lower']), F(parameters[k]['upper'])) for k in ('g0', 'gs')]
    data['g'] = [root_intervals[0], *[root_intervals[1]]*3]
    ex, iv = field.full.exact, field.full.iv
    native, geom = data['native'], data['geom']
    _, u = np.linalg.eigh(native['frame'][0])
    v = u[:, -1]
    X = (np.outer(v, v.conj())-np.eye(4)/4)/np.sqrt(1.5)
    coeff = 2*np.einsum('aij,ji->a', geom['deltas'][:, 0], X).real
    normal = np.einsum('a,amij->mij', coeff, geom['deltas'])
    local = -np.ones(64)/64
    local[0] += 1
    local /= np.linalg.norm(local)
    fd = local[:, None, None, None]*normal[None]
    fields = np.broadcast_to(native['frame'], (64, 4, 4, 4)).copy()
    links = np.broadcast_to(np.eye(4), (192, 4, 4)).copy().astype(complex)
    ld = np.zeros_like(links)
    state = np.zeros(256, complex)
    state[:4] = v
    reconstruction = float(np.max(abs(normal[0]-X)))
    other = float(np.max(abs(normal[1:])))
    tangent_norm = float(np.sqrt(2*np.sum(abs(fd)**2)))
    if max(reconstruction, other, abs(tangent_norm-1)) > 1e-12:
        raise ArithmeticError('The proposed normal-direction witness is not realized')
    rows = []
    first = ex(0)
    second = ex(0)
    F1 = ex(F(1)+F(9, 8)+512*R)
    F2 = ex(1025)
    for profile, old in zip(data['profiles'], proof['actual_full65_occupied_projector_Kato_prices'], strict=True):
        if profile['role'] != old['role']:
            raise ArithmeticError('Profile ordering changed')
        _, dh = source.h_jet(profile, native, fields, fd, links, ld)
        ray = float(abs(np.vdot(state, dh@state)))
        observed = float(np.max(abs(np.linalg.eigvalsh(dh))))
        claimed = F(old['all_normal_first_h_derivative_upper']['upper'])
        onsite = abs(F(profile['onsite']))
        rational_lower = F(math.floor(onsite))*F(63, 64)*F(3, 5)
        if rational_lower <= claimed or ray <= float(claimed):
            raise ArithmeticError('The counterexample does not reject the stored premise')
        bounds = field.exact_operator_bounds(profile, data['g'])
        vp = ex(F(bounds['Phi_vertex_operator_upper']['upper']))
        vc = ex(F(bounds['colour_vertex_operator_upper']['upper']))
        pc = sum(p['length']*(abs(p['c0'])+abs(p['ci'])) for p in profile['paths'])
        cc = ex(0)
        for p in profile['paths']:
            c1 = abs(p['c0'])+abs(p['ci'])
            hop = abs(ex(p['c0']))*iv.sqrt(data['g'][0])+abs(ex(p['ci']))*iv.sqrt(data['g'][p['axis']+1])
            cc += ex(p['length']**2)*(hop+ex(c1)*F1*ex(R)/iv.sqrt(ex(2)))
        Bphi = 8*vp
        Bc = 16*vc+ex(pc)*F1*ex(R)
        D1 = Bphi*F1+Bc
        D2 = Bphi*F2+2*ex(pc)*F1+cc
        distance = ex(F(bounds['minimum_gap_unscaled']['lower'])/2)-2*D1*ex(R)
        if field.full.endpoint(distance, 0) <= 0:
            raise ArithmeticError('Repaired support-wide gap price fails')
        b1 = 8*D1/distance
        b2 = 32*D1**2/distance**2+8*D2/distance
        count = 128*profile['load']+(1 if profile['scale']!='heavy' and profile['load']==1 else 0)
        first += count*b1
        second += count*b2
        D1_upper = F(field.full.bounds(D1)['upper'])
        if observed >= float(D1_upper):
            raise ArithmeticError('Repaired envelope misses the actual native witness')
        rows.append({'role': profile['role'], 'load': profile['load'],
                     'old_stored_claimed_full_normal_D1': float(claimed),
                     'exact_rational_Rayleigh_lower': str(rational_lower),
                     'observed_Rayleigh_lower': ray, 'observed_native_derivative_opnorm': observed,
                     'observed_divided_old_claim': observed/float(claimed),
                     'repaired_global_D1': field.full.bounds(D1),
                     'repaired_global_D2': field.full.bounds(D2),
                     'supported_distance_to_zero_lower': field.full.bounds(distance),
                     'directional_Kato_first': field.full.bounds(b1),
                     'directional_Kato_second': field.full.bounds(b2),
                     'fixed_bulk_particle_count': count})
    captured = {**field.CAPTURED, **source.SOURCE_BYTES, Path(__file__).name: Path(__file__).read_bytes()}
    epochs = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'unchanged': data==(ROOT/name).read_bytes()}
              for name, data in captured.items()}
    if not all(record['unchanged'] for record in epochs.values()) or INPUT.read_bytes()!=raw:
        raise ArithmeticError('Control source/input changed during execution')
    return {'governing_plan': 'issue-222-induced-photon-delta13-continuation-20261008',
            'governing_revision': 'a0de34c80c9dae7a8c5ddddcbeb6c0dc5b3fe69213a72993e47e5f7379fdd5dc',
            'template_revision': '89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
            'input': {'path': str(INPUT), 'sha256': INPUT_SHA}, 'derivation': DERIVATION,
            'numerical_display_g': data['descriptor']['numerical_display_g'],
            'witness': {'normal_reconstruction_error': reconstruction, 'other_components': other,
                        'global_tangent_norm': tangent_norm, 'coefficient_norm': float(np.linalg.norm(coeff)),
                        'orbit_pairing': float(np.max(abs(sum(native['frame'][m]@normal[m]-normal[m]@native['frame'][m] for m in range(4)))))},
            'profiles': rows, 'safe_full_ambient_Fock_gradient_price': field.full.bounds(76*first),
            'directional_Fock_second_data_NOT_Hessian_norm': field.full.bounds(second),
            'verdict': 'Stored literal all-normal D1 premise REFUTED. Repaired support-wide h jet envelope derived conditionally; no revised whole-source admission yet.',
            'source_at_execution': epochs, 'own_source_utf8': Path(__file__).read_text(),
            'runtime': {'executable': sys.executable, 'python': sys.version},
            'elapsed_seconds': time.monotonic()-started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'verdict': result['verdict'], 'witness': result['witness'],
                      'profiles': [{k: v for k, v in row.items() if k not in ('repaired_global_D1', 'repaired_global_D2', 'supported_distance_to_zero_lower', 'directional_Kato_first', 'directional_Kato_second')} for row in result['profiles']],
                      'elapsed_seconds': result['elapsed_seconds']}, indent=2))
