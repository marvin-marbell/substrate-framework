"""NEW exact independent-centre rejection and joint-centre charged carrier control."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import sympy as s


def hs2(a):
    return s.simplify(s.trace(a.H * a))


def centre_controls(profiles):
    pauli = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
    gamma = [s.diag(1, 1, -1, -1)] + [s.kronecker_product(pauli[0], p) for p in pauli]
    centres = [1, s.I, -1, -s.I]
    rotation = s.eye(4)
    rotation[0, 0] = rotation[2, 2] = s.Rational(3, 5)
    rotation[0, 2] = s.Rational(4, 5)
    rotation[2, 0] = -s.Rational(4, 5)
    assert rotation.H * rotation == s.eye(4) and rotation.det() == 1
    assert rotation * gamma[0] != gamma[0] * rotation
    phase = (3 + 4 * s.I) / 5
    rows = []
    identity_rows = []
    for length in (1, 2):
        for mu in range(4):
            joint = s.zeros(4)
            bare = s.zeros(4)
            count = 0
            for labels in itertools.product(centres, repeat=length + 1):
                zp = s.prod(s.conjugate(labels[j]) * labels[j + 1] for j in range(length))
                joint += labels[0] * gamma[mu] * zp * s.conjugate(labels[-1])
                bare += labels[0] * gamma[mu] * s.conjugate(labels[-1])
                count += 1
            assert joint / count == gamma[mu] and bare == s.zeros(4)
            identity_rows.append({'length': length, 'mu': mu, 'Phi': 'C', 'U_links': 'I',
                                  'joint_kernel_exact': True, 'wrong_bare_kernel_zero': True})
    for n in (4, 6, 8):
        for profile in profiles[1:]:
            # Onsite covariance is an adjoint average, unlike a bare fundamental carrier.
            for j in range(4):
                assert sum((z * gamma[j] * s.conjugate(z) for z in centres), s.zeros(4)) / 4 == gamma[j]
            for path in profile['paths']:
                axis, length = path['axis'], path['length']
                start = n - length
                vertices = [(start + k) % n for k in range(length + 1)]
                frames = [rotation ** (vertex % 3) for vertex in vertices]
                links = [frames[k] * frames[k + 1].H for k in range(length)]
                k = s.Rational(path['c0']) * gamma[0] / 2 + s.I * s.Rational(path['ci']) * gamma[axis + 1] * s.Rational(2, 3)
                actual = -phase ** length * frames[0] * k * frames[-1].H
                assert hs2(actual) > 0
                joint = s.zeros(4)
                independent = s.zeros(4)
                count = 0
                # Compute actual frame/link matrices, not a principal-log approximation.
                cache = {}
                for edge, link in enumerate(links):
                    for zx, zy in itertools.product(centres, repeat=2):
                        gx, gy = frames[edge] * zx, frames[edge + 1] * zy
                        v = gx.H * link * gy
                        distances = [hs2(v - z * s.eye(4)) for z in centres]
                        z = centres[distances.index(min(distances))]
                        assert min(distances) == 0 and z == s.conjugate(zx) * zy
                        cache[edge, zx, zy] = z
                for labels in itertools.product(centres, repeat=length + 1):
                    zp = s.prod(cache[edge, labels[edge], labels[edge + 1]] for edge in range(length))
                    bare_scalar = labels[0] * s.conjugate(labels[-1])
                    joint_scalar = bare_scalar * zp
                    assert s.simplify(joint_scalar) == 1
                    joint += joint_scalar * actual
                    independent += bare_scalar * actual
                    count += 1
                joint = joint.applyfunc(s.simplify) / count
                independent = independent.applyfunc(s.simplify) / count
                assert joint == actual and independent == s.zeros(4)
                rows.append({'N': n, 'role': profile['role'], 'axis': axis, 'length': length,
                             'centre_assignments_exact': count, 'AP': -1,
                             'once_shared_integer_flow': vertices[:-1],
                             'joint_kernel_carrier_HS_squared': str(hs2(joint)),
                             'wrong_bare_independent_kernel_carrier_HS_squared': '0',
                             'joint_kernel_error_squared': '0'})
    return {'rows': rows, 'all36_native_profile_path_cases_passed': len(rows) == 36,
            'exact_identity_body_controls': identity_rows,
            'independent_bare_kernel_kills_fundamental_carrier': True,
            'existing_joint_z_product_kernel_retains_exact_native_carrier': True,
            'onsite_adjoint_centre_average_retained': True,
            'frames_noncommuting_SU4_not_a_global_adoption': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for name, digest in request['pins'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    profiles = json.loads(Path(request['profiles']).read_text())['primitive_profiles']
    prior = json.loads(Path(request['prior_receipt']).read_text())
    result = {'kind': 'NEW_EXACT_JOINT_CENTRE_KERNEL_CONTROL_AND_PINNED_SOURCE_COMPOSITION',
              'centre_controls': centre_controls(profiles),
              'quantitative_bound_from_immutable_epoch1': prior['quantitative_bound'],
              'prior_receipt_read_only_not_executed': request['prior_receipt'],
              'unchanged_proof_comparator': 'g_x K z_p g_y^dag with z_e nearest(g_x^dag U_e g_y), z_p=ordered product',
              'law_changed': False, 'actual_rhoN_or_entireP0_replaced': False,
              'H_domain_or_phase_adoption': False, 'historical_consumer_import_or_execution': False}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'joint_centre_cases': 36,
                      'wrong_independent_centre_kernel_rejected': True, 'existing_joint_kernel_exact': True}))


if __name__ == '__main__':
    main()
