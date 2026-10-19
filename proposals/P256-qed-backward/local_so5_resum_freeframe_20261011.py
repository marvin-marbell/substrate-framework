"""Exact native binary profile and analytic free5 density frame for exterior consumer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import sympy as s
import bulk_electric_phase_consumer as bound


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    own = Path(__file__).read_bytes()
    members = dict(bound.CAPTURED)
    members[Path(__file__).name] = own
    for name in [bound.INPUTS[0], bound.INPUTS[1], args.receipt]:
        members[name] = (root/name).read_bytes()
    observed = json.loads(members[args.receipt])
    for name in [bound.INPUTS[0], bound.INPUTS[1]]:
        assert hashlib.sha256(members[name]).hexdigest() == observed['source_and_input_sha256'][name]
    frame, scale = bound.frozen.exact_frame(members[bound.INPUTS[0]])
    cfg = json.loads(members[bound.INPUTS[1]])['input_descriptor']['original_micro_config_UNCHANGED']
    profiles = bound.frozen.profiles(cfg, frame, scale)
    rat = lambda x: s.Rational(x.numerator, x.denominator)
    cm = lambda matrix: s.Matrix([[rat(z[0])+s.I*rat(z[1]) for z in row] for row in matrix])
    f1 = cm(frame[1])
    q = s.trace(f1*f1)/4
    assert f1*f1 == q*s.eye(4)
    gamma = f1/s.sqrt(q)
    spin = s.Matrix.hstack(s.eye(4)[:, 0], gamma[:, 0], s.eye(4)[:, 1], gamma[:, 1])
    assert s.simplify(spin.H*spin) == s.eye(4)
    m, d, e, c = s.symbols('m d e c', real=True, nonzero=True)
    rt = s.sqrt(2)
    l = s.Matrix([(e+m)/(2*e), -s.I*d/(rt*e), (m-e)/(2*e)])
    middle = s.Matrix([-s.I*d/(rt*e), m/e, -s.I*d/(rt*e)])
    high = s.Matrix([(m-e)/(2*e), -s.I*d/(rt*e), (e+m)/(2*e)])
    triplet = s.Matrix([[1,0,0],[0,1/rt,0],[0,1/rt,0],[0,0,1],[0,0,0]])
    anti = s.Matrix([0, 1/rt, -1/rt, 0, 0])
    zero = s.eye(5)[:, 4]
    u = s.Matrix.hstack(s.I*triplet*l, (anti+zero)/rt, triplet*middle, (anti-zero)/rt, -s.I*triplet*high)
    a = s.diag(-2, 0, 0, 2, 0)
    b = 2*(anti*zero.T+zero*anti.T)
    raising = s.zeros(5)
    raising[1,0] = raising[3,2] = raising[2,0] = raising[3,1] = 1
    ci = s.I*(raising-raising.T)
    rho = s.diag(0, 1, -1, 0, 0)
    energies = s.diag(-2*e, 2*c, 0, -2*c, 2*e)
    response = s.zeros(5)
    for column, value in [(0,d/(2*e)), (2,m/(rt*e)), (4,-d/(2*e))]:
        for row in [1,3]:
            response[row,column] = response[column,row] = value
    def vanishes(matrix: s.Matrix) -> bool:
        for expression in matrix:
            numerator = s.cancel(expression).as_numer_denom()[0]
            remainder = s.Poly(numerator,e).rem(s.Poly(e**2-m**2-d**2,e)).as_expr()
            if s.simplify(remainder) != 0:
                return False
        return True
    assert vanishes(u.H*u-s.eye(5))
    assert vanishes((m*a+c*b+d*ci)*u-u*energies)
    assert vanishes(u.H*rho*u-response)
    records = []
    for profile, measured in zip(profiles, observed['parameters']):
        path = next(p for p in profile['paths'] if p['axis']==0 and p['length']==1)
        hopping = cm(path['matrix'])
        t = s.simplify(spin.H*hopping*spin)
        c0 = s.simplify(-t[0,0]); native_d = s.simplify(t[0,1]/s.I)
        block = s.Matrix([[-c0,s.I*native_d],[s.I*native_d,c0]])
        assert s.simplify(t-s.diag(block,block)) == s.zeros(4)
        mass = rat(profile['onsite']); native_e = s.sqrt(mass**2+native_d**2)
        assert abs(float(c0)-measured['c0']) < 1e-12 and abs(float(native_d)-measured['ci']) < 1e-12
        record = {'role': profile['role'], 'spin_replicas': 2*profile['count'], 'm_exact': str(mass), 'c0_exact': str(c0), 'c_positive_exact': str(abs(c0)), 'd_exact': str(native_d), 'd_squared_exact': str(s.simplify(native_d**2)), 'e_exact': str(native_e), 'e_squared_exact': str(s.simplify(mass**2+native_d**2)), 'e_float': float(native_e), 'binary_cl2_frame_square_q_exact': str(q), 'native_spin_split_exact': True, 'free_density_amplitudes_exact': {'L_to_Bboth': str(native_d/(2*native_e)), 'M_to_Bboth': str(mass/(rt*native_e)), 'H_to_Bboth': str(-native_d/(2*native_e))}}
        records.append(record)
    output = {'kind': 'EXACT_NATIVE_COMMON_FREE5_DENSITY_TRANSFORM_FOR_PAID_EXTERIOR', 'profiles': records, 'free_basis_order': ['L=-2e','Bplus=2c0=-2c','M=0','Bminus=-2c0=2c','H=2e'], 'native_basis_order': ['ac','ab','cd','bd','(ad+bc)/sqrt2'], 'U_columns_exact_symbolic': [[str(u[i,j]) for j in range(5)] for i in range(5)], 'rho_free_exact_symbolic': [[str(response[i,j]) for j in range(5)] for i in range(5)], 'algebraic_relation': 'e^2=m^2+d^2, e>0,m>0,c0<0', 'symbolic_unitarity_free_H_density_identities': True, 'current_Ritz_mu': observed['zero_u']['energy'], 'current_complete_H_sigma': observed['zero_u']['complete_H_residual'], 'current_complete_H_sigma_squared': observed['zero_u']['complete_H_variance'], 'current_receipt': args.receipt, 'stage': 'EXACT_LOCAL_REPRESENTATION_PLUS_NUMERICAL_RITZ_NOT_GROUND_CERTIFICATE', 'source_and_input_sha256': {name: hashlib.sha256(raw).hexdigest() for name,raw in members.items()}, 'source_and_inputs_unchanged': {name:(root/name).read_bytes()==raw for name,raw in members.items()}, 'actual_command': [sys.executable,*sys.argv]}
    assert all(output['source_and_inputs_unchanged'].values())
    name = 'local_so5_resum_freeframe_proof_20261011.json'
    (root/name).write_text(json.dumps(output,indent=2)+'\n')
    with zipfile.ZipFile(root/'local_so5_resum_freeframe_capsule_20261011.zip','w',zipfile.ZIP_DEFLATED) as capsule:
        for member,raw in members.items():
            capsule.writestr(member,raw)
        capsule.writestr(name,(root/name).read_bytes())
    print(json.dumps({'proof':name,'symbolic_identities':True,'profiles':records,'mu':output['current_Ritz_mu'],'sigma':output['current_complete_H_sigma']}),flush=True)


if __name__ == '__main__':
    main()
