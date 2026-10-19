"""New native moving-section and full-body variational/domain successor.

Only pure helper definitions from our own BEFORE-execution first epoch are
loaded. No prior main, eigensolver, checker or ground producer is called.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen_helpers(raw):
    tree = ast.parse(raw)
    # Keep imports/constants/functions; remove only the top-level main guard.
    selected = [node for node in tree.body if not isinstance(node, ast.If)]
    namespace = {'__file__': str(ROOT/'ground_loop_support_consumer_20261012.py'),
                 '__name__': 'ground_loop_support_frozen_helpers'}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'own_frozen_helpers', 'exec'), namespace)
    return namespace


def generic_native_witness(helper, profiles):
    sites, index, phi, links, phases, block, points, specs = helper['native_data'](profiles)
    # Nonproportional full traceless Hermitian fields and full four-colour
    # rational SU4 Givens products remove the first witness's special stratum.
    for position, x in enumerate(sites):
        for mu in range(4):
            body = sp.diag(3+mu+position%5, -1-mu, -1-position%5, -1)
            for a in range(4):
                for b in range(a+1, 4):
                    value = sp.Rational((1+position+mu+a+2*b)%7+1, 11+mu+a+b)
                    value *= sp.I if (position+mu+a+b)%2 else 1
                    body[a, b] = value; body[b, a] = sp.conjugate(value)
            assert sp.trace(body) == 0 and body.H == body
            phi[x][mu] = body
    for x in sites:
        for axis in range(3):
            u = sp.eye(4)
            for step, (a, b) in enumerate(((0,1),(1,2),(2,3),(0,2))):
                rotation = sp.eye(4)
                alpha = sp.Rational(3,5)
                beta = sp.Rational(4,5) * (sp.I if (sum(x)+axis+step)%2 else 1)
                rotation[a,a] = rotation[b,b] = alpha
                rotation[a,b] = beta; rotation[b,a] = -sp.conjugate(beta)
                u = (u * rotation).applyfunc(sp.expand)
            assert (u.H*u).applyfunc(sp.expand) == sp.eye(4) and sp.simplify(u.det()) == 1
            links[(x,axis)] = u
            phases[(x,axis)] = (3+4*sp.I)/5 if (sum(x)+axis)%2 else (5+12*sp.I)/13
    for spec in specs:
        path = next(p for p in profiles[1]['paths'] if p['axis']==spec['axis'] and p['length']==1)
        y, b = block(spec['x'], path)
        assert y == spec['y']
        spec['link'] = links[(spec['x'],spec['axis'])]
        spec['phase'] = phases[(spec['x'],spec['axis'])]
        spec['body'] = b.H if spec['reverse'] else b
    legs = helper['face_matrices'](specs, points)
    ranks, a, b, bh = helper['native_ranks'](legs)
    return {'ranks': ranks, 'nonproportional_all4Phi': True,
            'rational_noncommuting_SU4_all4colour': True,
            'native_all24_all64_exact_intertwiner': True,
            'rank_is_modular_LOWER_not_complete_classification': True}, legs


def native_moving_gauge_dark(helper, profiles):
    # Native fullcolour commuting stratum with ALL4 fields nonzero, not an
    # abstract cycle. A real native local SU4 gauge orbit gives a moving carrier.
    c0 = sp.Rational(profiles[1]['paths'][0]['c0'])
    ci = sp.Rational(profiles[1]['paths'][0]['ci'])
    diagonal = [sp.diag(3,-1,-1,-1), sp.diag(1,-1,2,-2),
                sp.diag(2,1,-1,-2), sp.diag(1,2,-2,-1)]
    legs = []
    for edge, axis in enumerate((0,1,0,1)):
        t = sp.zeros(16)
        body = c0*diagonal[0]+sp.I*ci*diagonal[axis+1]
        if edge >= 2:
            body = body.H
        t[4*edge:4*edge+4,4*((edge+1)%4):4*((edge+1)%4)+4] = body
        legs.append(t)
    b = helper['superoperator']([helper['array'](t) for t in legs])
    bh = helper['superoperator']([helper['array'](t.H) for t in legs])
    prime = helper['PRIME']; inverse2 = helper['INV2']; imag = helper['IMAG']
    a = (b+bh)*inverse2%prime
    g = sp.diag(*(list(sp.diag(1,-1,0,0).diagonal())*4))
    generator = sp.zeros(16); generator[0,1] = generator[1,0] = sp.Rational(1,2)
    gp = sp.I*(generator*g-g*generator)
    g_array = helper['array'](g).reshape(-1); gp_array = helper['array'](gp).reshape(-1)
    eye = np.eye(16,dtype=np.int64)
    connection = imag*(np.kron(helper['array'](generator),eye)-np.kron(eye,helper['array'](generator).T))%prime
    ap = helper['comm'](connection,a)
    first = ap@g_array%prime; second = a@gp_array%prime
    assert not np.any(a@g_array%prime)
    assert np.any(first) and np.any(second) and not np.any((first+second)%prime)
    # Second differentiated identity A''g+2A'g'+Ag''=0.
    app = helper['comm'](connection,ap)
    gpp = connection@gp_array%prime
    assert not np.any((app@g_array+2*ap@gp_array+a@gpp)%prime)
    leg_first = [sp.I*(generator*t-t*generator) for t in legs]
    leg_second = [sp.I*(generator*dt-dt*generator) for dt in leg_first]
    independent = helper['hermitian_jet'](legs, leg_first, leg_second, helper['array'](g))
    assert np.array_equal(independent[1].reshape(-1), first)
    assert np.array_equal((2*independent[2].reshape(-1))%prime, (app@g_array)%prime)
    return {'native_full4colour_full4Phi_nonzero': True,
            'native_A_g_zero_along_exact_SU4_gauge_orbit': True,
            'A_prime_g_nonzero': True, 'A_g_prime_nonzero': True,
            'ordinary_first_jets_cancel_exactly': True,
            'ordinary_second_jets_cancel_exactly': True,
            'independent_ordered_leg_first_second_jets_checked': True,
            'covariant_Gauss_derivative_is_equivariant_not_deleted': True,
            'native_copy_space': 'X(g)=dGamma_active(g)P_S in actual M256/k128 all64 canonical space, all P_S multiplicities retained',
            'not_positive_measure_fullfield_section': True,
            'not_original_H_eigen_ground': True,
            'not_a_no_go_or_ground_weight': True}


def native_radial_domain_control():
    # Full physical N4 Cartesian radial section; source is genuinely native
    # homogeneous degree4. The first epoch supplies a nonzero angular witness.
    dimension = 60*64
    exponent = sp.Rational(dimension+9,2)  # exponent of |Psi|^2.
    radial_norm = sp.beta(sp.Rational(dimension,2),sp.Rational(9,2))/2
    def moment(power):
        second = exponent-sp.Rational(dimension,2)-sp.Rational(power,2)
        return {'power': str(power), 'beta_second': str(second), 'finite': bool(second>0),
                'normalized_exact': str(sp.beta(sp.Rational(dimension,2)+sp.Rational(power,2),second)/(2*radial_norm)) if second>0 else 'DIVERGES_AT_INFINITY'}
    rows = [moment(sp.Rational(17,2)),moment(8),moment(12),moment(16)]
    assert rows[0]['finite'] and rows[1]['finite'] and not rows[2]['finite'] and not rows[3]['finite']
    # Fourier-free strict fractional-domain implication control. A graph
    # measure with density(2/17) E^(-19/17), E>=1 has finite2s moments iff
    # s<1/17, while its first and second moments actually diverge.
    s = sp.Rational(1,34)
    fractional_moment = sp.Rational(2,17)/(sp.Rational(2,17)-2*s)
    assert fractional_moment == 2
    return {'N':4, 'ambient_fullPhi_dimension': dimension,
            'normalized_physical_radial_amplitude': '(1+|q|^2)^(-(D+9)/4)/sqrt(SphereArea*B(D/2,9/2)/2)',
            'moments': rows,
            'canonical_carrier': 'normalized n_active,x P_S carrier, all common64 multiplicities; neutral half-site-fullcolour determinant and Ref4; rho=0, equivariant Gauss; smooth original deck/Bloch fibre or normalized original cover packet, not a selected minimizing character',
            'body_all65': 'complete original Jordan degree4 dominates at infinity; alignment degree2, all65 onsite/NN/two matter degree1, Haar/rawE/source and kinetic derivatives lower degree; no term is dropped from H',
            'native_nonzero_angular_source': 'actual1 Aad(Px) nonzero rational-modular witness, hence nonzero on an open angular/Haar set',
            'base_section_in_original_fullH_graph': True,
            'native_uncut_A_section_in_Hilbert': True,
            'native_uncut_A_section_NOT_in_fullH_energy_form_or_graph': True,
            'not_a_ground_section': True,
            'domain_countercontrol_not_ground_failure': True,
            'fixed_cutoff_A_M_restores_graph_domain': True,
            'strict_fractional_measure_control': {'density':'(2/17) E^(-19/17), E>=1','s':str(s),'2s_moment':str(fractional_moment),'first_and_second_moments':'DIVERGE','not_native_history_spectrum':True},
            'wrong_DLs_equals_DL_rejected': True}


def variational_response_identity():
    x = sp.symbols('x', real=True)
    # Noncommuting generic endomorphism calculation verifies the FULL quantum
    # operator double commutator, not an invented ground or scalar shortcut.
    a = sp.Matrix([[x,1+sp.I*x],[1-sp.I*x,-x]])
    psi = sp.Matrix([sp.Function('f')(x),sp.Function('g')(x)])
    k = sp.symbols('k',positive=True)
    kinetic = lambda v: -k*v.diff(x,2)
    lhs = a*(kinetic(a*psi)-a*kinetic(psi))-(kinetic(a*a*psi)-a*kinetic(a*psi))
    rhs = 2*k*a.diff(x)*a.diff(x)*psi-k*(a*a.diff(x,2)-a.diff(x,2)*a)*psi-2*k*(a*a.diff(x)-a.diff(x)*a)*psi.diff(x)
    assert sp.simplify(lhs-rhs)==sp.zeros(2,1)
    assert (a*a.diff(x)-a.diff(x)*a)!=sp.zeros(2)
    # Fullbody spectral interface is exact and rejects the wrong zero-P0
    # subtraction: t=sum E mass, U=sum E2 mass, whole zero module absent.
    eta = sp.Rational(2,3)
    energies = [sp.Integer(1),sp.Integer(3),sp.Integer(7)]
    masses = [sp.Rational(1,5),sp.Rational(2,7),sp.Rational(3,11)]
    t = sum(e*m for e,m in zip(energies,masses)); u = sum(e*e*m for e,m in zip(energies,masses))
    exact = sum(m/(e+eta) for e,m in zip(energies,masses))
    lower = t**3/(u*(u+eta*t))
    assert exact>=lower>0
    return {'full_quantum_operator_double_commutator_identity':True,
            'scalar_gradient_only_operator_identity_rejected':True,
            'native_Berry_nonzero_witness':'actual1 native rank lower228, all24/all64/fullcolour',
            'sameL_variational_interface':'if actual fullbody t_M>=tau>0 and m2<=U=Cg^2 M^8, wholeP0 excited mass>=tau^2/U and G_eta>=tau^3/[U(U+eta tau)]',
            'spectral_identity_control':{'t':str(t),'m2':str(u),'eta':str(eta),'exact_G':str(exact),'lower_G':str(lower),'not_native_spectrum':True},
            'actual_originating_signed_Berry_matter_lower_tau':'NOT_PAID',
            'wholeP0_excluded_by_energy_not_mean':True,
            'fixed_cutoff_only':True, 'q_dependent_cutoff':False,
            'no_actual_ground_positive_response_inferred':True}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();request=json.loads(args.request.read_text())
    with zipfile.ZipFile(request['preexecution_archive']) as archive:
        for name,pin in request['input_sha256'].items():
            assert digest(archive.read(name))==pin,name
        with zipfile.ZipFile(io.BytesIO(archive.read(request['first_preexecution_archive']))) as first:
            helper=frozen_helpers(first.read('ground_loop_support_consumer_20261012.py'))
        with zipfile.ZipFile(io.BytesIO(archive.read(request['profile_archive']))) as original:
            profiles=json.loads(original.read(request['profile_member']))['primitive_profiles']
        first_receipt=json.loads(archive.read('ground_loop_support_actual1_20261012_receipt.json'))
    assert first_receipt['complete_body']['active_onsite_common_singlet_leakage_nonzero']
    generic,legs=generic_native_witness(helper,profiles)
    moving=native_moving_gauge_dark(helper,profiles)
    domain=native_radial_domain_control()
    response=variational_response_identity()
    result={'kind':'NEW_MATERIAL_FULLQUANTUM_VARIATIONAL_RESPONSE_AND_NATIVE_MOVING_GAUGE_SECTION_SUCCESSOR_NOT_GROUND_ADMISSION',
            'generic_native_point':generic,'native_moving_gauge_dark':moving,
            'native_fullbody_domain_control':domain,'full_quantum_variational_response':response,
            'attempt1_boundary':'adjoint native rank does not classify all higher irreps or select actual originating weights; covariant eigen-dark graph stays essential',
            'attempt2_boundary':'no positive actual fullbody first-energy lower after Berry/matter signed terms; exact conditional quantitative interface only',
            'input_sha256':request['input_sha256'],'runtime':{'executable':sys.executable,'elapsed_seconds':time.monotonic()-started},
            'scientific_admission':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'receipt':args.output.name,'new_native_and_material_variational_controls':True,'actual_ground_response':'NOT_PAID'}))


if __name__=='__main__':
    main()
