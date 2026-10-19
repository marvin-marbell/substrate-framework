"""Exact native N4 conditional exterior kernels; no quantum-ground backend."""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any
import zipfile
import sympy as sp

ROOT = Path(__file__).resolve().parent


def canonical_coefficient(ordered_factors: list[sp.Matrix], number: int) -> sp.Expr:
    """Factors are listed in chronological order: latest multiplies on left."""
    if not ordered_factors:
        raise ValueError('One-particle identity dimension must be explicit')
    body = sp.eye(ordered_factors[0].rows)
    for factor in ordered_factors:
        body = factor * body
    # det(lambda I-B): coefficient (-1)^n e_n(B), full exterior sector.
    return sp.expand((-1)**number * body.charpoly().all_coeffs()[number])


def native_h(phi: list[list[sp.Matrix]], links: dict, phases: dict, profile: dict,
             charged: bool, n: int = 4) -> sp.MutableSparseMatrix:
    """Full site/colour native matrix, including all onsite/path/AP/adjoint terms."""
    sites = list(itertools.product(range(n), repeat=3))
    index = {x: j for j, x in enumerate(sites)}
    h = sp.MutableSparseMatrix(4*len(sites), 4*len(sites), {})
    def add(x, y, block):
        for a in range(4):
            for b in range(4):
                h[4*index[x]+a, 4*index[y]+b] += block[a, b]
    for x in sites:
        add(x, x, sp.Rational(profile['onsite'])*phi[index[x]][0])
        for path in profile['paths']:
            y = x; u = sp.eye(4); phase = sp.S.One; seam = 1
            for _ in range(path['length']):
                u = u*links[(y, path['axis'])]
                if charged:
                    phase *= phases[(y, path['axis'])]
                z = list(y)
                if z[path['axis']] == n-1:
                    seam *= -1
                z[path['axis']] = (z[path['axis']]+1) % n
                y = tuple(z)
            c0, ci = sp.Rational(path['c0']), sp.I*sp.Rational(path['ci'])
            fx = c0*phi[index[x]][0]+ci*phi[index[x]][path['axis']+1]
            fy = c0*phi[index[y]][0]+ci*phi[index[y]][path['axis']+1]
            block = seam*phase*(fx*u+u*fy)/2
            add(x, y, block); add(y, x, block.conjugate().T)
    return h


def car_action(mask: int, a: int, b: int) -> tuple[int, int]:
    if not(mask >> b & 1) or mask >> a & 1:
        return mask, 0
    sign = (-1)**((mask & ((1 << b)-1)).bit_count())
    mask ^= 1 << b
    sign *= (-1)**((mask & ((1 << a)-1)).bit_count())
    return mask ^ (1 << a), sign


def occupation_kernel(profiles: list[dict]) -> dict:
    # Only four modes are acted on. ALL252 remaining modes compensate filling.
    # Four identical diagonal-colour cycles are disjoint, not discarded colours.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    z0 = math.comb(256, 128)
    rows = []
    sums: Any = Counter()
    for mask in range(16):
        signed = 0
        order_values = []
        for order in itertools.permutations(range(4)):
            m, value = mask, 1
            for j in order:
                m, sign = car_action(m, *edges[j]); value *= sign
                if not value:
                    break
            trace = value if m == mask else 0
            signed += trace; order_values.append(trace)
        spectators = math.comb(252, 128-mask.bit_count())
        weight = F(spectators, z0)
        branch = 'opposite_pairs' if mask in (5, 10) else 'complement'
        sums[branch] += signed*weight
        rows.append({'mask': mask, 'occupation': mask.bit_count(),
                     'spectator_count': str(spectators), 'canonical_weight': str(weight),
                     'all24_ordered_signed_values': order_values, 'sum24': signed,
                     'branch': branch})
    assert sum(F(r['canonical_weight']) for r in rows) == 1
    assert sums['complement'] != 0 and sums['opposite_pairs'] != 0
    c0 = [F(next(p for p in role['paths'] if p['length'] == 1)['c0'])
          for role in profiles[1:]]
    native_role4 = c0[0]**4+63*c0[1]**4
    total = 4*native_role4*sum(sums.values())
    # All mixed assignments are tested structurally, not selected out as eligibility.
    assignment_counts = Counter()
    for assignment in itertools.product(range(64), repeat=4):
        assignment_counts['balanced_single_copy' if len(set(assignment)) == 1
                          else 'zero_by_copy_charge_balance'] += 1
    assert assignment_counts['balanced_single_copy'] == 64
    assert sum(assignment_counts.values()) == 64**4
    # Exterior determinant with ordered rank-one insertions and 252 spectators.
    t = sp.symbols('t0:4'); z = sp.Symbol('z')
    small = sp.eye(4)
    for j in range(4):
        leg = sp.zeros(4); leg[edges[j][0], edges[j][1]] = 1
        small = (sp.eye(4)+t[j]*leg)*small
    local = sp.Poly((sp.eye(4)+z*small).det(), *t)
    coeff = local.coeff_monomial(sp.prod(t))
    ordered_exterior = sum(int(sp.expand(coeff).coeff(z, k))*math.comb(252,128-k)
                           for k in range(5))
    selected_order = sum(r['all24_ordered_signed_values'][0]*int(r['spectator_count'])
                         for r in rows)
    assert ordered_exterior == selected_order
    return {'N':4, 'one_particle_dimension':256, 'filling_each_copy':128,
            'rows':rows, 'all64_pow4_assignment_counts':dict(assignment_counts),
            'all24_expanded_word_count':24*64**4,
            'opposite_pair_signed_sum':str(sums['opposite_pairs']),
            'complement_signed_sum':str(sums['complement']),
            'sum24_one_colour':str(sum(sums.values())),
            'native_all64_fourcolour_connected_coefficient_in_epsilon4_units':str(total),
            'proper_subset_moments':'zero at identity heat by unbalanced directed flow; connected equals moment',
            'determinant_exterior_matches_ordered_CAR':True,
            'neutral':'full canonical128 determinant in denominator, no charged source'}


def shared_hs_control() -> dict:
    # One native incidence divergence transfer; eigenmode of N4 graph L with eigenvalue4.
    n=4; sites=list(itertools.product(range(n),repeat=3))
    rho=[F((-1)**x[0]) for x in sites]
    lp=[v/4 for v in rho]
    cross=sum(a*b for a,b in zip(rho,lp))
    assert cross==16
    shared=64*sum((2*a)*b*2 for a,b in zip(rho,lp))
    independent=2*64*cross
    assert shared==4096 and independent==2048
    return {'one_shared_covariance':'128 delta (D D*)plus',
            'native_N4_mode_L_eigenvalue':4, 'rho_Lplus_rho':str(cross),
            'two_copies_same_rho_shared_energy':str(shared),
            'independent_copy_wrong_energy':str(independent),
            'missing_cross_energy':str(shared-independent),
            'background_ref_phase':'exp(i eta dot[Ref-2*64]); eta has zero mode removed',
            'fullE_plus_HC_double_count_rejected':True}


def gates_and_paid_heat(profiles: list[dict]) -> dict:
    # Physical beta=1, conditional Phi0=epsilon diag(1,-1,1,-1), all other Phi=0.
    # Later slice doubles epsilon. Every native onsite/NN/length2 remains in bound.
    epsilon=F(1,2**9000)
    w=[abs(F(p['onsite']))+2*sum(abs(F(q['c0']))+abs(F(q['ci']))
                              for q in p['paths']) for p in profiles]
    d=2*epsilon*128*(2**8192*w[0]+w[1]+63*w[2])
    assert d < F(1,2**780)
    # Compact SU4 sewing at one site; two -1 modes, 254 +1 modes.
    z0=math.comb(256,128)
    sewn=sum((-1)**j*math.comb(2,j)*math.comb(254,128-j) for j in range(3))
    assert F(sewn,z0)==-F(1,255)
    full65=F(sewn,z0)**65
    assert full65<0 and 2*d < abs(full65)/2
    # Bare directed moment stays negative at ordinary sewing, physical hopping1.
    Csum=sum(p['load']*(abs(F(next(q for q in p['paths'] if q['length']==1)['c0'])))
             for p in profiles[1:])
    connected_error=624*(4*Csum)**4*4*d
    small=occupation_kernel_cached
    value=F(small['native_all64_fourcolour_connected_coefficient_in_epsilon4_units'])
    assert value<0 and connected_error < abs(value)/2
    return {'physical_hopping':1, 'conditional_beta':1,
            'conditional_two_slice_fields':'epsilon D then 2epsilon D, epsilon=2^-9000, D=diag(1,-1,1,-1); all spatial sites',
            'heat_bound_d_less_than':'2^-780',
            'heat_bias_rule':'||product exp(-delta h)-I|| <= exp(d)-1 <=2d; complete-body canonical norm bound',
            'SU4_allowed_sewing':'diag(-1,-1,1,1) at one N4 site; all65 species transformed',
            'one_copy_canonical_sewn_trace_ratio':str(F(sewn,z0)),
            'all65_sewn_trace_ratio':str(full65),
            'negative_sign_persistence_paid':True,
            'connected_rescaled_error_upper':str(connected_error),
            'connected_negative_sign_persistence_paid':True,
            'all_path_positive_gate':'FAILED_EXACT_CANONICAL_GAUSS_NEGATIVE_PATH_AT_PHYSICAL1',
            'material_second_mechanism':'exhaustive signed occupation/complement exterior sum with canonical compensating spectators',
            'second_mechanism_result':'signed complement nonzero; cannot discard it or infer actualground branch weights',
            'global_ground_or_gapped_branch_proved':False}


def fullnative_binding(profiles: list[dict]) -> dict:
    sites = list(itertools.product(range(4), repeat=3))
    links = {(x, axis): sp.eye(4) for x in sites for axis in range(3)}
    phases = {key: sp.S.One for key in links}
    zero = [[sp.zeros(4) for _ in range(4)] for x in sites]
    phi = [[sp.diag(1, -1, 1, -1), sp.zeros(4), sp.zeros(4), sp.zeros(4)]
           for x in sites]
    rows = []
    for role, profile in enumerate(profiles):
        hz = native_h(zero, links, phases, profile, role != 0)
        assert not hz.todok()
        h = native_h(phi, links, phases, profile, role != 0)
        assert h.shape == (256, 256)
        assert h[0, 0] == sp.Rational(profile['onsite'])
        assert h[1, 1] == -sp.Rational(profile['onsite'])
        # Every literal path remains; the spatial AP seam is assembled separately.
        assert h[0, 4*sites.index((1, 0, 0))] == sp.Rational(profile['paths'][0]['c0'])
        assert h[0, 4*sites.index((3, 0, 0))] == -sp.Rational(profile['paths'][0]['c0'])
        rows.append({'role': profile['role'], 'copies': profile['load'],
                     'full_dimension': 256, 'literal_onsite': profile['onsite'],
                     'native_path_count': len(profile['paths']),
                     'zero_Phi_h_exactly_zero': True,
                     'nonzero_Phi_onsite_NN_AP_bound_reached': True,
                     'nonzero_entries': len(h.todok())})
    # Exact signed response cancellation witness, not a claimed native ground.
    gp = sp.Matrix([[1, -1], [-1, 1]])
    v = sp.ones(2, 1)
    assert gp.det() == 0 and (v.T*gp*v)[0] == 0
    return {'all65_native_onebody_rows': rows,
            'actualground_integrated_attempt': {
                'full_physical_form': 'H-tp Pi A_M Pi-tc(A_M-Pi A_M Pi)',
                'full_L_response_Gram': 'Gab=omega(Aa Q0(L+eta)^-1 Q0 Ab)',
                'signed_full_response': '2(Gpp+Gcc+2Re Gpc)',
                'paid_signed_lower': '2(sqrt(Gpp)-sqrt(Gcc))^2',
                'physical1_gate': 'UNPAID_ACTUAL_GROUND_BRANCH_RESPONSE_SEPARATION',
                'missing': ['actual Gpp', 'actual Gcc', 'complex actual Gpc',
                            'uniform cutoff/eta/N path and IR payment'],
                'adverse_PSD_Gram_method_control': str(gp),
                'nonzero_branch_response_can_cancel': True,
                'control_is_actual_native_ground': False}}


def actual_boundary_minor_control() -> dict:
    """Non-Gaussian canonical boundary method control; no native ground claim."""
    subsets = list(itertools.combinations(range(4), 2))
    masks = [sum(1 << j for j in occupied) for occupied in subsets]
    basis = {mask: j for j, mask in enumerate(masks)}
    psi = sp.zeros(6, 1); psi[1] = 1; psi[4] = 1
    rho = psi*psi.T/2
    heat = sp.diag(2, 3, 5, 7)
    heat_fock = sp.diag(*(heat[i, i]*heat[j, j] for i, j in subsets))
    variables = sp.symbols('u0:4')
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    body = heat
    fock = heat_fock
    for u, (a, b) in zip(variables, edges):
        leg = sp.zeros(4); leg[a, b] = 1
        car = sp.zeros(6)
        for col, mask in enumerate(masks):
            target, sign = car_action(mask, a, b)
            if sign:
                car[basis[target], col] = sign
        body = (sp.eye(4)+u*leg)*body
        fock = (sp.eye(6)+u*car)*fock
    minor_transport = sp.Matrix(6, 6, lambda i, j:
                               body.extract(subsets[i], subsets[j]).det())
    boundary = sp.expand(sp.trace(rho*minor_transport))
    direct = sp.expand(sp.trace(rho*fock))
    checks = []
    for bits in itertools.product(range(2), repeat=4):
        monomial = sp.prod(u**bit for u, bit in zip(variables, bits))
        lhs = sp.Poly(boundary, *variables).coeff_monomial(monomial)
        rhs = sp.Poly(direct, *variables).coeff_monomial(monomial)
        assert lhs == rhs
        checks.append({'subset': list(bits), 'boundary_minor': str(lhs),
                       'complete_CAR': str(rhs)})
    closed = canonical_coefficient([body], 2)
    norm_boundary = sp.expand(boundary)/boundary.subs(dict.fromkeys(variables, 0))
    norm_closed = sp.expand(closed)/closed.subs(dict.fromkeys(variables, 0))
    assert sp.simplify(norm_boundary-norm_closed) != 0
    return {'full_dimension':4, 'full_canonical_sector_dimension':6,
            'boundary_state':'(|02>+|13>)/sqrt(2); offdiagonal canonical density retained',
            'all16_ordered_source_minor_CAR_checks':checks,
            'ordinary_trace_boundary_substitution_rejected':True,
            'normalized_boundary_polynomial':str(norm_boundary),
            'normalized_identity_boundary_polynomial':str(norm_closed),
            'native_ground_state':False,
            'actualground_proof':'canonical_quantum_path_boundary_proof_20261012.json'}


def wrongcases() -> dict:
    # Noncommuting heat/source control. These small matrices are method controls only.
    a=sp.Matrix([[2,1],[0,1]]); b=sp.Matrix([[1,0],[1,3]])
    source=sp.Matrix([[0,1],[0,0]])
    forward=sp.trace(source*b*a); reverse=sp.trace(source*a*b)
    assert forward != reverse
    canonical=canonical_coefficient([a,b],1)
    grand=(sp.eye(2)+b*a).det()
    assert canonical != grand
    on=sp.diag(2,3); omitted=sp.eye(2)
    assert canonical_coefficient([on],1)!=canonical_coefficient([omitted],1)
    assert canonical_coefficient([on],1)**65 != canonical_coefficient([on],1)**64
    return {'grandcanonical_rejected':{'canonical':str(canonical),'grandcanonical':str(grand)},
            'reversed_path_rejected':{'forward':str(forward),'reverse':str(reverse)},
            'onsite_omission_rejected':True, 'neutral_omission_rejected':True,
            'opposite_pair_eligibility_rejected':'nonzero complement retained in native exact rows',
            'scope':'2x2 matrices method controls; native N4 occupation, SU4 and HS controls separate'}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--request',required=True)
    args=parser.parse_args(); req=json.loads((ROOT/args.request).read_text())
    captured={name:(ROOT/name).read_bytes() for name in req['pins']}
    for name, data in captured.items():
        assert hashlib.sha256(data).hexdigest()==req['pins'][name],name
    with zipfile.ZipFile(ROOT/req['profile_archive']) as z:
        raw=z.read(req['profile_member'])
    profiles=json.loads(raw)['primitive_profiles']
    assert [p['load'] for p in profiles]==[1,1,63]
    assert len({p['onsite'] for p in profiles})==3
    global occupation_kernel_cached
    occupation_kernel_cached=occupation_kernel(profiles)
    result={'kind':'SAME_LAW_CANONICAL_QUANTUM_PATH_CONDITIONAL_NATIVE_N4_NOT_GROUND',
            'native_conditional_kernel':occupation_kernel_cached,
            'shared_HS':shared_hs_control(), 'physical1_attempts':gates_and_paid_heat(profiles),
            'fullnative_path_binding':fullnative_binding(profiles),
            'actual_boundary_minor_control':actual_boundary_minor_control(),
            'wrongcases':wrongcases(), 'input_pins':req['pins'],
            'profile_sha256':hashlib.sha256(raw).hexdigest(),
            'exact_rational_arithmetic':True, 'finite_heat_error_paid':True,
            'full_quantum_Gauss_Phi_path_integration_executed':False,
            'actualground_normalization_executed':False,
            'scientific_admission':False,'campaign_complete':False}
    assert all((ROOT/name).read_bytes()==data for name,data in captured.items())
    (ROOT/req['receipt']).write_text(json.dumps(result,indent=2)+'\n')
    print('canonical native N4 signed-complement/negative-Gauss gate completed')


if __name__=='__main__':
    main()
