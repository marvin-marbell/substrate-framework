"""New whole-canonical signed positive-residual Galerkin calculation.
Only frozen pure libraries are called, never historical main/eigensolvers.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import time
import numpy as np
import sympy as sp
import vacuum_connected_native_scientist_20261011 as native

ROOT = Path(__file__).resolve().parent
PREFIX = 'positive_residual_'
car, frozen = native.car, native.frozen
GAMMA = F(46955312929, 34359738368)


def herm(terms):
    return car.collect(car.scale(terms, F(1, 2)) + car.scale(native.dagger(terms), F(1, 2)))


def exact_quadratic(terms):
    """Exact Hermitian matrix and exact LDL bisection, no numerical eigensolver."""
    h = [[car.ZERO for _ in range(8)] for _ in range(8)]
    scalar = car.ZERO
    for z, w, m in terms:
        if any(m):
            raise ArithmeticError('Cannot erase totalrho')
        if not w:
            scalar = car.ca(scalar, z)
        elif len(w) == 2 and w[0][3] and not w[1][3] and all(c == 1 for c, s, p, cr in w):
            i, j = 4*w[0][1]+w[0][2], 4*w[1][1]+w[1][2]
            h[i][j] = car.ca(h[i][j], z)
        else:
            raise ArithmeticError(('Not entire eight-mode quadratic', w))
    if scalar[1] or any(h[i][j] != (h[j][i][0], -h[j][i][1]) for i in range(8) for j in range(8)):
        raise ArithmeticError('Hermitian proxy not exact')
    rowr = [sum((car.norm(h[i][j]) for j in range(8) if j != i), F(0)) for i in range(8)]
    global_lo = min(h[i][i][0]-rowr[i] for i in range(8))-1
    global_hi = max(h[i][i][0]+rowr[i] for i in range(8))+1
    intervals = []
    counts = []
    for i in range(8):
        if intervals and counts[-1][1] >= i+1:
            intervals.append(intervals[-1]); counts.append(counts[-1]); continue
        lo, hi = global_lo, global_hi
        lc, hc = 0, 8
        while hi-lo > F(1, 1 << 20):
            mid = (lo+hi)/2
            try:
                count = native.inertia(h, mid)
            except ArithmeticError:
                # Avoid a true eigenvalue or zero intermediate LDL pivot exactly.
                mid = (2*lo+hi)/3
                try:
                    count = native.inertia(h, mid)
                except ArithmeticError:
                    mid = (3*lo+hi)/4
                    count = native.inertia(h, mid)
            if count <= i:
                lo, lc = mid, count
            else:
                hi, hc = mid, count
        intervals.append((lo, hi)); counts.append((lc, hc))
    return scalar[0], intervals, h, counts


def spectrum(roles, errors):
    lower, upper, payload = [], [], []
    scalar = F(0)
    for role, mult in ((0, 1), (1, 63)):
        c, intervals, h, counts = exact_quadratic(roles[role])
        scalar += mult*c
        lower.extend([v[0] for v in intervals]*mult)
        upper.extend([v[1] for v in intervals]*mult)
        payload.append(dict(role=role, physical_count=mult, scalar_exact=str(c), onebody_exact=[[[str(x) for x in z] for z in row] for row in h], eigenvalue_intervals=[frozen.encode(v) for v in intervals], exact_LDL_endpoint_inertias=counts))
    error = errors[0]+63*errors[1]
    return (scalar+sum(sorted(lower)[:256])-error, scalar+sum(sorted(upper)[-256:])+error), payload, error


def combine(left, right, c):
    return [car.collect(left[r]+car.scale(right[r], c)) for r in range(2)]


def controls():
    K = sp.diag(0, sp.Rational(3, 2), 4)
    X = sp.Matrix([[5, 1, 0, 0], [1, 7, 1, 0], [0, 1, 9, 1], [0, 0, 1, 11]])
    J = sp.Matrix([[1, 2, 0], [0, 1, 1], [1, 0, 2], [2, 1, 1]])
    a, E = sp.Rational(6), sp.Rational(1, 4)
    H = K+(a-E)*sp.eye(3); FF = H.inv(); R = (X-E*sp.eye(4)).inv()
    T = X*J-J*K-a*J; G = J.T*J
    B = (J.T*T+T.T*J)/2
    W = J.T*R*J; V = (G*FF+FF*G)/2-FF*B*FF
    P = FF*T.T*R*T*FF
    assert W == V+P
    assert G*K != K*G and B*FF != FF*B
    assert W != (G*FF+FF*G)/2-B*FF**2+P
    assert B*FF**2 != (B*FF**2).T
    assert W != V
    assert J.T*T-T.T*J == K*G-G*K
    assert (X-4*sp.eye(4)).is_positive_definite
    assert P.is_positive_definite
    assert (FF*T.T*T*FF/(4-E)-P).is_positive_definite
    # Same whole-domain operator-anchor Galerkin identity, noncommuting H.
    Z = sp.Matrix([[0, 1, 0], [1, 2, 1], [0, 1, 1]])/3
    HH = H+Z; FY = HH.inv(); TY = (X-E*sp.eye(4))*J-J*HH
    BY = (J.T*TY+TY.T*J)/2
    assert W == (G*FY+FY*G)/2-FY*BY*FY+FY*TY.T*R*TY*FY
    # A one-sided upper bound alone cannot pay an absolute energy norm.
    bad = -5*sp.eye(2); kq = sp.eye(2)
    assert (kq-bad).is_positive_definite and not (kq+bad).is_positive_definite
    return dict(exact_noncommuting_identity=True, wrong_B_F_squared_rejected=True, wrong_deleted_positive_residual_rejected=True, exact_commutator_relation=True, full_operator_residual_enclosure=True, noncommuting_operator_anchor_checked=True, one_sided_not_absolute_discriminator=True, W_exact=[[str(v) for v in row] for row in W.tolist()], positive_residual_exact=[[str(v) for v in row] for row in P.tolist()])


def run(epoch):
    started = time.monotonic()
    request = json.loads((ROOT/f'{PREFIX}{epoch}_request.json').read_bytes())
    for name, pin in request['input_sha256'].items():
        if native.sha(ROOT/name) != pin:
            raise ArithmeticError(('Input pin mismatch', name))
    baseline = json.loads((ROOT/'vacuum_connected_native_parent_budget3_20261011_receipt.json').read_bytes())
    iface = json.loads((ROOT/'central_neutral_vertex_interface_20261011.json').read_bytes())
    assert iface['controls']['actual_unequal_role_commutator'][1]['wrong_template_active_h_difference'] > 0
    boundary = json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
    _, _, _, exact = frozen.formation.derive_physical(boundary)
    matrices = {}
    for axis in range(3):
        M = np.array([[complex(frozen.formation.sp.sympify(z)) for z in row] for row in exact[axis]])
        for s, t in itertools.product(range(2), repeat=2):
            matrices[axis, s, t] = (-1)**(s+t)*M
    inventory, _ = frozen.native.inventory(8)
    edges = [next(e for e in inventory if e['axis'] == a and e['central_site'] == 0 and e['central_is_tail']) for a in (1, 2)]
    kernel = frozen.DynamicKernel(); star = frozen.Star(kernel, edges, matrices); central = car.Central(kernel, star)
    cap = F(1, 65536); infidelity = (cap/GAMMA)**2
    assert infidelity == F(274877906944, 2204801412260314559041)
    eta = frozen.sqrt_hi(2*infidelity)
    density = json.loads((ROOT/'three_cell_density_formation_typed3_20261011_interface.json').read_bytes())
    delta = max(F(r['delta_trace_norm_exact']) for r in density['errors'].values())
    beta = 2500*frozen.formation.gamma(8192)+frozen.formation.gamma(1 << 24)+128*frozen.formation.gamma(256)
    price = delta+beta+2*eta
    d = F(baseline['domains']['whole_operator_two_cell_floor_exact'])
    axes = []
    for edge in range(2):
        anchor0 = car.CANDIDATES[edge]
        grams = {name: [] for name in ('G', 'JC', 'CC', 'JT', 'Tgram')}
        errors = {name: [] for name in grams}
        accounting, memory = [], []
        for role, mult in ((0, 1), (1, 63)):
            combined = {name: [] for name in grams}
            combined_error = {name: F(0) for name in grams}
            for orientation in (-1, 1):
                first = [(car.dec(z), w, (0, 0, 0)) for z, w in star.root(edge, orientation, role)]
                flow, u = star.flow_u((1, edge, orientation, 1-edge, 0))
                C = central.action(first, flow, u)
                assert not any(any(m) for z, w, m in C)
                T = car.collect(C+car.scale(first, -anchor0))
                outer = 0 if edge == 0 else 2
                for name, left, right in (('G', first, first), ('JC', first, C), ('CC', C, C), ('JT', first, T), ('Tgram', T, T)):
                    terms, err, account = native.projected_product(kernel, left, right, outer, role, price, beta)
                    combined[name] += terms; combined_error[name] += err
                    accounting.append(dict(role=role, physical_count=mult, orientation=orientation, operator=name, accounting=account))
                next_action = central.action(T, flow, u)
                memory.append(dict(role=role, orientation=orientation, totalrho_cap_exact='128', next_action_exact=[dict(coefficient_exact=[str(x) for x in z], word=w, TOTALrho_right_powers=m) for z, w, m in next_action], nonzero_totalrho_terms=sum(any(m) for z, w, m in next_action)))
            for name in grams:
                grams[name].append(car.collect(combined[name])); errors[name].append(combined_error[name])
        S = [herm(g) for g in grams['JC']]
        for r in range(2):
            assert not car.collect(grams['JT'][r]+car.scale(grams['JC'][r], -1)+car.scale(grams['G'][r], anchor0))
            expected = car.collect(grams['CC'][r]+car.scale(S[r], -2*anchor0)+car.scale(grams['G'][r], anchor0**2))
            assert not car.collect(expected+car.scale(grams['Tgram'][r], -1))
        Gspec, Gpayload, Gerror = spectrum(grams['G'], errors['G'])
        G0 = native.expectation(kernel, grams['G'], price)
        G0 = (max(F(0), G0[0]-Gerror), G0[1]+Gerror)
        l, u = Gspec[0]-G0[1], Gspec[1]-G0[0]
        gc = max(abs(l), abs(u))
        # K^-1/2 Gc K^-1/2 has interval [min(l,0)/gamma,max(u,0)/gamma].
        lq, uq = min(l, F(0)), max(u, F(0))
        sweep = []
        for anchor in (F(64), F(80), F(96), anchor0, F(112), F(128), F(160), F(192), F(256)):
            B = combine(S, grams['G'], -anchor)
            Berr = [errors['JC'][r]+anchor*errors['G'][r] for r in range(2)]
            TT = [car.collect(grams['CC'][r]+car.scale(S[r], -2*anchor)+car.scale(grams['G'][r], anchor**2)) for r in range(2)]
            Terr = [errors['CC'][r]+2*anchor*errors['JC'][r]+anchor**2*errors['G'][r] for r in range(2)]
            if anchor == anchor0:
                B = [herm(t) for t in grams['JT']]
                Berr = errors['JT']
                TT, Terr = grams['Tgram'], errors['Tgram']
            Bspec, Bpayload, Berror = spectrum(B, Berr)
            Tspec, Tpayload, Terror = spectrum(TT, Terr)
            rows = []
            for oldrow in baseline['axes'][edge]['energies']:
                E = F(oldrow['E_exact']); A = anchor-E; de = d-E
                assert A > 0 and de > 0
                w0 = tuple(F(x) for x in oldrow['vacuum_w0']['exact'])
                # Joint -B+T*T/(d-E), collected before all512-mode canonical filling.
                joint = combine([car.scale(v, -1) for v in B], TT, 1/de)
                jerr = [Berr[r]+Terr[r]/de for r in range(2)]
                Jspec, Jpayload, Jerror = spectrum(joint, jerr)
                den = GAMMA*(A+GAMMA)**2
                mismatch = (G0[0]/A-w0[1], G0[1]/A-w0[0])
                sym_abs = gc/(GAMMA*(A+GAMMA))
                # Exact normalized-contraction anticommutator bound AB+BA>=-I/4.
                sym_upper = (uq+(uq-lq)/8)/(GAMMA*(A+GAMMA))
                sym_lower = (-lq+(uq-lq)/8)/(GAMMA*(A+GAMMA))
                positive_full = max(F(0), Tspec[1])/(de*den)
                joint_upper = max(F(0), Jspec[1])/den
                negative_B = max(F(0), Bspec[1])/den
                scalar_lower = G0[1]/(A*(A+GAMMA))
                upper = sym_upper+max(F(0), mismatch[1])/GAMMA+joint_upper
                lower = min(sym_abs, sym_lower)+scalar_lower+max(F(0), -mismatch[0])/GAMMA+negative_B
                # Additional W>=0 bound is valid but not a promoted vacuum/root claim.
                lower = min(lower, w0[1]/GAMMA)
                absolute = max(upper, lower)
                rows.append(dict(E_exact=str(E), vacuum_w0=oldrow['vacuum_w0'], joint_minus_B_plus_Tgram_over_floor=frozen.encode(Jspec), joint_operator_onebody_exact=Jpayload, joint_operator_error_exact=str(Jerror), components_exact=dict(centered_G_anticommutator_upper=str(sym_upper), centered_G_absolute=str(sym_abs), signed_scalar_counterterm_lower=str(scalar_lower), scalar_mismatch_interval=[str(x) for x in mismatch], signed_B_lower_price=str(negative_B), whole_positive_residual_price=str(positive_full), signed_joint_upper_price=str(joint_upper)), every_Q_one_sided_upper_relative_exact=str(upper), every_Q_one_sided_lower_relative_exact=str(lower), every_Q_absolute_relative_exact=str(absolute), every_Q_one_sided_upper_display=float(upper), every_Q_one_sided_lower_display=float(lower), every_Q_absolute_display=float(absolute), stability_upper_reaches_one=upper < 1, absolute_reaches_one=absolute < 1, historical_linear_triangle_cap_display=oldrow['whole_excited_connected_relative_cap_display']))
            sweep.append(dict(anchor_exact=str(anchor), signed_B_spectrum=frozen.encode(Bspec), signed_B_onebody_exact=Bpayload, TdagT_spectrum=frozen.encode(Tspec), TdagT_onebody_exact=Tpayload, signed_B_error_exact=str(Berror), TdagT_error_exact=str(Terror), energies=rows))
        best_upper = min(sweep, key=lambda r: F(r['energies'][1]['every_Q_one_sided_upper_relative_exact']))
        best_abs = min(sweep, key=lambda r: F(r['energies'][1]['every_Q_absolute_relative_exact']))
        axes.append(dict(axis=edges[edge]['axis'], G_spectrum=frozen.encode(Gspec), G_onebody_exact=Gpayload, G_ground_expectation=frozen.encode(G0), G_operator_error_exact=str(Gerror), signed_JC_roles_exact=[[dict(coefficient_exact=[str(x) for x in z], word=w, rho=m) for z, w, m in terms] for terms in S], projection_accounting=accounting, full_TOTALrho_memory=memory, scalar_anchor_sweep=sweep, best_E0_one_sided_anchor_exact=best_upper['anchor_exact'], best_E0_absolute_anchor_exact=best_abs['anchor_exact'], best_E0_one_sided_upper_display=best_upper['energies'][1]['every_Q_one_sided_upper_display'], best_E0_absolute_display=best_abs['energies'][1]['every_Q_absolute_display']))
    result = dict(kind='NEW_NATIVE_WHOLE_CANONICAL_SIGNED_POSITIVE_RESIDUAL_GALERKIN', epoch=epoch, governing_plan=request['governing_plan'], input_sha256=request['input_sha256'], source_sha256=native.sha(Path(__file__)), formation=dict(accepted_residual_exact=str(cap), included_formation_exact='1/262144', infidelity_exact=str(infidelity), eta_sqrt_transfer_exact=str(eta), density_price_exact=str(delta), beta_exact=str(beta), projected_operator_price_exact=str(price), Gram_formation_semantics='MAX(input_term_pair_L1,post_normal_order_pre_outer_collection_L1)', historical_density_machine_operation_premises_conditional=True, historical_multicell_source_price_preserved=baseline['formation']['charged_multicell_moment_price_exact']), domains=baseline['domains'], axes=axes, independent_noncommuting_controls=controls(), actual_whole_domain_anchor_sweep_exercised=True, positive_residual_paid_by_actual_whole_TdagT=True, no_old_main_eigensolver_root_checker_rerun=True, new_exact_LDL_bisection_not_eigensolver_rerun=True, frozen_reference_not_physicalPhi_ground=True, interacting_root_paid=False, higher_charge_intercell_paid=False, phase_paid=False, photon_paid=False, campaign_complete=False, elapsed_seconds=time.monotonic()-started)
    result['physical1_one_sided_verdict'] = 'PASS_SCOPED_STABILITY_ONLY' if all(a['best_E0_one_sided_upper_display'] < 1 for a in axes) else 'FAILED_SUFFICIENT_GUARD_NOT_NO_GO'
    result['physical1_absolute_verdict'] = 'PASS_SCOPED_ABSOLUTE_ONLY' if all(a['best_E0_absolute_display'] < 1 for a in axes) else 'FAILED_SUFFICIENT_GUARD_NOT_NO_GO'
    result['sources_inputs_unchanged'] = all(native.sha(ROOT/n) == p for n, p in request['input_sha256'].items())
    assert result['sources_inputs_unchanged']
    native.dump(ROOT/f'{PREFIX}{epoch}_receipt.json', result)
    print(json.dumps(dict(receipt=f'{PREFIX}{epoch}_receipt.json', axes=[dict(axis=a['axis'], best_one_sided=a['best_E0_one_sided_upper_display'], best_absolute=a['best_E0_absolute_display']) for a in axes], one_sided=result['physical1_one_sided_verdict'], absolute=result['physical1_absolute_verdict'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    run(parser.parse_args().epoch)
