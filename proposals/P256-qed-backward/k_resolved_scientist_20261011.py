"""NEW whole-canonical joint K pencil and collective-rho positive-row certificates.
Immutable contracted data only; no historical entrypoint or numerical eigensolver.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
import sympy as sp
import positive_residual_scientist_20261011 as pure

ROOT = Path(__file__).resolve().parent
PREFIX = 'k_resolved_'
native, car, frozen = pure.native, pure.car, pure.frozen
GAMMA = pure.GAMMA
SCALE = 1024
SLOPES = (F(0), F(8), F(16), F(32), F(64), F(128), F(256), F(512), F(1024), F(2048), F(4096))


def decode(payload):
    return [(F(p['scalar_exact']), [[(F(z[0]), F(z[1])) for z in row] for row in p['onebody_exact']]) for p in payload]


def free_matrices():
    rows = json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())['profiles']
    out = []
    for row in rows:
        m, c, d = [F(row[k]) for k in ('m_exact', 'c0_exact', 'd_exact')]
        h = [[car.ZERO for _ in range(8)] for _ in range(8)]
        for s in range(2):
            for p in range(4):
                h[4*s+p][4*s+p] = (m*(-1 if p % 2 == 0 else 1), F(0))
            for p in range(2):
                h[4*s+p][4*s+p+2] = (c*(1 if p == 0 else -1), F(0))
                h[4*s+p+2][4*s+p] = h[4*s+p][4*s+p+2]
                h[4*s+p][4*s+3-p] = (F(0), d)
                h[4*s+3-p][4*s+p] = (F(0), -d)
        out.append(h)
    return out


def exact_brackets(h):
    """Rational LDL only; reuse identical spin blocks after exact equality."""
    assert all(h[i][j] == (h[j][i][0], -h[j][i][1]) for i in range(8) for j in range(8))
    repeated = all(h[i][j] == h[i+4][j+4] for i in range(4) for j in range(4)) and all(h[i][j] == car.ZERO and h[j][i] == car.ZERO for i in range(4) for j in range(4, 8))
    a = [row[:4] for row in h[:4]] if repeated else h
    n = len(a)
    rad = [sum((car.norm(a[i][j]) for j in range(n) if i != j), F(0)) for i in range(n)]
    bottom = min(a[i][i][0]-rad[i] for i in range(n))-1
    top = max(a[i][i][0]+rad[i] for i in range(n))+1
    brackets, counts = [], []
    for rank in range(n):
        if counts and counts[-1][1] >= rank+1:
            brackets.append(brackets[-1]); counts.append(counts[-1]); continue
        lo, hi, lc, hc = bottom, top, 0, n
        while hi-lo > F(1, 256):
            for divisor in (2, 3, 4, 5, 7):
                mid = ((divisor-1)*lo+hi)/divisor
                try:
                    count = native.inertia(a, mid)
                    break
                except ArithmeticError:
                    continue
            else:
                raise ArithmeticError('All exact LDL pivots vanished')
            if count <= rank:
                lo, lc = mid, count
            else:
                hi, hc = mid, count
        brackets.append((lo, hi)); counts.append((lc, hc))
    return brackets*(2 if repeated else 1), dict(exact_identical_spin_blocks=repeated, intervals=[frozen.encode(b) for b in brackets], endpoint_inertias=counts)


def pencil(operators, hfree, error, slope, shift, eg_upper) -> dict[str, Any]:
    """D <= slope*K+v by free joint filling and an ACTUAL collective square."""
    levels, scalar, payload = [], F(0), []
    for role, multiplicity in ((0, 1), (1, 63)):
        c, d = operators[role]
        h = [[car.ca(d[i][j], car.cs(hfree[role][i][j], -slope)) for j in range(8)] for i in range(8)]
        for i in range(8):
            # -64u rho^2 = -64u(rho-z)^2 -128u z rho +64u z^2.
            h[i][i] = car.ca(h[i][i], (-128*slope*shift*F(1 if i % 4 < 2 else -1, 2), F(0)))
        brackets, certificate = exact_brackets(h)
        levels.extend([b[1] for b in brackets]*multiplicity)
        scalar += c*multiplicity
        payload.append(dict(role=role, count=multiplicity, pencil_onebody_exact=[[[str(x) for x in z] for z in row] for row in h], certificate=certificate))
    v = scalar+sum(sorted(levels)[-256:])+slope*eg_upper+64*slope*shift**2+error
    return dict(design='JOINT_FREE_PENCIL_WITH_RETAINED_COLLECTIVE_SQUARE', slope_exact=str(slope), rho_shift_exact=str(shift), intercept_exact=str(v), intercept_display=float(v), coefficient_error_exact=str(error), exact_positive_square_coefficient=str(64*slope), all512_global256_fill=True, roles=payload)


def ceil_fraction(x):
    return -((-x.numerator)//x.denominator)


def local_row_costs(d, hfree, slope, weights):
    """Whole local 8-mode Fock, including spin mixing and arbitrary labelled N."""
    weights = tuple(weights)*2
    a = [[car.ca(d[i][j], car.cs(hfree[i][j], -slope)) for j in range(8)] for i in range(8)]
    costs = {}
    controls = []
    for mask in range(256):
        occupied = [i for i in range(8) if mask & (1 << i)]
        diag = sum((a[i][i][0] for i in occupied), F(0))
        # D_weight^-1 A D_weight row: occupied j -> empty i.
        rad = sum((car.norm(a[i][j])*weights[i]/weights[j] for j in occupied for i in range(8) if i not in occupied), F(0))
        exact = diag+rad
        nplus = sum(i % 4 < 2 for i in occupied)
        nminus = len(occupied)-nplus
        cost = ceil_fraction(SCALE*exact)
        costs[nplus, nminus] = max(costs.get((nplus, nminus), cost), cost)
        controls.append(dict(occupation_mask=mask, nplus=nplus, nminus=nminus, cost_exact=str(exact), outward_integer=cost))
    return costs, controls


def collective_row(operators, hfree, error, slope, eg_upper, weights) -> dict[str, Any]:
    """Exact integer max-plus containing ALL256 canonical rows and total rho."""
    impossible = -(1 << 60)
    dp = np.full((257, 257), impossible, dtype=np.int64)
    dp[0, 0] = 0
    payload = []
    slots = 0
    scalar = F(0)
    for role, count in ((0, 1), (1, 63)):
        c, d = operators[role]
        costs, control = local_row_costs(d, hfree[role], slope, weights)
        bound = max(abs(v) for v in costs.values())
        if bound*64 >= (1 << 58):
            raise OverflowError('Exact integer DP cannot silently wrap')
        scalar += count*c
        payload.append(dict(role=role, count=count, local_modes=8, local_whole_Fock_rows=control))
        for _ in range(count):
            slots += 1
            target = np.full_like(dp, impossible)
            oldmax = min(256, 4*(slots-1))
            for (p, m), cost in costs.items():
                pm = min(oldmax, 256-p)+1
                mm = min(oldmax, 256-m)+1
                block = dp[:pm, :mm]
                value = np.where(block == impossible, impossible, block+cost)
                np.maximum(target[p:p+pm, m:m+mm], value, out=target[p:p+pm, m:m+mm])
            dp = target
    profile = []
    best = None
    bestplus = None
    for nplus in range(257):
        value = int(dp[nplus, 256-nplus])
        if value == impossible:
            continue
        rho = F(2*nplus-256, 2)
        row = scalar+F(value, SCALE)-64*slope*rho**2+slope*eg_upper+error
        profile.append(dict(Nplus=nplus, Nminus=256-nplus, TOTALrho_exact=str(rho), total_row_upper_exact=str(row)))
        if best is None or row > best:
            best, bestplus = row, nplus
    assert best is not None
    return dict(design='ACTUAL_COLLECTIVE_RHO_WEIGHTED_POSITIVE_ROW', slope_exact=str(slope), weights_exact=[str(w) for w in weights]*2, intercept_exact=str(best), intercept_display=float(best), worst_Nplus=bestplus, rounding_quantum_exact=f'1/{SCALE}', max_total_row_rounding_exact=str(F(64, SCALE)), no_independent_rho_fields=True, all256_canonical_rows_contained=True, roles=payload, whole_totalrho_profile=profile)


def scalar_price(slope, intercept, A, weight, counter_g=F(0), mismatch=F(0)):
    """Outward interval covering scalar sup, not a sampled eigenvalue maximum.
    weight='Q': divide by k; weight='S': divide by k+gamma.
    Positive terms decrease; negative terms relaxed to zero only at this step.
    """
    k0 = GAMMA if weight == 'Q' else F(0)
    positive_cuts = [GAMMA*(F(17, 16)**j) for j in range(230)]
    cuts = ([F(0)] if weight == 'S' else [])+positive_cuts
    # All positive ratios have known tails; finite intervals use rational bounds.
    bounds = []
    for lo, hi in zip(cuts, cuts[1:]):
        wlo = lo if weight == 'Q' else lo+GAMMA
        whi = hi if weight == 'Q' else hi+GAMMA
        numerator = slope*hi+intercept
        first = numerator/(wlo*(lo+A)**2) if numerator >= 0 else numerator/(whi*(hi+A)**2)
        scalar = -counter_g*lo/(A*(hi+A)*whi)
        mass = mismatch/wlo if mismatch >= 0 else mismatch/whi
        bounds.append(first+scalar+mass)
    # Tail safely relaxes negative terms and uses k/(k+gamma)<=1.
    last = cuts[-1]
    tail = slope/(last+A)**2+max(intercept, F(0))/(last*(last+A)**2)+max(mismatch, F(0))/last
    bounds.append(tail)
    return max(bounds)


def controls():
    K = sp.Matrix([[2, 1], [1, 3]])
    D = sp.Matrix([[7, -2], [-2, 1]])
    H = K+4*sp.eye(2)
    Fm = H.inv()
    # Exact joint pencil remains a true order through F congruence without [D,K]=0.
    u, v = sp.Rational(2), sp.Rational(7)
    slack = u*K+v*sp.eye(2)-D
    assert slack[0, 0] > 0 and slack.det() > 0 and K*D != D*K
    congruence = Fm*(u*K+v*sp.eye(2))*Fm-Fm*D*Fm
    assert congruence[0, 0] > 0 and congruence.det() > 0
    # Tempting independent extrema c=lambda_max(D)/lambda_max(K) is NOT domination.
    Kwrong = sp.diag(1, 10)
    Dwrong = sp.diag(9, 1)
    assert (sp.Rational(9, 10)*Kwrong-Dwrong)[0, 0] < 0
    U = sp.Matrix([[1, 0], [0, 0]])
    V = sp.Matrix([[sp.Rational(1, 2)]*2]*2)
    assert ((U*V+V*U)/2).det() < 0
    X = sp.Matrix([[8, 1], [1, 11]])
    J = sp.Matrix([[1, 2], [-1, 1]])
    T = X*J-J*H
    B = (J.T*T+T.T*J)/2
    G = J.T*J
    W = J.T*X.inv()*J
    residual = Fm*T.T*X.inv()*T*Fm
    variational = (G*Fm+Fm*G)/2-Fm*B*Fm
    assert W-variational == residual and residual.trace() > 0
    assert ((W-variational)[0, 0] > 0)
    assert B*Fm**2 != Fm*B*Fm
    # Collective square is not sum of independent local squares, even canonically.
    rho1, rho2 = sp.diag(1, -1), sp.diag(-1, 1)
    assert (rho1+rho2)**2 == sp.zeros(2) and rho1**2+rho2**2 != sp.zeros(2)
    assert (U*V-V*U) != sp.zeros(2)
    # Full Pg/Qg K-only relative order must fail if a nonzero offdiagonal remains.
    C = sp.Matrix([[0, 1], [1, 0]])
    assert (10*sp.diag(0, 2)-C).det() < 0
    return dict(exact_noncommuting_joint_congruence=True, independent_extrema_ratio_rejected=True, positive_product_order_rejected=True, omitted_positive_residual_rejected=True, wrong_B_F_squared_rejected=True, independent_collective_square_rejected=True, full_PgQg_pure_K_claim_rejected=True, noncommuting_K_D=True, no_native_theorem_from_control=True)


def run(epoch):
    began = time.monotonic()
    request = json.loads((ROOT/f'{PREFIX}{epoch}_request.json').read_bytes())
    assert all(native.sha(ROOT/name) == pin for name, pin in request['runtime_sha256'].items())
    old = json.loads((ROOT/'positive_residual_actual1_20261011_receipt.json').read_bytes())
    energy = json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_bytes())
    eg = F(energy['ground_energy_interval_exact'][1])
    hfree = free_matrices()
    assert old['domains']['physical_hopping'] == '1'
    allaxes = []
    for axis in old['axes']:
        # Original native anchor only: this is NOT a further scalar-anchor sweep.
        original = axis['scalar_anchor_sweep'][3]
        assert original['anchor_exact'] == str(car.CANDIDATES[axis['axis']-1])
        outputs = []
        for row in original['energies']:
            E, anchor = F(row['E_exact']), F(original['anchor_exact'])
            A = anchor-E
            joint = decode(row['joint_operator_onebody_exact'])
            error = F(row['joint_operator_error_exact'])
            candidates = [pencil(joint, hfree, error, slope, F(0), eg) for slope in SLOPES]
            # A genuinely different exact containing-space resolution retains rho^2
            # per global canonical row, rather than just a nonnegative-square slack.
            row_candidates = []
            if E == 0:
                for slope in (F(32), F(128), F(512)):
                    for weights in ((F(1),)*4, (F(4), F(1), F(4), F(1))):
                        row_candidates.append(collective_row(joint, hfree, error, slope, eg, weights))
            candidates.extend(row_candidates)
            g0lo, g0hi = [F(x) for x in axis['G_ground_expectation']['exact']]
            wlo, whi = [F(x) for x in row['vacuum_w0']['exact']]
            mismatch = g0hi/A-wlo
            components = row['components_exact']
            bestQ = min(candidates, key=lambda p: scalar_price(F(p['slope_exact']), F(p['intercept_exact']), A, 'Q', g0lo, mismatch))
            bestS = min(candidates, key=lambda p: scalar_price(F(p['slope_exact']), F(p['intercept_exact']), A, 'S', g0lo, mismatch))
            qjoint = scalar_price(F(bestQ['slope_exact']), F(bestQ['intercept_exact']), A, 'Q', g0lo, mismatch)
            sjoint = scalar_price(F(bestS['slope_exact']), F(bestS['intercept_exact']), A, 'S', g0lo, mismatch)
            gspec = [F(x) for x in axis['G_spectrum']['exact']]
            lq, uq = min(gspec[0]-g0hi, F(0)), max(gspec[1]-g0lo, F(0))
            centeredQ = (uq+(uq-lq)/8)/(GAMMA*(A+GAMMA))
            centeredS = (uq+(uq-lq)/8)/(GAMMA*A)
            upperQ = centeredQ+qjoint
            upperS = centeredS+sjoint
            lowerQ = F(row['every_Q_one_sided_lower_relative_exact'])
            # SAME complete lower proof with positive allcentral weight.
            centered_abs_S = max(abs(gspec[0]-g0hi), abs(gspec[1]-g0lo))/(GAMMA*A)
            bupper = F(original['signed_B_spectrum']['exact'][1])
            lowerS = min(centered_abs_S+g0hi/(A*(A+GAMMA))+max(F(0), whi-g0lo/A)/GAMMA+max(F(0), bupper)/(GAMMA*A*A), whi/GAMMA)
            for p in candidates:
                p['every_Q_joint_plus_scalar_price_exact'] = str(scalar_price(F(p['slope_exact']), F(p['intercept_exact']), A, 'Q', g0lo, mismatch))
                p['every_Q_joint_price_without_scalar_exact'] = str(scalar_price(F(p['slope_exact']), F(p['intercept_exact']), A, 'Q'))
            outputs.append(dict(E_exact=str(E), anchor_exact=str(anchor), best_Q_design=bestQ['design'], best_Q_slope_exact=bestQ['slope_exact'], best_Q_intercept_exact=bestQ['intercept_exact'], best_S_design=bestS['design'], best_S_slope_exact=bestS['slope_exact'], K_resolved_joint_and_negative_scalar_Q_exact=str(qjoint), K_resolved_joint_and_negative_scalar_S_exact=str(sjoint), centered_G_upper_Q_exact=str(centeredQ), centered_G_upper_S_exact=str(centeredS), every_Q_upper_exact=str(upperQ), every_Q_upper_display=float(upperQ), every_Q_lower_exact=str(lowerQ), every_Q_absolute_exact=str(max(upperQ, lowerQ)), every_Q_absolute_display=float(max(upperQ, lowerQ)), allcentral_S_upper_exact=str(upperS), allcentral_S_upper_display=float(upperS), allcentral_S_lower_exact=str(lowerS), allcentral_S_absolute_exact=str(max(upperS, lowerS)), allcentral_S_absolute_display=float(max(upperS, lowerS)), old_upper_display=row['every_Q_one_sided_upper_display'], old_joint_price_exact=components['signed_joint_upper_price'], whole_positive_residual_old_price_exact=components['whole_positive_residual_price'], candidates=candidates))
        allaxes.append(dict(axis=axis['axis'], role_counts=[1, 63], energies=outputs))
    result = dict(kind='NEW_JOINT_K_ENERGY_WHOLE_CANONICAL_CERTIFICATE_NOT_PHYSICAL_ADMISSION', epoch=epoch, governing_plan=request['governing_plan'], source_sha256=native.sha(Path(__file__)), input_sha256=request['input_sha256'], formation=old['formation'], domains=old['domains'], actual_Hcentral='sum_role dGamma(h_role) +64 TOTALrho_all^2', ground_energy_interval_exact=energy['ground_energy_interval_exact'], controls=controls(), axes=allaxes, conditional_historical_ground_density_operations=True, no_historical_entrypoint_eigensolver_root_checker=True, complete_canonical256_of512=True, full_PgQg_allcentral_positive_weight='S=K+gamma I', physical1_phase_root_admitted=False, campaign_complete=False, elapsed_seconds=time.monotonic()-began)
    result['physical1_Q_one_sided'] = all(F(a['energies'][1]['every_Q_upper_exact']) < 1 for a in allaxes)
    result['physical1_Q_absolute'] = all(F(a['energies'][1]['every_Q_absolute_exact']) < 1 for a in allaxes)
    result['sources_inputs_unchanged'] = all(native.sha(ROOT/name) == pin for name, pin in request['runtime_sha256'].items())
    assert result['sources_inputs_unchanged']
    native.dump(ROOT/f'{PREFIX}{epoch}_receipt.json', result)
    print(json.dumps(dict(receipt=f'{PREFIX}{epoch}_receipt.json', axes=[dict(axis=a['axis'], Q_upper=a['energies'][1]['every_Q_upper_display'], Q_absolute=a['energies'][1]['every_Q_absolute_display'], S_upper=a['energies'][1]['allcentral_S_upper_display'], design=a['energies'][1]['best_Q_design'], slope=a['energies'][1]['best_Q_slope_exact']) for a in allaxes], physical1_Q_one_sided=result['physical1_Q_one_sided'], physical1_Q_absolute=result['physical1_Q_absolute'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True)
    run(parser.parse_args().epoch)
