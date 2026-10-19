"""Independent bottom-right exact spectral and whole-form consumer of frozen outputs."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
GAMMA = F(46955312929, 34359738368)
Z = tuple[F, F]


def mul(a: Z, b: Z) -> Z:
    return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]


def sub(a: Z, b: Z) -> Z:
    return a[0]-b[0], a[1]-b[1]


def divide_real(a: Z, value: F) -> Z:
    return a[0]/value, a[1]/value


def inertia(h: list[list[Z]], shift: F) -> int:
    m = [[sub(z, (shift if i == j else F(0), F(0)))
          for j, z in enumerate(row)] for i, row in enumerate(h)]
    negative = 0
    while m:
        # Reverse elimination differs from the producer's top-left LDL.
        pivots = [i for i in range(len(m)) if m[i][i][0] != 0]
        assert pivots, 'Selected endpoint has no nonzero diagonal pivot'
        p = pivots[-1]
        assert m[p][p][1] == 0
        value = m[p][p][0]
        negative += int(value < 0)
        keep = [i for i in range(len(m)) if i != p]
        m = [[sub(m[i][j], divide_real(mul(m[i][p], m[p][j]), value))
              for j in keep] for i in keep]
    return negative


def spectrum(roles: list[dict[str, Any]], error: F) -> tuple[F, F, int]:
    lower: list[F] = []; upper: list[F] = []; scalar = F(0); checked = 0
    assert sorted(r['physical_count'] for r in roles) == [1, 63]
    for role in roles:
        h = [[(F(z[0]), F(z[1])) for z in row] for row in role['onebody_exact']]
        assert all(h[i][j] == (h[j][i][0], -h[j][i][1]) for i in range(8) for j in range(8))
        cache: dict[F, int] = {}
        count = role['physical_count']; scalar += count*F(role['scalar_exact'])
        for i, interval in enumerate(role['eigenvalue_intervals']):
            lo, hi = map(F, interval['exact'])
            assert lo <= hi
            if lo not in cache: cache[lo] = inertia(h, lo)
            if hi not in cache: cache[hi] = inertia(h, hi)
            assert cache[lo] <= i < cache[hi]
            assert [cache[lo], cache[hi]] == role['exact_LDL_endpoint_inertias'][i]
            lower.extend([lo]*count); upper.extend([hi]*count); checked += 2
    assert len(lower) == len(upper) == 512
    return scalar+sum(sorted(lower)[:256], F(0))-error, scalar+sum(sorted(upper)[-256:], F(0))+error, checked


def capsule(name: str, prefix: str) -> int:
    cap = json.loads((ROOT/name).read_text())
    assert cap['returncode'] == 0 and cap['natural_exit'] and cap['sources_inputs_unchanged']
    pre = ROOT/(prefix+'_preexecution.zip'); exe = ROOT/(prefix+'_executed.zip')
    receipt = ROOT/(prefix+'_receipt.json')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(pre) == cap['preexecution_sha256'] and sha(exe) == cap['executed_sha256']
    assert sha(receipt) == cap['receipt_sha256']
    with zipfile.ZipFile(pre) as a, zipfile.ZipFile(exe) as b:
        for n in a.namelist():
            target = prefix+'_request.json' if n == 'request.json' else n
            assert a.read(n) == b.read(target)
        assert hashlib.sha256(b.read(receipt.name)).hexdigest() == sha(receipt)
        return len(a.namelist())


def main() -> None:
    sys.set_int_max_str_digits(0)
    p = argparse.ArgumentParser(); p.add_argument('--request', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path); args = p.parse_args()
    request = json.loads(args.request.read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h for n,h in request['pins'].items())
    members = capsule('positive_residual_actual1_20261011_capsule.json', 'positive_residual_actual1_20261011')
    members += capsule('positive_residual_global_actual1_20261011_capsule.json', 'positive_residual_global_actual1_20261011')
    value = json.loads((ROOT/'positive_residual_actual1_20261011_receipt.json').read_text())
    global_value = json.loads((ROOT/'positive_residual_global_actual1_20261011_receipt.json').read_text())
    d = F(value['domains']['whole_operator_two_cell_floor_exact'])
    checked = 0; rows = []
    for axis, global_axis in zip(value['axes'], global_value['axes'], strict=True):
        gl, gu, n = spectrum(axis['G_onebody_exact'], F(axis['G_operator_error_exact'])); checked += n
        assert [gl, gu] == list(map(F, axis['G_spectrum']['exact']))
        g0l, g0u = map(F, axis['G_ground_expectation']['exact'])
        l, u = gl-g0u, gu-g0l; lq, uq = min(l,F(0)), max(u,F(0)); gc=max(abs(l),abs(u))
        for sweep, gs in zip(axis['scalar_anchor_sweep'], global_axis['all_domain_scalar_anchor_sweep'], strict=True):
            a = F(sweep['anchor_exact']); assert str(a) == gs['anchor_exact']
            bl, bu, n = spectrum(sweep['signed_B_onebody_exact'], F(sweep['signed_B_error_exact'])); checked += n
            tl, tu, n = spectrum(sweep['TdagT_onebody_exact'], F(sweep['TdagT_error_exact'])); checked += n
            assert [bl, bu] == list(map(F,sweep['signed_B_spectrum']['exact']))
            assert [tl, tu] == list(map(F,sweep['TdagT_spectrum']['exact']))
            for row, grow in zip(sweep['energies'], gs['energies'], strict=True):
                E = F(row['E_exact']); A = a-E; de=d-E
                jl, ju, n = spectrum(row['joint_operator_onebody_exact'], F(row['joint_operator_error_exact'])); checked += n
                assert [jl, ju] == list(map(F,row['joint_minus_B_plus_Tgram_over_floor']['exact']))
                wl, wu = map(F,row['vacuum_w0']['exact']); ml,mu=g0l/A-wu,g0u/A-wl
                up = (uq+(uq-lq)/8)/(GAMMA*(A+GAMMA))+max(mu,F(0))/GAMMA+max(ju,F(0))/(GAMMA*(A+GAMMA)**2)
                low = min(gc, -lq+(uq-lq)/8)/(GAMMA*(A+GAMMA))+g0u/(A*(A+GAMMA))+max(-ml,F(0))/GAMMA+max(bu,F(0))/(GAMMA*(A+GAMMA)**2)
                low = min(low,wu/GAMMA)
                assert up == F(row['every_Q_one_sided_upper_relative_exact'])
                assert low == F(row['every_Q_one_sided_lower_relative_exact'])
                assert max(up,low) == F(row['every_Q_absolute_relative_exact'])
                assert max(tu,F(0))/(de*GAMMA*(A+GAMMA)**2) == F(row['components_exact']['whole_positive_residual_price'])
                gup=(uq+(uq-lq)/8)/(GAMMA*A)+max(mu,F(0))/GAMMA+max(ju,F(0))/(GAMMA*A*A)
                glow=min(wu/GAMMA,gc/(GAMMA*A)+g0u/(A*(A+GAMMA))+max(-ml,F(0))/GAMMA+max(bu,F(0))/(GAMMA*A*A))
                assert gup==F(grow['upper_relative_exact']) and glow==F(grow['lower_relative_exact'])
                assert max(gup,glow)==F(grow['absolute_relative_exact'])
                rows.append({'axis':axis['axis'],'anchor':str(a),'E':str(E),'Q_upper':str(up),'Q_lower':str(low),'Q_absolute':str(max(up,low)),'all_central_positive_weight_absolute':str(max(gup,glow))})
    assert len(rows)==54 and all(F(r['Q_absolute'])>1 for r in rows)
    result={'kind':'INDEPENDENT_EXACT_NATIVE_SIGNED_SPECTRA_FORM_CONSUMER_NOT_PHASE','archive_members_verified':members,'endpoint_inertia_checks':checked,'whole_domain_rows':rows,'every_Q_guard_failed':True,'whole_Pg_Qg_positive_weight_not_K_only':True,'historical_formation_conditional':True,'physical1_phase_or_photon':False,'campaign_complete':False}
    with args.output.open('x') as out: json.dump(result,out,indent=2);out.write('\n')
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h for n,h in request['pins'].items())
    print(json.dumps({'exact_endpoint_checks':checked,'whole_domain_rows':len(rows),'physical1_guard_failed':True}))


if __name__=='__main__': main()
