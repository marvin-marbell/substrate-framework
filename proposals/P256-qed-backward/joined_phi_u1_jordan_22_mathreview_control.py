"""Independent exact certificate audit; never imports/reruns the science producer."""
from __future__ import annotations

from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from math import isqrt
from pathlib import Path
import time
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_jordan_22_mathreview_'
TARGET = 'joined_phi_u1_jordan_22_typed_actual_20261010'
SOURCE_PIN = '3bbe20dc7dd9203683eef16aa80cbb0735be91e398486c9b5fe7e0832911877e'
RECEIPT_PIN = 'c6d62aeadd356a4b5d04be0d34ea7129bdf60b774704695827be7bbae8d36edb'


def sqrt_upper(q: F) -> F:
    scale = 1 << 160
    n = q.numerator * scale * scale // q.denominator
    r = isqrt(n)
    return F(r + (F(r * r, scale * scale) < q), scale)


def dot(a: list[F], b: list[F]) -> F:
    return sum((x * y for x, y in zip(a, b)), F(0))


def block_q(t: list[F], v: list[F]) -> F:
    s, h = dot(t, t), dot(v, v)
    return h*h + s*s + 4*dot(t, v)**2 + 2*s*h


def orbit(c: F, s: F) -> list[tuple[F, F]]:
    assert c*c+s*s == 1
    return [(c,s),(-s,c),(-c,-s),(s,-c)]


def exact_controls(profile: dict[str, Any]) -> dict[str, Any]:
    a = F(profile['onsite'])
    c0 = [F(z['c0']) for z in profile['paths']]
    ci = [F(z['ci']) for z in profile['paths']]
    t = [F(z,16) for z in (1,1,0,0)]
    vectors = [[F(z,16) for z in row] for row in ((2,0,1,0),(0,3,0,1))]
    # Diagonal Pauli fields: all spectra, moments and occupation sums are rational.
    base = [(F(3,5),F(4,5)),(F(5,13),F(12,13)),(F(8,17),F(15,17))]
    sheets = []
    for sheet in product((0,1), repeat=3):
        bands, moments = [], []
        for b in range(2):
            # theta2 = pi*n-theta1; p=(pi+theta)/4, hence p2=pi*(2+n)/4-p1.
            # Components n=0 can be rational circle reflected by pi/2; n=1 has sqrt2.
            # Exact rational orbit controls use independently relaxed characters; allangle
            # determinant inclusion is proven separately, not claimed by these controls.
            circles = base if b == 0 else [(s,-c) if n == 0 else (-c,-s) for (c,s),n in zip(base,sheet)]
            xs = [[a+sum(2*c0[j]*z[j][0] for j in range(3)),
                   *[-2*ci[j]*z[j][1] for j in range(3)]]
                  for z in product(*(orbit(c,s) for c,s in circles))]
            tb = t if b == 0 else [-z for z in t]
            v = vectors[b]
            values = [dot(x,tb)**2+dot(x,v)**2 for x in xs]
            mean = sum(values,F(0))/64
            variance = sum(((z-mean)**2 for z in values),F(0))/64
            k00 = t[0]**2+v[0]**2
            protected = 8*a*a*sum(z*z for z in c0)*k00*k00
            assert variance >= protected
            projection = sum((y*z[0][0] for y,z in zip(values,product(*(orbit(c,s) for c,s in circles)))),F(0))/64
            assert projection == 2*a*c0[0]*k00
            eigen = [dot(x,tb)+sign*dot(x,v) for x in xs for sign in (-1,1)]
            bands.append(eigen)
            moments.append({'variance':str(variance),'protected':str(protected),'frequency1_cos_projection':str(projection)})
        global128 = sum(sorted(bands[0]+bands[1])[-128:],F(0))
        fixed = sum((sum(sorted(b)[-64:],F(0)) for b in bands),F(0))
        halfabs = sum((abs(z) for z in bands[0]+bands[1]),F(0))/2
        assert sum(bands[0]+bands[1],F(0)) == 0 and global128 <= halfabs
        sheets.append({'independent_character_index':list(sheet),'global128':str(global128),'wrong_block64_sum':str(fixed),'deficit':str(global128-fixed),'half_absolute_trace':str(halfabs),'moments':moments})
    assert max(F(row['deficit']) for row in sheets) > 0
    q = sum((block_q(t,v) for v in vectors),F(0))
    direct = sum((sum(((tm+vm)*(tn+vn)+(tn+vn)*(tm+vm))**2 + ((tm-vm)*(tn-vn)+(tn-vn)*(tm-vm))**2 for tm,vm in zip(t,v) for tn,vn in zip(t,v)) for v in vectors),F(0))
    assert direct == 8*q
    stripped = sum((dot(v,v)**2 for v in vectors),F(0))
    assert q > stripped
    # Exact contrary case to replacing discrete N4 fourth moments by continuum ones.
    c,s = F(3,5),F(4,5)
    discrete = sum((cc*cc*ss*ss for cc,ss in orbit(c,s)),F(0))/4
    assert discrete == F(144,625) and discrete > F(1,8)
    alias_inner = sum(((cc*cc-ss*ss)*(2*cc*ss) for cc,ss in orbit(c,s)),F(0))/4
    assert alias_inner == -F(168,625)
    assert block_q([F(0)]*4,[F(0)]*4) == 0
    assert block_q([F(0)]*4,vectors[0]) == dot(vectors[0],vectors[0])**2
    return {'scope':'exact rational independent-character static neutral256 bands; relaxed determinant control, not native lattice or allangle experiment',
            't':list(map(str,t)),'v1':list(map(str,vectors[0])),'v2':list(map(str,vectors[1])),
            'quartic_direct':str(256*direct),'quartic_formula':str(2048*q),'wrong_scalar_drop_quartic':str(2048*stripped),
            'finite_exact_character_controls':sheets,
            'contrary_alias':{'cos_p':'3/5','sin_p':'4/5','mean_cos2_sin2':str(discrete),'wrong_continuum_value':'1/8','frequency2_cos_sin_inner':str(alias_inner),'trigger':'assuming shifted N4 frequency2 cos/sin orthogonality or continuous fourth moments; not an inequality used by reviewed proof'},
            'zero_and_s0_rank1':'exactly exercised; scalar-only has Q=s^2,K=tt^T,r=s; rank3 follows general polynomial and rank-H moment proof'}


def audit_cells(route: dict[str, Any]) -> dict[str, Any]:
    c = list(map(F,route['C_exact']))
    d3, spatial = F(route['D3_exact']), sum(z*z for z in c[1:])
    assert d3 == sum(z*z for z in sorted(c,reverse=True)[:3])
    d = F(route['defect']['lower'])
    variance = F(route['protected_variance_exact'])
    if route['route'] == 'moment':
        assert d == 0
    elif route['route'] == 'curvature':
        support = F(route['crude_support_exact'])
        assert d * 8 * support * sqrt_upper(support) <= variance
    else:
        fourth = sqrt_upper(sqrt_upper(d3))
        denom_upper = 2*fourth*(sqrt_upper(F(route['support_exact']))+fourth)**2
        assert d*denom_upper <= variance
    records = route['cell_bounds']
    assert len(records) == route['closed_cells'] == 16384
    for index, cell in enumerate(records):
        lo,hi,p = F(cell['lo']),F(cell['hi']),F(cell['pressure_upper'])
        assert cell['index'] == index and lo == F(index,16384) and hi == F(index+1,16384)
        z = p + d*lo*lo
        assert z >= 0
        q = z*z
        # Direct rational squared inequalities: no regenerated envelope or angle sweep.
        assert q*q >= d3 or (q >= c[0]*hi and (q-c[0]*hi)**2 >= spatial*(1-lo*lo))
    alpha = F(route['alpha_upper'])
    assert alpha == max(F(z['pressure_upper']) for z in records)
    low,high = F(route['lower']['lower']),F(route['lower']['upper'])
    assert (-low/3)**3 >= 256*alpha**4 >= (-high/3)**3
    return {'route':route['route'],'closed_cells_checked':len(records),'all_rational_upper_inequalities':True,'radial_endpoints_cubed_exactly':True,'defect_lower_outward_checked_independently_at_bits':160}


def run() -> dict[str, Any]:
    start = time.monotonic()
    names = [Path(__file__).name,'joined_phi_u1_jordan_22_consumer.py',TARGET+'.json',TARGET+'.capsule.json',TARGET+'.input_source.zip',TARGET+'.execution_source.zip',
             'joined_phi_u1_matrix_jordan_flat_actual_20261010_receipt.json','joined_phi_u1_jordan_22_final_checker_20261010.json']
    raw = {name:(ROOT/name).read_bytes() for name in names}
    pins = {name:sha256(value).hexdigest() for name,value in raw.items()}
    assert pins['joined_phi_u1_jordan_22_consumer.py'] == SOURCE_PIN and pins[TARGET+'.json'] == RECEIPT_PIN
    with zipfile.ZipFile(ROOT/(PREFIX+'20261010.epoch.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in raw.items():archive.writestr(name,value)
    receipt = json.loads(raw[TARGET+'.json'])
    caps = json.loads(raw[TARGET+'.capsule.json'])
    assert caps['exit_code'] == 0 and caps['natural_completed_exit'] and caps['receipt_sha256'] == RECEIPT_PIN
    assert caps['source_input_sha256_before'] == caps['source_input_sha256_after']
    assert caps['archive_sha256'] == pins[TARGET+'.execution_source.zip']
    assert receipt['immutable_archive_sha256'] == pins[TARGET+'.input_source.zip']
    checked = []
    for suffix,mapping in (('.execution_source.zip',caps['source_input_sha256_before']),('.input_source.zip',receipt['source_input_sha256_before'])):
        with zipfile.ZipFile(ROOT/(TARGET+suffix)) as archive:
            for name,pin in mapping.items():
                assert sha256(archive.read(name)).hexdigest() == pin
                assert sha256((ROOT/name).read_bytes()).hexdigest() == pin
        checked.append({'archive':TARGET+suffix,'member_and_live_sha256_count':len(mapping)})
    prior = json.loads(raw['joined_phi_u1_matrix_jordan_flat_actual_20261010_receipt.json'])
    profile = prior['primitive_profiles'][0]
    a,c0,ci = F(profile['onsite']),F(profile['paths'][0]['c0']),F(profile['paths'][0]['ci'])
    aa,b = 2*abs(c0),4*ci*ci
    assert b > 3*aa*aa
    support = a*a+3*b+3*(a*aa)**2/(b-3*aa*aa)
    for route in receipt['routes']:
        assert F(route['support_exact']) == support
        assert F(route['protected_variance_exact']) == 24*a*a*c0*c0
        assert list(map(F,route['C_exact'])) == [a*a+6*c0*c0,*([2*ci*ci]*3)]
    cells = [audit_cells(route) for route in receipt['routes']]
    controls = exact_controls(profile)
    lower = F(receipt['sector_bracket']['lower']['lower'])
    central = F(prior['exact_central_convex_residual']['upper'])
    assert F(receipt['gap_to_central_trial']['lower']) == central-lower > 0
    assert F(receipt['lower_improvement']['lower']) == lower-F(receipt['routes'][0]['lower']['lower']) > 0
    assert not receipt['excluded']
    assert [z['sheet'] for z in receipt['all8_continuous_angle_controls']] == [list(z) for z in product((0,1),repeat=3)]
    # Character-sum selection rule: degrees1/2 differences cannot be nonzero multiples4.
    assert all(sum(complex(0,1)**(d*k) for k in range(4)) == 0 for d in (-3,-2,-1,1,2,3))
    unchanged = all((ROOT/name).read_bytes() == value for name,value in raw.items())
    assert unchanged
    result = {'kind':'INDEPENDENT_EXACT_STATIC_JORDAN22_PROOF_REVIEW_CONTROL','pins':pins,'archive_bindings':checked,
              'producer_natural_exit':'archived evidence inspected, not rerun','producer_elapsed_seconds':caps['elapsed_seconds'],
              'certificate_audit':cells,'independent_controls':controls,'support_and_units':'neutral onebody coefficients; 256=sumx 1/(16a), block trace anticommutator=8Q, hence2048Q; energy/Lambda',
              'allangle_proof':'root-of-unity selection rule protects frequency1 against every degree2 term even aliased2; no angle grid proves continuum',
              'sector_decision':'VALID lower bracket tightening; 2+2 remains eligible; no dominance/fullground/pole admission',
              'elapsed_seconds':time.monotonic()-start,'source_inputs_unchanged':unchanged}
    (ROOT/(PREFIX+'control_actual_20261010.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    report = run()
    print(json.dumps({'kind':report['kind'],'elapsed_seconds':report['elapsed_seconds'],'certificate_routes':report['certificate_audit'],'source_inputs_unchanged':report['source_inputs_unchanged']},indent=2))
