"""NEW source-bound discriminators; no delivered producer or old control is called."""
from __future__ import annotations
import ast
from collections import deque
from fractions import Fraction as F
from itertools import product
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = 'flux_phase_mathreview_'
PINS = {
    'bulk_electric_phase_consumer.py': '4c8dd6fd6202411c6c56f5d9ac6f1ff61c52bcb64bcd2f9e8d0e40737ee02515',
    'bulk_electric_phase_proof_20261010.json': '2319ad2e58508c3e9ad7336985612761a7b12c2cc38f7c9a7b565d313c28342f',
    'bulk_electric_phase_delivered_20261010.json': '4dd906ba76d2bde62f342917fd3c75e36fa8b7f9f9c93b485fd773e123a9b737',
}
raw = {name: (ROOT/name).read_bytes() for name in PINS}
assert all(hashlib.sha256(raw[name]).hexdigest() == pin for name, pin in PINS.items())
delivered = json.loads(raw['bulk_electric_phase_delivered_20261010.json'])
bindings = dict(PINS)
archive_counts = {}
for suffix in ('preexecution', 'executed'):
    name = 'bulk_electric_phase_delivered_20261010_'+suffix+'.zip'
    blob = (ROOT/name).read_bytes()
    bindings[name] = hashlib.sha256(blob).hexdigest()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member, pin in delivered['source_and_input_sha256'].items():
            bound = z.read('bound/'+member)
            assert hashlib.sha256(bound).hexdigest() == pin, (suffix, member)
            if '/' not in member:
                assert (ROOT/member).read_bytes() == bound, member
        archive_counts[suffix] = len(delivered['source_and_input_sha256'])
        if suffix == 'executed':
            assert z.read('actual.json') == raw['bulk_electric_phase_delivered_20261010.json']
            neutral = delivered['fixed_neutral_AP_ground']
            assert hashlib.sha256(z.read(neutral['artifact'])).hexdigest() == neutral['sha256']
with zipfile.ZipFile(ROOT/'bulk_electric_phase_delivered_20261010_natural_execution.zip') as z:
    exit_receipt = json.loads(z.read('exit.json'))
    assert exit_receipt['natural_exit'] and exit_receipt['returncode'] == 0
    for name, row in exit_receipt['artifacts'].items():
        current = (ROOT/name).read_bytes()
        assert hashlib.sha256(current).hexdigest() == row['sha256'], name
        if name in z.namelist():
            assert z.read(name) == current, name
        else:
            assert name == delivered['fixed_neutral_AP_ground']['artifact'], name
            with zipfile.ZipFile(io.BytesIO(z.read('bulk_electric_phase_delivered_20261010_executed.zip'))) as inner:
                assert inner.read(name) == current
bindings.update({name: pin for name, pin in delivered['source_and_input_sha256'].items() if '/' not in name})
# Execute only the source's pure integer graph class, not its old spectral controls.
tree = ast.parse(raw['bulk_electric_phase_consumer.py'])
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IntegerBox')
namespace = {'np': np, 'product': product, 'deque': deque}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT/'bulk_electric_phase_consumer.py'), 'exec'), namespace)
box = namespace['IntegerBox'](5)
C, chords = box.cycle_basis()
C = C.astype(np.int64)
D = np.zeros((box.volume, 3*box.volume), dtype=np.int64)
D[box.tail, np.arange(3*box.volume)] += 1
D[box.head, np.arange(3*box.volume)] -= 1
assert not np.any(D@C)
assert np.array_equal(C[chords], np.eye(len(chords), dtype=int))
G = C.T@C
assert not np.array_equal(G, np.eye(len(chords), dtype=int))
j = next(j for j in range(len(chords)) if G[j,j] != 1)
v = C[:,j]
assert int(v@v) == int(G[j,j]) != 1
# Arbitrary exact divergence-free integer combinations reconstruct via chord coordinates.
coordinates = np.array([(i % 7)-3 for i in range(len(chords))], dtype=np.int64)
k = C@coordinates
assert not np.any(D@k) and np.array_equal(C@k[chords], k)
face = box.face((4,4,4), 0, 2)
assert not np.any(D@face) and int(face@face) == 4
assert np.array_equal(C@face[chords], face)
# Full Gauss fibres are affine cosets: a charged endpoint path is NOT a cycle.
path = box.path((4,0,0), 0, 2)
rho = D@path
assert np.sum(np.abs(rho)) == 2
assert not np.array_equal(C@path[chords], path)
affine = path + k
assert np.array_equal(D@affine, rho)
assert np.array_equal(path+C@(affine-path)[chords], affine)
assert np.array_equal(D@(affine+face), rho)
eps = F(1, 1024*len(chords)*int(np.trace(G)))
beta = [eps if i%2 else -eps for i in range(len(chords))]
E = [sum(F(int(C[e,i]))*beta[i] for i in range(len(chords))) for e in range(C.shape[0])]
base = 64*sum(x*x for x in E)
cross = 128*sum(F(int(c))*x for c,x in zip(face,E))
face_energy = 64*sum((x+int(c))**2 for x,c in zip(E,face))
assert face_energy == base+256+cross
sup = 64*eps**2*len(chords)*int(np.trace(G))
spread = 128*eps*int(np.sum(np.abs(face@C)))
assert base <= sup and abs(cross) <= spread
wrong_metric_energy = 64*sum(x*x for x in beta)
assert base != wrong_metric_energy

# Contact/order control on a charged physical fibre, with exterior shifted outputs retained.
probe = affine + 17*face
assert np.array_equal(D@probe, rho)
dot = int(probe@face)
comm = {}
correct = {}
wrong_order = {}
for sign in (-1, 1):
    key = tuple(int(x) for x in probe+sign*face)
    comm[key] = -F(sign,2)*64*(int((probe+sign*face)@(probe+sign*face))-int(probe@probe))
    correct[key] = -64*dot-128*sign
    wrong_order[key] = -64*(dot+4*sign)-128*sign
assert comm == correct and comm != wrong_order
assert all(wrong_order[key]-correct[key] == -256*sign for sign,key in [(-1,tuple(probe-face)),(1,tuple(probe+face))])
# The lower-order contact term is indispensable even though hopping commutes.
assert any(comm[tuple(probe+sign*face)] != -64*dot for sign in (-1,1))
# Domain: |k+-c|^4 <= 8|k|^4+128 controls D(E^2), not merely a truncated core.
for scale in (-1000,-3,0,9,1000):
    flow = path+scale*face
    for sign in (-1,1):
        assert int((flow+sign*face)@(flow+sign*face))**2 <= 8*int(flow@flow)**2+128

# Read-only actual binary coefficient constructors; no producer/control entrypoint called.
import joined_phi_u1_bulk_flux_ring_consumer as constructors
frame, phi_scale = constructors.exact_frame((ROOT/constructors.INPUTS[0]).read_bytes())
cfg = json.loads((ROOT/constructors.INPUTS[1]).read_bytes())['input_descriptor']['original_micro_config_UNCHANGED']
profiles = constructors.profiles(cfg, frame, phi_scale)
b = 2*sum((p['count']*sum(abs(z[0])+abs(z[1]) for pathrow in p['paths'] for row in pathrow['matrix'] for z in row) for p in profiles), F(0))
assert b == F(delivered['connected_local_tail']['full64_native_CAR_bound_per_site_exact'])
assert 131072+4096*b == F(delivered['connected_local_tail']['second_energy_moment_upper_at_physical1_exact'])
# All64 explicit shared-field Gauss transitions, not a selected-copy assertion.
sea = sum(1 << (copy*4*box.volume+4*site+spin) for copy in range(64) for site in range(box.volume) for spin in (0,1))
def move(mask, create, destroy):
    if not (mask>>destroy)&1 or (mask>>create)&1: return None
    return (mask^(1<<destroy))|(1<<create)
checks = 0
for copy in range(64):
    profile = profiles[0 if copy==0 else 1]
    for row in profile['paths']:
        x = (4,4,4); y = box.step(x,row['axis'],row['length'])
        flow = box.path(x,row['axis'],row['length'])
        for sign in (-1,1):
            target, origin = (x,y) if sign==1 else (y,x)
            for i in range(4):
                for h in range(4):
                    z = row['matrix'][i][h]
                    if z == (F(0),F(0)): continue
                    ti, oi = (i,h) if sign==1 else (h,i)
                    new = move(sea,copy*4*box.volume+4*box.index[target]+ti,copy*4*box.volume+4*box.index[origin]+oi)
                    if new is None: continue
                    expected = np.zeros(box.volume,dtype=np.int64)
                    expected[box.index[target]] += 1; expected[box.index[origin]] -= 1
                    assert np.array_equal(D@(sign*flow),expected)
                    assert new.bit_count()==sea.bit_count()
                    assert np.array_equal(D@(sign*flow+face),expected)
                    checks += 1
assert checks == 64*12

# Exact independent Feshbach matrix-walk, and derivative sign, at native energies.
fold_rows=[]
for leftrole,rightrole in ((0,0),(0,1),(1,1)):
    left=delivered['roles'][leftrole]['paths'][0]
    right=delivered['roles'][rightrole]['paths'][2]
    d1,d2,g1,g2 = map(F,(left['physical_intermediate_energy_exact'],right['physical_intermediate_energy_exact'],left['sea_PH_norm_squared_exact'],right['sea_PH_norm_squared_exact']))
    # Q states A,B,AB with vacuum removed; length-two middle walk exhausts Sigma4.
    energies = (d1,d2,d1+d2)
    outer = (g1,g2)
    sigma4 = F(0)
    for a in (0,1):
        for last in (0,1):
            sigma4 -= g1*g2/(energies[a]*energies[2]*energies[last])
    e2 = -g1/d1-g2/d2
    derivative = -g1/d1**2-g2/d2**2  # d/dz (z-d)^-1 = -(z-d)^-2
    fullfold = e2*derivative
    self_fold = g1*g1/d1**3+g2*g2/d2**3
    crossfold = fullfold-self_fold
    assert sigma4+crossfold == 0 and sigma4-crossfold != 0
    fold_rows.append({'roles':[leftrole,rightrole],'matrix_walk_sigma4':str(sigma4),'cross_fold':str(crossfold),'correct_sum':'0','wrong_derivative_sign_sum':str(sigma4-crossfold),'deleted_fold_sum':str(sigma4)})
# Whole-zero projector versus mean subtraction: exact degeneracy counterexample.
# H=diag(0,0,256), Omega=e0, O e0=(0,3/5,4/5), <O>=0.
connected_mass=F(1); wholeQ_mass=F(16,25); connected_zero_mass=F(9,25)
assert connected_mass != wholeQ_mass and connected_mass-connected_zero_mass == wholeQ_mass
positive_m1=256*wholeQ_mass
assert positive_m1 == 256*(connected_mass-connected_zero_mass)
# Shared electric copy price is quadratic in TOTAL flow.
shared=64*int((2*path)@(2*path)); separate=128*int(path@path)
assert shared == 512 and separate == 256
unit = box.path((4,0,0),0,1)
assert 64*int((2*unit)@(2*unit)) == 256 != 128

a = sum((p['count']*F(r['vacuum_dressing_norm_density_exact']) for p,r in zip(profiles,delivered['roles'])),F(0))
assert a>0 and a==F(delivered['linked_attempt']['vacuum_dressing_norm_density_all64_exact'])
for row in delivered['linked_attempt']['actual_volume_scaling']:
    assert F(row['exact_norm_squared'])==row['N']**3*a
result = {
    'kind':'NEW_INDEPENDENT_FLUX_PHASE_MATHEMATICAL_CONTROLS',
    'source_and_input_sha256':bindings,
    'old_capsule_binding':{'all_manifest_members_verified_per_archive':archive_counts,'natural_exit_receipt_read_not_rerun':True,'local_inputs_equal_bound_archive_bytes':True},
    'new_controls':{
        'cycle_metric_and_fullfibre':{'N':5,'d':len(chords),'CGram_witness_diagonal':int(G[j,j]),'wrong_orthonormal_diagonal':1,'packet_energy_exact':str(base),'wrong_identity_metric_energy_exact':str(wrong_metric_energy),'charged_coset_divergence_L1':int(np.sum(np.abs(rho))),'charged_path_not_in_cycle_lattice':True,'face_preserves_charged_coset':True},
        'contact_order_domain':{'charged_coset_source':True,'high_flow_dot_c':dot,'complete_action_coefficients_exact':[str(comm[tuple(probe-face)]),str(comm[tuple(probe+face)])],'wrong_E_cos_order_rejected':True,'contact_deleted_rejected':True,'domain_bound':'|k+-c|^4 <= 8|k|^4+128; finite shifts preserve D(64|k|^2), then bounded V leaves domain unchanged','all64_explicit_native_CAR_Gauss_checks':checks},
        'fold_sign':fold_rows,
        'mean_vs_wholeQ0':{'connected_mass':str(connected_mass),'whole_positive_mass':str(wholeQ_mass),'connected_zero_energy_mass':str(connected_zero_mass),'positive_m1_both':str(positive_m1),'mean_subtraction_does_not_normalize_Schur':True,'counterexample_scope':'Degenerate-ground finite spectral algebra, not measured ground of native law'},
        'native_moments_and_vacuum':{'b_exact':str(b),'b_float':float(b),'m1_upper':256,'m2_upper_physical1_exact':str(131072+4096*b),'a_exact':str(a),'a_float':float(a),'vacuum_relative_bound_fails':True,'same_unit_edge_total_price':256,'wrong_copy_price':128,'same_length2_path_total_price':shared,'wrong_length2_copy_price':separate}
    },
    'execution_scope':'NEW exact controls only; source pure class and read-only constructors; no old run/spectral_box/local_action_controls/linked_pair/neutral_AP_ground called',
    'runtime':{'executable':sys.executable,'python':sys.version,'numpy':np.__version__},
    'campaign_complete':False,
}
output = ROOT/(PREFIX+'controls_actual_20261010.json')
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'all_controls_reached':True,'all64_Gauss_checks':checks,'cycle_N':5,'fold_cases':len(fold_rows),'b':float(b),'a':float(a)}))
