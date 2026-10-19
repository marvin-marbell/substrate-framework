"""Independent mathematical control: literal full N8 exterior CAR, subset DP.
No existing ring consumer/capsule/helper is imported or run.  The frozen input
and primitive scalar law supply data only; this is not a physical ground test.
"""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import hashlib
import json
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
N = 8
selected = [
    'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json',
    'joined_phi_u1_bulk_flux_ring_corrected_metric_20261010_actual.json',
]
inputs = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in selected}
with np.load(ROOT/selected[0], allow_pickle=False) as data:
    phi = data['Phi_binary_section']
cfg = json.loads((ROOT/selected[1]).read_text())['input_descriptor']['original_micro_config_UNCHANGED']
receipt = json.loads((ROOT/selected[2]).read_text())

def exact(z):
    return sp.Rational(float(z.real)) + sp.I*sp.Rational(float(z.imag))

# Reconstruct the exact normalized eigenframe and literal primitive scalar law,
# not the producer's frame/profiles/ring functions.
h = sp.Matrix([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])
frame = [h.T*sp.Matrix(4,4,lambda i,j: exact(phi[m,i,j]))*h/2 for m in range(4)]
s = sp.Rational(float(phi[0,0,2].real))
assert frame[0] == sp.diag(-s,-s,s,s)
vertices = [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
site_ids = [(x*N+y)*N+z for x,y,z in vertices]
sea = frozenset(4*x+j for x in range(N**3) for j in (0,1))
assert len(sea) == 2*N**3

def exterior_apply(occupied, create, remove, parity=True):
    if remove not in occupied or create in occupied:
        return None
    lo, hi = sorted((create, remove))
    sign = (-1)**sum(lo < i < hi for i in occupied) if parity else 1
    return occupied.difference((remove,)).union((create,)), sign

def calculate(mass, one_ph_only=False, parity=True, incoming=0):
    onsite = sp.Rational(float(mass + 4.5*cfg['light_r']/cfg['a']))*s
    scalar = sp.Rational(float(-cfg['light_r']/cfg['a']))
    axial = -sp.I*sp.Rational(float(cfg['light_v']/(2*cfg['a'])))
    forward = [scalar*frame[0] + axial*frame[axis+1] for axis in (0,1)]
    matrices = [forward[0], forward[1], forward[0].conjugate().T, forward[1].conjugate().T]
    channels = []
    for edge, matrix in enumerate(matrices):
        channels.append([(4*site_ids[edge]+i, 4*site_ids[(edge+1)%4]+j, matrix[i,j])
                         for i in range(4) for j in range(4) if matrix[i,j] != 0])
    # Used-edge subsets merge temporal orders after each step.  Every amplitude
    # carries a differentiated weight, computed from its cumulative sum 1/E.
    current = {(0,sea): (sp.S.One, sp.S.Zero)}
    surviving_ph = set()
    for depth in range(1,5):
        following = defaultdict(lambda: [sp.S.Zero,sp.S.Zero])
        for (used,occupied),(amplitude,derivative) in current.items():
            for edge in range(4):
                if used & (1 << edge):
                    continue
                new_used = used | (1 << edge)
                for create,remove,entry in channels[edge]:
                    moved = exterior_apply(occupied,create,remove,parity)
                    if moved is None:
                        continue
                    destination,sign = moved
                    a = sp.expand(sign*entry*amplitude)
                    d = sp.expand(sign*entry*derivative)
                    if depth < 4:
                        # Q removes ALL source-sea returns, independent of flow.
                        if destination == sea:
                            continue
                        ph = len(destination.difference(sea))
                        if one_ph_only and ph > 1:
                            continue
                        surviving_ph.add(ph)
                        electric = 64*sum((incoming+int(bool(new_used & (1 << e))))**2 for e in range(4))
                        energy = 2*onsite*ph + electric
                        assert energy > 0
                        a /= -energy
                        d = d/(-energy) + a/energy
                    key = (new_used,destination)
                    following[key][0] += a
                    following[key][1] += d
        current = {key: (sp.expand(a),sp.expand(d)) for key,(a,d) in following.items()}
    coefficient,derivative = current.get((15,sea),(sp.S.Zero,sp.S.Zero))
    assert coefficient.is_Rational and derivative.is_Rational
    return coefficient, derivative, sorted(surviving_ph)

rows = []
for role,mass,count in [('active',cfg['active_m'],1),('auxiliary',cfg['auxiliary_m'],63)]:
    c,d,ph = calculate(mass)
    one,_,_ = calculate(mass,one_ph_only=True)
    no_car,_,_ = calculate(mass,parity=False)
    shifted,_,_ = calculate(mass,incoming=1)
    recorded = next(row for row in receipt['rows'] if row['role']==role)
    assert c == sp.Rational(recorded['actual']['coefficient_exact'])
    assert d == sp.Rational(recorded['actual']['z_derivative_exact'])
    assert one == sp.Rational(recorded['wrong_one_PH_projection']['coefficient_exact'])
    assert no_car == -c and no_car != c
    assert shifted == sp.Rational(recorded['incoming_one_loop']['coefficient_exact']) and shifted != c
    rows.append(dict(role=role,count=count,coefficient_exact=str(c),z_derivative_exact=str(d),
                     source_metric_exact=str(-d),coefficient=float(c),surviving_intermediate_PH_counts=ph,
                     wrong_one_PH_exact=str(one),wrong_one_PH=float(one),wrong_no_CAR=float(no_car),
                     incoming_one_loop=float(shifted),all_plausible_wrongcases_rejected=True))
c = sum(sp.Rational(row['coefficient_exact'])*row['count'] for row in rows)
metric = -sum(sp.Rational(row['z_derivative_exact'])*row['count'] for row in rows)
assert c == sp.Rational(receipt['coefficient_of_positive_oriented_shift_exact'])
assert metric == sp.Rational(receipt['normalization']['fourth_order_source_metric_character_exact'])
r = receipt['controlled_remainder']
B,delta,t = (sp.Rational(r[name]) for name in ('hopping_norm_upper_exact','QH0Q_gap_lower_exact','test_lambda_exact'))
remainder = t**5*B**5/(delta**4*(1-t*B/delta))
assert remainder == sp.Rational(r['remainder_after_order4_upper_exact'])
assert c*t**4 + remainder < 0 and t < delta/B < 1

# Fresh contrary character: at N8, four length-two axial paths close endpoints,
# satisfy zero divergence, but carry a full winding vector, NOT a unit face.
flow = {}
x = 0
for hop in range(4):
    for segment in range(2):
        edge = 3*((x*N+0)*N+0)
        flow[edge] = flow.get(edge,0)+1
        x = (x+1)%N
face = {3*((0*N+0)*N+0):1,3*((1*N+0)*N+0)+1:1,
        3*((0*N+1)*N+0):-1,3*((0*N+0)*N+0)+1:-1}
assert x == 0 and flow != face and sum(flow.values())//N == 1
# Separate exact projection control in dimension three: full resolvent in Q's
# place resums forbidden P returns and changes even the leading kernel.
z = sp.S.Zero
H = sp.Matrix([[2,1,2],[1,3,1],[2,1,5]])
b = H[0,1:3]
D = H[1:3,1:3]
irreducible = (b*(z*sp.eye(2)-D).inv()*b.T)[0]
full_substitution = (b*(z*sp.eye(3)-H).inv()[1:3,1:3]*b.T)[0]
assert irreducible == -sp.Rational(13,14) and full_substitution == -sp.Rational(26,15)
assert full_substitution != irreducible
print(json.dumps(dict(kind='INDEPENDENT_FULL_N8_EXTERIOR_CAR_SUBSET_DP_MATH_CONTROL',
                     inputs_sha256=inputs,whole_sea_particles_per_species=len(sea),
                     active_vertex_ambient_order=site_ids,rows=rows,coefficient=float(c),source_metric=float(metric),
                     exact_receipt_coefficient_and_metric_agree=True,
                     controlled_interval_checked=dict(radius=float(delta/B),test_lambda=float(t),
                                                      upper=float(c*t**4+remainder),physical_one_not_controlled=True),
                     fresh_winding_wrongcase=dict(N=N,native_lengths=[2,2,2,2],closed_endpoints=True,
                                                  integer_edge_count=len(flow),homology=[1,0,0],
                                                  not_unit_face=True,endpoint_only_test_rejected=True),
                     independent_three_dimensional_schur_control=dict(irreducible=str(irreducible),
                                                                      wrong_full_resolvent=str(full_substitution),
                                                                      forbidden_source_returns_rejected=True),
                     physical_nonclaims=['not fullPhi ground','not physical hopping1 control','not stiffness','not pole']),
                 sort_keys=True))
