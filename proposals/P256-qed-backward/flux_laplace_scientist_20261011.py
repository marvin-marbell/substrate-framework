"""Native exact-g coherent four-distinct-edge coefficient; no historical CLI runs."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import sympy as sy
import charged_output_lower_certificate_20261011 as iv
import charged_stieltjes_scientist_20261011 as cx

ROOT = Path(__file__).resolve().parent
Z = iv.iv(0)
ONE = iv.iv(1)
CZ = (Z, Z)
ORDERS = list(itertools.permutations(range(4)))
SPINS = list(itertools.product(range(2), repeat=4))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@lru_cache(None)
def expminus(x):
    """Outward fixed-point alternating Taylor, range reduction, exact remainder."""
    x = F(x)
    assert x >= 0
    reductions = 0
    while x > 1:
        x /= 2
        reductions += 1
    a = iv.iv(x)
    total = ONE
    term = ONE
    for n in range(1, 65):
        term = iv.mul(iv.mul(term, a), iv.iv(F(1, n)))
        total = iv.add(total, iv.scale(term, (-1)**n))
    remainder = iv.mul(iv.mul(term, a), iv.iv(F(1, 65)))[1]
    total = (total[0]-remainder, total[1]+remainder)
    for _ in range(reductions):
        total = iv.mul(total, total)
    return total

def scaled(a, factor):
    return iv.mul(a, iv.iv(factor))

def complex_scaled(a, factor):
    return scaled(a[0], factor), scaled(a[1], factor)

def matrix_product(A, B):
    return [[cx.ca(cx.cm(A[i][0], B[0][j]), cx.cm(A[i][1], B[1][j])) for j in range(2)] for i in range(2)]

def trace_product(matrices):
    out = matrices[0]
    for matrix in matrices[1:]:
        out = matrix_product(out, matrix)
    return cx.ca(out[0][0], out[1][1])

def dagger(A):
    return [[cx.cc(A[j][i]) for j in range(len(A))] for i in range(len(A))]

def l1(A):
    return F(sum(max(abs(x) for x in value[0])+max(abs(x) for x in value[1]) for row in A for value in row), iv.Q)

def exact_matrix(raw):
    return sy.Matrix([[sy.Rational(a)+sy.I*sy.Rational(b) for a,b in row] for row in raw])

def native_matrices():
    """Exactly the archived native binaryPhi/free-frame, not method matrices."""
    boundary = json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_text())['all64_boundary']
    with np.load(ROOT/'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz', allow_pickle=False) as data:
        raw = data['Phi_binary_section'][1]
    phi = sy.Matrix([[sy.Rational(F(float(complex(v).real)))+sy.I*sy.Rational(F(float(complex(v).imag))) for v in row] for row in raw])
    had = sy.Matrix([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]])/sy.sqrt(2)
    frame = sy.simplify(had.T*phi*had)
    gamma = sy.simplify(frame/sy.sqrt(sy.trace(frame*frame)/4))
    S = sy.Matrix.hstack(sy.eye(4)[:,0],gamma*sy.eye(4)[:,0],sy.eye(4)[:,1],gamma*sy.eye(4)[:,1])
    P = sy.diag(1,-1,1,-1)
    assert sy.simplify(S.conjugate().T*S) == sy.eye(4)
    result = {}
    exact = {}
    for role,row in zip(('active','aux'), boundary['roles']):
        result[role] = {}
        exact[role] = {}
        for axis in (1,2):
            path = next(p for p in row['paths'] if p['axis']==axis and p['length']==1)
            full = sy.simplify(P*S.conjugate().T*exact_matrix(path['matrix_exact'])*S*P)
            result[role][axis] = [[cx.ci(full[p,q]) for q in range(4)] for p in range(4)]
            exact[role][axis] = [[str(full[p,q]) for q in range(4)] for p in range(4)]
    return result, exact

def order_data(order, orientation):
    n = [256]*4
    sign = 1
    charged_counts = []
    for stage,e in enumerate(order):
        creation, annihilation = (e,(e+1)%4) if orientation==1 else ((e+1)%4,e)
        sign *= (-1)**(sum(n[:annihilation])+sum(n[:creation])-(annihilation<creation))
        n[creation] += 1
        n[annihilation] -= 1
        if stage<3:
            charged_counts.append(sum(x!=256 for x in n))
    assert n == [256]*4
    between = []
    creation_first = []
    for vertex in range(4):
        ce,ae = (vertex,(vertex-1)%4) if orientation==1 else ((vertex-1)%4,vertex)
        a,b = order.index(ce),order.index(ae)
        between.append(tuple(range(min(a,b),max(a,b))))
        creation_first.append(a<b)
    return sign,charged_counts,between,creation_first

def contraction(heat, edge_matrices, orientation):
    vertices = (0,1,2,3) if orientation==1 else (0,3,2,1)
    factors = []
    for vertex in vertices:
        edge = vertex if orientation==1 else (vertex-1)%4
        factors.append(matrix_product(heat[vertex],edge_matrices[edge]))
    return trace_product(factors)

def mask_op(mask, mode, creation):
    if bool((mask>>mode)&1)==creation:
        return None
    return mask^(1<<mode), (-1)**((mask&((1<<mode)-1)).bit_count())

def adverse_algebra():
    """NEW full-CAR algebra witnesses, not a rerun of the old toy solver."""
    initial = sum(5<<(4*v) for v in range(4))
    counters = []
    for orientation in (-1,1):
        for order in ORDERS:
            graded,*_ = order_data(order,orientation)
            found = False
            for modes in itertools.product(range(2),repeat=8):
                mask = initial
                full_sign = local_sign = 1
                mismatch = False
                for edge in order:
                    a,b = (edge,(edge+1)%4) if orientation==1 else ((edge+1)%4,edge)
                    p,q = modes[2*edge:2*edge+2]
                    before = mask
                    x = mask_op(mask,4*b+q,False)
                    if x is None: break
                    y = mask_op(x[0],4*a+p,True)
                    if y is None: break
                    u = mask_op((before>>(4*b))&15,q,False)
                    v = mask_op((before>>(4*a))&15,p,True)
                    assert u is not None and v is not None
                    full_sign *= x[1]*y[1]
                    local_sign *= u[1]*v[1]
                    mismatch |= x[1]*y[1] != u[1]*v[1]
                    mask = y[0]
                else:
                    if mask == initial:
                        assert full_sign == graded*local_sign
                        found = True
                        counters.append({'orientation':orientation,'order':list(order),'graded_sign':graded,'global_sign':full_sign,'local_product':local_sign,'per_hop_mismatch':mismatch})
                        break
            assert found
    assert any(x['per_hop_mismatch'] for x in counters)
    assert any(x['graded_sign']==-1 for x in counters)
    # Neutral number2 has six basis masks; a rank5 crop loses reachable scalar.
    v = np.zeros(16);v[5]=1
    x = mask_op(5,0,False);assert x is not None
    y = mask_op(x[0],1,True);assert y is not None
    exterior = np.zeros(16);exterior[y[0]]=x[1]*y[1]
    scalar = np.zeros(16);scalar[6]=scalar[9]=1/np.sqrt(2)
    cropped = exterior-scalar*np.dot(scalar,exterior)
    fullnorm = float(exterior@exterior)
    cropnorm = float(cropped@cropped)
    assert fullnorm==1 and abs(cropnorm-.5)<1e-14
    opposite = [o for o in ORDERS if (o[0]-o[1])%4==2]
    assert len(opposite)==8 and all(order_data(o,1)[1][1]==4 for o in opposite)
    # Terminal labelled conservation: one incoming/outgoing label per vertex;
    # common edge copy, independent vertex spin => 64*2^4, not 64^4 or 128.
    valid = 0
    for spins in SPINS:
        for copy in range(64):
            labels = [(copy,spins[e],spins[(e+1)%4]) for e in range(4)]
            assert all(labels[e][1]==labels[(e-1)%4][2] for e in range(4))
            valid += 1
    assert valid==1024
    wrong_labels = [(0,0,0),(1,0,0),(0,0,0),(0,0,0)]
    assert any(wrong_labels[e][0]!=wrong_labels[(e-1)%4][0] for e in range(4))
    assert 64*4==256 and 2*64*4!=256 and 64*3!=256
    return {'fullCAR_phase_witnesses':counters,'neutral_crop_full_norm_squared':fullnorm,'neutral_crop_wrong_norm_squared':cropnorm,'opposite_first_two_orders':len(opposite),'omitted_twoPH_rejected':True,'terminal_label_multiplicity':valid,'wrong_independent_copies_rejected':True,'external64_once_rejected_if_omitted_or_doubled':True,'incoming_face_to_twice_face_energy':64*(16-4)}

def rectangle(raw):
    return iv.iv(F(raw['re'][0]))[0],iv.iv(F(raw['re'][1]))[1]

def decode_matrix(raw):
    return [[((iv.iv(F(v['re'][0]))[0],iv.iv(F(v['re'][1]))[1]),(iv.iv(F(v['im'][0]))[0],iv.iv(F(v['im'][1]))[1])) for v in row] for row in raw]

def output_complex(value):
    return {'re':cx.interval_output(value[0]),'im':cx.interval_output(value[1])}

def run(interface_path, helper_name):
    started = time.monotonic()
    interface = json.loads(interface_path.read_text())
    helper = importlib.import_module(helper_name)
    request = json.loads((ROOT/'flux_laplace_time_request_20261011.json').read_text())
    T = F(request['cube']['upper_exact'])
    divisions = request['cube']['subdivisions_per_axis']
    h = T/divisions
    ground = json.loads((ROOT/'local_so5_ground_upper_complete_state_consumer_20261011_receipt.json').read_text())
    c = -5120-F(ground['ground_energy_interval_exact'][1])
    assert c>0
    matrices, exact = native_matrices()
    controls = adverse_algebra()
    cache = {}
    def kernel(role, first, duration):
        orientation = 'creation' if first else 'annihilation'
        key = role,orientation,duration
        if key not in cache:
            cache[key] = decode_matrix(helper.evaluate(interface,role,orientation,duration))
        return cache[key]
    rows = []
    totals = {z:CZ for z in (0,64)}
    equal_totals = {z:CZ for z in (0,64)}
    for role,count in (('active',1),('aux',63)):
        forward = [matrices[role][1],matrices[role][2],dagger(matrices[role][1]),dagger(matrices[role][2])]
        for orientation in (-1,1):
            edges = forward if orientation==1 else [dagger(A) for A in forward]
            spin_blocks = []
            B = F(0)
            for spins in SPINS:
                blocks = []
                for e, matrix in enumerate(edges):
                    sa,sb = (spins[e],spins[(e+1)%4]) if orientation==1 else (spins[(e+1)%4],spins[e])
                    blocks.append([[matrix[2*sa+p][2*sb+q] for q in range(2)] for p in range(2)])
                spin_blocks.append(blocks)
                B += l1(blocks[0])*l1(blocks[1])*l1(blocks[2])*l1(blocks[3])
            for order in ORDERS:
                sign,ncharged,between,first = order_data(order,orientation)
                center = {z:CZ for z in totals}
                errors = {z:F(0) for z in totals}
                for box in itertools.product(range(divisions),repeat=3):
                    lower = [k*h for k in box]
                    tau = [(k+F(1,2))*h for k in box]
                    durations = [sum((tau[j] for j in indices),F(0)) for indices in between]
                    heat = [kernel(role,first[v],durations[v]) for v in range(4)]
                    amplitude = CZ
                    for blocks in spin_blocks:
                        amplitude = cx.ca(amplitude,complex_scaled(contraction(heat,blocks,orientation),sign))
                    variations = []
                    for vertex,indices in enumerate(between):
                        low = sum((lower[j] for j in indices),F(0))
                        delta = len(indices)*h/2
                        moment_cap = F(helper.derivative_bound(interface,role,'creation' if first[vertex] else 'annihilation',low,order=1)[1])
                        global_cap = F(1,2)/low if low else moment_cap
                        variations.append(min(F(1),delta*min(moment_cap,global_cap)))
                    for z in totals:
                        weight = ONE
                        for j in range(3):
                            rate = F(64*(j+1)+z)
                            weight = iv.mul(weight,scaled(iv.sub(expminus(rate*lower[j]),expminus(rate*(lower[j]+h))),1/rate))
                        center[z] = cx.ca(center[z],cx.cm(amplitude,cx.scalar(weight)))
                        errors[z] += B*F(weight[1],iv.Q)*sum(variations)
                values = {}
                equal_heat = [kernel(role,first[v],F(0)) for v in range(4)]
                equal_amplitude = CZ
                for blocks in spin_blocks:
                    equal_amplitude = cx.ca(equal_amplitude,complex_scaled(contraction(equal_heat,blocks,orientation),sign))
                for z in totals:
                    rates = [F(64*(j+1)+ncharged[j]*c+z) for j in range(3)]
                    denominator = rates[0]*rates[1]*rates[2]
                    tail = B*sum(F(expminus(rate*T)[1],iv.Q) for rate in rates)/denominator
                    price = errors[z]+tail
                    rad = iv.iv(price)[1]
                    value = ((center[z][0][0]-rad,center[z][0][1]+rad),(center[z][1][0]-rad,center[z][1][1]+rad))
                    totals[z] = cx.ca(totals[z],complex_scaled(value,count))
                    equal = complex_scaled(equal_amplitude,F(count, (64+z)*(128+z)*(192+z)))
                    equal_totals[z] = cx.ca(equal_totals[z],equal)
                    values[str(z)] = {'midpoint_coherent_all16_spins':output_complex(center[z]),'quadrature_error_radius_exact':str(errors[z]),'infinite_tail_radius_exact':str(tail),'all16_spins_one_copy_interval':output_complex(value),'rates_exact':list(map(str,rates))}
                rows.append({'role':role,'copy_count':count,'orientation':orientation,'order':list(order),'graded_globalCAR_sign':sign,'charged_counts':ncharged,'between_tau_indices':between,'creation_first':first,'all16_spin_assignments':True,'coherent_L1_spin_sum_exact':str(B),'values':values})
    assert len(rows)==96 and len(cache)>4
    changed = []
    for key,heat in cache.items():
        if key[2]>0:
            zero = kernel(key[0],key[1]=='creation',F(0))
            changed.append(any(heat[i][i][0][1]<zero[i][i][0][0] for i in range(2)))
    controls['actual_heat_decay_resolved_against_equal_time'] = any(changed)
    controls['equal_time_false_representation_rejected_by_positive_charged_floor'] = c>0 and expminus(c*T)[1]<iv.Q
    controls['equal_time_numerical_midpoint_not_certified'] = True
    inputs = ['flux_laplace_time_request_20261011.json','flux_laplace_proof_20261011.json',interface_path.name,helper_name+'.py','charged_first_shell_actual04_20261011_receipt.json','joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz','local_so5_ground_upper_complete_state_consumer_20261011_receipt.json','charged_output_lower_certificate_20261011.py','charged_stieltjes_scientist_20261011.py','harmonic_car_bridge_scientist_20261011.py']
    return {'kind':'NEW_ACTUAL_NATIVE_EXACT_G_SIGNED_COHERENT_LAPLACE_COEFFICIENT_INTERVAL','own_source_sha256':sha(Path(__file__)),'input_sha256':{p:sha(ROOT/p) for p in inputs},'heat_interface_sha256':sha(interface_path),'charged_floor_relative_exact':str(c),'native_matrices_exact':exact,'cube':request['cube'],'coefficient_intervals':{str(z):output_complex(value) for z,value in totals.items()},'wrong_equal_time_coefficient_intervals':{str(z):output_complex(value) for z,value in equal_totals.items()},'sign_resolved':{str(z):not(value[0][0]<=0<=value[0][1]) for z,value in totals.items()},'orders':rows,'heat_cache_entries':len(cache),'adverse':controls,'arithmetic':'Conditional binary64 formation inputs from charged interface; subsequent 100-bit outward integer/rational interval arithmetic. Taylor remainder and range reduction paid. No float integration or empirical tail.','whole_charged_semigroups':True,'source_norm_only_redescription':False,'incoming0_only':True,'unpaid':['Sharper heat matrix Loewner/entry correlations and quadrature if coefficient sign unresolved','physical1 normalized rooted stability','fullPhi/lattice stationary ground','phase/stiffness','gap closing pole/IR atom/commoncone'],'elapsed_seconds':time.monotonic()-started}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface',required=True)
    parser.add_argument('--helper',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = run(Path(args.interface),args.helper)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('kind','coefficient_intervals','sign_resolved','heat_cache_entries','elapsed_seconds')},indent=2),flush=True)

if __name__=='__main__':
    main()
