"""NEW all-time native charged flux consumer; no historical consumer import/CLI."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools as it
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import sympy as sy
import charged_output_lower_certificate_20261011 as iv
from typing import cast
import charged_stieltjes_scientist_20261011 as cx

ROOT = Path(__file__).resolve().parent
# Exact trusted rational proof endpoints can exceed Python's decimal-display cap.
# Preserve exact endpoints, rather than truncating/rounding the proof payload.
sys.set_int_max_str_digits(0)
Z = iv.iv(0)
CZ = (Z, Z)
ORDERS = list(it.permutations(range(4)))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def cscale(x, q):
    r = iv.iv(q)
    return iv.mul(x[0], r), iv.mul(x[1], r)

def mm(A, B):
    n = len(A)
    return [[sumc(cx.cm(A[i][k], B[k][j]) for k in range(n)) for j in range(n)] for i in range(n)]

def sumc(values):
    s = CZ
    for x in values:
        s = cx.ca(s, x)
    return s

def dagger(A):
    return [[cx.cc(A[j][i]) for j in range(len(A))] for i in range(len(A))]

def lift(A):
    return [[A[i % 2][j % 2] if i // 2 == j // 2 else CZ for j in range(4)] for i in range(4)]

def mscale(A, q):
    return [[cscale(x, q) for x in row] for row in A]

def madd(*As):
    return [[sumc(A[i][j] for A in As) for j in range(len(As[0]))] for i in range(len(As[0]))]

def frob(A):
    s = sum(iv.sq_upper(v[k]) for row in A for v in row for k in (0, 1))
    return F(iv.sqrt((s, s))[1], iv.Q)

def hermitian_op(A):
    # Hermitian 2x2 Gershgorin and Frobenius upper bounds, no entrywise Loewner use.
    off = iv.sqrt((0, iv.sq_upper(A[0][1][0]) + iv.sq_upper(A[0][1][1])))[1]
    return min(frob(A), F(max(max(abs(x) for x in A[i][i][0]) + off for i in range(2)), iv.Q))

def native_matrices():
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
    result, exact, norms = {}, {}, {}
    for role, row in zip(('active','aux'), boundary['roles']):
        result[role], exact[role], norms[role] = {}, {}, {}
        for axis in (1,2):
            path = next(p for p in row['paths'] if p['axis']==axis and p['length']==1)
            physical = sy.Matrix([[sy.Rational(a)+sy.I*sy.Rational(b) for a,b in rr] for rr in path['matrix_exact']])
            full = sy.simplify(P*S.conjugate().T*physical*S*P)
            result[role][axis] = [[cx.ci(full[p,q]) for q in range(4)] for p in range(4)]
            exact[role][axis] = [[str(full[p,q]) for q in range(4)] for p in range(4)]
            gram = sy.simplify(full.conjugate().T*full)
            # Bound the exact PSD Gram operator, without assuming unitarity.
            G = [[cx.ci(gram[p,q]) for q in range(4)] for p in range(4)]
            rows = []
            for p in range(4):
                off = sum(F(iv.sqrt((0,iv.sq_upper(G[p][q][0])+iv.sq_upper(G[p][q][1])))[1],iv.Q) for q in range(4) if q!=p)
                rows.append(F(G[p][p][0][1],iv.Q)+off)
            op = min(frob(result[role][axis]),F(iv.sqrt(iv.iv(max(rows)))[1],iv.Q))
            norms[role][axis] = (op, frob(result[role][axis]))
    return result, exact, norms

def order_data(order, orientation):
    n, sign, counts = [256]*4, 1, []
    for stage, edge in enumerate(order):
        c, a = (edge,(edge+1)%4) if orientation==1 else ((edge+1)%4,edge)
        sign *= (-1)**(sum(n[:a])+sum(n[:c])-(a<c))
        n[c] += 1
        n[a] -= 1
        if stage<3:
            counts.append(sum(v!=256 for v in n))
    assert n == [256]*4
    between, first = [], []
    for vertex in range(4):
        c,a = (vertex,(vertex-1)%4) if orientation==1 else ((vertex-1)%4,vertex)
        i,j = order.index(c),order.index(a)
        between.append(tuple(range(min(i,j), max(i,j))))
        first.append(i<j)
    return sign, counts, between, first

@lru_cache(None)
def duration_monomial(indices, degree):
    result = {(0,0,0): F(1)}
    for _ in range(degree):
        out = {}
        for p,c in result.items():
            for j in indices:
                q = tuple(v+(k==j) for k,v in enumerate(p))
                out[q] = out.get(q,F(0))+c
        result = out
    return tuple(result.items())

@lru_cache(None)
def integral_weight(rates, durations, degrees):
    poly = {(0,0,0): F(1)}
    for indices, degree in zip(durations,degrees):
        out = {}
        for p,c in poly.items():
            for q,d in duration_monomial(indices, degree):
                r = tuple(p[j]+q[j] for j in range(3))
                out[r] = out.get(r,F(0))+c*d
        poly = out
    return sum((c*math.prod(F(math.factorial(p[j]),rates[j]**(p[j]+1)) for j in range(3)) for p,c in poly.items()),F(0))

def scalar_integral(functions, between, base):
    total = F(0)
    durations = tuple(between)
    for terms in it.product(*functions):
        rates = tuple(base[j]+sum(t[0] for v,t in enumerate(terms) if j in between[v]) for j in range(3))
        assert min(rates)>0
        total += math.prod(t[2] for t in terms)*integral_weight(rates,durations,tuple(t[1] for t in terms))
    return total

def encode(A):
    return [[{'re':iv.exact_endpoints(v[0]),'im':iv.exact_endpoints(v[1])} for v in row] for row in A]

def heat_data(interface, row):
    M = [[[tuple((iv.iv(F(v[k][0]))[0],iv.iv(F(v[k][1]))[1]) for k in ('re','im')) for v in rr] for rr in row['moments'][name]] for name in ('M0','M1','M2')]
    # JSON stores annihilator-source Gram; native minus is its transpose.
    if row['orientation']=='annihilation':
        M = [[[A[j][i] for j in range(2)] for i in range(2)] for A in M]
    b = F(interface['charged_floor_exact'])
    eta = F(interface['ground_distance_exact'][1])
    tr0 = sum(F(sum(M[0][i][i][0]),2*iv.Q) for i in range(2))
    tr1 = sum(F(sum(M[1][i][i][0]),2*iv.Q) for i in range(2))
    a = max(b+1,tr1/tr0)
    r = a-b
    V = madd(M[2],mscale(M[1],-2*a),mscale(M[0],a*a))
    B = mscale(V,1/(2*r*r))
    C0 = madd(M[0],mscale(B,-1))
    C1 = madd(mscale(M[0],a),mscale(M[1],-1),mscale(B,-r))
    terms = [(a,0,lift(C0)),(a,1,lift(C1)),(b,0,lift(B))]
    d = [(b,0,1/(r*r)),(a,0,-1/(r*r)),(a,1,-1/r)]
    vtrace = F(sum(V[i][i][0][1] for i in range(2)),iv.Q)
    vop = hermitian_op(V)
    assert vtrace>=0 and vop>=0
    source_norm = F(iv.sqrt(iv.iv(F(interface['source_norm_squared_exact'])))[1],iv.Q)
    transfer = eta*(1+source_norm)
    h = min(F(1),hermitian_op(M[0])+transfer)
    # Schatten-2 cap of Ispin tensor G, priced from actual source M0 and exact-g transfer.
    hf = min(F(2),F(iv.sqrt(iv.iv(2))[1],iv.Q)*frob(M[0])+2*transfer)
    return {'terms':terms,'a':a,'b':b,'V':V,'vtrace':vtrace,'vop':vop,'h':h,'hf':hf,'d':d,'source':[(b,0,transfer)],'coefficients':{'C_a0':encode(C0),'C_a1':encode(C1),'C_b0':encode(B)},'moment_rectangle_widths':{name:[[max(F(z[k][1]-z[k][0],iv.Q) for k in (0,1)) for z in rr] for rr in A] for name,A in zip(('M0','M1','M2'),M)}}

def central_integral(data, edges, orientation, between, base):
    vertices = (0,1,2,3) if orientation==1 else (0,3,2,1)
    total = CZ
    durations = tuple(between)
    for terms in it.product(*(d['terms'] for d in data)):
        rates = tuple(base[j]+sum(t[0] for v,t in enumerate(terms) if j in between[v]) for j in range(3))
        weight = integral_weight(rates,durations,tuple(t[1] for t in terms))
        factors = [mm(terms[v][2],edges[v if orientation==1 else (v-1)%4]) for v in vertices]
        out = factors[0]
        for A in factors[1:]:
            out = mm(out,A)
        total = cx.ca(total,cscale(sumc(out[i][i] for i in range(4)),weight))
    return total

def integrated_error(data, edge_norms, between, base, use_operator, include_source, schatten):
    total = F(0)
    components = {}
    op_product = math.prod(n[0] for n in edge_norms)
    for bits in it.product((0,1),repeat=4):
        if not any(bits):
            continue
        funcs, ratios = [], []
        for v, flag in enumerate(bits):
            d = data[v]
            if flag:
                cap = d['vop'] if use_operator else d['vtrace']
                funcs.append([(rate,degree,c*cap/2) for rate,degree,c in d['d']]+(d['source'] if include_source else []))
                ratios.append(F(2))
            else:
                funcs.append([(d['b'],0,d['h'])])
                ratios.append(d['hf']/d['h'])
        trace_cap = F(4)
        if schatten:
            # Generalized Schatten Holder over all eight factors. Pick ANY two S2 factors.
            ratios += [n[1]/n[0] for n in edge_norms]
            trace_cap = min(trace_cap,min(x*y for x,y in it.combinations(ratios,2)))
        value = op_product*trace_cap*scalar_integral(funcs,between,base)
        assert value>=0, ('positive pointwise remainder integrated negative',bits,value)
        components[''.join(map(str,bits))] = str(value)
        total += value
    return total, components

def complex_output(z):
    return {'re':cx.interval_output(z[0]),'im':cx.interval_output(z[1])}

def controls():
    # Independent finite spectral toy: tensor resolvents, not our monomial routine.
    spectral = [(F(2),F(1,3)),(F(5),F(2,3))]
    between = ((0,1),(1,2),(0,),(2,))
    base = (F(7),F(11),F(13))
    functions = [[(rate,0,weight) for rate,weight in spectral] for _ in range(4)]
    exact = scalar_integral(functions,between,base)
    independent = F(0)
    for states in it.product(spectral,repeat=4):
        denominator = math.prod(base[j]+sum(s[0] for v,s in enumerate(states) if j in between[v]) for j in range(3))
        independent += math.prod(s[1] for s in states)/denominator
    assert exact == independent
    # Polynomial toy checked by independent successive symbolic integration.
    x,y,z = sy.symbols('x y z',positive=True)
    expr = (1+x+y)*(2+y+z)*(3+x)*(4+z)*sy.exp(-7*x-11*y-13*z)
    independent_poly = sy.integrate(expr,(x,0,sy.oo),(y,0,sy.oo),(z,0,sy.oo))
    polynomial = scalar_integral([[(F(0),0,F(v)),(F(0),1,F(1))] for v in (1,2,3,4)],between,base)
    assert sy.Rational(polynomial.numerator,polynomial.denominator)==independent_poly
    signed = scalar_integral([[(F(2),0,F(1)),(F(3),0,F(-1))]]+ [[(F(0),0,F(1))]]*3,((0,),(),(),()),base)
    unsigned = scalar_integral([[(F(2),0,F(1)),(F(3),0,F(1))]]+ [[(F(0),0,F(1))]]*3,((0,),(),(),()),base)
    assert signed!=unsigned and signed==F(1,11*13)*(F(1,9)-F(1,10))
    tail = sy.exp(-sy.Rational(7,5))/sy.Integer(7*11*13)
    assert tail.is_positive
    # Every index i=2 sigma+p. All sixteen genuinely distinct spin assignments.
    G = [sy.Matrix([[v+1,sy.I],[-sy.I,v+2]]) for v in range(4)]
    E = [sy.Matrix(4,4,lambda i,j:(v+1)*(i+1)+2*(j+1)+sy.I*(i-j)) for v in range(4)]
    spectral_blocks = [sy.diag(sy.Rational(1,3),0),sy.diag(0,sy.Rational(2,3))]
    toy_terms = [(rate,0,lift([[cx.ci(A[i,j]) for j in range(2)] for i in range(2)])) for (rate,_),A in zip(spectral,spectral_blocks)]
    toy_edges = [[[cx.ci(A[i,j]) for j in range(4)] for i in range(4)] for A in E]
    coherent = central_integral([{'terms':toy_terms}]*4,toy_edges,1,between,base)
    direct = sy.Integer(0)
    for states in it.product(range(2),repeat=4):
        rates = [base[j]+sum(spectral[s][0] for v,s in enumerate(states) if j in between[v]) for j in range(3)]
        product = sy.eye(4)
        for v,s in enumerate(states):
            product *= sy.diag(spectral_blocks[s],spectral_blocks[s])*E[v]
        direct += sy.trace(product)/sy.Rational(math.prod(rates).numerator,math.prod(rates).denominator)
    direct = sy.expand(direct)
    for part,exact_part in zip(coherent,direct.as_real_imag()):
        exact_part = cast(sy.Rational, exact_part)
        rational = F(int(exact_part.p),int(exact_part.q))
        assert F(part[0],iv.Q)<=rational<=F(part[1],iv.Q)
    witnesses = []
    for orientation in (-1,1):
        edges = E if orientation==1 else [A.conjugate().T for A in E]
        vertices = (0,1,2,3) if orientation==1 else (0,3,2,1)
        full = sy.eye(4)
        for v in vertices:
            full *= sy.diag(G[v],G[v])*edges[v if orientation==1 else (v-1)%4]
        values = []
        for spins in it.product(range(2),repeat=4):
            out = sy.eye(2)
            for v in vertices:
                e = v if orientation==1 else (v-1)%4
                sr,sc = (spins[e],spins[(e+1)%4]) if orientation==1 else (spins[(e+1)%4],spins[e])
                block = edges[e][2*sr:2*sr+2,2*sc:2*sc+2]
                out *= G[v]*block
            values.append(sy.expand(sy.trace(out)))
        assert sy.expand(sy.trace(full)-sum(values))==0
        assert len(set(values))>1 and sy.expand(sum(values)-16*values[0])!=0
        witnesses.append({'orientation':orientation,'full_trace':str(sy.expand(sy.trace(full))),'distinct_spin_values':len(set(values)),'wrong_16_singleblock':str(16*values[0])})
    return {'spectral_rational_exact':str(exact),'independent_spectral_resolvent_equal':True,'coherent_matrix_spectral_resolvent_exact':str(direct),'coherent_matrix_algorithm_enclosure':complex_output(coherent),'independent_symbolic_polynomial_exact':str(independent_poly),'signed_cancellation_exact':str(signed),'absolute_coefficients_wrong':str(unsigned),'omitted_time_tail_rejected_exact':str(tail),'spin_convention':'i=2*sigma+p; i//2=sigma, i%2=p; vertex labels independently summed','nonidentical_spin_witnesses':witnesses,'wrong_spin_blocks_rejected':True}

def run():
    started = time.monotonic()
    interface_path = ROOT/'charged_cell_heat_interface_20261011.json'
    assert sha(interface_path)=='41d524542ded29c183426646bbdecf2f34b613bee7dcd13bf98e2a427c9c330f'
    interface = json.loads(interface_path.read_text())
    matrices, exact, norms = native_matrices()
    adverse = controls()
    data = {(r['role'],r['orientation']=='creation'):heat_data(interface,r) for r in interface['roles']}
    totals = {z:CZ for z in (0,64)}
    budgets = {z:{k:F(0) for k in ('trace_baseline','operator_remainder','coherent_remainder','source_increment')} for z in totals}
    rows = []
    for role,count in (('active',1),('aux',63)):
        forward = [matrices[role][1],matrices[role][2],dagger(matrices[role][1]),dagger(matrices[role][2])]
        en = [norms[role][j] for j in (1,2,1,2)]
        for orientation in (-1,1):
            edges = forward if orientation==1 else [dagger(A) for A in forward]
            for order in ORDERS:
                sign,counts,between,first = order_data(order,orientation)
                ds = [data[role,f] for f in first]
                values = {}
                for z in totals:
                    base = tuple(F(64*(j+1)+z) for j in range(3))
                    central = cscale(central_integral(ds,edges,orientation,between,base),sign)
                    baseline,_ = integrated_error(ds,en,between,base,False,False,False)
                    operator,_ = integrated_error(ds,en,between,base,True,False,False)
                    coherent,components = integrated_error(ds,en,between,base,True,False,True)
                    paid,paid_components = integrated_error(ds,en,between,base,True,True,True)
                    assert paid>=coherent
                    for k,val in zip(budgets[z],(baseline,operator,coherent,paid-coherent)):
                        budgets[z][k] += count*val
                    radius = iv.iv(paid)[1]
                    enclosed = tuple((part[0]-radius,part[1]+radius) for part in central)
                    totals[z] = cx.ca(totals[z],cscale(enclosed,count))
                    values[str(z)] = {'central_interval':complex_output(central),'trace_baseline_radius_exact':str(baseline),'operator_radius_exact':str(operator),'coherent_loewner_radius_exact':str(coherent),'source_ground_increment_exact':str(paid-coherent),'complete_radius_exact':str(paid),'subset_loewner_integrals_exact':components,'subset_paid_integrals_exact':paid_components,'alltime_coefficient_interval':complex_output(enclosed)}
                rows.append({'role':role,'count':count,'orientation':orientation,'order':order,'signedCAR':sign,'charged_counts':counts,'between':between,'creation_first':first,'values':values})
    assert len(rows)==96
    return {'kind':'NEW_NATIVE_ANALYTIC_WHOLE_TIME_CHARGED_FLUX_COEFFICIENT','own_sha256':sha(Path(__file__)),'heat_sha256':sha(interface_path),'coefficient_intervals':{str(z):complex_output(v) for z,v in totals.items()},'sign_verdict':{str(z):('POSITIVE' if v[0][0]>0 else 'NEGATIVE' if v[0][1]<0 else 'UNRESOLVED_CONTAINS_ZERO') for z,v in totals.items()},'error_budgets':{str(z):{k:{'exact':str(v),'display':float(v)} for k,v in budget.items()} for z,budget in budgets.items()},'heat_coefficients':{f'{role}_{first}':{'anchor_exact':str(d['a']),'floor_exact':str(d['b']),'V_interval':encode(d['V']),'V_trace_cap':str(d['vtrace']),'V_operator_cap':str(d['vop']),'G0_operator_cap':str(d['h']),'lifted_G0_schatten2_cap':str(d['hf']),'coefficient_intervals':d['coefficients'],'moment_rectangle_widths':{k:[[str(x) for x in rr] for rr in A] for k,A in d['moment_rectangle_widths'].items()}} for (role,first),d in data.items()},'native_matrices_exact':exact,'native_operator_schatten2_caps':{r:{str(j):list(map(str,n)) for j,n in ns.items()} for r,ns in norms.items()},'orders':rows,'adverse':adverse,'quadrature_error_exact':'0','omitted_time_tail_exact':'0; all tau>=0 integrated, pointwise remainder global','arithmetic':'100-bit outward rational coefficient intervals; factorial/rational-rate moments exact Fraction before outward scaling. Conditional historical correctly-rounded binary64 premise retained. Moment/source s differs from exact ground g.','retained':['all24 orders/both orientations','opposite first edges twoPH','all active1+aux63','native full4x4 coherent trace = sixteen nonidentical vertex spin blocks','signedCAR','external64 once','half affine/incoming0','whole charged containing-space floors'],'unpaid':['If interval contains0, sign remains unresolved despite operator and coherent Schatten improvement','physical1 normalized allorder root','same physical Phi/lattice stationary ground','phase/stiffness','gap closing IR atom/pole/common cone/originalSOURCE'],'elapsed_seconds':time.monotonic()-started}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = run()
    central = {str(z):CZ for z in (0,64)}
    for row in result['orders']:
        for z,value in row['values'].items():
            raw = value['central_interval']
            decoded = tuple((iv.iv(F(raw[k]['exact'][0]))[0],iv.iv(F(raw[k]['exact'][1]))[1]) for k in ('re','im'))
            central[z] = cx.ca(central[z],cscale(decoded,row['count']))
    result['central_coefficient_intervals'] = {z:complex_output(v) for z,v in central.items()}
    result['moment_coefficient_interval_halfwidths_exact'] = {z:{k:str(F(v[j][1]-v[j][0],2*iv.Q)) for j,k in enumerate(('re','im'))} for z,v in central.items()}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('kind','coefficient_intervals','sign_verdict','error_budgets','elapsed_seconds')},indent=2),flush=True)
