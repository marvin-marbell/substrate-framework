"""NEW native shared-star reduced-resolvent consumer; never executes an old CLI.

The ground fourth coefficient is enclosed, not identified with a static Gram.
All displayed Ritz values are approximations; exact rational residual prices
retain the entire H0 spectrum and the actual neutral P0 subtraction.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import three_cell_native_scientist_20261011 as native
import three_cell_metric_transfer_consumer_20261011 as formation

ROOT = Path(__file__).resolve().parent
PREFIX = 'shared_star_dynamic_'
INPUTS = [*native.PINS, 'three_cell_native_scientist_20261011.py',
          'three_cell_metric_transfer_consumer_20261011.py',
          'connected_root_exterior_scientist_20261011.py',
          'harmonic_car_bridge_scientist_20261011.py',
          'three_cell_density_formation_typed3_20261011_interface.json',
          'three_cell_reference_formation_parent_attempt4_20261011_receipt.json',
          'joined_phi_u1_three_cell_reference_dynamical_scoped_acceptance_20261011.json']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sqrt_hi(x):
    x = max(F(0), F(x)); scale = 1 << 100
    k = math.isqrt(x.numerator * scale * scale // x.denominator)
    return F(k + int(F(k*k, scale*scale) < x), scale)


def interval(c, e):
    return (F(float(c)) - e, F(float(c)) + e)


def add(x, y):
    return x[0] + y[0], x[1] + y[1]


def scale(x, c):
    return (c*x[0], c*x[1]) if c >= 0 else (c*x[1], c*x[0])


def mul(x, y):
    z = [a*b for a in x for b in y]
    return min(z), max(z)


def encode(x):
    return {'exact': [str(v) for v in x], 'display': [float(v) for v in x]}


@lru_cache(maxsize=None)
def normal(word):
    """Global signed CAR, retaining scalar contractions rather than dropping folds."""
    for i in range(len(word)-1):
        a, b = word[i:i+2]
        ka, kb = (not a[3], a[:3]), (not b[3], b[:3])
        if a == b:
            return ()
        if ka > kb:
            result = defaultdict(int)
            swapped = word[:i] + (b, a) + word[i+2:]
            for sign, w in normal(swapped):
                result[w] -= sign
            if not a[3] and b[3] and a[:3] == b[:3]:
                for sign, w in normal(word[:i]+word[i+2:]):
                    result[w] += sign
            return tuple((v, w) for w, v in result.items() if v)
    return ((1, word),)


def collect(terms):
    result = defaultdict(complex)
    for z, word, memory in terms:
        for sign, w in normal(word):
            result[w, memory] += sign*z
    return [(z, w, m) for (w, m), z in result.items() if z != 0]


def product(a, b):
    return collect([(z*q, w+v, (0, 0, 0)) for z, w in a for q, v in b])


class DynamicKernel(native.Kernel):
    def __init__(self):
        super().__init__()
        self.fields = {}
        with np.load(ROOT/native.DENSITY, allow_pickle=False) as archive:
            for roles, key in [((0,0),'active_pair'), ((1,1),'auxiliary_pair'), ((0,1),'mixed_pair')]:
                U = native.car.pair_basis(roles[0] == roles[1]); d = U.shape[1]
                for left, right in itertools.product(range(2), repeat=2):
                    raw = archive[key][left*d:(left+1)*d, right*d:(right+1)*d]
                    self.fields[roles, left, right] = U@raw.T@U.T/self.N0
        for left, right in itertools.product(range(2), repeat=2):
            self.fields[(1,0),left,right] = self.fields[(0,1),left,right].reshape(5,5,5,5).transpose(1,0,3,2).reshape(25,25)
        self.h = []
        profiles = json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_bytes())['profiles']
        for row in profiles:
            m,c,d = [float(F(row[key])) for key in ('m_exact','c0_exact','d_exact')]
            self.h.append(m*np.diag([-1,1,-1,1])+c*np.kron([[0,1],[1,0]],np.diag([1,-1]))+d*np.block([[np.zeros((2,2)),1j*np.array([[0,1],[1,0]])],[-1j*np.array([[0,1],[1,0]]),np.zeros((2,2))]]))
        self.independent = False
        self.delete_memory = False
        self.counters = defaultdict(int)

    @lru_cache(maxsize=300000)
    def local_memory(self, word, left, right):
        if not word:
            rho = self.fields[(0,0),left,right]
            return np.trace(rho)
        balance = defaultdict(int)
        for slot,p,creation in word:
            balance[slot] += 1 if creation else -1
        if any(balance.values()):
            return 0j
        slots = sorted(balance)
        if len(slots) > 2:
            raise ArithmeticError('Unsupported third-slot datum, never silently factorized')
        inv = sum(word[i][0] > word[j][0] for i in range(len(word)) for j in range(i+1,len(word)))
        ops = []
        for slot in slots:
            op = np.eye(16, dtype=complex)
            for owner,p,creation in word:
                if owner == slot:
                    op = op@(self.ann[p].T if creation else self.ann[p])
            ops.append(self.W.conj().T@op@self.W)
        roles = tuple(0 if slot < 2 else 1 for slot in slots)
        if len(slots) == 1:
            rho = np.einsum('ikjk->ij',self.fields[(roles[0],roles[0]),left,right].reshape(5,5,5,5))
            op = ops[0]
        else:
            rho = self.fields[roles,left,right]
            op = np.kron(*ops)
            if self.independent and left == right == 0:
                r0 = np.einsum('ikjk->ij',rho.reshape(5,5,5,5))
                r1 = np.einsum('ikil->kl',rho.reshape(5,5,5,5))
                rho = np.kron(r0,r1)
        return (-1)**inv*np.einsum('ij,ji->',op,rho)

    def moment_memory(self, word, left, right):
        inv = sum(word[i][0] > word[j][0] for i in range(len(word)) for j in range(i+1,len(word)))
        z = complex((-1)**inv)
        for cell in range(3):
            local = tuple((s,p,c) for owner,s,p,c in word if owner == cell)
            z *= self.local_memory(local,left[cell],right[cell])
            if z == 0:
                break
        return z

    @lru_cache(maxsize=600000)
    def local_overlap(self, left, right, lm, rm, independent):
        dagger=tuple((s,p,not creation) for s,p,creation in reversed(left))
        return self.local_memory(dagger+right,lm,rm)

    def gram_terms(self, left, right):
        # Product CELL sources have even parity. Grouping each global CAR word
        # into increasing cell order gives its exact graded-tensor sign. Never
        # factor copies WITHIN central B: local_overlap uses the joint density.
        if self.delete_memory:
            left=[t for t in left if not any(t[2])]
            right=[t for t in right if not any(t[2])]
        if not left or not right:
            return 0j
        def decompose(terms):
            coefficients=[]; keys=[[] for _ in range(3)]
            for z,word,memory in terms:
                inv=sum(word[i][0]>word[j][0] for i in range(len(word)) for j in range(i+1,len(word)))
                coefficients.append(z*(-1)**inv)
                for cell in range(3):
                    keys[cell].append((tuple((s,p,c) for owner,s,p,c in word if owner==cell),memory[cell]))
            return np.asarray(coefficients),keys
        lc,lk=decompose(left); rc,rk=decompose(right)
        value=lc.conj()[:,None]*rc[None,:]
        for cell in range(3):
            lu=list(dict.fromkeys(lk[cell])); ru=list(dict.fromkeys(rk[cell]))
            li={key:i for i,key in enumerate(lu)}; ri={key:i for i,key in enumerate(ru)}
            small=np.asarray([[self.local_overlap(lw,rw,lm,rm,self.independent)
                               for rw,rm in ru] for lw,lm in lu])
            lindex=np.asarray([li[key] for key in lk[cell]])
            rindex=np.asarray([ri[key] for key in rk[cell]])
            value*=small[lindex[:,None],rindex[None,:]]
        self.counters['Gram_term_pairs']+=len(left)*len(right)
        return np.sum(value)

    def action(self, terms, flow, u):
        """[H0,O]g+external64 O g; full Fock CAR and TOTAL density memory."""
        out = []
        for z,word,memory in terms:
            if any(memory):
                raise ArithmeticError('Only O and first action are required; no invented higher rho fields')
            change = [F(0),F(0),F(0)]
            for cell,slot,p,creation in word:
                change[cell] += (1 if creation else -1)*F(1 if p < 2 else -1,2)
            for i,(cell,slot,p,creation) in enumerate(word):
                h = self.h[0 if slot < 2 else 1]
                for q in range(4):
                    coefficient = h[q,p] if creation else -h[p,q]
                    if coefficient:
                        out.append((z*coefficient,word[:i]+((cell,slot,q,creation),)+word[i+1:],memory))
            constant = 64*sum(x*x for x in flow)
            for cell in range(3):
                delta = change[cell]+u[cell]
                constant += 64*delta*delta
                if delta:
                    mem = tuple(int(j == cell) for j in range(3))
                    out.append((z*float(128*delta),word,mem))
            if constant:
                out.append((z*float(constant),word,memory))
        return collect(out)


def assignments():
    """All partitions of FOUR physical copy labels with no singleton block."""
    result = [((0,0,0,0),1,'all_active'),((1,1,1,1),63,'all_auxiliary')]
    for pattern in ((0,0,1,1),(0,1,0,1),(0,1,1,0)):
        for roles,mult,name in [((0,1),63,'active_aux'),((1,0),63,'aux_active'),((1,2),3906,'aux_aux')]:
            result.append((tuple(roles[i] for i in pattern),mult,str(pattern)+name))
    return result


class Star:
    def __init__(self,k,edges,matrices):
        self.k=k; self.edges=edges; self.matrices=matrices
        self.cache={}
        self.max_sizes=[0,0,0]

    @lru_cache(maxsize=None)
    def root(self, edge, orientation, copy):
        e = self.edges[edge]
        return tuple(native.root_terms(e,orientation,0 if edge == 0 else 2,self.matrices[e['axis'],e['central_site'],e['outer_site']],copy))

    def flow_u(self, component):
        _,first,of,second,os = component
        flow=[0,0]; u=[F(0),F(0),F(0)]
        for edge,orientation in [(first,of),(second,os)]:
            e=self.edges[edge]; flow[edge] += orientation
            ct=1 if e['central_is_tail'] else (0 if edge == 0 else 2)
            ch=(0 if edge == 0 else 2) if e['central_is_tail'] else 1
            st=e['central_site'] if ct == 1 else e['outer_site']
            sh=e['central_site'] if ch == 1 else e['outer_site']
            u[ct] += orientation*F(-1 if st == 0 else 1,2)
            u[ch] -= orientation*F(-1 if sh == 0 else 1,2)
        return tuple(flow),tuple(u)

    def vectors(self,component,copies):
        weight,first,of,second,os=component
        key=(first,of,second,os,copies)
        if key not in self.cache:
            O=product(self.root(first,of,copies[0]),self.root(second,os,copies[1]))
            flow,u=self.flow_u(component)
            A=self.k.action(O,flow,u)
            if any(any(m) for _,_,m in A):
                raise ArithmeticError('Original external root should exactly cancel internal affine electric change')
            B=self.k.action(A,flow,u)
            self.cache[key]=(O,A,B)
            self.max_sizes=[max(x,len(v)) for x,v in zip(self.max_sizes,(O,A,B))]
        return self.cache[key]

    def moments(self,components,active_only=False):
        m=np.zeros(5,dtype=complex); budget=np.zeros(5)
        for left,right in itertools.product(components,repeat=2):
            if self.flow_u(left)[0] != self.flow_u(right)[0]:
                continue
            for copies,mult,name in assignments():
                if active_only and any(copies):
                    continue
                lv=self.vectors(left,copies[:2]); rv=self.vectors(right,copies[2:])
                weight=left[0]*right[0]*mult
                for j,(li,ri) in enumerate(((0,0),(0,1),(1,1),(1,2),(2,2))):
                    value=self.k.gram_terms(lv[li],rv[ri]); m[j] += weight*value
                    ln=sum(abs(z)*128**sum(mem) for z,w,mem in lv[li])
                    rn=sum(abs(z)*128**sum(mem) for z,w,mem in rv[ri])
                    budget[j] += abs(weight)*ln*rn
        if max(abs(m.imag)) > 1e-5*max(1,max(abs(m.real))):
            raise ArithmeticError(('Hermitian moment failed',m.tolist()))
        return m.real,budget

    def projection(self,components):
        p=0j; budget=0.
        for component in components:
            if self.flow_u(component)[0] != (0,0):
                continue
            for copies,mult in [((0,0),1),((1,1),63)]:
                O=self.vectors(component,copies)[0]
                p += component[0]*mult*self.k.gram_terms([(1.,(),(0,0,0))],O)
                budget += abs(component[0])*mult*sum(abs(z) for z,w,mem in O)
        return p.real,budget


def moment_intervals(m,budget,p,pbudget,price):
    # Price uncollapsed paths: normal ordering of four CAR factors has at most
    # four folds, dGamma commutator row-sum <256, external energy <=128.
    # Second-action electric scalar plus rho support128 contributes <100000.
    raw=max(F(1),F(float(budget[0])))*16
    safe=[raw,raw*384,raw*384**2,raw*384**2*100000,
          raw*384**2*100000**2]
    result=[interval(v,max(F(float(b)),s)*price) for v,b,s in zip(m,budget,safe)]
    pi=interval(p,F(float(pbudget))*price)
    pp=mul(pi,pi)
    result[0]=add(result[0],scale(pp,F(-1)))
    result[0]=(max(F(0),result[0][0]),result[0][1])
    if result[0][1] < 0:
        raise ArithmeticError('Ground-projected norm upper is negative')
    return result,pi


def resolvent(m,mi,gap):
    """Two-dimensional polynomial Ritz inverse; residual, not sampled tails."""
    H=np.array([[m[1],m[2]],[m[2],m[3]]])
    target=np.array([m[0],m[1]])
    try:
        c=np.linalg.solve(H,target)
    except np.linalg.LinAlgError:
        c=np.array([m[0]/m[1],0.])
    c0,c1=map(lambda z:F(float(z)),c)
    response=add(scale(mi[0],c0),scale(mi[1],c1))
    coefficients=[F(1),-2*c0,c0*c0-2*c1,2*c0*c1,c1*c1]
    residual=(F(0),F(0))
    for x,a in zip(mi,coefficients):
        residual=add(residual,scale(x,a))
    if residual[1] < 0:
        raise ArithmeticError('Negative outward residual squared')
    error=sqrt_hi(max(F(0),mi[0][1])*max(F(0),residual[1]))/gap
    enclosure=add(response,(-error,error))
    enclosure=(max(F(0),enclosure[0]),min(mi[0][1]/gap,enclosure[1]))
    if enclosure[0] > enclosure[1]:
        raise ArithmeticError('Resolvent enclosure intersection empty')
    scalar=m[0]*m[0]/m[1] if m[1] > 0 else 0.
    numeric=float(c0)*m[0]+float(c1)*m[1]
    return {'polynomial_exact':[str(c0),str(c1)],'Ritz_response_float':numeric,
            'scalar_response_float':scalar,'response':encode(enclosure),
            'residual_square':encode(residual),'response_error_exact':str(error),
            'all_spectral_tails_paid_by_residual_over_gap':True},enclosure


def run(epoch):
    started=time.monotonic(); pins={name:sha(ROOT/name) for name in INPUTS}
    for name,pin in native.PINS.items():
        if pins[name] != pin:
            raise ValueError(('Frozen input changed',name))
    boundary=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_bytes())['all64_boundary']
    _,_,L,exact_matrices=formation.derive_physical(boundary)
    matrices={}
    for axis in range(3):
        M=np.asarray([[complex(formation.sp.sympify(z)) for z in row] for row in exact_matrices[axis]])
        for s,t in itertools.product(range(2),repeat=2):
            matrices[axis,s,t]=(-1)**(s+t)*M
    inventory,rows=native.inventory(8)
    e=next(e for e in inventory if e['axis']==1 and e['central_site']==0 and e['central_is_tail'])
    f=next(e for e in inventory if e['axis']==2 and e['central_site']==0 and e['central_is_tail'])
    if e['raw_edge']==f['raw_edge'] or e['outer_cell']==f['outer_cell']:
        raise ArithmeticError('Not two distinct external edges')
    k=DynamicKernel(); star=Star(k,[e,f],matrices)
    parent=json.loads((ROOT/'three_cell_reference_formation_parent_attempt4_20261011_receipt.json').read_bytes())
    gap=F(parent['conditional_reference_gap_exact'])
    b=F(150034528033,34359738368); single_gap=64+2*b
    density=json.loads((ROOT/'three_cell_density_formation_typed3_20261011_interface.json').read_bytes())
    delta=max(F(r['delta_trace_norm_exact']) for r in density['errors'].values())
    I=F(274877906944,2204801412260314559041)
    eta3=sqrt_hi(6*I)
    residual=F(1,65536); rayleigh=F(46955575073,1538631694057472)
    # Conditional accepted gamma operation model, scaled by memory support128.
    # beta controls compressed full-CAR, pair embedding and <=2^28 reductions.
    beta=2500*formation.gamma(8192)+formation.gamma(1<<28)+128*formation.gamma(256)
    price=3*delta+beta+2*eta3
    singles=[]
    norms=[]
    for edge in range(2):
        G=T=U=0.; budget=[0.,0.,0.]
        for orientation in (-1,1):
            for copy,mult in [(0,1),(1,63)]:
                root=[(z,w,(0,0,0)) for z,w in star.root(edge,orientation,copy)]
                comp=(1.,edge,orientation,edge,-orientation)
                # Single source affine u is half of that component's first hop.
                single_component=(1.,edge,orientation,1-edge,0)
                flow,u=star.flow_u(single_component)
                A=k.action(root,flow,u)
                moments=[k.gram_terms(root,root).real,k.gram_terms(root,A).real,k.gram_terms(A,A).real]
                G+=mult*moments[0]; T+=mult*moments[1]; U+=mult*moments[2]
                for j,(left,right) in enumerate(((root,root),(root,A),(A,A))):
                    budget[j]+=mult*sum(abs(z)*128**sum(mem) for z,w,mem in left)*sum(abs(z)*128**sum(mem) for z,w,mem in right)
        gi,ti,ui=[interval(v,F(float(p))*price) for v,p in zip((G,T,U),budget)]
        alpha=F(float(T/G)); variance=add(add(ui,scale(ti,-2*alpha)),scale(gi,alpha*alpha))
        epsilon=sqrt_hi(max(F(0),variance[1]))/(alpha*single_gap)
        A0=scale(gi,1/alpha); B0=scale(gi,1/(alpha*alpha))
        ae=sqrt_hi(max(F(0),gi[1]))*epsilon
        be=2*sqrt_hi(max(F(0),gi[1]))/alpha*epsilon+epsilon*epsilon
        ai=add(A0,(-ae,ae)); bi=add(B0,(-be,be))
        ai=(max(F(0),ai[0]),ai[1]);bi=(max(F(0),bi[0]),bi[1])
        singles.append({'G':encode(gi),'T':encode(ti),'U':encode(ui),'alpha_exact':str(alpha),
                        'whole_charged_outer_resolvent_error_exact':str(epsilon),
                        'A':encode(ai),'B':encode(bi),'firstH_source_residual_price_exact':str(3*(residual+rayleigh)*sqrt_hi(gi[1])),
                        'source_moment_centers_float':[G,T,U]})
        norms.append((alpha,epsilon,gi[1],ai,bi))
    aa=[(1./float(norms[0][0]),0,o,0,-o) for o in (-1,1)]
    bb=[(1./float(norms[1][0]),1,o,1,-o) for o in (-1,1)]
    mixed=[]
    for oa,ob in itertools.product((-1,1),repeat=2):
        mixed += [(1./float(norms[1][0]),0,oa,1,ob),(1./float(norms[0][0]),1,ob,0,oa)]
    groups={'neutral_plus':aa+bb,'neutral_minus':aa+[(-w,a,o,b,p) for w,a,o,b,p in bb],'mixed':mixed}
    outputs={}; enclosures={}; source_bounds={}
    for name,components in groups.items():
        m,budget=star.moments(components); p,pbudget=star.projection(components)
        mi,pi=moment_intervals(m,budget,p,pbudget,price)
        projected=m.copy();projected[0]-=p*p
        result,enclosure=resolvent(projected,mi,gap)
        outputs[name]={'moments_float':m.tolist(),'moment_intervals':[encode(x) for x in mi],
                       'P0_amplitude':encode(pi),'P0_subtracted_norm_float':projected[0],
                       'L1_action_budgets_float':budget.tolist(),'resolvent':result}
        enclosures[name]=enclosure;source_bounds[name]=sqrt_hi(mi[0][1])
        print(json.dumps({'source':name,'Ritz_response':result['Ritz_response_float'],'interval':result['response']['display'],'max_action_terms':star.max_sizes}),flush=True)
    # 2Re <Qyaa,R Qybb> = (response_plus-response_minus)/2.
    chain=add(scale(add(enclosures['neutral_plus'],scale(enclosures['neutral_minus'],F(-1))),F(1,2)),enclosures['mixed'])
    Ka=[]
    for edge in (e,f):
        Ka.append(2*sum(F(r['count'])*F(next(p['fullCAR_hop_norm_upper_exact'] for p in r['paths'] if p['axis']==edge['axis'] and p['length']==1)) for r in boundary['roles']))
    da=Ka[0]*norms[0][1];db=Ka[1]*norms[1][1]
    dm=Ka[0]*norms[1][1]+Ka[1]*norms[0][1]
    # The exact neutral sources are bounded individually by half the +/- sums.
    na=nb=(source_bounds['neutral_plus']+source_bounds['neutral_minus'])/2
    outer_error=(2*(da*nb+db*na+da*db)+dm*(2*source_bounds['mixed']+dm))/gap
    chain=add(chain,(-outer_error,outer_error))
    fold=add(mul(norms[0][3],norms[1][4]),mul(norms[1][3],norms[0][4]))
    connected=add(chain,scale(fold,F(-1)))
    numeric_chain=(outputs['neutral_plus']['resolvent']['Ritz_response_float']-outputs['neutral_minus']['resolvent']['Ritz_response_float'])/2+outputs['mixed']['resolvent']['Ritz_response_float']
    numeric_fold=singles[0]['source_moment_centers_float'][0]*singles[1]['source_moment_centers_float'][0]*(1/float(norms[0][0])/float(norms[1][0])**2+1/float(norms[1][0])/float(norms[0][0])**2)
    # Discriminating dynamic controls on the SAME material central fourpoint.
    control_components=mixed
    baseline=outputs['mixed']['moments_float']
    # Independent global signed-word path checks the optimized CELL grouping
    # on actual native O/A/B, including a density-memory B sample.
    sample=star.vectors(mixed[0],(0,1))
    grouping_controls=[]
    for order,terms in enumerate(sample):
        selected=terms[:12]+[t for t in terms if any(t[2])][:12]
        direct=0j
        for z,w,lm in selected:
            dagger=tuple((c,s,p,not cr) for c,s,p,cr in reversed(w))
            for q,v,rm in selected:
                direct+=z.conjugate()*q*k.moment_memory(dagger+v,lm,rm)
        fast=k.gram_terms(selected,selected)
        grouping_controls.append({'action_order':order,'direct_float':[direct.real,direct.imag],
                                  'vectorized_float':[fast.real,fast.imag],'absolute_difference':abs(direct-fast)})
        if abs(direct-fast)>1e-8*max(1,abs(direct)):
            raise ArithmeticError('Independent signed CAR grouping control rejected')
    k.independent=True;k.local_memory.cache_clear()
    independent,_=star.moments(control_components)
    k.independent=False;k.local_memory.cache_clear()
    k.delete_memory=True
    omitted,_=star.moments(control_components)
    k.delete_memory=False
    active_only,_=star.moments(control_components,active_only=True)
    wrong_flow=[sum(math.comb(j,i)*128**(j-i)*baseline[i] for i in range(j+1)) for j in range(5)]
    controls={'independent_copy_moments_float':independent.tolist(),
              'independent_signed_CAR_grouping_controls':grouping_controls,
              'independent_copy_difference_float':(independent-np.array(baseline)).tolist(),
              'omitted_collective_density_moments_float':omitted.tolist(),
              'omitted_collective_density_difference_float':(omitted-np.array(baseline)).tolist(),
              'omitted_auxiliary_role_moments_float':active_only.tolist(),
              'omitted_auxiliary_role_difference_float':(active_only-np.array(baseline)).tolist(),
              'repeated_edge_wrong_dynamic_moments_float':wrong_flow,
              'repeated_edge_wrong_dynamic_difference_float':(np.array(wrong_flow)-np.array(baseline)).tolist(),
              'P0_not_removed_wrong_neutral_plus_norm_float':outputs['neutral_plus']['moments_float'][0],
              'P0_correct_neutral_plus_norm_float':outputs['neutral_plus']['P0_subtracted_norm_float'],
              'correct_external_two_edge_energy':128,'repeated_edge_wrong_energy':256,
              'role_partitions':assignments(),
              'correct_physical_copy_total':64,'wrong_drop_auxiliary_copy_total':1,
              'copy_singleton_Gram_exactly_zero_by_outer_fixed_labelled_number':True,
              'z11_exact':0,'z11_reason':'One A and one C hop leaves two distinct nonzero integer external flows and changes outer labelled numbers; overlap with g is zero.',
              'omit_central_neutral_wrong_chain_float':outputs['mixed']['resolvent']['Ritz_response_float'],
              'full_chain_float':numeric_chain}
    q=2*(Ka[0]+Ka[1])/gap
    radius=gap/(2*(Ka[0]+Ka[1]))
    result={'kind':'NEW_FULL_H0_SHARED_STAR_CONNECTED_REDUCED_RESOLVENT_OUTWARD_ATTEMPT',
            'governing_plan':{'id':'issue-222-induced-photon-delta13-continuation-20261008','captured_revision':'09fba2e1e86dc1f15f08e797760118b6cd91aa01c1f8567cba541f706e11afa1','template':'general','template_revision':'89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe','work_item':'induced-photon'},
            'status':'COMPLETED_PARENT_REVIEW_PENDING_FAIL_TO_CERTIFY_PHYSICAL1',
            'edges':[e,f],'N':8,'selected_graph_only_not_entire_lattice_perturbation':True,
            'normalization':'R=Q(H0-Eg)^-1Q, Q=1-|g><g|; coefficient lim_T (logZ)_22/T=-E22=chain-fold; ordinary power-series coefficients, mixed derivative is2!2! times coefficient.',
            'sources':outputs,'single_roots':singles,'reference_gap_exact':str(gap),'single_flow_floor_exact':str(single_gap),
            'meaningful_resolvent_time_scale_exact':str(1/gap),'whole_time_integration':'Reduced resolvent integrates the entire t>=0 semigroup; no finite-time cube, tiny-T or quadrature substitution. Residual/gap pays entire spectral tail.',
            'chain':encode(chain),'normalization_fold':encode(fold),'connected_per_time_coefficient':encode(connected),
            'Ritz_connected_float':numeric_chain-numeric_fold,'Ritz_chain_float':numeric_chain,'Ritz_normalization_fold_float':numeric_fold,
            'outer_approximation_price_exact':str(outer_error),'controls':controls,
            'formation_and_transfer':{'conditional_binary64_operation_model':True,'density_trace_price_exact':str(delta),'new_action_and_contraction_beta_exact':str(beta),'threecell_ground_distance_upper_exact':str(eta3),'combined_relative_L1_expectation_price_exact':str(price),
                'source_norm_exact':density['source_norm_squared_exact'],'represented_cache_N0_exact':density['common_N0_exact'],
                'residual_exact':str(residual),'residual_formation_included_exact':'1/262144','Rayleigh_shift_exact':str(rayleigh),
                'memory_domain':'GR/RR total-rho archived field rows on complete two-slot extraction; fields widened by128^(left+right) times field0 formation plus rho-multiplication gamma256. Source-to-g uses entire bounded CAR/rho polynomial norms, not an unpriced K^2 residual.',
                'moment_meaning':'Exact-g K moments enclosed through [H,O]g and [H,[H,O]]g identities evaluated on s; archived s not renamed g. These are NOT asserted to be K^j s moments.',
                'full_action':'Every neutral, scalar6/exterior, ground fold and collective rho spectator is retained. No PHP squared, no cropped source-image H, no independent charged-factor substitution.',
                'unsupported_machine_attestation':True},
            'normalized_rooted_physical1_attempt':{'edge_operator_norm_upper_exact':[str(x) for x in Ka],
                'gap_guard_2sumV_over_gap_exact':str(q),'physical1_passes':q<1,
                'certified_selected_pair_analytic_radius_upper_attempt_exact':str(radius),
                'criterion':'2|lambda| (||Va||+||Vb||)<gap gives isolated finite-reference eigenbranch and its normalized coefficient expansion, NOT a spatial allorders polymer theorem.',
                'failed_bound_not_no_go':True,'allorder_linked_normalization_unpaid':True},
            'input_sha256':pins,'own_source_sha256':sha(Path(__file__)),'natural_main_finished':True,
            'max_source_first_second_action_terms':star.max_sizes,'operation_counters':dict(k.counters),'elapsed_seconds':time.monotonic()-started,
            'nonclaims':['finite_pair_not_allorders','no_physicalPhi','no_lattice_ground','no_phase_or_stiffness','no_IRatom_pole_commoncone_originalSOURCE','matching_gap_preserving_continuation_excludes_masslessness']}
    if any(sha(ROOT/name)!=pin for name,pin in pins.items()):
        raise ValueError('Input epoch changed during computation')
    path=ROOT/f'{PREFIX}{epoch}_receipt.json'
    with path.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'receipt':path.name,'connected_interval':result['connected_per_time_coefficient']['display'],'connected_Ritz_float':result['Ritz_connected_float'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',required=True)
    run(parser.parse_args().epoch)
