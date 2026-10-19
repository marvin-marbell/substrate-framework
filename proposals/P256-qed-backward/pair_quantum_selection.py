#!/usr/bin/env python3
"""Exact invariant blocks and outward-rounded global pair harmonic selection.

No grid or optimizer is used as proof. The full finite-Lambda quantum ground
is not numerically computed. See the companion proof for asymptotic scope.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast
if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[variable]='1'
import mpmath
# The interval SDK constructs arithmetic operators dynamically, beyond its stubs.
mp: Any = mpmath
import numpy as np
import sympy as sp
from scipy.linalg import expm, null_space
from full_band_gauss_section import FullBandGaussSection
from full_band_packet_bound import packet_point_controls
from pair_global_field_bound import coefficients
from pair_minimizer_moduli import (canonical_fields, full_tangents,
    principal_spectrum, projector_metric, stabilizer_vacuum_control,
    source_hashes as inherited_hashes)
from pair_separated_charge_transfer import full_field_angular_control
from rotation_band import RotationBand
from unrestricted_field_minimum import (ALIGNMENT, BARE, STIFFNESS, Functional,
    coordinates, check, local_potential)
import pair_quantum_selection_probe_symbolic as algebra
import pair_quantum_selection_probe_partition as partition
ROOT=Path(__file__).resolve().parent

class Jet:
    """Value, first derivative and second derivative, all interval-enclosed."""
    def __init__(self,v,d=0,dd=0):
        self.v=v if hasattr(v,'_mpi_') else mp.iv.mpf(v)
        self.d=d if hasattr(d,'_mpi_') else mp.iv.mpf(d)
        self.dd=dd if hasattr(dd,'_mpi_') else mp.iv.mpf(dd)
    @staticmethod
    def lift(x): return x if isinstance(x,Jet) else Jet(x)
    def __add__(self,x):
        x=self.lift(x); return Jet(self.v+x.v,self.d+x.d,self.dd+x.dd)
    __radd__=__add__
    def __neg__(self): return Jet(-self.v,-self.d,-self.dd)
    def __sub__(self,x): return self+-self.lift(x)
    def __rsub__(self,x): return self.lift(x)+-self
    def __mul__(self,x):
        x=self.lift(x)
        return Jet(self.v*x.v,self.d*x.v+self.v*x.d,
                   self.dd*x.v+2*self.d*x.d+self.v*x.dd)
    __rmul__=__mul__
    def reciprocal(self):
        return Jet(1/self.v,-self.d/self.v**2,
                   2*self.d**2/self.v**3-self.dd/self.v**2)
    def __truediv__(self,x): return self*self.lift(x).reciprocal()
    def __rtruediv__(self,x): return self.lift(x)*self.reciprocal()
    def __pow__(self,n):
        if n<0: return (self.reciprocal())**(-n)
        if n==0: return Jet(1)
        if n==1: return self
        return Jet(self.v**n,n*self.v**(n-1)*self.d,
            n*self.v**(n-1)*self.dd+n*(n-1)*self.v**(n-2)*self.d**2)
    def sqrt(self):
        root=mp.iv.sqrt(self.v)
        return Jet(root,self.d/(2*root),self.dd/(2*root)-self.d**2/(4*root**3))
    def sin(self):
        return Jet(mp.iv.sin(self.v),mp.iv.cos(self.v)*self.d,
                   mp.iv.cos(self.v)*self.dd-mp.iv.sin(self.v)*self.d**2)
    def cos(self):
        return Jet(mp.iv.cos(self.v),-mp.iv.sin(self.v)*self.d,
                   -mp.iv.sin(self.v)*self.dd-mp.iv.cos(self.v)*self.d**2)

def ends(x):
    return mp.mpf(x._mpi_[0]),mp.mpf(x._mpi_[1])

def enclosure(x):
    """Serialize dyadic endpoints with EXACT outward decimal rounding."""
    scale=10**30
    result=[]
    for index,(sign,mantissa,exponent,_) in enumerate(x._mpi_):
        numerator=(-1 if sign else 1)*mantissa*scale
        denominator=1
        if exponent>=0: numerator<<=exponent
        else: denominator<<=-exponent
        integer=numerator//denominator if index==0 else -((-numerator)//denominator)
        prefix='-' if integer<0 else ''
        whole,fraction=divmod(abs(integer),scale)
        result.append(f'{prefix}{whole}.{fraction:030d}')
    return result

def positive(x,label):
    if ends(x)[0]<=0: raise ArithmeticError(f'Interval positivity unresolved: {label} {enclosure(x)}')

def expression_function(expr):
    """Compile exact rational algebra without conversion through binary floats."""
    variables=(algebra.delta,algebra.a1,algebra.a2,algebra.y1,algebra.y2,algebra.R)
    def build(node):
        if node in variables: return lambda args,index=variables.index(node):args[index]
        if node.is_Integer: return lambda args,n=int(node):Jet(n)
        if node.is_Rational:
            return lambda args,p=int(node.p),q=int(node.q):Jet(p)/q
        if node.is_Add:
            children=[build(x) for x in node.args]
            return lambda args:sum((f(args) for f in children),Jet(0))
        if node.is_Mul:
            children=[build(x) for x in node.args]
            def product(args):
                result=Jet(1)
                for f in children: result=result*f(args)
                return result
            return product
        if node.is_Pow and node.exp.is_Integer:
            child=build(node.base); power=int(node.exp)
            return lambda args:child(args)**power
        raise TypeError(f'Unsupported exact-expression node {node}')
    return build(expr)

def rational(value):
    """An exact source rational as an outward interval constant."""
    return Jet(int(value.p))/int(value.q)

def rank_coefficients(matrix):
    power=sp.eye(matrix.rows); traces=[]; result=[sp.Integer(1)]
    for degree in range(1,matrix.rows+1):
        power=(power*matrix).applyfunc(algebra.norm)
        traces.append(algebra.norm(sp.trace(power)))
        result.append(algebra.norm(sum((-1)**(i-1)*result[degree-i]*traces[i-1]
                                    for i in range(1,degree+1))/degree))
    return result

def derive_blocks():
    ranks=(2,3,2,2,2,2,3,4,2,1,2)
    multiplicities=(1,1,2,2,2,2,1,1,1,1,1)
    rows=[]
    for group,rank,mult in zip(algebra.groups,ranks,multiplicities):
        matrix=sp.Matrix([[algebra.entry(n,m) for m in group] for n in group])
        es=rank_coefficients(matrix)
        if any(e!=0 for e in es[rank+1:]):
            raise AssertionError(f'Exact characteristic null factor failed: {group}')
        if rank==4:
            for root in (2*algebra.J,2*algebra.J+2*algebra.t**2/algebra.R):
                polynomial=sum((-1)**i*es[i]*root**(4-i) for i in range(5))
                if algebra.norm(polynomial)!=0:
                    raise AssertionError('Exact constant-rank4 factor identification failed')
        rows.append(dict(names=group,rank=rank,multiplicity=mult,matrix=matrix,
            coefficients=es[1:rank+1],functions=[expression_function(e) for e in es[1:rank+1]],
            variable=any(e.has(algebra.delta,algebra.a1,algebra.a2,algebra.y1,algebra.y2)
                         for e in es[1:rank+1])))
    return rows

def cubic_trace(coefficients):
    """Positive root t=sum sqrt(lambda) of (t²-e1)²=4e2+8sqrt(e3)t.

    Strict convexity and the unique crossing above sqrt(e1), with outward
    interval bisection, enclose the root. Implicit differentiation encloses
    both derivatives without diagonalization or repeated-root divisions.
    """
    e1,e2,e3=coefficients
    for i,e in enumerate(coefficients): positive(e.v,f'cubic e{i+1}')
    g=e3.sqrt()
    def phi(t): return (t*t-e1.v)**2-4*e2.v-8*g.v*t
    lo=ends(mp.iv.sqrt(e1.v))[0]
    hi=ends(mp.iv.sqrt(3*e1.v))[1]
    threshold=ends(mp.iv.sqrt(e1.v))[1]
    # Each bound is certified by the appropriate outward interval sign.
    left,right=lo,hi
    for _ in range(150):
        mid=(left+right)/2
        a,b=ends(phi(mp.iv.mpf(mid)))
        if b<0: left=mid
        elif a>0 and mid>=threshold: right=mid
        else: break
    # Parameter ranges can straddle phi(mid)=0. Retain both proved brackets.
    root=mp.iv.mpf([left,right])
    denominator=4*root*(root**2-e1.v)-8*g.v
    positive(denominator,'cubic trace implicit denominator')
    first=(2*(root**2-e1.v)*e1.d+4*e2.d+8*g.d*root)/denominator
    # Differentiate phi(t,x)=0; the mixed partial includes the derivative of g.
    second=(2*(root**2-e1.v)*e1.dd-2*e1.d**2+4*e2.dd+8*g.dd*root
            +(8*root*e1.d+16*g.d)*first
            -(12*root**2-4*e1.v)*first**2)/denominator
    return Jet(root,first,second)

def trace_root(es):
    if len(es)==1:
        positive(es[0].v,'rank-one eigenvalue'); return es[0].sqrt()
    if len(es)==2:
        positive(es[1].v,'rank-two positive product')
        return (es[0]+2*es[1].sqrt()).sqrt()
    if len(es)==3: return cubic_trace(es)
    # Only the exactly constant rank4 block reaches here. Two factors are
    # 2J and 2J+2kappa*v; exact factor verification is performed separately.
    if any(ends(e.d)!=(0,0) or ends(e.dd)!=(0,0) for e in es):
        raise AssertionError('A variable rank4 block requires another trace formula')
    R=radius()
    first=rational(2*algebra.J); second=first+rational(2*algebra.t**2)/R
    rest_sum=es[0]-first-second
    rest_product=es[3]/(first*second)
    positive(rest_product.v,'constant rank4 residual product')
    return first.sqrt()+second.sqrt()+(rest_sum+2*rest_product.sqrt()).sqrt()

def radius():
    alpha_exact=cast(sp.Rational,algebra.alpha)
    alpha=mp.iv.mpf(int(alpha_exact.p))/int(alpha_exact.q)
    return Jet(mp.iv.exp(mp.iv.ln(alpha)/3))

def arguments(x):
    theta=x*(mp.iv.pi/4)
    q=rational(algebra.q)
    delta=q*(2*theta).cos()
    yy=(2*q).sqrt()
    y1,y2=yy*theta.sin(),yy*theta.cos()
    a1=(1-q+delta).sqrt(); a2=(1-q-delta).sqrt()
    return delta,a1,a2,y1,y2,radius()

def coefficient_jet(rows,x,*,variable_only=False):
    args=arguments(x); total=Jet(0)
    for row in rows:
        if variable_only and not row['variable']: continue
        es=[f(args) for f in row['functions']]
        total+=row['multiplicity']*trace_root(es)/2
    if not variable_only:
        R=args[-1]; hu=rational(2*algebra.c**2)/R; hv=rational(2*algebra.t**2)/R
        electric=rational(2*algebra.J)
        # Active constant singles: eleven anti-spatial and one symmetric.
        total+=(11*(hv+electric).sqrt()+hv.sqrt())/2
        # COMPLETE unused60: 26 quartic soft zeros and34 positive modes.
        total+=hu.sqrt()+hv.sqrt()+(hu+electric).sqrt()+(hv+electric).sqrt()+13*electric.sqrt()
    return total

def interval_certificate(rows,max_depth=22):
    """Cover the entire closed x interval; certify endpoint curvature and
    interior derivative. A point evaluation alone can never accept a cell.
    """
    accepted=[]; pending=[(mp.mpf(0),mp.mpf(1),0)]
    while pending:
        lo,hi,depth=pending.pop()
        x=Jet(mp.iv.mpf([lo,hi]),1)
        try:
            jet=coefficient_jet(rows,x,variable_only=True)
            first,second=ends(jet.d),ends(jet.dd)
            kind=None
            if lo==0 and second[0]>0: kind='endpoint-positive-curvature'
            elif hi==1 and second[1]<0: kind='Clifford-negative-curvature'
            elif first[0]>0: kind='positive-first-derivative'
            if kind:
                accepted.append(dict(x=[str(lo),str(hi)],proof=kind,
                    first_derivative=enclosure(jet.d),second_derivative=enclosure(jet.dd)))
                continue
        except ArithmeticError:
            pass
        if depth>=max_depth:
            raise ArithmeticError(f'Global certificate unresolved on [{lo},{hi}] at depth {depth}')
        mid=(lo+hi)/2
        pending.extend(((mid,hi,depth+1),(lo,mid,depth+1)))
    accepted.sort(key=lambda row:mp.mpf(row['x'][0]))
    boundary=mp.mpf(0)
    for row in accepted:
        if mp.mpf(row['x'][0])!=boundary: raise AssertionError('Interval-cover gap')
        boundary=mp.mpf(row['x'][1])
    if boundary!=1: raise AssertionError('Interval-cover missing endpoint')
    # Positive PSD roots satisfy lambda_min >= e_rank/e1^(rank-1).
    # This gives a continuum normal-gap certificate without root isolation.
    hessian_gap=mp.inf
    for cell in accepted:
        args=arguments(Jet(mp.iv.mpf([mp.mpf(v) for v in cell['x']])))
        for row in rows:
            es=[f(args) for f in row['functions']]
            positive(es[0].v,'Block trace in uniform normal-gap bound')
            lower=ends((es[-1]/es[0]**(row['rank']-1)).v)[0]
            if lower<=0: raise ArithmeticError('A positive block rank is not interval-certified')
            hessian_gap=min(hessian_gap,lower)
    R=radius()
    for constant in (rational(2*algebra.c**2)/R,rational(2*algebra.t**2)/R,rational(2*algebra.J)):
        hessian_gap=min(hessian_gap,ends(constant.v)[0])
    return dict(method='Outward-rounded interval automatic differentiation of exact invariant characteristic factors; full closed interval cover',
        arithmetic=f'mpmath interval {mp.iv.dps} decimal digits',
        maximum_subdivision_depth=max_depth,
        minimum_positive_Hessian_eigenvalue_certified_lower=enclosure(mp.iv.mpf(hessian_gap))[0],
        coordinate='x=4theta/pi in [0,1], delta=q cos(2theta); coordinate is only for a scalar proof, not a measure or kinetic operator',
        cells=accepted,cell_count=len(accepted),
        exact_endpoint_derivatives_zero='Even extension at theta0 exchanges the y-sign cross blocks; at theta pi/4 the complete coefficient is even in delta',
        strict_global_minimizer='delta=q, uniquely in the COMPLETE classified quotient',
        all_domain_monotonicity_certified=True)

def analytic_controls():
    """Rigorous scalar enclosures for global tails and all-minimum fibres."""
    R=radius(); c,d,t,k,J=map(rational,(algebra.c,algebra.d,algebra.t,algebra.k,algebra.J))
    q=rational(algebra.q); gap=2*R*(1-d/c)
    local_gap=2*R/c*(1-2*q).sqrt()
    weight=((1+(1-2*q).sqrt())/2)**8
    derivative_sum=30*(4*c**2+2*d**2+2*t**2)
    a=k/16
    b=4*rational(max(algebra.c**2+algebra.d**2,algebra.t**2)).sqrt()
    candidate=(2*b/a).v
    r0=Jet(mp.iv.exp(mp.iv.ln(candidate)/3))
    if ends(r0.v)[0]<1: r0=Jet(mp.iv.mpf([1,max(1,ends(r0.v)[1])]))
    return dict(arithmetic='Outward-rounded intervals, exact actual-native source inputs',
        radius=enclosure(R.v),q=enclosure(q.v),
        heavy_gap_all_minima_bound_expression=enclosure(gap.v),
        local_Phi0_gap_all_minima_bound_expression=enclosure(local_gap.v),
        paired_projection_weight_all_minima_bound_expression=enclosure(weight.v),
        one_species_field_derivative_HS_square_sum=enclosure(derivative_sum.v),
        paired_quantum_metric_Born_Huang_all_minima_upper_expression=enclosure((derivative_sum/gap**2).v),
        paired_quantum_metric_Born_Huang_half_gap_tube_upper_expression=enclosure((4*derivative_sum/gap**2).v),
        global_field_tail=dict(a=enclosure(a.v),b=enclosure(b.v),r0=enclosure(r0.v),
            light_B0=enclosure((b*r0).v),light_B1=enclosure((2*b/a).v),
            inequalities='F>=a*rho^4-b*rho; ||HL||<=b*rho; rho<=r0+(2/a)*(F+3R), uniformly over all fields/U/T2',
            scope='Exact inequalities; interval expressions enclose their constants. No global light gap is used.'))

def native_controls(rows,ue4):
    band=RotationBand(**BARE); model=FullBandGaussSection(band,heavy_scale=1.)
    owner=Functional('pair',band,model,model.basis)
    paulis=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    basis=np.array([np.kron(paulis[i],paulis[j]) for i in range(4) for j in range(4) if i+j])
    custom=Functional('pair',band,model,basis)
    idx=np.array([site*60+mu*15+a for site in range(2) for mu in range(2) for a in range(15)])
    change=np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2),np.eye(30))
    name_to_index={name:i for i,name in enumerate(algebra.names)}
    controls=[]
    for fraction in (0.,.013,.37,.999,1.):
        fields,R,q=canonical_fields(model,fraction)
        H=owner.hessian(coordinates(fields,model.basis).reshape(-1))
        derived=custom.hessian(coordinates(fields,basis).reshape(-1))
        active=change@derived[np.ix_(idx,idx)]@change.T
        aa=np.sqrt([1-q+q*fraction,1-q-q*fraction]); yy=np.sqrt([q-q*fraction,q+q*fraction])
        substitutions={algebra.delta:q*fraction,algebra.a1:aa[0],algebra.a2:aa[1],algebra.y1:yy[0],algebra.y2:yy[1],algebra.R:R}
        residual=0.; characteristic=0.
        for row in rows:
            indices=[name_to_index[name] for name in row['names']]
            actual=active[np.ix_(indices,indices)]
            exact=np.array(row['matrix'].subs(substitutions).evalf(),dtype=float)
            residual=max(residual,float(np.linalg.norm(actual-exact)))
            values=np.linalg.eigvalsh(actual)
            expected=[float(e.subs(substitutions)) for e in row['coefficients']]
            actual_poly=np.poly(values)
            characteristic=max(characteristic,max(abs(actual_poly[i+1]-(-1)**(i+1)*e) for i,e in enumerate(expected)))
        check(residual,'Derived invariant blocks versus actual active60 Hessian',2e-12)
        check(characteristic,'Exact characteristic factors versus native spectrum',3e-12)
        positive_values=np.linalg.eigvalsh(H); positive_values=positive_values[positive_values>2e-7]
        native=float(np.sqrt(positive_values).sum()/2)
        x=mp.acos(mp.mpf(str(fraction)))/mp.pi*2
        exact=coefficient_jet(rows,Jet(x))
        distance=max(abs(native-float(v)) for v in ends(exact.v))
        check(distance,'Exact whole120 trace versus native Cartesian coefficient',2e-11)
        row_out: dict[str,Any]=dict(fraction=fraction,derived_block_residual=residual,
            characteristic_residual=characteristic,full120_positive_count=len(positive_values),
            native_C0=native,exact_C0_interval=enclosure(exact.v),trace_residual=distance)
        section=model.state(fields,np.eye(4),(0.,0.))
        predicted_weight=float(np.prod(((1+aa)/2)**4))
        check(abs(section.weight-predicted_weight),'Native full4900 paired projection weight formula',2e-13)
        row_out['native_paired_projection_weight']=section.weight
        row_out['exact_principal_angle_weight']=predicted_weight
        if fraction in (0.,.37,1.):
            pullback,gauges,D=full_tangents(model,fields)
            row_out['actual_full135_principal']=principal_spectrum(H,pullback,gauges,D,Lambda=1e12,ue4=ue4)
            row_out['actual_native_Gauss']=packet_point_controls(model,fields,np.eye(4),(0.,0.))
            row_out['actual_field_angular_Gauss']=full_field_angular_control(model,fields,np.eye(4),(0.,0.))
            row_out['isotropy_paired_vacuum']=stabilizer_vacuum_control(model,fields,H,pullback,D)
            row_out['actual_Born_Huang']=projector_metric(model,fields)
        controls.append(row_out)
    fields,R,q=canonical_fields(model,1.)
    H=owner.hessian(coordinates(fields,model.basis).reshape(-1))
    active_coords=coordinates(fields[0,:2],model.basis)
    soft_basis=null_space(active_coords,rcond=1e-10)
    soft_rows=[]
    for configuration in ('one-component','two-orthogonal','two-parallel'):
        vectors=np.zeros((2,15))
        if configuration=='one-component': vectors[0]=soft_basis[:,0]
        elif configuration=='two-orthogonal': vectors=soft_basis[:,:2].T/np.sqrt(2)
        else: vectors[:]=soft_basis[:,0]/np.sqrt(2)
        direction=np.zeros((2,4,15));direction[:,2:]=vectors/np.sqrt(2)
        check(abs(np.linalg.norm(direction)-1),'Actual unitfull120 soft coordinate')
        check(np.linalg.norm(H@direction.reshape(-1)),'Entire coupled soft26 Hessian kernel')
        quartic=STIFFNESS/4*float(np.sum((vectors@vectors.T)**2))
        center=local_potential(fields,((0,1),),(np.eye(4),))[0]
        steps=[]
        for step in (.03,.01):
            moved=fields+step*owner.fields(direction.reshape(-1))
            elastic=local_potential(moved,((0,1),),(np.eye(4),))[0]
            check(abs(elastic-center-quartic*step**4),'Native coupled soft26 exact quartic',2e-15)
            steps.append(dict(step=step,actual_elastic_excess=elastic-center,exact_quartic_excess=quartic*step**4))
        soft_rows.append(dict(configuration=configuration,unitfull120_quartic_coefficient=quartic,steps=steps))
    actual_one=soft_rows[0]['steps'][0]['actual_elastic_excess']
    wrong=STIFFNESS*.03**4
    if abs(wrong-actual_one)<1e-6:
        raise AssertionError('Per-site/full120 normalization adverse control did not discriminate')
    wrong_model=FullBandGaussSection(RotationBand(a=1.,m=float(algebra.c),r=0.,v=.3,improved=False),heavy_scale=1.)
    wrong_functional=Functional('pair',wrong_model.bands[0],wrong_model,wrong_model.basis)
    wrong_hessian=wrong_functional.hessian(coordinates(fields,wrong_model.basis).reshape(-1))
    omitted_d_defect=float(np.linalg.norm(H-wrong_hessian))
    if omitted_d_defect<1e-3:
        raise AssertionError('Dropping the commuting d term must change the actual Hessian')
    gl=expm(1j*np.einsum('a,aij->ij',.31*np.sin(np.arange(15)*.71+.3),model.generators))
    gr=expm(1j*np.einsum('a,aij->ij',.23*np.cos(np.arange(15)*.53+.2),model.generators))
    moved=fields.copy();moved[0]=gl@fields[0]@gl.conj().T;moved[1]=gr@fields[1]@gr.conj().T
    noncentral=dict(actual_native_Gauss=packet_point_controls(model,moved,gl@gr.conj().T,(.137,-.081)),
        actual_field_angular_Gauss=full_field_angular_control(model,moved,gl@gr.conj().T,(.137,-.081)),
        link_noncentral_defect=float(np.linalg.norm(gl@gr.conj().T-np.trace(gl@gr.conj().T)/4*np.eye(4))))
    c,d,t=coefficients(model)
    if tuple(map(sp.Rational,(c,d,t)))!=(algebra.c,algebra.d,algebra.t):
        raise RuntimeError('Native source coefficients differ from the exact dyadic certificate')
    derivative_hs_sum=float(np.sum(abs(owner.directions)**2))
    gap=2*R*(1-abs(d)/c)
    minimum_weight=((1+np.sqrt(1-2*q))/2)**8
    uniform=dict(heavy_gap_on_all_minima_lower=gap,
        local_Phi0_positive_negative_gap_lower=2*R/c*np.sqrt(1-2*q),
        paired_native_projection_weight_on_all_minima_lower=minimum_weight,
        one_species_field_derivative_HS_square_sum=derivative_hs_sum,
        paired_field_Born_Huang_on_minima_upper=derivative_hs_sum/gap**2,
        uniform_small_tube_field_Born_Huang_upper=4*derivative_hs_sum/gap**2,
        scope='Analytic compact-minimum bounds; the tube bound assumes both heavy/light gaps >= half the displayed gap. No global light gap on all T2 is asserted or required by the variational lower bound.')
    uniform['numeric_status']='Floating evaluation of analytic expressions, not an interval certificate; rigorous scalar enclosures are in analytic_uniform_controls.'
    return dict(rows=controls,soft26=soft_rows,
        adverse=dict(per_site_soft_quartic_wrong_excess=wrong,actual_unitfull120_excess=actual_one,
            dropped_commuting_d_term_Hessian_defect=omitted_d_defect),
        noncentral_native=noncentral,uniform_fibre_controls=uniform)

def run(precision=45,max_depth=22,ue4=.3,native=True) -> dict[str,Any]:
    if precision<30 or max_depth<4 or not np.isfinite(ue4) or ue4<=0:
        raise ValueError('Require precision>=30, max-depth>=4 and positive finite native UE4')
    mp.mp.dps=precision+15; mp.iv.dps=precision
    bindings=inherited_hashes()
    for name in ('pair_quantum_selection.py','pair_quantum_selection.md',
                 'pair_quantum_selection_probe_symbolic.py','pair_quantum_selection_probe_partition.py'):
        bindings[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    rows=derive_blocks()
    exact_partition=partition.exact_partition()
    cert=interval_certificate(rows,max_depth)
    cliff=coefficient_jet(rows,Jet(1)); endpoint=coefficient_jet(rows,Jet(0))
    gap=cliff.v-endpoint.v; positive(gap,'Strict endpoint advantage')
    native_rows=native_controls(rows,ue4) if native else None
    frozen=inherited_hashes()
    for name in ('pair_quantum_selection.py','pair_quantum_selection.md',
                 'pair_quantum_selection_probe_symbolic.py','pair_quantum_selection_probe_partition.py'):
        frozen[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if frozen!=bindings: raise RuntimeError('Owner/source/proof changed during execution')
    return dict(source_sha256=bindings,source_bindings_unchanged=True,
        algorithm_inputs=dict(precision=precision,max_depth=max_depth,ue4=ue4,native_controls_enabled=native,
            mpmath_version=mp.__version__,sympy_version=sp.__version__),
        analytic_uniform_controls=analytic_controls(),
        exact_source_coefficients={name:str(value) for name,value in zip(
            ('c','d','t','kappa','J'),(algebra.c,algebra.d,algebra.t,algebra.k,algebra.J))},
        exact_active_partition=exact_partition,
        invariant_blocks=[dict(names=row['names'],rank=row['rank'],multiplicity=row['multiplicity'],
            coefficients=[str(e) for e in row['coefficients']],variable=row['variable']) for row in rows],
        unused60=dict(positive=34,quartic_soft=26,moduli_independent=True,unit_full120_soft_coefficient='kappa/4'),
        interval_certificate=cert,Clifford_C0=enclosure(cliff.v),endpoint_C0=enclosure(endpoint.v),
        strict_endpoint_advantage=enclosure(gap),native_controls=native_rows,
        quantum_ground_computed=False,campaign_complete=False,all_checks_pass=True,
        proof_status='Computer-assisted global C0 endpoint theorem and complete-domain leading variational semiclassical construction; see companion proof. No finite-Lambda quantum ground was computed.',
        full_domain_asymptotic=dict(law='Unchanged fullR120 x SU4 x T2 x native4900 neutral primitive Gauss fibre; fixed positive UE4,UE1 and fixed K',
            statement='E0=-3R Lambda + endpoint_C0 Lambda^(3/4) + o(Lambda^(3/4)); neutral ground-projector density concentrates on the delta=q minimum stratum',
            proof_kind='Analytic variational lower and admissible native paired-section upper construction in companion proof; not a numerical ground solve or a sampled BO extrapolation',
            unresolved='Finite-Lambda error constants and subleading effective soft/tangent spectrum are not computed; charged gap/spin, bulk/thermodynamic photon phase and the joined campaign remain open'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',type=int,default=45)
    parser.add_argument('--max-depth',type=int,default=22)
    parser.add_argument('--ue4',type=float,default=.3)
    parser.add_argument('--interval-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.precision,args.max_depth,args.ue4,not args.interval_only)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),proof_status=report['proof_status'],
        cells=report['interval_certificate']['cell_count'],
        Clifford_C0=report['Clifford_C0'],endpoint_C0=report['endpoint_C0'],
        strict_endpoint_advantage=report['strict_endpoint_advantage'],all_checks_pass=True),indent=2))
if __name__=='__main__': main()
