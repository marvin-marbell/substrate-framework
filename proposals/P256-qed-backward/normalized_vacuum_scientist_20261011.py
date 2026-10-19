"""New normalized full-CAR/one-shared-flow control and native all-time source consumer.
No historical main, root or solver is called. Finite two-role control is not g256.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import itertools as it
import json
import math
from pathlib import Path
import time
from typing import Any, Callable
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, expm_multiply, spsolve
import analytic_flux_time_scientist_20261011 as heat
import charged_output_lower_certificate_20261011 as iv
import charged_stieltjes_scientist_20261011 as cx

ROOT = Path(__file__).resolve().parent


class ComplexOperator(LinearOperator):
    """Native operator-adapter pattern with separate complex adjoint action."""

    def __init__(self, dimension: int, action: Callable[..., Any], reverse: Callable[..., Any]):
        self.action = action
        self.reverse = reverse
        super().__init__(np.dtype(np.complex128), (dimension, dimension))

    def _matvec(self, x):
        return self.action(x)

    def _rmatvec(self, x):
        return self.reverse(x)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(n, norms):
    """Actual raw supports include middle cells; no endpoint-only graph."""
    cell = lambda x: (x[0] // 2, x[1], x[2])
    incident = {}
    classes = {}
    for x, axis, length in it.product(it.product(range(n), repeat=3), range(3), (1, 2)):
        if axis == 0 and length == 1 and x[0] % 2 == 0:
            continue
        sites = [tuple((x[j] + (r if j == axis else 0)) % n for j in range(3)) for r in range(length + 1)]
        support = set(map(cell, sites))
        key = (axis, length, len(support))
        classes[key] = classes.get(key, 0) + 1
        for c in support:
            incident[c] = incident.get(c, F(0)) + norms[axis, length]
    assert len(set(incident.values())) == 1 and max(k[2] for k in classes) == 3
    return {'N': n, 'cells': n**3//2, 'J_exact': str(max(incident.values())),
            'path_classes': [{'axis': a, 'length': l, 'support_cells': s, 'positive_paths': count} for (a,l,s), count in sorted(classes.items())],
            'all_middle_raw_edges_retained': True}


def scalar_product_integral(a, b, base, moment=0):
    total = F(0)
    for ra, pa, ca in a:
        for rb, pb, cb in b:
            power = pa + pb + moment
            total += ca*cb*math.factorial(power)/(base+ra+rb)**(power+1)
    return total


def native_source(interface):
    """Two charged legs on actual transverse NN path; alltime not equal-time.
    This selected source contraction supplements, not replaces, all-path guards.
    """
    matrices, exact, norms = heat.native_matrices()
    rows = []
    for axis, role in it.product((1,2), ('active','aux')):
        plus = heat.heat_data(interface, next(r for r in interface['roles'] if r['role']==role and r['orientation']=='creation'))
        minus = heat.heat_data(interface, next(r for r in interface['roles'] if r['role']==role and r['orientation']=='annihilation'))
        M = matrices[role][axis]
        Md = heat.dagger(M)
        midpoint = [heat.CZ, heat.CZ]
        for rp, pp, Ap in plus['terms']:
            for rm, pm, Am in minus['terms']:
                product = heat.mm(heat.mm(heat.mm(Ap,M),Am),Md)
                trace = heat.sumc(product[j][j] for j in range(4))
                for moment in (0,1):
                    factor = F(math.factorial(pp+pm+moment),1)/(64+rp+rm)**(pp+pm+moment+1)
                    midpoint[moment] = cx.ca(midpoint[moment], heat.cscale(trace,factor))
        def error(data):
            return [(r,p,c*data['vop']/2) for r,p,c in data['d']] + data['source']
        ep, em = error(plus), error(minus)
        hp, hm = [(plus['b'],0,plus['h'])], [(minus['b'],0,minus['h'])]
        radii = []
        for moment in (0,1):
            radius = 4*norms[role][axis][0]**2*(scalar_product_integral(ep,hm,64,moment)+scalar_product_integral(hp,em,64,moment)+scalar_product_integral(ep,em,64,moment))
            assert radius >= 0
            radii.append(radius)
        mult = 1 if role == 'active' else 63
        entry = {'axis':axis,'role':role,'copy_multiplicity':mult,'spin_sum':'ALL four independent spin-pair blocks, native4x4 contraction',
                 'external_shared_rate_exact':'64','energies_have_no_upper_cutoff':True,'integration_domain':'[0,infinity)',
                 'native_matrix_exact':exact[role][axis], 'bounds':[]}
        for moment,name in ((0,'inverse_energy_source_weight'),(1,'inverse_square_dressing_weight')):
            center = midpoint[moment][0]
            radius = iv.iv(radii[moment])
            lo = max(0,center[0]-radius[1])
            hi = max(0,center[1]+radius[1])
            entry['bounds'].append({'name':name,'one_orientation_one_copy_exact':[str(F(lo,iv.Q)),str(F(hi,iv.Q))],
                                   'two_orientations_allcopies_exact':[str(2*mult*F(lo,iv.Q)),str(2*mult*F(hi,iv.Q))],
                                   'signed_midpoint_exact':iv.exact_endpoints(center),'whole_spectrum_error_exact':str(radii[moment]),
                                   'imaginary_midpoint_exact':iv.exact_endpoints(midpoint[moment][1])})
        rows.append(entry)
    return {'scope':'CURRENT g256 heat/source premises, transverse NN native same-endpoint site0 paths, all1+63, all spin channels; not finite-control ground',
            'rows':rows,'source_error_and_literal_normalizer_paid':True,
            'not_used_as_allorder_transfer_kernel':True}


def lift_onebody(h, masks):
    lookup = {m:i for i,m in enumerate(masks)}
    rr, cc, vv = [], [], []
    for col,m in enumerate(masks):
        for p,q in zip(*np.nonzero(h)):
            if not m>>q&1:
                continue
            right=m^(1<<int(q)); sign=(-1)**((m&((1<<int(q))-1)).bit_count())
            if right>>p&1:
                continue
            target=right|(1<<int(p)); sign*=(-1)**((right&((1<<int(p))-1)).bit_count())
            rr.append(lookup[target]);cc.append(col);vv.append(sign*h[p,q])
    return sparse.csr_matrix((vv,(rr,cc)),shape=(len(masks),)*2,dtype=complex)


def finite_control(profiles):
    """Two cells, active+aux labelled four-mode slots each, complete canonical8.
    Every label has n0..4; one raw external integer k shared by both roles.
    The two-role reduction is an honest METHOD control, never a native64 ground.
    """
    def profile_h(row):
        m,c,d=(float(F(row[k])) for k in ('m_exact','c0_exact','d_exact'))
        return m*np.diag([-1,1,-1,1])+c*np.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]])+d*np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
    hc=sparse.block_diag([profile_h(r) for r in profiles]).toarray()
    local_masks=[m for m in range(256) if m.bit_count()==4]
    Dl=lift_onebody(hc,local_masks).toarray()
    rho8=np.tile([.5,.5,-.5,-.5],2)
    rl=np.array([sum(rho8[p] for p in range(8) if m>>p&1) for m in local_masks])
    vals,vecs=np.linalg.eigh(Dl+np.diag(64*rl**2))
    g=vecs[:,0]; Eg=float(vals[0])
    masks=[m for m in range(65536) if m.bit_count()==8]
    lookup={m:i for i,m in enumerate(masks)}
    Hm=lift_onebody(sparse.block_diag([hc,hc]).toarray(),masks)
    k=np.array([(m&255).bit_count()-4 for m in masks])
    rA=np.array([sum(rho8[p] for p in range(8) if m>>p&1) for m in masks])
    rB=np.array([sum(rho8[p] for p in range(8) if m>>(p+8)&1) for m in masks])
    L=Hm+sparse.diags(64*((rA+k/2)**2+(rB+k/2)**2+k*k)-2*Eg)
    omega=np.zeros(len(masks),complex)
    for a,ma in enumerate(local_masks):
        for b,mb in enumerate(local_masks):
            omega[lookup[ma|(mb<<8)]]=g[a]*g[b]
    assert np.linalg.norm(L@omega)<1e-10
    # Actual common native c0,d endpoint coefficient in one spin replica.
    c,d=(float(F(profiles[0][key])) for key in ('c0_exact','d_exact'))
    endpoint=np.array([[-c,1j*d],[1j*d,-c]],complex)
    hop=np.zeros((16,16),complex)
    for role in (0,1):
        for p,q in it.product(range(2),repeat=2):
            hop[4*role+2+p,8+4*role+q]=endpoint[p,q]
    V=lift_onebody(hop+hop.conj().T,masks)
    vc=V.tocoo()
    assert np.max(np.abs(k[vc.row]-k[vc.col]))==1
    T=F(1,200)
    def coefficients(action):
        def apply(x,reverse=False):
            y=np.asarray(x).reshape(5,-1)
            out=np.stack([-L@v for v in y])
            for j in range(4):
                if reverse:
                    out[j]-=action(y[j+1])
                else:
                    out[j+1]-=action(y[j])
            return (float(T)*out).reshape(-1)
        size=5*len(masks)
        augmented=ComplexOperator(size, apply, lambda x:apply(x,True))
        initial=np.zeros((5,len(masks)),complex);initial[0]=omega
        out=expm_multiply(augmented,initial.reshape(-1),traceA=float((-5*L.diagonal().sum()*float(T)).real)).reshape(5,-1)
        return np.array([np.vdot(omega,x) for x in out])
    z=coefficients(lambda x:V@x)
    # Full neutral P_flow0 has dimension4900; it is NOT spectral P_ground.
    neutral=(k==0).astype(float)
    def Q(x):
        return neutral*x-omega*np.vdot(omega,x)
    def wrong(x):
        v=V@x
        return v-Q(v)-V@Q(x)
    zw=coefficients(wrong)
    filtered=expm_multiply(-float(T/2)*(L+V),omega,traceA=float((-float(T/2)*(L+V).diagonal().sum()).real))
    Z=float(np.vdot(filtered,filtered).real)
    obs=float(np.vdot(filtered,k*k*filtered).real/Z)
    nonzero=np.flatnonzero(k)
    chi=np.zeros_like(omega);chi[nonzero]=spsolve(L[nonzero][:,nonzero],(V@omega)[nonzero])
    a2=float(np.vdot(chi,chi).real);b2=float(np.vdot(chi,k*k*chi).real)
    volumes=[1,2,8,64,1024]
    volume=[{'disjoint_bond_copies':m,'log_exact_Z':m*math.log(Z),'normalized_root_observable':obs,
             'wrong_undressed_root_observable':b2/(1+m*a2),'undressed_norm_squared':1+m*a2,
             'normalized_product_firstorder_root_observable':b2/(1+a2)} for m in volumes]
    cumulant=float((z[4]-z[2]**2/2).real)
    wrong_subtraction=float((z[4]-z[2]**2).real)
    reset=float((zw[4]-zw[2]**2/2).real)
    assert abs(z[1])+abs(z[3])<1e-12 and abs(z[0]-1)<1e-10
    assert abs(z[2]-zw[2])<1e-10
    assert abs(cumulant-reset)>1e-9 and abs(cumulant-wrong_subtraction)>1e-9
    assert volume[-1]['wrong_undressed_root_observable']<volume[0]['wrong_undressed_root_observable']/2
    return {'scope':'NEW two-role full-CAR shared-flow method control at current native profile values, NOT all1+63 g256',
            'canonical_number':8,'full_CAR_modes':16,'dimension':len(masks),'neutral_P_flow0_dimension':int(sum(k==0)),
            'labelled_occupations':'all0..4 for every active/aux slot in each cell','all_flow_values':sorted(set(map(int,k))),
            'affine_offsets':'uA=uB=k/2; one external64k^2, no per-copy field',
            'T_exact':str(T),'physical_lambda_test':1,'local_method_ground_energy':Eg,'Z_filter_norm_squared':Z,
            'root_k_squared_normalized':obs,'lambda_Dyson_coefficients':[{'re':float(v.real),'im':float(v.imag)} for v in z],
            'connected_lambda4_log_coefficient':cumulant,'wrong_double_log_subtraction':wrong_subtraction,
            'wrong_reset_all_neutral_memory_to_ground':reset,'reset_same_lambda2':float(zw[2].real),
            'normalization_volume_controls':volume,'undressed_one_bond_inverse_square_weight':a2,
            'numeric_scope':'binary64 finite observation, not interval proof or native ground identification',
            'independent_of_parent_threecell_toy':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    started=time.monotonic();request=json.loads(Path(args.request).read_text())
    for name,pin in request['pins'].items():
        assert sha(ROOT/name)==pin,name
    boundary=json.loads((ROOT/'charged_first_shell_actual04_20261011_receipt.json').read_text())['all64_boundary']
    norms={(r['axis'],r['length']):F(r['complete_all64_shared_pointwise_norm_upper_exact']) for r in boundary['all64_once_shared_pointwise_norms']}
    graphs=[geometry(n,norms) for n in (8,10)]
    J=F(graphs[0]['J_exact']);assert graphs[0]['J_exact']==graphs[1]['J_exact']
    gamma=F(46955312929,34359738368);C=64;s=3;theta2=F(1,64)
    radius=gamma/(72*C*s*J)
    r1=36*C*s*J/gamma
    source=native_source(json.loads((ROOT/'charged_cell_heat_typed3_20261011_interface.json').read_text()))
    finite=finite_control(json.loads((ROOT/'local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json').read_text())['profiles'])
    result={'kind':'NORMALIZED_VACUUM_LINKED_NEW_ACTUAL_CONSUMER', 'graphs':graphs,
            'allorder_guard':{'gamma_exact':str(gamma),'J_exact':str(J),'projector_split_factor':C,'maximum_support_cells':s,
                             'source_boundary_ratio_squared_upper_exact':str(theta2),'N_uniform_radius_exact':str(radius),
                             'N_uniform_radius_display':float(radius),'r_at_physical_lambda1_exact':str(r1),'r_at_physical_lambda1_display':float(r1),
                             'r_required':'<=1/2 using e^2<9','physical1_verdict':'FAIL_TO_CERTIFY; positive-coefficient majorant divergent, not physical no-go',
                             'KP_event_guard_upper_at_radius_exact':'9/32','KP_time_guard_upper_at_radius_exact':'1/192',
                             'scope':'conditional on named inherited entire-law caps and independently reviewed defect/rod/Mayer proof, not phase or spectral-gap claim'},
            'native_alltime_source_attempt':source,'finite_full_CAR_shared_flow_normalization_control':finite,
            'pins':request['pins'],'source_sha256':sha(Path(__file__)),
            'all_pins_unchanged':all(sha(ROOT/name)==pin for name,pin in request['pins'].items()),
            'elapsed_seconds':time.monotonic()-started,'historical_main_runs':0,'natural_main_finished':True}
    with Path(args.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'radius':float(radius),'physical1_r':float(r1),'finite_lambda4_log':finite['connected_lambda4_log_coefficient'],'natural_main_finished':True}))


if __name__=='__main__':
    main()
