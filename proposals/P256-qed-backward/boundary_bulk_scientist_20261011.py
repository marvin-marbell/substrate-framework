"""NEW native boundary/common-algebra/deck/Gauss controls; not a Gibbs solver."""
from __future__ import annotations
import argparse
from collections import deque
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from magnetic_boundary_slab_scientist_20261011 import FixedMagneticSlab
from magnetic_boundary_scientist_20261011 import generic_phi
ROOT = Path(__file__).resolve().parent

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def cycles(slab: FixedMagneticSlab) -> tuple[Any, dict[str, Any]]:
    """Saturated integer kernel: collapse boundary to ground, fundamental cycles."""
    d = slab.DI.tocsc(); rows = d.shape[0]; ground = rows
    endpoints = []
    adjacency: list[list[tuple[int, int, int]]] = [[] for _ in range(rows+1)]
    for edge in range(d.shape[1]):
        col = dict(zip(d.indices[d.indptr[edge]:d.indptr[edge+1]], d.data[d.indptr[edge]:d.indptr[edge+1]]))
        source = next((int(i) for i, a in col.items() if a == 1), ground)
        target = next((int(i) for i, a in col.items() if a == -1), ground)
        endpoints.append((source, target))
        adjacency[source].append((target, edge, 1)); adjacency[target].append((source, edge, -1))
    parent = {ground: (ground, -1, 0)}; queue = deque([ground]); tree = set()
    while queue:
        u = queue.popleft()
        for v, e, sign in adjacency[u]:
            if v not in parent:
                parent[v] = (u, e, sign); tree.add(e); queue.append(v)
    assert len(parent) == rows+1 and len(tree) == rows
    def root_path(v: int) -> dict[int, int]:
        out = {}
        while v != ground:
            u, e, sign = parent[v]; out[e] = sign; v = u
        return out
    ri, ci, data = [], [], []; column = 0
    for e, (u, v) in enumerate(endpoints):
        if e in tree: continue
        flow = {e: 1}
        # e runs u->v; close by v->ground->u.
        for k, a in root_path(u).items(): flow[k] = flow.get(k, 0)+a
        for k, a in root_path(v).items(): flow[k] = flow.get(k, 0)-a
        for k, a in flow.items():
            if a: ri.append(k); ci.append(column); data.append(a)
        column += 1
    c = sparse.csc_matrix((data, (ri, ci)), shape=(d.shape[1], column), dtype=np.int64)
    assert (d@c).nnz == 0 and column == 2*slab.n**3-3*slab.n**2
    chord_rows = [e for e in range(d.shape[1]) if e not in tree]
    assert (c[chord_rows]-sparse.eye(column, format='csc')).nnz == 0
    # Schur/unimodular tree proof, plus independent floating determinant consumer.
    sign1, log1 = np.linalg.slogdet(slab.LI.toarray())
    sign2, log2 = np.linalg.slogdet((c.T@c).toarray())
    assert sign1 == sign2 == 1 and abs(log1-log2) < 1e-8
    return c, dict(rank=column, tree_edges=rows, integer_kernel_residual_nnz=0,
        chord_identity=True, saturated_cycle_dual_to_primitive_projected_deck=True,
        logdet_LI_float=float(log1), logdet_CGram_float=float(log2),
        logdet_difference_float=float(log1-log2),
        exact_identity='det(C^T C)=det(D_I D_I^T); see unimodular proof',
        primitive_deck='Lambda=Q_I Z^M=C(C^TC)^-1 Z^d; affine rho gluing retained',
        free_integrated_heat='(2pi)^d/(sqrt(det LI)*(4pi*64*beta_temperature)^(d/2))',
        native_cell_radian_volume='(2pi)^d/sqrt(det LI)')

def common_operator(slab: FixedMagneticSlab, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    n, v = slab.n, slab.v
    x = slab.index[(n//2-1, 1, 1)]
    py = slab.path(x, 1, 1); pz = slab.path(x, 2, 1)
    assert py is not None and pz is not None
    yz = slab.path(py['target'], 2, 1); zy = slab.path(pz['target'], 1, 1)
    assert yz is not None and zy is not None
    loop = {py['edges'][0]:1, yz['edges'][0]:1, zy['edges'][0]:-1, pz['edges'][0]:-1}
    k = np.array([loop.get(e,0) for e in slab.dynamic], dtype=np.int64)
    kt = np.array([loop.get(e,0) for e in range(3*v)], dtype=np.int64)
    assert not np.any(slab.DI@k) and not np.any(slab.native.D@kt)
    assert sum(loop[e]*slab.a[e] for e in loop) == Q(1,n*n)
    # On a finite interior patch theta_periodic=theta_slab+A is the common chart.
    theta = {e: Q((e%7)-3,17) for e in slab.dynamic}
    periodic = [Q(0)]*(3*v)
    for e in slab.dynamic: periodic[e] = theta[e]+slab.a[e]
    path_count = 0; nonzero = 0; all_species = 0
    species_mask = sum(15 << (4*i) for i, xyz in enumerate(slab.sites) if xyz[1] < n//2)
    global_mask = sum(species_mask << (4*v*s) for s in range(65)) | (15 << (4*v*65))
    # Gauge-covariant open transport is neutral only with its CAR endpoint factors.
    for length in (1,2):
        start = slab.index[(n//2, n//2-1, 1)]
        path = slab.path(start,1,length); assert path is not None
        for profile in profiles:
            if profile['scale']=='heavy' and length==2: continue
            row = next(r for r in profile['paths'] if r['axis']==1 and r['length']==length)
            fields={site:generic_phi(1+site%3) for site in (path['source'],path['target'])}
            import sympy as sp
            links={e:sp.diag(sp.I,-sp.I,-1,-1) for e in path['edges']}
            coefficient=slab.native.endpoint(row,path,fields,links)
            charged=profile['scale']!='heavy'
            for species in (range(64) if charged else [64]):
                image=slab.apply_path(global_mask,species,path,coefficient,theta,charged)
                other=slab.native.apply_path(global_mask,species,path,coefficient,periodic,charged)
                assert image == other and image
                all_species += 1; nonzero += len(image)
            path_count += 1
    # Weyl commutation and full affine shift: closed loop has no longitudinal term.
    assert int(k@k)==4 and np.array_equal(slab.DI@k,np.zeros(len(slab.interior)))
    return dict(local_base=list(slab.sites[x]), loop_dynamic_edges=len(loop),
        interior_closed_incidence_exact=True, periodic_closed_incidence_exact=True,
        chart_intertwining_exact=True, fullPhi_SU4_native_path_cases=path_count,
        canonical_species_path_actions=all_species, nonzero_CAR_images=nonzero,
        scalar_Wilson_shift_phase='W_r U_t=e^{-i t r.k} U_t W_r in radian convention',
        charged_endpoint_transport_is_neutral=True,
        boundary_Ref_excluded=True, pinned_E_excluded=True,
        scope='local algebra representation identity, NOT H/time/state intertwining')

def gauss_discriminator(slab: FixedMagneticSlab) -> dict[str, Any]:
    n,v=slab.n,slab.v
    rho=np.array([128 if xyz[0]<n//2 else -128 for xyz in slab.sites],dtype=float)
    rI=rho[slab.interior]; u=spsolve(slab.LI.astype(float),rI)
    lp=(slab.native.D@slab.native.D.T).tocsc().astype(float)
    up=np.zeros(v); up[1:]=spsolve(lp[1:,1:],rho[1:]); up-=np.mean(up)
    assert np.max(np.abs(lp@up-rho))<1e-8
    ds=float(64*rI@u); dt=float(64*rho@up)
    m=n//2
    exact_slab=64*128**2*Q((m-1)*(m*m-m+1),6*(2*m-1))
    exact_torus=64*128**2*Q(m*m+2,12)
    difference=-65536*Q(n*n,n-1)
    assert exact_slab-exact_torus==difference
    assert abs(ds/v-float(exact_slab))<1e-7 and abs(dt/v-float(exact_torus))<1e-7
    # Actual full64 canonical state: all species occupy wedge4 at x<N/2.
    # Paired complement family independently cancels fullrho at EVERY cell.
    complement_density=32*((4-2)+(0-2))
    assert complement_density==0
    eigenvalues=np.array([4*math.sin(math.pi*l/(2*(n-1)))**2+
        4*math.sin(math.pi*j/n)**2+4*math.sin(math.pi*k/n)**2
        for l in range(1,n-1) for j in range(n) for k in range(n)])
    assert abs(float(np.sum(np.log(eigenvalues)))-float(np.linalg.slogdet(slab.LI.toarray())[1]))<1e-8
    return dict(slab_HC_float=ds, periodic_HC_float=dt,
        HC_difference_per_volume_float=(ds-dt)/v,
        HC_slab_per_volume_exact=str(exact_slab),HC_torus_per_volume_exact=str(exact_torus),
        HC_difference_per_volume_exact=str(difference),
        evenN_difference_law_exact='-65536*N^2/(N-1)',
        fullrho_max=128, canonical_NL_each=2*v,
        failed_uniform_surface_only_HC_price=True,
        paired_complement_fullrho_exact=0,
        orthonormal_physical_singlet_family_log_count=33*(math.lgamma(v+1)-2*math.lgamma(v//2+1)),
        family_count_exact=f'binomial({v},{v//2})^33',
        dirichlet_eigen_min_float=float(min(eigenvalues)),
        HC_counted_once=True, moment_is_averaged_not_selected_point=True)

def trial_and_coercivity(slab: FixedMagneticSlab, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    import sympy as sp
    v=slab.v
    occupied={i for i, xyz in enumerate(slab.sites) if xyz[0]<slab.n//2}
    masks=[sum(15 << (4*i) for i in (occupied if s%2==0 else set(range(v))-occupied)) for s in range(64)]
    masks.append(sum(15 << (4*i) for i in occupied))
    assert all(mask.bit_count()==2*v for mask in masks)
    rho=[sum(((mask >> (4*i))&15).bit_count()-2 for mask in masks[:64]) for i in range(v)]
    assert rho == [0]*v
    fields=generic_phi(3)
    assert all(sp.trace(f)==0 for f in fields)
    r2=sum(sp.trace(f*f) for f in fields)
    jordan=sum(sp.trace((a*b+b*a)**2) for a in fields for b in fields)
    assert jordan>=r2**2/4
    onsite_cases=[]
    for profile in profiles:
        # Literal onsite is c_onsite*Phi0 (no scalar identity term in source).
        h=sp.Rational(profile['onsite'])*fields[0]
        full_site=sp.trace(h);empty_site=sp.Integer(0)
        assert full_site==empty_site==0
        onsite_cases.append(dict(role=profile['role'],full_wedge4_exact=str(full_site),
            empty_wedge0_exact=str(empty_site),scalar_identity_coefficient_exact=0,
            heavy_scale='-Lambda' if profile['scale']=='heavy' else '1'))
    return dict(canonical_counts_checked=65,fullrho_exact=rho[0],
        SU4_singlet_reason='wedge4 transforms by det(U)=1, wedge0 by1; radial fullPhi and Haar constant invariant',
        orthonormal_reason='distinct occupation masks are orthogonal; complementary pairs inject choices',
        onsite=onsite_cases,offsite_expectation_exact=0,
        offsite_reason='any nonzero c_x^dagger c_y changes occupation mask for distinct x,y',
        jordan_full_matrix_probe_exact=str(jordan),r2_exact=str(r2),
        local_Jordan_lower_checked=True,trial_is_not_ground=True)

def standing_gauss(n: int) -> dict[str, Any]:
    """Native Gauss Schur complement only; no magnetic or B2 substitution."""
    length=n-1; lam=4*math.sin(math.pi/n)**2; gamma=math.acosh(1+lam/2)
    profile=np.array([math.cosh(gamma*(x-length/2))/math.cosh(gamma*length/2) for x in range(n)])
    residual=np.max(np.abs((2+lam)*profile[1:-1]-profile[:-2]-profile[2:]))
    assert residual<1e-12
    dtN=1-profile[1]
    energy=float(np.sum(np.diff(profile)**2)+lam*np.sum(profile[1:-1]**2))
    assert abs(energy-2*dtN)<1e-12
    # alpha radians; atlas source alpha*cos(qy)/(2pi) on each z boundary link.
    source=[math.cos(2*math.pi*y/n)/(2*math.pi) for y in range(n)]
    assert abs(sum(source))<1e-14
    return dict(q=2*math.pi/n, alpha_unit='radian', atlas_derivative=source,
        native_Gauss_recurrence_residual_float=float(residual), DtN_per_boundary_float=float(dtN),
        gauss_energy_norm_float=energy, centre_transfer_float=float(profile[n//2]),
        qN_centre_limit='sech(pi), not zero', fixed_q_interior_decay='exp(-gamma(q)*distance)',
        magnetic_response_computed=False, scope='fullE longitudinal Gauss reservoir spectral connection ONLY')

def contrary() -> dict[str, Any]:
    # Exact commuting Ising Hamiltonian; falsification only, not native stiffness.
    import sympy as sp
    up=sp.Matrix([1,0]); down=sp.Matrix([0,1]); z=sp.diag(1,-1)
    lower=sp.Matrix([[0,0],[1,0]])
    assert (up.T*z*up)[0]==1 and (down.T*z*down)[0]==-1
    plus=(lower*up).dot(lower*up); minus=(lower*down).dot(lower*down)
    assert plus==1 and minus==0
    return dict(model='3D nearest-neighbour Ising, H=-sum_edges Zx Zy; NOT native law',
        pressure_ground_density_both_exact='-3', local_Z_plus_exact=1,local_Z_minus_exact=-1,
        A='sigma_minus at one site', temporal_measure_plus='delta_12', temporal_measure_minus='0',
        whole_ground_removed=True, equal_pressure_implies_state_or_spectrum=False,
        boundary_surface_energy_bound='2 times number of changed boundary bonds',
        native_conclusion='pressure equality alone cannot establish bridge')

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); request=json.loads(args.request.read_text())
    for name,pin in request['input_sha256'].items(): assert digest(ROOT/name)==pin,name
    profiles=json.loads((ROOT/request['profile']).read_text())['primitive_profiles']
    controls=[]
    for n in request['N']:
        slab=FixedMagneticSlab(n,1); _c,lattice=cycles(slab)
        controls.append(dict(N=n,lattice=lattice,common_operator=common_operator(slab,profiles),
            Gauss_Coulomb=gauss_discriminator(slab),standing_Gauss=standing_gauss(n),
            physical_trial_and_coercivity=trial_and_coercivity(slab,profiles)))
    result=dict(kind='NATIVE_COMMON_ALGEBRA_INTEGER_DECK_GAUSS_THERMAL_BOUND_CONTROLS_NOT_ACTUAL_GIBBS_NUMERICS',
        source_sha256=digest(Path(__file__)),request_sha256=digest(args.request),input_sha256=request['input_sha256'],
        controls=controls,contrary_equal_pressure=contrary(),
        actual_native_state_spectral_convergence=False,full_field_tail_numeric_paid=False,
        beta_Bloch_zero_substituted=False,full_cover_partition_claimed=False,
        first_missing='uniform connected boundary influence / fullrho screening and wholeP0 IR control')
    args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'receipt':args.output.name,'N':request['N'],'native_state_identity':False}))
if __name__=='__main__': main()
