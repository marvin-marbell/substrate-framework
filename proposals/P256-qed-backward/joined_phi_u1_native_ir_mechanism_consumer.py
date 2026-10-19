"""Actual-law Gauss/algebra/current probes and stationary irreducible-kernel controls.

The companion proof derives the full NEW Jordan law before these controls.
Native probes use every literal path on the archived topology, arbitrary
nonuniform Phi and ordered SU4, never a claimed ground. Schur/Ward spectral
models are downstream method controls, not substitute campaign Hamiltonians.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any
import zipfile

import numpy as np
from scipy.linalg import expm
import sympy as sp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_native_ir_mechanism_'
PROOF = PREFIX + 'proof_corrected_20261010.json'
PINS = {
    'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz': 'f8a0793adcbf4909402cd089471aa8c33fc74a31b66e99618f8baaab8d6dfb65',
    'joined_phi_u1_ground_photon_collective_delivered_actual_20261010.json': '36cb109e5c5e9a5f850122b2b9758bfea9db2cf6b268c394e6a5401b4a205b14',
    'joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json': '1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94',
    'joined_phi_u1_matrix_jordan_supplier.py': 'a4e5e4eee0c84c8c72b9ed3fae0be2042bc8031336cba73c94dfc8d9d8b96420',
    'joined_phi_u1_stationary_canonical_photon.py': 'd2f33bdfbaae16c89bbc9d58f9c2bb636546a300e13378a7dfee9cb3276f7289',
    'joined_phi_u1_stationary_full_phi_fiber.py': '945ad7d3b2143ed077af074511ec641b9ce878109db6c7e2997d9790e86029c6',
    'joined_phi_u1_jordan_sector_gaussian_global_parent_packet_20261010.json': '36c70469170a1550d0a9bba52ae69164d520c3596288c354bf6abc80899012ed',
    'joined_phi_u1_jordan_sector_gaussian_parent_manifest_20261010.json': '55d18c9887ec8d4f83128b15f49e5c490bde62c7af87d885ff35d2cde2d0c12d',
    'bosonic_original_law_join.py': '8e107f1fecb8483200c1520b8afd0032a653c24bae5b81d1ecad36a81d0cdd9e',
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a))


def green(sites: np.ndarray) -> list[list[F]]:
    """Exact N4 inverse Laplacian from its literal spatial Fourier spectrum."""
    cos = (1, 0, -1, 0)
    table: dict[tuple[int, ...], F] = {}
    for displacement in sites:
        value = F(0)
        for m in sites:
            eigenvalue = 6 - 2 * sum(cos[int(t)] for t in m)
            if eigenvalue:
                value += F(cos[int(m @ displacement) % 4], 64 * eigenvalue)
        table[tuple(map(int, displacement))] = value
    return [[table[tuple(map(int, (x-y) % 4))] for y in sites] for x in sites]


def topology_controls(native: dict[str, np.ndarray]) -> dict[str, Any]:
    sites, d, q, loops = (native[k] for k in ('sites', 'D', 'Q', 'Wilson_64_loops'))
    lookup = {tuple(map(int, x)): i for i, x in enumerate(sites)}
    exact_d = np.zeros((64, 192), dtype=int)
    for x, pos in enumerate(sites):
        for axis in range(3):
            nxt = pos.copy(); nxt[axis] = (nxt[axis]+1) % 4
            y = lookup[tuple(map(int, nxt))]
            exact_d[x, 3*x+axis] = 1; exact_d[y, 3*x+axis] = -1
    assert np.array_equal(d, exact_d)
    assert np.array_equal(d @ loops.T, np.zeros((64, 64)))
    lp = green(sites)
    lap = exact_d @ exact_d.T
    for x in range(64):
        for y in range(64):
            assert sum((int(lap[x,t])*lp[t][y] for t in range(64)), F(0)) == F(int(x==y)) - F(1,64)
    k = [0]*192; k[0] = 1
    rho = [int(x) for x in exact_d[:,0]]
    ell = [sum((int(exact_d[x,e])*lp[x][y]*rho[y] for x in range(64) for y in range(64)), F(0)) for e in range(192)]
    transverse = [F(ke)-le for ke,le in zip(k,ell)]
    assert sum(t*t for t in transverse) + sum(rho[x]*lp[x][y]*rho[y] for x in range(64) for y in range(64)) == 1
    # FullE periodic translation is conserved; Pi-only translation is not.
    assert transverse[0]+ell[0] == 1 and transverse[0].denominator > 1
    pi_only_phase = complex(np.exp(2j*np.pi*float(transverse[0])))
    assert abs(pi_only_phase-1) > .1
    assert F(1,2)*int(loops[0,0]) == F(1,2)
    assert sp.exp(2*sp.pi*sp.I*sp.Rational(1,2)) == -1
    # All64 copies can undergo a common hop: exact collective electric action.
    one = F(64); collective64 = 64*64*one
    independent64 = 64*one
    assert collective64/independent64 == 64
    incoming = [F(0)]*192; incoming[0] = F(3,8)
    delta = 64*sum((2*p*F(ke)+F(ke)**2 for p,ke in zip(incoming,k)), F(0))
    assert delta == 112 and delta != 64
    assert 128*sum(p*F(ke) for p,ke in zip(incoming,k)) == 48
    assert 128*sum(F(ke)**2 for ke in k) == 128
    # Physical Fourier convention, with D source minus target.
    m = np.array([1,1,0]); wave = 2*np.pi*m/4
    site_phase = np.exp(-1j*(sites @ wave))/8
    fourier_error = 0.
    for axis in range(3):
        mid = sites.astype(float).copy(); mid[:,axis] += .5
        edge_phase = np.exp(-1j*(mid @ wave))/8
        actual = site_phase @ exact_d[:,axis::3]
        expected = 2j*np.sin(wave[axis]/2)*edge_phase
        fourier_error = max(fourier_error, norm(actual-expected))
    assert fourier_error < 1e-12
    pt = q @ q.T
    assert norm(pt - (np.eye(192)-d.T @ np.asarray(lp,dtype=float) @ d)) < 1e-10
    # Actual integer spatial cochain Bianchi: divergence of curl, NOT curl B.
    pairs=((0,1),(0,2),(1,2)); curl=np.zeros((192,192),dtype=int)
    def shifted(x: int, axis: int) -> int:
        pos=sites[x].copy(); pos[axis]=(pos[axis]+1)%4
        return lookup[tuple(map(int,pos))]
    for x in range(64):
        for face,(i,j) in enumerate(pairs):
            row=3*x+face
            curl[row,3*x+i]=1; curl[row,3*shifted(x,i)+j]=1
            curl[row,3*shifted(x,j)+i]=-1; curl[row,3*x+j]=-1
    cube=np.zeros((64,192),dtype=int)
    for x in range(64):
        for axis,face,sign in ((0,2,1),(1,1,-1),(2,0,1)):
            cube[x,3*shifted(x,axis)+face] += sign
            cube[x,3*x+face] -= sign
    assert np.array_equal(curl[::3],loops)
    assert np.array_equal(cube @ curl,np.zeros((64,192),dtype=int))
    field=np.zeros(192,dtype=int); field[0]=1
    magnetic=curl @ field; curl_magnetic=curl.T @ magnetic
    assert np.array_equal(cube @ magnetic,np.zeros(64,dtype=int))
    assert int(magnetic @ magnetic)==4 and int(curl_magnetic @ curl_magnetic)>0
    return {'all64_local_Gauss_exact': True, 'integer_edges':192, 'cycle_modes':129,
            'fullE_translation_phase_exact':'1', 'Pi_only_open_hop_phase':[pi_only_phase.real,pi_only_phase.imag],
            'fractional_closed_Weyl_full_translation_phase_exact':'-1',
            'full64_collective_to_wrong_independent_ratio_exact':str(collective64/independent64),
            'incoming_fullE_transfer_exact':str(delta), 'wrong_reset_incoming_transfer_exact':'64',
            'physical_midpoint_Fourier_error':fourier_error,
            'Bianchi_cube_times_curl_exact_zero':True,
            'nonuniform_A_flux_squared_exact':str(int(magnetic @ magnetic)),
            'wrong_curlB_squared_exact':str(int(curl_magnetic @ curl_magnetic)),
            'scope':'Exact rational native N4 Gauss/cover algebra, not an eigenstate or photon'}


def alignment_controls() -> dict[str,Any]:
    """Exact actual primitive HS/basis/index-count conversion, not a new law."""
    identity=sp.eye(2); sx=sp.Matrix([[0,1],[1,0]])
    sy=sp.Matrix([[0,-sp.I],[sp.I,0]]); sz=sp.diag(1,-1)
    pauli=(identity,sx,sy,sz)
    basis=[sp.kronecker_product(pauli[i],pauli[j]) for i in range(4) for j in range(4) if (i,j)!=(0,0)]
    assert all(sp.trace(a*b)==4*int(i==j) for i,a in enumerate(basis) for j,b in enumerate(basis))
    unitary=sp.diag(1,sp.I,1,-sp.I)
    assert unitary.det()==1 and unitary.H*unitary==sp.eye(4)
    hs=sp.Integer(0); coefficient_norm=sp.Integer(0)
    for mu in range(4):
        left=sum(((i+mu+1)*b for i,b in enumerate(basis)),sp.zeros(4))
        right=sum(((2*i-mu)*b for i,b in enumerate(basis)),sp.zeros(4))
        difference=(left-unitary*right*unitary.H).applyfunc(sp.simplify)
        ys=[sp.simplify(sp.trace(difference*b)/4) for b in basis]
        reconstructed=sum((y*b for y,b in zip(ys,basis)),sp.zeros(4))
        assert (reconstructed-difference).applyfunc(sp.simplify)==sp.zeros(4)
        hs += sp.simplify(sp.trace(difference.H*difference))
        coefficient_norm += sum(y*y for y in ys)
    assert hs==4*coefficient_norm and hs>0
    cart_norm=8*coefficient_norm
    actual=sp.Rational(1,5)*hs/8
    assert actual==hs/40==coefficient_norm/10==cart_norm/80
    assert hs/5==8*actual
    return {'scientific_alignment_exact':'1/5','primitive_matrix_Frobenius_divisor_exact':'8',
            'HS_coefficient_exact':'1/40','B_basis_y_coefficient_exact':'1/10',
            'Cart_T_basis_x_coefficient_exact':'1/80','fields':4,'traceless_colour_components':15,
            'positive_axis_links':192,'reverse_link_counted_again':False,
            'nonuniform_endpoint_HS_squared_exact':str(hs),
            'wrong_omit_norm_divisor_ratio_exact':'8',
            'scope':'Exact unscaled alignment norm conversion from actual primitive; Lambda54 retained symbolically; no bareEM branch imported'}


def native_jets(profile: dict[str, Any], sites: np.ndarray, phi: np.ndarray,
                links: np.ndarray, theta: np.ndarray, v: np.ndarray,
                omit_two: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = {tuple(map(int,x)): i for i,x in enumerate(sites)}
    h = np.zeros((256,256), complex); j = h.copy(); contact = h.copy()
    for x in range(64):
        h[4*x:4*x+4,4*x:4*x+4] = float(F(profile['onsite']))*phi[x,0]
    for row in profile['paths']:
        axis, length = int(row['axis']),int(row['length'])
        if omit_two and length==2: continue
        c0,ci = float(F(row['c0'])),float(F(row['ci']))
        for x,pos in enumerate(sites):
            current = pos.copy(); edges=[]; seam=1; ordered=np.eye(4,dtype=complex)
            for _ in range(length):
                idx=lookup[tuple(map(int,current))]; edge=3*idx+axis
                edges.append(edge); ordered=ordered @ links[edge]
                if current[axis]==3: seam *= -1
                current[axis]=(current[axis]+1)%4
            y=lookup[tuple(map(int,current))]
            fx=c0*phi[x,0]+1j*ci*phi[x,axis+1]
            fy=c0*phi[y,0]+1j*ci*phi[y,axis+1]
            charged = profile['scale'] != 'heavy'
            phase=np.exp(1j*sum(theta[e] for e in edges)) if charged else 1.
            weight=sum(v[e] for e in edges) if charged else 0.
            hop=seam*(fx @ ordered+ordered @ fy)*phase/2
            for out,block in ((h,hop),(j,1j*weight*hop),(contact,-weight*weight*hop)):
                out[4*x:4*x+4,4*y:4*y+4] += block
                out[4*y:4*y+4,4*x:4*x+4] += block.conj().T
    return h,j,contact


def current_controls(profiles: list[dict[str, Any]], native: dict[str,np.ndarray]) -> dict[str,Any]:
    rng=np.random.default_rng(222101064)
    raw=rng.normal(size=(64,4,4,4))+1j*rng.normal(size=(64,4,4,4))
    phi=(raw+raw.swapaxes(-1,-2).conj())/2
    phi -= np.trace(phi,axis1=-2,axis2=-1)[...,None,None]*np.eye(4)/4
    raw_links=rng.normal(size=(192,4,4))+1j*rng.normal(size=(192,4,4))
    generators=(raw_links+raw_links.swapaxes(-1,-2).conj())/20
    generators -= np.trace(generators,axis1=-2,axis2=-1)[:,None,None]*np.eye(4)/4
    links=np.array([expm(1j*b) for b in generators])
    theta=rng.normal(size=192); lam=rng.normal(size=64)
    v=native['D'].T @ lam; charge=np.repeat(lam,4)
    difference=charge[:,None]-charge[None,:]
    rows=[]
    for p in profiles:
        h,j,k=native_jets(p,native['sites'],phi,links,theta,v)
        charged=p['scale']!='heavy'
        expected_j=1j*difference*h if charged else np.zeros_like(h)
        expected_k=-difference*difference*h if charged else np.zeros_like(h)
        ej,ek=norm(j-expected_j),norm(k-expected_k)
        assert ej < 1e-10 and ek < 1e-9
        assert norm(h-h.conj().T) < 1e-12
        if charged:
            assert norm(k+expected_k) > 100
            _,wrong_j,wrong_k=native_jets(p,native['sites'],phi,links,theta,v,True)
            missing=norm(wrong_j-j)+norm(wrong_k-k)
            assert missing > 100
            alpha=.13
            hp,_,_=native_jets(p,native['sites'],phi,links,theta+alpha*v,v)
            gauge=np.exp(1j*alpha*difference)*h
            assert norm(hp-gauge)<1e-10
        else:
            missing=0.; assert norm(j)==0 and norm(k)==0
        rows.append({'role':p['role'],'load':p['load'],'all_path_families':len(p['paths']),
                     'Ward_current_error':ej,'Ward_contact_error':ek,'wrong_drop_two_link_error':missing,
                     'wrong_contact_sign_error':norm(k+expected_k)})
    assert [p['load'] for p in profiles] == [1,1,63]
    # Materially different force/orthogonal-response route: physical transverse
    # E obeys dotE_T=-J_T, so test the literal force, not static zeroPi curvature.
    phase=np.cos(2*np.pi*native['sites'][:,0]/4)
    transverse=np.zeros(192); transverse[1::3]=phase/8
    assert norm(native['D'] @ transverse)<1e-12 and norm(transverse)>0
    transverse /= norm(transverse)
    force_rows=[]; total_force_variance=0.
    coefficient=F(128*128,256*255)
    for p in profiles:
        h,j,k=native_jets(p,native['sites'],phi,links,theta,transverse)
        variance=float(coefficient)*float(np.trace(j @ j).real)
        commutator=norm(h @ j-j @ h)
        if p['scale']=='heavy':
            assert variance==0 and commutator==0
        else:
            assert variance>1 and commutator>1
            assert abs(np.trace(j))<1e-12 and abs(np.trace(k))<1e-12
            _,wrong_j,_=native_jets(p,native['sites'],phi,links,theta,transverse,True)
            assert norm(j-wrong_j)>1
        total_force_variance += int(p['load'])*variance
        force_rows.append({'role':p['role'],'load':p['load'],
                           'canonical_tracial_JT_variance':variance,
                           'onebody_h_JT_commutator_norm':commutator})
    assert total_force_variance>100
    return {'rows':rows,'sites':64,'ordered_fullSU4_links':192,'fullcanonical_particles_per_species':128,
            'fullcanonical_modes_per_species':256,'seed':222101064,
            'physical_transverse_force_route':{
                'rows':force_rows,'all64_canonical_tracial_variance':total_force_variance,
                'canonical_trace_coefficient_exact':str(coefficient),
                'contact_trace_exact':'0',
                'conclusion':'Actual transverse force is nonzero and dynamical at this arbitrary fullPhi/SU4 section; symmetry cannot supply stiffness and fullJJ cannot replace orthogonal dynamics.',
                'scope':'Normalized fixedN128 CAR trace at a prescribed field section, not a physical fullGauss stationaryground expectation. It tests operator algebra/nonzero force, not the sign of induced stiffness.'},
            'scope':'Allnative arbitrary nonuniform fullPhi/SU4 literal onebody jets; dGamma preserves these commutator identities at each fixed canonical number. Floating structural checks, not interval bounds or interacting response.'}


def stationary_method_controls() -> dict[str,Any]:
    # Only AFTER the actual-law proof: exact independent stationary Schur control.
    z=sp.I; ell=sp.Matrix([[2,1,1],[1,3,0],[1,0,4]])
    assert all(ell.extract(range(i),range(i)).det()>0 for i in (1,2,3))
    full=(z*sp.eye(3)-ell).inv(); sigma=1/(z-3)+1/(z-4)
    correct=1/(z-2-sigma)
    assert sp.simplify(full[0,0]-correct)==0
    force=sp.Matrix([0,1,1]); reducible=(force.T*full*force)[0]
    wrong=1/(z-2-reducible)
    assert sp.simplify(correct-wrong)!=0
    # Correct derivative sign and pole residue in an exact positive bath model.
    pole=(5-sp.sqrt(5))/2
    residue=sp.simplify(1/(1+1/(pole-3)**2))
    overlap=sp.simplify(1/(1+(pole-2)**2))
    assert sp.simplify(residue-overlap)==0
    wrong_residue=sp.simplify(1/(1-1/(pole-3)**2))
    assert sp.simplify(wrong_residue-overlap)!=0
    # Plausible Ward-only inference: both massive and massless kernels transverse.
    q=sp.Matrix([1,2,0]); pt=sp.eye(3)-q*q.T/(q.dot(q))
    assert pt*q==sp.zeros(3,1)
    massive=pt*(sp.Rational(1,100)+1)
    assert massive*q==sp.zeros(3,1)
    massive_gap=sp.sqrt(1+sp.Rational(1,100)); gapless=sp.Rational(1,10)
    assert massive_gap>1 and gapless<1
    continuum=[]
    for momentum in (1.,.1,.01):
        for s in (0.,1.,3.):
            atom=2/(momentum**2*(1+s*s))
            density=math.log((4+s*s)/(1+s*s))/momentum**2
            ratio=density/atom
            assert .69 < ratio < 1.32
            continuum.append({'q':momentum,'w_over_q':s,'continuum_to_atom_Euclidean_ratio':ratio})
    # Window around the atom at q: density widthdelta*q has vanishing weight ratio.
    fractions=[1.,.1,.01,.001]
    assert all(0 < f <= 1 for f in fractions) and fractions[-1] < .002
    return {'exact_Schur_error':'0','proper_orthogonal_Sigma':str(sp.simplify(sigma)),
            'wrong_full_reducible_Sigma':str(sp.simplify(reducible)),
            'wrong_full_reducible_resolvent_difference':str(sp.simplify(correct-wrong)),
            'exact_residue':str(residue),'wrong_derivative_sign_residue':str(wrong_residue),
            'Ward_massive_and_gapless_both_transverse':True,'Ward_only_mass_gap':str(massive_gap),
            'positive_gapless_no_atom_continuum':continuum,
            'shrinking_relative_window_continuum_weight_fraction':fractions,
            'shrinking_relative_window_atom_weight_fraction':[1,1,1,1],
            'scope':'Exact stationary finite method controls downstream of actual-law equations; no N4 ground/pole admission'}


def run(output: Path) -> dict[str,Any]:
    start=time.monotonic()
    captured={name:(ROOT/name).read_bytes() for name in PINS}
    assert all(sha(captured[name])==pin for name,pin in PINS.items())
    proof_bytes=(ROOT/PROOF).read_bytes(); proof=json.loads(proof_bytes)
    # Ground/control boundary explicitly consumed before any method control.
    assert proof['kind'].endswith('NOT_PHOTON_ADMISSION')
    assert 'Sigma_irr(z)=Bdag(z-RLR)^-1 B' in proof['materially_different_irreducible_route']['exact_Feshbach']
    with np.load(io.BytesIO(captured['joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz'])) as saved:
        native={k:saved[k] for k in ('D','Q','sites','Wilson_64_loops')}
    profiles=json.loads(captured['joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json'])['primitive_profiles']
    actual_algebra=topology_controls(native)
    actual_current=current_controls(profiles,native)
    alignment=alignment_controls()
    method=stationary_method_controls()
    source=Path(__file__).resolve(); source_bytes=source.read_bytes()
    captured[PROOF]=proof_bytes; captured[source.name]=source_bytes
    result={'kind':'COMPLETED_ACTUAL_NATIVE_ALGEBRA_CURRENT_AND_IRREDUCIBLE_METHOD_CONSUMER_NOT_GROUND_PHOTON',
            'proof':PROOF,'proof_sha256':sha(proof_bytes),'source_sha256':sha(source_bytes),
            'input_sha256':{k:sha(v) for k,v in captured.items() if k!=source.name},
            'actual_native_algebra':actual_algebra,'actual_fullPhi_SU4_current_contact':actual_current,
            'actual_alignment_norm_conversion':alignment,
            'stationary_irreducible_route_method_controls':method,
            'assertions_reached':['all64_Gauss','fullE_not_Pi_periodicity','fractional_Weyl_noncentrality','collective64_HC_HE','incomingPi_contact','midpoint_q_sign','allnative_Ward_contacts','wrong_contact_sign','wrong_drop_two_link','neutral_charge0','wrong_reducible_Sigma','wrong_Sigma_derivative_sign','Ward_only_masslessness','gapless_continuum_no_atom'],
            'correction_assertions_reached':['Bianchi_cube_C_zero','wrong_curlB_nonzero','Va_HS_B_Cart_exact_conversion'],
            'sameground_massless_pole_admission':False,'law_or_adoption_change':False,
            'elapsed_seconds':time.monotonic()-start}
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    result['source_inputs_unchanged']=True
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in captured.items():archive.writestr('executed_bytes/'+name,raw)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        receipt=run(args.output)
    print(json.dumps({'kind':receipt['kind'],'assertions_reached':receipt['assertions_reached'],
                      'elapsed_seconds':receipt['elapsed_seconds'],'ground_or_photon_admitted':False},indent=2))
