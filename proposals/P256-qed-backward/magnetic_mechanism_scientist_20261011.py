"""Native magnetic algebra and full-law form controls, never a ground solver."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
import sympy as sp

ROOT = Path(__file__).resolve().parent
PAIRS = ((0, 1), (0, 2), (1, 2))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def incidence(n: int) -> tuple[Any, Any, Any]:
    """Native positive-edge, positively oriented jk face, xyz cube incidence."""
    volume = n ** 3
    sites = list(np.ndindex(n, n, n))
    index = {x: i for i, x in enumerate(sites)}
    def target(x: tuple[int, ...], axis: int) -> int:
        y = list(x)
        y[axis] = (y[axis] + 1) % n
        return index[tuple(y)]
    dr, dc, dv, cr, cc, cv, sr, sc, sv = ([] for _ in range(9))
    for x in sites:
        i = index[x]
        for j in range(3):
            dr.extend((i, target(x, j))); dc.extend((3*i+j, 3*i+j)); dv.extend((1, -1))
        for p, (j, k) in enumerate(PAIRS):
            cr.extend((3*i+p,) * 4)
            cc.extend((3*i+j, 3*target(x,j)+k, 3*target(x,k)+j, 3*i+k))
            cv.extend((1, 1, -1, -1))
        # dB = forward_x B_yz - forward_y B_xz + forward_z B_xy.
        for axis, face, sign in ((0,2,1),(1,1,-1),(2,0,1)):
            sr.extend((i,i)); sc.extend((3*target(x,axis)+face,3*i+face)); sv.extend((sign,-sign))
    D = sparse.csr_matrix((dv,(dr,dc)),shape=(volume,3*volume),dtype=np.int64)
    C = sparse.csr_matrix((cv,(cr,cc)),shape=(3*volume,3*volume),dtype=np.int64)
    S = sparse.csr_matrix((sv,(sr,sc)),shape=(volume,3*volume),dtype=np.int64)
    return D, C, S


def lattice_control(n: int) -> dict[str, Any]:
    D, C, S = incidence(n)
    volume = n**3
    assert (S@C).nnz == 0 and (C@D.T).nnz == 0
    # Exact integer units pi/11: rawcurl is exact, wrapping is an integer operation.
    theta_units = np.random.default_rng(222+n).integers(-11,12,size=3*volume,dtype=np.int64)
    raw = C@theta_units
    principal = (raw+11)%22-11
    monopoles = S@principal
    assert np.array_equal(S@raw,np.zeros(volume,dtype=np.int64))
    assert np.all(monopoles%22 == 0) and np.count_nonzero(monopoles) > 0
    sin_div = S@np.sin(np.pi*raw/11)
    assert np.max(np.abs(sin_div)) > .1
    surfaces = []
    for z in range(n):
        f = np.zeros((n,n,n,3),dtype=np.int64); f[:,:,z,0]=1
        r = f.reshape(-1)
        assert np.array_equal(C.T@r,np.zeros(3*volume,dtype=np.int64))
        surfaces.append({'z':z,'raw_flux_pi_over_11_units':int(r@raw),
                         'principal_flux_2pi_units':int(r@principal)//22,
                         'surface_generator_edge_coefficients_nonzero':int(np.count_nonzero(C.T@r))})
        assert int(r@raw)==0 and int(r@principal)%22==0
    deck = np.zeros(3*volume,dtype=np.int64); deck[0]=22
    shifted = C@(theta_units+deck)
    assert np.array_equal((shifted+11)%22-11,principal)
    assert np.count_nonzero(shifted-raw)==4
    # Real normalized xy-face Fourier cosine; its coexact edge image is legal under Gauss.
    x = np.indices((n,n,n))[0]
    f = np.zeros((n,n,n,3)); f[...,0]=np.cos(2*np.pi*x/n)
    f=f.reshape(-1); f/=np.linalg.norm(f)
    b=C.T@f
    kappa=2*np.sin(np.pi/n)
    assert np.max(np.abs(D@b)) < 1e-12
    assert abs(float(b@b)-kappa*kappa)<1e-12
    # Arbitrary regular transverse Gaussian covariance v I on kerD:
    # Cov(CTheta)=v C C^T. Closed face rows have norm^2=4, neighbours overlap=-1.
    bounded = []
    q=2*np.pi/n
    xy=C[::3,:]
    overlap=(xy@xy.T).toarray()
    face_vector=f[::3]
    raw_mode_variance=float(b@b)/kappa
    electric_mode_variance=kappa/4
    assert abs(raw_mode_variance-kappa)<1e-12
    for v in (.01,1.,16.):
        sin_diagonal=-np.expm1(-8*v)/2
        cos_diagonal=(1+np.exp(-8*v))/2
        sin_neighbour=np.exp(-3*v)*(-np.expm1(-2*v))/2
        cos_neighbour=(np.exp(-3*v)+np.exp(-5*v))/2
        m0=sin_diagonal-2*sin_neighbour*(np.cos(q)+1)
        m1=64*(4*cos_diagonal-2*cos_neighbour*(np.cos(q)+1))
        q0_m1=256*(cos_diagonal-cos_neighbour)
        assert m0>0 and m1>0 and q0_m1>0
        # Independent contraction on the actual incidence, not a restated formula.
        sin_cov=np.exp(-4*v)*np.sinh(v*overlap)
        cos_gram=np.exp(-4*v)*np.cosh(v*overlap)
        native_m0=float(face_vector@sin_cov@face_vector)
        native_m1=float(64*face_vector@(overlap*cos_gram)@face_vector)
        assert abs(native_m0-m0)<1e-12 and abs(native_m1-m1)<1e-10
        bounded.append({'transverse_coordinate_variance':v,'sin_flux_total_variance':float(m0),
                        'sin_flux_form_gradient_budget_64':float(m1),
                        'q0_budget_nonzero':float(q0_m1),
                        'native_incidence_contracted_variance':native_m0,
                        'native_incidence_contracted_gradient_budget':native_m1,
                        'not_ground_spectral_moment':True})
    # Whole transverse Gaussian with mode variance1/kappa supplies raw variance~kappa.
    modes=np.indices((n,n,n)); eig=sum(4*np.sin(np.pi*modes[j]/n)**2 for j in range(3))
    inverse_root=np.zeros_like(eig); np.divide(1,np.sqrt(eig),out=inverse_root,where=eig>0)
    covariance_density=float(2*np.sum(inverse_root)/volume)
    assert covariance_density <= 1.5+3.5/n <= 19/8
    return {'N':n,'V':volume,'SC_exact_nonzero':(S@C).nnz,'CDtranspose_exact_nonzero':(C@D.T).nnz,
            'raw_surface_flux':surfaces,'principal_monopole_charge_values':sorted(set((monopoles//22).tolist())),
            'principal_monopole_count':int(np.count_nonzero(monopoles)),
            'net_principal_monopole_charge':int(np.sum(monopoles)//22),
            'sin_curl_cube_divergence_max':float(np.max(np.abs(sin_div))),
            'deck_changes_rawcurl_but_not_periodic_algebra':True,
            'all64_Gauss_closed_field':float(np.max(np.abs(D@b))),
            'raw_mode_kappa':float(kappa),'raw_mode_form_gradient_budget_64':float(64*b@b),
            'regular_order_candidate_raw_variance':raw_mode_variance,
            'regular_order_candidate_electric_variance':electric_mode_variance,
            'regular_candidate_covariance_trace_per_site_two_transverse':covariance_density,
            'bounded_native_gaussian_controls':bounded,
            'scope':'Exact native integer incidence/wrapped adverse control; floating Gaussian contractions. Regular candidates are full physical finite-form trials, NOT stationary ground data.'}


def algebra_control() -> dict[str, Any]:
    t, phi, u, b = sp.symbols('Theta Phi SU4 b',real=True)
    psi=sp.Function('Psi')(t,phi,u)
    F=sp.sin(b*t)
    E=lambda v:-sp.I*sp.diff(v,t)
    H=lambda v:64*E(E(v))
    double=sp.simplify(2*F*H(F*psi)-F*F*H(psi)-H(F*F*psi))
    assert sp.simplify(double-128*b*b*sp.cos(b*t)**2*psi)==0
    B=sp.Function('B')(phi,u)
    matter=sp.simplify(F*B*sp.exp(sp.I*t)*psi-B*sp.exp(sp.I*t)*F*psi)
    assert matter==0
    alpha=sp.symbols('alpha',real=True)
    hopping_change=sp.simplify(B*sp.exp(sp.I*(t+alpha))-B*sp.exp(sp.I*t))
    assert sp.simplify(hopping_change.subs(alpha,2*sp.pi))==0
    return {'double_commutator':str(double),'all_quantum_matter_multiplier_commutator':str(matter),
            'continuous_flat_shift_changes_charge1_hop':str(hopping_change),
            'integer_deck_shift_commutes_with_every_integer_path':True,
            'fullHE_coefficient':64,'species':'active1+aux63; heavy neutral and Ref retained',
            'fixedPhi_or_bareMaxwell_used':False}


def run(request_path: Path, output: Path) -> None:
    request=json.loads(request_path.read_text())
    for name,pin in request['input_sha256'].items():
        assert digest(ROOT/name)==pin,name
    profiles=json.loads((ROOT/request['profile']).read_text())['primitive_profiles']
    assert [(p['load'],p['scale']) for p in profiles]==[(1,'heavy'),(1,'charged'),(63,'charged')]
    assert all(any(int(row['length'])==1 for row in p['paths']) for p in profiles)
    result={'kind':'ACTUAL_NATIVE_MAGNETIC_ALGEBRA_AND_ORDER_HYPOTHESIS_DISCRIMINATION_NOT_GROUND',
            'source_sha256':digest(Path(__file__)),'request_sha256':digest(request_path),
            'input_sha256':request['input_sha256'],'algebra':algebra_control(),
            'N_controls':[lattice_control(n) for n in (4,6,8)],
            'full_ground_or_pole_proved':False,'old_program_entrypoints_executed':False}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'receipt':output.name,'N':[4,6,8],'full_ground_or_pole':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.request,args.output)
