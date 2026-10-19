"""Independent full-law form, Fourier, Gaussian and contrary IR controls; no ground solver."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
PAIRS = ((0, 1), (0, 2), (1, 2))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curl_roll(theta: np.ndarray) -> np.ndarray:
    """Independent native positive-face action, spatial axes preceding final edge axis."""
    return np.stack([theta[..., j] + np.roll(theta[..., k], -1, axis=j)
                     - np.roll(theta[..., j], -1, axis=k) - theta[..., k]
                     for j, k in PAIRS], axis=-1)


def xy_incidence(n: int) -> np.ndarray:
    # Obtain the entire xy map by its action on basis inputs, not the target constructor.
    identity = np.eye(3*n**3).reshape(n, n, n, 3, 3*n**3)
    return (identity[..., 0, :] + np.roll(identity[..., 1, :], -1, axis=0)
            - np.roll(identity[..., 0, :], -1, axis=1) - identity[..., 1, :]).reshape(n**3, 3*n**3)


def full_law_symbolic() -> dict[str, Any]:
    t, s, phi, u, a0, a1, chi = sp.symbols('t s phi u A0 A1 chi', real=True)
    psi = sp.Function('Psi')(t, s, phi, u)
    source = sp.Function('F')(t, s)
    W = sp.Function('W_Jordan_alignment_plaquette_CAR_HC')(t, s, phi, u)
    def electric(value: Any, x: Any, connection: Any) -> Any:
        return -sp.I*sp.diff(value, x) + connection*value
    def primitive(value: Any, label: str) -> Any:
        if label == 'raw_HE':
            return 64*(electric(electric(value, t, a0), t, a0)
                       + electric(electric(value, s, a1), s, a1))
        if label == 'Phi_kinetic':
            return -4*chi*sp.diff(value, phi, 2)
        if label == 'SU4_Casimir':
            return -sp.Rational(3, 20)*chi*sp.diff(value, u, 2)
        return W*value
    results = {}
    for label in ('raw_HE', 'Phi_kinetic', 'SU4_Casimir', 'all_multipliers'):
        op = lambda value: primitive(value, label)
        double = sp.expand(2*source*op(source*psi)-source**2*op(psi)-op(source**2*psi))
        expected = 128*(sp.diff(source,t)**2+sp.diff(source,s)**2)*psi if label == 'raw_HE' else 0
        residual = sp.simplify(double-expected)
        assert residual == 0
        results[label] = str(residual)
    # A non-scalar internal coefficient retains noncommuting quantum/CAR action.
    internal = sp.Matrix([[phi, sp.I*u], [-sp.I*u, -phi]])
    assert source*internal-internal*source == sp.zeros(2)
    results['matrix_internal_commutator_zero'] = True
    results['interpretation'] = 'Generic F derivatives and affine longitudinal connections retained. Phi/SU4 differential probes are coordinate-independence controls; the analytic review covers all 60V Phi and all Haar Lie fields. No quantum field frozen.'
    return results


def fourier(n: int) -> dict[str, Any]:
    coordinates = np.indices((n, n, n))
    worst = 0.0
    modes = ((1,0,0), (0,1,0), (1,1,0), (1,2,1), (n//2,n//2,n//2))
    rows = []
    for m in modes:
        q = 2*np.pi*np.asarray(m)/n
        d = np.exp(1j*q)-1
        kappa2 = float(np.vdot(d,d).real)
        Cq = np.zeros((3,3),complex)
        for row,(j,k) in enumerate(PAIRS):
            Cq[row,j],Cq[row,k] = -d[k],d[j]
        projector = np.eye(3)-np.outer(d,d.conj())/kappa2
        defect = float(np.max(np.abs(Cq.conj().T@Cq-kappa2*projector)))
        assert defect < 1e-12
        phase = np.exp(1j*sum(q[j]*coordinates[j] for j in range(3)))
        vectors = []
        for axis in range(3):
            theta = np.zeros((n,n,n,3),complex); theta[...,axis]=phase
            actual = curl_roll(theta)
            error = float(np.max(np.abs(actual-phase[...,None]*Cq[:,axis])))
            assert error < 2e-13
            worst = max(worst,error)
        # Two independent physical polarization vectors selected without an eigensolver.
        for axis in range(3):
            vector=projector[:,axis].copy()
            for previous in vectors:
                vector-=previous*np.vdot(previous,vector)
            norm=float(np.linalg.norm(vector))
            if norm>1e-9:
                vectors.append(vector/norm)
            if len(vectors)==2:
                break
        assert len(vectors)==2
        for vector in vectors:
            assert abs(np.vdot(d,vector))<1e-12
            assert abs(np.vdot(Cq@vector,Cq@vector).real-kappa2)<1e-12
        rows.append({'m':list(m),'kappa_squared':kappa2,'two_transverse':True,'symbol_identity_error':defect})
    return {'N':n,'roll_vs_symbol_max_error':worst,'modes':rows,'harmonic_q0':'C(0)=0, three separately normalized real harmonic coordinates'}


def gaussian(n: int, variances: list[float], recorded: dict[str, Any]) -> dict[str, Any]:
    A = xy_incidence(n)
    G = A@A.T
    assert set(np.unique(G))=={-1.0,0.0,4.0}
    x,y,_ = np.indices((n,n,n))
    rows=[]
    recorded_rows={row['transverse_coordinate_variance']:row for row in recorded['bounded_native_gaussian_controls']}
    for v in variances:
        sincov=(np.exp(-v*(4-G))-np.exp(-v*(4+G)))/2
        cosgram=(np.exp(-v*(4-G))+np.exp(-v*(4+G)))/2
        for m in ((0,0),(1,0),(1,1)):
            qx,qy=2*np.pi*np.asarray(m)/n
            f=np.exp(-1j*(qx*x+qy*y)).reshape(-1)/np.sqrt(n**3)
            variance=float(np.vdot(f,sincov@f).real)
            budget=float(64*np.vdot(f,(G*cosgram)@f).real)
            neighbour_sin=(np.exp(-3*v)-np.exp(-5*v))/2
            neighbour_cos=(np.exp(-3*v)+np.exp(-5*v))/2
            expected_var=-np.expm1(-8*v)/2-2*neighbour_sin*(np.cos(qx)+np.cos(qy))
            expected_budget=64*(2*(1+np.exp(-8*v))-2*neighbour_cos*(np.cos(qx)+np.cos(qy)))
            assert abs(variance-expected_var)<2e-12
            assert abs(budget-expected_budget)<2e-10
            if m==(0,0):
                positive_q0=128*(-np.expm1(-3*v))*(-np.expm1(-5*v))
                assert positive_q0>0 and abs(budget-positive_q0)<2e-10
            discrepancy=None
            if m==(1,0) and v in recorded_rows:
                old=recorded_rows[v]
                discrepancy=max(abs(variance-old['sin_flux_total_variance']),abs(budget-old['sin_flux_form_gradient_budget_64']))
                assert discrepancy<2e-10
            rows.append({'v':v,'m_xy':list(m),'total_trial_variance':variance,'gradient_budget_64':budget,
                         'recorded_scientist_max_difference':discrepancy,'ground_spectral_moment':False})
    return {'N':n,'incidence_basis_action':'independent roll/basis map','face_overlap_values':sorted(set(G.ravel())),'rows':rows}


def sampling_positive_control(samples: int, seed: int) -> dict[str, Any]:
    n=4; v=0.4; rng=np.random.default_rng(seed); A=xy_incidence(n)
    theta=rng.normal(0,np.sqrt(v),(samples,3*n**3))
    B=theta@A.T
    # Longitudinal and harmonic components do not enter curl. Their projection gives
    # precisely the same B and gradient law as the legal physical quotient Gaussian.
    f=np.cos(2*np.pi*np.indices((n,n,n))[0]/n).reshape(-1); f/=np.linalg.norm(f)
    values=np.sin(B)@f
    gradients=(np.cos(B)*f)@A
    costs=64*np.sum(gradients**2,axis=1)
    sin_neighbour=(np.exp(-3*v)-np.exp(-5*v))/2
    cos_neighbour=(np.exp(-3*v)+np.exp(-5*v))/2
    exact_var=-np.expm1(-8*v)/2-2*sin_neighbour
    exact_budget=64*(2*(1+np.exp(-8*v))-2*cos_neighbour)
    variance=float(np.mean(values**2)); budget=float(np.mean(costs))
    var_se=float(np.std(values**2,ddof=1)/np.sqrt(samples)); budget_se=float(np.std(costs,ddof=1)/np.sqrt(samples))
    assert abs(variance-exact_var)<8*var_se and abs(budget-exact_budget)<8*budget_se
    return {'N':n,'v':v,'samples':samples,'seed':seed,'sample_variance':variance,'exact_variance':float(exact_var),
            'sample_budget':budget,'exact_budget':float(exact_budget),'variance_standard_error':var_se,'budget_standard_error':budget_se,
            'interpretation':'independent native Gaussian sampling positive control, not rigorous intervals or ground data'}


def shells(ns: list[int]) -> list[dict[str, Any]]:
    rows=[]
    for n in ns:
        modes=np.indices((n,n,n)).reshape(3,-1).T
        centered=(modes+n//2)%n-n//2
        lengths=np.max(np.abs(centered),axis=1)
        kappa=np.sqrt(np.sum(4*np.sin(np.pi*centered/n)**2,axis=1))
        nonzero=lengths>0
        assert np.all(kappa[nonzero]+1e-14>=4*np.linalg.norm(centered[nonzero],axis=1)/n)
        for ell in range(1,n//2+1):
            assert np.count_nonzero(lengths==ell)<=24*ell**2+2
        density=float(2*np.sum(1/kappa[nonzero])/n**3)
        bound=1.5+3.5/n
        assert density<=bound<=19/8
        rows.append({'N':n,'two_transverse_covariance_density':density,'analytic_bound':bound,'harmonic_density_if_width_vh':'3*vh/N^3, excluded from 19/8 coexact bound'})
    return rows


def contrary_cases() -> dict[str, Any]:
    k,c=sp.symbols('kappa c',positive=True)
    low,high=32*k/c,96*k/c
    density=c*k/(high-low)
    mass=sp.simplify(density*(high-low))
    m1=sp.simplify(density*(high**2-low**2)/2)
    assert mass==c*k and m1==64*k**2
    F=sp.Matrix([[0,sp.sqrt(k),k],[sp.sqrt(k),0,0],[k,0,0]])
    H=sp.diag(0,0,64); Omega=sp.Matrix([1,0,0]); P0=sp.diag(1,1,0)
    exc=(sp.eye(3)-P0)*F*Omega
    ordinary=(F*Omega).dot(F*Omega)
    excmass=exc.dot(exc); excmoment=exc.dot(H*exc)
    assert ordinary==k+k**2 and excmass==k**2 and excmoment==64*k**2
    return {'atomless':{'measure':'density c^2/64 on [32*kappa/c,96*kappa/c]','mass':str(mass),'m1':str(m1),'atoms':False,'lowenergy_mass_at_128kappa_over_c':str(mass)},
            'whole_groundblock':{'H':'diag(0,0,64)','F':str(F),'Omega':'e0','whole_P0':'diag(1,1,0)',
                                'ordinary_variance':str(ordinary),'excitation_mass':str(excmass),'excitation_m1':str(excmoment),
                                'lesson':'ordinary covariance >= kappa and moment64*kappa^2 do not imply whole-ground-removed covariance >= c*kappa as kappa->0; not claimed native ground'}}


def run(request_path: Path, output: Path) -> None:
    request=json.loads(request_path.read_text())
    for name,pin in request['input_sha256'].items():
        assert sha(ROOT/name)==pin,name
    recorded=json.loads((ROOT/request['recorded_receipt']).read_text())
    original={row['N']:row for row in recorded['N_controls']}
    result={'kind':'INDEPENDENT_ACTUAL_LAW_MAGNETIC_IR_REVIEW_CONTROLS_NOT_GROUND',
            'source_sha256':sha(Path(__file__)),'request_sha256':sha(request_path),'input_sha256':request['input_sha256'],
            'full_law_symbolic':full_law_symbolic(),'fourier':[fourier(n) for n in request['Ns']],
            'gaussian':[gaussian(n,request['variances'],original[n]) for n in request['Ns']],
            'native_sampling_positive_control':sampling_positive_control(request['samples'],request['seed']),
            'shells':shells(request['shell_Ns']),'contrary_cases':contrary_cases(),
            'historical_programs_imported_or_executed':False,'ground_phase_or_pole_proved':False}
    with output.open('x') as stream:
        json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps({'receipt':output.name,'kind':result['kind'],'ground_phase_or_pole':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.request,args.output)
