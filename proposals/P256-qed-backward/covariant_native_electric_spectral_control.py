#!/usr/bin/env python3
"""Bounded native primitive-electric spectral control; no macro admission."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
import scipy
from scipy.linalg import eigh
import mpmath as mp
ROOT = Path(__file__).resolve().parent
# Snapshot before importing any repository module; retain loaded members only.
INITIAL_BYTES = {p.name: p.read_bytes() for p in ROOT.glob('*.py')}
PLAN = Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_SHA = hashlib.sha256(PLAN.read_bytes()).hexdigest()
from covariant_induced_photon import Parameters, phi_frame
from covariant_native_source_gram_moments import matched_tree_half
from covariant_native_closed_auxiliary_loop import auxiliary_path_half
from covariant_shared_split_route_photon import split_route_half
import covariant_native_haar_range_jets as native

PROOF = '''Declared Hilbert space only: fixed native finite CAR sectors (active N=4 or 5 of 8; auxiliary N=2 of 4), Phi3, and untruncated integer U(1) cycles on each specified finite graph in ONE total-density Gauss sector. No finite electric cut is used in this theorem.
For scalar B, Gershgorin gives b_l±=1/(2*.09)±1 and b_a±=1/(2*.04)±.5 uniformly in all link angles. Every path-average phase has modulus at most one, for both same-path and split-route laws. On y in [-.09,.09], the active row AND column sums of |F| are bounded by (1.09*|Gamma1_01|+|Gamma2_01|)/2; auxiliary bound is |Gamma2_01|/2. Thus ||F||2 <= sqrt(||F||1 ||F||infinity) <= f.
T=Jdag diag(B^-1,B) J with J=[[I,-F],[0,I]], and both ||J|| and ||J^-1|| <= 1+f. Hence min(1/bmax,bmin)/(1+f)^2 <= T <= max(1/bmin,bmax)*(1+f)^2. S=sqrt(T) has the square-root bounds. The full canonical exterior power Gamma_N(S), in ALL occupation states, has eigenvalues products of N eigenvalues of S. The scalar sqrt(det B) lies between bmin^(sites/2) and bmax^(sites/2). Multiply active and auxiliary bounds to obtain uniform positive mI<=SF<=MI. Pointwise bounds survive the physical Gauss restriction because the covariant local multiplication operator preserves that ONE sector; no middle Haar projection is inserted. Its inverse is bounded there.
For every allowed total density, choose an integer path solution Epath. All solutions are E=Epath+Ck, k integer, with G=C^T C. Completing squares gives ||E||²=(k+G^-1 C^T Epath)^T G(k+G^-1 C^T Epath)+||Epath-C G^-1 C^T Epath||². The longitudinal term is nonnegative. The shift is generally fractional and is NOT reset to zero. Poisson summation gives theta_G(a)=pi^(d/2)/sqrt(det G) sum_n exp(-pi² n^T G^-1 n) cos(2pi n.a) <= theta_G(0). Consequently Tr exp(-sum E²) <= fullCARcount*theta_G(0) per Phi node at UE1.
Let A=Phiheat_half SF, D=exp(-sum E²/2), F=A D, K=F Fdag. Phi3 heat is invertible, SF is uniformly bounded below, so A is bounded invertible. D is Hilbert-Schmidt and injective (its inverse need not be bounded). Fdag=D Adag is injective. K is therefore positive injective compact trace class with Tr K=||F||HS² <= ||Phiheat_half||² M²*3*fullCARcount*theta_G(0). Spectral theorem yields positive eigenvalues of finite multiplicity tending to zero. For positive neutral lambdaN=||KN||, H=-log(K/lambdaN) is the densely defined self-adjoint spectral logarithm, with domain sum_j |log(lambda_j/lambdaN)|²|c_j|²<infinity and discrete spectrum tending to +infinity. The charged sector may have a finite negative chemical shift relative to this SAME neutral normalization; nonnegativity is not asserted without its spectrum.
The physical sin/q multiplication field, provided it is nonzero almost everywhere (as the actual sine or nontrivial two-face Fourier combination is), is bounded and injective on L2 of the cycle torus. For any nonzero Omega, Bq Omega is nonzero, and injectivity of sqrt(KN) gives Zph=||sqrt(KN/lambdaN)Bq Omega||²>0. This proves neither literal source M0>0 nor any pole/residue, masslessness, or continuum cone.
Global full Cartesian quartic coercivity, internal kinetics, full SU4/Phi Hilbert space and macro P2 remain OPEN. This is a conditional declared primitive-electric finite-graph construction, not an induced-electric explanation or isolated-pole/ground-admission gate.'''

def bounds(bmin: float, bmax: float, f: float, sites: int, number: int) -> dict[str, float]:
    lo = min(1/bmax, bmin)/(1+f)**2
    hi = max(1/bmin, bmax)*(1+f)**2
    return dict(bmin=bmin, bmax=bmax, F_bound=f, T_min=lo, T_max=hi,
                canonical_min=bmin**(sites/2)*lo**(number/2),
                canonical_max=bmax**(sites/2)*hi**(number/2))

def canonical(half: np.ndarray, pref: float, number: int) -> np.ndarray:
    occupations = np.array(list(itertools.combinations(range(len(half)), number)))
    return pref*np.linalg.det(half[occupations[:, None, :, None], occupations[None, :, None, :]])

def theta(gram: list[list[int]]) -> dict[str, Any]:
    mp.mp.dps = 110
    G = mp.matrix(gram); d = len(gram); R = 6
    lam = mp.mpf(4 if d == 1 else 3)
    total = mp.mpf(0)
    for k in itertools.product(range(-R, R+1), repeat=d):
        v = mp.matrix(k); total += mp.exp(-(v.T*G*v)[0])
    # Gaussian shell union bound. Geometric ratio decreases beyond R+1.
    tail1 = mp.exp(-lam*(R+1)**2)/(1-mp.exp(-lam*(2*R+3)))
    theta1_bound = 1+2*mp.exp(-lam)/(1-mp.exp(-3*lam))
    tail = 2*d*tail1*theta1_bound**(d-1)
    rounding = mp.mpf('1e-90')
    return dict(G=gram, summation_box=R, precision_digits=110,
                finite_sum=mp.nstr(total, 95), positive_tail_bound=mp.nstr(tail, 95),
                arithmetic_allowance=str(rounding), lower=mp.nstr(total-rounding, 95),
                upper=mp.nstr(total+tail+rounding, 95),
                tail_proof='G>=lambda I; union over coordinates outside box; two one-dimensional Gaussian tails times bounded full sums in remaining coordinates')

def run() -> dict[str, Any]:
    gamma4, metric = phi_frame(Parameters(kappa_power=400))
    ev, U = eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]); embedding = U[:, ev>0]
    _, frame = eigh(embedding.conj().T@gamma4[0]@embedding); embedding = embedding@frame[:, ::-1]
    gamma = np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    fl = float((1.09*abs(gamma[1,0,1])+abs(gamma[2,0,1]))/2)
    fa = float(abs(gamma[2,0,1])/2)
    aux = bounds(12., 13., fa, 2, 2)
    sector_bounds = {}; comparisons = []
    for N in (4,5):
        light = bounds(1/.18-1, 1/.18+1, fl, 4, N)
        m = light['canonical_min']*aux['canonical_min']; M = light['canonical_max']*aux['canonical_max']
        sector_bounds[str(N)] = dict(active=light, auxiliary=aux, SF_min=m, SF_max=M, fullCARcount=420 if N==4 else 336)
        for y in (-.09,0.,.09):
            actual = gamma.copy(); actual[1] *= 1+y
            for L,R in ((.37,-.81),(1.23,.56),(2.71,-1.17)):
                half,pref = matched_tree_half(actual,L,0.)
                active = canonical(half,pref,N); al = eigh(active,eigvals_only=True)
                for law in ('two-path','split-route'):
                    pb = complex((np.exp(1j*L)+1)/2)
                    pf = pb if law=='two-path' else complex((np.exp(1j*L)+np.exp(1j*(L-R)))/2)
                    half,pref = auxiliary_path_half(actual,.04,pb) if law=='two-path' else split_route_half(actual,.04,pb,pf)
                    heavy = canonical(half,pref,2); ah = eigh(heavy,eigvals_only=True)
                    full = np.kron(active,heavy); values = eigh(full,eigvals_only=True)
                    assert values[0]>=m and values[-1]<=M
                    assert al[0]>=light['canonical_min'] and al[-1]<=light['canonical_max']
                    assert ah[0]>=aux['canonical_min'] and ah[-1]<=aux['canonical_max']
                    comparisons.append(dict(N=N,y=y,angles=[L,R],law=law,phase_moduli=[abs(pb),abs(pf)],
                        full_matrix_dimension=len(full),actual_min=float(values[0]),actual_max=float(values[-1]),
                        uniform_min=m,uniform_max=M,hermitian_defect=float(np.linalg.norm(full-full.conj().T))))
    y = np.array([-.09,0.,.09]); sigma=.03; radius=.12
    k=np.exp(-y*y/(4*sigma*sigma)-sigma*sigma*y*y/(radius*radius*(radius*radius-y*y))); k/=np.linalg.norm(k)
    D=np.zeros((2,3))
    for j in range(2):
        D[j,j]=np.sqrt(k[j+1]/k[j])/.09; D[j,j+1]=-np.sqrt(k[j]/k[j+1])/.09
    he,U=eigh(2*sigma*sigma*D.T@D); heat=(U*np.exp(-he/2))@U.T
    heatnorm=float(np.linalg.norm(heat,2)); traces={name:theta(G) for name,G in [('square',[[4]]),('shared',[[4,1],[1,4]])]}
    trace_bounds=[]
    for name,t in traces.items():
        for N,b in sector_bounds.items():
            value=mp.mpf(str(heatnorm))**2*mp.mpf(str(b['SF_max']))**2*3*b['fullCARcount']*mp.mpf(t['upper'])
            trace_bounds.append(dict(graph=name,N=int(N),upper=mp.nstr(value,70)))
    # Actual all-density completing-square checks, including nonzero fractional cosets.
    density_rows=[]
    for charge in (0,1):
        for l,a in itertools.product(itertools.combinations(range(8),4+charge),itertools.combinations(range(4),2)):
            rho=native.density(l,a,charge)
            for name,C in [('square',np.ones((4,1))),('shared',np.array([[1,0],[1,0],[1,0],[1,1],[0,-1],[0,-1],[0,-1]]))]:
                path=np.array([rho[0],rho[0]+rho[1],rho[0]+rho[1]+rho[2],0]+([] if name=='square' else [0,0,0]),float)
                G=C.T@C; shift=np.linalg.solve(G,C.T@path); longitudinal=float(path@path-shift@G@shift)
                assert longitudinal>=-1e-12
                density_rows.append(dict(charge=charge,graph=name,rho=list(map(int,rho)),shift=shift.tolist(),longitudinal=longitudinal))
    descriptor=dict(UE=1,active_kappa=.09,auxiliary_kappa=.04,Phi3=y.tolist(),normal_interval=[-.09,.09],
        gamma_real=np.real(gamma).tolist(),gamma_imag=np.imag(gamma).tolist(),graphs={'square':[[4]],'shared':[[4,1],[1,4]]},
        allCARcounts=[420,336],plan_id=PLAN.stem,plan_sha256_at_IMPORT=PLAN_SHA,scope='bounded native finite CAR/Phi3, untruncated finite-graph U1 cycles only')
    sources={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and Path(filename).parent==ROOT and Path(filename).name in INITIAL_BYTES:
            name=Path(filename).name; data=INITIAL_BYTES[name]
            sources[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=data.decode())
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        sources_at_IMPORT=sources,proof=PROOF,uniform_bounds=sector_bounds,actual_nonzero_field_fullcanonical_comparisons=comparisons,
        rotor_heat_traces=traces,traceK_upper_bounds=trace_bounds,all_density_affine_checks=density_rows,
        Phiheat_half=dict(norm=heatnorm,min_eigenvalue=float(eigh(heat,eigvals_only=True)[0])),physical_metric=metric,
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__,platform=platform.platform()))

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    start=time.monotonic(); result=run(); result['elapsed_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),comparisons=len(result['actual_nonzero_field_fullcanonical_comparisons']),uniform_bounds=result['uniform_bounds'],rotor_heat_traces=result['rotor_heat_traces'])))
if __name__=='__main__':
    main()
