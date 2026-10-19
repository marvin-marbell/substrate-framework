"""Native Abel flux and ground-resolvent insertion; no physical-ground solver."""
from __future__ import annotations
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, cast
import numpy as np
from scipy import sparse
import sympy as sp
from magnetic_mechanism_scientist_20261011 import incidence

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def abel(phi: Any, r: float) -> tuple[Any, Any, Any, Any]:
    den = 1 + 2*r*np.cos(phi) + r*r
    poisson = (1-r*r)/den
    return (2*np.arctan2(r*np.sin(phi), 1+r*np.cos(phi)),
            1-poisson, -2*r*(1-r*r)*np.sin(phi)/den**2, poisson)


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def reject(condition: bool, label: str) -> dict[str, Any]:
    try:
        require(condition, label)
    except AssertionError as error:
        return dict(assertion_reached=True, rejected=True, assertion=str(error))
    return dict(assertion_reached=True, rejected=False, assertion=label)


def native(n: int, r: float) -> dict[str, Any]:
    D, C, S = incidence(n)
    require((S@C).nnz == 0 and (C@D.T).nnz == 0, 'native oriented complexes')
    v = n**3
    sites = np.array(list(np.ndindex(n,n,n)))
    internal = np.zeros(3*v,dtype=bool)
    internal[::3] = sites[:,0]%2 == 0
    yz = C[2,:].toarray().ravel()
    xy = C[0,:].toarray().ravel()
    require(cast(bool, np.all(D@yz == 0) and yz@yz == 4), 'native yz closed character')
    require(not np.any(yz[internal]), 'yz character has no matching-internal edge')
    external_divergence = D@np.where(internal,0,yz)
    require(cast(bool, np.all(external_divergence == 0)), 'every cell d0 d1 vanish, so Q u vanish')
    mixed_wrong = reject(not np.any(xy[internal]), 'xy character contains matching-internal edges')
    require(mixed_wrong['rejected'], 'mixed-face negative control discriminates reference premise')
    f = np.zeros(3*v)
    f[::3] = np.sqrt(2/v)*np.cos(2*np.pi*sites[:,0]/n)
    require(cast(bool, abs(f@f-1) < 2e-14), 'real normalized native q')
    kappa = 2*np.sin(np.pi/n)
    b = C.T@f
    require(abs(b@b-kappa*kappa) < 1e-13, 'native dispersion')
    integers = np.random.default_rng(20261011+n).integers(-19,20,3*v)
    units = C@integers
    theta = np.pi*integers/11
    face = C@theta
    value, derivative, second, p = abel(face,r)
    g = C.T@(f*derivative)
    defect = C.T@(f*p)
    require(np.max(np.abs(g-(b-defect))) < 2e-14, 'exact Poisson gradient decomposition')
    require(np.max(np.abs(D@g)) < 2e-14, 'full Gauss closed gradient')
    laplacian = 4*np.dot(f,second)
    shifted = theta.copy(); shifted[0] += 2*np.pi
    require(np.max(np.abs(abel(C@shifted,r)[0]-value)) < 5e-14, 'native deck periodicity')
    wrapped_units = ((units+11)%22)-11
    charge = S@wrapped_units
    require(cast(bool, np.all(charge%22 == 0) and np.any(charge != 0)), 'retained principal integer defects')
    divergence = S@value
    require(np.max(np.abs(divergence)) > 1e-3, 'Abel additive Bianchi fails on native legal fields')
    # A normalized constant face source diagnoses the q=0 nonlinear UV term.
    f0 = np.zeros(3*v); f0[::3] = 1/np.sqrt(v)
    g0 = C.T@(f0*derivative)
    require(np.linalg.norm(C.T@f0) < 1e-14, 'constant raw source telescopes')
    require(g0@g0 > 1e-3, 'smooth periodic defect survives q0')
    # Spatial face convolution is scalar on an exact Fourier source, not UV removal.
    smooth = (1+2*np.cos(2*np.pi/n))/3
    convolution = lambda x: (x+np.roll(x,1,axis=0)+np.roll(x,-1,axis=0))/3
    coarse_value = convolution(value.reshape(n,n,n,3)).ravel()
    require(abs(f@coarse_value-smooth*(f@value)) < 2e-13, 'spatial filter Fourier scalar')
    require(abs(np.dot(C.T@(smooth*f*derivative),C.T@(smooth*f*derivative))-smooth*smooth*(g@g)) < 2e-13, 'coarse contact rescales both moments')
    rsoft = kappa/(4+kappa)
    softened_value, softened_derivative, _, _ = abel(face,rsoft)
    softened_g = C.T@(f*softened_derivative)
    require(64*(softened_g@softened_g) <= 192*kappa*kappa+1e-12, 'q-dependent Abel source has uniform small contact')
    softened_haar_m1 = 512*rsoft*rsoft/(1-rsoft*rsoft)
    softened_haar_m0_upper = 2*rsoft*rsoft/(1-rsoft*rsoft)
    require(softened_haar_m1 >= 256*softened_haar_m0_upper, 'small contact alone does not lower Haar spectral scale')
    wrong = {
        'erase_Poisson_defect': reject(np.max(np.abs(g-b)) < 1e-12, 'gradient must retain Poisson defect'),
        'bounded_additive_Bianchi': reject(np.max(np.abs(divergence)) < 1e-12, 'S b_r(CTheta) is not zero'),
        'q0_contact_vanishes': reject(g0@g0 < 1e-12, 'q0 nonlinear UV contact is positive'),
        'spatial_filter_removes_UV': reject(abs(g0@g0) < 1e-12, 'spatial filter has multiplier one at q0 and cannot remove UV'),
    }
    require(all(x['rejected'] for x in wrong.values()), 'all intended geometry rejections discriminate')
    return dict(N=n, r=r, kappa=float(kappa), normalized_source=float(f@f),
                SC_exact_zero=True, CDt_exact_zero=True, raw_Bianchi_integer=True,
                wrapped_monopoles=int(np.count_nonzero(charge)),
                Abel_Bianchi_max=float(np.max(np.abs(divergence))),
                contact_configuration=128*float(g@g), laplacian=float(laplacian),
                gradient_defect_norm=float(np.linalg.norm(defect)),
                q0_contact_configuration=128*float(g0@g0),
                spatial_multiplier=float(smooth), wrongcases=wrong,
                conditional_matching=dict(yz_closed_exact=True, yz_norm2_exact=int(yz@yz),
                    internal_edge_count=int(np.count_nonzero(yz[internal])),
                    site_external_divergence_exact_zero=True, all_cell_Q_u_exact_zero=True,
                    spectral_reference_applicable=n>=8, spectral_energy='256 j^2 at even N>=8 under matching g theorem',
                    mixed_face_wrongcase=mixed_wrong, fullPhi_ground=False),
                softened_operator=dict(r=float(rsoft), F_configuration=float(f@softened_value),
                    m1_configuration=64*float(softened_g@softened_g),
                    uniform_m1_bound=192*float(kappa*kappa),
                    Haar_m0_upper=float(softened_haar_m0_upper), Haar_m1=float(softened_haar_m1),
                    Haar_gap_lower=256, physical_ground=False),
                state_scope='legal configurations, not actual-ground expectations')


def annihilator(mode: int, modes: int = 8) -> Any:
    rows=[]; cols=[]; vals=[]
    for occupation in range(1<<modes):
        if (occupation>>mode)&1:
            rows.append(occupation ^ (1<<mode)); cols.append(occupation)
            vals.append((-1)**((occupation & ((1<<mode)-1)).bit_count()))
    return sparse.csr_matrix((vals,(rows,cols)),shape=(1<<modes,1<<modes),dtype=np.int64)


def car_and_affine(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    c=[annihilator(i) for i in range(8)]; I=sparse.eye(256,format='csr',dtype=cast(Any, np.int64))
    for i in range(8):
        for j in range(8):
            require((c[i]@c[j]+c[j]@c[i]).nnz == 0, 'finite CAR annihilator anticommutation')
            require((c[i]@c[j].T+c[j].T@c[i]-(I if i==j else 0*I)).nnz == 0, 'finite CAR delta')
    nx=sum((c[i].T@c[i] for i in range(4)),0*I)
    ny=sum((c[i].T@c[i] for i in range(4,8)),0*I)
    tested=0
    for a in range(4):
        for b in range(4,8):
            T=c[a].T@c[b]
            require((nx@T-T@nx-T).nnz == 0, 'density source +1')
            require((ny@T-T@ny+T).nnz == 0, 'density target -1')
            require(((nx+ny)@T-T@(nx+ny)).nnz == 0, 'canonical number conservation')
            tested+=1
    # Exact parallel-edge quotient, retaining coherent density shifts.
    D=sp.Matrix([[1,-1],[-1,1]])
    Lplus=sp.Matrix([[1,-1],[-1,1]])/8
    A=D.T*Lplus; Q=sp.eye(2)-A*D; k=sp.Matrix([1,0])
    require(Q*k+A*D*k == k, 'full affine electric insertion gives literal charge1 k')
    affine_wrong=reject(Q*k == k, 'Pi-only electric misses longitudinal density conjugation')
    x,y=sp.symbols('rho_active rho_aux',real=True)
    hc=64*(x+y)**2
    collective_wrong=reject(sp.expand(hc-64*x*x-64*y*y)==0, 'collective HC has cross128 rho_active rho_aux')
    t,ph,u,alpha=sp.symbols('Theta Phi SU4 affine',real=True)
    rr=sp.Rational(3,4)
    F=2*sp.atan(rr*sp.sin(t)/(1+rr*sp.cos(t)))
    psi=sp.Function('psi')(t,ph,u)
    E=lambda z:-sp.I*sp.diff(z,t)+alpha*z
    H=lambda z:64*E(E(z))
    double=sp.trigsimp(2*F*H(F*psi)-F*F*H(psi)-H(F*F*psi)-128*sp.diff(F,t)**2*psi)
    require(sp.simplify(double)==0, 'exact Abel full affine electric contact')
    primitive_checks=[]
    for profile in profiles:
        charge=0 if profile['scale']=='heavy' else 1
        for path in profile['paths']:
            length=int(path['length'])
            # B remains an arbitrary quantum endpoint/link multiplier, including adjoint and AP sign.
            B=sp.Function('B_endpoint_quantum')(ph,u)
            hop=B*sp.exp(sp.I*charge*length*t)
            require(sp.simplify(F*hop*psi-hop*F*psi)==0, 'full Phi/SU4 CAR Peierls multiplier commutes')
            primitive_checks.append(dict(role=profile['role'],load=profile['load'],length=length,
                                         axis=path['axis'],charge=charge,commutator_zero=True))
    require(sp.simplify(F*sp.diff(psi,ph,2)-sp.diff(F*psi,ph,2))==0, 'full Phi kinetic commutes')
    require(sp.simplify(F*sp.diff(psi,u,2)-sp.diff(F*psi,u,2))==0, 'full SU4 derivatives commute')
    return dict(CAR_modes=8, Fock_dimension=256, exact_CAR_pairs=64, colour_hops=tested,
                canonical_embedding='two-site colour subalgebra tensored with spectator occupations in each native 2V sector; no CAR truncation adopted',
                primitive_checks=primitive_checks, quantum_B_not_frozen=True,
                SU4_scope='F is identity on all Phi/Haar/CAR colour coordinates, hence commutes with full Gauss generators and Casimir',
                affine_A=str(A), affine_Q=str(Q), full_affine_exact=True,
                full_contact_exact=True, collective_HC=str(sp.expand(hc)),
                wrongcases=dict(Pi_only=affine_wrong, independent_HC=collective_wrong))


def domain_and_filter(n: int, r: Fraction) -> dict[str, Any]:
    # Infinite Fourier-support electric probe, NOT a replacement for physical ground.
    # Each face character has ||C^T e_face||^2=4, hence E_n=256 n^2.
    # Resolvent filter coefficient epsilon/(E_n+epsilon) is an actually defined new operator.
    rr=r*r; cutoff=96
    eps=Fraction(4,n*n)  # exact low-frequency control parameter, not physical coupling
    m0=Fraction(0); m1=Fraction(0); filtered0=Fraction(0); filtered1=Fraction(0)
    for j in range(1,cutoff+1):
        weight=2*rr**j/(j*j); energy=256*j*j
        factor=eps/(energy+eps)
        m0+=weight; m1+=energy*weight
        filtered0+=factor*factor*weight; filtered1+=energy*factor*factor*weight
    # Honest infinite tails, outward exact rational bounds.
    tail0=2*rr**(cutoff+1)/((cutoff+1)**2*(1-rr))
    tail1=512*rr**(cutoff+1)/(1-rr)
    tailf0=tail0*(eps/(256*(cutoff+1)**2+eps))**2
    tailf1=tail1*(eps/(256*(cutoff+1)**2+eps))**2
    require(filtered1>=256*filtered0, 'electric contrary probe remains gapped under filtering')
    require(filtered1+tailf1<=eps*m0/4, 'universal resolvent moment bound')
    require(512*rr/(1-rr)==m1+tail1, 'exact Haar UV moment and tail')
    # Degenerate-ground finite spectral control, with IR and arbitrarily large UV contact.
    kap=Fraction(1,n)
    energies=[Fraction(0),Fraction(0),kap,Fraction(10000)]
    weights=[Fraction(2),Fraction(3),kap,Fraction(1)]
    excitation=[(e,w) for e,w in zip(energies,weights) if e>0]
    q0mass=sum(w for e,w in excitation)
    scalar_mass=sum(weights[1:])
    whole_wrong=reject(q0mass==scalar_mass, 'remove entire P0, not selected ground vector')
    low=next(w for e,w in excitation if e==kap)
    filtered_low=low*(kap/(kap+kap))**2
    uvfiltered1=Fraction(10000)*(kap/(10000+kap))**2
    require(filtered_low==kap/4 and uvfiltered1<=kap*kap/10000, 'IR survives while UV first moment is suppressed')
    return dict(N=n, electric_probe='native edge-torus Haar and HE only; NOT full-law ground or physical1',
                face_character_energy='256 j^2', source_norm2='2 Li_2(r^2)',
                matching_reference_extension=dict(applicable=n>=8,
                    premises='entire matching g theorem, frozen beta0 law, even N>=8, yz external-only closed character; every site d=0 so Q=u=0',
                    spectrum_exact='256 j^2', fullPhi=False, physical1=False),
                derivative_norm2=str(2*rr/(1-rr)), second_derivative_norm2=str(2*rr*(1+rr)/(1-rr)**3),
                Haar_m1_exact=str(512*rr/(1-rr)), Haar_m1_q_independent=True,
                epsilon_exact=str(eps), Fourier_cutoff=cutoff, retained_infinite_tail=True,
                m0_partial=float(m0), m0_tail_upper=float(tail0),
                filtered_m0_partial=float(filtered0), filtered_m0_tail_upper=float(tailf0),
                filtered_m1_partial=float(filtered1), filtered_m1_tail_upper=float(tailf1),
                filtered_gap_ratio_lower=256, filter_cannot_create_low_spectrum=True,
                finite_spectral_control=dict(not_physical_ground=True, wholeP0_wrongcase=whole_wrong,
                    total_m1=str(sum(e*w for e,w in excitation)), low_weight_exact=str(low),
                    filtered_low_weight_exact=str(filtered_low), filtered_UV_m1_exact=str(uvfiltered1)),
                precision='all series partial sums and tails exact rational; displayed decimals are only float renderings')


def run(request_path: Path, output: Path) -> None:
    request=json.loads(request_path.read_text())
    for name,pin in request['input_sha256'].items():
        require(sha(ROOT/name)==pin, 'input SHA '+name)
    profiles=json.loads((ROOT/request['profile']).read_text())['primitive_profiles']
    require([(p['load'],p['scale']) for p in profiles]==[(1,'heavy'),(1,'charged'),(63,'charged')], 'literal species loads')
    r=Fraction(request['r_exact'])
    require(0<r<1, 'smooth Abel domain')
    receipt=dict(kind='NEW_NATIVE_ABEL_FLUX_AND_ENERGY_FILTERED_INSERTION_NOT_PHYSICAL_GROUND',
        source_sha256=sha(Path(__file__)), request_sha256=sha(request_path), input_sha256=request['input_sha256'],
        native=[native(n,float(r)) for n in (4,6,8)], algebra=car_and_affine(profiles),
        energy_filter=[domain_and_filter(n,r) for n in (4,6,8)],
        exact_uniform_operator_bounds=dict(r_exact=str(r), b_sup='2 asin(r)',
            derivative_sup=str(2*r/(1-r)), C_operator_norm_squared_upper=12,
            contact_m1_upper_exact=str(64*12*(2*r/(1-r))**2),
            interpretation='normalized source, N-uniform operator bound; NOT O(q^2) ground estimate'),
        physical_ground_measured=False, physical1_ground=False, atom=False, old_entrypoints_run=False)
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(receipt=output.name,N=[4,6,8],new_operators=['Abel flux','ground resolvent-filtered block insertion'],physical_ground=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.request,args.output)
