#!/usr/bin/env python3
"""Native quantumEM × full64 centre Slaters in the original noncompact Href.

This is a normalized pure variational state, not a photon vacuum or pole.
The macro222 root remains implicit; its enclosure midpoint is used ONLY for
this finite-box diagnostic, with the frame image error reported separately.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from scipy.linalg import null_space
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import betaln

from induced_phi_backreaction import PhiConfig, SELECTED, models, scalar_paths, IMPORT_BYTES
from induced_phi_ir_well_transfer import rational, sqrt_bounds

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = dict(IMPORT_BYTES)
for _module in tuple(sys.modules.values()):
    _file = getattr(_module, '__file__', None)
    if _file is not None:
        _path = Path(_file).resolve()
        if _path.parent == ROOT and _path.suffix == '.py':
            SOURCE_BYTES[_path.name] = _path.read_bytes()
SOURCE_BYTES[Path(__file__).name] = Path(__file__).read_bytes()
SOURCE_HASHES = {name: hashlib.sha256(raw).hexdigest() for name, raw in SOURCE_BYTES.items()}


def _frame_image(mean: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    enclosure = mean['centre_enclosure']
    midpoints = [(rational(enclosure[k + '_lower']) + rational(enclosure[k + '_upper'])) / 2 for k in ('g0', 'gs')]
    scales = np.sqrt([float(midpoints[0]), float(midpoints[1]), float(midpoints[1]), float(midpoints[1])])
    prices = []
    for k, midpoint, image in zip(('g0', 'gs'), midpoints, scales[:2]):
        lower, upper = sqrt_bounds(rational(enclosure[k + '_lower']))[0], sqrt_bounds(rational(enclosure[k + '_upper']))[1]
        error = max(abs(Fraction(float(image)) - lower), abs(Fraction(float(image)) - upper))
        prices.append(dict(label=k, image_scale=float(image), exact_scale_error_upper=float(error), midpoint_float=float(midpoint)))
    return scales[:, None, None] * SELECTED, dict(scope='NATIVE midpoint diagnostic only; not a new macro centre', scales=prices)


def matter_hamiltonian(native: dict[str, Any], profile: dict[str, Any], theta: np.ndarray) -> np.ndarray:
    """Literal original Peierls matrix, site-major 4*x+spin, edge3*x+axis."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3 * native['V'],):
        raise ValueError('theta must be a physical real edge vector')
    h = np.kron(np.eye(native['V']), profile['onsite'])
    for path in profile['paths']:
        x, y = path['source'], path['target']
        hop = path['hop'] * np.exp(1j * np.sum(theta[list(path['edges'])]))
        h[4*x:4*x+4, 4*y:4*y+4] += hop
        h[4*y:4*y+4, 4*x:4*x+4] += hop.conj().T
    return h


def edge_current(native: dict[str, Any], profile: dict[str, Any], theta: np.ndarray, edge: int) -> np.ndarray:
    """Hermitian literal first derivative including wrapped two-link endpoints."""
    if not 0 <= edge < 3*native['V']:
        raise ValueError('Physical edge index is outside this box')
    theta=np.asarray(theta,dtype=float)
    if theta.shape!=(3*native['V'],):
        raise ValueError('theta must be a physical real edge vector')
    current=np.zeros((4*native['V'],4*native['V']),dtype=complex)
    for path in profile['paths']:
        multiplicity=path['edges'].count(edge)
        if multiplicity:
            x,y=path['source'],path['target']
            hop=1j*multiplicity*path['hop']*np.exp(1j*np.sum(theta[list(path['edges'])]))
            current[4*x:4*x+4,4*y:4*y+4]+=hop
            current[4*y:4*y+4,4*x:4*x+4]+=hop.conj().T
    return current


def build_native(meanreceipt: dict[str, Any] | str | Path, size: int = 6) -> dict[str, Any]:
    """Fixed public geometry/matter ABI shared with field-dependent Kato work."""
    if size < 2:
        raise ValueError('Native periodic box must contain at least two sites per axis')
    mean: dict[str, Any] = meanreceipt if isinstance(meanreceipt, dict) else json.loads(Path(meanreceipt).read_bytes())
    frame, image = _frame_image(mean)
    cfg = PhiConfig(Lambda=1., **mean['microscopic_config'])
    sites = np.array(list(np.ndindex((size, size, size))), dtype=int)
    V = len(sites)
    index = {tuple(site): i for i, site in enumerate(sites)}
    D = np.zeros((V, 3*V))
    for x, site in enumerate(sites):
        for axis in range(3):
            target = site.copy(); target[axis] = (target[axis]+1) % size
            D[x, 3*x+axis] = 1
            D[index[tuple(target)], 3*x+axis] = -1
    L = D @ D.T
    values, vectors = np.linalg.eigh(L)
    gap = 4 * np.sin(np.pi / size)**2
    positive = values > gap/2
    Lplus = (vectors[:, positive] / values[positive]) @ vectors[:, positive].T
    Q = null_space(D)
    PT = Q @ Q.T
    harmonics = np.zeros((3*V, 3))
    for axis in range(3):
        harmonics[axis::3, axis] = 1 / np.sqrt(V)
    native: dict[str, Any] = dict(N=size, size=size, V=V, M=2*V+1, a=cfg.a, UE=1.,
        D=D, Q=Q, Lplus=Lplus, PT=PT, frame=frame, sites=sites, harmonics=harmonics,
        g0=float(np.trace(frame[0]@frame[0]).real/4), gs=float(np.trace(frame[1]@frame[1]).real/4),
        frame_image=image, laplacian_gap=float(values[positive][0]), Pprofiles=[])
    phases = 2*np.pi*(sites+.5)/size
    U = np.exp(1j * sites @ phases.T) / np.sqrt(V)
    for load, model, role in models(cfg)[1:]:
        onsite_scalar, scalarpaths = scalar_paths(model)
        onsite = onsite_scalar * frame[0]
        primitive = np.broadcast_to(onsite, (V, 4, 4)).copy()
        paths = []
        for displacement, scalars in scalarpaths:
            axis = int(np.flatnonzero(displacement)[0]); length = int(displacement[axis])
            if length not in (1, 2):
                raise ArithmeticError('Unexpected original directed path')
            hop = np.einsum('m,mij->ij', scalars, frame)
            phase = np.exp(1j * phases @ displacement)
            primitive += phase[:,None,None]*hop + phase.conj()[:,None,None]*hop.conj().T
            for x, site in enumerate(sites):
                y = index[tuple((site+displacement)%size)]
                edges = []
                for step in range(length):
                    vertex=site.copy(); vertex[axis]=(vertex[axis]+step)%size
                    edges.append(3*index[tuple(vertex)]+axis)
                seam = -1 if site[axis]+length >= size else 1
                paths.append(dict(source=x, target=y, edges=tuple(edges), axis=axis, length=length,
                    hop=seam*hop, seam=seam, scalars=scalars, displacement=displacement))
        eigenvalues, eigenvectors = np.linalg.eigh(primitive)
        Pp = eigenvectors[:,:,:2] @ eigenvectors[:,:,:2].conj().transpose(0,2,1)
        P = np.einsum('xp,pab,yp->xayb', U, Pp, U.conj(), optimize=True).reshape(4*V,4*V)
        blocks = P.reshape(V,4,V,4).transpose(0,2,1,3)
        occupation = np.trace(blocks[np.arange(V),np.arange(V)], axis1=1, axis2=2).real
        S = np.diag(occupation) - np.einsum('xyab,yxba->xy', blocks, blocks, optimize=True).real
        profile: dict[str, Any] = dict(label=role, load=int(load), model=model, onsite=onsite,
            paths=paths, P=P, Pmomenta=Pp, S=S, blocks=blocks, primitive=primitive,
            occupied_vectors=eigenvectors[:,:,:2], onebody_gap=float(np.min(eigenvalues[:,2])),
            Pocc=np.einsum('xp,pak->xapk',U,eigenvectors[:,:,:2]).reshape(4*V,2*V),
            onsite_energy=float(np.einsum('ab,xba->',onsite,blocks[np.arange(V),np.arange(V)]).real))
        profile['hcentre'] = matter_hamiltonian(native, profile, np.zeros(3*V))
        profile['path_expectations'] = np.array([np.trace(p['hop'] @ blocks[p['target'],p['source']]) for p in paths])
        incidence = np.zeros((len(paths),3*V))
        for i,path in enumerate(paths):
            incidence[i,list(path['edges'])]=1
        profile['path_incidence'] = incidence
        profile['path_Q'] = incidence @ Q
        profile['path_norm2'] = np.einsum('tm,tm->t',profile['path_Q'],profile['path_Q'])
        native['Pprofiles'].append(profile)
    native['paths'] = [p['paths'] for p in native['Pprofiles']]
    native['matterhops'] = native['paths']
    native['charge_covariance'] = np.sum([p['load']*p['S'] for p in native['Pprofiles']], axis=0)
    native['Coulomb'] = native['UE']/native['a'] * float(np.trace(Lplus @ native['charge_covariance']))
    native['static_edges'] = tuple(sorted(set([0,1,2,3,3*(V//2),3*(V//2)+1,3*V-3,3*V-2,3*V-1])))
    errors = dict(divergence=float(np.linalg.norm(D@Q)), orthonormal=float(np.linalg.norm(Q.T@Q-np.eye(2*V+1))),
        projector=float(np.linalg.norm(PT-(np.eye(3*V)-D.T@Lplus@D))),
        harmonics=float(np.linalg.norm(PT@harmonics-harmonics)),
        mean_charge=max(float(np.max(abs(np.trace(p['blocks'][np.arange(V),np.arange(V)],axis1=1,axis2=2)-2))) for p in native['Pprofiles']),
        charge_sum=float(np.max(abs(native['charge_covariance'].sum(axis=0)))))
    if Q.shape!=(3*V,2*V+1) or max(errors.values())>1e-9:
        raise ArithmeticError(f'Physical quotient or full-sea neutrality failed: {errors}')
    native['geometry_errors']=errors
    return native


def packet_characteristic(coordinates: np.ndarray, radius: float = .125) -> np.ndarray:
    """Exact convergent moment series of normalized density (1-u²/R²)^4.

    Here |R*t|<1 for the literal finite one/two-link sums; 18 terms have a
    remainder below 1/(36!) per coordinate. No Gaussian replacement is used.
    """
    coordinates=np.asarray(coordinates)
    z=radius*coordinates
    if np.max(abs(z),initial=0.)>=1:
        raise ValueError('Moment-series receipt requires |R*t|<1')
    single=np.ones_like(z)
    for k in range(1,18):
        moment=np.exp(betaln(k+.5,5)-betaln(.5,5))
        single += (-1)**k * z**(2*k) * moment/math.factorial(2*k)
    return np.prod(single,axis=-1)


def _characteristic(native: dict[str, Any], coordinates: np.ndarray, variance: float, packet: bool) -> np.ndarray:
    if packet:
        return packet_characteristic(coordinates)
    return np.exp(-variance*np.einsum('...m,...m->...',coordinates,coordinates)/2)


def static_current_correlator(native: dict[str, Any], variance: float, packet: bool = False) -> dict[str, Any]:
    """Exact symmetrized equal-time connected CAR+field current on named edges.

    Species are independent conditional on theta: CAR Wick scales as63, while
    the shared-field classical covariance uses the SUM of all64 means.
    """
    selected=np.array(native['static_edges'])
    template=native['Pprofiles'][0]
    chosen=[i for i,p in enumerate(template['paths']) if any(e in native['static_edges'] for e in p['edges'])]
    incidence=template['path_incidence'][chosen][:,selected]
    ell=np.concatenate((incidence,-incidence))
    q=template['path_Q'][chosen]
    q=np.concatenate((q,-q))
    phases=_characteristic(native,q[:,None,:]+q[None,:,:],variance,packet)
    single=_characteristic(native,q,variance,packet)
    quantum=np.zeros(phases.shape,dtype=complex)
    totalmean=np.zeros(len(q),dtype=complex)
    for profile in native['Pprofiles']:
        paths=[profile['paths'][i] for i in chosen]
        x=np.array([p['source'] for p in paths]); y=np.array([p['target'] for p in paths])
        a=np.concatenate((x,y)); b=np.concatenate((y,x))
        hops=np.array([p['hop'] for p in paths]); hops=np.concatenate((hops,hops.conj().transpose(0,2,1)))
        blocks=profile['blocks']
        mean=np.einsum('tab,tba->t',hops,blocks[b,a])
        totalmean+=profile['load']*mean
        Pbc=blocks[b[:,None],a[None,:]]
        Rda=-blocks[b[None,:],a[:,None]]
        same=(b[None,:]==a[:,None])
        Rda+=same[:,:,None,None]*np.eye(4)
        wick=np.einsum('tij,tsjk,skl,tsli->ts',hops,Pbc,hops,Rda,optimize=True)
        quantum+=profile['load']*wick
    current=(1j*ell.T@(totalmean*single)).real
    moment=-np.real(ell.T@((quantum+totalmean[:,None]*totalmean[None,:])*phases)@ell)
    connected=(moment+moment.T)/2-np.outer(current,current)
    return dict(edges=selected, symmetrized_connected=connected, current=current,
        charge_connected=native['charge_covariance'],
        scope='Equal-time symmetrized literal Peierls current; no spectral or retarded inference')


def _evaluate(native: dict[str, Any], variance: float, packet: bool, correlator: bool) -> dict[str, Any]:
    if variance<=0 or not np.isfinite(variance):
        raise ValueError('Density variance must be finite and positive')
    current=np.zeros(3*native['V']); contact=np.zeros((3*native['V'],3*native['V']))
    matter=0.; onsite=0.; hopping=0.; rows=[]
    for profile in native['Pprofiles']:
        char=_characteristic(native,profile['path_Q'],variance,packet)
        z=profile['path_expectations']*char
        load=profile['load']; incidence=profile['path_incidence']
        contribution=load*(profile['onsite_energy']+2*np.sum(z.real))
        matter+=contribution; onsite+=load*profile['onsite_energy']; hopping+=load*2*np.sum(z.real)
        current+=load*(-2*z.imag)@incidence
        contact+=load*incidence.T@((-2*z.real)[:,None]*incidence)
        rows.append(dict(label=profile['label'],load=load,energy=float(contribution),path_characteristic=char))
    kinetic=native['UE']/native['a']*native['M']*(192 if packet else 1/(4*variance))
    energy=float(kinetic+native['Coulomb']+matter)
    result=dict(variance=variance,energy=energy,energy_per_site=energy/native['V'],
        energy_parts=dict(EMkinetic=float(kinetic),Coulomb=native['Coulomb'],matter=float(matter),onsite=float(onsite),hopping=float(hopping)),
        current=current,contact=contact,EMcov=variance*native['PT'],
        momentum_covariance=(192 if packet else 1/(4*variance))*native['PT'],
        Coulomb=native['Coulomb'],charge_covariance=native['charge_covariance'],profiles=rows,
        lambda_dressing=0.,normalization=1.,state=f'g(u) tensor product of64 full centre occupied{2*native["V"]}-orbital Slaters, then real Gauss lift')
    if correlator:
        result['static_current_connected']=static_current_correlator(native,variance,packet)
    return result


def evaluate_gaussian(native: dict[str, Any], variance: float, *, correlator: bool = True) -> dict[str, Any]:
    return _evaluate(native,float(variance),False,correlator)


def evaluate_old_packet(native: dict[str, Any], *, correlator: bool = True) -> dict[str, Any]:
    return _evaluate(native,1/704,True,correlator)


def optimize_variance(native: dict[str, Any]) -> dict[str, Any]:
    """Optimize the actual original-H expectation, not a fitted RPA frequency."""
    norms=native['Pprofiles'][0]['path_norm2']
    amplitude=sum(p['load']*2*p['path_expectations'].real for p in native['Pprofiles'])
    constant=native['Coulomb']+sum(p['load']*p['onsite_energy'] for p in native['Pprofiles'])
    kinetic=native['UE']/native['a']*native['M']/4
    def objective(logvariance: float) -> float:
        variance=np.exp(logvariance)
        return float(constant+kinetic/variance+amplitude@np.exp(-variance*norms/2))
    grid=np.linspace(-12,18,301)
    prices=np.array([objective(x) for x in grid]); i=int(np.argmin(prices))
    if i in (0,len(grid)-1):
        raise ArithmeticError('Variance optimum is not bracketed by the native actual-energy scan')
    result=minimize_scalar(objective,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-12})
    variance=float(np.exp(result.x))
    derivative=-kinetic/variance**2 - .5*np.sum(amplitude*norms*np.exp(-variance*norms/2))
    curvature=2*kinetic/variance**3+.25*np.sum(amplitude*norms**2*np.exp(-variance*norms/2))
    return dict(variance=variance,energy=float(result.fun),derivative=float(derivative),curvature=float(curvature),
        scanned_logvariance=[-12.,18.],infinite_spread_energy=float(constant),optimizer_success=bool(result.success))


def descriptor(native: dict[str, Any], variance: float) -> dict[str, Any]:
    M=native['M']; s2=variance; V=native['V']
    # Operator norm: each of2V occupied particles sees the original onebody row sum.
    potential=0.
    for p in native['Pprofiles']:
        rowcap=float(np.linalg.norm(p['onsite'],2))+2*sum(float(np.linalg.norm(q['hop'],2)) for q in p['paths'] if q['source']==0)
        potential+=p['load']*2*V*rowcap
    coulombcap=native['UE']/native['a']*(128**2*V)/native['laplacian_gap']
    kinetic_graph2=(native['UE']/native['a'])**2*M*(M+2)/(16*s2**2)
    lognorm=-M/4*math.log(2*math.pi*s2)
    return dict(N=native['N'],size=native['size'],V=V,M=M,a=native['a'],UE=native['UE'],
        g0_diagnostic=native['g0'],gs_diagnostic=native['gs'],chargedloads=[1,63],Qbasis=native['Q'],fullharmonics=native['harmonics'],
        s_density_variance=s2,lambda_dressing=0.,normalization=1.,log_amplitude_normalizer=lognorm,
        positive_covariance=dict(physical_Q_coordinate_eigenvalue=s2,multiplicity=M,edge_covariance='s² PT; null only in gauge-gradient directions'),
        normalizedstateformula='(2*pi*s²)^(-M/4) exp(-u.u/(4*s²)) tensor Ωactive(0) tensor Ωaux(0)^tensor63; θ=Q u; real Gauss lift exp(i sum_x chi_x rho_x), Pi_long=D.T Lplus rho',
        density_formula='product M normalized N(0,s²) densities in real physical Q coordinates',
        physical_charge='mobile sum64(n_x-2); Ref nref4 Qref1 background-4 net0; neutralheavy0 unchanged',
        lift_isometry='quotient additive real site gauge volume; operator rho, not a TF or mean density; Coulomb retained',
        moments=dict(norm_u_squared=M*s2,norm_u_fourth=M*(M+2)*s2**2,total_momentum_squared=M/(4*s2),kinetic_graph_norm_squared=kinetic_graph2),
        tail=dict(radius=math.sqrt(2*M*s2),norm_tail_upper=math.exp(-M*(2-1-math.log(2))/2),formula='P(||u||²>=c*M*s²)<=exp[-M(c-1-log c)/2], c=2'),
        graph_prices=dict(matter_operator_upper=potential,Coulomb_operator_upper=coulombcap,
            Href_graph_norm_squared_upper=2*kinetic_graph2+2*(potential+coulombcap)**2,
            scope='FiniteN original-H domain price only; H2 Gaussian Schwartz section and bounded finiteCAR Peierls/Coulomb'))


def _independent_controls(native: dict[str, Any], variance: float) -> dict[str, Any]:
    """Literal256-dimensional local CAR marginal and independent integrals."""
    profile=native['Pprofiles'][0]
    path=next(p for p in profile['paths'] if p['source']==0 and p['length']==1)
    x,y=path['source'],path['target']
    modes=np.r_[np.arange(4*x,4*x+4),np.arange(4*y,4*y+4)]
    C=profile['P'][np.ix_(modes,modes)]
    occupation,vectors=np.linalg.eigh(C)
    if np.min(occupation)<=0 or np.max(occupation)>=1:
        raise ArithmeticError('Selected exact CAR marginal is not faithful')
    annihilators=np.zeros((8,256,256),dtype=complex)
    for mode in range(8):
        bit=1<<mode
        for column in range(256):
            if column&bit:
                annihilators[mode,column^bit,column]=(-1)**((column&(bit-1)).bit_count())
    bilinears=np.einsum('aij,bjk->abik',annihilators.conj().transpose(0,2,1),annihilators,optimize=True)
    logit=(vectors*np.log((1-occupation)/occupation))@vectors.conj().T
    H=np.einsum('ab,abij->ij',logit,bilinears)
    eigenvalues,eigenvectors=np.linalg.eigh(H)
    weights=np.exp(-(eigenvalues-eigenvalues[0])); weights/=weights.sum()
    density=(eigenvectors*weights)@eigenvectors.conj().T
    J=np.zeros((8,8),dtype=complex)
    J[:4,4:]=1j*path['hop']; J[4:,:4]=J[:4,4:].conj().T
    B=np.diag([1.]*4+[-1.]*4)
    fockJ=np.einsum('ab,abij->ij',J,bilinears)
    fockB=np.einsum('ab,abij->ij',B,bilinears)
    def connected(operator: np.ndarray) -> float:
        return float(np.trace(density@operator@operator).real-np.trace(density@operator).real**2)
    wickJ=float(np.trace(J@C@J@(np.eye(8)-C)).real)
    wickB=float(np.trace(B@C@B@(np.eye(8)-C)).real)
    wrongB=float(np.trace(B@C@B@(np.eye(8)+C)).real)
    reconstructed=np.einsum('ij,abji->ab',density,bilinears).T
    path_index=next(i for i,p in enumerate(profile['paths']) if p is path)
    norm2=float(profile['path_norm2'][path_index])
    gaussian_integral=quad(lambda t: math.cos(math.sqrt(variance*norm2)*t)*math.exp(-t*t/2)/math.sqrt(2*math.pi),-np.inf,np.inf,epsabs=2e-13)[0]
    q=profile['path_Q'][path_index]
    packet_integral=math.prod(quad(lambda t:315/256*(1-t*t)**4*math.cos(.125*float(coordinate)*t),-1.,1.,epsabs=2e-14)[0] for coordinate in q)
    packet_norm=quad(lambda t:315/256*(1-t*t)**4,-1.,1.,epsabs=1e-14)[0]
    packet_kinetic=quad(lambda t:315/256*16*t*t*(1-t*t)**2/.125**2,-1.,1.,epsabs=1e-12)[0]
    edge=int(native['static_edges'][0]); direct=0.
    for p in native['Pprofiles']:
        fullJ=edge_current(native,p,np.zeros(3*native['V']),edge)
        support=np.flatnonzero(np.any(abs(fullJ)>0,axis=0))
        localJ=fullJ[np.ix_(support,support)]; localP=p['P'][np.ix_(support,support)]
        direct+=p['load']*float(np.trace(localJ@localP@localJ@(np.eye(len(support))-localP)).real)
    integrated=float(static_current_correlator(native,0.)['symmetrized_connected'][0,0])
    errors=dict(CAR_marginal=float(np.max(abs(reconstructed-C))),CAR_current_variance=abs(connected(fockJ)-wickJ),
        CAR_charge_variance=abs(connected(fockB)-wickB),Gaussian_characteristic=abs(gaussian_integral-math.exp(-variance*norm2/2)),
        packet_characteristic=abs(packet_integral-float(packet_characteristic(q))),
        packet_normalization=abs(packet_norm-1.),packet_kinetic=abs(packet_kinetic-192.),
        static_current_zero_width_limit=abs(integrated-direct))
    if max(errors.values())>1e-7:
        raise ArithmeticError(f'Independent literal CAR/integration controls failed: {errors}')
    return dict(errors=errors,local_CAR_dimension=256,local_current_variance_literal=connected(fockJ),
        local_charge_variance_literal=connected(fockB),wrong_plus_CAR_charge_variance=wrongB,
        actual_charge_variance_Wick=wickB,Gaussian_characteristic_quadrature=gaussian_integral,
        packet_characteristic_quadrature=packet_integral,packet_normalization_quadrature=packet_norm,
        packet_kinetic_quadrature=packet_kinetic,static_current_dense_onebody=direct,
        static_current_integrated_zero_width=integrated)


def _adverse(native: dict[str, Any], gaussian: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    profile=native['Pprofiles'][0]
    i=next(i for i,p in enumerate(profile['paths']) if p['length']==2)
    edges=profile['paths'][i]['edges']; s2=gaussian['variance']
    actual=float(np.exp(-s2*profile['path_norm2'][i]/2))
    wrong=float(np.exp(-s2*sum(native['PT'][e,e] for e in edges)/2))
    theta=np.zeros(3*native['V']); theta[edges[0]]=.23; theta[edges[1]]=-.19
    step=1e-4; direction=np.zeros_like(theta);direction[list(edges)]=[.7,-.4]
    exact_first=0.; exact_second=0.
    for p in native['Pprofiles']:
        for path,z in zip(p['paths'],p['path_expectations']):
            phase=np.exp(1j*sum(theta[list(path['edges'])])); velocity=sum(direction[list(path['edges'])])
            exact_first+=p['load']*(-2*(z*phase).imag)*velocity
            exact_second+=p['load']*(-2*(z*phase).real)*velocity**2
    def energy_at(t: float) -> float:
        return float(sum(p['load']*np.trace(matter_hamiltonian(native,p,theta+t*direction)@p['P']).real for p in native['Pprofiles']))
    values=[energy_at(t) for t in (-step,0.,step)]
    fd1=(values[2]-values[0])/(2*step); fd2=(values[2]-2*values[1]+values[0])/step**2
    # A literal2-site CAR Wick positive control, using the fullsea local marginal.
    a,b=0,1; p=profile['P']; wick=float((p[a,a]*(1-p[b,b])+p[a,b]*p[b,a]).real)
    density_joint=float((p[a,a]*p[b,b]-p[a,b]*p[b,a]).real)
    varied=evaluate_gaussian(native,s2*2,correlator=False)['energy']
    return dict(dropping_connected_Coulomb_changes_energy=native['Coulomb'],
        incorrect_auxiliary_squared_Coulomb=(63**2-63)*native['UE']/native['a']*float(np.trace(native['Lplus']@native['Pprofiles'][1]['S'])),
        twolink_cross_covariance=float(native['PT'][edges[0],edges[1]]),actual_twolink_characteristic=actual,
        dropping_twolink_cross_characteristic=wrong,characteristic_difference=actual-wrong,
        doubled_variance_energy_penalty=varied-gaussian['energy'],
        literal_current_finite_difference=dict(exact=exact_first,finite_difference=fd1,absolute_error=abs(exact_first-fd1)),
        literal_contact_finite_difference=dict(exact=exact_second,finite_difference=fd2,absolute_error=abs(exact_second-fd2)),
        CAR_local_exchange=dict(cdag_a_c_b_cdag_b_c_a=wick,n_a_n_b=density_joint,occupation_a=float(p[a,a].real),sum_error=abs(wick+density_joint-p[a,a].real)),
        Gaussian_at_old_variance_energy=evaluate_gaussian(native,1/704,correlator=False)['energy'],old_packet_energy=old['energy'])


def serialize(value: Any) -> Any:
    if isinstance(value,np.ndarray):
        if np.iscomplexobj(value):return dict(real=value.real.tolist(),imag=value.imag.tolist())
        return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,complex):return dict(real=value.real,imag=value.imag)
    raise TypeError(type(value).__name__)


def run(mean_path: Path, size: int = 6) -> dict[str, Any]:
    started=time.monotonic(); raw=mean_path.read_bytes(); mean=json.loads(raw)
    native=build_native(mean,size); optimized=optimize_variance(native)
    gaussian=evaluate_gaussian(native,optimized['variance']); old=evaluate_old_packet(native)
    adverse=_adverse(native,gaussian,old)
    independent=_independent_controls(native,optimized['variance'])
    improvement=old['energy']-gaussian['energy']
    if improvement<=0 or native['Coulomb']<=0 or adverse['characteristic_difference']==0 or adverse['doubled_variance_energy_penalty']<=0:
        raise ArithmeticError('Actual same-H variational improvement or adverse discriminator failed')
    if adverse['CAR_local_exchange']['sum_error']>1e-12:
        raise ArithmeticError('Literal CAR exchange failed')
    image_error=max(row['exact_scale_error_upper'] for row in native['frame_image']['scales'])
    scalarcap=sum(p['load']*2*native['V']*(abs(float(scalar_paths(p['model'])[0]))+2*sum(float(np.linalg.norm(s,1)) for _,s in scalar_paths(p['model'])[1])) for p in native['Pprofiles'])
    frameprice=scalarcap*image_error
    return dict(status='native_same_original_Href_quantumEM_variational_improvement',
        scope=f'Actual N{size} finiteCAR+realEM Gaussian integrals; no promotion to macro222, globalground, photonvacuum, pole or speed',
        input_sha256={mean_path.name:hashlib.sha256(raw).hexdigest()},source_sha256_at_import=SOURCE_HASHES,
        independent_controls=independent,
        source_snapshots_at_import={name:value.decode() for name,value in SOURCE_BYTES.items()},
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform(),argv=sys.argv),
        descriptor=descriptor(native,optimized['variance']),geometry_errors=native['geometry_errors'],
        diagnostic_frame_image=native['frame_image'],diagnostic_fixedstate_matter_image_energy_price=frameprice,
        original_Href='UE/a*(sum allM PiQ² + rho.T Lplus rho)+full64 original onsite+Peierls paths; no bareB² or compact/integer electric substitution',
        optimization=optimized,Gaussian=gaussian,old_packet=old,
        old_packet_definition=dict(radius=.125,density='315/(256*R)*(1-u²/R²)^4 on[-R,R], product over SAME real Q',normalization=1.,density_variance=1/704,kinetic_per_coordinate=192.,moment_series_absolute_remainder_upper=native['M']/math.factorial(36)),
        same_native_improvement=dict(total=improvement,per_site=improvement/native['V']),adverse=adverse,
        profile_controls=[dict(label=p['label'],load=p['load'],particles=2*native['V'],onebody_gap=p['onebody_gap'],
            connected_Coulomb=native['UE']/native['a']*p['load']*float(np.trace(native['Lplus']@p['S'])),
            centre_filled_energy=float(p['load']*np.trace(p['hcentre']@p['P']).real),
            binary_coefficients=[dict(displacement=d,scalars=s) for d,s in scalar_paths(p['model'])[1]]) for p in native['Pprofiles']],
        elapsed_seconds=time.monotonic()-started)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mean',type=Path,default=ROOT/'induced_phi_ir_mean_certificate_normal_transfer_final_receipt.json')
    parser.add_argument('--output',type=Path,default=ROOT/'induced_phi_em_gaussian_variational_final_receipt_20261008.json')
    parser.add_argument('--size',type=int,default=6)
    args=parser.parse_args(); result=run(args.mean,args.size)
    args.output.write_text(json.dumps(result,default=serialize,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],variance=result['optimization']['variance'],Gaussian_energy_per_site=result['Gaussian']['energy_per_site'],old_packet_energy_per_site=result['old_packet']['energy_per_site'],improvement=result['same_native_improvement'],Coulomb=result['Gaussian']['Coulomb'],elapsed_seconds=result['elapsed_seconds']),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
