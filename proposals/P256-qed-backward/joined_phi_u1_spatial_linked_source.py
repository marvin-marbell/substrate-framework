"""Finite-density native plaquette Slater circuit; conditional frozen-Phi body only.

The proof below prices the COMPLETE body action, rather than (PHP)^2. The
finite float observations are not interval enclosures. No old T lift is used.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.abc
import importlib.machinery
import json
from fractions import Fraction
from pathlib import Path
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_spatial_linked_'
SOURCE_BYTES = {Path(__file__).name: Path(__file__).read_bytes()}

class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes()
        SOURCE_BYTES[Path(self.path).name] = raw
        return self.source_to_code(raw, self.path)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.origin and Path(spec.origin).parent == ROOT and spec.origin.endswith('.py'):
            spec.loader = Loader(fullname, spec.origin)
        return spec

finder = Finder()
sys.meta_path.insert(0, finder)
import induced_phi_nonuniform_field_car as field
sys.meta_path.remove(finder)
T = 1 / 128
COPIES = 64
GRID = 32
LOOP_R = 1 / 4
LOOP_Z = 1 + LOOP_R**2 / 2
PROOF = {
    'state': 'For every even N>=4 tile each xy plane by disjoint 2x2 plaquettes. Each cell has16 orbitals and8 occupied columns W0 of onsite Gamma0 negative projector. Four ordered native link gates g=I+(cos(t)-1)J-sin(t)S, S=Gamma0 hci/(ci sqrt(gs)), Sdag=-S, S^2=-J, act on the corresponding two sites. U=g3 g2 g1 g0; wedge(UW0) is exactly normalized pointwise. Tensor all V/4 cells then active1+aux63 copies over ONE shared native Haar field. This is the actual complete exterior wedge, not one PH per species, not an independent-field product. Angle t=1/128 is a trial shape, not a physical coupling adoption.',
    'paths': 'Hamiltonian remains the full original onsite and all forward/reverse length1 and ordered length2 paths with AP seam (-1) for each path crossing box boundary. Circuit uses only internal positive-edge cell paths; no seam crosses a cell. All intercell Hamiltonian matrix blocks are retained, though expectation is zero because every cell has exactly8 particles in each species. Their action is NOT zero.',
    'gauge': 'Every gate covaries g(theta+lambda_x-lambda_y)=G g(theta) Gdag. W0 wedge transforms by exp(2i sum_x lambda_x); subtracting uniform2-per-copy background restores gauge invariance. Ref4 is one fixed filled origin spectator, rho=sum64(n_s,x-2)+(nRef-4)delta_x0. Neutral has canonical2V and charge0; it is the exact frozen neutral negative physical-energy Fourier sea of -Lambda h_neutral, unchanged and an exact eigenstate. CommonZ4 is trivial. State is periodic in integer edge characters and is in beta0; other beta sectors are not minimized.',
    'electric': 'On this compatible Gauss sector HE64+HC64=64 sum_e Efull_e^2. Acting on a covariant circuit restores full native incidence, so Efull=-i full edge derivative of its background-centered lift. Local R_e=-i Udag d_e U. Mean Tr(P0 R_e)=0 by Clifford-even spin trace and Tr R_e=0. All Wick crosscopy terms are evaluated as n Var(R)+n^2 mean(R)^2, not discarded by independence. Density mean2 similarly follows even Clifford algebra; density fluctuations and longitudinal energy remain nonzero.',
    'action_bound': 'A cell contains8 occupied orbitals. For its single edge gate ||gprime||<=s=|sin t|, ||gsecond||<=s, so the wedge derivative norm<=8s and second derivative norm<=8s+56s^2 by the exact exterior product rule. Across64 copies the second derivative norm<=64(8s+56s^2)+64*63(8s)^2. There are4(V/4) active edges. The scalar-loop price is stated in shared_loop_action. Triangle with complete matter action gives ||(Hbody-Eref)Psi||<=2V sum_charged_roles load_s(Lh_s+|onsite_s|sqrt(g0))+electric_bound; Lh_s=|onsite_s|sqrt(g0)+2 sum_paths (|c0_s|sqrt(g0)+|ci_s|sqrt(gs)). Active1 and auxiliary63 have different onsite masses; equal hopping paths do NOT permit64 active Hamiltonians. The directed hop is nonnormal; its operator bound is the SUM, not sqrt(c0^2g0+ci^2gs). This prices every onsite/intercell/ordered exterior path and all second derivative/crosscopy terms. Variance<=this squared, since variance is inf_c ||(H-c)Psi||^2. Neutral frozen eigenenergy is removed exactly in Eref; frozen bosonic potentials are constants. No newJordan fullPhi/SU4 action bound is claimed.',
    'stationarity_lower': 'Select z-nearest links between successive xy-plane cells. Electric action and intracell matter preserve8 particles/cell/copy. Intercell hopping changes cell particle numbers and is orthogonal to them, so it cannot cancel against any electric term. For an ordered adjacent z-cell pair, A=Pempty_a h_ab Poccupied_b. At onsite reference ||A0||HS=sqrt(8 ci_z^2 gs), independent of link phases (four distinct vertical edges). ||h_ab||HS=4 sqrt(c0_z^2 g0+ci_z^2 gs). Since ||P-P0||<=2||U-I||<=8t, ||A-A0||HS<=16t ||h_ab||HS. Hence variance>=64*(V/4)*2*max(0, sqrt(8 ci_z^2 gs)-64t sqrt(c0_z^2 g0+ci_z^2 gs))^2. At N4,N6 no length2 z path connects adjacent planes; this selected sector is unique and AP signs do not alter norms. It proves a positive uniform extensive complete-H residual when the bracket is positive; observations are float64 displays of this analytic bound, not certified interval arithmetic.',
    'integration': 'Matter energy uses the actual load-weighted active1 and auxiliary63 onebody Hamiltonians; intercell expectation zero. Gauge fixing three cell edges leaves one uniform plaquette flux. Cell projectors and derivatives are finite Laurent polynomials (four gates);32-point trapezoidal integration resolves local energies and PH pair covariance exactly in algebra, evaluated in float64. Pointwise complete matter variances Tr(P h_s(1-P)h_s) are retained separately per role and are NOT Haar full-H variances. For the conditional overlap diagnostic o(phi)=|<wedge(U(phi)W0),wedge(U(phi+.73)W0)>|^2, use the once-shared normalized weight w=A^2: mean(w o^64), not mean((w o)^64). The conservative degree bound2050 is below the4096-point integration period. This is a conditional overlap observable under the source probability, not the translated full-source fidelity including shifted scalar factors.',
    'limits': 'Conditional frozen Clifford Phi and identity SU4, beta0 body only. No fullPhi radial/SU4/Haar lift or full-beta minimizer; old-law T CANNOT certify NEW Jordan. No stationary ground, stiffness, pole, common cone, originalSOURCE or campaign closure.',
    'shared_loop_amplitude': 'The corrected real trial multiplies the complete pointwise-normalized64-copy circuit by ONE shared A(phi)=(1+cos(phi)/4)/sqrt(33/32) per disjoint plaquette. Integral A^2 dphi/(2pi)=1 exactly; loop characters are zero-homology faces. This is a new normalized source shape, not a bareB2 Hamiltonian term. Scalar derivatives are retained: cell electric expectation64[ A^2(64 sum VarR+64^2 sum meanR^2)+4(Aprime)^2 ]; HC has only weighted matter density covariance. Circuit PH-loop trace alone was not a valid witness in the first executed attempt.',
    'shared_loop_action': 'For each active edge use full second derivative of A times64-copy wedge: ||d2(AW)||L2 <= r/sqrt(Z)+2r/sqrt(Z)*512s+(1+r)/sqrt(Z)*(512s+261632s^2). Other cell factors have L2 norm1. Electric action norm <=64V times this bracket, including Asecond, gradient-Wick and all crosscopy terms. Thus the reported corrected complete-H bound replaces the unweighted formula in action_bound. Orthogonal exterior lower bound remains unchanged because the bound is pointwise and A^2 has integral1.',
    'shared_copy_correlation': 'For distinct charged copies, the actual conditional PH expectation f(phi)=Tr(Pplus P(phi)) gives connected common-field covariance mean(w(f-mean(wf))^2). The independently sampled-field product instead has zero connected covariance. This is a correlation witness, not an entanglement claim. The overlap difference may be blind; no huge threshold or repetition of A in its power is accepted. Repeated-A normalization remains a separate wrongcase.',
}

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def archive(path: Path, table: dict[str, bytes]) -> dict[str, Any]:
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(table.items()):
            z.writestr(name, raw)
    return {'path': str(path), 'sha256': sha(path.read_bytes()), 'members': {n:sha(b) for n,b in table.items()}}

def cell(theta: np.ndarray, gam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = np.kron(np.eye(4), (np.eye(4)-gam[0])/2)
    U = np.eye(16, dtype=complex)
    du = np.zeros((4,16,16), complex)
    # Literal positive edges: bottom x, right y, top x, left y.
    for k,(x,y,axis) in enumerate(((0,1,1),(1,3,2),(2,3,1),(0,2,2))):
        a,b = slice(4*x,4*x+4),slice(4*y,4*y+4)
        S = np.zeros((16,16), complex)
        v = np.exp(1j*theta[k])*1j*gam[0]@gam[axis]
        S[a,b] = v; S[b,a] = -v.conj().T
        J = np.zeros((16,16), complex); J[a,a]=np.eye(4);J[b,b]=np.eye(4)
        g = np.eye(16)+(np.cos(T)-1)*J-np.sin(T)*S
        dS = np.zeros_like(S);dS[a,b]=1j*v;dS[b,a]=1j*v.conj().T
        du = np.einsum('ij,ejk->eik',g,du)
        du[k] += -np.sin(T)*dS@U
        U = g@U
    return U@p0@U.conj().T,U,du

def tiles(native: dict[str, Any]) -> list[list[int]]:
    N=native['N']; idx={tuple(v):i for i,v in enumerate(native['sites'])}
    return [[idx[(x,y,z)],idx[(x+1,y,z)],idx[(x,y+1,z)],idx[(x+1,y+1,z)]]
            for x in range(0,N,2) for y in range(0,N,2) for z in range(N)]

def literal(native: dict[str, Any], profile: dict[str, Any], theta: np.ndarray) -> np.ndarray:
    V=native['V']; h=np.kron(np.eye(V),float(profile['onsite'])*native['frame'][0]).astype(complex)
    for displacement,c in profile['raw_paths']:
        support=[p for p in native['Pprofiles'][0]['paths'] if np.array_equal(displacement,p['displacement'])]
        assert len(support)==V
        for p in support:
            a,b=slice(4*p['source'],4*p['source']+4),slice(4*p['target'],4*p['target']+4)
            phase=1. if profile['scale']=='heavy' else np.exp(1j*sum(theta[e] for e in p['edges']))
            hop=p['seam']*phase*np.einsum('m,mij->ij',c,native['frame'])
            h[a,b]+=hop;h[b,a]+=hop.conj().T
    return -h if profile['scale']=='heavy' else h

def local_h(phi: float, profile: dict[str, Any], frame: np.ndarray) -> np.ndarray:
    h=np.kron(np.eye(4),float(profile['onsite'])*frame[0]).astype(complex)
    for k,(x,y,axis) in enumerate(((0,1,0),(1,3,1),(2,3,0),(0,2,1))):
        row=next(r for r in profile['paths'] if r['axis']==axis and r['length']==1)
        block=np.exp(1j*(phi if k==3 else 0))*(float(row['c0'])*frame[0]+1j*float(row['ci'])*frame[axis+1])
        a,b=slice(4*x,4*x+4),slice(4*y,4*y+4)
        h[a,b]+=block;h[b,a]+=block.conj().T
    return h

def neutral_state(native: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    N=native['N'];V=native['V'];frame=native['frame']
    k=2*np.pi*(native['sites']+.5)/N
    primitive=np.broadcast_to(float(profile['onsite'])*frame[0],(V,4,4)).copy().astype(complex)
    for displacement,c in profile['raw_paths']:
        hop=np.einsum('m,mij->ij',c,frame);phase=np.exp(1j*k@displacement)
        primitive+=phase[:,None,None]*hop+phase.conj()[:,None,None]*hop.conj().T
    band=np.sqrt(np.trace(primitive@primitive,axis1=1,axis2=2).real/4)
    # Exact definition uses positive unscaled Clifford projector, occupying
    # the negative physical -Lambda h bands; no rounded eigenvectors define it.
    pm=(np.eye(4)+primitive/band[:,None,None])/2
    fourier=np.exp(1j*native['sites']@k.T)/np.sqrt(V)
    P=np.einsum('xp,pab,yp->xayb',fourier,pm,fourier.conj(),optimize=True).reshape(4*V,4*V)
    h=literal(native,profile,np.zeros(3*V));other=literal(native,profile,np.ones(3*V)*.27)
    result={'state':'Exact wedge of upper unscaled Clifford bands at ALL AP momenta; physical -Lambda sign retained. Projector (I+h(k)/E(k))/2 defines full Fourier exterior, not a PH truncation.','rank_observed':float(np.trace(P).real),'rank_exact':2*V,'projector_defect':float(np.max(abs(P@P-P))),'unscaled_eigen_commutator_defect':float(np.max(abs(h@P-P@h))),'unscaled_energy':float(np.trace(P@h).real),'exact_physical_energy_expression':'-2^8192 * 2 sum_ALL_AP_k sqrt(sum_mu g_mu xi_neutral_mu(k)^2)','unscaled_variance_observed':float(np.trace(P@h@(np.eye(4*V)-P)@h).real),'exact_variance':0,'neutral_U1_independence_defect':float(np.max(abs(h-other))),'AP_band_positive':band.tolist()}
    assert result['projector_defect']<1e-9 and result['unscaled_eigen_commutator_defect']<1e-8 and result['neutral_U1_independence_defect']==0
    return result

def shared_overlap_observation(gam: np.ndarray, w0: np.ndarray) -> dict[str, Any]:
    angles = 2*np.pi*np.arange(4096)/4096
    weight = (1+LOOP_R*np.cos(angles))**2/LOOP_Z
    overlap = []
    for phi in angles:
        u = cell(np.array([0.,0.,0.,phi]),gam)[1]
        v = cell(np.array([0.,0.,0.,phi+.73]),gam)[1]
        overlap.append(float(abs(np.linalg.det((u@w0).conj().T@(v@w0)))**2))
    values = np.array(overlap)
    shared = float(np.mean(weight*values**COPIES))
    independent = float(np.mean(weight*values)**COPIES)
    return dict(grid=4096,degree_upper=2050,normalizer_mean=float(np.mean(weight)),
                shared_mean_weight_once=shared,independent_power_of_weighted_mean=independent,
                difference=shared-independent,
                constant_overlap_weight_once=float(np.mean(weight)),
                wrong_constant_overlap_repeated_weight=float(np.mean(weight**COPIES)),
                semantics='Conditional normalized Slater overlap under ONE common source probability; may be blind, not full-source translated fidelity.')


def run_case(data: dict[str, Any], N: int) -> dict[str, Any]:
    native=field.full.supplier.quantum.build_native(data['mean'],N)
    V=native['V']; cells=tiles(native); frame=native['frame']; gam=frame/np.sqrt(np.trace(frame@frame,axis1=1,axis2=2).real/4)[:,None,None]
    charged = data['profiles'][1:]
    assert [int(r['load']) for r in charged] == [1,63]
    p = charged[0]
    assert p['paths'] == charged[1]['paths']
    assert p['onsite'] != charged[1]['onsite']
    assert sum(int(r['load']) for r in charged) == COPIES
    p0=np.kron(np.eye(4),(np.eye(4)-gam[0])/2); vals,w=np.linalg.eigh(p0);w=w[:,vals>.5]
    phis=2*np.pi*np.arange(GRID)/GRID
    weights=(1+LOOP_R*np.cos(phis))**2/LOOP_Z
    scalar_grad=LOOP_R**2*np.sin(phis)**2/LOOP_Z
    energy=[];elec=[];ph=[];over=[];berries=[];hcs=[];defects=[]
    energy_by_role: list[list[float]] = [[] for _ in charged]
    # Use actual Green matrix of this N, not an N4 replacement.
    cs=cells[0]; lp=native['Lplus'][np.ix_(cs,cs)]
    for phi in phis:
        P,U,du=cell(np.array([0.,0.,0.,phi]),gam)
        R=-1j*np.einsum('ij,ejk->eik',U.conj().T,du)
        means=np.trace(np.einsum('ij,ejk->eik',p0,R),axis1=1,axis2=2)
        ev=sum(np.trace(p0@r@(np.eye(16)-p0)@r).real for r in R)
        blocks=P.reshape(4,4,4,4).transpose(0,2,1,3)
        den=np.trace(blocks[np.arange(4),np.arange(4)],axis1=1,axis2=2).real
        cov=np.diag(den)-np.einsum('xyij,yxji->xy',blocks,blocks).real
        hc=float(np.sum(lp*cov))
        j=len(energy);wt=weights[j]
        local_energies = [wt*float(np.trace(P@local_h(phi,r,frame)).real) for r in charged]
        for values, value in zip(energy_by_role,local_energies):
            values.append(value)
        energy.append(sum(int(r['load'])*value for r,value in zip(charged,local_energies)))
        elec.append(float(wt*(COPIES*ev+COPIES**2*np.sum(abs(means)**2))+4*scalar_grad[j]))
        hcs.append(wt*COPIES*hc); berries.append(float(max(abs(means))))
        ph.append(wt*float(np.trace((np.eye(16)-p0)@P).real))
        other=cell(np.array([0.,0.,0.,phi+.73]),gam)[1]
        over.append(float(abs(np.linalg.det((U@w).conj().T@(other@w)))**2))
        defects.append(max(float(np.max(abs(U.conj().T@U-np.eye(16)))),float(np.max(abs(P@P-P))),float(max(abs(den-2)))))
    # Actual nonconstant literal full matrix; its exterior is never projected out.
    rng=np.random.default_rng(2221010+N);theta=.23*rng.normal(size=3*V)
    P=np.zeros((4*V,4*V),complex); cellid=np.zeros(V,int)
    allR=np.zeros((V,16,16),complex)
    for c,sites in enumerate(cells):
        edges=[3*sites[0],3*sites[1]+1,3*sites[2],3*sites[0]+1]
        pc,u,du=cell(theta[edges],gam);inds=np.array([4*x+j for x in sites for j in range(4)])
        P[np.ix_(inds,inds)]=pc;cellid[sites]=c
        allR[4*c:4*c+4]=-1j*np.einsum('ij,ejk->eik',u.conj().T,du)
    h=literal(native,p,theta)
    mask=np.repeat(cellid,4)[:,None]==np.repeat(cellid,4)[None,:]
    complement=np.eye(4*V)-P
    role_variances: list[dict[str, Any]] = []
    for r in charged:
        role_h = h if r is p else literal(native,r,theta)
        role_within = np.where(mask,role_h,0)
        role_external = role_h-role_within
        role_variances.append(dict(role=r['role'],load=int(r['load']),
            full=float(np.trace(P@role_h@complement@role_h).real),
            PHP=float(np.trace(P@role_within@complement@role_within).real),
            exterior=float(np.trace(P@role_external@complement@role_external).real)))
    exterior = role_variances[0]['exterior']
    decomposition = max(abs(r['full']-r['PHP']-r['exterior']) for r in role_variances)
    # Literal local gauge covariance, including full Hamiltonian AP/two paths.
    lam=.3*rng.normal(size=V);shift=native['D'].T@lam;G=np.exp(1j*np.repeat(lam,4))
    hg=literal(native,p,theta+shift)
    gauge=float(np.max(abs(hg-G[:,None]*h*G.conj()[None,:])))
    sourceg=0.
    for sites in cells:
        edges=[3*sites[0],3*sites[1]+1,3*sites[2],3*sites[0]+1]
        pc=cell((theta+shift)[edges],gam)[0];old=cell(theta[edges],gam)[0];gg=np.exp(1j*np.repeat(lam[sites],4))
        sourceg=max(sourceg,float(np.max(abs(pc-gg[:,None]*old*gg.conj()[None,:]))))
    # Exact geometric homology, not a Fourier label inferred from N4.
    character=np.zeros(3*V,int);sites=cells[0];edges=[3*sites[0],3*sites[1]+1,3*sites[2],3*sites[0]+1]
    character[edges]=[1,1,-1,-1]
    homology=[int(character[axis::3].sum()) for axis in range(3)]
    winding=np.zeros(3*V,int)
    idx={tuple(v):i for i,v in enumerate(native['sites'])}
    for x in range(N):winding[3*idx[(x,0,0)]]=1
    g0=native['g0'];gs=native['gs']
    role_prices: list[dict[str, Any]] = []
    for r,values in zip(charged,energy_by_role):
        onsite = abs(float(r['onsite']))*np.sqrt(g0)
        Lh = onsite+2*sum(abs(float(q['c0']))*np.sqrt(g0)+abs(float(q['ci']))*np.sqrt(gs) for q in r['paths'])
        role_prices.append(dict(role=r['role'],load=int(r['load']),onsite_exact=str(r['onsite']),
            cell_energy_weighted_integrand=values,
            matter_energy=float((V//4)*np.mean(values)),onsite_norm=float(onsite),h_norm=float(Lh),
            matter_action_norm_upper=float(2*V*(Lh+onsite))))
    s=np.sin(T)
    derivative_bracket=LOOP_R/np.sqrt(LOOP_Z)+2*LOOP_R/np.sqrt(LOOP_Z)*512*s+(1+LOOP_R)/np.sqrt(LOOP_Z)*(512*s+261632*s*s)
    ebound=64*V*derivative_bracket
    mbound=sum(r['load']*r['matter_action_norm_upper'] for r in role_prices)
    z=next(r for r in p['paths'] if r['axis']==2 and r['length']==1)
    bracket=np.sqrt(8*float(z['ci'])**2*gs)-64*T*np.sqrt(float(z['c0'])**2*g0+float(z['ci'])**2*gs)
    lower=64*(V//4)*2*max(0.,bracket)**2
    loop=complex(np.mean(np.array(ph)*np.exp(-1j*phis)))
    overlaps = shared_overlap_observation(gam,w)
    conditional_PH = np.array(ph)/weights
    PH_mean = float(np.mean(weights*conditional_PH))
    shared_PH_covariance = float(np.mean(weights*(conditional_PH-PH_mean)**2))
    # Paid generic Berry crosscopy control: multiply each normalized copy by
    # exp(i plaquette_flux). This is an actual periodic, contractible phase.
    berry_missing=64*64*63*4*(V//4)
    result = {
        'N':N,'V':V,'charged_species':COPIES,'particles_per_species':2*V,
        'neutral_particles':2*V,'Ref_particles':4,'total_particles':65*2*V+4,
        'cells':V//4,'angle_exact':'1/128',
        'frozen_G_display':{'g0':float(g0),'gs':float(gs),'semantics':'Finite frozen-field display, not an adopted quantum mean or root existence certificate.'},
        'PH_number_per_species':float(np.mean(ph)*(V//4)),'PH_density_per_site':float(np.mean(ph)/4),
        'norm_projector_density_defect':max(defects),'Berry_max':max(berries),
        'gauge_H_defect':gauge,'gauge_source_defect':sourceg,
        'matter_energy':float((V//4)*np.mean(energy)),
        'HEHC_energy':float(64*(V//4)*np.mean(elec)),
        'HC_energy':float(64*(V//4)*np.mean(hcs)),
        'HE_energy':float(64*(V//4)*(np.mean(elec)-np.mean(hcs))),
        'reference_charged_energy':float(-2*V*np.sqrt(g0)*sum(int(r['load'])*float(r['onsite']) for r in charged)),
        'charged_role_observations':role_prices,
        'loop_PH_character_real':loop.real,'loop_PH_character_imag':loop.imag,
        'contractible_character':{'edges':edges,'coefficients':[1,1,-1,-1],'divergence_max':float(max(abs(native['D']@character))),'winding_numerator':homology,'norm_squared':int(character@character),'harmonic_norm_squared':float(np.linalg.norm(native['harmonics'].T@character)**2)},
        'winding_control':{'divergence_max':float(max(abs(native['D']@winding))),'winding':[int(winding[a::3].sum())//N for a in range(3)],'norm_squared':int(winding@winding)},
        'pointwise_matter_variance_by_role':role_variances,
        'pointwise_matter_variance_total_conditional_field':sum(r['load']*r['full'] for r in role_variances),
        'omitted_exterior_variance_per_copy':exterior,'variance_decomposition_defect':decomposition,
        'complete_H_variance_upper_bound':float((ebound+mbound)**2),
        'electric_action_norm_bound':float(ebound),'complete_matter_action_norm_bound':float(mbound),
        'complete_H_variance_lower_bound':float(lower),
        'stationarity':'FAIL: positive orthogonal intercell leakage' if lower>0 else 'not certified',
        'conditional_shared_overlap':overlaps,
        'shared_PH_pair_connected_covariance_per_cell':shared_PH_covariance,
        'phase_Berry_control_missing_crosscopy_energy_exact':berry_missing,
        'raw_cell_phi':phis.tolist(),'raw_cell_PH_weighted_integrand':ph,
        'raw_cell_total_charged_matter_energy_weighted_integrand':energy,
        'raw_cell_HEHC_weighted_integrand':elec,'raw_cell_onecopy_overlap_squared':over,
        'profiles':[{'role':r['role'],'load':r['load'],'onsite':str(r['onsite']),'paths':[{k:str(v) for k,v in q.items()} for q in r['paths']]} for r in data['profiles']],
    }
    control_phi=2*np.pi*np.arange(512)/512
    scalar=(1+LOOP_R*np.cos(control_phi))/np.sqrt(LOOP_Z)
    result['loop_probability_character_exact']='8/33'
    result['loop_probability_character_observed']=float(np.mean(weights*np.cos(phis)))
    result['shared_scalar_norm_observed']=float(np.mean(weights))
    result['wrong_replica_shared_scalar_norm_per_cell']=float(np.mean(scalar**128))
    result['wrong_independent_scalar_norm_per_cell']=float(np.mean(scalar**2)**64)
    result['overlap_control_semantics'] = overlaps['semantics']
    result['shared_scalar_HE_energy']=float(64*(V//4)*np.mean(4*scalar_grad))
    result['complete_body_energy_relative_neutral_eigenenergy']=result['matter_energy']+result['HEHC_energy']
    result['delta_energy_to_onsite_charged_reference']=result['complete_body_energy_relative_neutral_eigenenergy']-result['reference_charged_energy']
    result['neutral_full_Fourier_exterior']=neutral_state(native,data['profiles'][0])
    result['operator_path_counts']={'forward_ordered_paths':sum(len(native['sites']) for _ in p['raw_paths']),'all_reverse_retained':True,'two_link_forward':V*sum(r['length']==2 for r in p['paths']),'AP_seam_paths':sum(q['seam']==-1 for q in native['Pprofiles'][0]['paths'])}
    unweighted_matter=float((V//4)*np.mean(np.array(energy)/weights))
    unweighted_electric=float(64*(V//4)*np.mean((np.array(elec)-4*scalar_grad)/weights))
    unweighted_energy=unweighted_matter+unweighted_electric
    unweighted_PH=np.array(ph)/weights
    unweighted_loop=complex(np.mean(unweighted_PH*np.exp(-1j*phis)))
    loop_cost=result['complete_body_energy_relative_neutral_eigenenergy']-unweighted_energy
    result['same_H_A_equals_1_control']={'matter_energy':unweighted_matter,'HEHC_energy':unweighted_electric,'complete_body_energy_relative_neutral_eigenenergy':unweighted_energy,'delta_energy_to_onsite_reference':unweighted_energy-result['reference_charged_energy'],'PH_density_per_site':float(np.mean(unweighted_PH)/4),'PH_loop_character_real':unweighted_loop.real,'PH_loop_character_imag':unweighted_loop.imag,'complete_H_variance_lower_bound':float(lower),'complete_H_variance_upper_bound':float((mbound+64*V*(512*s+261632*s*s))**2),'stationarity':'FAIL: same positive orthogonal intercell leakage'}
    result['imposed_loop_delta_same_H']=float(loop_cost)
    result['retained_cheaper_trial']='A=1' if loop_cost>0 else 'A=(1+cos(phi)/4)/sqrt(33/32)'
    result['imposed_loop_interpretation']='Explicitly imposed contractible variational amplitude, not evidence of actual-ground flux ordering. Both actual normalized candidates fail stationarity through intercell leakage.'
    assert gauge<1e-9 and sourceg<1e-9 and max(defects)<1e-9 and max(berries)<1e-9
    assert exterior>1 and decomposition<1e-6 and lower>0
    assert homology==[0,0,0] and max(abs(native['D']@character))==0
    assert abs(loop)>1e-12 and shared_PH_covariance>0
    return result

def run(output: Path) -> dict[str, Any]:
    started=time.monotonic()
    names=['induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json','induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json','induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json']
    inputs={n:(ROOT/n).read_bytes() for n in names}
    saved=json.loads(inputs[names[2]])['input_descriptor']['exact_NEW_allq_BO_mean']
    centres=[float(Fraction(v)) for v in saved['centre_exact_rational']]
    captured={**SOURCE_BYTES,**field.CAPTURED,**inputs}
    pre=archive(output.with_name(output.stem+'_preexecution.zip'),captured)
    data=field.load_mean(g_override=[centres[0],centres[1],centres[1],centres[1]],mean_descriptor=saved)
    with threadpool_limits(limits=1):cases=[run_case(data,N) for N in (4,6)]
    executed=archive(output.with_name(output.stem+'_executed.zip'),captured)
    result={'kind':'ACTUAL_EXTENSIVE_NORMALIZED_NATIVE_LINKED_PLAQUETTE_SLATER_AND_COMPLETE_BODY_ACTION_BOUND','proof':PROOF,'cases':cases,'preexecution':pre,'executed':executed,'runtime':{'python':sys.executable,'version':sys.version,'numpy':np.__version__},'source_inputs_unchanged':all((ROOT/n).read_bytes()==raw for n,raw in captured.items()),'elapsed_seconds':time.monotonic()-started,'next_source_decision':'Positive extensive intercell residual rules out this disjoint-cell stationary state, not the physics. Replace the cell-number superselection by a gauge-covariant overlapping/intercell canonical circuit or a whole-exterior Slater response, retaining these local contractible channels and the priced full action. FullPhi/SU4 newJordan lift and beta minimization remain unpaid.'}
    assert result['source_inputs_unchanged']
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    r=run(args.output)
    print(json.dumps({'receipt':str(args.output),'sha256':sha(args.output.read_bytes()),'cases':[{k:c[k] for k in ('N','PH_density_per_site','loop_PH_character_real','matter_energy','HEHC_energy','complete_H_variance_lower_bound','complete_H_variance_upper_bound','stationarity')} for c in r['cases']],'elapsed_seconds':r['elapsed_seconds']},indent=2))
