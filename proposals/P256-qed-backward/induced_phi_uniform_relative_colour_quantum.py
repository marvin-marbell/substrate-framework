"""NEW originalFullForm uniform45 Phi + uniform45 relative SU4 colour quantum supplier.

All65 filled CAR species and one/two-link physical endpoint transports are retained.
Frozen f63/2832 scalar sources, field laws, EM and spacing are never modified.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
CAPTURED: dict[str, bytes] = {Path(__file__).name:Path(__file__).read_bytes()}


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        raw = Path(self.path).read_bytes(); CAPTURED[Path(self.path).name] = raw
        return self.source_to_code(raw,self.path)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        spec = importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.origin is not None:
            p = Path(spec.origin)
            if p.parent==ROOT and p.suffix=='.py':
                spec.loader = Loader(fullname,str(p)); return spec
        return None


finder = Finder(); sys.meta_path.insert(0,finder)
import induced_phi_uniform_selfconsistent_full_normal_quantum as stationary
import bosonic_original_law_join as original
sys.meta_path.remove(finder)
CAPTURED.update(stationary.CAPTURED)
import mpmath as mp
import numpy as np
from scipy.linalg import expm

full = stationary.full
iv: Any = mp.iv
exact,endpoint,enclosure,bounds = full.exact,full.endpoint,full.enclosure,full.bounds
PREDECESSOR_SHA = '5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354'
PREDECESSOR_SOURCE_SHA = 'f63c1590af4fa9922757615b1bfccb788f430760d401d7c66372e8656163251c'
COLOUR_COMETRIC = F(3,10)


def original_contract() -> dict[str,str]:
    # The frozen owner declares a constant operator dictionary in an instance
    # method. Read that exact AST instead of fabricating an unused self object.
    tree = ast.parse(CAPTURED[Path(original.__file__).name])
    owner = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='OriginalBosonicLaw')
    method = next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name=='full_form_contract')
    returned = next(node for node in method.body if isinstance(node,ast.Return))
    call = returned.value
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='dict' or call.args:
        raise ArithmeticError('Original operator contract is not its declared constant dictionary')
    result: dict[str,str] = {}
    for keyword in call.keywords:
        value = ast.literal_eval(keyword.value)
        if keyword.arg is None or not isinstance(value,str):
            raise ArithmeticError('Original operator contract contains a nonliteral field')
        result[keyword.arg] = value
    return result


def colour_geometry(native: dict[str,Any], geom: dict[str,Any]) -> dict[str,Any]:
    T = geom['M']/np.sqrt(8)
    comm = 1j*(T[:,None]@native['frame'][None]-native['frame'][None]@T[:,None])
    H = np.einsum('amij,bmji->ab',comm,comm).real/4
    casimir = np.einsum('aij,ajk->ik',T,T)
    errors = dict(generator_Gram=float(np.max(abs(np.einsum('aij,bji->ab',T,T)-np.eye(15)/2))),fundamental_Casimir=float(np.max(abs(casimir-15*np.eye(4)/8))),orbit_offdiagonal=float(np.max(abs(H-np.diag(np.diag(H))))))
    if max(errors.values())>2e-14:
        raise ArithmeticError('Actual colour generator/Casimir/orbit normalization failed')
    X = np.einsum('a,aij->ij',np.arange(1,16)/80,T); U = expm(1j*X)
    def character(v: np.ndarray) -> complex:
        return np.trace(v)
    curves = []
    for step in (.02,.01):
        lap = sum(-(character(expm(1j*step*t)@U)-2*character(U)+character(expm(-1j*step*t)@U))/step**2 for t in T)
        curves.append(lap)
    richardson = (4*curves[1]-curves[0])/3
    curve_error = float(abs(richardson-15*character(U)/8))
    if curve_error>1e-7:
        raise ArithmeticError('Actual original E4 left-curve Casimir check failed')
    return dict(T=T,H=H,r=np.diag(H),errors=errors,curve_error=curve_error,
                original_full_form=original_contract(),
                coordinate='U_x,axis=exp(i sum_A T_A C_axis,A/8), T_A=M_A/sqrt8, Tr(T_A T_B)=delta_AB/2; E4_A=−i left derivative exp(i epsilon T_A)U',
                physical_cometric='(3/10) chi I45. OriginalFullForm Hamiltonian is .3chi E4²/2; old LAW05 .3chi E4² and .6chi cometric are a separate twice-coefficient branch.',
                Casimir='sum_A T_A²=15/8 I4; actual fundamental character left-curve Richardson check',
                stiffness='Kc_primitive=Lambda^(5/4) I3 tensor Horbit/5, Horbit_AB=sum_mu Tr(i[T_A,Phi_mu]i[T_B,Phi_mu])/4. Uniform q0 plaquette quadratic is exactly zero, but original .7Lambda^(5/4) plaquette quartic is retained.')


def bloch_species(p: dict[str,Any], native: dict[str,Any], geom: dict[str,Any], colour: dict[str,Any]) -> dict[str,Any]:
    theta = (2*np.pi*np.asarray(list(product(range(4),repeat=3)))+np.pi)/4
    T = colour['T']; sign = -1 if p['scale']=='heavy' else 1
    x = np.zeros((64,4)); x[:,0] = float(p['onsite'])
    for v in p['paths']:
        axis,L = v['axis'],v['length']
        x[:,0] += 2*float(v['c0'])*np.cos(L*theta[:,axis])
        x[:,axis+1] -= 2*float(v['ci'])*np.sin(L*theta[:,axis])
    vertices = np.zeros((64,90,4,4),complex)
    cross = np.zeros((64,45,45,4,4),complex)
    cc = np.zeros((64,45,45,4,4),complex)
    path_derivatives = []
    for disp,c in p['raw_paths']:
        axis = int(np.flatnonzero(disp)[0]); L = int(disp[axis])
        hop = np.einsum('m,mij->ij',c,native['frame']); phase = np.exp(1j*L*theta[:,axis])
        anti = hop@T+T@hop
        cv = 1j*L/16*anti
        vertices[:,45+15*axis:60+15*axis] += phase[:,None,None,None]*cv[None]+phase.conj()[:,None,None,None]*cv.conj().transpose(0,2,1)[None]
        dH = np.einsum('m,amij->aij',c,geom['deltas'])/8
        mix = 1j*L/16*(dH[:,None]@T[None]+T[None]@dH[:,None])
        cross[:,:,15*axis:15*(axis+1)] += phase[:,None,None,None,None]*mix[None]+phase.conj()[:,None,None,None,None]*mix.conj().transpose(0,1,3,2)[None]
        ab = T[:,None]@T[None]+T[None]@T[:,None]
        second = -L*L/256*(hop@ab+ab@hop)
        cc[:,15*axis:15*(axis+1),15*axis:15*(axis+1)] += phase[:,None,None,None,None]*second[None]+phase.conj()[:,None,None,None,None]*second.conj().transpose(0,1,3,2)[None]
        path_derivatives.append(dict(displacement=disp,axis=axis,L=L,hop=hop,phi=dH,colour=cv,cross=mix,cc=second))
    h = sign*np.einsum('km,mij->kij',x,native['frame'])
    vertices[:,:45] = np.einsum('km,amij->kaij',x,geom['deltas'])/8
    vertices *= sign; cross *= sign; cc *= sign
    energies,U = np.linalg.eigh(h); W,Z = U[:,:,:2],U[:,:,2:]
    delta = energies[:,2]-energies[:,1]
    amps = W.conj().transpose(0,2,1)[:,None]@vertices@Z[:,None]
    C = np.einsum('kaij,kbij->kab',amps,amps.conj())
    B = 2*np.einsum('kab,k->ab',C,1/delta).real
    P = W@W.conj().transpose(0,2,1)
    contact = np.zeros((90,90)); contact[:45,45:] = np.einsum('kij,kabji->ab',P,cross).real
    contact[45:,:45] = contact[:45,45:].T
    contact[45:,45:] = np.einsum('kij,kabji->ab',P,cc).real
    sea = contact-B
    force = np.einsum('kij,kaji->a',P,vertices).real
    return dict(profile=p,sign=sign,h=h,vertices=vertices,delta=delta,amps=amps,C=C,B=B,contact=contact,sea=sea,force=force,paths=path_derivatives)


def public_species(row: dict[str,Any], geom: dict[str,Any]) -> dict[str,Any]:
    p = row['profile']; sea = row['sea']; pc = sea[:45,45:]; cc = sea[45:,45:]
    dynamic = []
    for om in (0,1,10,100):
        response = np.sum(row['C']/(row['delta'][:,None,None]-1j*om)+row['C'].conj()/(row['delta'][:,None,None]+1j*om),axis=0)
        dynamic.append(dict(unscaled_Euclidean_frequency=om,PH=full.encode(response),odd_antisymmetric_norm=float(np.linalg.norm((response-response.T)/2,2))))
    return dict(role=p['role'],load=p['load'],physical_scale=p['scale'],occupancy='negative physical −Lambda h = upper h BOTH bands' if row['sign']<0 else 'negative h BOTH bands',actual_Phi_colour_static_matrix=pc.tolist(),actual_colour_static_matrix=cc.tolist(),actual_contact= row['contact'].tolist(),actual_static_PH=row['B'].tolist(),actual_complex_PH_point_matrices=full.encode(row['C']),actual_PH_amplitudes=full.encode(row['amps']),actual_PH_gaps=row['delta'].tolist(),actual_dynamic_complex_gyro=dynamic,Phi_colour_cross_norm=float(np.linalg.norm(pc,2)),Phi_colour_cross_absmax=float(np.max(abs(pc))),colour_static_eigenvalues=np.linalg.eigvalsh(cc).tolist(),colour_force_norm=float(np.linalg.norm(row['force'][45:])),Phi35_colour_cross=pc[geom['index35']].tolist())


def literal_species(row: dict[str,Any],native: dict[str,Any],geom: dict[str,Any]) -> dict[str,Any]:
    V = native['V']; sign = row['sign']; p = row['profile']; zero = np.zeros(3*V)
    theta = (2*np.pi*np.asarray(list(product(range(4),repeat=3)))+np.pi)/4
    fourier = np.kron(np.exp(1j*native['sites']@theta.T)/8,np.eye(4))
    paths = []
    for data in row['paths']:
        support = [v for v in native['Pprofiles'][0]['paths'] if np.array_equal(v['displacement'],data['displacement'])]
        if len(support)!=64:
            raise ArithmeticError('Literal one/two-link native transport support incomplete')
        paths.extend(dict(v,hop=v['seam']*data['hop'],derivatives=data) for v in support)
    profile = dict(model=p['model'],onsite=float(p['onsite'])*native['frame'][0],paths=paths)
    h = sign*full.supplier.quantum.matter_hamiltonian(native,profile,zero)
    Fs = np.zeros((90,256,256),complex)
    for a,df in enumerate(geom['deltas']):
        dp = dict(profile,onsite=float(p['onsite'])*df[0]/8,paths=[dict(v,hop=v['seam']*v['derivatives']['phi'][a]) for v in paths])
        Fs[a] = sign*full.supplier.quantum.matter_hamiltonian(native,dp,zero)
    # Colour coefficients are assembled on the actual site-major paths, not inferred from a Fourier echo.
    for path in paths:
        data = path['derivatives']; i,j = int(path['source']),int(path['target']); axis = data['axis']
        sl,tl = slice(4*i,4*i+4),slice(4*j,4*j+4)
        coeff = sign*path['seam']*data['colour']; Fs[45+15*axis:60+15*axis,sl,tl] += coeff
        Fs[45+15*axis:60+15*axis,tl,sl] += coeff.conj().transpose(0,2,1)
    energies,U = np.linalg.eigh(h); W,Z = U[:,:128],U[:,128:]
    if not energies[127]<0<energies[128]:
        raise ArithmeticError('Literal256 both physical negative bands are not filled')
    delta = energies[None,128:]-energies[:128,None]; amps = W.conj().T@Fs@Z
    B = 2*np.einsum('aij,bij,ij->ab',amps,amps.conj(),1/delta).real
    P = W@W.conj().T; contact = np.zeros((90,90))
    for path in paths:
        data = path['derivatives']; i,j = int(path['source']),int(path['target']); axis = data['axis']
        block = P[4*j:4*j+4,4*i:4*i+4]; f = sign*path['seam']
        contact[:45,45+15*axis:60+15*axis] += 2*np.einsum('ij,abji->ab',block,f*data['cross']).real
        contact[45+15*axis:60+15*axis,45+15*axis:60+15*axis] += 2*np.einsum('ij,abji->ab',block,f*data['cc']).real
    contact[45:,:45] = contact[:45,45:].T
    expectedh = np.zeros_like(h); expectedF = np.zeros_like(Fs)
    for k in range(64):
        expectedh[4*k:4*k+4,4*k:4*k+4] = row['h'][k]
        expectedF[:,4*k:4*k+4,4*k:4*k+4] = row['vertices'][k]
    errors = dict(full256_h=float(np.max(abs(fourier.conj().T@h@fourier-expectedh))),ALL90_full256_currents=float(np.max(abs(fourier.conj().T@Fs@fourier-expectedF))),ALL90_static_PH=float(np.max(abs(B-row['B']))),ALL90_contacts=float(np.max(abs(contact-row['contact']))))
    gyro = []
    for om in (0,1,10,100):
        actual = np.einsum('aij,bij,ij->ab',amps,amps.conj(),1/(delta-1j*om))+np.einsum('aij,bij,ij->ab',amps.conj(),amps,1/(delta+1j*om))
        predicted = np.sum(row['C']/(row['delta'][:,None,None]-1j*om)+row['C'].conj()/(row['delta'][:,None,None]+1j*om),axis=0)
        error = float(np.max(abs(actual-predicted))); errors['gyro_'+str(om)] = error
        gyro.append(dict(unscaled_frequency=om,absolute_error=error))
    if max(errors.values())>3e-10:
        raise ArithmeticError('Actual literal256 all90 currents/contact/complex PH join failed')
    return dict(role=p['role'],load=p['load'],literal_shape=list(Fs.shape),PH_shape=list(amps.shape),errors=errors,gyro=gyro)


def colour_zpe_radial(g0: Any, gs: Any, ctx: Any) -> tuple[Any,list[Any],list[list[Any]]]:
    # Exact Clifford anticommuting-generator multiplicities, independently
    # checked against the actual literal orbit Gram by colour_geometry.
    categories = ((0,3,1),(1,2,3),(1,1,3),(0,2,3),(1,0,1),(0,1,3),(1,3,1))
    prefactor = 3*ctx.sqrt(stationary.number(F(3,50),ctx))/2
    energy = ctx.mpf(0); gradient = [ctx.mpf(0),ctx.mpf(0)]
    Hessian = [[ctx.mpf(0),ctx.mpf(0)],[ctx.mpf(0),ctx.mpf(0)]]
    for b0,bs,multiplicity in categories:
        b = (b0,bs); r = (b0*g0+bs*gs)/2; sr = ctx.sqrt(r)
        energy += prefactor*multiplicity*sr
        for i in range(2):
            gradient[i] += prefactor*multiplicity*b[i]/(4*sr)
            for j in range(2):
                Hessian[i][j] -= prefactor*multiplicity*b[i]*b[j]/(16*r*sr)
    return energy,gradient,Hessian


def feedback_force_jacobian(g0: Any, gs: Any, rows: list[dict[str,Any]], ctx: Any) -> tuple[list[Any],list[list[Any]],Any]:
    force,J,energy = stationary.radial_force_jacobian(g0,gs,rows,ctx)
    zpe,dz,H = colour_zpe_radial(g0,gs,ctx)
    ratio = stationary.number(F(2)**-5120,ctx)
    for i,volume in enumerate((64,192)):
        force[i] += ratio*dz[i]/volume
        for j in range(2):
            J[i][j] += ratio*H[i][j]/volume
    return force,J,energy+ratio*zpe/64


def feedback_root(rows: list[dict[str,Any]], bare: dict[str,Any]) -> dict[str,Any]:
    mp.mp.dps = 2600
    seed = [stationary.number(F(v),mp.mp) for v in bare['centre_exact_rational']]
    def f(x: Any,y: Any) -> tuple[Any,Any]:
        values,_,_ = feedback_force_jacobian(x,y,rows,mp.mp)
        return values[0],values[1]
    def jac(x: Any,y: Any) -> Any:
        _,J,_ = feedback_force_jacobian(x,y,rows,mp.mp)
        return mp.matrix(J)
    root = mp.findroot(f,seed,J=jac,tol=mp.mpf(10)**-2500,maxsteps=20,solver='mdnewton')
    centres = [F(str(mp.nstr(v,2250))) for v in root]; radius = F(1,10**2150)
    iv.dps = 2300
    point = [exact(c) for c in centres]; box = [enclosure(c-radius,c+radius) for c in centres]
    fp,Jp,_ = feedback_force_jacobian(point[0],point[1],rows,iv)
    fb,Jb,energy = feedback_force_jacobian(box[0],box[1],rows,iv)
    Jr = [[stationary.midpoint(v) for v in row] for row in Jp]
    determinant = Jr[0][0]*Jr[1][1]-Jr[0][1]*Jr[1][0]
    C = [[Jr[1][1]/determinant,-Jr[0][1]/determinant],[-Jr[1][0]/determinant,Jr[0][0]/determinant]]
    contraction = F(0); image = []
    for i in range(2):
        derivative = [exact(int(i==j))-sum(exact(C[i][k])*Jb[k][j] for k in range(2)) for j in range(2)]
        lipschitz = sum(endpoint(abs(v),1) for v in derivative)
        contraction = max(contraction,lipschitz)
        image.append(endpoint(abs(sum(exact(C[i][j])*fp[j] for j in range(2))),1)+lipschitz*radius)
    if contraction>=1 or max(image)>=radius:
        raise ArithmeticError('Actual original-colour quantum feedback implicit root inclusion failed')
    force45 = iv.sqrt((16*iv.sqrt(box[0])/iv.sqrt(8)*abs(fb[0]))**2+3*(16*iv.sqrt(box[1])/iv.sqrt(8)*abs(fb[1]))**2)
    zpe,gradient,H = colour_zpe_radial(box[0],box[1],iv)
    return dict(centres=centres,box=box,parameters=dict(g0=bounds(box[0]),gs=bounds(box[1])),centre_exact_rational=[str(v) for v in centres],root_radius=bounds(exact(radius)),contraction=bounds(exact(contraction)),image_radii=[bounds(exact(v)) for v in image],Newton_preconditioner_exact_rational=[[str(v) for v in row] for row in C],radial_force_intervals=[bounds(v) for v in fb],ALL45_force_divided_Lambda_upper=bounds(force45),colour_ZPE_divided_Lambda3over8=bounds(zpe),colour_ZPE_radial_gradient_divided_Lambda3over8=[bounds(v) for v in gradient],colour_ZPE_radial_Hessian_divided_Lambda3over8=[[bounds(v) for v in row] for row in H],quantum_shift_from_bare_f63_centre=[mp.nstr(v-b,100) for v,b in zip(root,seed,strict=True)],effective_energy_per_site_divided_Lambda=bounds(energy),proof='Exact preconditioned Newton inclusion and contraction prove the positive root of Vclassical_ALL65+Ecolour_primitive. Full uniform Ecolour=(3/2)Lambda^(3/8)Tr15 sqrt((3/50)Horbit), not45 duplicated scalar Phi modes. Weighted Clifford double commutators make every shape/offdiagonal Phi gradient exactly zero; four diagonal grade1 forces remain, and actual cubic AP permutations make three spatial equations identical. Hence ALL45 implicit-root effective forces are EXACT ZERO. This selected BO-feedback preparation is NOT the full nonlinear or PH-dynamical quantum ground; subleading colour-PH and Phi-ZPE force prices are separately REQUIRED.')


def fast_phi_derivatives(native: dict[str,Any], geom: dict[str,Any], colour: dict[str,Any]) -> dict[str,Any]:
    T = colour['T']; comm = 1j*(T[:,None]@native['frame'][None]-native['frame'][None]@T[:,None])
    dframe = geom['deltas']/8
    derivative = 1j*(T[None,:,None]@dframe[:,None]-dframe[:,None]@T[None,:,None])
    first = np.einsum('amij,ubmji->uab',comm,derivative).real/4
    first += first.transpose(0,2,1).copy()
    second = np.einsum('uamij,vbmji->uvab',derivative,derivative,optimize=True).real/4
    second += second.transpose(1,0,2,3).copy()
    r = colour['r']; sr = np.sqrt(r)
    prefactor = 1.5*np.sqrt(.06)
    grad = prefactor*.5*np.einsum('uaa,a->u',first,1/sr)
    loewner = -1/(2*sr[:,None]*sr[None,:]*(sr[:,None]+sr[None,:]))
    Hessian = prefactor*(.5*np.einsum('uvaa,a->uv',second,1/sr)+np.einsum('ab,uab,vba->uv',loewner,first,first,optimize=True))
    radial = np.zeros(45); g0=float(native['g0']); gs=float(native['gs'])
    _,dz,_ = colour_zpe_radial(g0,gs,mp.fp)
    radial[0] = float(dz[0])*np.sqrt(g0)/(4*np.sqrt(8))
    radial[1:4] = float(dz[1])*np.sqrt(gs)/(12*np.sqrt(8))
    error = float(np.max(abs(grad-radial)))
    if error>2e-14:
        raise ArithmeticError('Actual full45 fast quantum gradient is not the proved radial Clifford gradient')
    return dict(first=first,second=second,gradient=grad,Hessian=Hessian,public=dict(actual_Horbit_first_Phi45_derivative=first.tolist(),actual_Efast_Phi45_gradient_divided_Lambda3over8=grad.tolist(),actual_Efast_Phi45_Hessian_divided_Lambda3over8=Hessian.tolist(),radial_exact_trace_identity_error=error,gradient_norm=float(np.linalg.norm(grad)),Hessian_norm=float(np.linalg.norm(Hessian,2)),exact_derivative_ABI='H_AB(Phi)=sum_mu Tr(i[T_A,Phi_mu]i[T_B,Phi_mu])/4; dPhi_mu/dq_i=deltaPhi_i,mu/8. dE=(3/4)sqrt(3/50)Lambda^(3/8)Tr(H^−1/2 dH); D²TrsqrtH = .5Tr(H^−1/2 D²H)−sum_AB dH_i,AB dH_j,BA/[2sqrt(r_A)sqrt(r_B)(sqrt(r_A)+sqrt(r_B))]. Exact Clifford matrices and implicit-root algebraic Q define coefficients; these arrays are observations, never wave coefficients.'))


def exact_static_species(p: dict[str,Any],geom: dict[str,Any],g: list[Any],tensors: Any) -> dict[str,Any]:
    """All90 static PH and contacts from exact integer Clifford traces."""
    M = geom['M']; gamma = full.supplier.primitive.SELECTED
    roots = [iv.sqrt(v) for v in g]; half = iv.sqrt(exact(2))/2
    co1,si1,si2 = (1,-1,-1,1),(1,1,-1,-1),(1,-1,1,-1)
    anti = M[:,None]@M[None]+M[None]@M[:,None]
    anti_g = gamma[:,None]@M[None]+M[None]@gamma[:,None]
    colour_expansion = []
    for m in range(4):
        row = []
        for A in range(15):
            row.append([(B,F.from_float(float(np.trace(anti_g[m,A]@M[B]).real/4))) for B in range(15) if np.trace(anti_g[m,A]@M[B]).real!=0])
        colour_expansion.append(row)
    pc_trace = np.empty((15,15,5),object); cc_trace = np.empty((4,15,15,5),object)
    for A,B in product(range(15),repeat=2):
        pc_trace[A,B,0] = F.from_float(float(np.trace(anti[A,B]).real/2))
        for j in range(4):
            pc_trace[A,B,j+1] = F.from_float(float(np.trace(gamma[j]@anti[A,B]).real/2))
        for m in range(4):
            second = gamma[m]@anti[A,B]+anti[A,B]@gamma[m]
            cc_trace[m,A,B,0] = F.from_float(float(np.trace(second).real/2))
            for j in range(4):
                cc_trace[m,A,B,j+1] = F.from_float(float(np.trace(gamma[j]@second).real/2))
    B = [[exact(0) for _ in range(90)] for _ in range(90)]
    contact = [[exact(0) for _ in range(90)] for _ in range(90)]
    sign = -1 if p['scale']=='heavy' else 1; occupied = -sign
    gap_lower = None
    for k in product(range(4),repeat=3):
        x = [exact(p['onsite']),exact(0),exact(0),exact(0)]
        phases = []
        for path in p['paths']:
            axis,L = path['axis'],path['length']
            co = exact(co1[k[axis]])*half if L==1 else exact(0)
            si = exact(si1[k[axis]])*half if L==1 else exact(si2[k[axis]])
            x[0] += 2*exact(path['c0'])*co; x[axis+1] -= 2*exact(path['ci'])*si
            phases.append((path,co,si))
        h = [r*y for r,y in zip(roots,x,strict=True)]
        energy = iv.sqrt(sum(v*v for v in h)); n = [v/energy for v in h]
        gl = endpoint(2*energy,0); gap_lower = gl if gap_lower is None else min(gap_lower,gl)
        D = [[exact(0) for _ in range(15)] for _ in range(15)]
        for A,Z in product(range(15),repeat=2):
            t0,linear,quad = tensors[A][Z]; re = exact(t0)
            for i,r,_ in linear:
                re += occupied*exact(r)*n[i]
            for i,j,r,_ in quad:
                re += exact(r)*n[i]*n[j]
            D[A][Z] = re
        vectors: list[dict[int,Any]] = []
        for column in geom['columns']:
            vector: dict[int,Any] = {}
            for slot,c in column.items():
                m,A = divmod(slot,15); vector[A] = vector.get(A,exact(0))+c*x[m]/8
            vectors.append(vector)
        vectors.extend({} for _ in range(45))
        def tr(coefficients: Any) -> Any:
            return exact(coefficients[0])+occupied*sum(exact(coefficients[j+1])*n[j] for j in range(4))
        for path,co,si in phases:
            axis,L = path['axis'],path['length']
            # Im(e^ikL c_m), Re(e^ikL c_m) for the actual exact c0+i ci paths.
            imag = ((0,exact(path['c0'])*si),(axis+1,exact(path['ci'])*co))
            real = ((0,exact(path['c0'])*co),(axis+1,-exact(path['ci'])*si))
            for A in range(15):
                v = vectors[45+15*axis+A]
                for m,im in imag:
                    for Z,coefficient in colour_expansion[m][A]:
                        v[Z] = v.get(Z,exact(0))-L*im*roots[m]*exact(coefficient)/8
            for i,column in enumerate(geom['columns']):
                for slot,c in column.items():
                    m,A = divmod(slot,15)
                    for m2,im in imag:
                        if m==m2:
                            for Z in range(15):
                                contact[i][45+15*axis+Z] -= sign*L*im*c*tr(pc_trace[A,Z])/512
            for A,Z in product(range(15),repeat=2):
                contact[45+15*axis+A][45+15*axis+Z] -= sign*L*L*sum(re*roots[m]*tr(cc_trace[m,A,Z]) for m,re in real)/1024
        for a in range(90):
            for b in range(a,90):
                value = sum(c*d*D[A][Z] for A,c in vectors[a].items() for Z,d in vectors[b].items())/energy
                B[a][b] += value
                if a!=b:
                    B[b][a] += value
    for i,j in product(range(45),repeat=2):
        contact[45+j][i] = contact[i][45+j]
    sea = [[contact[i][j]-B[i][j] for j in range(90)] for i in range(90)]
    return dict(B=B,contact=contact,sea=sea,gap_lower=exact(gap_lower))


def matrix_norm_upper(matrix: list[list[Any]]) -> Any:
    return iv.sqrt(sum(abs(v)**2 for row in matrix for v in row))


def joint_quantum(geom: dict[str,Any],g: list[Any],profiles: list[dict[str,Any]],algebra: list[dict[str,Any]],fast: dict[str,Any],root: dict[str,Any]) -> dict[str,Any]:
    Lambda = exact(F(2)**8192); chi = exact(F(2)**-4096)
    rho = exact(F(2)**-2048); cross_ratio = exact(F(2)**-1024)
    Kprim,_ = full.primitive_matrix(geom,g,exact(F(1,64)))
    matter = [[algebra[0]['sea'][i][j]+(algebra[1]['sea'][i][j]+63*algebra[2]['sea'][i][j])/Lambda for j in range(90)] for i in range(90)]
    Kphi = [[Kprim[i][j]+matter[i][j] for j in range(45)] for i in range(45)]
    eig,U,error,orth = full.eigen_certificate(Kphi)
    phi_min = exact(F.from_float(float(eig[0])))-error
    pc = [row[45:] for row in matter[:45]]; cc = [row[45:] for row in matter[45:]]
    pc_norm = matrix_norm_upper(pc); cc_norm = matrix_norm_upper(cc)
    # Horbit_min=.5 gs, primitive Kc_min=gs/10, exact at the implicit root.
    colour_min = g[1]/10-rho*cc_norm
    eta = cross_ratio*pc_norm/iv.sqrt(phi_min*colour_min)
    if endpoint(phi_min,0)<=0 or endpoint(colour_min,0)<=0 or endpoint(eta,1)>=1:
        raise ArithmeticError('Actual rescaled full90 static quantum stiffness is not positive')
    Gp = 8*chi; Gc = exact(COLOUR_COMETRIC)*chi
    covphi = iv.sqrt(Gp/Lambda)/(2*iv.sqrt(phi_min)*iv.sqrt(1-eta))
    covcolour = iv.sqrt(Gc/exact(F(2)**10240))/(2*iv.sqrt(colour_min)*iv.sqrt(1-eta))
    conditional_M = rho*pc_norm/((1-eta)*colour_min)
    # Exact Riccati C block A_CC²=(K_CC−Gp A_CP A_PC)/Gc and
    # Sylvester A_PP Gp A_PC+A_PC Gc A_CC=K_PC improve colour bounds.
    ACCrelative = (rho*cc_norm/(g[1]/10)+(Gp/Gc)*rho*rho*pc_norm**2/((1-eta)*colour_min*(g[1]/10)))/(1-eta)
    phi_coordinate_norm = iv.sqrt(45*covphi)
    boxforce = exact(F(root['ALL45_force_divided_Lambda_upper']['upper']))*Lambda
    # Universal operator bounds price dPH/contact without inserting fitted force.
    derivative_bounds = []; phi_derivative_bounds = []
    for profile,alg in zip(profiles,algebra,strict=True):
        onsite = abs(exact(profile['onsite']))
        csum = sum(abs(exact(v['c0']))+abs(exact(v['ci'])) for v in profile['paths'])
        bphi = (onsite+2*csum)/(8*iv.sqrt(2))
        bcolour = sum(v['length']*(abs(exact(v['c0']))*iv.sqrt(g[0])+abs(exact(v['ci']))*iv.sqrt(g[1])) for v in profile['paths'])/(4*iv.sqrt(8))
        bprime = sum(v['length']*(abs(exact(v['c0']))+abs(exact(v['ci']))) for v in profile['paths'])/128
        contact_cap = sum(v['length']**2*(abs(exact(v['c0']))*iv.sqrt(g[0])+abs(exact(v['ci']))*iv.sqrt(g[1])) for v in profile['paths'])/256
        contact_prime = sum(v['length']**2*(abs(exact(v['c0']))+abs(exact(v['ci']))) for v in profile['paths'])/(2048*iv.sqrt(2))
        gap = alg['gap_lower']
        per_entry = 512*bphi*contact_cap/gap+128*contact_prime+128*(20*bphi*bcolour*bcolour/gap**2+8*bcolour*bprime/gap)
        derivative_bounds.append(iv.sqrt(45)*45*per_entry)
        phi_derivative_bounds.append(iv.sqrt(45)*45*128*20*bphi**3/gap**2)
    dcc = derivative_bounds[0]+(derivative_bounds[1]+63*derivative_bounds[2])/Lambda
    # ||D O||F <=sqrt(15)/32 and ||O||<=sqrt(sum g/2) give an
    # explicit all45 normal derivative bound, with no fitted coefficients.
    dKprim_cap = 2*iv.sqrt(sum(g)/2)*iv.sqrt(45*15)/(32*5)
    colour_PH_force = iv.sqrt(Gc*exact(F(2)**10240))*45*rho*(dcc/iv.sqrt(colour_min)+dKprim_cap*cc_norm/(2*iv.sqrt(colour_min)**3))/4
    # Slow Phi ZPE is explicitly not included in this BO mean. Its source
    # force action is priced, never called an exact nonlinear quantum root.
    # Vprimitive/Lambda=256||Phi Phi^T||F² and ||dPhi/dq||F=1/sqrt128.
    dKphi_cap = iv.sqrt(45)*6144*2*iv.sqrt(sum(g))/iv.sqrt(128)**3+phi_derivative_bounds[0]+(phi_derivative_bounds[1]+63*phi_derivative_bounds[2])/Lambda
    phi_ZPE_force = iv.sqrt(Gp*Lambda)*45*dKphi_cap/(4*iv.sqrt(phi_min))
    omega = [iv.sqrt(exact(F(3,50))*sum(g[m] for m in range(4) if np.max(abs(geom['M'][A]@full.supplier.primitive.SELECTED[m]+full.supplier.primitive.SELECTED[m]@geom['M'][A]))==0)/2)*exact(F(2)**3072) for A in range(15)]
    # Certified reference matrix display. Exact joint covariance remains the
    # non-block-diagonal matrix functional below; Loewner disks enclose it.
    refphi = (U*(1/np.sqrt(eig))[None])@U.T/2
    return dict(actual_Phi_stiffness_divided_Lambda=[[full.centre(v) for v in row] for row in Kphi],actual_Phi_colour_stiffness_divided_Lambda=[[full.centre(v) for v in row] for row in pc],actual_colour_sea_stiffness_divided_Lambda=[[full.centre(v) for v in row] for row in cc],certified_Phi_eigenvalue_lower_divided_Lambda=bounds(phi_min),certified_colour_eigenvalue_lower_divided_Lambda5over4=bounds(colour_min),certified_Phi_colour_cross_norm_divided_Lambda=bounds(pc_norm),certified_colour_sea_norm_divided_Lambda=bounds(cc_norm),reference_relative_stiffness_offdiagonal_norm_upper=bounds(eta),reference_Phi_covariance_units_sqrt_Gphi_over_Lambda=refphi.tolist(),actual_joint_Phi_covariance_operator_upper=bounds(covphi),actual_joint_conditional_colour_covariance_operator_upper=bounds(covcolour),actual_joint_conditional_mean_M_operator_upper=bounds(conditional_M),actual_joint_ACC_relative_primitive_alpha0_operator_upper=bounds(ACCrelative),actual_primitive_colour_frequencies=[bounds(v) for v in omega]*3,actual_root_box_force_times_Phi_coordinate_norm_upper=bounds(boxforce*phi_coordinate_norm),actual_omitted_static_colour_PH_ZPE_force_times_Phi_coordinate_norm_upper=bounds(colour_PH_force*phi_coordinate_norm),actual_omitted_Phi_ZPE_force_times_Phi_coordinate_norm_upper=bounds(phi_ZPE_force*phi_coordinate_norm),exact_joint_wave_ABI=dict(coordinate='q=(Phi45,C_axisMajor45); native edges remain site-major3*site+axis. Phi_x=Phi*+sum_i q_i deltaPhi_i/8; U_x,axis=exp(i sum_A C_axis,A T_A/8).',G='diag(8chi I45,(3/10)chi I45)',K90='[[Lambda Kprim+sea_PhiPhi, sea_PhiC],[sea_CP, Lambda^(5/4) I3 tensor Horbit/5+sea_CC]], sea=sum_all65 load*(physical contact−exact static PH bubble). Primitive plaquette has exactly zero q0 Hessian.',A='G^−1/2 (G^1/2 K90 G^1/2)^1/2 G^−1/2 EXACT positive matrix functional at implicit NEW feedback root; solves A G A=K90. No NumPy coefficients, scalar duplication, or finite-precision huge clock phases define this wave.',wave='Psi=(det A/pi^90)^1/4 exp(−q^T A q/2). Cov=.5A^−1. Conditional mean C=−M Phi, M=A_CC^−1 A_CP; conditional C covariance=.5A_CC^−1; Phi marginal covariance=.5(A_PP−A_PC A_CC^−1 A_CP)^−1.', quadratic_PH_reference_law='All actual complex PH amplitudes, both occupied/empty bands and load63 independent CAR channels are retained. The SELECTED quadratic reference is Href=Pi G Pi/2+q Kbare q/2+sum Delta bdag b+sum q_a(A_aj bdag_j+A*_aj b_j), Kbare=primitive+filled-sea contact. Its dynamic kernel contains BOTH C/(Delta−i omega)+C*/(Delta+i omega), including complex antisymmetric gyro. This zero-linear-force REFERENCE is not the full physical nonlinear Hamiltonian and its wave is not asserted to equal the PH-dynamical Gaussian ground.', physical_bare_force_phi='At the NEW BO root, the actual classical all65 force is NONZERO: fclass_Phi=−Lambda^(3/8) dZcolour/dq, exactly. It cancels the primitive colour-ground expectation of the actual alignment Phi C² term, NOT an omitted force. Colour force is exactly zero by filled-projector Clifford trace and actual AP sums. The source consumer MUST retain fclass·Phi plus full original nonlinear alignment or its explicit normal-ordered Phi C²/resolvent treatment.', field_following_fast_colour='At any nearby Phi, Horbit_AB(Phi)=sum_mu Tr(i[T_A,Phi_mu]i[T_B,Phi_mu])/4. Kc(Phi)=Lambda^(5/4) I3 tensor Horbit(Phi)/5+all65 static contact−PH CC block. Primitive fast conditional precision alpha0(Phi)=sqrt(Kc_primitive(Phi)/Gc), covariance=.5alpha0^−1 and centre C0(Phi)=0 for primitive alignment. Matter mixing requires the EXACTJOINT90 M=A_CC^−1 A_CP, not the naive primitive product. Efast_primitive=(3/2)Lambda^(3/8)Tr15 sqrt((3/50)Horbit(Phi)); all45 derivatives are supplied. Static-PH Efast correction force and slow Phi ZPE force have explicit independent prices; the latter is adverse, not suppressed.'),proof='Congruence scales Phi by Lambda^−1/2 and C by Lambda^−5/8. Full90 K has positive diagonal blocks and norm cross eta<1; (1−eta)Kref<=K90<=(1+eta)Kref. Operator monotonicity of square root/inverse square root certifies all90 Gaussian covariance without numerically discarding e−617 frequency hierarchy. Riccati and positive Sylvester bounds certify actual conditional M and ACC−alpha0. Actual nonzero mixed block is NEVER dropped in the exact wave.',physical_limits='Uniform relative-colour/Phi90 selected static-PH Gaussian at actual primitive-colour-ZPE feedback mean; not full nonlinear/PH-dynamical/nonuniform/fullEM vacuum or original SOURCE/commoncone/QED admission. Root omitted-Phi-ZPE force price is adverse, not suppressed.')


def run(path: Path, mixing_only: bool, root_output: Path | None) -> dict[str,Any]:
    started = time.monotonic(); raw = path.read_bytes(); cap = json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=PREDECESSOR_SHA or hashlib.sha256(CAPTURED[Path(stationary.__file__).name]).hexdigest()!=PREDECESSOR_SOURCE_SHA:
        raise ArithmeticError('Frozen f63/5b97 selfconsistent predecessor required')
    root = cap['actual_NEW_finiteN4_stationary_mean']
    older_path = Path(cap['input_descriptor']['input_receipt']['path'])
    if not older_path.is_absolute():
        older_path = ROOT/older_path
    older_raw = older_path.read_bytes()
    if hashlib.sha256(older_raw).hexdigest()!=stationary.FROZEN_RECEIPT_SHA:
        raise ArithmeticError('Original native mean descriptor predecessor changed')
    old = json.loads(older_raw)['input_descriptor']['actual_root_mean_input_UNCHANGED']
    profiles = full.scalar.fast.primitive_profiles(old)
    if not mixing_only:
        if root_output is None:
            raise ValueError('Actual quantum feedback execution requires --root-output')
        root = feedback_root(stationary.radial_data(profiles),root)
        root_public = {k:v for k,v in root.items() if k not in ('centres','box')}
        root_output.write_text(json.dumps(dict(actual_NEW_colour_BO_feedback_mean=root_public,predecessor_sha256=PREDECESSOR_SHA,source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in CAPTURED.items()}),indent=2)+'\n')
    centres = [F(v) for v in root['centre_exact_rational']]; mean = stationary.input_mean(old,centres)
    native = full.supplier.quantum.build_native(mean,4); iv.dps = 80
    g0,gs = (exact(v) for v in centres)
    if mixing_only:
        g = [g0,gs,gs,gs]
    else:
        g = [enclosure(F(root['parameters']['g0']['lower']),F(root['parameters']['g0']['upper'])),*[enclosure(F(root['parameters']['gs']['lower']),F(root['parameters']['gs']['upper'])) for _ in range(3)]]
    geom = full.normal_geometry(native,g); colour = colour_geometry(native,geom)
    species = [bloch_species(p,native,geom,colour) for p in profiles]
    public = [public_species(row,geom) for row in species]
    literal = [] if mixing_only else [literal_species(row,native,geom) for row in species]
    fast = fast_phi_derivatives(native,geom,colour)
    quantum = None; exact_errors = []
    if not mixing_only:
        tensors = full.trace_tensors(geom['M'],full.supplier.primitive.SELECTED)
        algebra = [exact_static_species(p,geom,g,tensors) for p in profiles]
        for observed,alg in zip(species,algebra,strict=True):
            errors = {key:float(np.max(abs(observed[key]-np.asarray([[full.centre(v) for v in row] for row in alg[key]])))) for key in ('B','contact','sea')}
            if max(errors.values())>3e-10:
                raise ArithmeticError('Exact Clifford all90 static PH/contact differs from actual Bloch derivative')
            exact_errors.append(dict(role=observed['profile']['role'],errors=errors))
        quantum = joint_quantum(geom,g,profiles,algebra,fast,root)
    desc = dict(predecessor=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),original_native_descriptor=dict(path=str(older_path),sha256=hashlib.sha256(older_raw).hexdigest()),plan=cap['input_descriptor']['plan'],scientific_execution_revision=full.REVISION,parameters=dict(N=4,V=64,a='1/64',UEphys=1,u=64,Lambda='2^8192',chi='2^-4096',GPhi='8chi',Gcolour='3chi/10',f='1/8',charged_loads=[1,63],neutral_heavy_load=1),physical_colour_law=colour['original_full_form'],mean_scope='f63 classical root mixing discriminator ONLY' if mixing_only else 'NEW implicit uniform Vclassical_ALL65+primitive_colour_ZPE BO feedback root; omitted subleading ZPE prices explicit',exact_feedback_mean={k:v for k,v in root.items() if k not in ('centres','box')},species_coefficients=[dict(role=p['role'],load=p['load'],scale=p['scale'],onsite=str(p['onsite']),paths=[dict(axis=v['axis'],length=v['length'],c0=str(v['c0']),ci=str(v['ci'])) for v in p['paths']]) for p in profiles],original_SOURCE_receiver_probability_UNCHANGED=True,no_bare_EMcurl_or_speed_fit=True)
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_NEW_colour_BO_feedback_mean={k:v for k,v in root.items() if k not in ('centres','box')},actual_colour_normalization={k:v for k,v in colour.items() if k not in ('T','H','r')},actual_colour_Horbit=colour['H'].tolist(),actual_Efast_full45_derivatives=fast['public'],actual_species=public,actual_literal_ALL90=literal,actual_exact_ALL90_static_Clifford_join=exact_errors,actual_joint90_quantum=quantum,Phi_labels=geom['labels'],source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in CAPTURED.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in CAPTURED.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root-output',type=Path)
    parser.add_argument('--mixing-only',action='store_true')
    args = parser.parse_args(); result = run(args.predecessor,args.mixing_only,args.root_output)
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Scientific source changed during uniform colour execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=result['input_sha256'],species=[dict(role=v['role'],Phi_colour_cross_norm=v['Phi_colour_cross_norm'],colour_force_norm=v['colour_force_norm']) for v in result['actual_species']],elapsed_seconds=result['elapsed_seconds']),indent=2))
