#!/usr/bin/env python3
"""NEW real-cover reflected C law: LEFT SF, field heat, one joint Gauss.

This is NOT frozen Alpha56's ordered transfer. The chart column is
F_ch=J_ch chi SF56 R_ch G and the outside column is
F_out=sqrt(1-chi^2) exp(-H_out/2) G; F=[F_ch,F_out], K=FFdag.
The operator definition below retains the complete Cartesian/SU4 field and
UNCOMPACTIFIED real electromagnetic quotient. Photon heat is REFERENCE RPA,
not an induced-Maxwell claim. Harmonics have real free heat, never a Gibbs trace.
A normalized packet boundary W=|Xi><Xi| supplies all powers/source consumers.
The actual regulator keeps all five real quotient modes, complete70/56 carrier
columns and all four native statistical centre-cycle branches. Its one normal
and one Lie direction do not stand in for all45/45 quantum field coordinates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

if __name__ == '__main__':
    for _variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[_variable] = '1'

import mpmath as mp
import numpy as np
from scipy.linalg import eigh, expm

ROOT = Path(__file__).resolve().parent
SOURCE_BYTES = Path(__file__).read_bytes()
LEDGER_HASH = '567985093c8e3e512d78b064427996aa92a5a7abab36886d7cf21b4e1d623c87'
FROZEN_NATIVE_HASH = '569e932d9cee150395fa4a27dd9a4355dace8b7de367db9c271b6c7bcca84d51'
LEDGER_CAPSULE_BYTES = (ROOT/'physical_c_completed_ledger_same_halfstep_capsule.json').read_bytes()
LEDGER_BYTES = json.loads(LEDGER_CAPSULE_BYTES)['execution']['source_bytes_at_IMPORT'].encode()
FROZEN_NATIVE_BYTES = (ROOT/'physical_alpha_compton_Cartesian_final_native_capsule.json').read_bytes()
if hashlib.sha256(LEDGER_BYTES).hexdigest()!=LEDGER_HASH or hashlib.sha256(FROZEN_NATIVE_BYTES).hexdigest()!=FROZEN_NATIVE_HASH:
    raise RuntimeError('The NEW law requires the exact captured ledger and Alpha56 capsule')
_directory = tempfile.TemporaryDirectory(prefix='physical-C-real-cover-')
_snapshot = Path(_directory.name)
(_snapshot/'physical_c_completed_ledger.py').write_bytes(LEDGER_BYTES)
(_snapshot/'physical_alpha_compton_Cartesian_final_native_capsule.json').write_bytes(FROZEN_NATIVE_BYTES)
_spec = importlib.util.spec_from_file_location('_physical_C_real_cover_ledger',_snapshot/'physical_c_completed_ledger.py')
if _spec is None or _spec.loader is None:
    raise RuntimeError('Cannot load the captured C-law primitive')
ledger = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ledger
_spec.loader.exec_module(ledger)
alpha = ledger.alpha


@dataclass(frozen=True)
class NativeParameters:
    photon_grade: int = 1
    quadrature_order: int = 2
    normal_order: int = 3
    colour_order: int = 2
    normal_sigma: float = .01
    colour_sigma: float = .003
    chart_radius: float = .02
    centre_penalty: float = 10.
    harmonic_free_coefficient: float = .02
    outside_quartic: float = .02
    outside_couplings: tuple[float,float,float] = (.07,.02,.03)
    prefix_steps: int = 4
    critical_bare_mass: float = -2.0282053185612114
    spacing: float = .01


def real_graph() -> dict[str,np.ndarray]:
    """2x1x1 periodic cubic graph: SIX links, FIVE real quotient coordinates."""
    d = np.zeros((2,6))
    d[:,0],d[:,3] = [1.,-1.],[-1.,1.]
    q = np.zeros((6,5))
    q[[1,4],0] = [1.,-1.]
    q[[2,5],1] = [1.,-1.]
    q[[0,3],2] = 1.
    q[[1,4],3] = 1.
    q[[2,5],4] = 1.
    q /= math.sqrt(2)
    laplace = d@d.T
    inverse = np.linalg.pinv(laplace)
    if np.linalg.norm(d@q)>1e-12 or np.linalg.norm(q.T@q-np.eye(5))>1e-12:
        raise ArithmeticError('Actual real quotient incidence/orthonormality failed')
    return dict(D=d,Q=q,L=laplace,Lplus=inverse)


def slice_geometry(reference: np.ndarray) -> dict[str,Any]:
    """Actual Cart Tr(TaTb)=4delta, SU4 t=Cart/sqrt8, regular45 normal."""
    cart = alpha.matrix_unit_basis()
    generators = cart/math.sqrt(8)
    orbit = np.array([[np.trace(1j*(t@reference[mu]-reference[mu]@t)@a).real/4
                       for mu in range(4) for a in cart] for t in generators]).T
    frame,singular,_ = np.linalg.svd(orbit,full_matrices=True)
    if np.sum(singular>1e-10)!=15:
        raise ArithmeticError('The actual Alpha Clifford field has no regular15 orbit')
    normal = frame[:,15:]
    return dict(cart=cart,generators=generators,orbit=orbit,normal=normal,
                R0=orbit.T@orbit,normal_defect=float(np.linalg.norm(orbit.T@normal)),
                normal_dimension=normal.shape[1],diagonal_connected_cover_degree=4)


def haar_jacobian(x: np.ndarray) -> float:
    eigenvalues = eigh(x,eigvals_only=True)
    product = 1.
    for a in range(4):
        for b in range(a):
            z = (eigenvalues[a]-eigenvalues[b])/2
            product *= np.sinc(z/np.pi)**2
    return float(product)


def full_family_definition() -> dict[str,Any]:
    """Exact operator-level NEW law; no claim of executing huge NF matrices."""
    with mp.workdps(180):
        naux,volume = mp.mpf(10)**1150/16,mp.mpf(2)**72
        radius = mp.mpf('1e-1000')
        norm_logs = []
        for count,k in ((mp.mpf(1),mp.mpf('.1256305951044483')),(naux,mp.mpf('1e-250'))):
            bmin = 1/(2*k)-3-k/4
            bmax = 1/(2*k)+3+k*(2*radius+4*radius**2)
            fmax = 3+8*k*radius
            m = max(bmax,1/bmin)*(1+fmax)**2
            norm_logs.append(count*volume*(mp.log(bmax)+mp.log(m)))
        energy = mp.log(1/(2*mp.mpf('1e-250'))-3)
        coupler = mp.mpf('4.4')*(naux+1)
        young = 3*coupler**(mp.mpf(4)/3)/(mp.power(4,mp.mpf(4)/3)*mp.root(mp.mpf('1e4000')/2,3))
        outside_lower = volume*(2*energy*naux-young)
        if outside_lower<=0:
            raise ArithmeticError('Fresh C-completed outside form lower bound failed')
        return dict(law='NEW_LEFT_SF_REAL_C_REFLECTED_CHART_OUTSIDE',
            field_domain='R^(60V) Cartesian traceless Hermitian Phi x SU4^(3V) x R^(2V+1) real EM quotient, finite primitive4-component CAR/species plus protected Ref4/Ref3 source sectors. Real q is never identified modulo2pi.',
            gauge='ONE joint restriction: normalized compact localSU4 Haar projection and sumrho=0 on the EXACT real-Gauss quotient. DPi=rho is solved by Pilong=Ddag Lplus rho, theta=Qq+Ddag lambda and equivariant phase exp(i lambda.rho). There is NO finite Haar average over noncompact siteR.',
            regular_chart='Phi_x=V_x(C+N0Y_x)V_xdag; Uxy=Vx zxy exp(i xi.t) Vy†. Full linked cover has connected DIAGONAL Z4 degree4, not4^V. ALL divergence-compatible centre characters and cycle labels are retained. Total fundamental particlecount including Ref is0 mod4; active+1 and Ref4->3 preserve it.',
            chart_measure='Actual weight Vol(SU4)^(V-L) product(det[O(C)†O(C+N0Y)]/sqrt(det R0)) product HaarJac(xi)/4, plus normalized Haar Gauss. Cart basisTr(TaTb)=4delta; t=Cart/sqrt8. Physical bundle QFock(V) implements the same component frame in kinetic, potential and source.',
            chart_kinetic='ALPHA-OWN law: unitary measure pullback of Dirichlet Mehler normal45/site and colour15/link generators; normal tube radius1e-1000 and colour Lie radius8sigma, sigma=1e-1500 BEFORE Naux, excitationgap1 in TRANSFER units. Physical Dminus=Q d Qdag, with the SAME Jacobian/centre descent. Every compatible centre sector retains own flux/cycle coefficient10.',
            EM_heat='REFERENCE56/RPA coexact Mehler heat with beta1e150 and lattice frequencies; REAL free harmonics with coefficientnu=.02. Exact real Gauss adds positive TRANSFER Coulomb nu rho.Lplus.rho/(2beta), primitive active chargeQ and aux charge1. This declared REFERENCE input is NOT induced Maxwell.',
            exact_order='F_s=[Jch chi SF56_s Rch_s G_s, sqrt(1-chi²) exp(-Hout_TRANSFER_s/2) G_s]; physicalHout=Hout_TRANSFER/a. ChartSF is LEFT of unchanged field heat. SF56=exp((pref56+dGamma(hC56))/2), including normal-order scalar. Chi is a smooth C/gauge-invariant tube cutoff; repeatedK connects its overlap shell. No source/Pe filter.',
            outside_H='ALPHA-OWN Friedrichs TRANSFER form: -sigma² CartesianLaplace +sigma² positiveSU4Laplace +REFERENCE realEM heat/Coulomb +1e4000 sum_site||Phi||Cart^4 + actual linearPhi C-completed normal-ordered ALL-Fock Wilson hops with(.7,.2,.3), PLUS RETAINED2Eaux/site/species. No occupation0 deletion, no negative-mass subtraction.',
            chart_SF_norm_log_upper=mp.nstr(sum(norm_logs),80),
            outside_global_H_lower=mp.nstr(outside_lower,80),
            bounded_positive_transfer='Chart multiplication SF is bounded on compact regular field chart, uniformly in REAL q since every Wilson link is unitary. OutsideH has the displayed positive global lower bound, so its semigroup is bounded. Orthogonal jointGauss and fieldheat are contractions. Thus the actual orderedF is bounded and K=FFdag is bounded positive; NO canonical photon trace is invoked.',
            neutral_normalizer='lambda_N=||K_N||, ONE neutral norm used in K_C/lambda_N and every source/current/receiver. Essential-supremum norm does NOT assert an isolated eigenvalue on real harmonic cover.',
            boundary='Actual normalized smooth compact chart/colour section times Schwartz Gaussian REAL coexact/harmonic packet and fixed-canonical filledWilson CAR/Ref4 state, projected by ONE jointGauss and normalized by its ACTUAL norm. W_N=|Xi_N><Xi_N| is traceclass. PreparedXi=(K_N/lambda_N)^m Xi_N / its ACTUAL vectornorm. Powers act on W/Xi, not on infinite globalTrK^p.',
            optional_NEW_neutral_preparation='Use SAME conditioned REFERENCE boundary f(-log(KN0/lambdaN0))Xi for actual/reference kernels. f(E)=exp(-E²/(Delta²-E²)) on0<=E<Delta, zero above, Delta=1/Nt; smooth at0 on the physical nonnegative spectrum and flat at upperedge. Record actual preparationGram/cost, retain original Gaussian control. Reference tau>=exp(-mDelta); no isolated real-harmonic ground or macro preparationGram is inferred from finite Galerkin.',
            photon_source_receiver='Full continuum incoming a_fdag is UNBOUNDED and needs its actual weighted q/p graph moments and actual normalization. Native Pgrade1 a_fdag Pgrade1 is a DECLARED projection, not fullCCR. Physical packet receiver is rank1 effect built from r=a_outdag S Xi_target, normalized by ACTUAL sum||r||²; no globally bounded continuum annihilator or source gate is asserted.',
            source='Frozen56 four-spin packet Psi=[exp(iQ theta00)U00;I]/2, Ref4->3: source and receiver retain their exact primitive/worldline path and allspin channels. Outside source is the same globally smooth bounded Wilsonline CAR operator, independent of Phi; graphcore follows from smooth Phi bump, compactSU4 packet and REAL Schwartz EM packet with finite literalQ derivatives.',
            current='Differentiate EXACT F chart/outside, including scalarpref, normal-order and actual Duhamel contacts: Kx=FxFdag+FFxdag; Kxy=FxyFdag+FxFy†+FyFx†+FFxy†. Physical spatial H=-log(K/lambda_N)/a uses Frechet log derivatives of THIS K, not h56. NEW temporal Pe kernel is K(A0,A)=exp(i rho.A0)K(A); its endpoint Ward/current/contact are kernel generating insertions, not a claim that H(A0) is Hermitian.',
            relative_factor='For SAME exact quantum colour U, heat/cutoff/Gauss and outside column, set ONLY normalY=0 in the reference chart. F1chart=M F0chart, M=SF1 SF0^-1. FullF1 is NOT M FullF0: DeltaK=D Kchart0+Kchart0 D†+D Kchart0 D†. Weighted SAME-boundary powers/source and regulator prices are required.',
            C_scope='Intrinsic C-completed potential is covariant under actual Cart/colour/EM reflection. Protected Ref4->3 maps to the explicit dual Ref0->1 hole/conjugate branch, not a false Ref3 neutral C singlet. Native code measures the PLUS branch only.',
            no_global_photon_trace=True,no_induced_Maxwell_claim=True,no_mass1_claim=True,
            no_Root6911_or_Lambda8192_receipt=True)


def hermite_rule(order: int) -> tuple[np.ndarray,np.ndarray]:
    nodes,weights = np.polynomial.hermite.hermgauss(order)
    return nodes,np.sqrt(weights/math.sqrt(np.pi))


def oscillator_h(order: int) -> tuple[np.ndarray,np.ndarray]:
    a = np.diag(np.sqrt(np.arange(1,order)),1)
    p2 = (2*np.diag(np.arange(order)) +np.eye(order)-a@a-a.T@a.T)/2
    return np.diag(np.arange(order)),p2


def field_regulator(p: NativeParameters) -> dict[str,Any]:
    if p.photon_grade not in (0,1) or p.quadrature_order<2:
        raise ValueError('Implemented complete photon vacuum/all5single-excitation regulators use grade0 or1')
    x,w = hermite_rule(p.quadrature_order)
    photon_nodes = np.array(list(itertools.product(x,repeat=5)))
    photon_weights = np.array([np.prod(v) for v in itertools.product(w,repeat=5)])
    photon_basis = photon_weights[:,None]*np.column_stack([np.ones(len(photon_nodes))]+([math.sqrt(2)*photon_nodes[:,i] for i in range(5)] if p.photon_grade else []))
    yn,yw = hermite_rule(p.normal_order);xn,xw = hermite_rule(p.colour_order)
    ybasis = np.column_stack([yw*np.polynomial.hermite.hermval(yn,[0.]*i+[1.])/math.sqrt(2**i*math.factorial(i)) for i in range(p.normal_order)])
    xbasis = np.column_stack([xw*np.polynomial.hermite.hermval(xn,[0.]*i+[1.])/math.sqrt(2**i*math.factorial(i)) for i in range(p.colour_order)])
    basis = np.kron(np.kron(photon_basis,ybasis),xbasis)
    omega = 2*math.asinh(1.)
    covers = photon_nodes.copy();covers[:,:2] /= math.sqrt(math.sinh(omega))
    points = [(a,math.sqrt(2)*p.normal_sigma*y,math.sqrt(2)*p.colour_sigma*c)
              for a,y,c in itertools.product(covers,yn,xn)]
    phdim = photon_basis.shape[1]
    ph_h = np.zeros((phdim,phdim))
    ph_h[0,0] = 3*p.harmonic_free_coefficient/4
    if p.photon_grade:
        ph_h[1,1]=ph_h[2,2]=omega+ph_h[0,0]
        for j in range(3,6):ph_h[j,j]=ph_h[0,0]+p.harmonic_free_coefficient/2
    # EXACT compressed real harmonic halfheat, not exp(-P Pi² P/2).
    a = 1+p.harmonic_free_coefficient/4
    b = 1+p.harmonic_free_coefficient/2
    ph_r = np.full(phdim,a**(-1.5))
    ph_r2_full = np.full(phdim,b**(-1.5))
    if p.photon_grade:
        ph_r[1:3] *= math.exp(-omega/2)
        ph_r2_full[1:3] *= math.exp(-omega)
        ph_r[3:] /= a
        ph_r2_full[3:] /= b
    heat_leak = float(math.sqrt(max(ph_r2_full-ph_r**2)))
    yh,yp2 = oscillator_h(p.normal_order);xh,xp2 = oscillator_h(p.colour_order)
    h = np.kron(np.kron(ph_h,np.eye(p.normal_order)),np.eye(p.colour_order))
    h += np.kron(np.kron(np.eye(phdim),yh),np.eye(p.colour_order))
    h += np.kron(np.kron(np.eye(phdim),np.eye(p.normal_order)),xh)
    outside_h = np.kron(np.kron(ph_h,np.eye(p.normal_order)),np.eye(p.colour_order))
    outside_h += np.kron(np.kron(np.eye(phdim),yp2/2),np.eye(p.colour_order))
    outside_h += np.kron(np.kron(np.eye(phdim),np.eye(p.normal_order)),xp2/2)
    if np.linalg.norm(basis.T@basis-np.eye(basis.shape[1]))>1e-11:
        raise ArithmeticError('Actual complete Hermite coordinate quadrature is not orthonormal')
    exact_r = np.kron(np.kron(np.diag(ph_r),expm(-yh/2)),expm(-xh/2))
    return dict(basis=basis,points=points,H=h,Houtside=outside_h,R=exact_r,
                photon_dimension=phdim,dimension=basis.shape[1],
                real_harmonic_exact_halfheat_compression=True,
                actual_halfheat_omitted_Hermite_operator_norm=heat_leak,
                outside_heat_regulator='exp(-P Hout P/2) from the declared Galerkin quadratic form; NOT P exp(-Hout/2) P, and no uniform continuum source-price is inferred.',
                no_normalizable_harmonic_ground_claim=True)


def cutoff(radius: float,limit: float) -> float:
    if radius<=limit/2:return 1.
    if radius>=limit:return 0.
    s=2*radius/limit-1
    a,b=math.exp(-1/s),math.exp(-1/(1-s))
    return b/(a+b)


def compress(basis: np.ndarray,blocks: np.ndarray) -> np.ndarray:
    result=np.einsum('ni,nfg,nj->ifjg',basis,blocks,basis,optimize=True)
    return result.reshape(basis.shape[1]*blocks.shape[1],-1)


def native_law(p: NativeParameters) -> tuple[Any,np.ndarray,np.ndarray]:
    law=alpha.CartesianCWilson(alpha.gamma_frame(),alpha.Parameters(mass=p.critical_bare_mass,spacing=p.spacing),np.array([[1,0,0],[0,1,1]]))
    original=json.loads(FROZEN_NATIVE_BYTES)
    gram=json.loads(original['capsule']['input_json'])['NEW_law_inputs']['Phi_reference_gram']
    reference=np.sqrt(gram)[:,None,None]*alpha.SELECTED
    direction=2*alpha.cartesian_C_normal_geometry(law,reference)['direction']
    return law,reference,direction


def native_factor(p: NativeParameters,number: int,centre: int,field: dict[str,Any],
                  directions: tuple[np.ndarray,np.ndarray] | None = None,
                  reference_fields: bool = False) -> dict[str,Any]:
    """Actual COMPLETE columns of BOTH ordered factors, including outside heat."""
    law,reference,direction=native_law(p)
    graph=real_graph();e=alpha.charge_contract(alpha.Parameters())['bare_e']
    _,masks=alpha.second_quantized(np.eye(8),number);size=len(masks)
    rho=np.array([[(mask & 15).bit_count()-2,(mask >> 4).bit_count()-(2 if number==4 else 3)] for mask in masks],float)
    if np.max(abs(np.sum(rho,axis=1)))>0:raise ArithmeticError('Ref4->3 real-Gauss total charge failed')
    coulomb=.5*p.harmonic_free_coefficient*e**2*np.einsum('ni,ij,nj->n',rho,graph['Lplus'],rho)
    basis=field['basis']; chart=[];outside=[];psis=[];chis=[]
    keys=('',) if directions is None else ('','x','y','xy')
    chartjets={key:[] for key in keys};outsidejets={key:[] for key in keys}
    for cover,y,xi in field['points']:
        phases=(e*graph['Q']@cover).reshape(2,3)
        colour=np.broadcast_to(np.eye(4),(2,3,4,4)).astype(complex).copy()
        colour[0,0]=np.exp(1j*np.pi*centre/2)*expm(1j*xi*alpha.matrix_unit_basis()[7]/math.sqrt(8))
        remainder=y*np.broadcast_to(direction,(2,4,4,4))
        potential=law.potential(phases,np.zeros_like(remainder) if reference_fields else remainder,colour,directions)
        generators=[]
        for key in keys:
            g,_=alpha.second_quantized(potential['h'+key],number)
            generators.append((g+potential['pref'+key]*np.eye(size))/2)
        values=(expm(generators[0]),) if directions is None else alpha.matrix_exp_jet(*generators)
        chi=cutoff(abs(y),p.chart_radius);chis.append(chi)
        for key,value in zip(keys,values,strict=True):chartjets[key].append(chi*value)
        op=law.outside_potential(phases,np.broadcast_to(reference,(2,4,4,4))+remainder,colour,p.outside_couplings,directions)
        for key in keys:
            h,_=alpha.second_quantized(op['H'+key],number)
            outsidejets[key].append(h+op['offset'+key]*np.eye(size))
        packet=law.source_packet_jet(phases,colour,(np.zeros((2,3)),np.zeros((2,3))))
        psis.append(packet['psi'])
    dim=field['dimension']*size
    r=np.kron(field['R'],np.diag(np.exp(-coulomb/2)))
    chart={key:compress(basis,np.array(chartjets[key]))@r for key in keys}
    coupler=2*(p.outside_couplings[0]+3*sum(p.outside_couplings[1:]))
    young=3*coupler**(4/3)/(4**(4/3)*(p.outside_quartic/2)**(1/3))
    quartic=np.array([sum((np.linalg.norm(reference+y*direction)**2/4)**2 for _ in range(2)) for _,y,_ in field['points']])
    h0=np.kron(field['Houtside'],np.eye(size))+compress(basis,np.array(outsidejets['']))
    h0+=np.kron(basis.T@(p.outside_quartic*quartic[:,None]*basis),np.eye(size))+2*young*np.eye(dim)
    h0+=np.kron(np.eye(field['dimension']),np.diag(coulomb))
    generators=[-h0/2]+[-compress(basis,np.array(outsidejets[key]))/2 for key in keys[1:]]
    outexp=(expm(generators[0]),) if directions is None else alpha.matrix_exp_jet(*generators)
    outside_weight=np.kron(basis.T@(np.sqrt(1-np.array(chis)**2)[:,None]*basis),np.eye(size))
    out={key:outside_weight@value for key,value in zip(keys,outexp,strict=True)}
    f={key:np.hstack((chart[key],out[key])) for key in keys}
    centre_energy=p.centre_penalty*(1-math.cos(np.pi*centre/2))
    f0=f[''];kernel=f0@f0.conj().T
    values,vectors=eigh((kernel+kernel.conj().T)/2)
    if values[0]<=0:raise ArithmeticError('Actual complete NEW C real-cover positive spectrum unresolved')
    result=dict(values=values*math.exp(-centre_energy),vectors=vectors,masks=masks,
                centre_energy=centre_energy,dimension=dim,complete_column_count=f0.shape[1],
                outside_H_min=float(eigh(h0,eigvals_only=True)[0]),outside_Young_shift=2*young,
                outside_factor_norm=float(np.linalg.norm(out[''],'fro')),
                chart_outside_overlap=float(np.linalg.norm(chart['']@out[''].conj().T,'fro')),
                source_packet_blocks=np.array(psis),normal_radius_coordinates=[float(abs(y)) for _,y,_ in field['points']],
                literal_real_cover=True)
    if directions is not None:result['F']=f
    return result


def source_coefficients(field: dict[str,Any],packets: np.ndarray) -> np.ndarray:
    basis=field['basis']
    return np.einsum('ni,nba,nj->baij',basis,packets,basis,optimize=True)


def creation_matrices(masks4: list[int],masks5: list[int]) -> np.ndarray:
    lookup={m:i for i,m in enumerate(masks5)}
    result=np.zeros((8,len(masks5),len(masks4)))
    for j,mask in enumerate(masks4):
        for bit in range(8):
            if not mask & (1<<bit):result[bit,lookup[mask|(1<<bit)],j]=(-1)**((mask&((1<<bit)-1)).bit_count())
    return result


def apply_source(coefficients: np.ndarray,creation: np.ndarray,state: np.ndarray) -> np.ndarray:
    tensor=state.reshape(coefficients.shape[-1],creation.shape[-1])
    return np.einsum('baji,if,bgf->ajg',coefficients,tensor,creation,optimize=True).reshape(4,-1)


def apply_receiver(coefficients: np.ndarray,creation: np.ndarray,state: np.ndarray) -> np.ndarray:
    tensor=state.reshape(4,coefficients.shape[-1],creation.shape[1])
    return np.einsum('baji,ajg,bgf->if',coefficients.conj(),tensor,creation,optimize=True).reshape(-1)


def physical_H_jet(k: np.ndarray,kx: np.ndarray,kxy: np.ndarray,
                   neutral_lambda: float,spacing: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """THIS reflected transfer's spatial H/current/contact, fixed neutral norm."""
    logarithm,first,_,contact=alpha.matrix_log_jet(k,kx,kx,kxy)
    return (-(logarithm-math.log(neutral_lambda)*np.eye(len(k)))/spacing,
            -first/spacing,-contact/spacing)


def temporal_pe_jet(k: np.ndarray,kx: np.ndarray,kxy: np.ndarray,
                    temporal_density: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Exact exp(i rho.A0)K external Pe insertions, including temporal seagull."""
    density=temporal_density
    return (1j*density@k+kx,
            -density@density@k+2j*density@kx+kxy)


def native_consumer(p: NativeParameters, preparation: str = 'Gaussian',
                    factors: tuple[Any,...] | None = None) -> dict[str,Any]:
    field=field_regulator(p);law,_,_=native_law(p)
    mean=law.potential(np.zeros((2,3)),np.zeros((2,4,4,4)),np.broadcast_to(np.eye(4),(2,3,4,4)).copy())
    generator,_=alpha.second_quantized(mean['h'],4)
    _,frame=eigh(generator);sea=frame[:,-1]
    if factors is None:
        neutral=[];charged=[];reference_neutral=[]
        for z in range(4):
            neutral.append(native_factor(p,4,z,field));charged.append(native_factor(p,5,z,field))
            reference_neutral.append(native_factor(p,4,z,field,reference_fields=True))
    else:
        neutral,charged,reference_neutral=factors
    reference=max(row['values'][-1] for row in neutral)
    reference0=max(row['values'][-1] for row in reference_neutral)
    boundary=[];prepared=[];prepared0=[];normalizer=normalizer0=0.
    for row,row0 in zip(neutral,reference_neutral,strict=True):
        xi=np.zeros(row['dimension'],complex);xi[:70]=sea/2
        if preparation=='smooth_reference_energy':
            energy=-np.log(row0['values']/reference0)
            delta=1/2**20
            bump=np.zeros_like(energy);support=(energy>=0)&(energy<delta)
            ratio=energy[support]/delta
            bump[support]=np.exp(-ratio**2/(1-ratio**2))
            xi=row0['vectors']@(bump*(row0['vectors'].conj().T@xi))
        coefficients=row['vectors'].conj().T@xi
        evolved=row['vectors']@((row['values']/reference)**p.prefix_steps*coefficients)
        evolved0=row0['vectors']@((row0['values']/reference0)**p.prefix_steps*(row0['vectors'].conj().T@xi))
        boundary.append(xi);prepared.append(evolved);prepared0.append(evolved0)
        normalizer+=float(np.linalg.norm(evolved)**2);normalizer0+=float(np.linalg.norm(evolved0)**2)
    preparation_gram=sum(float(np.linalg.norm(x)**2) for x in boundary)
    if preparation_gram<=0:raise ArithmeticError('Actual neutral preparation Gram is zero')
    boundary=[x/math.sqrt(preparation_gram) for x in boundary]
    prepared=[x/math.sqrt(preparation_gram) for x in prepared]
    prepared0=[x/math.sqrt(preparation_gram) for x in prepared0]
    normalizer/=preparation_gram;normalizer0/=preparation_gram
    weighted_relative=math.sqrt(sum(float(np.linalg.norm(a-b)**2) for a,b in zip(prepared,prepared0,strict=True))/normalizer0)
    normalized_distance=math.sqrt(sum(float(np.linalg.norm(a/math.sqrt(normalizer)-b/math.sqrt(normalizer0))**2) for a,b in zip(prepared,prepared0,strict=True)))
    n,n0=neutral[0],reference_neutral[0]
    adverse=n0['vectors'][:,0]
    adverse0=(n0['values'][0]/reference0)**p.prefix_steps*adverse
    adverse1=n['vectors']@((n['values']/reference)**p.prefix_steps*(n['vectors'].conj().T@adverse))
    adverse_ratio=float(np.linalg.norm(adverse1-adverse0)/np.linalg.norm(adverse0))
    if normalizer<=0:raise ArithmeticError('Actual boundary/prefix norm is zero')
    prepared=[state/math.sqrt(normalizer) for state in prepared]
    reference_prepared=[state/math.sqrt(normalizer0) for state in prepared0]
    phdim=field['photon_dimension']
    if phdim!=6:raise ValueError('Native photon/source consumer needs all five single-excitation modes')
    b_in=np.zeros((phdim,phdim));b_in[1,0]=1
    b_out=np.zeros((phdim,phdim));b_out[0,2]=1
    incoming=np.kron(np.kron(b_in,np.eye(p.normal_order)),np.eye(p.colour_order))
    outgoing=np.kron(np.kron(b_out,np.eye(p.normal_order)),np.eye(p.colour_order))
    input_states=[(incoming@state.reshape(field['dimension'],70)).reshape(-1) for state in prepared]
    photon_gram=sum(float(np.linalg.norm(state)**2) for state in input_states)
    if photon_gram<=0:raise ArithmeticError('Actual incoming reference photon packet vanishes')
    input_states=[state/math.sqrt(photon_gram) for state in input_states]
    reference_inputs=[(incoming@state.reshape(field['dimension'],70)).reshape(-1) for state in reference_prepared]
    reference_photon_gram=sum(float(np.linalg.norm(state)**2) for state in reference_inputs)
    actual_photon_weighted_error=math.sqrt(sum(float(np.linalg.norm(a*math.sqrt(photon_gram)-b)**2) for a,b in zip(input_states,reference_inputs,strict=True))/reference_photon_gram)
    raw=half=moment=pe4=negative=0.;vectors=[];source_data=[]
    for n,c,state in zip(neutral,charged,input_states,strict=True):
        coefficients=source_coefficients(field,n['source_packet_blocks'])
        creation=creation_matrices(n['masks'],c['masks'])
        source=apply_source(coefficients,creation,state)
        spectral=source@c['vectors'].conj()
        energies=-np.log(c['values']/reference)
        weighted=spectral*np.sqrt(c['values']/reference)
        weights=np.sum(abs(weighted)**2,axis=0)
        raw+=float(np.linalg.norm(source)**2);half+=float(sum(weights));moment+=float(weights@energies)
        pe4+=float(sum(weights[(energies>=0)&(energies<=4)]))
        negative+=float(sum(weights[energies<0]))
        vectors.append((weighted,energies));source_data.append((coefficients,creation))
    if half<=0:raise ArithmeticError('Actual complete charged halfstep source vanishes')
    receivers=[]
    for n,target in zip(neutral,prepared,strict=True):
        coefficients=source_coefficients(field,n['source_packet_blocks'])
        creation=creation_matrices(n['masks'],charged[0]['masks'])
        source_target=apply_source(coefficients,creation,target).reshape(4,field['dimension'],56)
        receivers.append(np.einsum('ij,ajf->aif',outgoing.conj().T,source_target,optimize=True).reshape(4,-1))
    receiver_gram=sum(float(np.linalg.norm(r)**2) for r in receivers)
    if receiver_gram<=0:raise ArithmeticError('Actual packet receiver Gram vanishes')
    events=[]
    for t in (0.,.25,.5,1.):
        amplitude=0j
        for c,receiver,(weighted,energies) in zip(charged,receivers,vectors,strict=True):
            state=(weighted*np.exp(-1j*t*energies/p.spacing))@c['vectors'].T
            amplitude+=np.vdot(receiver,state)/math.sqrt(receiver_gram*half)
        probability=float(abs(amplitude)**2)
        if probability>1+1e-9:raise ArithmeticError('Actual rank1 packet receiver probability exceeds1')
        events.append(dict(time_PHYSICAL=t,actual_reference_packet_receiver_probability=probability))
    gaps=[float(-math.log(c['values'][-1]/reference)) for c in charged]
    result=dict(actual_same_neutral_lambda_N=float(reference),actual_boundary_W_trace=sum(float(np.linalg.norm(x)**2) for x in boundary),
        boundary_preparation=preparation,actual_preparation_Gram=preparation_gram,
        actual_preparation_cost_minus_log=-math.log(preparation_gram),
        reference_tau_lower=math.exp(-p.prefix_steps/2**20) if preparation=='smooth_reference_energy' else None,
        actual_boundary_prefix_norm_squared=normalizer,prefix_steps=p.prefix_steps,
        actual_log_tau=.5*math.log(normalizer),
        actual_reference_log_tau=.5*math.log(normalizer0),
        actual_SAME_boundary_weighted_relative_prefix_error=weighted_relative,
        actual_normalized_prefix_vector_error=normalized_distance,
        relative_weighted_normalized_error_upper=2*weighted_relative,
        actual_near_null_same_boundary_weighted_relative_error=adverse_ratio,
        actual_near_null_reference_log_tau=math.log(float(np.linalg.norm(adverse0))),
        exact_heat_regulator=dict(actual_harmonic_halfheat_omitted_Hermite_operator_norm=field['actual_halfheat_omitted_Hermite_operator_norm'],
            chart_photon_heat='P exp(-Hreference/2) P, exact compressed freeharmonic matrix; no normalizableharmonicground.',
            outside=field['outside_heat_regulator']),
        actual_incoming_reference_photon_Gram=photon_gram,actual_raw_Ref4_to3_source_Gram=raw,
        actual_REFERENCE_incoming_projected_photon_Gram=reference_photon_gram,
        actual_projected_photon_weighted_relative_prefix_error=actual_photon_weighted_error,
        actual_packet_receiver_Gram=receiver_gram,
        actual_complete_charged_half_Gram=half,actual_first_output_mean_TRANSFER=moment/half,
        literal_TRANSFER_Pe4_Gram=pe4,literal_negative_energy_weight=negative,
        actual_charged_top_gap_TRANSFER=gaps,actual_charged_top_gap_PHYSICAL=[g/p.spacing for g in gaps],
        conditional_Markov_Pe4_lower=half*(1-moment/half/4) if negative==0 else None,
        Jensen_upper_TRANSFER=-math.log(half/raw) if min(gaps)>=0 else None,
        actual_receiver_CP_events=events,
        centre_rows=[dict(centre=z,neutral_dimension=n['dimension'],charged_dimension=c['dimension'],
            neutral_complete_columns=n['complete_column_count'],charged_complete_columns=c['complete_column_count'],
            neutral_all_positive_spectral_count=len(n['values']),charged_all_positive_spectral_count=len(c['values']),
            centre_energy=n['centre_energy'],outside_H_min=n['outside_H_min'],outside_factor_norm=n['outside_factor_norm'],
            chart_outside_overlap=n['chart_outside_overlap']) for z,(n,c) in enumerate(zip(neutral,charged,strict=True))],
        scope='ACTUAL NEW ordered fieldword, real5-coordinate Hermite regulator, all70/56 columns and all4native centre-cycle branches, bothchart/outside columns. Native hasNaux0, one45normal direction/one15Lie direction; macro allPhi/allSU4/allNaux and physical mass1/error are NOT inferred from this regulator.')
    if preparation=='Gaussian':
        result['optional_NEW_smooth_neutral_preparation']=native_consumer(p,'smooth_reference_energy',(neutral,charged,reference_neutral))
    return result


def gauss_and_real_cover_probe() -> dict[str,Any]:
    graph=real_graph();q=np.array([.13,-.21,.32,-.17,.29]);lam=np.array([.3,-.3]);rho=np.array([1.,-1.])
    theta=graph['Q']@q+graph['D'].T@lam
    pi=graph['D'].T@graph['Lplus']@rho
    longitudinal_error=float(np.linalg.norm(graph['D']@pi-rho))
    quotient_error=float(np.linalg.norm(graph['Q'].T@theta-q))
    # A real-cover Gaussian distinguishes q from a large-link-period translation.
    shift=graph['Q'].T@(2*np.pi*np.array([1.,0.,0.,0.,0.,0.]))
    packet_overlap=math.exp(-float(shift@shift)/4)
    return dict(actual_incidence=graph['D'].tolist(),actual_real_cover_basis=graph['Q'].tolist(),
        real_cover_dimension=5,retained_coexact_modes=2,retained_REAL_harmonics=3,
        Gauss_longitudinal_error=longitudinal_error,quotient_reconstruction_error=quotient_error,
        actual_real_Gaussian_large_link_translation_overlap=packet_overlap,
        wrong_compactified_overlap=1.,normalized_noncompact_Haar_projection_exists=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--quadrature-order',type=int,default=2)
    parser.add_argument('--prefix-steps',type=int,default=4)
    args=parser.parse_args();started=time.time()
    parameters=NativeParameters(quadrature_order=args.quadrature_order,prefix_steps=args.prefix_steps)
    law,reference,_=native_law(parameters);geometry=slice_geometry(reference)
    result=dict(input_parameters=asdict(parameters),full_family_definition=full_family_definition(),
        actual_real_Gauss_cover=gauss_and_real_cover_probe(),
        actual_regular_slice=dict(normal_dimension=geometry['normal_dimension'],normal_defect=geometry['normal_defect'],
            min_R0_eigenvalue=float(eigh(geometry['R0'],eigvals_only=True)[0]),connected_diagonal_Z4_degree=4),
        actual_complete_native_source_consumer=native_consumer(parameters))
    inputs=json.dumps(dict(parameters=asdict(parameters),frozen_Alpha56=ledger.FROZEN_HASH,
                          captured_ledger=LEDGER_HASH,new_fieldword=True,real_EM_cover=True),sort_keys=True,separators=(',',':'))
    result['execution']=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),runtime_seconds=time.time()-started,
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__,platform=platform.platform()),
        captured_ledger_sha256=LEDGER_HASH,captured_ledger_bytes=LEDGER_BYTES.decode(),
        captured_Alpha56_native_capsule_sha256=FROZEN_NATIVE_HASH,
        frozen_Alpha56_source_at_IMPORT=alpha.SOURCE_HASHES,frozen_Alpha56_source_bytes_at_IMPORT={key:value.decode() for key,value in alpha.SOURCE_BYTES.items()},
        own_startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    native=result['actual_complete_native_source_consumer']
    print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],input_sha256=result['execution']['input_sha256'],
        same_lambda_N=native['actual_same_neutral_lambda_N'],source_half_Gram=native['actual_complete_charged_half_Gram'],
        Pe4_Gram=native['literal_TRANSFER_Pe4_Gram'],mean_TRANSFER=native['actual_first_output_mean_TRANSFER'],runtime_seconds=result['execution']['runtime_seconds'])))


if __name__=='__main__':
    main()
