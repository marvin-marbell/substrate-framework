#!/usr/bin/env python3
"""Original full-Fock Dirac-Maxwell-Coulomb reference, not a one-ray closure.

The physical initial fn/Gamma/SOURCE/photon/ordinary phases are unchanged.
Only paid light/Ref fast connections and Phi/colour light coefficients are
removed. Full photon/current, Coulomb multipair, heavy and neutral RIGHT stay.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any
if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from functional_source_photon_bank import Full108ReferenceMarks
from improved_phi_band import NSTAR, UE
from prepared_neutral_shared_photon_state import load_shared_native_cache, loaded_source_epoch
from prepared_neutral_slow_light_reference import C2_SHA, ceil_log2
from prepared_neutral_full_field import BETA
from original_charged_principal import portable
from sea_selection import SELECTED

FIELD_SHA='9c05f1bc54a0a6576f0208897f81314d877f305d1986dd79fb20e48306c2492c'
ORDINARY_SHA='64612effb995b3715d7dbc7d49cf0a1b7d3daef415986e730e32198924df47f4'
DESCRIPTOR_SHA='020fb175c9231b607fb0e53b016cf97528af9065a634dedc7ddea82bdac3556c'


@dataclass(frozen=True)
class CoulombPacketAction:
    """RIGHT-ordered neutral density, never its expectation or weak matrix.

    [HC,B_S]=sum_x B_{S_x}(2UE V_x(rho)+UE W_x); negative transfer
    reverses the first term only. rho=n_light-2+(Nref-4)delta_FAR.
    HC has NO factor1/2. Density action creates actual multipair states.
    """
    marks: Full108ReferenceMarks
    density_kernel: np.ndarray
    self_energy: np.ndarray

    def on_density_eigenvector(self, rho: np.ndarray) -> Full108ReferenceMarks:
        """Scalar specialization ONLY when the right input is a density eigenvector."""
        rho=np.asarray(rho,float)
        if rho.shape!=self.self_energy.shape:
            raise ValueError('One original density eigenvalue per site required')
        potential=self.density_kernel@rho
        return Full108ReferenceMarks(
            np.repeat(potential+self.self_energy,4)[:,None]*self.marks.plus,
            self.marks.minus*np.repeat(-potential+self.self_energy,4)[None,:])


class MeanPacketIntertwiner:
    """Existing arbitrary full108/Ref4 ABI; never an eight-ray packet fit."""
    def __init__(self, model: Any):
        self.model=model

    def light(self, A: np.ndarray, first: np.ndarray | None=None,
              second: np.ndarray | None=None) -> np.ndarray:
        return self.model.law.onebody(self.model.chart.mean_phi,self.model.chart.mean_links,
                                     A,first=first,second=second)

    @staticmethod
    def onebody_action(marks: Full108ReferenceMarks, body: np.ndarray,
                       reference: np.ndarray | None=None) -> Full108ReferenceMarks:
        value=marks.light_action(body)
        if reference is None:
            return value
        right=marks.reference_action(reference)
        return Full108ReferenceMarks(value.plus+right.plus,value.minus+right.minus)

    def coulomb_action(self, marks: Full108ReferenceMarks) -> CoulombPacketAction:
        return coulomb_packet_action(marks,self.model.green,self.model.ue,self.model.reference_site)


def coulomb_packet_action(marks: Full108ReferenceMarks, green: np.ndarray,
                          ue: float, reference: int) -> CoulombPacketAction:
    """All sites/all Ref and both maps; source denominator remains caller-owned."""
    green=np.asarray(green,float)
    volume=len(green)
    if green.shape!=(volume,volume) or marks.plus.shape!=(4*volume,4) or marks.minus.shape!=(4,4*volume):
        raise ValueError('Full site Green and uncropped physical packet marks required')
    if not 0<=reference<volume or ue<0:
        raise ValueError('Original reference and nonnegative electric coefficient required')
    return CoulombPacketAction(marks,2*ue*(green-green[reference]),
        ue*(np.diag(green)+green[reference,reference]-2*green[:,reference]))


def photon_multiplication_commutator(gradient: np.ndarray, laplacian: complex,
                                    electric_coefficient: float) -> dict[str, Any]:
    """[e Pi²,F]=-e Delta F-2i e grad F.Pi with Pi on the RIGHT.

    e=UE in physical coexact A coordinates, e=UE/N for macrotheta.
    Magnetic multiplication commutes F. This is not a vacuum compression.
    """
    return dict(multiplication=-electric_coefficient*laplacian,
                right_momentum_coefficients=-2j*electric_coefficient*np.asarray(gradient),
                right_momentum_not_vacuum_expectation=True)


def full_fock_reference_contract(field: dict[str, Any], c2: dict[str, Any]) -> dict[str, Any]:
    """State-independent comparison operator identities + paid actual rows.

    Href retains Phi/heavy normal kinetics with Qheavy and actual normal
    Noether/timeconnection, FULL EM/HC, mean-colour light/orbital and derived
    mean light/Ref Gauss Noether. Removing LR covariant charges changes only
    cross/vertical terms; no second reference-state graph is needed.
    """
    legal=field['legal'];volume=int(legal['LEGAL_N'])**3
    v=ceil_log2(volume);lam=int(c2['science']['selected_log2_Lambda_lower']);clock=int(legal['clock'])
    # Conservative original V=sum_j sup|v_j|<=3 Rout (not a fitted constant).
    Rout=2684354560000;velocity=3*Rout
    cb=6+(v+1)//2
    # Original D has gInv<=160, J<=16 and radius<=1. Very loose
    # differentiated J/Y bounds 2^16(1+velocity), with ||Y0||<=
    # 2^16(1+velocity)sqrt(V), bound rG=gGG^-1 JG†Y and the ACTUAL
    # beta chi gGG^-1 gGN gNN^-1 a. gGN(0)=0 in the mean normal slice.
    # In particular gNN^-1(q)gNN(0) may cost160*256; it is NOT cancelled.
    # Full-CAR ||Q_LR||row<=32sqrt(V), ||B0||<=64sqrt(V); integral
    # (|Omega|+|beta|)<=32. 2^75 V²(1+velocity) dominates their product.
    geo_coefficient=75+2*v+ceil_log2(1+velocity)
    rows={}
    for label,row in legal['results'].items():
        mix=row['full_Dirichlet_mixed_graph']
        phi=row['relative_Phi_kinetic']
        light=row['actual_Dirichlet_light_freeze_norm_error_log2_upper']+cb+1
        # d-variation of original scalar_orbital: every oriented nearest
        # edge unitary difference <=2d, |v_j| bounded by the original guard.
        # Generic full encoded triangle <=2^8 V(1+velocity)d.
        # Integral|Omega|<=8, no T multiplier for an angular generator.
        # q2 is recovered from exact actual C2 mixed coefficient, never fitted.
        d2c=mix['distance_second_moment_coefficient_log2_upper']
        dlog=(d2c+1)//2-lam//4
        geometric=geo_coefficient+dlog
        budgets=[light,phi['cross_log2_upper'],
            phi['density_and_coefficient_derivative_log2_upper'],
            phi['full_vertical_Gauss_log2_upper'],
            phi['connection_square_log2_upper'],geometric,
            row['original_whole_to_Dirichlet_norm_error_log2_upper'],
            row['initial_SAME_Z_BminusB0_norm_error_log2_upper'],
            row['actual_auxiliary_projection']['combined_auxiliary_norm_error_log2_upper']]
        rows[label]=dict(d2_coefficient_log2_upper=d2c,
            light_commutator_integral_log2_upper=light,
            absolute_LR_fast_split_cross_integral_log2_upper=phi['cross_log2_upper'],
            absolute_LR_density_divergence_integral_log2_upper=phi['density_and_coefficient_derivative_log2_upper'],
            absolute_LR_full_vertical_integral_log2_upper=phi['full_vertical_Gauss_log2_upper'],
            absolute_LR_connection_square_integral_log2_upper=phi['connection_square_log2_upper'],
            mechanical_AND_actual_fixed_boost_mean_freeze_integral_log2_upper=geometric,
            total_reference_comparison_log2_upper=max(budgets)+4,
            preserved_original_whole_boundary_log2_upper=row['original_whole_to_Dirichlet_norm_error_log2_upper'],
            actual_auxiliary=row['actual_auxiliary_projection'],
            reference_moment_not_assumed=True)
    return dict(reference='FULL even-CAR/Fock+EM/harmonic on flat-P directsum/all8 twists/all64 aliases; actual encodedQ and PDQ forcing paid, NOT whole-PH CAR or range of one B_f Xi',
        Phi_fast='Actual Phi/colour positive form with Qheavy only, full heavy Hamiltonian/orbitals; actual normal Noether and IC/timeconnection remain common',
        light='Mean actual Phi/colour in ORIGINAL ImprovedPhiLaw, arbitrary full Peierls A; original current/contact are its derivatives',
        Coulomb='UE rho^T Lplus rho with actual light+Ref density and global FAR compensation; right-density terms create multipair, never discarded',
        photon='Original UE Pi²+(K/2)curl², canonical all coexact/harmonic; literal sine Noether and Gauss reconstruction unchanged',
        mean_Noether='Derived actual YG0=omega_mech0, true Ref RIGHT; scalar_orbital with mean colour and full Peierls, literal EM/rho.lambda_C',
        domain='Original H1_0 union, physical Gauss body reconstruction; slow EM common closed-form domain; full finite Fock',
        mechanical_comparison=dict(coefficient_log2_upper=geo_coefficient,
            original_guard_velocity_upper=velocity,original_radius_upper=1,
            metric_upper=256,cometric_upper=160,
            geometric_AND_actual_fixed_boost_varies=True,
            no_pointwise_boost_cancellation=True,
            mean_generator_is_derived_NOT_assigned_halfspin=True),
        orientation='P PsiD(t)-Uref_P(t)P PsiD(0)=-i integral Uref_P(t,s)[(P H_D P-Href_P)P PsiD(s)+P D Q PsiD(s)]ds; add actualQ and original whole-boundary remainder WITHOUT physically projecting state',
        no_state_sourceZ_or_detector_replacement=True,
        source='Original B0_f Xi is the prepared charged packet, not an invariant one-particle subspace; fullFock identification is proofP/codeflow, not physical projection',
        conserved_Ref='Full H/G/J/filter preserve neutral Ref4/Ref0 and charged Ref3/Ref1; true RIGHT acts on the latter, not assigned neutral spin',
        optional_transfer='T_t=Uslow,C B0 Uslow,N^dagger is explicitly smaller mean slow reference, NOT defined using actual fullU; q-independent, norm C_B preserved, full Coul/photon/multipair retained',
        scalar_Ref_bound='On original Ref4/Ref0 roots ||B_f Xi||<=max(||S||F,||T||F)||Xi||; does not treat right-density operator-valued S as a scalar',
        orbital_colour_difference_coefficient_log2_upper=8+v+ceil_log2(1+velocity),
        original_Rout=Rout,clock=clock,rows=rows,
        still_to_join=['Core actual mean mechanical group and selected narrower LEGAL SOFT, not wide native SOFT','Original current finiteT photon transition/remainder from Transport','Quantum law/parameters'])


def full_fock_coulomb_bound() -> dict[str, Any]:
    """Full physical light+Ref Fock, not a one-particle Coulomb potential.

    At every site nL-2 is in[-2,2]; only FAR adds nRef-4 in[-4,0].
    Thus sum rho_x²<=4V+32. Discrete gap4sin²(pi/N)>=16/N².
    """
    volume=int(NSTAR)**3
    bound=UE*F(int(NSTAR)**2*(4*volume+32),16)
    return dict(original_UE_numerator=str(UE.numerator),original_UE_denominator=str(UE.denominator),
        original_N=int(NSTAR),all_Fock_rho_squared_upper=4*volume+32,
        Green_operator_upper='N²/16',HC_operator_log2_upper=ceil_log2(bound),
        no_factor_half=True,multipair_retained=True,
        caution='This norm bound alone MUST NOT drop HC at UE/V Born amplitude scale; use FULL HC interaction picture or linked-current correction',
        CAR_domain='flat-P fullFock directsum, actual encodedQ separately paid')


def native_massive_mobile(model: Any) -> dict[str, Any]:
    """Exact translation symbol of the ORIGINAL mean law, not compression.

    This is native evidence; selected-SOFT dispersion/remainder and physical
    mean group belong to Core. No physical mass is inferred from an8-profile fit.
    """
    law=model.law;phi=model.chart.mean_phi[0]
    def symbol(p: np.ndarray) -> np.ndarray:
        out=law.onsite*phi[0].copy()
        for displacement,scalar in law.paths:
            block=np.einsum('m,mij->ij',scalar,phi)
            phase=np.exp(1j*np.dot(displacement,p));out+=phase*block+phase.conjugate()*block.conj().T
        return out
    h0=MeanPacketIntertwiner(model).light(np.zeros((law.volume,3)))
    embedding=np.tile(np.eye(4),(law.volume,1))/math.sqrt(law.volume)
    zero=symbol(np.zeros(3));error=float(np.linalg.norm(h0@embedding-embedding@zero))
    eig=np.linalg.eigvalsh(zero);mass=float(eig[2])
    if mass<=0 or error>1e-11:raise ValueError('Original translational positive-mass symbol failed')
    speeds=[]
    for step in (1e-3,5e-4):
        row=[]
        for j in range(3):
            p=np.zeros(3);p[j]=step;energy=float(np.linalg.eigvalsh(symbol(p))[2])
            row.append((energy**2-mass**2)/step**2)
        speeds.append(row)
    if min(speeds[-1])<=0:raise ValueError('Actual massive symbol is not mobile')
    return dict(exact_native_translation_symbol_error=error,original_rest_eigenvalues=eig,
        native_positive_rest_mass=mass,energy_squared_curvature_estimates=speeds,
        hole_symbol='-h^T, NOT refilled projector or new law',
        fullFock_lift='[dGamma(h),B_S]=B_hS; exact for ANY neutral input on flatP, including multipair',
        law_parameters_not_derived=True,physical_SOFT_and_halfspin_not_inferred_from_native_symbol=True)



def ordinary_reference_contract(epoch: dict[str, Any]) -> dict[str, Any]:
    """Ordinary Omega=0/T6, NEVER relabel the long driven C2 science.

    The original actual ordinary normal/right-form proof's light/Berry rows
    remove the light/Ref fast coupling; all heavy/normal, EM/Coulomb remain.
    Weighted rows dominate unweighted comparison; no creator acts on a residual.
    """
    lam=int(epoch['ordinary_weighted_OUT']['selected_log2_lambda_lower'])
    def power(value: dict[str,str]) -> F:
        return F(int(value['exact_numerator_hex'],16),int(value['exact_denominator_hex'],16))
    rows=epoch['ordinary_weighted_OUT']['weighted_light_normal_Duhamel_terms']
    errors=[row['log']+math.ceil(power(row['power'])*lam) for row in rows]
    v=ceil_log2(int(NSTAR)**3);qlog=7-math.floor(F(lam,8));aux=qlog+14+(v+1)//2
    return dict(protocol='ORIGINAL ordinary Omega=0/T6',duration=6,
        actual_initial_state=epoch['initial_energy']['identity'],
        original_H_energy_origin='UNSHIFTED Cidentity=0, actual absolute8V Lambda',
        actual_ordinary_normal_reference=epoch['light_normal_shadow']['shadow'],
        original_field_rows=rows,field_comparison_log2_upper=max(errors)+2,
        actual_Q_norm_log2_upper=qlog,actual_Q_AND_original_beta_PD_Q_log2_upper=aux,
        total_unweighted_norm_error_excluding_separately_SUPEREXP_Agmon_log2_upper=max(max(errors),aux)+3,
        whole_boundary='Original ORDINARY SUPEREXP Agmon plus conserved firstKEM graph, NOT driven C2 boundary; already in frozen ordinary ledger',
        actual_encoded_state_not_projected=True,CAR_only_flatP=True,
        no_driven_energy_or_photon_graph_retag=True,no_OUT_creator_through_norm_error=True,
        true_OUT='a_gout_j†(Bplus_j+Bminus_j)U_Nactual(6)Xi; fixed original gout and ONE globalG/E2',
        global_initial_annihilator_tensors=epoch['ordinary_weighted_OUT']['initial_annihilator_tensor_bounds_neutral_and_normalized_SOURCE'],
        global_initial_truephoto_tensors=epoch['ordinary_weighted_OUT']['initial_annihilator_tensor_bounds_normalized_true_creator'],
        original_allmode_displacement_square_upper='16 UE/c',
        initial_absolute_energy=epoch['initial_energy']['absolute_initial_energy_upper'],
        whole_phi_quartic=dict(rho_squared='sum_xmu Tr(Phi_xmu²)/4',
            original_stiffness=2,original_elastic_lower='rho^4/(4V)',
            heavy_PLUS_light_coefficient_triangle_upper=65536,
            original_beta_lower='-V',Young='2^16 sqrtV rho <= rho^4/(8V)+2^22 V',
            rho_fourth_expectation_upper='2^26 V²',HS_Phi_fourth_expectation_upper='2^30 V²',
            no_global_CAR_assumed=True),
        remaining_OUT='Transport owns GLOBAL ordinaryN² propagation, lateB/B0 and actualcreator interpolation plus literal jointGram/current')



def global_source_coframe_bound(model: Any, source_descriptor: dict[str, Any]) -> dict[str, Any]:
    """Exact generic coframe certificate, original a.e SOURCE; no fallback.

    Gram-positive4-tuples in Herm0(4)^4 form a connected real-analytic domain.
    Whiten analytically. Discriminant and resultant of L=system†system with
    determinant of reshaped adjugate-column are analytic. ONE exact point
    with both nonzero proves their exceptional set null on that domain,
    covering ALL eigenbranches, not merely the selected minimum at Clifford.
    Finite numeric guard thresholds are NOT physical projections.
    """
    import sympy as sp
    prime=101;imag=10
    I=sp.eye(2);X=sp.Matrix([[0,1],[1,0]])
    Y=sp.Matrix([[0,-imag],[imag,0]]);Z=sp.diag(1,-1);pauli=[I,X,Y,Z]
    basis=[sp.kronecker_product(a,b) for i,a in enumerate(pauli)
           for j,b in enumerate(pauli) if i or j]
    vector_values=[1,3,4,7,11,13,16,17,20,23,25,29,31,37,41]
    vector=sp.Matrix(vector_values)
    denominator=int(vector.dot(vector))
    house=(sp.eye(15)-2*pow(denominator,-1,prime)*vector*vector.T).applyfunc(lambda a:int(a)%prime)
    phi=[]
    for j in range(4):
        entry=sp.zeros(4)
        for i in range(15):entry+=house[i,j]*basis[i]
        phi.append(entry.applyfunc(lambda a:int(a)%prime))
    gamma=[]
    for actual in SELECTED:
        if any(z.real!=int(z.real) or z.imag!=int(z.imag) for z in actual.flat):
            raise ValueError('Original SELECTED is not the observed exact Gaussian-integer Clifford representation')
        gamma.append(sp.Matrix([[int(z.real)+imag*int(z.imag) for z in row] for row in actual]))
    # EXACT original vec_F layout, not an assigned spin representation.
    kernel=(sp.kronecker_product(sp.eye(4),sum((f*f for f in phi),sp.zeros(4)))+4*sp.eye(16)
        -2*sum((sp.kronecker_product(g.T,f) for f,g in zip(phi,gamma)),sp.zeros(16))).applyfunc(lambda a:int(a)%prime)
    x=sp.Symbol('x');polynomial=sp.Poly(kernel.charpoly(x).as_expr(),x,modulus=prime)
    raw_discriminant=polynomial.discriminant()
    if not isinstance(raw_discriminant,sp.Integer):
        raise TypeError('Finite-field characteristic discriminant must be an exact integer')
    discriminant=int(raw_discriminant)%prime
    coefficients=[int(a)%prime for a in polynomial.all_coeffs()]
    matrix=np.asarray(kernel.tolist(),dtype=np.int64)
    e=np.zeros(16,dtype=np.int64);e[0]=1;columns=[e]
    for k in range(1,16):columns.append((matrix@columns[-1]+coefficients[k]*e)%prime)
    adj=[sp.Poly(sum(int(columns[k][j])*x**(15-k) for k in range(16)),x,modulus=prime)
         for j in range(16)]
    determinant=sp.Poly(0,x,modulus=prime)
    for permutation in itertools.permutations(range(4)):
        sign=(-1)**sum(permutation[i]>permutation[j] for i in range(4) for j in range(i+1,4))
        term=sp.Poly(sign,x,modulus=prime)
        for i in range(4):term=term.mul(adj[i+4*permutation[i]]).rem(polynomial)
        determinant=determinant.add(term).rem(polynomial)
    gcd=determinant.gcd(polynomial)
    if not discriminant or gcd.degree()!=0:
        raise ValueError('Exact original SELECTED generic full-polar-rank witness failed')
    norms={name:float(np.vdot(profile,profile).real) for name,profile in model.source_bank.profiles.items()}
    beta_squared=sum(F.from_float(float(z.real))**2+F.from_float(float(z.imag))**2 for z in BETA)
    eta_squared=sum(F.from_float(z['real'])**2+F.from_float(z['imag'])**2
        for z in source_descriptor['source_eta'])
    if beta_squared>2 or eta_squared>2:
        raise ValueError('Original serialized LEGAL SOURCE19 quartet/profile norm premises fail')
    def coefficient(law: Any) -> F:
        return 8*abs(F.from_float(float(law.onsite)))+16*sum(
            abs(F.from_float(float(z.real)))+abs(F.from_float(float(z.imag)))
            for _,scalar in law.paths for z in scalar)
    heavy=coefficient(model.heavy_law);light=coefficient(model.law)
    if 4*heavy+light>65536:raise ValueError('Actual Phi-linear fullencoded coefficient triangle exceeds bound')
    return dict(actual_SELECTED_gaussian_integer=True,original_vecF_kernel=True,
        modulus=prime,imaginary_unit=imag,Householder_denominator=denominator,
        Householder_vector=vector_values,
        characteristic_coefficients_modp=coefficients,discriminant_nonzero_modp=discriminant,
        cofactor_determinant_modp=[int(a)%prime for a in determinant.all_coeffs()],
        cofactor_reshape_determinant_gcd_degree=int(gcd.degree()),
        conclusion='Exact nonzero mod101 proves nonzero overQ(i); all eigenvectors rank4 at this rational GramI point. Analytic discriminant/resultant on connected Gram>0 gives original least-mode polar unitary a.e ALLPhi',
        native_profile_norm_squared=norms,original_beta_squared_upper=2,
        LEGAL_SOURCE19_eta_squared_exact_fraction=str(eta_squared),
        LEGAL_SOURCE19_eta_squared_upper=2,LEGAL_SOURCE19_descriptor_sha256=DESCRIPTOR_SHA,
        LEGAL_SOFT_profile_norm_not_inferred_from_native_restriction=True,
        actual_heavy_linear_entry_triangle_upper=float(heavy),actual_light_linear_entry_triangle_upper=float(light),
        heavy_PLUS_light_linear_global_upper=65536,
        source_point_HS='nu_xHS=||beta|| |eta_x|: four ORIGINAL quaternion labels HSorthogonal, Fx/Fr unitary, signed screenPRODUCT modulus<=1',
        unrestricted_Phi_and_fullencoded_SOURCE_upper='8||beta||||eta||2 sqrtV for ANY original profile; <=16sqrtV for exact serialized LEGAL SOURCE19 (eta²<=2, beta²<=2); individualevenbilinear norm<=1, NO globalCAR',
        screen_proof='psi0=sum positiveweights, psi1=sum sameweights timesZ4 phase; ratio modulus<=1 on ALLSU4',
        global_coexact_number_commutation=True,
        no_physical_SOURCE_extension_or_fallback='Original a.e formula unchanged; nullsets irrelevant in L2. Native finite guard code remains unchanged',
        no_global_SOURCE_derivative_bound='M1 used ONLY on original regularD; outside use global multiplication bound',
        all_finite_paths_and512_nullset_union=True)
def transfer_matrix(marks: Full108ReferenceMarks, charge: int) -> np.ndarray:
    n=marks.plus.shape[0];out=np.zeros((n+4,n+4),complex)
    if charge==1:out[:n,n:]=marks.plus
    elif charge==-1:out[n:,:n]=marks.minus
    else:raise ValueError('Original plus/minus transfer required')
    return out


def apply_onebody(body: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Independent exact small-Fock algebra witness, no weak-fibre crop."""
    out=np.zeros_like(state)
    entries=list(zip(*np.nonzero(body)))
    for mask in np.flatnonzero(state):
        mask=int(mask)
        for i,j in entries:
            i=int(i);j=int(j)
            if not (mask>>j)&1:continue
            removed=mask^(1<<j)
            if (removed>>i)&1:continue
            sign=(-1)**((mask&((1<<j)-1)).bit_count()+(removed&((1<<i)-1)).bit_count())
            out[removed^(1<<i)]+=sign*body[i,j]*state[mask]
    return out


def native_small_fock(model: Any) -> dict[str, Any]:
    """Actual original native two-site restriction, ALL12-mode Fock, both roots.

    This discriminates ordering/multipair/factor1/2 errors; not a replacement
    preparation or a full108/LEGAL manybody simulation.
    """
    law=model.law;ref=int(model.reference_site);other=int(law.target[ref,0])
    sites=[ref,other];ids=np.concatenate([np.arange(4*x,4*x+4) for x in sites])
    A=np.random.default_rng(2222043).normal(size=(law.volume,3))*4
    light=MeanPacketIntertwiner(model).light(A);h=light[np.ix_(ids,ids)]
    values,modes=np.linalg.eigh(h);occupied=modes[:,values<0]
    if occupied.shape!=(8,4):raise ValueError('Actual two-site negative sea cluster required')
    psi=np.zeros(4096,complex)
    for indices in itertools.combinations(range(8),4):
        mask=sum(1<<i for i in indices)|(15<<8)
        psi[mask]=np.linalg.det(occupied[list(indices)])
    psi/=np.linalg.norm(psi)
    minus=np.zeros_like(psi);minus[(1<<8)-1]=1
    raw=model.source_bank.seed_fibre.source_vectors_at(model.chart.mean_phi,model.chart.mean_links,
        law.size*A.mean(axis=0),A,model.source_bank.profiles['source19'],BETA)
    marks=Full108ReferenceMarks(raw['full108_plus_coefficients'][ids],raw['full108_minus_coefficients'][:,ids])
    green=model.green[np.ix_(sites,sites)];ue=model.ue
    action=coulomb_packet_action(marks,green,ue,0)
    rho=np.array([[sum((mask>>(4*x+j))&1 for j in range(4))-2+
        (sum((mask>>(8+j))&1 for j in range(4))-4 if x==0 else 0) for x in range(2)] for mask in range(4096)],float)
    HC=ue*np.einsum('mi,ij,mj->m',rho,green,rho)
    body=np.zeros((12,12),complex);body[:8,:8]=h
    rows=[]
    for charge,state in ((1,psi),(-1,minus)):
        T=transfer_matrix(marks,charge);source=apply_onebody(T,state)
        left=HC*source-apply_onebody(T,HC*state)
        rhs=np.zeros_like(state);density_only=np.zeros_like(state)
        for mask in np.flatnonzero(state):
            single=np.zeros_like(state);single[mask]=state[mask]
            full=action.on_density_eigenvector(rho[mask]);rhs+=apply_onebody(transfer_matrix(full,charge),single)
            wrong=action.on_density_eigenvector(np.zeros(2));density_only+=apply_onebody(transfer_matrix(wrong,charge),single)
        packet=MeanPacketIntertwiner.onebody_action(marks,h)
        bilinear=apply_onebody(body,source)-apply_onebody(T,apply_onebody(body,state))
        expected=apply_onebody(transfer_matrix(packet,charge),state)
        rows.append(dict(charge=charge,source_raw_norm_squared=float(np.vdot(source,source).real),
            exact_Coulomb_commutator_error=float(np.linalg.norm(left-rhs)),
            wrong_density_expectation_zero_error=float(np.linalg.norm(left-density_only)),
            wrong_half_Coulomb_error=float(np.linalg.norm(left-rhs/2)),
            exact_onebody_fullFock_commutator_error=float(np.linalg.norm(bilinear-expected)),
            no_SOURCE_renormalization=True))
    return dict(original_native_sites=sites,modes=12,Fock_dimension=4096,
        original_two_site_mean_light_restriction=True,original_HC_without_half=True,
        rows=rows,scope='Independent exact SMALLFock restriction witness, not actualfull108/LEGAL state evolution; operator identities are state-independent')


def native_full_packet(model: Any) -> dict[str, Any]:
    rng=np.random.default_rng(2222044)
    A=(model.em['legs']@rng.normal(size=model.em['legs'].shape[1])*5).reshape(model.law.volume,3)
    A+=rng.normal(size=3)*5/model.law.size
    law=model.law;api=MeanPacketIntertwiner(model);first=model.em['legs'][:,0].reshape(law.volume,3)
    raw=model.source_bank.seed_fibre.source_vectors_at(model.chart.mean_phi,model.chart.mean_links,
        law.size*A.mean(axis=0),A,model.source_bank.profiles['source19'],BETA)
    marks=Full108ReferenceMarks(raw['full108_plus_coefficients'],raw['full108_minus_coefficients'])
    result={}
    for name,body in (('Hlight',api.light(A)),('Joriginal',api.light(A,first)),('contact',api.light(A,first,first))):
        size=body.shape[0];combined=np.zeros((size+4,size+4),complex);combined[:size,:size]=body
        packet=api.onebody_action(marks,body)
        result[name]={str(q):float(np.linalg.norm(combined@transfer_matrix(marks,q)-transfer_matrix(marks,q)@combined-transfer_matrix(packet,q))) for q in (1,-1)}
    return dict(full_light_modes=108,Ref=4,original_first_and_second_Peierls_derivatives=True,
        packet_matrix_commutator_errors=result,no_active_crop_or_8profile_fit=True,
        statement='CAR lifts these exact matrices to full even-Fock on proofP; full neutral RIGHT and multipair not inferred from a compressed matrix')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-npz','cache-manifest','field-capsule','C2-LEGAL','ordinary-epoch','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();start=time.monotonic();body=args.field_capsule.read_bytes()
    if hashlib.sha256(body).hexdigest()!=FIELD_SHA:raise ValueError('Exact exercised original field/aux supplier required')
    c2body=getattr(args,'C2_LEGAL').read_bytes()
    if hashlib.sha256(c2body).hexdigest()!=C2_SHA:raise ValueError('Exact original C2 state/law required')
    c2=json.loads(c2body)
    ordinarybody=args.ordinary_epoch.read_bytes()
    if hashlib.sha256(ordinarybody).hexdigest()!=ORDINARY_SHA:raise ValueError('Exact ordinary source/field epoch required')
    ordinary=json.loads(ordinarybody)
    field=json.loads(body);model,data=load_shared_native_cache(args.cache_npz,args.cache_manifest)
    manifest=json.loads(args.cache_manifest.read_bytes())
    descriptor_path=Path(manifest['argv'][manifest['argv'].index('--descriptor')+1])
    descriptor_body=descriptor_path.read_bytes()
    if hashlib.sha256(descriptor_body).hexdigest()!=DESCRIPTOR_SHA:
        raise ValueError('Exact original LEGAL SOURCE19 serialized descriptor required')
    source_descriptor=json.loads(descriptor_body)
    epoch=loaded_source_epoch()
    result=dict(reference=full_fock_reference_contract(field,c2),ordinary_reference=ordinary_reference_contract(ordinary),
        native_small_Fock=native_small_fock(model),
        original_fullFock_Coulomb_bound=full_fock_coulomb_bound(),
        native_exact_massive_mobile_symbol=native_massive_mobile(model),
        global_original_SOURCE_domain_and_bound=global_source_coframe_bound(model,source_descriptor),
        native_full108=native_full_packet(model),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        field_capsule_sha256=FIELD_SHA,cache_sha256=hashlib.sha256(args.cache_npz.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),loaded_sources=epoch,
        loaded_sources_stable=epoch==loaded_source_epoch(),program_wall_seconds=time.monotonic()-start,
        scope='New explicit controlled FULLFock reference and original packet/Coul/current identities, not single-particle invariant range or completed physical group/Born')
    text=json.dumps(portable(result),sort_keys=True);args.output.write_text(text);os.chmod(args.output,0o600);print(text)


if __name__=='__main__':main()
