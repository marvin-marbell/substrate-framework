#!/usr/bin/env python3
"""Prepared full-Gauss neutral and source-first finite-time photon transfer.

The native quantum slice is a control, not a discretization of every legal-N
coordinate. No actual-ground, stationary-spectrum or imported QED law is used.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from fractions import Fraction as F
from pathlib import Path
from typing import Any, cast

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from improved_phi_band import C, NSTAR, UE, ImprovedPhiLaw, exterior, selected_band
from bosonic_original_law_join import NativeOriginalModel, quartic_vacuum_coefficients_from_l1
from noncompact_quantitative_normal_budget import (
    ceil_log2, native_normal_budget, sqrt_upper, whole_heavy_projector_jets,
)
from noncompact_screened_gauss_dressing import frame_chart, centre_scalar
from original_charged_principal import portable
from rotation_band import RotationBand
from sea_selection import SELECTED
from unrestricted_field_minimum import BARE, local_potential

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-fullfield-closedrotation-continuation-20261006'
PLAN_REVISION = 'c0930d8a36cfce96344dd62a7fcc05c31f093e73f4145481fe92e11bfdbd87b1'
IDENTITY = 'PNFF-v2-explicit-unfiltered-fullGauss-rightaction'
BETA = np.array([1., .3j, -.2+.4j, .6])


def require(condition: bool | np.bool_, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def reject(value: float, label: str, tolerance: float = 1e-7) -> dict[str, Any]:
    require(math.isfinite(value) and value > tolerance, 'adverse did not discriminate: '+label)
    return dict(label=label, rejected=True, defect=float(value), tolerance=tolerance)


def l1(band: RotationBand) -> F:
    law = ImprovedPhiLaw(3, band)
    result = 4*abs(F.from_float(float(law.onsite)))
    for _, coefficient in law.paths:
        result += 8*sum((abs(F.from_float(float(z.real)))+abs(F.from_float(float(z.imag)))
                         for z in coefficient), F())
    return result


def neutral_descriptor() -> dict[str, Any]:
    """One immutable functional identity for both scientific suppliers.

    W here denotes the general full Haar/Kato/Gauss chart isometry, NOT the
    frozen low-band embedding with its local normal ground inserted.
    """
    return dict(identity=IDENTITY, N=NSTAR, V=NSTAR**3, time_units='unscaled microscopic H time; hbar=1, a=1; dimensionless t_in_a_over_c=c*t/a',
        C=str(C), UE=str(UE), K='C*C/(2*UE)',
        light_parameters=selected_band()[2]['parameters'], heavy_parameters=BARE,
        Lambda='UNCHANGED finite selected Lambda=t^80 from normal receipt',
        epsilon='Lambda^(-1/4)=t^(-20)',
        normal_dimension=90*NSTAR**3,
        quotient='60V Cartesian Phi +45V SU4 link -15V local SU4 Gauss; coexact and harmonic EM separate',
        normal_wave='A_D*epsilon^(-D/2)*exp(-|y/epsilon|^2/2-1/(1-|y/epsilon|^2)) for |y|<epsilon; zero otherwise',
        normal_normalization='A_D^-2=integral_{|z|<1} exp(-|z|^2-2/(1-|z|^2)) dz; positive finite',
        normal_not_local_ground=True,
        light_sea='entire 2V negative band Slater of actual improved baseline H_L(theta); PH-fixed whole Slater phase; no positive-half substitute',
        heavy_sea='entire 2V positive occupied cluster of original H_H; exact whole-cluster Kato transport, not heavy BO scalar replacement',
        coexact='exact full Fock vacuum of all 2(V-1) quadratic curl modes',
        harmonic='normalized real even Gaussian in theta=N*A_h; physical harmonic A_h in R^3; paired Bloch kappa/-kappa and every ell retained, not kappa0 ground',
        harmonic_wave='(N*N/pi)^(3/4)*exp(-N*N*|A_h|^2/2); equivalently pi^(-3/4) exp(-|theta|²/2) in theta measure',
        auxiliary='exact local-loop code preparation; finite positive Delta and nonzero total beta=1/16 remain in actual Hamiltonian',
        reference='full local Ref4 singlet; source produces actual wedge3/bar4 and all four internal holes',
        branches='all64 disjoint initial chart supports with amplitude1/8; retain ordinary E_theta; no exact superselection or branchwise energy subtraction',
        symmetry='use C_eff-real sea phase; even normal/harmonic/coexact factors and C_full W=W C_eff; if necessary select larger of Re_C xi, Im_C xi and normalize once',
        C_real_projection='Re_C xi=(xi+Cxi)/2; Im_C xi=(xi-Cxi)/(2i); norms squared sum1, selected norm>=1/sqrt2',
        full_state='xi=W(input)/||W(input)||; after optional C-real selection norm1; evolves only by actual full H_neutral',
        filtered_state_is_not_selected=True,
        source='literal whole-Haar eta18 alpha2^371 source19; quartet [1,.3i,-.2+.4i,.6] normalized once; compensated fullGauss reference',
        B_coexact_commutator='[B,a_g]=[B,a_g_dagger]=0; longitudinal Coulomb and harmonic boundary NOT discarded',
        current='J_f=d H(A+s f)/ds|0; contact C_ff=d2 H(A+s f)/ds2|0; every length2 path differentiated twice',
        spatial_packet_independent_of_neutral=True,
        quantum_law_debt=True, microscopic_parameter_debt=True,
        actual_ground_claimed=False, native_LAW01_04_roots_untouched=True)


def bump_graph_bounds(dimension: int) -> dict[str, Any]:
    """Exact normalized IBP moments; polynomial cost, no polar ratio 2^D."""
    D, n = dimension, dimension+1
    constants = {0:F(1),1:F(1),2:F(1)}
    for k in range(1,7):
        constants[k+2]=((4+2*k)*constants[k+1]+constants[k]+2*constants[k-1])/4
    root_n = math.isqrt(n)+1
    gradient_squared = F(D,2)+D*n+9*n*root_n
    hessian_squared = 6*(1+16*constants[4]*n*n+16*constants[8]*n**4
        +64*constants[6]*n**3+D+4*D*constants[4]*n*n)
    return dict(dimension=D, constants={str(k):str(v) for k,v in constants.items()},
        first_L2_log2_upper=(ceil_log2(gradient_squared)+1)//2,
        second_Frobenius_L2_log2_upper=(ceil_log2(hessian_squared)+1)//2,
        Laplacian_L2_log2_upper=(ceil_log2(D*hessian_squared)+1)//2,
        recurrence='4M[k+2]=(4+2k)M[k+1]+(D-2k-2)M[k]+2M[k-1]',
        proof='normalized divergence IBP; Mj<=Cj*(D+1)^(j/2), M0=1, M1<=sqrt(D+1), M2<=D+1; trace adds sqrt(D)')


def full_field_bounds(normal: dict[str, Any], time: F) -> dict[str, Any]:
    """Conserved-energy quartic and literal Peierls commutator bounds.

    Constants deliberately grow with V; no static occupation or independent
    normal-ground ceiling enters. Lambda is symbolic, never allocated.
    """
    require(normal['N'] == NSTAR, 'SAME selected legal N')
    budget = native_normal_budget(NSTAR)
    D, V = budget.dimension, budget.volume
    band = selected_band()[0]
    heavy_l1, light_l1 = l1(RotationBand(**BARE)), l1(band)
    a, b, cc = quartic_vacuum_coefficients_from_l1(V, F(2), heavy_l1, light_l1)
    chart = budget.chart_majorant
    jets = whole_heavy_projector_jets(budget, 2)['Kato_section_jet_upper']
    # Actual normalized inverse-cutoff moments control derivatives even though
    # the logarithmic derivative diverges at the compact support boundary.
    bump = bump_graph_bounds(D)
    derivative1_log = bump['first_L2_log2_upper']
    derivative2_log = bump['Laplacian_L2_log2_upper']
    geometric_log = ceil_log2(chart**4*(1+jets[1]+jets[2])*(105*V+1)**4)
    kinetic_log = derivative2_log+2*derivative1_log+geometric_log+8
    # Support epsilon<=1 and rho_star<=sqrt(2V)+1. Scalar multiplication
    # bounds and heavy/light all-state occupations are finite, not vacuum means.
    rho_star = sqrt_upper(F(2*V))+1
    multiplication = (1+V)*(1+rho_star)**4*(1+heavy_l1+light_l1)
    multiplication_log = ceil_log2(multiplication)+16
    selector_log = max(kinetic_log, multiplication_log)+4
    # Lambda >= 2^selector_log absorbs the bump/Kato kinetic term, beta and
    # harmonic/coexact energy. Actual selected Lambda must satisfy this NEW
    # admission; old static sourcefloor is not automatically reused.
    selected_local_lambda_log = int(normal['actual_same_N_mass_capability']['finite_Lambda']['local_log2_Lambda'])
    # Keep certificate explicit even when the receipt shape only has a symbolic
    # global selector. A selector is not a silently paid admission.
    energy_coefficient = 16*multiplication+1
    A4 = F(1, 4*V)
    # H >= kinetic+EM+(A4 Lambda/2)rho^4 - B'_N, allocating half quartic
    # to lower-bound heavy/light matter uniformly over every auxiliary sector.
    a2,b2,c2 = quartic_vacuum_coefficients_from_l1(V, F(1), heavy_l1, light_l1)
    energy_form = energy_coefficient+a2+b2+c2+1
    rho4 = 2*energy_form/A4
    rho_upper = F(2)**((ceil_log2(rho4)+3)//4)
    omega_min = 4*C/NSTAR
    modes = 2*(V-1)
    # For |nu|<=2 canonical LAW05 has [a,A]=sqrt(UE/omega).
    # The deliberately loose endpoint l1 majorant retains every path.
    current_bound = light_l1*(2*V+1)*rho_upper
    leg_squared = 16*UE/omega_min*current_bound**2
    all_mode_squared = modes*leg_squared
    cloud_squared = time*time*all_mode_squared
    target = UE/(V*2**40)
    # Photon amplitude small is not the stringent charged-response target.
    return dict(state_identity=IDENTITY, N=NSTAR, fullN_analytic=True,
        state_DH=True, state_norm=1, fullGauss=True, actual_prepared_not_ground=True,
        seed_energy_bound='E_xi <= Cseed*Lambda once new kinetic selector admitted',
        Cseed=str(energy_coefficient),
        epsilon_log_Lambda_coefficient='-1/4',
        kinetic_scaling='chi/epsilon^2=1; colour/Phi cutoff derivatives paid by actual bump/Kato graph',
        alignment_scaling='Lambda^(5/4)*epsilon^2=Lambda^(3/4)<=Lambda',
        bump_first_L2_log2_upper=derivative1_log,
        bump_Laplacian_L2_log2_upper=derivative2_log,
        bump_graph=bump,
        chart_Kato_log2_upper=geometric_log,
        required_new_log2_Lambda=selector_log,
        supplied_local_log2_Lambda=selected_local_lambda_log,
        new_selector_admitted=(selected_local_lambda_log is not None and selected_local_lambda_log>=selector_log),
        no_old_static_cap_reused=True,
        Root_BN_coefficients=[str(a),str(b),str(cc)],
        half_quartic_BN_coefficients=[str(a2),str(b2),str(c2)],
        rho4_bound=str(rho4), rho_norm_upper=str(rho_upper),
        occupation_all_state='0<=n_x<=4, total N_L=2V; charged total2V+1, Ref4 exact; NOT n_x(t)=2',
        current_norm_upper=str(current_bound), contact_norm_upper=str(2*current_bound),
        omega_min_lower=str(omega_min), coexact_mode_count=modes,
        photon_leg_squared_upper=str(leg_squared),
        allmode_photon_cloud_squared_upper=str(cloud_squared),
        allmode_photon_cloud_squared_log2_upper=ceil_log2(cloud_squared),
        photon_relative_amplitude_log2_upper=(ceil_log2(time*time*leg_squared)+1)//2,
        normal_stress='sum(PhiPi^2+.3 E4^2)<=2 Lambda^(1/2)*(E_xi+B_N); NOT ground moment',
        source_first='Hq B-B Hn=[kinetic,B]+hL*B+longitudinal Coulomb/harmonic/ref; scalar/heavy multiplication commutes B',
        source_jet_owner='actual eta18 alpha2^371 whole-Haar q0..6; new bump derivatives and neutral momentum must multiply them',
        source_first_normal_term='chi*q1*||Pi xi||+chi*q2/2; crude conserved-energy Pi gives Lambda^(1/4) q1, not a small transferred normal error',
        source_first_normal_residual_paid=False,
        exact_rightaction='Hn a_g^dagger-a_g^dagger Hn-omega_g a_g^dagger=[H_L,a_g^dagger] for monochromatic g; finite packet adds ||(omega-omega0)g||',
        no_global_neutral_quasimode_needed=True,
        resolution_bound='||U(T)a_g^dagger xi - a_{exp(-i omega T)g}^dagger U(T)xi|| <= |T| sqrt(leg_squared); allmode photon production <=T^2 allmode_squared',
        spectral_statement='finite-time nonstationary transported photon leg and positive two-time kernel; NOT a stationary ground spectral measure',
        ordinary_offsets_all64_retained=True,
        charged_response_target=str(target),
        photon_cloud_over_charged_target_log2_upper=ceil_log2(cloud_squared/target),
        long_clock_admission='T^2*allmode_squared must fit caller error; cannot inherit T6 or full2pi clocks',
        physical_highmode_derivative='quartic energy does NOT bound spatial derivatives/coarse-SO3 defect; derivative/alignment excess requires separate actual Etheta subtraction',
        full_field_massive_normal_transfer_paid=False,
        actual_equilibrium_photon_phase_paid=False,
        quantum_law_debt=True, parameter_debt=True)


def gaussian_filter_certificate(sigma: float, delta: float) -> dict[str, Any]:
    """Failed materially different energy-filter certificate, not selected state."""
    log_amplification = sigma*sigma/(2*delta*delta)
    return dict(delta=delta, sigma=sigma, log_amplitude_amplification=log_amplification,
        positive_norm_squared_lower_log=-sigma*sigma/(delta*delta),
        residual_log_upper=math.log(delta)-.5+log_amplification,
        universal_certificate_minimum=sigma, minimizing_delta=sigma,
        reason='Jensen lower Z alone cannot improve certified seed variance', selected=False)


class PreparedQuantumSlice:
    """Primitive full-root/full-heavy quantum Galerkin control.

    Full256 edge Pauli, all6 heavy two-of-four states, actual light negative
    two-of-four seed, three normal and two colour coordinate nodes, photon3,
    harmonic3. Spatial N3 onebody is compressed to one fixed uniform bulk
    mode; this is NOT the full N3 sea and NOT a full-field phase proof.
    """
    def __init__(self):
        self.band, self.frame, _ = selected_band()
        self.law = ImprovedPhiLaw(3, self.band)
        self.heavy = ImprovedPhiLaw(3, RotationBand(**BARE))
        self.model = NativeOriginalModel()
        self.rootdim = len(self.model.identity)
        self.Lambda, self.ue = 4., .0025
        self.chi = self.Lambda**-.5
        self.K = float(C)**2/(2*self.ue)
        self.normal_scale, self.colour_scale = .18, .2
        self.phomega = float(C)*math.sqrt(3)
        self.aux_penalty, self.beta = 20., .13
        self.modes = [3,2,3,3]
        self.heavy_dim = 6
        self.internal = self.heavy_dim*self.rootdim
        rng = np.random.default_rng(222100601)
        normal = rng.normal(size=(self.law.volume,4,15))
        normal = self.law.fields(normal); normal /= np.linalg.norm(normal)
        self.normal = normal
        self.fold = np.tile(np.eye(4), (self.law.volume,1))/math.sqrt(self.law.volume)
        self.f = np.zeros((self.law.volume,3))
        self.f[:,1] = np.cos(2*math.pi*self.law.sites[:,0]/3)
        self.f /= np.linalg.norm(self.f)
        self.harmonic = np.zeros_like(self.f); self.harmonic[:,1] = 1/self.law.size
        self.phscale = math.sqrt(2*self.ue/self.phomega)
        self.coordinates, self.transforms, self.p2 = [], [], []
        for n in self.modes[:3]:
            a = np.diag(np.sqrt(np.arange(1,n)),1)
            x = (a+a.T)/math.sqrt(2)
            nodes, trans = np.linalg.eigh(x)
            self.coordinates.append(nodes); self.transforms.append(trans)
            # Full p^2 compression, rather than square compressed p.
            p2 = np.diag(np.arange(n)+.5)
            for j in range(n-2): p2[j,j+2]=p2[j+2,j]=-math.sqrt((j+1)*(j+2))/2
            self.p2.append(trans.T@p2@trans)
        theta = 2*math.pi*np.arange(3)/3
        ell = np.array([-1,0,1])
        fourier = np.exp(1j*theta[:,None]*ell)/math.sqrt(3)
        self.coordinates.append(theta); self.transforms.append(fourier)
        self.p2.append((fourier@np.diag(ell*ell)@fourier.conj().T).real)
        a = np.diag(np.sqrt(np.arange(1,3)),1)
        self.annihilator = self.axis_operator(2,self.transforms[2].T@a@self.transforms[2])
        self.creator = self.annihilator.conj().T
        self.fields, self.links, self.asites = [], [], []
        self.scalar, self.root_h, self.heavy_h, self.sources, self.jroots, self.croots = [], [], [], [], [], []
        labels = np.array([np.eye(4), *(SELECTED[0]@SELECTED[j] for j in range(1,4))])
        quartet = np.einsum('a,aij->ij',BETA,labels)/2
        masks6, W6 = self.model.code_isometry(6)
        # Full root neutral seed: bulk2/ref4, not bulk0/ref4 or an occupation cap.
        h0 = self.fold.conj().T@self.law.onebody(np.tile(self.frame,(self.law.volume,1,1,1)),
            np.tile(np.eye(4),(self.law.volume,3,1,1)),np.zeros_like(self.f))@self.fold
        hh0 = self.fold.conj().T@self.heavy.onebody(np.tile(self.frame,(self.law.volume,1,1,1)),
            np.tile(np.eye(4),(self.law.volume,3,1,1)),np.zeros_like(self.f))@self.fold
        _, light_vec = np.linalg.eigh(h0); _, heavy_vec = np.linalg.eigh(hh0)
        _, heavy_masks = exterior(np.zeros((4,4)),2)
        self.hseed = np.array([np.linalg.det(heavy_vec[[i for i in range(4) if m>>i&1],2:]) for m in heavy_masks])
        coefficients = np.zeros(len(masks6),complex)
        for k, mask in enumerate(masks6):
            if mask & 240 != 240: continue
            occupied=[i for i in range(4) if mask>>i&1]
            if len(occupied)==2: coefficients[k]=np.linalg.det(light_vec[occupied,:2])
        self.lseed = W6@coefficients
        require(abs(np.linalg.norm(self.lseed)-1)<1e-10,'actual half-filled light seed')
        loop = self.model.stabilizers[0]; I = self.model.identity
        refcharge=sum((self.model.n(i) for i in range(4,8)),np.zeros_like(I))-4*I
        # Uniform folded bulk density is killed by the compensated Green form.
        p=2*math.pi*self.law.sites/self.law.size
        laplacian=4*np.sum(np.sin(p/2)**2,axis=1)
        green_rr=float(np.sum(1/laplacian[laplacian>1e-12])/self.law.volume)
        coulomb=self.ue*green_rr*refcharge@refcharge
        for z0,z1,z2,theta0 in itertools.product(*self.coordinates):
            phi=np.tile(self.frame,(self.law.volume,1,1,1))+self.normal_scale*z0*normal
            u=np.tile(np.eye(4,dtype=complex),(self.law.volume,3,1,1))
            for site in range(self.law.volume):
                u[site,0]=expm(1j*self.colour_scale*z1*self.law.generators[3])
            em=self.phscale*z2*self.f+theta0*self.harmonic
            h=self.fold.conj().T@self.law.onebody(phi,u,em)@self.fold
            hh=self.fold.conj().T@self.heavy.onebody(phi,u,np.zeros_like(em))@self.fold
            j=self.fold.conj().T@self.law.onebody(phi,u,em,self.f)@self.fold
            cmat=self.fold.conj().T@self.law.onebody(phi,u,em,self.f,self.f)@self.fold
            onsite=self.law.onsite*np.mean(phi[:,0],axis=0)
            rh=self.model.encode_even(onsite,self.model.bulk_modes)+loop@self.model.encode_even(h-onsite,self.model.bulk_modes)
            rh+=self.aux_penalty*(I-loop)/2+self.beta*self.model.defect
            rh+=coulomb
            hj,_=exterior(-self.Lambda*hh,2)
            self.root_h.append(sparse.csr_matrix(rh)); self.heavy_h.append(hj)
            self.jroots.append(sparse.csr_matrix(loop@self.model.encode_even(j,self.model.bulk_modes)))
            self.croots.append(sparse.csr_matrix(loop@self.model.encode_even(cmat,self.model.bulk_modes)))
            edges=[(x,int(self.law.target[x,k])) for x in range(self.law.volume) for k in range(3)]
            weights=[u[x,k] for x in range(self.law.volume) for k in range(3)]
            # Original scalar quartic and alignment, actual quadratic coexact curl.
            elastic, alignment=local_potential(phi,edges,weights)
            scalar=self.Lambda*float(elastic)+self.Lambda**1.25*float(alignment)+self.K/2*3*(self.phscale*z2)**2
            self.scalar.append(scalar)
            fx,fy=frame_chart(phi[0])[0],frame_chart(phi[-1])[0]
            path=u[0,0]@u[int(self.law.target[0,0]),0]
            ss=centre_scalar(fx.conj().T@path@fy,18.,self.law.generators)[0]
            mark=ss*fx@quartet@fy.conj().T
            source=np.zeros((8,8),complex);source[:4,4:]=mark
            self.sources.append(sparse.csr_matrix(self.model.encode_even(source)))
            self.fields.append(phi);self.links.append(u);self.asites.append(em)
        self.H=self.hamiltonian()
        self.J=sparse.block_diag([sparse.kron(sparse.eye(6),j) for j in self.jroots],format='csr')
        self.contact=sparse.block_diag([sparse.kron(sparse.eye(6),c) for c in self.croots],format='csr')
        self.B=sparse.block_diag([sparse.kron(sparse.eye(6),b) for b in self.sources],format='csr')
        factors=[trans[0].conj() if k<3 else trans[:,1] for k,trans in enumerate(self.transforms)]
        scalar_seed=factors[0]
        for f in factors[1:]:scalar_seed=np.kron(scalar_seed,f)
        self.seed=np.kron(scalar_seed,np.kron(self.hseed,self.lseed))
        self.seed/=np.linalg.norm(self.seed)
        self.energy=float(np.vdot(self.seed,self.H@self.seed).real)
        self.H=self.H-self.energy*sparse.eye(len(self.seed),format='csr')
        self.trace=complex(self.H.diagonal().sum())
        self.rho2=np.array([np.linalg.norm(p)**2/4 for p in self.fields])
        self.rho=sparse.kron(sparse.diags(np.sqrt(self.rho2)),sparse.eye(self.internal),format='csr')
        self.nbulk=self.axis_internal(sum((self.model.n(i) for i in range(4)),np.zeros_like(I)))
        self.nref=self.axis_internal(sum((self.model.n(i) for i in range(4,8)),np.zeros_like(I)))
        self.complement=self.axis_internal((I-loop)/2)

    def axis_internal(self, matrix: np.ndarray):
        return sparse.kron(sparse.eye(math.prod(self.modes)),sparse.kron(sparse.eye(6),sparse.csr_matrix(matrix)),format='csr')

    def axis_operator(self, axis: int, matrix: np.ndarray):
        result=sparse.csr_matrix([[1.]])
        for k,n in enumerate(self.modes):result=sparse.kron(result,sparse.csr_matrix(matrix) if k==axis else sparse.eye(n),format='csr')
        return sparse.kron(result,sparse.eye(self.internal),format='csr')

    def hamiltonian(self, external: float = 0.):
        blocks=[]
        loop=self.model.stabilizers[0]
        for phi,u,em,rh,hh,scalar in zip(self.fields,self.links,self.asites,self.root_h,self.heavy_h,self.scalar):
            root=rh
            if external:
                h0=self.fold.conj().T@self.law.onebody(phi,u,em)@self.fold
                hs=self.fold.conj().T@self.law.onebody(phi,u,em+external*self.f)@self.fold
                root=root+sparse.csr_matrix(loop@self.model.encode_even(hs-h0,self.model.bulk_modes))
            blocks.append(sparse.kron(sparse.eye(6),root)+sparse.kron(hh,sparse.eye(self.rootdim))+scalar*sparse.eye(self.internal))
        H=sparse.block_diag(blocks,format='csr')
        for k,coefficient in enumerate((self.chi/(2*self.normal_scale**2),.3*self.chi/self.colour_scale**2,self.ue/self.phscale**2,self.ue/self.law.size)):
            H+=coefficient*self.axis_operator(k,self.p2[k])
        return H

    def correlated_hessian_control(self) -> tuple[np.ndarray, dict[str,Any]]:
        """Material changed preparation using actual native well Hessian.

        It is not PNFF-v2 and is not promoted to a legalN covariance proof.
        """
        def classical(y):
            phi=np.tile(self.frame,(self.law.volume,1,1,1))+y[0]*self.normal
            u=np.tile(np.eye(4,dtype=complex),(self.law.volume,3,1,1))
            for site in range(self.law.volume):
                u[site,0]=expm(1j*y[1]*self.law.generators[3])
            em=np.zeros_like(self.f)
            h=self.fold.conj().T@self.law.onebody(phi,u,em)@self.fold
            hh=self.fold.conj().T@self.heavy.onebody(phi,u,em)@self.fold
            edges=[(x,int(self.law.target[x,k])) for x in range(self.law.volume) for k in range(3)]
            weights=[u[x,k] for x in range(self.law.volume) for k in range(3)]
            elastic,alignment=local_potential(phi,edges,weights)
            return (self.Lambda*float(elastic)+self.Lambda**1.25*float(alignment)
                -self.Lambda*float(np.linalg.eigvalsh(hh)[2:].sum())
                +float(np.linalg.eigvalsh(h)[:2].sum()))
        step=2e-4
        zero=np.zeros(2);E=classical(zero);hessian=np.zeros((2,2));gradient=np.zeros(2)
        for i in range(2):
            ei=np.eye(2)[i]*step
            gradient[i]=(classical(ei)-classical(-ei))/(2*step)
            hessian[i,i]=(classical(ei)+classical(-ei)-2*E)/(step*step)
        hessian[0,1]=hessian[1,0]=(classical(np.array([step,step]))-classical(np.array([step,-step]))
            -classical(np.array([-step,step]))+classical(np.array([-step,-step])))/(4*step*step)
        g=np.diag([self.chi,.6*self.chi]);gs=np.sqrt(g)
        vals,vec=np.linalg.eigh(gs@hessian@gs)
        require(float(vals[0])>0,'actual native Hessian positive for changed covariance')
        A=np.linalg.inv(gs)@(vec@np.diag(np.sqrt(vals))@vec.T)@np.linalg.inv(gs)
        factors=[]
        for z0,z1,z2,theta in itertools.product(*self.coordinates):
            i=int(np.argmin(abs(self.coordinates[0]-z0)))
            j=int(np.argmin(abs(self.coordinates[1]-z1)))
            k=int(np.argmin(abs(self.coordinates[2]-z2)))
            l=int(np.argmin(abs(self.coordinates[3]-theta)))
            y=np.array([self.normal_scale*z0,self.colour_scale*z1])
            weight=self.transforms[0][0,i]*self.transforms[1][0,j]
            weight*=math.exp((z0*z0+z1*z1)/2-float(y@A@y)/2)
            factors.append(weight*self.transforms[2][0,k]*self.transforms[3][l,1])
        state=np.kron(np.asarray(factors),np.kron(self.hseed,self.lseed))
        state/=np.linalg.norm(state)
        energy=float(np.vdot(state,self.H@state).real)
        return state,dict(hessian=hessian,gradient=gradient,kinetic_cometric=g,covariance_precision=A,
            covariance= np.linalg.inv(2*A),actual_H_relative_energy=energy,
            actual_H_variance=float(np.linalg.norm(self.H@state-energy*state)**2),
            classical_floor_at_template=E,template_is_not_native_stationary=bool(np.linalg.norm(gradient)>1e-7),
            lawful_global_floor_proved=False,fullN_covariance_not_selected=True,
            canonical_state_identity_unchanged=IDENTITY)


def lanczos_filter(H, vector: np.ndarray, steps: int, delta: float):
    """Finite native filter control; exact residual evaluated using actual H."""
    basis=[]; aa=[]; bb=[]; v=vector/np.linalg.norm(vector); previous=np.zeros_like(v); beta=0.
    for _ in range(steps):
        basis.append(v.copy()); w=H@v-beta*previous
        alpha=float(np.vdot(v,w).real);w-=alpha*v
        for q in basis:w-=np.vdot(q,w)*q
        beta=float(np.linalg.norm(w));aa.append(alpha)
        if beta<1e-12:break
        bb.append(beta);previous,v=v,w/beta
    tri=np.diag(aa)+np.diag(bb[:len(aa)-1],1)+np.diag(bb[:len(aa)-1],-1)
    eig,unit=np.linalg.eigh(tri)
    f=np.exp(-eig*eig/(2*delta*delta))
    coefficients=unit@(f*unit[0].conj())
    unnormalized=np.asarray(coefficients@np.asarray(basis))
    Z=float(np.linalg.norm(unnormalized));require(Z>0,'native filter nonzero')
    prepared=unnormalized/Z
    return prepared,dict(norm=Z,energy=float(np.vdot(prepared,H@prepared).real),
        residual=float(np.linalg.norm(H@prepared)),steps=len(aa),delta=delta)


def native_probe(time: float) -> dict[str, Any]:
    model=PreparedQuantumSlice();H,J,Q=model.H,model.J,model.contact
    xi=model.seed
    sigma=float(np.linalg.norm(H@xi))
    filtered,filterrow=lanczos_filter(H,xi,70,.15)
    check,filtercheck=lanczos_filter(H,xi,100,.15)
    filter_convergence=float(np.linalg.norm(filtered-check))
    # These are actual nonstationary SAME-neutral legs; never freeze the right.
    evolve=lambda mat,vec: np.asarray(expm_multiply(-1j*time*mat,vec,traceA=-1j*time*complex(mat.diagonal().sum())))
    nt=evolve(H,xi)
    correlated,covariance_row=model.correlated_hessian_control()
    ct=evolve(H,correlated)
    covariance_row['finite_time_relative_photon_error']=float(np.linalg.norm(
        evolve(H,model.creator@correlated)-np.exp(-1j*model.phomega*time)*(model.creator@ct)))
    photon=model.creator@xi
    lhs=evolve(H,photon)
    rhs=np.exp(-1j*model.phomega*time)*(model.creator@nt)
    relative=float(np.linalg.norm(lhs-rhs))
    wrong_static=float(np.linalg.norm(lhs-np.exp(-1j*model.phomega*time)*photon))
    residual=H@photon-model.creator@(H@xi)-model.phomega*photon
    # Split oscillator cutoff boundary from literal nonlinear matter term.
    Hph=model.ue/(2*model.phscale**2)*model.axis_operator(2,model.p2[2])
    values=model.coordinates[2]
    Hph+=model.K*3*model.phscale**2*model.axis_operator(2,np.diag(values*values))
    cutoff=Hph@photon-model.creator@(Hph@xi)-model.phomega*photon
    matter_leg=residual-cutoff
    number_before=float(np.linalg.norm(model.annihilator@xi)**2)
    number_after=float(np.linalg.norm(model.annihilator@nt)**2)
    source=model.B@xi;source_norm=float(np.linalg.norm(source));require(source_norm>1e-6,'actual global Ref4 source norm')
    source/=source_norm
    st=evolve(H,source)
    source_first=H@(model.B@xi)-model.B@(H@xi)
    source_boundary_norm=float(np.linalg.norm(source_first))
    delta=2e-4
    Hp=model.hamiltonian(delta)-model.energy*sparse.eye(len(xi),format='csr')
    Hm=model.hamiltonian(-delta)-model.energy*sparse.eye(len(xi),format='csr')
    jerror=float(np.linalg.norm(((Hp-Hm)@source)/(2*delta)-J@source))
    cerror=float(np.linalg.norm(((Hp+Hm-2*H)@source)/(delta*delta)-Q@source))
    require(jerror<2e-8 and cerror<2e-5,'same-H current/contact finite differences')
    # Actual first/second static-drive response via augmented Duhamel system.
    zero=sparse.csr_matrix(H.shape)
    generator=sparse.bmat([[H,zero,zero],[J,H,zero],[Q,2*J,H]],format='csr')
    stacked=np.concatenate([source,np.zeros_like(source),np.zeros_like(source)])
    response=evolve(generator,stacked).reshape(3,-1)
    fp=evolve(Hp,source);fm=evolve(Hm,source)
    r1=float(np.linalg.norm((fp-fm)/(2*delta)-response[1]))
    r2=float(np.linalg.norm((fp+fm-2*response[0])/(delta*delta)-response[2]))
    require(r1<2e-6 and r2<2e-4,'actual driven response matches same J/contact')
    no_contact=sparse.bmat([[H,zero,zero],[J,H,zero],[zero,2*J,H]],format='csr')
    wrong_response=evolve(no_contact,stacked).reshape(3,-1)
    # Photon spectral diagnostic: actual finite-time transported support,
    # positive Gram Fourier window. No stationary ground spectral label.
    times=np.linspace(0,time,17)
    legs=cast(Any,expm_multiply)(-1j*H,photon,start=0.,stop=time,num=len(times),traceA=-1j*model.trace)
    correlations=np.asarray(legs)@photon.conj()
    spectral=[]
    for frequency in (0.,model.phomega,2*model.phomega):
        weights=np.exp(1j*frequency*times)*np.hanning(len(times))
        weights/=np.sum(weights)
        spectral.append(float(np.linalg.norm(weights@np.asarray(legs))**2))
    occ_before=float(np.vdot(xi,model.nbulk@xi).real)
    occ_after=float(np.vdot(nt,model.nbulk@nt).real)
    colour_occ=model.axis_internal(model.model.n(0))
    colour_change=float(abs(np.vdot(nt,colour_occ@nt)-np.vdot(xi,colour_occ@xi)))
    rho_before=float(np.linalg.norm(model.rho@xi));rho_after=float(np.linalg.norm(model.rho@nt))
    kinetic_normal=model.axis_operator(0,model.p2[0])/model.normal_scale**2
    normal_momentum=float(np.vdot(nt,kinetic_normal@nt).real)
    alignment_norms=[]
    for phi,u in zip(model.fields,model.links):
        alignment_norms.append(sum(np.linalg.norm(phi[x]-u[x,k]@phi[int(model.law.target[x,k])]@u[x,k].conj().T)**2/8
            for x in range(model.law.volume) for k in range(3)))
    probabilities=np.sum(abs(nt.reshape(math.prod(model.modes),model.internal))**2,axis=1)
    alignment_moment=float(probabilities@np.asarray(alignment_norms))
    missing_normal=H-model.chi/(2*model.normal_scale**2)*model.axis_operator(0,model.p2[0])
    wrong_normal=float(np.linalg.norm(evolve(missing_normal,source)-st))
    wrong_occupation=1.-float(np.vdot(xi,colour_occ@xi).real)
    return dict(state_identity=IDENTITY,native_scope=PreparedQuantumSlice.__doc__,
        dimension=len(xi),full_root_dimension=model.rootdim,quantum_heavy_dimension=6,
        coordinates=dict(normal=3,colour=2,coexact=3,harmonic=3),
        microscopic_native_parameters=dict(Lambda=model.Lambda,chi=model.chi,UE=model.ue,K=model.K,c=float(C),
            normal_scale=model.normal_scale,colour_scale=model.colour_scale,aux_penalty=model.aux_penalty,beta=model.beta),
        native_UE_not_selected_UE=True,native_fullGauss_continuum_proof=False,
        native_sea='actual negative2/4 light Slater and positive2/4 heavy Slater of fixed uniform spatial compression; full6 heavy quantum Hilbert, primitive256 Pauli root and staticRef4',
        normalized_once=True,seed_norm=float(np.linalg.norm(xi)),source_norm_before_normalization=source_norm,
        seed_energy=model.energy,seed_actual_H_variance=sigma*sigma,
        unfiltered_prepared_selected=True,
        changed_correlated_Hessian_preparation_control=covariance_row,
        gaussian_filter_control=filterrow,gaussian_filter_convergence=filter_convergence,
        gaussian_filter_100=filtercheck,jensen_failed_certificate=gaussian_filter_certificate(sigma,.15),
        time=time,evolution_norm_defect=abs(float(np.linalg.norm(nt))-1),
        exact_neutral_rightaction_relative_photon_error=relative,
        photon_relative_commutator_norm=float(np.linalg.norm(residual)),
        photon_cutoff_boundary_norm=float(np.linalg.norm(cutoff)),
        actual_matter_photon_leg_norm=float(np.linalg.norm(matter_leg)),
        photon_number_before=number_before,photon_number_after=number_after,
        finite_window_frequencies=[0.,model.phomega,2*model.phomega],
        positive_finite_window_photon_weights=spectral,
        photon_two_time_correlation=correlations,
        nonstationary_finite_resolution_not_ground_spectrum=True,
        source_first_boundary_norm=source_boundary_norm,
        current_derivative_error=jerror,contact_derivative_error=cerror,
        actual_current_norm=float(np.linalg.norm(J@st)),actual_contact_norm=float(np.linalg.norm(Q@st)),
        first_driven_response_norm=float(np.linalg.norm(response[1])),second_driven_response_norm=float(np.linalg.norm(response[2])),
        first_response_difference_error=r1,second_response_difference_error=r2,
        bulk_occupation_before=occ_before,bulk_occupation_after=occ_after,
        reference_occupation_after=float(np.vdot(st,model.nref@st).real),
        actual_colour_occupation_change=colour_change,
        rho_norm_before=rho_before,rho_norm_after=rho_after,
        actual_normal_momentum_squared=normal_momentum,actual_covariant_alignment_moment=alignment_moment,
        actual_aux_complement_norm=float(np.linalg.norm(model.complement@nt)),
        adverse=dict(frozen_neutral_rightaction=reject(wrong_static,'frozen stationary neutral instead of actual rightaction'),
            missing_normal_kinetic=reject(wrong_normal,'omitted quantum normal field'),
            static_filled_colour_occupation=reject(abs(wrong_occupation),'replace evolving CAR occupation by occupied colour1'),
            missing_current=reject(float(np.linalg.norm(response[1])),'erase actual current'),
            missing_contact=reject(float(np.linalg.norm(wrong_response[2]-response[2])),'erase actual contact from shared response')))


def source_bindings() -> dict[str,str]:
    names=['prepared_neutral_full_field.py','prepared_neutral_full_field.md','improved_phi_band.py',
        'bosonic_original_law_join.py','noncompact_quantitative_normal_budget.py',
        'noncompact_screened_gauss_dressing.py','rotation_band.py','sea_selection.py']
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def run(normal_path: Path,time: float) -> dict[str,Any]:
    hashes=source_bindings()
    raw=normal_path.read_bytes();normal=json.loads(raw)
    return dict(plan=PLAN,plan_revision=PLAN_REVISION,work_item='prepared-neutral-full-field',
        state=neutral_descriptor(),fullN=full_field_bounds(normal,F.from_float(time)),
        native=native_probe(time),inputs={str(normal_path):hashlib.sha256(raw).hexdigest()},
        source_sha256=hashes,source_unchanged=(hashes==source_bindings()),
        conclusion='Explicit prepared neutral and actual finite-time rightaction photon/shared-response route; equilibrium phase and full normal/coarse spatial transfer remain unpaid',
        frozen_suppliers_rerun=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal-receipt',type=Path,required=True)
    parser.add_argument('--time',type=float,default=1.)
    args=parser.parse_args()
    require(args.time>0,'positive finite native flight time')
    print(json.dumps(portable(run(args.normal_receipt,args.time)),indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
