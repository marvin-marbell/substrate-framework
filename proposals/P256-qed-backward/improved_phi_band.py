#!/usr/bin/env python3
"""Original-heavy, primitive range-two light Phi law and physical band supplier."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from fractions import Fraction as F
from pathlib import Path
from typing import Any

if __name__ == '__main__':
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from flint import arb, ctx

from rotation_band import RotationBand
from sea_selection import SELECTED, matrix_unit_basis, reframe
from local_orbital_stability import scalar_paths
from original_charged_principal import portable
from unrestricted_field_minimum import BARE, local_potential
from bosonic_original_law_join import (
    NativeOriginalModel, quartic_vacuum_coefficients, gauss_trial_vacuum_upper_contract,
)
from noncompact_screened_gauss_dressing import (
    frame_chart, centre_scalar, orientation_core_halfspin_bound,
    physical_orientation_lie_majorant, physical_raw_noether_geometry_majorant,
    relative_photon_noether_graph,
)
from noncompact_quantitative_normal_budget import sqrt_lower, sqrt_upper, ceil_log2

ROOT = Path(__file__).resolve().parent
NSTAR = 5 * 2**96
C = F(1130202988249777, 9007199254740992)
UE = F.from_float(1.5429442570866347e-148) / 2**4096
TRIAL_G0 = .4187253908699568
TRIAL_GS = .0632221974274214
CALIBRATED_V = .4990360130634525
LIGHT_PARAMETERS: dict[str, Any] = dict(a=1., m=.2, r=.12, v=CALIBRATED_V, e=1., improved=True)
GRAPH_LOG2 = 66048
GAP_LOG2 = 132096


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def accept(error: float, label: str, tolerance: float = 3e-7) -> float:
    require(math.isfinite(error) and error <= tolerance, f'{label}: {error} > {tolerance}')
    return float(error)


def reject(error: float, label: str, tolerance: float = 3e-7) -> dict[str, Any]:
    require(math.isfinite(error) and error > tolerance, f'adverse control not discriminating: {label}')
    return dict(rejected=True, defect=float(error), acceptance_tolerance=tolerance)


def exact(value: Any) -> F:
    if isinstance(value, dict):
        return F(int(value['exact_numerator_hex'], 16), int(value['exact_denominator_hex'], 16))
    return F(value)


def selected_band(metric_payload: dict[str, Any] | None = None,
                  size: int = NSTAR) -> tuple[RotationBand, np.ndarray, dict[str, Any]]:
    """Executed trial; exact physical calibration is c/sqrt(G_infinity,11).

    The metric belongs to the ORIGINAL heavy/scalar minimizer. It is recovered
    from anticommutators of the actual Phi Hamiltonian, never source whitening.
    Finite-N and binary64 calibration errors remain in the returned certificate.
    """
    require(size >= 25, 'legal metric onset N>=25')
    band = RotationBand(**LIGHT_PARAMETERS)
    G = np.diag([TRIAL_G0, TRIAL_GS, TRIAL_GS, TRIAL_GS])
    frame = np.sqrt(np.diag(G))[:, None, None] * SELECTED
    result: dict[str, Any] = dict(parameters=LIGHT_PARAMETERS,
        heavy_parameters=BARE, G=G, mass=band.m*math.sqrt(TRIAL_G0),
        charged_cone=band.v*math.sqrt(TRIAL_GS), photon_c=float(C),
        exact_v_definition='c/sqrt((G_infinity_heavy)11)',
        metric_owner='original heavy/scalar convex full10 Gram, NOT source Gram')
    if metric_payload is not None:
        cert = metric_payload['certificate']
        require(cert['admission'] is True and cert['parameters']['improved'] is False,
                'actual original heavy metric required')
        require(cert['trial_decimal'] == [str(TRIAL_G0), str(TRIAL_GS)], 'authoritative heavy trial')
        require(cert['parameters']['a'] == 1 and cert['parameters']['kappa'] == 2,
                'actual heavy/scalar scale')
        require('analytic_strip' in cert and 'full_ten_gradient' in cert, 'full10 remainder proof')
        ctx.prec = 192
        residual = arb(str(cert['full_ten_gradient']['Frobenius_residual_upper']))
        remainder = arb(str(cert['continuum_quadrature_error_upper']))
        sigma = arb(str(cert['analytic_strip']['sigma']))
        majorant = arb(str(cert['analytic_strip']['matrix_gradient_integrand_HS_majorant']))
        # Explicit positive cap; never floating underflow masquerading as zero.
        require(sigma*size > arb(1024)*arb(2).log(), 'selected legalN analytic alias cap')
        alias = arb(2)**-1024
        alias_sum = 2*alias*(3+alias**2)/(1-alias)**3
        distance = (residual+remainder+majorant*alias_sum)/2
        d = F(str(float(distance.upper())))*F(1000000000001, 1000000000000)
        g0, gs = F(str(TRIAL_G0)), F(str(TRIAL_GS))
        schur = g0-d-d*d/(2*(gs-d))
        mass = F.from_float(band.m)*sqrt_lower(schur)
        cone_error = F.from_float(band.v)*d/(sqrt_lower(gs-d)+sqrt_lower(gs)) + F(1, 2**52)
        frame_error = d/(2*sqrt_lower(gs-d))
        result.update(N=size, all_real_shifts=True, all64_centres=True,
            full10_Frobenius_distance_upper=str(d), Schur_lower=str(schur),
            legal_mass_gap_lower=str(mass), legal_mass_gap_lower_float=float(mass),
            continuum_cone_calibration_debt_upper=str(cone_error),
            continuum_cone_calibration_debt_upper_float=float(cone_error),
            finiteN_alias_positive_upper='2^-1024',
            frame_distance_upper=str(frame_error),
            calibration_self_consistency='H_H/scalar leading law unchanged; light O(1) perturbation is paid by normal selector',
            exact_cone_statement='ideal v_star gives c exactly at G_infinity; executed v_hat and allrealshift legalN incur displayed debt')
    return band, frame, result


def physical_h(p: np.ndarray, frame: np.ndarray, band: RotationBand | None = None) -> np.ndarray:
    band = RotationBand(**LIGHT_PARAMETERS) if band is None else band
    return band.mass(p)*frame[0]+np.einsum('j,jik->ik', band.v*np.sin(p), frame[1:])


class ImprovedPhiLaw:
    """Literal full Cartesian Phi endpoints, SU4 path products and real U1 links."""
    def __init__(self, size: int = 3, band: RotationBand | None = None):
        require(size >= 3, 'periodic native N>=3')
        self.size = size
        self.band = RotationBand(**LIGHT_PARAMETERS) if band is None else band
        self.sites = np.asarray(list(itertools.product(range(size), repeat=3)))
        self.volume = len(self.sites)
        self.index = {tuple(site): k for k, site in enumerate(self.sites)}
        self.target = np.asarray([[self.index[tuple((site+step)%size)]
            for step in np.eye(3, dtype=int)] for site in self.sites],dtype=np.int64)
        self.basis = matrix_unit_basis()
        self.generators = self.basis/math.sqrt(8)
        self.onsite, self.paths = scalar_paths(self.band)

    def fields(self, values: np.ndarray) -> np.ndarray:
        return np.einsum('xma,aij->xmij', np.asarray(values).reshape(self.volume, 4, 15), self.basis)

    def terms(self, fields: np.ndarray, links: np.ndarray, a: np.ndarray):
        for x in range(self.volume):
            for displacement, scalar in self.paths:
                axis = int(np.flatnonzero(displacement)[0])
                length = int(displacement[axis])
                y, path, u, phase = x, [], np.eye(4, dtype=complex), 0.
                for _ in range(length):
                    path.append((y, axis))
                    u = u@links[y, axis]
                    phase += a[y, axis]
                    y = int(self.target[y, axis])
                fx = np.einsum('m,mij->ij', scalar, fields[x])
                fy = np.einsum('m,mij->ij', scalar, fields[y])
                yield x, y, (fx@u+u@fy)/2*np.exp(1j*phase), tuple(path)

    def onebody(self, fields: np.ndarray, links: np.ndarray, a: np.ndarray,
                first: np.ndarray | None = None, second: np.ndarray | None = None) -> np.ndarray:
        h = np.zeros((4*self.volume, 4*self.volume), complex)
        if first is None:
            for x in range(self.volume):
                h[4*x:4*x+4, 4*x:4*x+4] = self.onsite*fields[x, 0]
        for x, y, coefficient, path in self.terms(fields, links, a):
            if first is not None:
                coefficient = coefficient*(1j*sum(first[e] for e in path))
                if second is not None:
                    coefficient = coefficient*(1j*sum(second[e] for e in path))
            sx, sy = slice(4*x, 4*x+4), slice(4*y, 4*y+4)
            h[sx, sy] += coefficient
            h[sy, sx] += coefficient.conj().T
        return h


def primitive_native(frame: np.ndarray, band: RotationBand) -> dict[str, Any]:
    law = ImprovedPhiLaw(3, band)
    rng = np.random.default_rng(22210062)
    fields = np.tile(frame, (law.volume, 1, 1, 1))+law.fields(.003*rng.normal(size=(law.volume, 4, 15)))
    links = np.asarray([[expm(.03j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators))
                         for _ in range(3)] for _ in range(law.volume)])
    a = .04*rng.normal(size=(law.volume, 3))
    first, second = rng.normal(size=a.shape), rng.normal(size=a.shape)
    h = law.onebody(fields, links, a)
    current = law.onebody(fields, links, a, first)
    contact = law.onebody(fields, links, a, first, second)
    eps = 2e-5
    fd = (law.onebody(fields, links, a+eps*first)-law.onebody(fields, links, a-eps*first))/(2*eps)
    f2 = (law.onebody(fields, links, a+eps*second, first)-law.onebody(fields, links, a-eps*second, first))/(2*eps)
    e1 = accept(float(np.linalg.norm(fd-current)), 'all one/two-link current insertions', 2e-7)
    e2 = accept(float(np.linalg.norm(f2-contact)), 'all cross contact insertions', 3e-7)
    gauge = np.asarray([expm(.1j*np.einsum('a,aij->ij', rng.normal(size=15), law.generators)) for _ in range(law.volume)])
    chi = rng.normal(size=law.volume)
    moved = np.einsum('xij,xmjk,xlk->xmil', gauge, fields, gauge.conj())
    ul = np.asarray([[gauge[x]@links[x,j]@gauge[law.target[x,j]].conj().T for j in range(3)] for x in range(law.volume)])
    al = a+chi[:,None]-chi[law.target]
    q = sparse.block_diag([sparse.csr_matrix(np.exp(1j*chi[x])*gauge[x]) for x in range(law.volume)]).toarray()
    covariance = accept(float(np.linalg.norm(law.onebody(moved, ul, al)-q@h@q.conj().T)), 'full Phi SU4/U1 independent endpoint covariance')
    scaled = accept(float(np.linalg.norm(law.onebody(32*fields, links, a)-32*h)), 'unbounded fullPhi linear homogeneity')
    frozen = law.onebody(np.tile(frame, (law.volume,1,1,1)), links, a)
    near = ImprovedPhiLaw(3, RotationBand(**{**LIGHT_PARAMETERS, 'improved':False}))
    wrong_j = near.onebody(fields, links, a, first)
    gamma5=SELECTED[0]@SELECTED[1]@SELECTED[2]@SELECTED[3]
    sigma_fields=-np.einsum('ij,xmjk,kl->xmil',gamma5,fields,gamma5)
    sigma_links=np.einsum('ij,xajk,kl->xail',gamma5,links,gamma5)
    S=np.kron(np.eye(law.volume),gamma5)
    ph_error=accept(float(np.linalg.norm(law.onebody(sigma_fields,sigma_links,a)+S@h@S)),
        'changed fullPhi sigma/PH with all real photon A unchanged')
    uniform=np.tile(frame,(law.volume,1,1,1))
    flat=np.tile(np.eye(4),(law.volume,3,1,1)).astype(complex)
    wave=np.tile(np.eye(4),(law.volume,1))/math.sqrt(law.volume)
    momentum=np.array([.003,-.002,.001])
    twist=np.tile(momentum,(law.volume,1))
    primitive_bloch=wave.conj().T@law.onebody(uniform,flat,twist)@wave
    bloch_error=accept(float(np.linalg.norm(primitive_bloch-physical_h(momentum,frame,band))),
        'actual primitive Fourier dispersion and kinetic metric')
    rest_j=np.zeros((law.volume,3));rest_j[:,0]=1
    primitive_rest_current=wave.conj().T@law.onebody(uniform,flat,np.zeros_like(twist),rest_j)@wave
    measured_current=float(np.linalg.norm(primitive_rest_current,ord=2))
    accept(abs(measured_current-float(C)),'actual primitive current shared cone',2e-14)
    # The source is an actual field-dependent full-Haar polar-chart observable.
    source = []
    for x in (0, 1, 2):
        y = int(law.target[x,0]); fx = frame_chart(fields[x])[0]; fy = frame_chart(fields[y])[0]
        scalar = centre_scalar(fx.conj().T@links[x,0]@fy,18.,law.generators)[0]
        source.append(scalar*fx@fy.conj().T)
    source_change = float(np.linalg.norm(np.asarray(source)-np.eye(4)))
    return dict(native_N=3, fullCartesian_directions=60*law.volume, native_onebody_dimension=len(h),
        current_derivative_error=e1, contact_derivative_error=e2,
        SU4_U1_covariance_error=covariance, Phi_scaling_error=scaled,
        fullPhi_PH_sigma_error=ph_error,primitive_Bloch_dispersion_error=bloch_error,
        measured_physical_current_cone=measured_current,
        all_path_lengths=[1,2], source_field_dependence_measured=source_change,
        adverse=dict(frozen_Gamma=reject(float(np.linalg.norm(h-frozen)),'frozen endpoint Phi'),
            nearest_only_current=reject(float(np.linalg.norm(current-wrong_j)),'omitted range2 current'),
            source_metric_as_kinetic=reject(abs(band.v-float(C)),'source identity Gram versus physical Hamiltonian'),
            old_N9_scale=reject(abs(.41402766991208007*math.sqrt(TRIAL_GS)-float(C)),'N9 improved-heavy proxy scale')),
        scope='actual N3 full60V Phi and independent SU4/U1 native matrix; not huge legalN allocation')


def exterior(matrix: np.ndarray, number: int) -> tuple[sparse.csr_matrix, tuple[int,...]]:
    n = len(matrix)
    masks = tuple(sum(1<<i for i in row) for row in itertools.combinations(range(n), number))
    lookup = {m:k for k,m in enumerate(masks)}
    rr, cc, vv = [], [], []
    for column, mask in enumerate(masks):
        for source in range(n):
            if not mask>>source&1:
                continue
            rest = mask^(1<<source)
            sign = (-1)**((mask&((1<<source)-1)).bit_count())
            for target in range(n):
                if rest>>target&1 or matrix[target,source] == 0:
                    continue
                rr.append(lookup[rest|(1<<target)]); cc.append(column)
                vv.append(sign*(-1)**((rest&((1<<target)-1)).bit_count())*matrix[target,source])
    return sparse.csr_matrix((vv,(rr,cc)),shape=(len(masks),len(masks))), masks


def quantum_native(frame: np.ndarray, band: RotationBand) -> dict[str, Any]:
    """Actual three-site range2 CAR/source flight with quantum Phi/SU4/A slice.

    All charged seven-of-twelve occupations and all four STATIC reference
    columns are retained. The heavy Born--Oppenheimer potential uses the original
    heavy six-of-twelve minimum; this finite slice is not a microscopic ground.
    """
    rng = np.random.default_rng(22210663)
    basis = matrix_unit_basis(); generators = basis/math.sqrt(8)
    normal = np.einsum('xma,aij->xmij', rng.normal(size=(3,4,15)), basis)
    normal /= np.linalg.norm(normal)
    fields0 = np.tile(frame,(3,1,1,1))+.003*normal
    x = np.array([[0.,1.],[1.,0.]])/math.sqrt(2)
    nodes, modes = np.linalg.eigh(x)
    # Full p^2 compression, not (compressed p)^2.
    p2 = modes.T@np.diag([.5,1.5])@modes
    scales = (.02,.04,.07)
    ue = .0025; K = float(C)**2/(2*ue); chi = .004
    labels = np.asarray([np.eye(4), *(SELECTED[j]@SELECTED[k] for j,k in ((2,3),(3,1),(1,2)))])/2
    beta = np.asarray([1.,.3j,-.2+.4j,.6]); beta /= np.linalg.norm(beta)
    blocks = []; currents = []; contacts = []; source_columns = []
    onebody_blocks = []; onebody_currents = []; onebody_contacts = []
    source_marks = []; onsite_blocks = []; scalar_blocks = []
    masks7: tuple[int,...] = ()
    heavy = RotationBand(**BARE)
    def ring_h(phi, u, a, b, derivative=0):
        onsite, paths = scalar_paths(b)
        h = np.zeros((12,12),complex)
        if derivative == 0:
            for k in range(3): h[4*k:4*k+4,4*k:4*k+4] += onsite*phi[k,0]
        for site in range(3):
            for displacement, scalar in paths:
                # Fold y/z Bloch momenta zero; x remains three actual propagation sites.
                axis = int(np.flatnonzero(displacement)[0]); length = int(displacement[axis])
                target = (site+length)%3 if axis == 0 else site
                path = [(site+t)%3 for t in range(length)] if axis == 0 else []
                product = np.eye(4,dtype=complex)
                for t in path: product = product@u[t]
                fx = np.einsum('m,mij->ij',scalar,phi[site]); fy = np.einsum('m,mij->ij',scalar,phi[target])
                # A_y(x) is a transverse coexact profile in the folded y/z
                # Bloch slice; x colour transport and both y-Peierls links stay distinct.
                em_weight=length*a[site] if axis==1 else 0.
                coefficient = (fx@product+product@fy)/2*np.exp(1j*em_weight)
                if derivative:
                    coefficient *= (1j*length*(1.,-.4,-.6)[site] if axis==1 else 0.)**derivative
                sx,sy=slice(4*site,4*site+4),slice(4*target,4*target+4)
                h[sx,sy] += coefficient; h[sy,sx] += coefficient.conj().T
        return h
    for z0,z1,z2 in itertools.product(nodes, repeat=3):
        phi = fields0+scales[0]*z0*normal
        u = np.asarray([expm(1j*(.03*(k+1)+scales[1]*z1*(k+1))*generators[3]) for k in range(3)])
        a = scales[2]*z2*np.asarray([1.,-.4,-.6])
        h = ring_h(phi,u,a,band)
        hm, masks7 = exterior(h,7)
        jmat=ring_h(phi,u,a,band,1); cmat=ring_h(phi,u,a,band,2)
        j,_ = exterior(jmat,7)
        contact,_ = exterior(cmat,7)
        onebody_blocks.append(h); onebody_currents.append(jmat); onebody_contacts.append(cmat)
        hh = ring_h(phi,u,np.zeros(3),heavy)
        evals,ev = np.linalg.eigh(hh); sea = ev[:,:6]
        # Actual full six-particle Slater state, not positive-half source projection.
        _,masks6 = exterior(np.zeros_like(h),6)
        neutral = np.asarray([np.linalg.det(sea[[i for i in range(12) if mask>>i&1]]) for mask in masks6])
        fx,fy = frame_chart(phi[0])[0],frame_chart(phi[2])[0]
        product = u[0]@u[1]
        ss = centre_scalar(fx.conj().T@product@fy,18.,generators)[0]
        mark = ss*np.einsum('a,aij->ij',beta,fx@labels@fy.conj().T)
        source_marks.append(mark)
        onsite_blocks.append((band.m+4.5*band.r)*np.mean(phi[:,0],axis=0))
        profile = np.asarray([1.,.4j,-.2]); profile/=np.linalg.norm(profile)
        source = np.concatenate([profile[k]*mark for k in range(3)])/math.sqrt(2)
        out = np.zeros((len(masks7),4),complex); lookup7={mask:k for k,mask in enumerate(masks7)}
        for col,mask in enumerate(masks6):
            for mode in range(12):
                if mask>>mode&1: continue
                out[lookup7[mask|(1<<mode)]] += (-1)**((mask&((1<<mode)-1)).bit_count())*source[mode]*neutral[col]
        source_columns.append(out)
        scalar = sum(local_potential(phi,[(0,1),(1,2),(2,0)],list(u)))-float(evals[6:].sum())
        # A_y(x) transverse curl, full compensated Coulomb, kappa0 harmonic fibre.
        scalar += K*sum((a-np.roll(a,1))**2)
        scalar_blocks.append(scalar)
        occ=np.asarray([[sum(bool(mask>>i&1) for i in range(4*k,4*k+4)) for k in range(3)] for mask in masks7])
        density=occ-2-np.array([0,0,1])
        green=(np.eye(3)-np.ones((3,3))/3)/3
        coulomb=ue*np.einsum('xi,ij,xj->x',density,green,density)
        blocks.append(hm+sparse.diags(coulomb)+scalar*sparse.eye(len(masks7)))
        currents.append(j); contacts.append(contact)
    d = len(masks7)
    H = sparse.block_diag(blocks,format='csr')
    ident2=sparse.eye(2); identd=sparse.eye(d)
    for axis,coefficient in enumerate((chi/(2*scales[0]**2),.3*chi/(2*scales[1]**2),ue/(2*scales[2]**2))):
        factors=[sparse.csr_matrix(p2) if k==axis else ident2 for k in range(3)]
        kinetic=sparse.kron(sparse.kron(factors[0],factors[1]),factors[2])
        H += coefficient*sparse.kron(kinetic,identd,format='csr')
    source=np.concatenate(source_columns)/math.sqrt(8)
    gram=source.conj().T@source
    require(float(np.linalg.eigvalsh(gram)[0])>0,'actual four-reference quantum source metric')
    source/=np.linalg.norm(source)
    U=expm_multiply(-6j*H,source)
    accept(abs(float(np.linalg.norm(U))-1),'quantum native propagation norm',2e-8)
    J=sparse.block_diag(currents,format='csr'); Q=sparse.block_diag(contacts,format='csr')
    current_norm=float(np.linalg.norm(J@U)); contact_norm=float(np.linalg.norm(Q@U))
    occupation=np.asarray([[sum(bool(mask>>i&1) for i in range(4*k,4*k+4)) for mask in masks7] for k in range(3)])
    before=source.reshape(8,d,4); after=U.reshape(8,d,4)
    n0=np.einsum('nd,kd->k',np.sum(abs(before)**2,axis=(0,2))[None,:],occupation)
    n6=np.einsum('nd,kd->k',np.sum(abs(after)**2,axis=(0,2))[None,:],occupation)
    occupation_change=float(np.linalg.norm(n6-n0))
    require(occupation_change>1e-6,'actual native source flight changes physical occupation')
    # Dynamical primitive Pauli consumer of the same actual improved endpoint
    # coefficients. Spatial sites fold to the bulk block; ref4 remains STATIC.
    model=NativeOriginalModel(); masks,W=model.code_isometry(4)
    loop=model.stabilizers[0]; I=model.identity
    pauli_blocks=[]; pauli_j=[]; pauli_c=[]; pauli_source=[]
    fold=np.tile(np.eye(4),(3,1))/math.sqrt(3)
    ref_vector=W[:,masks.index(sum(1<<i for i in range(4,8)))]
    for h,jmat,cmat,mark_node,onsite_node,scalar in zip(onebody_blocks,onebody_currents,onebody_contacts,source_marks,onsite_blocks,scalar_blocks):
        probe=np.zeros((8,8),complex); probe[:4,:4]=fold.conj().T@h@fold
        onsite_probe=np.zeros_like(probe); onsite_probe[:4,:4]=onsite_node
        jp=np.zeros_like(probe); jp[:4,:4]=fold.conj().T@jmat@fold
        cp=np.zeros_like(probe); cp[:4,:4]=fold.conj().T@cmat@fold
        encoded=model.encode_even(onsite_probe)+loop@model.encode_even(probe-onsite_probe)
        pauli_blocks.append(sparse.csr_matrix(encoded+scalar*I+20*(I-loop)/2+.13*model.defect))
        pauli_j.append(sparse.csr_matrix(loop@model.encode_even(jp)))
        pauli_c.append(sparse.csr_matrix(loop@model.encode_even(cp)))
        columns=[]
        for ref in range(4):
            source_matrix=np.zeros_like(probe); source_matrix[:4,4+ref]=mark_node[:,ref]
            columns.append(model.encode_even(source_matrix)@ref_vector)
        pauli_source.append(np.column_stack(columns))
    HP=sparse.block_diag(pauli_blocks,format='csr')
    for axis,coefficient in enumerate((chi/(2*scales[0]**2),.3*chi/(2*scales[1]**2),ue/(2*scales[2]**2))):
        factors=[sparse.csr_matrix(p2) if k==axis else ident2 for k in range(3)]
        kinetic=sparse.kron(sparse.kron(factors[0],factors[1]),factors[2])
        HP+=coefficient*sparse.kron(kinetic,sparse.eye(len(I)),format='csr')
    sp=np.concatenate(pauli_source)
    require(float(np.linalg.eigvalsh(sp.conj().T@sp)[0])>0,'primitive Pauli static fullRef4 source')
    sp/=np.linalg.norm(sp)
    fp=expm_multiply(-6j*HP,sp)
    LP=sparse.kron(sparse.eye(8),sparse.csr_matrix((I-loop)/2))
    leakage=float(np.linalg.norm(LP@fp))
    require(leakage>1e-8,'finite nonzero defect actually populates auxiliary complement')
    pauli_current=float(np.linalg.norm(sparse.block_diag(pauli_j,format='csr')@fp))
    pauli_contact=float(np.linalg.norm(sparse.block_diag(pauli_c,format='csr')@fp))
    root_error=accept(abs(float(np.linalg.norm(fp))-1),'primitive Pauli quantum propagation norm')
    return dict(scope='three spatial x sites, y/z zero-Bloch folded; quantum fullPhi/SU4-Cartan/real-coexact-A Galerkin slice, original heavy BO potential, all7/12 charged CAR, STATICref4; not huge ground simulation',
        Hilbert_dimension=len(blocks)*d, charged_CAR_dimension=d, source_columns=4,
        source_Gram=gram, source_Gram_is_not_kinetic_metric=True,
        time=6, norm_defect=abs(float(np.linalg.norm(U))-1), current_state_norm=current_norm,
        contact_state_norm=contact_norm, occupation_before=n0, occupation_after=n6,
        occupation_change=occupation_change, Pauli_quantum_norm_error=root_error,
        Pauli_quantum_dimension=len(pauli_blocks)*len(I),Pauli_auxiliary_complement_norm=leakage,
        Pauli_actual_current_norm=pauli_current,Pauli_actual_contact_norm=pauli_contact,Pauli_nonzero_defect=.13,
        witness_parameters=dict(UE=ue,K=K,chi=chi,physical_photon_c=float(C)),
        selected_UE_not_numerically_replaced=True,
        parameter_scope='finite witness uses stated representable UE/chi; selected law and legal bounds use exact tiny UE and finite Lambda')


def spacetime(frame: np.ndarray, band: RotationBand, calibration: dict[str, Any]) -> dict[str, Any]:
    g0,gs=TRIAL_G0,TRIAL_GS
    mass=band.m*math.sqrt(g0); cone=band.v*math.sqrt(gs)
    rows=[]
    for radius in (.2,.1,.05):
        hbound=cone*radius**3/6+math.sqrt(g0)*band.r*radius**4/4
        jbound=cone*radius**2/2+math.sqrt(g0)*band.r*radius**3
        eh=ej=eu=ew=0.
        for direction in (np.array([1.,0.,0.]),np.ones(3)/math.sqrt(3),np.array([2.,-1.,1.])/math.sqrt(6)):
            p=radius*direction
            h=physical_h(p,frame,band)
            continuum=band.m*frame[0]+np.einsum('j,jik->ik',band.v*p,frame[1:])
            eh=max(eh,float(np.linalg.norm(h-continuum,ord=2)))
            j=np.asarray([reframe(row,frame) for row in band.current(p,p)])
            ej=max(ej,max(float(np.linalg.norm(x-y,ord=2)) for x,y in zip(j,band.v*frame[1:])))
            eu=max(eu,float(np.linalg.norm(expm(-6j*h)-expm(-6j*continuum),ord=2)))
            q=np.array([.011,-.007,.005])
            vertex=np.asarray([reframe(x,frame) for x in band.current(p+q,p-q)])
            ew=max(ew,float(np.linalg.norm(physical_h(p+q,frame,band)-physical_h(p-q,frame,band)-np.einsum('j,jik->ik',2*np.sin(q),vertex))))
        accept(max(0.,eh-hbound),'primitive physical Dirac remainder')
        accept(max(0.,ej-jbound),'same physical current remainder')
        accept(max(0.,eu-6*hbound),'T6 physical propagation remainder')
        accept(ew,'range2 exact Ward identity')
        rows.append(dict(cutoff=radius,H_error=eh,H_upper=hbound,current_error=ej,current_upper=jbound,T6_error=eu,T6_upper=6*hbound,Ward_error=ew))
    # Authoritative kinetic metric and current measured from actual primitive Bloch matrix.
    p=np.array([.003,0.,0.]); E=float(np.linalg.eigvalsh(physical_h(p,frame,band))[-1])
    measured=math.sqrt((E*E-mass*mass)/(p@p))
    j0=reframe(band.current(np.zeros(3),np.zeros(3))[0],frame)
    measured_current=float(np.linalg.norm(j0,ord=2))
    accept(abs(measured_current-float(C)),'actual rest current cone',2e-14)
    return dict(mass=mass,actual_current_cone=measured_current,small_p_dispersion_cone=measured,
        photon_c_exact=str(C),physical_kinetic_metric=np.diag([g0,gs,gs,gs]),
        kinetic_metric_definition='Tr{Phi_mu,Phi_nu}/8; h(Phi)^2=x_mu G_mu_nu x_nu I at selected heavy Clifford field',
        source_propagation='native source applies entire CAR creation before same primitive H flight; source Gram affects normalization only',
        rest_g=2.,g_scope='O(pi^2), (sigma.pi)^2=pi^2+charge sigma.B; W2 starts fourth order and cannot renormalize quadratic Pauli coefficient',
        gyromagnetic_parameter_debt='v_star calibrated on actual heavy G_infinity, not predicted; at finite momentum lattice/current/normal/calibration remainders apply',
        cutoff_rows=rows,legal_calibration=calibration,
        Compton_hard_momentum_outside_continuum_cutoff=True, exact_Lorentz_all_momenta=False)


def changed_bounds(normal: dict[str, Any], graph: dict[str, Any], band: RotationBand) -> dict[str, Any]:
    require(normal['N']==graph['N']==NSTAR,'SAME selectedN normal/graph')
    require(exact(normal['exact_family']['UE'])==exact(graph['actual_family']['UE'])==UE,'SAME exact UE')
    require(exact(graph['actual_family']['c'])==C,'SAME exact photon c')
    V=NSTAR**3
    # New light row majorant; old Cx=11/10 was not automatically reusable.
    onsite,paths=scalar_paths(band)
    coefficients=[]
    for displacement,scalar in paths:
        length=int(np.sum(abs(displacement)))
        coefficients.append((length,sum((abs(F.from_float(float(z.real)))+abs(F.from_float(float(z.imag))) for z in scalar),F())))
    row=F.from_float(float(onsite))+2*sum((v for _,v in coefficients),F())
    cap=F(16,5)
    require(row<cap and cap<4*F(11,10),'new primitive row coefficient domination')
    jets=[F.from_float(float(onsite))*(q==0)+2*sum((v*length**q for length,v in coefficients),F()) for q in range(7)]
    require(all(jets[q]<4*2**q*F(11,10) for q in range(7)),'actual q0..q6 path insertion domination')
    constants=normal['exact_normal_constants']
    A4=exact(normal['full_Phi_and_photon_graph_capability']['A4'])
    old_cd=exact(normal['full_Phi_and_photon_graph_capability']['CD'])
    # Fresh full-domain Young proof: quartic >=Lambda*A4*rho^4,
    # heavy >=-Lambda*BH*rho and new light >=-BL*rho. For Lambda>=1,
    # radius^3>=2(BH+BL)/A4 implies positive half-quartic outside radius.
    BL=2*(4*V+1)*cap
    BH=exact(normal['full_Phi_and_photon_graph_capability']['BH'])
    fresh_slope=BH+BL
    fresh_radius=F(2**max(0,(ceil_log2(2*fresh_slope/A4)+2)//3))
    added_cd=BL**2/A4+BL**2+1
    cd=old_cd+added_cd
    require(fresh_radius**3>=2*fresh_slope/A4 and cd>=fresh_slope*fresh_radius,
        'renewed Young baseline retains actual half-quartic coefficient')
    rho4=2*(cd+3)/A4
    rho_bits=(ceil_log2(rho4)+3)//4
    current=32*(2*V+1)*cap
    # Renew the actual changed-neutral quartic/pullthrough graphs. Coefficient
    # domination alone does not compare states from different Hamiltonians.
    crho=sqrt_upper(sqrt_upper(2*(cd+3)/A4))
    cq=sqrt_upper(sqrt_upper(2/A4))
    sharp_trial=64*V+UE*V*NSTAR**2/4+3*UE/(2*V)
    photon_shift=sharp_trial+1
    omega_min=C/NSTAR
    mode_weight=sqrt_upper(F(2*V)/omega_min)
    j_ground=current*crho
    un=mode_weight*j_ground/omega_min
    uqhalf=mode_weight*j_ground/(2*sqrt_lower(omega_min))
    uq=mode_weight*j_ground
    urho=crho*un+cq*sqrt_upper(uq*un)
    urho2=sqrt_upper(2/A4*(un*uq+(cd+3)*un*un))
    urho4=2/A4*(uq+(cd+3)*un+urho)
    urho3=sqrt_upper(urho2*urho4)
    uphhalf=sqrt_upper(un*uq+photon_shift*un*un)
    uphfull=uq+(sharp_trial+1)*un+BL*urho
    uchi_pi=sqrt_upper(2*(un*uq+(sharp_trial+8*exact(constants['C0'])+2)*un*un))
    renewed=dict(n=un,Q_half=uqhalf,Q_full=uq,rho=urho,rho2=urho2,
        rho3=urho3,rho4=urho4,current=current*urho,chiPi=uchi_pi,
        positive_photon_half=uphhalf,positive_photon_full=uphfull)
    graph_new={key:ceil_log2(value) for key,value in renewed.items()}
    source_half_log=ceil_log2(16*uphhalf+2**427*(NSTAR+1)*(un+urho+1))
    angular1=427+graph_new['rho']+1027+16
    angular2=850+graph_new['rho2']+2*1027+32
    mixed=850+graph_new['rho4']+2*ceil_log2(current)+2*1027+64
    require(max(max(graph_new.values()),source_half_log,angular1,angular2,mixed)+27868+2856+2547+64 < GRAPH_LOG2,
        'renewed changed-neutral fullPhi/source/current graph finite envelope')
    # Original selected global phase Lambda is retained as a dependency; increase
    # t by 2^256 and exterior ell by256. All unchanged positive heavy powers
    # then improve, while every changed light derivative/BL coefficient is paid.
    choice=normal['time6_operator_capability']['global_log2_Lambda_expression']
    common={**choice,'local_log2_t':int(choice['local_log2_t'])+256,'exterior_log2_t':int(choice['exterior_log2_t'])+256}
    loglambda=80*common['local_log2_t']
    require(loglambda>180375+512 and loglambda//4>GAP_LOG2+ceil_log2(BL)+64,'changed normal/source gap reserve')
    laplacian=F(2**20)*(105*V)**2*(1+2+cap)**2
    require(8*ceil_log2(laplacian)<5*loglambda,'actual changed squared-form quartic graph')
    hvac=exact(constants['Hvac'])
    require(8*ceil_log2(hvac)<loglambda,'renewed chiPi actual vacuum term paid to one')
    # Full exterior counting and path jets enlarge coefficients <=2^9. Raising t by
    # 2^256 reduces the slowest kinetic residual t^-5 by2^-1280, so every
    # original coefficient-weighted residual is still an upper bound here.
    residual=exact(normal['time6_operator_capability']['residual_upper'])
    require(residual<UE/(V*2**40),'actual retained nonzero normal residual')
    a,b,c=quartic_vacuum_coefficients(V,band)
    # Root coefficient formula with changed band bounds BOTH heavy/light; using
    # larger changed coefficients for heavy is conservative, not a heavy-law change.
    G=F(2**GAP_LOG2); beta=F(1,16)
    debt=F(2**GRAPH_LOG2)*(8*beta/F(2**(3*GAP_LOG2//4))+(4*beta+6*beta*beta)/G)
    target=UE/(V*2**40)
    require(debt<target,'absolute changed-law source/current/Noether graph budget')
    core=orientation_core_halfspin_bound(NSTAR//2+2,18.,.025,2.,.001)
    require(core['alpha_expression']=='2^371','unchanged full-Haar source alpha')
    lie=physical_orientation_lie_majorant(core,2.,.25,2.)
    geometry=physical_raw_noether_geometry_majorant(64.,1.)
    logue=math.log2(UE.numerator)-math.log2(UE.denominator)
    boundary=relative_photon_noether_graph(logue,float(C),1.,64.,1.,
        float(NSTAR**2),.25,coexact_singular_lower=4/float(NSTAR))
    boundary['relative_photon_Noether_prefactor_before_prepared_graph']=boundary['relative_photon_Noether_coefficient_before_sqrt_UE']
    boundary['relative_photon_Noether_prefactor_before_prepared_graph_log2']=boundary['relative_photon_Noether_coefficient_before_sqrt_UE_log2']
    boundary['relative_photon_Noether_prefactor_dyadic_upper_exponent']=boundary['relative_photon_Noether_dyadic_upper_exponent']
    boundary['relative_photon_Noether_coefficient_before_sqrt_UE_log2']+=source_half_log
    boundary['relative_photon_Noether_coefficient_before_sqrt_UE']=None
    boundary['relative_photon_Noether_upper_log2']+=source_half_log
    boundary['relative_photon_Noether_dyadic_upper_exponent']=math.ceil(boundary['relative_photon_Noether_upper_log2'])
    boundary['coefficient_scope']='prepared-state coefficient; logarithm authoritative when binary64 coefficient is unrepresentable'
    boundary['prefactor_scope']='before prepared Hph-half graph; prefactor dyadic exponent includes sqrtUE but not that graph'
    boundary['actual_prepared_source_photon_half_log2_upper']=source_half_log
    boundary['bound_uses_renewed_changed_Q_graph_not_original_state']=True
    s_second=-9/math.cosh(36)**2
    return dict(light_row_exact=str(row),light_row_upper=str(cap),old_row_upper='11/10',
        derivative_q0_q6_exact=[str(x) for x in jets],derivative_domination='Cq_new <= 4*2^q C0_old; path endpoint fields stay linear',
        full_CAR_light_rho_coefficient=str(BL),current_rho_coefficient=str(current),
        exterior_C0=str(8*exact(constants['C0'])),exterior_C1=str(2*BL/A4),
        A4_unchanged=str(A4),CD_changed=str(cd),rho_fourth_form_root_log2_upper=rho_bits,
        renewed_quartic=dict(Crho=str(crho),CQ=str(cq),
            rho4_form='rho4 <=(2/A4)[Lambda^-1 Q+(CDnew+3)n^2]',
            squared_form_Laplacian_majorant=str(laplacian),chiPi_vacuum_paid_to_one=True,
            fresh_heavy_light_slope=str(fresh_slope),fresh_Young_radius=str(fresh_radius),
            CD_dominates_fresh_slope_times_radius=True),
        renewed_actual_neutral=dict(sharp_trial_upper=str(sharp_trial),photon_form_shift_upper=str(photon_shift),
            current_ground_norm_upper=str(j_ground),omega_min_lower=str(omega_min),
            actual_changed_Q='H_new,neutral-E_new >=0; NOT original Omega/Q',
            U_definition='-sum gk/sqrtomega (Qnew+omega)^-1 Jnew,k Omega_new',
            uniform_all_2V_complex_unitL2_modes=True,
            graph_values={key:str(value) for key,value in renewed.items()},
            source_angular_log2_upper=[angular1,angular2,mixed]),
        nonzero_normal_residual_upper=str(residual),
        physical_source_Noether=dict(core=core,full_Haar_field_source_Lie=lie,
            primitive_vector_geometry=geometry,actual_scalar_EM_coexact_orbital_boundary=boundary,
            source19_orbital_coefficient_upper=192,static_far_reference_retained=True,
            statement='intrinsic physical halfspin core plus actual orbital/ref/EM/normal/loop terms; not totalJ eigenvalue',
            s_eta_second_curvature=s_second),
        changed_U_graph_log2_upper=graph_new,finite_combined_graph_log2=GRAPH_LOG2,
        common_Lambda=dict(expression='2^(80*(k_new+ell_new+8*2^(2^height)))',parameters=common,
            local_log2_Lambda=loglambda,original_phase_leading_law_unchanged=True,
            changed_only_light_constants=True,normal_complement_gap_lower_log2=GAP_LOG2,
            original_exact_Etheta_offsets_retained=True,all64=True),
        Rootpenalty=dict(vacuum_lower='-B_N(Lambda)',B_N='a_N Lambda+b_N+c_N/Lambda',a_N=str(a),b_N=str(b),c_N=str(c),
            G=f'max(2^{GAP_LOG2},B_N(Lambda)+U_trial(Lambda)+window+beta+1)',
            Delta='B_N(Lambda)+U_trial(Lambda)+window+beta+G',beta='1/16',local_nonzero_Z_defect=f'1/(96*{V})',
            actual_code_vacuum_upper=gauss_trial_vacuum_upper_contract(),
            trial_extension='Gaussian onsite Phi mean zero; all one/two-link hopping expectations zero for legalN>=3; all15 Gauss and static-reference trial unchanged',
            minimum_G_log2=GAP_LOG2,fullPhi_vacuum_advantage_paid=True),
        absolute_weighted_debt_log2_upper=ceil_log2(debt),debt_over_UE_V_2power40_log2_upper=ceil_log2(debt/target),
        source=dict(scattering_sites=19,eta=18,alpha='2^371',tree_word=NSTAR//2+2,
            q6_upper=[4,427,850,1274,1698,2122,2547],allRef4=True,
            RAW_profile_separate=dict(nativeN=41,screen_eta=18,coordinate_radius=20,count_upper=68921,tree_word=NSTAR//2+60,CY='2^64'),
            photon_after_B_half_graph_log2_upper=source_half_log,charged_norm_lower='1/sqrt8',
            source_Gram_never_kinetic_metric=True),
        domains=dict(Phi='entire R^(60V), full quartic form and squared-form integration by parts',
            colour='SU4^(3V), all15V Gauss, onsite four-mode Pauli trees',
            light='entire fixed-number raw CAR and opposite static reference; auxiliary sectors not projected away',
            PH='fullPhi parity+C, kappa0 or explicit kappa/-kappa paired histories; timed B not assumed even',
            physical_Noether='S_core + actual orbital + static ref/scalar-EM boundary + paid normal/loop remainder; no arbitrary-source totalJ=1/2',
            photon='real quadratic curl and Coulomb UE rho G rho and harmonic UE/V p^2; coexact raw B commutes exactly'),
        scalar_second_curvature='s_eta_second(0)=-(eta/2)*sech(2eta)^2; eta18 retained',
        bounds_reused='heavy/scalar quartic/gauge/normal chart/Kato PH/all64 exact offsets unchanged',
        bounds_recomputed='light row/exterior BL/current/contact q0..q6, Young CD/rho, graph reserve, commonLambda light error and finite-defect Root vacuum advantage',
        no_original_probability_copied=True)


def run(metric_path: Path, normal_path: Path, graph_path: Path) -> dict[str, Any]:
    started=time.perf_counter()
    inputs={name:path.read_bytes() for name,path in [('metric',metric_path),('normal',normal_path),('graph',graph_path)]}
    metric,normal,graph=(json.loads(inputs[name]) for name in ('metric','normal','graph'))
    band,frame,calibration=selected_band(metric,NSTAR)
    native=primitive_native(frame,band)
    quantum=quantum_native(frame,band)
    space=spacetime(frame,band,calibration)
    bounds=changed_bounds(normal,graph,band)
    require(calibration['legal_mass_gap_lower_float']>.1294,'legalN actual free mass gap')
    # W2 is the positive operator sum (1-(S_j+S_jdagger)/2)^2 for
    # arbitrary real U1 link histories. Hence the original temporal coercivity,
    # determinant and dense-range photon threshold apply with this same mu;
    # there is NO weak-photon gap loss. Transfer loses the actual residuals
    # plus charged-complement Schur and nonzero auxiliary-defect energies.
    mass_lower=F(calibration['legal_mass_gap_lower'])
    residual=F(bounds['nonzero_normal_residual_upper'])
    trial=F(bounds['renewed_actual_neutral']['sharp_trial_upper'])
    normal_gap=F(2**GAP_LOG2)
    require(normal_gap>trial+1,'actual charged Schur denominator')
    loss=2*residual+residual**2/(normal_gap-trial)+F(1,256)/normal_gap
    exact_charged=mass_lower-loss
    scale=10**12
    charged=F((exact_charged.numerator*scale)//exact_charged.denominator,scale)
    require(charged>F(129,1000) and charged<=exact_charged,'outward positive compensated microscopic charge gap')
    sources=['improved_phi_band.py','improved_phi_band.md','rotation_band.py','sea_selection.py',
        'local_orbital_stability.py','original_charged_principal.py','unrestricted_field_minimum.py',
        'bosonic_original_law_join.py','noncompact_screened_gauss_dressing.py','noncompact_quantitative_normal_budget.py']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    return dict(topic='improved primitive fullPhi legal-volume physical band',owner='ImprovedMicroscopicBand',
        achievement='new field-dependent one/two-link light Hamiltonian, full current/contact and quantum source/native propagation, original heavy phase retained',
        parameters=LIGHT_PARAMETERS,heavy_parameters_unchanged=BARE,N=NSTAR,V=NSTAR**3,UE=str(UE),c=str(C),
        calibration=calibration,primitive_native=native,quantum_native=quantum,spacetime=space,bounds=bounds,
        charged_gap=dict(effective_free_mass_lower=str(mass_lower),weak_photon_loss=0,
            normal_and_auxiliary_loss_log2_upper=ceil_log2(loss),
            microscopic_compensated_gap_lower=str(charged),microscopic_compensated_gap_lower_float=float(charged),both_charge_signs=True,
            gap_scope='infimum difference with compensated static reference, full photon and raw light sectors; no isolated charged pole'),
        input_sha256={name:hashlib.sha256(content).hexdigest() for name,content in inputs.items()},
        source_sha256=hashes,elapsed_seconds=time.perf_counter()-started,all_checks_pass=True,
        limits=['Full heavy/scalar phase and generic normal/Friedrichs analytic arguments are consumed owner dependencies, not repeated huge numerical ground verification.',
            'The exact calibrated continuum cone is a selected parameter relation; executed trial has the displayed nonzero interval/rounding debt.',
            'Native quantum witness is an explicitly stated finite Galerkin slice, not full huge Cartesian ground or exact Lorentz at hard Compton momenta.',
            'Compton probability and source-current flight are supplied separately by ImprovedPhysicalCompton, never copied from original branch.'])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric',type=Path,required=True)
    parser.add_argument('--normal',type=Path,required=True)
    parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=portable(run(args.metric,args.normal,args.graph))
    text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')


if __name__ == '__main__':
    main()
