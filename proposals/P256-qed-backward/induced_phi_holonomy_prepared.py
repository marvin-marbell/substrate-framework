#!/usr/bin/env python3
"""NEW preferred222 finiteN6 canonical quantum local-well execution.

Primitive formula bodies are copied explicitly from frozen7ef, not monkeypatched.
All fundamental paths, slow Hessian and native contacts use actual seam-twisted
momenta; every well/Berry/Agmon/graph/current price is recomputed on that mean.
The old000 packet and receipts remain byte-identical and separately scoped.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import mpmath as mp
from scipy.linalg import null_space
from induced_phi_backreaction import (PhiConfig,models,SELECTED,periodic_grid,
    scalar_paths,matrix_unit_basis,coefficients,elastic_hessian,
    clifford_cartesian_basis,natural_gauss_chart,canonical_geometry_report,
    colour_link_casimir_control,noncompact_EM_Gauss_control,
    flat_cut_ramp_support_control,joint_gauge_descent_control,orbit_categories,
    fast_zero_point_interval,global_centre_projection)
from induced_phi_centre_holonomy import (SOURCE_BYTES,coefficient_rows,
    data_for,functional,fast_pressure,vector,seam_control,native_band_control)
ROOT=Path(__file__).resolve().parent
IMPORT_BYTES={**SOURCE_BYTES,Path(__file__).name:Path(__file__).read_bytes()}
IMPORT_HASHES={name:hashlib.sha256(data).hexdigest() for name,data in IMPORT_BYTES.items()}
SOURCE_NAMES=tuple(IMPORT_BYTES)
GOVERNING_PLAN=dict(id='issue-222-induced-photon-delta13-continuation-20261008',
    revision='d13d7d28ebfd7cb8386e7442897c596481aeab95cd55def1a8b3a60c1e89f4dd',
    work_item='joined-explanatory-campaign')


def coupled_mean(cfg: PhiConfig,size: int,exponent: int,holonomy: Any) -> dict[str,Any]:
    if tuple(holonomy['preferred_twist'])!=(2,2,2):
        raise ValueError('This distinct construction is the selected222 well, not an extrapolated selector')
    if holonomy['input_descriptor']['size']!=size or holonomy['input_descriptor']['Lambda_exponent']!=exponent:
        raise ValueError('Selected-sector box/hierarchy mismatch')
    microscopic={key:value for key,value in asdict(cfg).items() if key!='Lambda'}
    if microscopic!=holonomy['input_descriptor']['config']:
        raise ValueError('Selected-sector microscopic parameters mismatch')
    precision=max(320,math.ceil(exponent*math.log10(2))+120)
    with mp.workdps(precision):
        seed=mp.matrix(holonomy['preferred_quantum_mean']['G_decimal'])
        g0=seed[0,0]; gs=sum(seed[i,i] for i in range(1,4))/3
        data=data_for(cfg,size,(2,2,2),exponent)
        scale=mp.mpf(2)**exponent; ratio=scale**(-mp.mpf(5)/8)
        selector=mp.zeros(10,2); selector[0,0]=1
        for index in (4,7,9):selector[index,1]=1
        tolerance=mp.mpf(10)**(-precision+40)
        for _ in range(20):
            literal=mp.diag([g0,gs,gs,gs])
            _,gradient,Hessian=functional(vector(literal),data,2/mp.mpf(cfg.a))
            zpe,pressure=fast_pressure(literal,size)
            full=gradient+ratio*vector(pressure)
            residual=mp.norm(full)
            if residual<tolerance:break
            step=mp.lu_solve(selector.T*Hessian*selector,selector.T*full)
            g0-=step[0];gs-=step[1]
            if min(g0,gs)<=0:raise ArithmeticError('Actual coupled cubic continuation left positive Gram')
        else:raise ArithmeticError('Actual coupled cubic mean did not converge')
        return dict(branch='NEW actual222 allthree-profile/canonicalfast-ZPE mean; exact cube symmetry imposed and fullten gradient executed',
            precision=precision,size=size,V=size**3,twist=[2,2,2],Lambda_exponent=exponent,Lambda_exact=str(2**exponent),
            g0_decimal=mp.nstr(g0,precision),gs_decimal=mp.nstr(gs,precision),
            G_float=np.diag([float(g0),float(gs),float(gs),float(gs)]).tolist(),
            full_normalized_uniform_gradient_decimal=mp.nstr(residual,precision),
            fast_zero_point_coefficient_decimal=mp.nstr(zpe,precision),
            fast_zero_point_gradient_decimal=[mp.nstr(pressure[0,0],precision),mp.nstr(sum(pressure[i,i] for i in range(1,4)),precision)],
            literal_cubic_centre_Gram_difference_from_allten_mean_decimal=mp.nstr(mp.norm(literal-seed),precision),
            physical_centre='Literal decimal g0/gs defines exactly diagonal positive Clifford Gram; tiny symmetry-rounding difference from original allten centre is displayed, not silently clamped. Fullten stationary error and shape price retained.',
            profiles=[[1,cfg.heavy_m,'neutral heavy minusLambda'],[1,cfg.active_m,'active charged'],[cfg.auxiliary_count,cfg.auxiliary_m,'charged auxiliary']],
            fundamental_momenta='2pi(n+1/2)/(N a); physicalSU4 links are -I only at N-1 seams, I elsewhere',source_at_import=IMPORT_HASHES)


def make_data(cfg: PhiConfig,size: int,mean: Any,heavy_only: bool) -> dict[str,Any]:
    sites=periodic_grid(size); gram=np.asarray(mean['G_float'])
    frame=np.sqrt(np.diag(gram))[:,None,None]*SELECTED
    momenta=2*np.pi*(sites+np.asarray(mean['twist'])/4)/(size*cfg.a)
    parts=[]
    for load,band,role in models(cfg):
        if heavy_only and not role.startswith('neutral'):continue
        parts.append((1. if heavy_only else load,band,coefficients(band,momenta),role))
    return dict(size=size,sites=sites,frame=frame,G=gram,basis=matrix_unit_basis(),parts=parts,twist=mean['twist'])



def literal_colour_current_contact(cfg: PhiConfig, size: int, mean: Any) -> dict[str,Any]:
    """Actual SU4 seam group curves vs analytic one/two-link current/contact."""
    data=make_data(cfg,size,mean,False)
    sites=data['sites']; volume=size**3; frame=data['frame']
    generator=matrix_unit_basis()[7]/np.sqrt(8)
    values,vectors=np.linalg.eigh(generator)
    qtheta=2*np.pi*np.asarray((1,2,1))/size
    ptheta=2*np.pi*(np.asarray((1,2,3))+np.asarray(mean['twist'])/4)/size
    wave=np.exp(1j*sites@ptheta)[:,None,None]*np.eye(4)[None]/np.sqrt(volume)
    cosine=np.cos(sites@qtheta)
    eps=1e-3
    links={}
    for step in (0.,eps,-eps,eps/2,-eps/2):
        group=(vectors[None]*np.exp(1j*step*cosine[:,None]*values[None])[:,None,:])@vectors.conj().T
        links[step]=np.stack([group*np.where(sites[:,axis]==size-1,-1.,1.)[:,None,None] for axis in range(3)],axis=1)
    rows=[]
    for _,model,role in models(cfg):
        onsite,paths=scalar_paths(model)
        def actual(step):
            output=float(onsite)*(frame[0]@wave)
            for displacement,scalars in paths:
                axis=int(np.flatnonzero(displacement)[0]); length=int(displacement[axis])
                direction=np.eye(3,dtype=int)[axis]
                F=np.einsum('m,mij->ij',scalars,frame)
                forward=np.broadcast_to(np.eye(4),(volume,4,4)).copy().astype(complex)
                reverse=forward.copy()
                for j in range(length):
                    fi=np.ravel_multi_index(((sites+j*direction)%size).T,(size,)*3)
                    ri=np.ravel_multi_index(((sites-displacement+j*direction)%size).T,(size,)*3)
                    forward=forward@links[step][fi,axis]
                    reverse=reverse@links[step][ri,axis]
                reverse=reverse.conj().transpose(0,2,1)
                target=np.ravel_multi_index(((sites+displacement)%size).T,(size,)*3)
                source=np.ravel_multi_index(((sites-displacement)%size).T,(size,)*3)
                output+=(F@forward+forward@F)@wave[target]/2
                output+=(F.conj().T@reverse+reverse@F.conj().T)@wave[source]/2
            return output
        j_expected=np.zeros_like(wave); jj_expected=np.zeros_like(wave)
        j_wrong=np.zeros_like(wave); jj_wrong=np.zeros_like(wave)
        for displacement,scalars in paths:
            axis=int(np.flatnonzero(displacement)[0]); length=int(displacement[axis])
            direction=np.eye(3,dtype=int)[axis]
            F=np.einsum('m,mij->ij',scalars,frame)
            forward_cos=[np.cos((sites+j*direction)@qtheta) for j in range(length)]
            reverse_cos=[np.cos((sites-displacement+j*direction)@qtheta) for j in range(length)]
            sf,sr=np.sum(np.asarray(forward_cos),axis=0),np.sum(np.asarray(reverse_cos),axis=0)
            phase=np.exp(1j*ptheta@displacement)
            fp=1j*(F@generator+generator@F)/2
            rp=-1j*(F.conj().T@generator+generator@F.conj().T)/2
            fpp=-(F@generator@generator+generator@generator@F)/2
            rpp=-(F.conj().T@generator@generator+generator@generator@F.conj().T)/2
            j_expected+=(sf[:,None,None]*phase*fp+sr[:,None,None]*phase.conjugate()*rp)@wave
            jj_expected+=(sf[:,None,None]**2*phase*fpp+sr[:,None,None]**2*phase.conjugate()*rpp)@wave
            j_wrong+=(sf[:,None,None]*phase*1j*(F@generator)-sr[:,None,None]*phase.conjugate()*1j*(generator@F.conj().T))@wave
            jj_wrong+=(np.sum(np.asarray(forward_cos)**2,axis=0)[:,None,None]*phase*fpp+np.sum(np.asarray(reverse_cos)**2,axis=0)[:,None,None]*phase.conjugate()*rpp)@wave
        zero,plus,minus,half_plus,half_minus=[actual(step) for step in (0.,eps,-eps,eps/2,-eps/2)]
        j=(4*(half_plus-half_minus)/eps-(plus-minus)/(2*eps))/3
        jj=(4*(half_plus+half_minus-2*zero)/(eps/2)**2-(plus+minus-2*zero)/eps**2)/3
        j_error=float(np.linalg.norm(j-j_expected)/(1+np.linalg.norm(j_expected)))
        jj_error=float(np.linalg.norm(jj-jj_expected)/(1+np.linalg.norm(jj_expected)))
        rows.append(dict(role=role,current_relative_error=j_error,contact_relative_error=jj_error,
            wrong_ordered_current_relative_error=float(np.linalg.norm(j-j_wrong)/(1+np.linalg.norm(j))),
            wrong_omitted_two_link_cross_contact_relative_error=float(np.linalg.norm(jj-jj_wrong)/(1+np.linalg.norm(jj)))))
        if j_error>1e-7 or jj_error>1e-5:
            raise ArithmeticError('Actual literal seam group curve disagrees with current/contact')
    if max(row['wrong_omitted_two_link_cross_contact_relative_error'] for row in rows)<.01:
        raise ArithmeticError('Wrong two-link contact control did not discriminate')
    return dict(rows=rows,step=eps,richardson='Independent actual exp(i epsilon cos(q.x)Ta) link products on literal -I seams; all216sites and4orbital columns; centered first/second derivatives with halfstep Richardson',
        scope='Double-precision diagnostic image of exact centre only. Physical packet uses the separate literal737digit centre, never this rounded diagnostic centre.')


def twisted_native_block(data, cfg: PhiConfig, q: np.ndarray) -> dict[str, Any]:
    """Actual 60 Phi + 45 SU4 link static block, including both contacts.

    Link coordinates sit at their source vertex. This is NOT an EM photon
    replacement: generators act on the same four orbital colours as Phi.
    """
    size = data['size']
    sites = data['sites']
    volume = len(sites)
    theta = 2 * np.pi * (sites + np.asarray(data['twist']) / 4) / size
    qtheta = 2 * np.pi * np.asarray(q) / size
    target = np.ravel_multi_index(((sites + q) % size).T, (size,) * 3)
    frame, basis = (data['frame'], data['basis'])
    generators = basis / np.sqrt(8)
    orbit = np.asarray([1j * (t @ frame - frame @ t) for t in generators])
    orbit_columns = np.einsum('tmij,aji->mat', orbit, basis).real.reshape(60, 15) / 4
    gauge = np.zeros((105, 15), complex)
    gauge[:60] = orbit_columns
    align_vertex = np.zeros((180, 105), complex)
    for axis in range(3):
        difference = 1 - np.exp(1j * qtheta[axis])
        gauge[60 + 15 * axis:75 + 15 * axis] = difference * np.eye(15)
        align_vertex[60 * axis:60 * (axis + 1), :60] = difference * np.eye(60)
        align_vertex[60 * axis:60 * (axis + 1), 60 + 15 * axis:75 + 15 * axis] = -orbit_columns
    curl = np.zeros((45, 105), complex)
    for face, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        curl[face * 15:(face + 1) * 15, 60 + i * 15:75 + i * 15] = (1 - np.exp(1j * qtheta[j])) * np.eye(15)
        curl[face * 15:(face + 1) * 15, 60 + j * 15:75 + j * 15] = (np.exp(1j * qtheta[i]) - 1) * np.eye(15)
    alignment = cfg.alignment * (align_vertex.conj().T @ align_vertex)
    colour = 0.7 * cfg.Lambda ** 1.25 / 2 * (curl.conj().T @ curl)
    bosonic = alignment + colour
    bosonic[:60, :60] += elastic_hessian(frame, basis, cfg.stiffness, 0.0)
    sea = np.zeros((105, 105), complex)
    contacts = np.zeros_like(sea)
    vertex_ward = []
    quantum_metric = np.zeros_like(sea)
    for load, band, x, role in data['parts']:
        h = np.einsum('pm,mij->pij', x, frame)
        values, vectors = np.linalg.eigh(h)
        sign = -1 if role.startswith('neutral') else 1
        occupied = slice(2, 4) if sign < 0 else slice(0, 2)
        empty = slice(0, 2) if sign < 0 else slice(2, 4)
        occupied_vectors = vectors[:, :, occupied]
        projector = occupied_vectors @ occupied_vectors.conj().transpose(0, 2, 1)
        vertices = np.zeros((105, volume, 4, 4), complex)
        vertices[:60] = np.einsum('pm,aij->mapij', (x + x[target]) / 2, basis).reshape(60, volume, 4, 4)
        contact = np.zeros_like(sea)
        _, paths = scalar_paths(band)
        for displacement, scalars in paths:
            axis = int(np.flatnonzero(displacement)[0])
            length = int(displacement[axis])
            F = np.einsum('m,mij->ij', scalars, frame)
            insertion = sum((np.exp(1j * qtheta[axis] * j) for j in range(length)))
            phase = np.exp(1j * theta @ displacement)
            reverse_phase = np.exp(-1j * (theta + qtheta) @ displacement)
            ft = (F @ generators + generators @ F) / 2
            fdt = (F.conj().T @ generators + generators @ F.conj().T) / 2
            vertices[60 + 15 * axis:75 + 15 * axis] += insertion * 1j * (phase[None, :, None, None] * ft[:, None] - reverse_phase[None, :, None, None] * fdt[:, None])
            rho_forward = np.einsum('p,pij->ij', phase, projector) / volume
            rho_reverse = np.einsum('p,pij->ij', phase.conj(), projector) / volume
            ab = generators[:, None] @ generators[None, :]
            ba = generators[None, :] @ generators[:, None]
            mf = -0.5 * length * (ab + ba)
            mr = mf.copy()
            for j in range(length):
                for k in range(j + 1, length):
                    z = np.exp(1j * qtheta[axis] * (k - j))
                    mf -= ab * z + ba * z.conjugate()
                    mr -= ba * z + ab * z.conjugate()
            forward = (F @ mf + mf @ F) / 2
            reverse = (mr @ F.conj().T + F.conj().T @ mr) / 2
            sl = slice(60 + 15 * axis, 75 + 15 * axis)
            contact[sl, sl] += np.einsum('ij,abji->ab', rho_forward, forward) + np.einsum('ij,abji->ab', rho_reverse, reverse)
            delta_f = (scalars[:, None, None, None] * basis[None]).reshape(60, 4, 4)
            z = np.exp(-1j * qtheta @ displacement)
            mixed_forward = 0.5j * insertion * (delta_f[:, None] @ generators[None] + z * (generators[None] @ delta_f[:, None]))
            delta_fd = (scalars.conj()[:, None, None, None] * basis[None]).reshape(60, 4, 4)
            mixed_reverse = -0.5j * insertion * (generators[None] @ delta_fd[:, None] + z * (delta_fd[:, None] @ generators[None]))
            mixed = np.einsum('ij,abji->ab', rho_forward, mixed_forward) + np.einsum('ij,abji->ab', rho_reverse, mixed_reverse)
            contact[:60, sl] += mixed
            contact[sl, :60] += mixed.conj().T
        bubble = np.zeros_like(sea)
        metric = np.zeros_like(sea)
        for target_sector, source_sector in ((empty, occupied), (occupied, empty)):
            tv = vectors[target, :, target_sector]
            sv = vectors[:, :, source_sector]
            features = np.einsum('pie,apij,pjo->apeo', tv.conj(), vertices, sv)
            denominator = np.abs(values[target, target_sector][:, :, None] - values[:, source_sector][:, None, :])
            weighted = (features / np.sqrt(denominator)[None]).reshape(105, -1)
            bubble -= weighted.conj() @ weighted.T / volume
            metric_features = (features / denominator[None]).reshape(105, -1)
            metric += 0.5 * (metric_features.conj() @ metric_features.T) / volume
        sea += load * (bubble + sign * contact)
        contacts += load * sign * contact
        quantum_metric += (1 if sign < 0 else load) * metric
        actual = np.einsum('apij,at->tpij', vertices, gauge)
        expected = np.asarray([1j * (t @ h - h[target] @ t) for t in generators])
        vertex_ward.append(dict(role=role, relative=float(np.linalg.norm(actual - expected) / (1 + np.linalg.norm(expected))), absolute=float(np.max(np.abs(actual - expected)))))
    full = (bosonic + sea + bosonic.conj().T + sea.conj().T) / 2
    normal = null_space(gauge.conj().T)
    physical = normal.conj().T @ full @ normal
    kinetic = normal.conj().T @ np.diag([cfg.chi] * 60 + [0.6 * cfg.chi] * 45) @ normal
    kv, ku = np.linalg.eigh(kinetic)
    kr = ku * np.sqrt(kv) @ ku.conj().T
    frequencies2 = np.linalg.eigvalsh(kr @ physical @ kr)
    return dict(q=q.tolist(), full=full, sea=sea, bosonic=bosonic, normal=normal, principal_cometric=kinetic, physical=physical, frequencies_squared=frequencies2, gauge=gauge, alignment_unit=align_vertex.conj().T @ align_vertex, colour_unit=curl.conj().T @ curl, quantum_metric=quantum_metric, vertex_Ward=vertex_ward, gauge_Hessian_residual=float(np.linalg.norm(full @ gauge)), omitted_contact_gauge_defect=float(np.linalg.norm((full - contacts) @ gauge)), orbit_Gram=orbit_columns.T @ orbit_columns)

def twisted_uniform_normal_packet(cfg: PhiConfig, size: int, quantum_mean, precision=200) -> dict[str, Any]:
    """Actual heavy-sea Hessian and normalized 45-dimensional slow Gaussian.

    This computes the entire non-Clifford normal Hessian, not only the two
    invariant Gram variables. Decimal packet coefficients refer to scaled
    coordinates y=Lambda^(3/8) z; the centre retains the separate full precision.
    """
    basis = clifford_cartesian_basis()
    tensors = np.rint(np.einsum('nij,ajk,mkl,bli->namb', SELECTED, basis, SELECTED, basis).real / 4).astype(int)
    commutators = np.rint(np.einsum('amij,bji->mab', 1j * (basis[:, None] @ SELECTED[None] - SELECTED[None] @ basis[:, None]), basis).real / 4).astype(int)
    with mp.workdps(precision):
        scales = [mp.sqrt(mp.mpf(quantum_mean[k])) for k in ('g0_decimal', 'gs_decimal', 'gs_decimal', 'gs_decimal')]
        orbit = mp.zeros(60, 15)
        used = set()
        vectors = []
        for a in range(15):
            support = []
            for mu in range(4):
                for b in range(15):
                    if commutators[mu, a, b]:
                        index = 15 * mu + b
                        orbit[index, a] = int(commutators[mu, a, b]) * scales[mu] / mp.sqrt(8)
                        support.append(index)
                        if index in used:
                            raise ArithmeticError('Clifford gauge columns unexpectedly overlap')
                        used.add(index)
            column = [orbit[index, a] for index in support]
            length = mp.sqrt(sum((z * z for z in column)))
            unit = [z / length for z in column]
            if len(support) > 1:
                householder = unit.copy()
                householder[0] -= 1
                norm = sum((z * z for z in householder))
                for j in range(1, len(support)):
                    vector = mp.zeros(60, 1)
                    for i, index in enumerate(support):
                        vector[index] = (1 if i == j else 0) - 2 * householder[i] * householder[j] / norm
                    vectors.append(vector)
        for index in range(60):
            if index not in used:
                vector = mp.zeros(60, 1)
                vector[index] = 1
                vectors.append(vector)
        normal = mp.matrix([[vector[i] for vector in vectors] for i in range(60)])
        if len(vectors) != 45:
            raise ArithmeticError('Actual full Cartesian normal dimension is not 45')
        rows = coefficient_rows(models(cfg)[0][1], size, quantum_mean['twist'])
        radii = [mp.sqrt(sum(((s * x) ** 2 for s, x in zip(scales, row)))) for row in rows]
        volume = len(rows)
        second = {}
        fourth = {}
        from itertools import product
        for indices in product(range(4), repeat=2):
            second[indices] = sum((row[indices[0]] * row[indices[1]] / radius for row, radius in zip(rows, radii))) / volume
        for indices in product(range(4), repeat=4):
            fourth[indices] = sum((mp.fprod((row[i] for i in indices)) / radius ** 3 for row, radius in zip(rows, radii))) / volume
        stiffness = 2 / mp.mpf(cfg.a)
        Hessian = mp.zeros(60)
        for mu, a, nu, b in product(range(4), range(15), range(4), range(15)):
            value = -(second[mu, nu] if a == b else 0)
            for i, j in product(range(4), repeat=2):
                coefficient = int(tensors[i, a, j, b])
                if coefficient:
                    value += coefficient * scales[i] * scales[j] * fourth[mu, nu, i, j]
            value += 2 * stiffness * ((scales[a] ** 2 if mu == nu and a == b and (a < 4) else 0) + (scales[nu] * scales[mu] if a == nu and b == mu else 0) + (scales[mu] ** 2 if mu == nu and a == b else 0))
            Hessian[15 * mu + a, 15 * nu + b] = value
        physical = normal.T * Hessian * normal
        eigenvalues, eigenvectors = mp.eigsy(physical)
        residual = max((abs(z) for z in physical * eigenvectors - eigenvectors * mp.diag(eigenvalues)))
        orth_error = max((abs(z) for z in eigenvectors.T * eigenvectors - mp.eye(45)))
        if eigenvalues[0] <= 0 or residual > mp.mpf(10) ** (-precision + 30):
            raise ArithmeticError('Actual high-precision full normal Hessian failed')
        covariance = eigenvectors * mp.diag([1 / (2 * mp.sqrt(z)) for z in eigenvalues]) * eigenvectors.T
        return dict(precision=precision, dimension=45, basis='15 exact signed Clifford Pauli monomials, Tr Ba Bb=4 delta; coordinate change only', minimum_stiffness=mp.nstr(eigenvalues[0], precision), maximum_stiffness=mp.nstr(eigenvalues[len(eigenvalues) - 1], precision), eigenvalues=[mp.nstr(z, precision) for z in eigenvalues], eigen_residual_decimal=mp.nstr(residual, precision), eigen_orthogonality_error_decimal=mp.nstr(orth_error, precision), normal_orthogonality_error_decimal=mp.nstr(max((abs(z) for z in normal.T * normal - mp.eye(45))), precision), normal_orbit_annihilation_decimal=mp.nstr(max((abs(z) for z in normal.T * orbit)), precision), normal_columns_decimal=[[mp.nstr(normal[i, j], precision) for j in range(45)] for i in range(60)], scaled_covariance_decimal=[[mp.nstr(covariance[i, j], precision) for j in range(45)] for i in range(45)], normalized_wavefunction='det(2pi Sigma)^(-1/4) exp(-y^T Sigma^-1 y/4), y=Lambda^(3/8) z_uniform; full precision centre independent of shape precision', normal_basis_definition='Physical basis is the exact disjoint-support Householder nullspace of O0 at the full literal decimal centre, using the displayed signed Clifford basis. normal_columns_decimal is a diagnostic approximation ONLY, never a replacement physical centre or rounded orbit. The displayed finite-decimal covariance defines the shape in that exact basis; shape error is priced separately.', scope='Actual normalized harmonic slow state. Joint fast, Haar/Gauss and boundary comparison are separate, not implicit.')

def preferred_prepared_report(cfg: PhiConfig, holonomy: Any, size=6, exponent=2048, cutoff_bits=512, clock=4096.0) -> dict[str, Any]:
    """Source-bound full-normal packet and explicit local-well comparison prices.

    The comparison is to the exact charged/EM Hamiltonian at the displayed
    mean, NOT to RPA or a continuum/Compton phase. The microscopic box is fixed.
    Scalar well ground energy is only an unobservable common reference phase.
    """
    start = time.monotonic()
    centre = global_centre_projection(cfg, size)
    if not centre['global_centre_admissible']:
        raise ArithmeticError('No positive filled-heavy Gauss packet on this odd box')
    quantum_mean = coupled_mean(cfg, size, exponent, holonomy)
    slow = twisted_uniform_normal_packet(cfg, size, quantum_mean)
    g = np.asarray(quantum_mean['G_float'])
    frame = np.sqrt(np.diag(g))[:, None, None] * SELECTED
    sites = periodic_grid(size)
    band = models(cfg)[0][1]
    data = make_data(cfg, size, quantum_mean, True)
    unit = replace(cfg, Lambda=1.0)
    coefficients_bound = 0.0
    metric_trace = 0.0
    Ward = 0.0
    rows = []
    zero = None
    for q in sites:
        block = twisted_native_block(data, unit, q)
        remainder = block['full'] - 0.2 * block['alignment_unit'] - 0.35 * block['colour_unit']
        norm = float(np.linalg.norm(remainder, 2))
        coefficients_bound = max(coefficients_bound, norm)
        metric_trace += float(np.trace(block['quantum_metric']).real)
        Ward = max(Ward, max((r['relative'] for r in block['vertex_Ward'])))
        rows.append(dict(q=q.tolist(), actual_heavy_quartic_remainder_norm=norm, actual_heavy_fibre_metric_trace=float(np.trace(block['quantum_metric']).real), relative_native_Ward=max((r['relative'] for r in block['vertex_Ward'])), relative_Hessian_gauge_residual=block['gauge_Hessian_residual'] / (1 + np.linalg.norm(block['full']) * np.linalg.norm(block['gauge'])), omitted_contact_gauge_defect=block['omitted_contact_gauge_defect']))
        if not any(q):
            zero = block
    if zero is None:
        raise ArithmeticError('Actual finite box omitted its zero Fourier block')
    normal = natural_gauss_chart(zero)['normal']
    native_eigenvalues = np.linalg.eigvalsh(normal.conj().T @ zero['full'][:60, :60] @ normal)
    precise_eigenvalues = np.asarray([float(z) for z in slow['eigenvalues']])
    discrepancy = float(np.max(np.abs(native_eigenvalues - precise_eigenvalues)))
    if discrepancy > 1e-09 * (1 + np.max(np.abs(precise_eigenvalues))):
        raise ArithmeticError('Independent high precision normal Hessian disagrees with native vertex/contact')
    control_cfg = replace(cfg, Lambda=float(2 ** 40))
    control_data = make_data(control_cfg, size, quantum_mean, False)
    control_blocks = [twisted_native_block(control_data, control_cfg, np.asarray(q)) for q in ((0, 0, 0), (1, 0, 0), (1, 2, 1))]
    geometry_control = canonical_geometry_report(control_cfg, size, control_data['frame'], control_blocks)
    Casimir_control = colour_link_casimir_control(control_cfg, control_blocks[0], size ** 3)
    EM_control = noncompact_EM_Gauss_control(size)
    cut_support_control = flat_cut_ramp_support_control()
    joint_descent = joint_gauge_descent_control(control_data['frame'])

    def path_cap(model):
        onsite, paths = scalar_paths(model)
        return abs(float(onsite)) + 2 * sum((float(np.sum(np.abs(s))) for _, s in paths))
    heavy_cap = path_cap(band)
    light_cap = sum((load * path_cap(model) for load, model, role in models(cfg) if not role.startswith('neutral')))
    with mp.workdps(200):
        scale = mp.mpf(2) ** exponent
        eta = mp.mpf(2) ** (-cutoff_bits)
        chi = scale ** (-mp.mpf(1) / 2)
        volume = size ** 3
        dimension = 90 * volume
        fast_dimension = dimension - 45
        g0, gs = (mp.mpf(quantum_mean['g0_decimal']), mp.mpf(quantum_mean['gs_decimal']))
        rho = mp.sqrt(g0 + 3 * gs)
        orbit_eigenvalues = [(b0 * g0 + bs * gs) / 2 for b0, bs, _ in orbit_categories()]
        rmin, rmax = (min(orbit_eigenvalues), max(orbit_eigenvalues))
        a0 = mp.mpf(slow['minimum_stiffness'])
        a1 = mp.mpf(slow['maximum_stiffness'])
        hfast = min(mp.mpf(1) / 5 * min((4 * sum((mp.sin(mp.pi * int(z) / size) ** 2 for z in q)) for q in sites if any(q))), rmin / 5)
        omega_min = mp.sqrt(3 * rmin / 25)
        kmax = 1 + 12 / rmin
        static_cap = mp.mpf(coefficients_bound) * 2
        normalized_perturbation = scale ** (-mp.mpf(1) / 4) * static_cap / hfast
        schur_denominator = 3 * hfast / 5 - scale ** (-mp.mpf(1) / 4) * (a1 + static_cap)
        if schur_denominator <= 0:
            raise ArithmeticError('Actual q0 fast block is not separated from the slow spectrum')
        schur = static_cap ** 2 / schur_denominator
        if normalized_perturbation >= mp.mpf('0.01') or scale ** (-mp.mpf(1) / 4) * schur > a0 / 4:
            raise ArithmeticError('Material hierarchy does not control full fast/slow harmonic blocks')
        coercivity_margin = hfast / 2 - scale ** (-mp.mpf(1) / 4) * (static_cap + 4 * static_cap ** 2 / a0)
        if coercivity_margin <= 0:
            raise ArithmeticError('Young inequality does not protect separate slow/fast potential masses')
        if scale ** (mp.mpf(1) / 4) * hfast < a0:
            raise ArithmeticError('Fast quadratic mass does not dominate the scalar radial mass')
        gap0 = scale ** (mp.mpf(1) / 4) * mp.sqrt(a0 / 2)
        zpe_interval = fast_zero_point_interval(quantum_mean, size, 200)
        zpe_upper = Fraction(int(zpe_interval['upper_exact']['numerator']), int(zpe_interval['upper_exact']['denominator']))
        fast_energy = 2 * scale ** (mp.mpf(3) / 8) * volume * (mp.mpf(zpe_upper.numerator) / zpe_upper.denominator)
        slow_energy = 45 * scale ** (mp.mpf(1) / 4) * mp.sqrt(static_cap)
        fast_variance_max = (mp.mpf(3) / 5 + 12 / rmin) / (2 * mp.sqrt(3 * rmin / 25 + mp.mpf(12) / 5))
        slow_variance_max = 1 / (2 * mp.sqrt(a0))
        mean_radius2 = fast_dimension * fast_variance_max * scale ** (-mp.mpf(7) / 8) + 45 * slow_variance_max * scale ** (-mp.mpf(3) / 4)
        cut_norm_lower = 1 - 4 * mean_radius2 / eta ** 2
        if cut_norm_lower <= mp.mpf('0.99'):
            raise ArithmeticError('Actual full Gaussian is not supported inside its Gauss chart')
        analytic_radius = min(mp.mpf('0.001'), rmin / (16 * mp.sqrt(rmax)))
        metric_cap = 4 + 512 / rmin
        metric_first = 8 * metric_cap / analytic_radius
        log_measure_first = 8 / analytic_radius
        log_measure_second = 32 / analytic_radius ** 2
        geometric_scalar = chi * (dimension ** 2 * (metric_first * log_measure_first + metric_cap * log_measure_second) / 4 + dimension * metric_cap * log_measure_first ** 2 / 8)
        electric_metric_relative = metric_first * eta / (mp.mpf(3) / 5)
        heavy_gap = 2 * mp.mpf(cfg.heavy_m) * mp.sqrt(g0)
        K1 = 8 * mp.mpf(heavy_cap) * (1 + rho)
        K2 = 16 * mp.mpf(heavy_cap) * (1 + rho)
        K3 = 32 * mp.mpf(heavy_cap) * (1 + rho)
        if heavy_gap <= 4 * K1 * eta:
            raise ArithmeticError('The actual heavy filled cluster is not gapped throughout the chart')
        occupied = 2 * volume
        third_heavy = occupied * (128 * K1 ** 3 / heavy_gap ** 2 + 32 * K1 * K2 / heavy_gap + K3)
        third_quartic = 24 * (2 / mp.mpf(cfg.a)) * (rho + 1)
        potential_relative = eta * (8 * (third_heavy + third_quartic) / a0 + 256 * (1 + rho) ** 2 / hfast + 4096 / rmin)
        rounded0 = mp.mpf(float(np.sqrt(g[0, 0]))) ** 2
        roundedS = mp.mpf(float(np.sqrt(g[1, 1]))) ** 2
        heavy_rows = coefficient_rows(band, size, quantum_mean['twist'])
        rounded_radii = [mp.sqrt(rounded0 * row[0] ** 2 + roundedS * sum((z * z for z in row[1:]))) for row in heavy_rows]
        rounded_f0 = 2 / mp.mpf(cfg.a) * rounded0 - sum((row[0] ** 2 / r for row, r in zip(heavy_rows, rounded_radii))) / volume
        rounded_fs = 6 / mp.mpf(cfg.a) * roundedS - sum((sum((z * z for z in row[1:])) / r for row, r in zip(heavy_rows, rounded_radii))) / volume
        rounded_force = 2 * mp.sqrt(rounded0 * rounded_f0 ** 2 + roundedS * rounded_fs ** 2 / 3)
        rounded_action_lower = scale ** (mp.mpf(5) / 8) * mp.sqrt(volume) * rounded_force / (2 * mp.sqrt(2 * mp.sqrt(a1)))
        form_relative = electric_metric_relative + potential_relative
        light_lipschitz = 8 * mp.sqrt(volume) * mp.mpf(light_cap) * (1 + rho)
        stationary_error = mp.mpf(quantum_mean['full_normalized_uniform_gradient_decimal'])
        linear_force = 4 * rho * mp.sqrt(volume) * scale ** (mp.mpf(3) / 8) * sum((abs(mp.mpf(z)) for z in quantum_mean['fast_zero_point_gradient_decimal'])) + light_lipschitz + 4 * scale * rho * mp.sqrt(volume) * stationary_error
        harmonic_energy_excess = scale ** (mp.mpf(1) / 8) * fast_dimension * static_cap * (fast_variance_max + kmax / omega_min) + 45 * schur / mp.sqrt(a0)
        trial_energy = fast_energy + slow_energy + harmonic_energy_excess
        normalization_excess = (1 / cut_norm_lower - 1) * trial_energy
        cut_derivative_energy = 8 * chi * metric_cap * mean_radius2 / eta ** 4
        cut_energy_excess = normalization_excess + 2 * mp.sqrt(trial_energy * cut_derivative_energy) + cut_derivative_energy
        shape_price = mp.mpf(10) ** (-slow['precision'] + 40) * dimension ** 2 * trial_energy
        scalar_energy_perturbation = form_relative * (trial_energy + gap0) + linear_force * eta + geometric_scalar
        baseline_excess = harmonic_energy_excess + cut_energy_excess + shape_price
        gap = gap0 - 4 * scalar_energy_perturbation - 2 * baseline_excess
        if gap < gap0 / 2:
            raise ArithmeticError('Actual full metric/potential form sandwich does not protect the well gap')
        energy_excess = harmonic_energy_excess + cut_energy_excess + shape_price + 2 * scalar_energy_perturbation
        spectral_distance = mp.sqrt(2 * energy_excess / gap)
        well_energy_upper = trial_energy + cut_energy_excess + shape_price + scalar_energy_perturbation
        weighted_energy_budget = well_energy_upper + linear_force * eta + geometric_scalar
        barrier = scale * a0 * eta ** 2 / 1024 - linear_force * eta - geometric_scalar
        shell_potential_lower = (1 - form_relative) * scale * a0 * eta ** 2 / 256 - linear_force * eta - geometric_scalar
        ramp_slope = eta * mp.sqrt(scale * a0 / (chi * metric_cap)) / 64
        ramp_kinetic_price = chi * metric_cap * ramp_slope ** 2 / 2
        if form_relative > mp.mpf('.01') or barrier < 16 * well_energy_upper or ramp_kinetic_price > shell_potential_lower / 8 or (shell_potential_lower - well_energy_upper - ramp_kinetic_price < barrier / 2):
            raise ArithmeticError('Repaired Agmon ramp does not lie below the actual well barrier')
        Agmon = eta ** 2 * mp.sqrt(scale * a0 / (chi * metric_cap)) / 1024
        weighted_norm2 = 1 + 2 * weighted_energy_budget / barrier
        weighted_gradient2 = 4 * weighted_energy_budget / (chi * mp.mpf('0.6')) + 2 * ramp_slope ** 2 * weighted_norm2
        boundary_prefactor = 2 * (chi * metric_cap * 8 / eta * mp.sqrt(weighted_gradient2) + chi / 2 * (dimension ** 2 * metric_first * 8 / eta + dimension * metric_cap * 96 / eta ** 2) * mp.sqrt(weighted_norm2))
        boundary_log2_upper = mp.log(boundary_prefactor, 2) - Agmon / mp.log(2)
        if boundary_log2_upper > -4096:
            raise ArithmeticError('Full chart boundary action has not been priced')
        D1 = 4 * mp.sqrt(dimension) * occupied * K1 / heavy_gap
        D2 = dimension * (64 * occupied ** 2 * K1 ** 2 / heavy_gap ** 2 + 16 * occupied * K2 / heavy_gap)
        derivative_norm = mp.sqrt(8 * weighted_energy_budget / (chi * mp.mpf('0.6')))
        charge_norm = mp.sqrt(volume) * (3 + cfg.auxiliary_count) * mp.sqrt(mp.mpf(5) / 2)
        fibre_action = chi * (4 * metric_cap * derivative_norm * (D1 + charge_norm / rmin) + metric_cap * D2 + dimension * metric_first * (D1 + charge_norm) + metric_cap * charge_norm ** 2 / rmin)
        charged_action = light_lipschitz * eta
        action = fibre_action + charged_action + mp.mpf(2) ** (-4096)
        distance = 2 * spectral_distance + mp.mpf(clock) * action
        em_radius = mp.mpf(1) / 8
        em_modes = 2 * volume + 1
        UElink = mp.mpf(cfg.inertia) / mp.mpf(cfg.a)
        smallest_laplacian = min((4 * sum((mp.sin(mp.pi * int(z) / size) ** 2 for z in q)) for q in sites if any(q)))
        em_graph = UElink * mp.sqrt(mp.mpf(63) * em_modes / 2 + 9 * em_modes * (em_modes - 1)) / em_radius ** 2
        charge_cap = 2 * (1 + cfg.auxiliary_count) + 4
        Coulomb_graph = UElink * volume * charge_cap ** 2 / smallest_laplacian
        charged_graph = 8 * volume * mp.mpf(light_cap) * rho
        reference_graph = em_graph + Coulomb_graph + charged_graph
        quartic_coercivity = scale / (8 * mp.mpf(cfg.a) * volume)
        global_matter_cap = 8 * mp.sqrt(volume) * (scale * mp.mpf(heavy_cap) + mp.mpf(light_cap))
        lower_energy_shift = 3 * global_matter_cap ** (mp.mpf(4) / 3) / (4 ** (mp.mpf(4) / 3) * quartic_coercivity ** (mp.mpf(1) / 3))
        packet_rho_cap = mp.sqrt(volume) * rho + eta
        inverse_fast_covariance = 2 * mp.sqrt(3 * rmax / 25 + mp.mpf(63) / 25) / (mp.mpf(3) / 5)
        kinetic_graph = metric_cap * dimension * (scale ** (mp.mpf(3) / 8) * (1 + inverse_fast_covariance) + scale ** (mp.mpf(1) / 4) * (1 + 2 * mp.sqrt(a1))) + 16 * metric_cap * dimension * chi / eta ** 2
        potential_graph = scale / mp.mpf(cfg.a) * volume * (rho + eta) ** 4 + 8 * scale * volume * mp.mpf(heavy_cap) * (rho + eta) + 128 * volume * scale ** (mp.mpf(5) / 4) * (1 + rho) ** 2 * eta ** 2
        graph_cap = kinetic_graph + potential_graph + reference_graph + action
        difference_graph = 2 * (graph_cap + lower_energy_shift)
        weighted_Phi_error = (difference_graph / quartic_coercivity) ** (mp.mpf(1) / 4) * distance ** (mp.mpf(3) / 4)
        current_cap = 16 * mp.mpf(cfg.a) * mp.sqrt(volume) * mp.mpf(light_cap)
        contact_cap = 32 * mp.mpf(cfg.a) ** 2 * mp.sqrt(volume) * mp.mpf(light_cap)
        current_action_error = current_cap * weighted_Phi_error
        contact_action_error = contact_cap * weighted_Phi_error
        current_expectation_error = 2 * current_action_error + current_cap * (1 + rho) * eta
        contact_expectation_error = 2 * contact_action_error + contact_cap * (1 + rho) * eta
        fast_rows = []
        for q in sites:
            lap = 4 * sum((mp.sin(mp.pi * int(z) / size) ** 2 for z in q))
            coefficients_rows = []
            for (b0, bs, mult), r in zip(orbit_categories(), orbit_eigenvalues):
                transverse = mp.mpf(3) / (10 * mp.sqrt(3 * r / 25 + 21 * lap / 100))
                longitudinal = (mp.mpf(3) / 5 + lap / r) / (2 * mp.sqrt(3 * r / 25 + lap / 5))
                coefficients_rows.append(dict(category=[b0, bs, mult], r=mp.nstr(r, 200), transverse_scaled_variance=mp.nstr(transverse, 200), longitudinal_scaled_variance=mp.nstr(longitudinal, 200)))
            fast_rows.append(dict(q=q.tolist(), lap=mp.nstr(lap, 200), Phi_scaled_variance=mp.nstr(1 / (2 * mp.sqrt(lap / 5)), 200) if any(q) else None, colour=coefficients_rows))
        price = lambda value: mp.nstr(value, 60)
        prices = dict(eta=price(eta), principal_metric_cap=price(metric_cap), normal_gap_lower=price(gap), fast_energy_upper=price(fast_energy), harmonic_reprepare_energy_excess=price(harmonic_energy_excess), Haar_FP_scalar_action_upper=price(geometric_scalar), nonlinear_form_relative=price(form_relative), actual_H2_cut_norm_squared_lower=dict(expression='1-deficit', deficit_upper=price(4 * mean_radius2 / eta ** 2)), cut_energy_excess=price(cut_energy_excess), shape_rounding_energy_excess=price(shape_price), mean_force_linear_action_on_chart=price(linear_force * eta), rounded_literal_centre_force_action_lower=price(rounded_action_lower), well_spectral_state_distance_upper=price(spectral_distance), Agmon_exponent_lower=price(Agmon), physical_boundary_action_log2_upper=price(boundary_log2_upper), heavy_Berry_Gauss_affine_action_upper=price(fibre_action), charged_EM_mean_freezing_action_upper=price(charged_action), reference_EM_CAR_graph_norm_upper=price(reference_graph), full_Hamiltonian_lower_shift=price(lower_energy_shift), quartic_graph_interpolation_coercivity=price(quartic_coercivity), difference_graph_norm_upper=price(difference_graph), weighted_Phi_action_error=price(weighted_Phi_error), all_flavour_current_action_error=price(current_action_error), all_flavour_contact_action_error=price(contact_action_error), all_flavour_current_expectation_error=price(current_expectation_error), all_flavour_contact_expectation_error=price(contact_expectation_error), physical_state_distance_at_clock_upper=price(distance), clock=clock)
        derivation = dict(coordinates=dict(real_space='45 exact normal Cart coordinates per site, 15 exponential Lie coordinates per oriented link; Fourier transform is real orthogonal and does not alter the radial cutoff', chart_radius=price(analytic_radius), well_radius=price(eta), flat_cut_inner=price(eta / 4), flat_cut_outer=price(eta / 2), FP='A=O0^T O(Phi), R0=O0^T O0. w_local=prod det(A)/sqrt(det(R0)) times prod det(Jleft). Joint Phi/U gauge tube has residual diagonalZ4 and cover degree4, not4^V. With normalized SU4 Haar and Maurer volume Vol4, w_joint=Vol4^(V-L) w_local/4; cubic L=3V gives Vol4^(-2V)w_local/4. W=w_joint^(-1/2) times matter-equivariant extension. Constant volume/cover factors drop from every log-measure derivative.', exact_basis=slow['normal_basis_definition']), joint_map='Phi_x=Vx(Cmean+N0Yx)Vx^dagger; Uxy=Vx zxy exp(iTheta_xy.T) Vy^dagger, fixed flat central seam zxy. Y and relativeTheta lie in displayed near-zero chart. Phi stabilizer isZ4 because full-rank Clifford frame generates M4; near-I link support and connectivity force all local centre choices equal.', descent_isometry='Psi(F(Y,Theta,V))=R_matter(V)w_joint^(-1/2) f(Y,Theta). Diagonal z acts by z^Ntotal; declared totalNmod4=0 gives single-valued descent. Degree4 change of variables gives dmu_boson=Vol4^(V-L)w_local dYdTheta dHaar(V)/4. Normalized Haar integral1 and unitary R_matter give integral||Psi||^2=integral||f||^2=1. Vol4 is geometrical total Maurer volume, not adjustable input.', ellipticity=dict(lower_exact='3/5', formula='pPhi=N0 pN+O0 A^(-T)[rho-div(Eleft)-O(Phi)^T N0 pN], Eleft=Jleft^(-T) ptheta. K/chi is I45 plus the positive Gauss-square and (3/5) Jleft^(-1)Jleft^(-T).', derivation='For real theta, Jleft=int_0^1 Ad(exp(it theta.T))dt averages orthogonal matrices, hence ||Jleft||<=1. N0 is exactly orthogonal to O0. Thus the homogeneous kinetic form is >=(3/5)(|pN|^2+|ptheta|^2), not a fitted normal metric.', complex_cap=price(metric_cap), cap_derivation='||delta O||<=2||delta C|| follows adjoint Casimir4. ||R0^-1/2 delta A R0^-1/2||<=2sqrt(rmax)R/rmin<=1/8. Hence ||O0 A^-T||<=2/sqrt(rmin). The six-edge divergence has norm<=sqrt(13), ||Jleft^-1||<=2 and ||O(Phi)^T N0||<=2R. The resulting principal norm is <=4+(208+16R^2)/rmin <=4+512/rmin.', metric_Cauchy_first=price(metric_first)), measure=dict(log_first_cap=price(log_measure_first), log_second_cap=price(log_measure_second), derivation='Per local site plus its three outgoing links, |log(w/w0)|<4: 15[-log(1-1/8)]<2.01 for FP, and the three small Haar determinants contribute <1.99 at R<=.001. Real-space Cauchy estimates on radius R/2 give 8/R and 32/R^2. Fourier coordinates are NOT used to claim a dimension-free bound on the sum log(w).', exact_scalar='U_w=chi div(Kbar grad log w)/4+chi (grad log w)^T Kbar grad log w/8, where Kbar=K/chi. Sum at most D^2 first/second-derivative terms and D gradient-square terms gives the displayed scalar cap.'), potential=dict(heavy_gap=price(heavy_gap), chart_heavy_gap_lower=price(heavy_gap - 2 * K1 * eta), K1=price(K1), K2=price(K2), K3=price(K3), heavy_third_cap=price(third_heavy), heavy_derivation="Actual one-link row/column path sum bounds h derivatives by K1,K2,K3. Perturbation <=K1 eta leaves cluster gap >=Delta/2. E'''=Tr(P h'''+2P' h''+P'' h'). With ambient dimension4V=2 nocc, projector differentiation gives nocc[K3+16K1K2/gap+32K1^3/gap^2]. Substituting gap>=Delta/2 gives the recorded128/32 coefficients.", quartic_third_cap=price(third_quartic), separate_mass_margin=price(coercivity_margin), quadratic_lower='Young: 2lambda C |s||f| >=-lambda amin |s|^2/4-4lambda C^2 |f|^2/amin. The recorded positive margin gives Vharm>=lambda amin |s|^2/4+lambda^(5/4)hfast |f|^2/4.', heavy_quartic_remainder='Their cubic remainder is <=lambda(Th+Tq) eta (|s|^2+|f|^2)/6. Comparing to the separate quadratic lower bounds is dominated by 8(Th+Tq)eta/amin.', alignment_remainder='Only fast coordinates enter the linear covariant difference; the uniform normal displacement has zero linear difference. The nonlinear difference is bounded by4(1+rho)eta |f|, the linear one by6(1+rho)|f|. Thus |Ralign|<=32 J(1+rho)^2 eta |f|^2, dominated relative to the separate fast mass by256(1+rho)^2 eta/hfast. No erroneous J eta/slow-mass bound is used.', curl_remainder='Each link occurs in four plaquettes. Expanding the four unitary exponentials on the radius<=.001 ball and summing the cubic remainder bounds it by a constant times lambda^(5/4) eta |theta|^2. Relative to the Higgs mass, the explicit4096 eta/rmin cap dominates this term.'), minmax=dict(exact_common='Write E_h for the exact full harmonic ground scalar. Identical exact V(mean), Efast and slow zero-point scalars cancel algebraically before every gap/excess comparison; numerical energies are used only as positive upper bounds.', roots='Fast block roots change by at most C lambda^(1/8)/omega_min; q0 slow roots obey the recorded Schur bound. Leading Gaussian energy minus E_h is <=harmonic_reprepare_energy_excess.', sandwich='(1-epsilon)Hharm-F eta-U <= Hwell-V(mean) <=(1+epsilon)Hharm+F eta+U on the Dirichlet ball.', endpoints='Cut trial upper E0<=E_h+X+P; minmax lower E1>=E_h+gap0-P; X=harmonic excess+cut excess+shape price and P=epsilon(Etrial_upper+gap0)+F eta+U. Therefore gap0-4P-2X is a conservative gap lower, and X+2P bounds trial-ground energy excess. The harmonic Dirichlet ground cutoff shift X is explicitly paid, not silently dropped.', spectral_overlap='If excess=X+2P and gap>0, 1-|<trial,ground>|^2<=excess/gap; phase-aligned distance<=sqrt(2 excess/gap).'), cutoffs=dict(H2_radial='b=(1-r^2/eta^2)^2_+. b and its first derivative vanish at r=eta. ||grad b||<=4r/eta^2, ||Hess b||<=12/eta^2, |Delta b|<=(4D+8)/eta^2. ||b Gaussian||^2>=1-4 E(r^2)/eta^2. Normalize by the exact positive integral, not by a float rounded to1.', flat='u=4r/eta-1; interior1, shell1-(10u^3-15u^4+6u^5), exterior0. For eta/4<=r<=eta/2: |grad cut|<=8/eta, ||Hess cut||<=96/eta^2, |Delta cut|<=(96+32D)/eta^2. This flat cutoff, not zero-extension of the Dirichlet ground, is used for the H2 quasimode.', barrier=price(barrier), Agmon=price(Agmon), agmon_derivation='Ramp begins eta/8, strictly inside first flat-cut derivatives eta/4. Slope=eta sqrt(lambda amin/(chi M))/64; weight kinetic price=lambda amin eta^2/8192. Actual shell potential lower=(1-epsilon)lambda amin eta^2/256-F eta-U. Price<=this shell lower/8 and shell lower-E_D-price>=recorded weaker barrier/2 are executed guards. This avoids incorrectly claiming price<=(unperturbed barrier-F eta-U)/8 at equality. E_D-V(mean)<=recorded well_energy_upper via normalized cut trial, barrier>16 times that upper. Distance eta/8 gives weight>=eta^2 sqrt(lambda amin/(chi M))/512; retained /1024 exponent is conservative.', boundary='Weighted Dirichlet identity gives outside coefficient V-E_D-chi grad(f)^T Kbar grad(f)/2>=barrier/2. Interior potential>=-F eta-U implies weighted norm^2<=1+2 budget/barrier and weighted gradient^2<=4 budget/(chi*.6)+2 slope^2 weighted norm^2. Commutator chi Kbar grad(cut).grad plus chi div(Kbar grad(cut))/2 uses actual8/eta,96/eta^2 constants and D^2 metric derivative terms. Executed boundary prefactor includes normalization factor2 and multiplies exp(-Agmon), not an exponent-only assertion.', well_energy_upper=price(well_energy_upper), weighted_energy_budget=price(weighted_energy_budget), ramp_slope=price(ramp_slope), ramp_kinetic_price=price(ramp_kinetic_price), shell_potential_lower=price(shell_potential_lower), weighted_norm_squared_upper=dict(expression='1+excess', excess_upper=price(2 * weighted_energy_budget / barrier)), weighted_gradient_squared_upper=price(weighted_gradient2), actual_boundary_prefactor=price(boundary_prefactor)), graphs=dict(coercivity='Quartic >=lambda rhoPhi^4/(4aV)=2c rhoPhi^4. Matter >=-L rhoPhi by actual unitary paths. The retained other half c rho^4-L rho>=-3L^(4/3)/(4^(4/3)c^(1/3))=-B. Positive kinetic/alignment/curl/electric/Coulomb terms give H+B>=c rhoPhi^4.', reference_at_all_times='Reference H_ref graph norm is conserved by its exact unitary. Direct unitary path and charged number bounds are uniform in EM coordinates; Phi cut support remains fixed. Kinetic Gaussian/cutoff/fibre derivative bounds and positive-potential path bounds thus bound the full H+B applied to the lifted reference at each time. The same initial full graph norm is conserved along the actual full evolution. Their sum is the recorded difference-graph cap.', EM_moments='For normalized 1D (1-(theta/eta)^2)^2_+, <p^2>=3/eta^2 and ||p^2 f||^2=63/(2eta^4). Orthogonal product modes give ||sum p_j^2||=sqrt(63D/2+9D(D-1))/eta^2. These are dimensionless canonical Peierls-phase coordinates, not physical A coordinates.', Heinz='rho^4<=c^-1(H+B) in forms. Operator monotonic square root gives rho^2<=c^-1/2(H+B)^1/2. Spectral interpolation then gives ||rho delta||<=c^-1/4 ||delta||^3/4 ||(H+B)delta||^1/4. Both graph norms are priced, not inferred merely from norm closeness.', consumer='Peierls phases are unitary for every quantum theta. Actual one/two-link path derivative sums give ||J delta||<=CJ||rho delta|| and ||contact delta||<=CJJ||rho delta||. Cauchy-Schwarz gives the displayed expectation differences.'), review_status='Explicit derivations and executed constants for parent mathematical review; not automatic theorem admission')
    microscopic = {key: value for key, value in asdict(cfg).items() if key != 'Lambda'}
    microscopic['Lambda_exact'] = str(2 ** exponent)
    descriptor = dict(config=microscopic, size=size, Lambda_exponent=exponent, cutoff_bits=cutoff_bits, clock=clock, bosonic_coefficients='Canonical LAW05 exact fractions J=Lambda^(5/4)/5, colour curl7Lambda^(5/4)/10, colour Hamiltonian3chi E4^2/10 and cometric3chi/5; scalar-path matter coefficients are exact binary64 inputs', colour_gauge_section='JOINT Phi/U near-relative-link chart; connected-link residual diagonalZ4, totalNmod4 descent. Normalized SU4 Haar, coverdegree4 and geometrical Maurer-volume constants included in W; no independent local-centre matter projection.', scalar_heavy_sector='fixed NH=2V upper-h occupied; minusLambda h, exact actual one-link paths', charged_reference='all64 actual CAR flavours, full linear Cartesian one/two-link Peierls h(physical exact mean, Ucolour=zxy, actual central seam links, A quantum); electric plus Coulomb reference retained', reference_EM_packet=dict(coexact_harmonic_modes=2 * size ** 3 + 1, radius='1/8', coordinates='dimensionless canonical Peierls phases theta_link=a Aphysical_link; primitive electric UEphys/a ptheta_link^2', Hilbert_domain='NONCOMPACT electromagnetic reference: L2(R^(2V+1)), quotient of theta_link in R^(3V) by additive real site gauges, with real-gradient Gauss solution. NOT compact U1/integer electric/Haar branch.', Coulomb_solution='Pi_long=D^T(DD^T)^+rho; H_C=UEphys/a rho^T(DD^T)^+rho, total charge0. Spectral norm<=1/[4sin(pi/N)^2] gives the same Coulomb graph cap used here.', wavefunction='normalized product (1-(theta_j/etaTheta)^2)^2_+ in real orthonormal coexact+harmonic coordinates', charge_sector='all64 filled valence doublets, Ref0/4 or extra active electron plus compensating Ref hole; total U1charge0 and SU4centreclass0'), quantum_inputs='hbar=1 canonical Cart/Haar/EM quantization and CAR declared inputs', twist=quantum_mean['twist'], minimum_compatible_bosonic_q_phys=2 * np.pi / (size * cfg.a), holonomy_input_sha256=holonomy['input_sha256'], governing_plan=GOVERNING_PLAN)
    return dict(branch='NEW preferred222 actual canonical full-normal Gauss local-well comparison; not000 price retag or vacuum/continuum promotion', input_descriptor=descriptor, input_sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest(), quantum_mean=quantum_mean, slow_packet=slow, fast_packet=fast_rows, physical_centre=centre, native_heavy_blocks=rows, actual_native_Ward_max=Ward, actual_heavy_metric_trace=metric_trace, independent_MP_native_Hessian_discrepancy=discrepancy, prices=prices, derivation_ledger=derivation, canonical_geometry_control=dict(microscopic=asdict(control_cfg), size=size, role='Geometry control at the actual preferred centre and independent Lambda2^40, not a stationary Lambda2^40 phase', report=geometry_control), canonical_one_link_Casimir_control=Casimir_control, noncompact_EM_Gauss_control=EM_control, flat_cut_Agmon_support_control=cut_support_control, joint_gauge_descent_control=joint_descent, normalized_state='All45 uniform slow+90V-45 fast real Fourier Gaussian modes at exact decimal centre, H2 radialcut and exact positive normalization. Joint Phi/U chart with relative links near fixed zxy; their Lie increments nearI, residual diagonalZ4. W=[Vol4^(-2V)prod(det A/sqrt det R0)prod HaarJac/4]^(-1/2) followed by unitary matter-equivariant extension. TotalNmod4=0 ensures descent; degree4 change of variables proves W isometry.', fast_zero_point_interval=zpe_interval, common_phase='Exact scalar Dirichlet-well eigenvalue E_D defines exp(-it E_D); no rounded numerical phase is used. V(mean) and the SAME exact algebraic/trigonometric Efast cancel before excitation-energy/excess comparison.', fast_covariance='z_fast=Lambda^-7/16 y. Per colour r: transverse/longitudinal projectors of d_axis=1-exp(iq_axis); q0 three identical colour variances. Nonzero-q Phi45 identity covariance; q0 Phi45 belongs slow packet only.', comparison_lemmas=['Native harmonic full block has fast-root variation <=C Lambda^1/8 and q0 slow Schur root correction O(1), with displayed actual remainder norm and scale-separated Schur margin.', 'On the chart ball, exact Gauss/Haar principal metric and scalar measure potential obey displayed Cauchy caps. Heavy Kato cluster third derivative and quartic/colour/alignment Taylor bounds give the displayed quadratic-form sandwich.', 'Dirichlet well gap >= harmonic gap minus twice both endpoint form-price errors. Trial energy excess over the well ground divided by the gap bounds squared state distance.', 'The displayed potential barrier and elliptic cometric give Agmon decay; a flat-interior smooth cutoff of the well ground has the displayed physical boundary-action bound.', 'Physical Gaussian is close to that quasimode. Exact unitarity preserves this preparation error; Duhamel adds actual gapped-heavy fibre, affine matter-Gauss and charged/EM freezing actions.', 'Retained half-quartic gives H+B>=c rhoPhi^4. Heinz interpolation yields ||rhoPhi deltaPsi||<=c^-1/4 ||deltaPsi||^3/4 ||(H+B)deltaPsi||^1/4. Actual all-mode H2 EM input and CAR/Coulomb caps price graph norm; linear current/contact Peierls-unitary caps then price the consumer.'], proof_status='Executed coefficients, normalized packet, native controls and analytic-bound arithmetic; comparison lemmas displayed for parent mathematical review, NOT an exact many-body simulation or an admitted full quantum phase.', limitations='Finite even box only, not macrobox/bulk stationarity. No common-cone repair. This is Phi/colour and weighted-current transfer to exact charged/EM reference, not validation of RPA photon vacuum or Compton kinetics.', source_at_import=IMPORT_HASHES, source_snapshot_at_import=dict(path='induced_phi_holonomy_prepared.py', sha256=IMPORT_HASHES['induced_phi_holonomy_prepared.py'], text=IMPORT_BYTES['induced_phi_holonomy_prepared.py'].decode('utf8')), source_unchanged_since_import={name: IMPORT_HASHES[name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}, runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, platform=platform.platform()), elapsed_seconds=time.monotonic() - start)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--holonomy',required=True)
    parser.add_argument('--selection-proof',required=True)
    parser.add_argument('--clock',type=float,default=4096.)
    parser.add_argument('--cutoff-bits',type=int,default=512)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    raw=Path(args.holonomy).read_bytes();holonomy=json.loads(raw)
    proof_raw=Path(args.selection_proof).read_bytes();proof=json.loads(proof_raw)
    if proof['input_descriptor']['holonomy_receipt_sha256']!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Outward selector proof is not the actual frozen holonomy input')
    margin=proof['strict_uniform_sector_selection_margin']
    if Fraction(int(margin['numerator']),int(margin['denominator']))<=0 or proof['preferred_twist']!=[2,2,2]:
        raise ValueError('The actual outward selected sector is not222')
    cfg=PhiConfig(Lambda=1.,**holonomy['input_descriptor']['config'])
    report=preferred_prepared_report(cfg,holonomy,size=holonomy['input_descriptor']['size'],
        exponent=holonomy['input_descriptor']['Lambda_exponent'],cutoff_bits=args.cutoff_bits,clock=args.clock)
    diagnostic_gram=mp.matrix(report['quantum_mean']['G_float'])
    report['literal_selected_seam_control']=seam_control(holonomy['input_descriptor']['size'],(2,2,2),diagnostic_gram)
    literal_band=native_band_control(cfg,holonomy['input_descriptor']['size'],(2,2,2),diagnostic_gram)
    literal_band.pop('frame')
    report['literal_selected_band_control']=literal_band
    report['literal_colour_current_contact_control']=literal_colour_current_contact(cfg,holonomy['input_descriptor']['size'],report['quantum_mean'])
    report['input_descriptor']['holonomy_receipt_sha256']=hashlib.sha256(raw).hexdigest()
    report['input_descriptor']['selection_proof_receipt_sha256']=hashlib.sha256(proof_raw).hexdigest()
    report['input_sha256']=hashlib.sha256(json.dumps(report['input_descriptor'],sort_keys=True).encode()).hexdigest()
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=args.output,source_at_import=report['source_at_import'],input_sha256=report['input_sha256'],
        twist=report['quantum_mean']['twist'],prices=report['prices'],native_Ward_max=report['actual_native_Ward_max'],
        Hessian_discrepancy=report['independent_MP_native_Hessian_discrepancy'],elapsed_seconds=report['elapsed_seconds']),indent=2))
