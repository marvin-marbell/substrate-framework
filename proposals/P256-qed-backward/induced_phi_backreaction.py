#!/usr/bin/env python3
"""NEW common-lattice Cartesian Phi mean and quantum prepared-state supplier.

No original source is modified. Uniform bounds hold for total half filling,
not just two particles per momentum. Spatial stability and RPA are separately
priced; a classical minimizer is never labelled a quantum ground state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import mpmath as mp
from scipy.linalg import null_space, orth
from rotation_band import RotationBand
from sea_selection import SELECTED, coefficients, gram, matrix_unit_basis, rank_report
from soft_budget import select_gram
from local_orbital_stability import periodic_grid, elastic_hessian, fourier_sea_hessian, scalar_paths
from induced_photon_response import ResponseConfig, SpectralResponse, mean_field_remainder
from induced_photon_mode_frame import compute_response, LOADED_SOURCE_BYTES as MODE_FRAME_BYTES
from induced_compton_transfer import SameCurrent, transfer

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ('induced_phi_backreaction.py', 'rotation_band.py', 'sea_selection.py',
                'soft_budget.py', 'local_orbital_stability.py', 'improved_phi_band.py',
                'induced_photon_response.py', 'induced_photon_mode_frame.py',
                'induced_compton_transfer.py', 'compton.py')
IMPORT_BYTES = {name: MODE_FRAME_BYTES if name=='induced_photon_mode_frame.py'
    else (ROOT/name).read_bytes() for name in SOURCE_NAMES}
IMPORT_HASHES = {name: hashlib.sha256(content).hexdigest() for name,content in IMPORT_BYTES.items()}


@dataclass(frozen=True)
class PhiConfig:
    a: float = .015625
    Lambda: float = 256.
    heavy_m: float = .2
    heavy_r: float = .12
    heavy_v: float = .3
    active_m: float = .2
    auxiliary_m: float = 2.
    auxiliary_count: int = 63
    light_r: float = .12
    light_v: float = .4990360130634525
    inertia: float = 1.

    @property
    def stiffness(self) -> float:
        return 2*self.Lambda/self.a

    @property
    def alignment(self) -> float:
        return .2*self.Lambda**1.25

    @property
    def chi(self) -> float:
        return self.Lambda**-.5


def models(cfg: PhiConfig):
    return [(cfg.Lambda, RotationBand(a=cfg.a, m=cfg.heavy_m, r=cfg.heavy_r,
                                     v=cfg.heavy_v, improved=False), 'neutral heavy -Lambda'),
            (1., RotationBand(a=cfg.a, m=cfg.active_m, r=cfg.light_r,
                              v=cfg.light_v, improved=True), 'active filled charged'),
            (float(cfg.auxiliary_count), RotationBand(a=cfg.a, m=cfg.auxiliary_m,
                r=cfg.light_r, v=cfg.light_v, improved=True), 'auxiliary filled charged')]


def uniform(cfg: PhiConfig, size: int, quadrature_order: int = 10):
    if size:
        sites = periodic_grid(size)
        p = 2*np.pi*sites/(size*cfg.a)
        momentum_weights = np.full(len(p),1/len(p))
    else:
        # The supplier's exact cubical-shell quadrature partitions the entire BZ.
        # Old metric only chooses a mesh scale, not an energy or mean constraint.
        mesh = SpectralResponse(ResponseConfig(a=cfg.a,m=cfg.active_m,
            r=cfg.light_r,v=cfg.light_v,flavors=1),quadrature_order)
        chunks = list(mesh.chunks())
        p = np.concatenate([theta/cfg.a for theta,_ in chunks])
        momentum_weights = np.concatenate([w for _,w in chunks])
        sites = None
    parts = [(load, band, coefficients(band, p), role)
             for load, band, role in models(cfg) if load > 0]
    total = sum(load for load, _, _, _ in parts)
    x = np.concatenate([part[2] for part in parts])
    w = np.concatenate([momentum_weights*load/total for load, _, _, _ in parts])
    # One global energy rescaling preserves select_gram's normalized-weight ABI.
    selected = select_gram(x, cfg.stiffness/total, 0., weights=w)
    g = selected['G']
    frame = np.sqrt(np.diag(g))[:, None, None]*SELECTED
    r = np.sqrt(np.einsum('pi,ij,pj->p', x, g, x))
    derivative = cfg.stiffness*g-total*np.einsum('p,pi,pj->ij', w/r, x, x)
    # Orthogonal coordinates on Sym(4): Frobenius metric, all TEN components.
    sym = []
    for i in range(4):
        z = np.zeros((4, 4)); z[i, i] = 1.; sym.append(z)
        for j in range(i+1, 4):
            z = np.zeros((4, 4)); z[i, j] = z[j, i] = 1/np.sqrt(2); sym.append(z)
    sym = np.asarray(sym)
    features = np.einsum('pi,tij,pj->pt', x, sym, x)
    h10 = cfg.stiffness*np.eye(10)+.5*total*np.einsum('p,pt,pu->tu', w/r**3, features, features)
    basis = matrix_unit_basis()
    orbit = np.asarray([1j*(b@frame-frame@b) for b in basis])
    orbit = orth(np.einsum('tmij,aji->mat', orbit, basis).real.reshape(60, 15)/4)
    return dict(size=size, sites=sites, parts=parts, x=x, weights=w, total=total,
                G=g, frame=frame, value=total*selected['value'], derivative=derivative,
                h10=h10, orbit=orbit, basis=basis,momentum_weights=momentum_weights,
                quadrature_order=quadrature_order if not size else None)


def objective(data, cfg: PhiConfig, g: np.ndarray):
    r = np.sqrt(np.maximum(0., np.einsum('pi,ij,pj->p', data['x'], g, data['x'])))
    return .5*cfg.stiffness*float(np.sum(g*g))-2*data['total']*float(data['weights']@r)


def uniform_report(data, cfg: PhiConfig):
    rng = np.random.default_rng(222)
    alternatives = []
    for label in ('rank1', 'rank2', 'rank3', 'anisotropic', 'nonClifford1', 'nonClifford2'):
        if label.startswith('rank'):
            g = data['G'].copy(); rank = int(label[-1]); g[rank:, rank:] = 0
            frame = np.sqrt(np.diag(g))[:, None, None]*SELECTED
        elif label == 'anisotropic':
            g = data['G'].copy(); g[1, 1] *= 1.5; g[2, 2] *= .5
            frame = np.sqrt(np.diag(g))[:, None, None]*SELECTED
        else:
            c = rng.normal(size=(4, 15))
            c *= np.sqrt(np.diag(data['G']))[:, None]/np.linalg.norm(c, axis=1)[:, None]
            frame = np.einsum('ma,aij->mij', c, data['basis']); g = gram(frame)
        sea = 0.
        for load, _, x, role in data['parts']:
            values = np.linalg.eigvalsh(np.einsum('pm,mij->pij', x, frame))
            if role.startswith('neutral'):
                values = -values
            # Actual fixed TOTAL weighted half filling, including heavy sign.
            order = np.argsort(values.ravel())
            occupation_weights = np.repeat(data['momentum_weights'],4)[order]
            fill = np.minimum(occupation_weights,np.maximum(0.,2-np.r_[0.,np.cumsum(occupation_weights)[:-1]]))
            sea += load*float(fill@values.ravel()[order])
        actual = sea+.5*cfg.stiffness*float(np.sum(g*g))
        lower = objective(data, cfg, g)
        alternatives.append(dict(label=label, actual_total_half_filling_energy=actual,
            universal_Gram_lower=lower, non_Clifford_slack=actual-lower,
            energy_above_selected=actual-data['value']))
    old = np.diag([.4187253908699568, .0632221974274214, .0632221974274214, .0632221974274214])
    r = np.sqrt(np.einsum('pi,ij,pj->p', data['x'], old, data['x']))
    old_gradient = cfg.stiffness*old-data['total']*np.einsum('p,pi,pj->ij', data['weights']/r, data['x'], data['x'])
    gaps = [dict(role=role, scale=load, mass=band.m,
                 single_particle_gap=(load if role.startswith('neutral') else 1.)*band.m*np.sqrt(data['G'][0, 0]))
            for load, band, _, role in data['parts']]
    return dict(size=data['size'], G=data['G'].tolist(), g0=float(data['G'][0, 0]),
        gs=float(data['G'][1, 1]), energy_per_site=data['value'],
        full10_gradient=data['derivative'].tolist(),
        full10_gradient_max=float(np.max(np.abs(data['derivative']))),
        quadrature_order=data['quadrature_order'],
        full10_Hessian_eigenvalues=np.linalg.eigvalsh(data['h10']).tolist(),
        cubic_isotropy='Unique strongly convex PSD Gram functional is invariant under cubic reflections/permutations; average any minimizer over that group. Offdiagonals vanish and spatial diagonals agree as outcome, not constraint.',
        global_bound='For traceless Hermitian h, arbitrary occupation E >= -Tr|h|/2 >= -sqrt(Tr h^2)=-2sqrt(xGx). Sum over momentum and species; fixed TOTAL half filling cannot be below unrestricted negative-spectrum filling. A Clifford square-root frame attains equality with exactly two negative states per momentum, including heavy sign reversal.',
        global_uniform_scope='All 60 real Cartesian traceless Hermitian coordinates; any PSD Gram rank. Strict convexity modulus stiffness proves unique Gram, not unique SU4 representative and not a full quantum phase.',
        rank=rank_report(data['x']), gaps=gaps, competing_configurations=alternatives,
        old_mean_control=dict(gradient_max=float(np.max(np.abs(old_gradient))),
                              excess_energy=objective(data,cfg,old)-data['value']))


def spatial_report(cfg: PhiConfig, size: int):
    data = uniform(cfg, size)
    frame, basis = data['frame'], data['basis']
    elastic = elastic_hessian(frame, basis, cfg.stiffness, 0.)
    bands = []
    for load, band, x, _ in data['parts']:
        h = np.einsum('pm,mij->pij', x, frame)
        values, vectors = np.linalg.eigh(h)
        bands.append((load, dict(sites=data['sites'], x=x, values=values, vectors=vectors)))
    blocks = []
    for q in data['sites']:
        sea = sum(load*fourier_sea_hessian(part,q,size,basis) for load,part in bands)
        lap = float(4*np.sum(np.sin(np.pi*q/size)**2))
        block = elastic+sea+cfg.alignment*lap*np.eye(60)
        eig = np.linalg.eigvalsh(block)
        blocks.append(dict(q=q.tolist(), laplacian=lap, minimum=float(eig[0]),
                          no_alignment_minimum=float(np.linalg.eigvalsh(elastic+sea)[0]),
                          eigenvalues=eig.tolist()))
    h0 = elastic+sum(load*fourier_sea_hessian(part,np.zeros(3,dtype=int),size,basis) for load,part in bands)
    normal = null_space(data['orbit'].T)
    eig = np.linalg.eigvalsh(normal.conj().T@h0@normal)
    return dict(size=size, G=data['G'].tolist(), domain='60 traceless Cartesian coordinates per site; no trace mode introduced',
        orbit_dimension=data['orbit'].shape[1], orbit_residual=float(np.linalg.norm(h0@data['orbit'])),
        normal_q0_eigenvalues=eig.tolist(), normal_gap=float(eig[0]),
        nonzero_q_minimum=min(b['minimum'] for b in blocks if any(b['q'])),
        alignment=cfg.alignment, Fourier_blocks=blocks,
        scope='Exact finite periodic filled seas and original alignment at U=identity; joint Phi/SU4 Hessian still needs gauge-horizontal colour/link completion.')


def joint_native_block(data, cfg: PhiConfig, q: np.ndarray) -> dict[str,Any]:
    """Actual 60 Phi + 45 SU4 link static block, including both contacts.

    Link coordinates sit at their source vertex. This is NOT an EM photon
    replacement: generators act on the same four orbital colours as Phi.
    """
    size = data['size']; sites = data['sites']; volume = len(sites)
    theta = 2*np.pi*sites/size
    qtheta = 2*np.pi*np.asarray(q)/size
    target = np.ravel_multi_index(((sites+q)%size).T,(size,)*3)
    frame,basis = data['frame'],data['basis']
    generators = basis/np.sqrt(8)
    orbit = np.asarray([1j*(t@frame-frame@t) for t in generators])
    orbit_columns = np.einsum('tmij,aji->mat',orbit,basis).real.reshape(60,15)/4
    gauge = np.zeros((105,15),complex)
    gauge[:60] = orbit_columns
    align_vertex = np.zeros((180,105),complex)
    for axis in range(3):
        difference = 1-np.exp(1j*qtheta[axis])
        gauge[60+15*axis:75+15*axis] = difference*np.eye(15)
        align_vertex[60*axis:60*(axis+1),:60] = difference*np.eye(60)
        align_vertex[60*axis:60*(axis+1),60+15*axis:75+15*axis] = -orbit_columns
    curl = np.zeros((45,105),complex)
    for face,(i,j) in enumerate(((0,1),(0,2),(1,2))):
        curl[face*15:(face+1)*15,60+i*15:75+i*15] = (1-np.exp(1j*qtheta[j]))*np.eye(15)
        curl[face*15:(face+1)*15,60+j*15:75+j*15] = (np.exp(1j*qtheta[i])-1)*np.eye(15)
    alignment = cfg.alignment*(align_vertex.conj().T@align_vertex)
    colour = .7*cfg.Lambda**1.25/2*(curl.conj().T@curl)
    bosonic = alignment+colour
    bosonic[:60,:60] += elastic_hessian(frame,basis,cfg.stiffness,0.)
    sea = np.zeros((105,105),complex)
    contacts = np.zeros_like(sea)
    vertex_ward = []
    quantum_metric = np.zeros_like(sea)
    for load,band,x,role in data['parts']:
        h = np.einsum('pm,mij->pij',x,frame)
        values,vectors = np.linalg.eigh(h)
        sign = -1 if role.startswith('neutral') else 1
        occupied = slice(2,4) if sign < 0 else slice(0,2)
        empty = slice(0,2) if sign < 0 else slice(2,4)
        occupied_vectors = vectors[:,:,occupied]
        projector = occupied_vectors@occupied_vectors.conj().transpose(0,2,1)
        vertices = np.zeros((105,volume,4,4),complex)
        vertices[:60] = np.einsum('pm,aij->mapij',(x+x[target])/2,basis).reshape(60,volume,4,4)
        contact = np.zeros_like(sea)
        _,paths = scalar_paths(band)
        for displacement,scalars in paths:
            axis = int(np.flatnonzero(displacement)[0]); length = int(displacement[axis])
            F = np.einsum('m,mij->ij',scalars,frame)
            insertion = sum(np.exp(1j*qtheta[axis]*j) for j in range(length))
            phase = np.exp(1j*theta@displacement)
            reverse_phase = np.exp(-1j*(theta+qtheta)@displacement)
            ft = (F@generators+generators@F)/2
            fdt = (F.conj().T@generators+generators@F.conj().T)/2
            vertices[60+15*axis:75+15*axis] += insertion*1j*(
                phase[None,:,None,None]*ft[:,None]-reverse_phase[None,:,None,None]*fdt[:,None])
            rho_forward = np.einsum('p,pij->ij',phase,projector)/volume
            rho_reverse = np.einsum('p,pij->ij',phase.conj(),projector)/volume
            ab = generators[:,None]@generators[None,:]
            ba = generators[None,:]@generators[:,None]
            mf = -.5*length*(ab+ba)
            mr = mf.copy()
            for j in range(length):
                for k in range(j+1,length):
                    z = np.exp(1j*qtheta[axis]*(k-j))
                    mf -= ab*z+ba*z.conjugate()
                    mr -= ba*z+ab*z.conjugate()
            forward = (F@mf+mf@F)/2
            reverse = (mr@F.conj().T+F.conj().T@mr)/2
            sl = slice(60+15*axis,75+15*axis)
            contact[sl,sl] += np.einsum('ij,abji->ab',rho_forward,forward)+np.einsum('ij,abji->ab',rho_reverse,reverse)
            delta_f = (scalars[:,None,None,None]*basis[None]).reshape(60,4,4)
            z = np.exp(-1j*qtheta@displacement)
            mixed_forward = .5j*insertion*(delta_f[:,None]@generators[None]+z*(generators[None]@delta_f[:,None]))
            delta_fd = (scalars.conj()[:,None,None,None]*basis[None]).reshape(60,4,4)
            mixed_reverse = -.5j*insertion*(generators[None]@delta_fd[:,None]+z*(delta_fd[:,None]@generators[None]))
            mixed = np.einsum('ij,abji->ab',rho_forward,mixed_forward)+np.einsum('ij,abji->ab',rho_reverse,mixed_reverse)
            contact[:60,sl] += mixed
            contact[sl,:60] += mixed.conj().T
        bubble = np.zeros_like(sea)
        metric = np.zeros_like(sea)
        for target_sector,source_sector in ((empty,occupied),(occupied,empty)):
            tv = vectors[target,:,target_sector]; sv = vectors[:,:,source_sector]
            features = np.einsum('pie,apij,pjo->apeo',tv.conj(),vertices,sv)
            denominator = np.abs(values[target,target_sector][:,:,None]-values[:,source_sector][:,None,:])
            weighted = (features/np.sqrt(denominator)[None]).reshape(105,-1)
            bubble -= weighted.conj()@weighted.T/volume
            metric_features = (features/denominator[None]).reshape(105,-1)
            metric += .5*(metric_features.conj()@metric_features.T)/volume
        sea += load*(bubble+sign*contact)
        contacts += load*sign*contact
        quantum_metric += (1 if sign < 0 else load)*metric
        actual = np.einsum('apij,at->tpij',vertices,gauge)
        expected = np.asarray([1j*(t@h-h[target]@t) for t in generators])
        vertex_ward.append(dict(role=role,relative=float(np.linalg.norm(actual-expected)/(1+np.linalg.norm(expected))),
                                absolute=float(np.max(np.abs(actual-expected)))))
    full = (bosonic+sea+bosonic.conj().T+sea.conj().T)/2
    normal = null_space(gauge.conj().T)
    physical = normal.conj().T@full@normal
    kinetic = normal.conj().T@np.diag([cfg.chi]*60+[.6*cfg.chi]*45)@normal
    kv,ku = np.linalg.eigh(kinetic)
    kr = (ku*np.sqrt(kv))@ku.conj().T
    frequencies2 = np.linalg.eigvalsh(kr@physical@kr)
    return dict(q=q.tolist(),full=full,sea=sea,bosonic=bosonic,normal=normal,
        principal_cometric=kinetic,physical=physical,frequencies_squared=frequencies2,
        gauge=gauge,alignment_unit=align_vertex.conj().T@align_vertex,
        colour_unit=curl.conj().T@curl,quantum_metric=quantum_metric,
        vertex_Ward=vertex_ward,gauge_Hessian_residual=float(np.linalg.norm(full@gauge)),
        omitted_contact_gauge_defect=float(np.linalg.norm((full-contacts)@gauge)),
        orbit_Gram=orbit_columns.T@orbit_columns)


def joint_native_report(cfg: PhiConfig, size: int) -> dict[str,Any]:
    data = uniform(cfg,size)
    rows = []
    for q in data['sites']:
        block = joint_native_block(data,cfg,q)
        rows.append(dict(q=block['q'],physical_dimension=block['normal'].shape[1],
            vertex_Ward=block['vertex_Ward'],gauge_Hessian_residual=block['gauge_Hessian_residual'],
            physical_stiffness_eigenvalues=np.linalg.eigvalsh(block['physical']).tolist(),
            principal_cometric_eigenvalues=np.linalg.eigvalsh(block['principal_cometric']).tolist(),
            actual_coupled_frequencies_squared=block['frequencies_squared'].tolist(),
            omitted_contact_gauge_defect=block['omitted_contact_gauge_defect'],
            full_sea_quantum_metric_trace=float(np.trace(block['quantum_metric']).real)))
    return dict(size=size,V=size**3,G=data['G'].tolist(),blocks=rows,
        domain='Actual60V Cartesian Phi +45V SU4 links, localGauss removes15V =>90V physicalnormal coordinates',
        original_primitive_coefficients=asdict(cfg),source_at_import=IMPORT_HASHES,
        scope='Native actual Phi/link matter vertex, both colour-colour and Phi-colour contact, heavy sign and all filled charged species, original alignment/colour curl. Principal cometric at the mean; full nonlinear connection/Berry action remains separately priced.',
        minimum_normal_stiffness=min(min(row['physical_stiffness_eigenvalues']) for row in rows),
        minimum_coupled_frequency_squared=min(min(row['actual_coupled_frequencies_squared']) for row in rows),
        Ward_max=max(r['relative'] for row in rows for r in row['vertex_Ward']),
        gauge_Hessian_max=max(row['gauge_Hessian_residual'] for row in rows))


def response_config(cfg: PhiConfig, data, orders=(4,6)):
    return ResponseConfig(a=cfg.a,m=cfg.active_m,r=cfg.light_r,v=cfg.light_v,
        g0=float(data['G'][0,0]),gs=float(data['G'][1,1]),flavors=1+cfg.auxiliary_count,
        inertia=cfg.inertia,orders=orders,q_values=(.03,.06,.12),omega_values=(.01,.02),
        species=((1,cfg.active_m),(cfg.auxiliary_count,cfg.auxiliary_m)))


def packet_report(cfg: PhiConfig, data, spatial):
    """Actual frozen-sea Gaussian action on Phi, NOT a ground-state claim.

    Gauge-horizontal normal oscillator numbers are exported for the full-Gauss
    chart owner. Simultaneously calculate a normalized 60-coordinate Cartesian
    product Gaussian with the same width, including its real PH matter variance.
    """
    stiffness = spatial['normal_gap']
    width = (cfg.chi/(4*stiffness))**.25  # each real C has variance width^2
    variance = width**2
    c = np.einsum('mij,aji->ma',data['frame'],data['basis']).real/4
    # Exact independent Gaussian Gram-square expectation in R^(4x15).
    g = data['G']; d = 15
    gram2 = float(np.sum(g*g))+variance*(2*(d+5)*float(np.trace(g)))+4*d*(d+5)*variance**2
    kinetic = 60*cfg.chi/(8*variance)
    quartic_excess = .5*cfg.stiffness*(gram2-float(np.sum(g*g)))
    # Alignment: independent site fields, each nearest bond mean ||dPhi||²/4=120 variance.
    alignment_excess = 3*cfg.alignment*60*variance
    ph_variance = 0.
    coefficient_caps = []
    for load, band, x, role in data['parts']:
        onsite, paths = scalar_paths(band)
        # Local fluctuation operator V_mu,a couples p to p' through average endpoints.
        vals, vec = np.linalg.eigh(np.einsum('pm,mij->pij', x,data['frame']))
        project_minus = vec[:,:,:2]@vec[:,:,:2].conj().transpose(0,2,1)
        project_plus = np.eye(4)-project_minus
        # Exact local-site PH variance: double momentum sum, normalized by V².
        acc = 0.
        for i in range(len(x)):
            t = (x[i]+x)/2
            if role.startswith('neutral'):
                psource, ptarget = project_plus[i], project_minus
            else:
                psource, ptarget = project_minus[i], project_plus
            transition = np.einsum('pij,ajk,kl,ali->pa',ptarget,data['basis'],psource,data['basis']).real
            acc += float(np.sum(np.sum(t*t,axis=1)*np.sum(transition,axis=1)))
        # One neutral species scales amplitude by Lambda; n charged species have independent PHs.
        factor = load**2 if role.startswith('neutral') else load
        ph_variance += variance*factor*acc/len(x)**2
        cap = abs(onsite)+2*sum(float(np.linalg.norm(s,1)) for _,s in paths)
        coefficient_caps.append(dict(role=role, load=load, coefficient_l1=cap))
    eta = width*np.sqrt(60/1e-3) # Chebyshev probability 1e-3 at initial time only.
    transfer = mean_field_remainder(response_config(cfg,data))
    return dict(state='normalized Cartesian Phi Gaussian centered at actual minimum, frozen exact filled sea; horizontal oscillator candidate supplied separately to full-Gauss owner',
        width_per_coordinate=width, variance_per_coordinate=variance,
        RMS_Cartesian_distance=np.sqrt(60*variance), initial_tail_radius_1e_minus3=eta,
        exact_Gaussian_Gram_square_expectation=gram2,
        kinetic_per_site=kinetic, quartic_excess_per_site=quartic_excess,
        alignment_excess_per_site=alignment_excess,
        exact_PH_matter_variance_per_site=ph_variance,
        actual_G_centre=data['G'].tolist(),finite_box_size=data['size'],
        frozen_sea_Duhamel_PH_residual_per_sqrt_site=np.sqrt(ph_variance),
        frozen_sea_PH_action_at_one_normal_period=2*np.pi*np.sqrt(ph_variance/(cfg.chi*stiffness)),
        normal_oscillator_frequency=np.sqrt(cfg.chi*stiffness),
        coefficient_caps=coefficient_caps, fixed_radius_supplier_bound=transfer,
        status='Full 60 Gaussian has SU4 orbit spreading and is NOT a gauge-horizontal full-Gauss quantum vacuum. Actual PH variance, not mere stationary Hessian, discriminates whether frozen sea can transfer. Full-Gauss chart and PH dressing must price these numbers; classical mean alone is insufficient.')


def mp_uniform_coefficients(band: RotationBand, size: int):
    onsite,paths = scalar_paths(band)
    rows = []
    for site in periodic_grid(size):
        row = [mp.mpf(float(onsite)),mp.mpf(0),mp.mpf(0),mp.mpf(0)]
        for displacement,scalars in paths:
            phase = mp.exp(2*mp.pi*mp.j*int(site@displacement)/size)
            for mu in range(4):
                s = mp.mpc(float(scalars[mu].real),float(scalars[mu].imag))
                row[mu] += 2*mp.re(s*phase)
        rows.append(row)
    return rows


def high_precision_heavy_mean(band: RotationBand, size: int, seed: np.ndarray) -> dict[str,Any]:
    """Refine the SAME weighted Gram equations using actual scalar-path bytes.

    A binary64 stationary estimate cannot centre a 1e-93 packet in a 1e245
    potential. Decimal coordinates, not a falsely exact float, define this state.
    """
    with mp.workdps(220):
        rows = mp_uniform_coefficients(band,size)
        squares = [(r[0]**2,sum(z*z for z in r[1:])) for r in rows]
        kappa = 2/mp.mpf(band.a)
        g0,gs = mp.mpf(float(seed[0,0])),mp.mpf(float(seed[1,1]))
        for _ in range(12):
            radii = [mp.sqrt(g0*m2+gs*k2) for m2,k2 in squares]
            f0 = kappa*g0-sum(m2/r for (m2,_),r in zip(squares,radii))/len(rows)
            fs = 3*kappa*gs-sum(k2/r for (_,k2),r in zip(squares,radii))/len(rows)
            if max(abs(f0),abs(fs)) < mp.mpf('1e-200'):
                break
            h00 = kappa+sum(m2*m2/(2*r**3) for (m2,_),r in zip(squares,radii))/len(rows)
            hss = 3*kappa+sum(k2*k2/(2*r**3) for (_,k2),r in zip(squares,radii))/len(rows)
            cross = sum(m2*k2/(2*r**3) for (m2,k2),r in zip(squares,radii))/len(rows)
            step = mp.lu_solve(mp.matrix([[h00,cross],[cross,hss]]),mp.matrix([f0,fs]))
            g0 -= step[0]; gs -= step[1]
        residual = max(abs(f0),abs(fs))
        if residual >= mp.mpf('1e-200'):
            raise ArithmeticError('High precision actual-coupling mean did not converge')
        encoded = np.diag([float(g0),float(gs),float(gs),float(gs)])
        frame_error = max(abs(mp.mpf(float(np.sqrt(encoded[i,i])))-mp.sqrt(z))
                          for i,z in enumerate((g0,gs,gs,gs)))
        rounded_radii = [mp.sqrt(mp.mpf(float(encoded[0,0]))*m2+
                                mp.mpf(float(encoded[1,1]))*k2) for m2,k2 in squares]
        rounded_f0 = kappa*mp.mpf(float(encoded[0,0]))-sum(
            m2/r for (m2,_),r in zip(squares,rounded_radii))/len(rows)
        rounded_fs = 3*kappa*mp.mpf(float(encoded[1,1]))-sum(
            k2/r for (_,k2),r in zip(squares,rounded_radii))/len(rows)
        rounded_force = 2*mp.sqrt(g0*rounded_f0**2+gs*rounded_fs**2/3)
        return dict(G=encoded.tolist(),g0_decimal=mp.nstr(g0,220),gs_decimal=mp.nstr(gs,220),
            frame_scales_decimal=[mp.nstr(mp.sqrt(z),220) for z in (g0,gs,gs,gs)],
            normalized_gradient_decimal=mp.nstr(residual,220),
            stationary_Gram_error_upper=float(2*residual/kappa),
            float_consumer_frame_operator_error=float(frame_error),
            rounded_centre_normalized_force=float(rounded_force),
            precision=220,actual_coefficient_source='scalar_paths(RotationBand), exact binary64 coefficient values continued to 220 decimal places; exact periodic phases')


def orbit_categories():
    """Simultaneous adjoint Clifford eigenvalues, std Tr(Ta Tb)=delta/2."""
    return ((0,3,1),(1,2,3),(1,1,3),(0,2,3),(1,0,1),(0,1,3),(1,3,1))


def leading_fast_frequencies(g0: float, gs: float, q: np.ndarray, size: int):
    lap = float(4*np.sum(np.sin(np.pi*np.asarray(q)/size)**2))
    result = [np.sqrt(.2*lap)]*45 if lap else []
    for b0,bs,multiplicity in orbit_categories():
        r = .5*(b0*g0+bs*gs)
        result += [np.sqrt(.12*r+.21*lap)]*(2*multiplicity)
        result += [np.sqrt(.12*r+.2*lap)]*multiplicity
    return np.sort(result)


def joint_ground_covariance(block) -> dict[str,Any]:
    """Actual normal Gaussian covariance in the full principal cometric."""
    kinetic = block['principal_cometric']
    physical = block['physical']
    kv,ku = np.linalg.eigh(kinetic)
    kr = (ku*np.sqrt(kv))@ku.conj().T
    kir = (ku/np.sqrt(kv))@ku.conj().T
    ov,ou = np.linalg.eigh(kr@physical@kr)
    if min(ov) <= 0:
        raise ArithmeticError('No normalized joint Gaussian at an unstable actual physical Hessian')
    root = (ou*np.sqrt(ov))@ou.conj().T
    inverse_root = (ou/np.sqrt(ov))@ou.conj().T
    position = .5*kr@inverse_root@kr
    momentum = .5*kir@root@kir
    full_position = block['normal']@position@block['normal'].conj().T
    energy = .5*float(np.sqrt(ov).sum())
    variance = (.5*np.trace(physical@position@physical@position)+
                .5*np.trace(kinetic@momentum@kinetic@momentum)-.25*np.trace(physical@kinetic)).real
    # Dropping colour/Coulomb kinetic structure is a physically wrong Gaussian,
    # not merely a differently labelled copy of the same state.
    wrong_kinetic = np.max(kv)*np.eye(len(kv))
    wo,wu = np.linalg.eigh(np.max(kv)*physical)
    wr = (wu*np.sqrt(wo))@wu.conj().T
    wir = (wu/np.sqrt(wo))@wu.conj().T
    wrong_position = .5*np.max(kv)*wir
    wrong_momentum = .5*wr/np.max(kv)
    wrong_variance = (.5*np.trace(physical@wrong_position@physical@wrong_position)+
        .5*np.trace(kinetic@wrong_momentum@kinetic@wrong_momentum)-.25*np.trace(physical@kinetic)).real
    return dict(position=position,momentum=momentum,full_position=full_position,
        energy=energy,variance_signed_roundoff=float(variance),
        normalized_quadratic_variance=abs(float(variance))/(1+energy**2),
        wrong_cometric_centered_variance=float(wrong_variance),
        Phi_coordinate_variance=float(np.trace(full_position[:60,:60]).real),
        colour_link_variance=float(np.trace(full_position[60:,60:]).real))


def natural_gauss_chart(block) -> dict[str,Any]:
    """Canonical 45 Cartesian-normal +45 link coordinates, not a flat metric.

    Gauge-invariant link fluctuation is theta-d R^-1 O^*deltaPhi.
    Transforming BOTH the normal covariance and cometric is essential.
    """
    orbit = block['gauge'][:60]
    difference = block['gauge'][60:]
    normal = null_space(orbit.conj().T)
    inverse = np.linalg.inv(block['orbit_Gram'])
    chart = np.zeros((90,105),complex)
    chart[:45,:60] = normal.conj().T
    chart[45:,:60] = -difference@inverse@orbit.conj().T
    chart[45:,60:] = np.eye(45)
    transform = chart@block['normal']
    kinetic = transform@block['principal_cometric']@transform.conj().T
    chi = float(np.trace(block['principal_cometric']).real/
        np.trace(block['normal'].conj().T@np.diag([1.]*60+[.6]*45)@block['normal']).real)
    expected = np.zeros((90,90),complex)
    expected[:45,:45] = chi*np.eye(45)
    expected[45:,45:] = chi*(.6*np.eye(45)+difference@inverse@difference.conj().T)
    if np.linalg.norm(kinetic-expected) > 1e-10*(1+np.linalg.norm(expected)):
        raise ArithmeticError('Actual canonical Gauss cometric disagrees with native quotient')
    return dict(chart=chart,normal=normal,transform=transform,kinetic=kinetic,
        gauge_annihilation=float(np.linalg.norm(chart@block['gauge'])),
        cometric_identity_error=float(np.linalg.norm(kinetic-expected)),
        flat_colour_omission_norm=float(np.linalg.norm(expected[45:,45:]-.6*chi*np.eye(45))))


def su4_left_frame(theta: np.ndarray) -> dict[str,Any]:
    """Actual left Maurer map and Haar density in exponential link coordinates."""
    from scipy.linalg import expm, expm_frechet
    generators = matrix_unit_basis()/np.sqrt(8)
    X = np.einsum('a,aij->ij',theta,generators)
    U = expm(1j*X)
    commutators = 1j*(X@generators-generators@X)
    adjoint = 2*np.einsum('aij,bji->ab',generators,commutators).real
    augmented = np.zeros((30,30))
    augmented[:15,:15] = adjoint
    augmented[:15,15:] = np.eye(15)
    J = expm(augmented)[:15,15:]
    actual = np.empty((15,15))
    for b,t in enumerate(generators):
        derivative = np.asarray(expm_frechet(1j*X,1j*t,compute_expm=False),dtype=np.complex128)
        actual[:,b] = 2*np.einsum('aij,ji->a',generators,-1j*derivative@U.conj().T).real
    error = float(np.linalg.norm(J-actual))
    if error>1e-12:
        raise ArithmeticError('Literal matrix exponential disagrees with actual Maurer connection')
    determinant = float(np.linalg.det(J))
    if determinant<=0:
        raise ArithmeticError('SU4 exponential chart has no positive Haar density')
    return dict(U=U,J_left=J,left_electric_from_canonical=np.linalg.inv(J.T),
        Haar_density=determinant,Frechet_control_error=error,
        nonidentity_frame_norm=float(np.linalg.norm(J-np.eye(15))))


def colour_link_casimir_control(cfg: PhiConfig, block, volume: int) -> dict[str,Any]:
    """Literal SU4 one-link Laplacian, not an assumed kinetic coefficient.

    Exterior-square matrix elements are the same NH=2 Wilson-pair state used
    by the fixed-heavy Gauss upper. Its pointwise norm is one; Casimir is 5/2.
    """
    from itertools import combinations
    from scipy.linalg import expm
    generators = matrix_unit_basis()/np.sqrt(8)
    wedge = np.zeros((16,6),complex)
    for column,(i,j) in enumerate(combinations(range(4),2)):
        wedge[4*i+j,column] = 1/np.sqrt(2)
        wedge[4*j+i,column] = -1/np.sqrt(2)
    representation = np.asarray([wedge.conj().T@(
        np.kron(t,np.eye(4))+np.kron(np.eye(4),t))@wedge for t in generators])
    casimir = np.einsum('aij,ajk->ik',representation,representation)
    fundamental = sum(t@t for t in generators)
    adjoint = []
    for t in generators:
        commutators = 1j*(t@generators-generators@t)
        adjoint.append(2*np.einsum('aij,bji->ab',generators,commutators).real)
    adjoint_casimir = sum(a.T@a for a in adjoint)
    theta = np.arange(1,16,dtype=float); theta *= .21/np.linalg.norm(theta)
    U = expm(1j*np.einsum('a,aij->ij',theta,generators))
    def exterior(value):
        return wedge.conj().T@np.kron(value,value)@wedge
    d = exterior(U)
    differences = []
    for step in (.04,.02):
        laplacian = np.zeros((6,6),complex)
        for t in generators:
            laplacian -= (exterior(expm(1j*step*t)@U)-2*d+
                exterior(expm(-1j*step*t)@U))/step**2
        differences.append(laplacian)
    richardson = (4*differences[1]-differences[0])/3
    expected = casimir@d
    curve_error = float(np.linalg.norm(richardson-expected))
    if curve_error>1e-6 or np.max(np.abs(casimir-2.5*np.eye(6)))>1e-12:
        raise ArithmeticError('Literal one-link exterior-square electric Casimir failed')
    chart = natural_gauss_chart(block)
    primitive_cometric = float(np.trace(chart['kinetic'][45:,45:]).real/(45*cfg.chi))
    Hamiltonian_factor = primitive_cometric/2
    if abs(Hamiltonian_factor-.3)>1e-12:
        raise ArithmeticError('Actual canonical LAW05 colour Hamiltonian coefficient is not .3chi E4^2')
    energy = Hamiltonian_factor*float(np.trace(casimir).real/6)
    return dict(generator_Gram_error=float(np.max(np.abs(
        np.einsum('aij,bji->ab',generators,generators)-np.eye(15)/2))),
        fundamental_Casimir_eigenvalues=np.linalg.eigvalsh(fundamental).tolist(),
        exterior_square_Casimir_eigenvalues=np.linalg.eigvalsh(casimir).tolist(),
        adjoint_Casimir_eigenvalues=np.linalg.eigvalsh(adjoint_casimir).tolist(),
        actual_group_curve_Richardson_error=curve_error,
        half_step_error_ratio=float(np.linalg.norm(differences[0]-expected)/
            np.linalg.norm(differences[1]-expected)),
        Wilson_pair_pointwise_norm=float(np.trace(d.conj().T@d).real/6),
        actual_native_cometric_factor=primitive_cometric,
        actual_Hamiltonian_factor=Hamiltonian_factor,
        Wilson_pair_energy_per_chi=energy,
        filled_heavy_dimer_upper_extra_per_V_chi=energy/2,
        actual_filled_heavy_dimer_upper_extra=energy*volume*cfg.chi/2,
        retired_half_colour_energy_per_chi=energy/2,
        scope='Actual one-link left-E curve and wedge2 state, canonical .3chi E4^2. HALFcolour .15chi E4^2 is a distinct rejected original-law control.')


def local_gauss_kinetic(frame, fields, links, normal_momenta, link_electric, matter_charge, chi) -> dict[str,Any]:
    """Exact principal quadratic form in a fixed local normal slice.

    matter_charge is rho in O(Phi)^*pPhi+div_U E=rho, with E the LEFT
    electric generators. Incoming fields transport by U^* E U. Haar/ordering
    and Berry terms are not hidden in this principal-form routine.
    """
    basis = matrix_unit_basis()
    generators = basis/np.sqrt(8)
    def orbit(phi):
        matrices = 1j*(generators[:,None]@phi[None]-phi[None]@generators[:,None])
        return np.einsum('tmij,aji->mat',matrices,basis).real.reshape(60,15)/4
    o0 = orbit(frame)
    normal = null_space(o0.T)
    shape = fields.shape[:3]
    if fields.shape != (*shape,4,4,4) or links.shape != (*shape,3,4,4):
        raise ValueError('Fields must be Phi[x,y,z,mu,4,4] and links U[x,y,z,axis,4,4]')
    if normal_momenta.shape != (*shape,45) or link_electric.shape != (*shape,3,15):
        raise ValueError('Actual 45 normal and 3*15 left-electric momenta are required')
    if matter_charge.shape != (*shape,15):
        raise ValueError('Every site requires an explicit matter-charge vector')
    angular_norm = 0.
    minimum_slice = float('inf')
    for site in np.ndindex(shape):
        current = orbit(fields[site])
        a = o0.T@current
        minimum_slice = min(minimum_slice,float(np.linalg.svd(a,compute_uv=False)[-1]))
        divergence = np.sum(link_electric[site],axis=0).astype(complex)
        for axis in range(3):
            incoming = list(site); incoming[axis] = (incoming[axis]-1)%shape[axis]
            incoming_site = tuple(incoming)
            u = links[incoming_site+(axis,)]
            e = np.einsum('a,aij->ij',link_electric[incoming_site+(axis,)],generators)
            transported = u.conj().T@e@u
            divergence -= 2*np.einsum('aij,ji->a',generators,transported)
        rhs = matter_charge[site]-divergence-current.T@normal@normal_momenta[site]
        angular = o0@np.linalg.solve(a.T,rhs)
        angular_norm += float(np.vdot(angular,angular).real)
    normal_norm = float(np.vdot(normal_momenta,normal_momenta).real)
    electric_norm = float(np.vdot(link_electric,link_electric).real)
    return dict(kinetic=.5*chi*(normal_norm+angular_norm+.6*electric_norm),
        angular_Gauss_kinetic=.5*chi*angular_norm,
        frozen_angular_wrong_kinetic=.5*chi*(normal_norm+.6*electric_norm),
        slice_minimum_singular_value=minimum_slice)


def canonical_geometry_report(cfg: PhiConfig, size: int, frame: np.ndarray, blocks) -> dict[str,Any]:
    """Native quotient, spatial Fourier and nonlinear chart positive controls."""
    shape = (size,)*3
    fields = np.broadcast_to(frame,(*shape,4,4,4)).copy()
    links = np.broadcast_to(np.eye(4),(*shape,3,4,4)).astype(complex).copy()
    p = np.zeros((*shape,45),complex)
    electric = np.zeros((*shape,3,15),complex)
    charge = np.zeros((*shape,15),complex)
    rows = []
    for block in blocks:
        chart = natural_gauss_chart(block)
        ground = joint_ground_covariance(block)
        covariance = np.asarray(chart['chart']@ground['full_position']@chart['chart'].conj().T,dtype=np.complex128)
        kv,ku = np.linalg.eigh(block['principal_cometric']/cfg.chi)
        kr = (ku*np.sqrt(kv))@ku.conj().T
        leading = block['normal'].conj().T@(
            .2*block['alignment_unit']+.35*block['colour_unit'])@block['normal']
        frequency2 = np.linalg.eigvalsh(kr@leading@kr)
        selected = frequency2 if any(block['q']) else frequency2[45:]
        if min(selected)<=0:
            raise ArithmeticError('Canonical actual leading fast operator is not positive')
        metric = gram(frame)
        analytic = leading_fast_frequencies(float(metric[0,0]),float(metric[1,1]),
            np.asarray(block['q']),size)
        frequency_error = float(np.max(np.abs(np.sqrt(selected)-analytic)))
        if frequency_error>1e-10:
            raise ArithmeticError('Canonical fast spectrum formula fails its actual full-normal matrix')
        rows.append(dict(q=block['q'],gauge_annihilation=chart['gauge_annihilation'],
            cometric_identity_error=chart['cometric_identity_error'],
            actual_leading_fast_frequency_max_error=frequency_error,
            leading_uniform_slow_null_residual=float(np.max(np.abs(frequency2[:45]))) if not any(block['q']) else None,
            flat_colour_omission_norm=chart['flat_colour_omission_norm'],
            canonical_coordinate_variances=np.diag(covariance).real.tolist(),
            normalized_quadratic_variance=ground['normalized_quadratic_variance'],
            wrong_cometric_centered_variance=ground['wrong_cometric_centered_variance']))
    nonzero = next(b for b in blocks if any(b['q']))
    q = np.asarray(nonzero['q'])
    phase = np.exp(2j*np.pi*(periodic_grid(size)@q)/size).reshape(shape)/np.sqrt(size**3)
    electric[...,0,0] = phase
    actual = local_gauss_kinetic(frame,fields,links,p,electric,charge,cfg.chi)
    chart = natural_gauss_chart(nonzero)
    vector = np.zeros(90,complex); vector[45]=1
    expected = .5*float(np.vdot(vector,chart['kinetic']@vector).real)
    if abs(actual['kinetic']-expected)>1e-10*(1+abs(expected)):
        raise ArithmeticError('Nonlinear local Gauss form fails its actual Fourier control')
    rng = np.random.default_rng(222427)
    displacement = rng.normal(size=45); displacement *= .001/np.linalg.norm(displacement)
    fields[0,0,0] += np.einsum('ma,aij->mij',
        (chart['normal']@displacement).real.reshape(4,15),matrix_unit_basis())
    angles = np.zeros(15); angles[1] = .002
    connection = su4_left_frame(angles)
    links[0,0,0,0] = connection['U']
    electric[0,0,0,0] = connection['left_electric_from_canonical']@electric[0,0,0,0]
    nonlinear = local_gauss_kinetic(frame,fields,links,p,electric,charge,cfg.chi)
    return dict(blocks=rows,Fourier_kinetic=actual,Fourier_expected=expected,
        nonlinear_kinetic=nonlinear,
        Maurer_Haar_control={key:connection[key] for key in
            ('Haar_density','Frechet_control_error','nonidentity_frame_norm')},
        nonlinear_change=nonlinear['kinetic']-actual['kinetic'],
        scope='Exact canonical principal form and harmonic Gaussian; nonlinear physical state/action requires the separately displayed measure, boundary and heavy-fibre estimates.')


def clifford_cartesian_basis():
    """Exact signed Pauli monomials, an orthogonal change of existing basis."""
    from itertools import combinations
    result = []
    for degree in range(1,5):
        for indices in combinations(range(4),degree):
            value = np.eye(4,dtype=complex)
            for index in indices:
                value = value@SELECTED[index]
            result.append((1j**(degree*(degree-1)//2))*value)
    basis = np.asarray(result)
    error = np.max(np.abs(np.einsum('aij,bji->ab',basis,basis)-4*np.eye(15)))
    if error != 0:
        raise ArithmeticError('Selected Clifford monomials are not exact Cartesian units')
    return basis


def precise_uniform_normal_packet(cfg: PhiConfig, size: int, quantum_mean, precision=200) -> dict[str,Any]:
    """Actual heavy-sea Hessian and normalized 45-dimensional slow Gaussian.

    This computes the entire non-Clifford normal Hessian, not only the two
    invariant Gram variables. Decimal packet coefficients refer to scaled
    coordinates y=Lambda^(3/8) z; the centre retains the separate full precision.
    """
    basis = clifford_cartesian_basis()
    tensors = np.rint(np.einsum('nij,ajk,mkl,bli->namb',
        SELECTED,basis,SELECTED,basis).real/4).astype(int)
    commutators = np.rint(np.einsum('amij,bji->mab',
        1j*(basis[:,None]@SELECTED[None]-SELECTED[None]@basis[:,None]),basis).real/4).astype(int)
    with mp.workdps(precision):
        scales = [mp.sqrt(mp.mpf(quantum_mean[k])) for k in
            ('g0_decimal','gs_decimal','gs_decimal','gs_decimal')]
        orbit = mp.zeros(60,15)
        used = set()
        vectors = []
        for a in range(15):
            support = []
            for mu in range(4):
                for b in range(15):
                    if commutators[mu,a,b]:
                        index = 15*mu+b
                        orbit[index,a] = int(commutators[mu,a,b])*scales[mu]/mp.sqrt(8)
                        support.append(index)
                        if index in used:
                            raise ArithmeticError('Clifford gauge columns unexpectedly overlap')
                        used.add(index)
            column = [orbit[index,a] for index in support]
            length = mp.sqrt(sum(z*z for z in column))
            unit = [z/length for z in column]
            if len(support)>1:
                householder = unit.copy(); householder[0] -= 1
                norm = sum(z*z for z in householder)
                for j in range(1,len(support)):
                    vector = mp.zeros(60,1)
                    for i,index in enumerate(support):
                        vector[index] = (1 if i==j else 0)-2*householder[i]*householder[j]/norm
                    vectors.append(vector)
        for index in range(60):
            if index not in used:
                vector = mp.zeros(60,1); vector[index] = 1; vectors.append(vector)
        normal = mp.matrix([[vector[i] for vector in vectors] for i in range(60)])
        if len(vectors)!=45:
            raise ArithmeticError('Actual full Cartesian normal dimension is not 45')
        rows = mp_uniform_coefficients(models(cfg)[0][1],size)
        radii = [mp.sqrt(sum((s*x)**2 for s,x in zip(scales,row))) for row in rows]
        volume = len(rows)
        second = {}
        fourth = {}
        from itertools import product
        for indices in product(range(4),repeat=2):
            second[indices] = sum(row[indices[0]]*row[indices[1]]/radius
                for row,radius in zip(rows,radii))/volume
        for indices in product(range(4),repeat=4):
            fourth[indices] = sum(mp.fprod(row[i] for i in indices)/radius**3
                for row,radius in zip(rows,radii))/volume
        stiffness = 2/mp.mpf(cfg.a)
        Hessian = mp.zeros(60)
        for mu,a,nu,b in product(range(4),range(15),range(4),range(15)):
            value = -(second[mu,nu] if a==b else 0)
            for i,j in product(range(4),repeat=2):
                coefficient = int(tensors[i,a,j,b])
                if coefficient:
                    value += coefficient*scales[i]*scales[j]*fourth[mu,nu,i,j]
            value += 2*stiffness*(
                (scales[a]**2 if mu==nu and a==b and a<4 else 0)+
                (scales[nu]*scales[mu] if a==nu and b==mu else 0)+
                (scales[mu]**2 if mu==nu and a==b else 0))
            Hessian[15*mu+a,15*nu+b] = value
        physical = normal.T*Hessian*normal
        eigenvalues,eigenvectors = mp.eigsy(physical)
        residual = max(abs(z) for z in physical*eigenvectors-
            eigenvectors*mp.diag(eigenvalues))
        orth_error = max(abs(z) for z in eigenvectors.T*eigenvectors-mp.eye(45))
        if eigenvalues[0]<=0 or residual>mp.mpf(10)**(-precision+30):
            raise ArithmeticError('Actual high-precision full normal Hessian failed')
        covariance = eigenvectors*mp.diag([1/(2*mp.sqrt(z)) for z in eigenvalues])*eigenvectors.T
        return dict(precision=precision,dimension=45,basis='15 exact signed Clifford Pauli monomials, Tr Ba Bb=4 delta; coordinate change only',
            minimum_stiffness=mp.nstr(eigenvalues[0],precision),
            maximum_stiffness=mp.nstr(eigenvalues[len(eigenvalues)-1],precision),
            eigenvalues=[mp.nstr(z,precision) for z in eigenvalues],
            eigen_residual_decimal=mp.nstr(residual,precision),
            eigen_orthogonality_error_decimal=mp.nstr(orth_error,precision),
            normal_orthogonality_error_decimal=mp.nstr(max(abs(z) for z in normal.T*normal-mp.eye(45)),precision),
            normal_orbit_annihilation_decimal=mp.nstr(max(abs(z) for z in normal.T*orbit),precision),
            normal_columns_decimal=[[mp.nstr(normal[i,j],precision) for j in range(45)] for i in range(60)],
            scaled_covariance_decimal=[[mp.nstr(covariance[i,j],precision) for j in range(45)] for i in range(45)],
            normalized_wavefunction='det(2pi Sigma)^(-1/4) exp(-y^T Sigma^-1 y/4), y=Lambda^(3/8) z_uniform; full precision centre independent of shape precision',
            normal_basis_definition='Physical basis is the exact disjoint-support Householder nullspace of O0 at the full literal decimal centre, using the displayed signed Clifford basis. normal_columns_decimal is a diagnostic approximation ONLY, never a replacement physical centre or rounded orbit. The displayed finite-decimal covariance defines the shape in that exact basis; shape error is priced separately.',
            scope='Actual normalized harmonic slow state. Joint fast, Haar/Gauss and boundary comparison are separate, not implicit.')


def colour_zero_point_quantum_mean(cfg: PhiConfig, size: int, exponent: int = 816) -> dict[str,Any]:
    """Precision-resolved all-sea mean plus actual leading full-Gauss fast ZPE.

    The fast normal spectrum is derived from native gauge geometry and checked
    against its 90-coordinate numerical block; no Maxwell or speed fitting.
    The exact slow/fast prepared-state kinetic remainder is a separate obligation.
    """
    heavy = models(cfg)[0][1]
    p = 2*np.pi*periodic_grid(size)/(size*cfg.a)
    initial = select_gram(coefficients(heavy,p),2/cfg.a,0.)['G']
    seed = high_precision_heavy_mean(heavy,size,initial)
    precision = max(320,int(np.ceil(exponent*np.log10(2)))+120)
    tolerance = mp.mpf(10)**(-precision+40)
    with mp.workdps(precision):
        scale = mp.mpf(2)**exponent
        kappa = 2/mp.mpf(cfg.a)
        profiles = []
        for _,band,role in models(cfg):
            rows = mp_uniform_coefficients(band,size)
            profiles.append((scale if role.startswith('neutral') else
                mp.mpf(1 if role.startswith('active') else cfg.auxiliary_count),
                [(r[0]**2,sum(z*z for z in r[1:])) for r in rows]))
        laps = [4*sum(mp.sin(mp.pi*int(z)/size)**2 for z in q) for q in periodic_grid(size)]
        volume = len(laps)
        ratio = scale**(-mp.mpf(5)/8)
        def fast(g0,gs):
            energy = mp.mpf(0); gradient = [mp.mpf(0),mp.mpf(0)]
            Hessian = mp.zeros(2)
            for lap in laps:
                energy += 45*mp.sqrt(lap/5)/2
                for b0,bs,mult in orbit_categories():
                    r = (b0*g0+bs*gs)/2
                    t = mp.mpf(3)*r/25+mp.mpf(21)*lap/100
                    l = mp.mpf(3)*r/25+lap/5
                    energy += mult*(mp.sqrt(t)+mp.sqrt(l)/2)
                    w = [mp.mpf(b0)/2,mp.mpf(bs)/2]
                    for i in range(2):
                        gradient[i] += mult*w[i]*(mp.mpf(3)/(50*mp.sqrt(t))+mp.mpf(3)/(100*mp.sqrt(l)))
                        for j in range(2):
                            Hessian[i,j] -= mult*w[i]*w[j]*(
                                mp.mpf(9)/(2500*t**mp.mpf('1.5'))+
                                mp.mpf(9)/(5000*l**mp.mpf('1.5')))
            return energy/volume,[g/volume for g in gradient],Hessian/volume
        g0,gs = mp.mpf(seed['g0_decimal']),mp.mpf(seed['gs_decimal'])
        for _ in range(8):
            zpe,gradient,joint_hessian = fast(g0,gs)
            derivative = mp.matrix([kappa*g0,3*kappa*gs])+ratio*mp.matrix(gradient)
            Hessian = mp.matrix([[kappa,0],[0,3*kappa]])+ratio*joint_hessian
            for load,squares in profiles:
                radii = [mp.sqrt(g0*m2+gs*k2) for m2,k2 in squares]
                derivative[0] -= load/scale*sum(m2/r for (m2,_),r in zip(squares,radii))/volume
                derivative[1] -= load/scale*sum(k2/r for (_,k2),r in zip(squares,radii))/volume
                for i in range(2):
                    for j in range(2):
                        Hessian[i,j] += load/scale*sum(
                            pair[i]*pair[j]/(2*r**3) for pair,r in zip(squares,radii))/volume
            residual = max(abs(d) for d in derivative)
            if residual < tolerance:
                break
            step = mp.lu_solve(Hessian,derivative)
            g0 -= step[0]; gs -= step[1]
        if residual >= tolerance:
            raise ArithmeticError('Full quantum-ZPE mean did not reach its required precision')
        bare0,bareS = mp.mpf(seed['g0_decimal']),mp.mpf(seed['gs_decimal'])
        force0,forceS = gradient
        return dict(branch='NEW nested fast-Phi/colour BO preparation candidate, not a full quantum phase',
            Lambda_exponent=exponent,Lambda_exact=str(2**exponent),precision=precision,size=size,V=volume,
            all_filled_profiles=[[1,cfg.heavy_m,'neutral heavy minusLambda'],[1,cfg.active_m,'active'],[cfg.auxiliary_count,cfg.auxiliary_m,'charged auxiliary']],
            g0_decimal=mp.nstr(g0,precision),gs_decimal=mp.nstr(gs,precision),
            G_float=np.diag([float(g0),float(gs),float(gs),float(gs)]).tolist(),
            g0_refinement_from_220_digit_seed=mp.nstr(g0-bare0,precision),
            gs_refinement_from_220_digit_seed=mp.nstr(gs-bareS,precision),
            fast_zero_point_coefficient_decimal=mp.nstr(zpe,precision),
            fast_zero_point_gradient_decimal=[mp.nstr(force0,precision),mp.nstr(forceS,precision)],
            fast_zero_point_energy_per_site_decimal=mp.nstr(scale**(mp.mpf(3)/8)*zpe,precision),
            full_normalized_uniform_gradient_decimal=mp.nstr(residual,precision),
            fast_normal_dimension=90*volume-45,slow_uniform_normal_dimension=45,
            physical_total_normal_dimension=90*volume,
            spectrum='Canonical LAW05 colour .3chi E4^2, cometric .6chi. 45 nonuniform Phi modes sqrt(.2lap), plus15 colour directions: 2sqrt(.12r+.21lap)+sqrt(.12r+.2lap); r=.5(sum anticommuting Gram scales). Frequencies multiply Lambda^(3/8).',
            neglected_force_scale='Subleading fast mass term force O(Lambda^1/8), slow zero-point force O(Lambda^1/4), bounded charged/ph excitations O(1). Their actions on the slow width scale Lambda^-1/4, Lambda^-1/8, Lambda^-3/8 respectively; coefficients require explicit pricing before physical-state promotion.',
            prepared_state_debt='Full conditional fast Gaussian, normal-slice connection, Haar curvature, Berry and boundary norms still must be executed/priced. This quantum stationary scalar functional is not silently a full ground state.',
            source_at_import=IMPORT_HASHES)


def fast_zero_point_interval(mean, size: int, digits=200) -> dict[str,Any]:
    """Outward-rounded ZPE interval, retaining exact binary rational endpoints.

    It evaluates the literal decimal packet centre, not a rounded float
    reconstruction. All frequencies are the canonical LAW05 ones.
    """
    from mpmath.ctx_iv import MPIntervalContext
    # mpmath installs interval arithmetic operators dynamically on context types.
    context: Any = MPIntervalContext()
    context.dps = digits
    g0 = context.mpf(mean['g0_decimal'])
    gs = context.mpf(mean['gs_decimal'])
    pi = context.mpf(context.pi)
    result = context.mpf(0)
    for q in periodic_grid(size):
        lap = 4*sum(context.sin(pi*int(z)/size)**2 for z in q)
        result += 45*context.sqrt(lap/5)/2
        for b0,bs,mult in orbit_categories():
            r = (b0*g0+bs*gs)/2
            result += mult*(context.sqrt(3*r/25+21*lap/100)+
                context.sqrt(3*r/25+lap/5)/2)
    result /= size**3
    endpoints = []
    for sign,mantissa,exponent,_ in result._mpi_:
        value = Fraction((-1 if sign else 1)*int(mantissa),1)
        value = value*(2**int(exponent)) if exponent>=0 else value/Fraction(2**(-int(exponent)),1)
        endpoints.append(value)
    encoded = lambda value: dict(numerator=str(value.numerator),denominator=str(value.denominator))
    with mp.workdps(max(digits+30,int(mean['precision']))):
        scalar = mp.mpf(mean['fast_zero_point_coefficient_decimal'])
        lower = mp.mpf(endpoints[0].numerator)/endpoints[0].denominator
        upper = mp.mpf(endpoints[1].numerator)/endpoints[1].denominator
        if not lower<=scalar<=upper:
            raise ArithmeticError('Independent canonical interval does not enclose actual high precision ZPE')
        width = upper-lower
        return dict(working_digits=digits,lower_exact=encoded(endpoints[0]),
            upper_exact=encoded(endpoints[1]),width_decimal=mp.nstr(width,digits),
            method='Independent MPIntervalContext outward sqrt/sin arithmetic of literal centre and exact rational LAW05 coefficients; endpoints are exact fractions',
            scalar_source='fast_zero_point_coefficient_decimal at full mean precision; float energy is neither produced nor consumed')

def joint_gauge_descent_control(frame: np.ndarray) -> dict[str,Any]:
    """Actual two-site link/fundamental matter test of the joint gauge section."""
    from itertools import combinations
    pairs = list(combinations(range(4),2))
    wedge = np.zeros((16,6),complex)
    dual = np.zeros((6,6))
    for j,pair in enumerate(pairs):
        i,k = pair
        wedge[4*i+k,j],wedge[4*k+i,j] = 1/np.sqrt(2),-1/np.sqrt(2)
        complement = next(candidate for candidate in pairs if all(z not in pair for z in candidate))
        sequence = pair+complement
        parity = sum(sequence[a]>sequence[b] for a in range(4) for b in range(a+1,4))
        dual[pairs.index(complement),j] = (-1)**parity
    def representation(u):
        return wedge.conj().T@np.kron(u,u)@wedge
    def state(u):
        return representation(u)@dual/np.sqrt(6)
    identity = np.eye(4,dtype=complex)
    left = 1j*identity
    right = identity
    transformed_link = left@identity@right.conj().T
    original = state(identity)
    transformed = state(transformed_link)
    matter = representation(left)@original@representation(right).T
    error = float(np.linalg.norm(transformed-matter))
    Phi_error = float(np.linalg.norm(left@frame@left.conj().T-frame))
    wrong_Phi_only = float(np.linalg.norm(original-matter))
    fixed_branch_distance = float(np.linalg.norm(transformed_link-identity))
    centres = (1,1j,-1,-1j)
    compatible = []
    for zx in centres:
        for zy in centres:
            body = np.conj(zx)*transformed_link*zy
            if np.linalg.norm(body-identity)<1e-12:
                lifted = zx**2*zy**2*original
                compatible.append(float(np.linalg.norm(lifted-transformed)))
    gx = su4_left_frame(np.arange(15)*.003)['U']
    gy = su4_left_frame(np.arange(15)[::-1]*-.002)['U']
    generic_error = float(np.linalg.norm(state(gx@gy.conj().T)-
        representation(gx)@original@representation(gy).T))
    norm = float(np.linalg.norm(original)**2)
    cover_norm = sum(float(np.linalg.norm(transformed)**2) for _ in compatible)/4
    wrong_local_cover_norm = sum(float(np.linalg.norm(transformed)**2) for _ in compatible)/16
    if (error>1e-12 or generic_error>1e-12 or Phi_error>1e-12 or
            len(compatible)!=4 or max(compatible)>1e-12 or
            wrong_Phi_only<1 or fixed_branch_distance<1 or abs(cover_norm-1)>1e-12):
        raise ArithmeticError('Actual linked matter section does not descend/isometrically cover the joint gauge chart')
    return dict(two_site_matter='Two primitive fundamental fermions per endpoint, exterior-square Wilson singlet D_wedge2(U) epsilon/sqrt6; totalN=4',
        local_centre='Vx=iI,Vy=I leaves Phi unchanged, moves actual U=I to iI and acts -1 on two-fermion x matter',
        Phi_adjoint_change=Phi_error,actual_link_change=fixed_branch_distance,
        actual_matter_link_covariance_error=error,generic_SU4_covariance_error=generic_error,
        wrong_Phi_only_wave_error=wrong_Phi_only,wrong_fixed_branch_excludes_physical_link=True,
        compatible_centre_frame_pairs=len(compatible),branch_state_disagreement=max(compatible),
        pointwise_matter_norm_squared=norm,finite_cover_quotient_norm_squared=cover_norm,
        wrong_independent_local_centre_cover_norm_squared=wrong_local_cover_norm,
        stabilizer='Connected links force zx=zy on every edge; only diagonalZ4 remains. It acts by z^totalN, not independent local occupation characters.')


def noncompact_EM_Gauss_control(size: int) -> dict[str,Any]:
    """Execute the real-gradient Gauss solution on the actual periodic box."""
    sites = periodic_grid(size)
    volume = len(sites)
    index = {tuple(site):i for i,site in enumerate(sites)}
    divergence = np.zeros((volume,3*volume))
    for i,site in enumerate(sites):
        for axis in range(3):
            target = site.copy()
            target[axis] = (target[axis]+1)%size
            divergence[i,3*i+axis] = 1
            divergence[index[tuple(target)],3*i+axis] = -1
    laplacian = divergence@divergence.T
    values,vectors = np.linalg.eigh(laplacian)
    expected_gap = 4*np.sin(np.pi/size)**2
    positive = values>expected_gap/2
    if int(np.count_nonzero(positive))!=volume-1:
        raise ArithmeticError('Real Gauss Laplacian has the wrong physical rank')
    green = (vectors[:,positive]/values[positive])@vectors[:,positive].T
    density = divergence[:,0]
    longitudinal = divergence.T@(green@density)
    string = np.zeros(3*volume)
    string[0] = 1
    residual = float(np.linalg.norm(divergence@longitudinal-density))
    energy = float(longitudinal@longitudinal)
    coulomb = float(density@green@density)
    if residual>1e-12 or abs(energy-coulomb)>1e-12:
        raise ArithmeticError('Actual real Gauss solution does not match Coulomb quadratic form')
    if float(np.linalg.norm(longitudinal-string))<.1:
        raise ArithmeticError('Real-gradient branch was incorrectly replaced by integer electric strings')
    return dict(Hilbert='L2(R^(2V+1)) after quotienting additive real site gauge transformations; theta_link in R^(3V), not compact U1 Haar',
        longitudinal_solution='Pi_long=D^T(DD^T)^+ rho, sum rho=0; electric Coulomb UEphys/a rho^T(DD^T)^+rho',
        coexact_plus_harmonic_dimension=3*volume-int(np.count_nonzero(positive)),
        actual_positive_Laplacian_gap=float(values[positive][0]),
        analytic_gap=float(expected_gap),Gauss_residual=residual,
        real_longitudinal_energy=energy,Coulomb_quadratic_form=coulomb,
        integer_string_energy=float(string@string),
        real_vs_integer_string_distance=float(np.linalg.norm(longitudinal-string)),
        domain='H2 compact-support product on real orthonormal coordinates; no periodization or projected integer-period lattice is asserted')


def flat_cut_ramp_support_control() -> dict[str,Any]:
    """Actual radial polynomial derivatives and repaired Agmon separation."""
    radius = np.linspace(.25,.5,1025)
    u = 4*radius-1
    derivative = -120*u**2*(1-u)**2
    second = -960*u*(1-u)*(1-2*u)
    first = .25+2.**-20
    first_u = 4*first-1
    first_derivative = -120*first_u**2*(1-first_u)**2
    old_ratio = ((first-.25)/16)/(1/1024)
    repaired_ratio = ((first-.125)/64)/(1/1024)
    if first_derivative==0 or old_ratio>=.001 or repaired_ratio<2:
        raise ArithmeticError('Actual flat-cut support does not discriminate the zero-separation Agmon error')
    if np.max(abs(derivative))>8 or np.max(abs(second))>96:
        raise ArithmeticError('Actual radial flat-cut derivatives exceed the claimed constants')
    return dict(flat_cut_derivative_support_closure=[.25,.5],
        old_ramp_start=.25,old_support_distance_infimum=0.,
        repaired_ramp_start=.125,repaired_support_distance_infimum=.125,
        first_sample_radius=first,actual_first_sample_radial_derivative=first_derivative,
        old_weight_to_claimed_exponent_ratio=old_ratio,
        repaired_weight_to_claimed_exponent_ratio=repaired_ratio,
        observed_dimensionless_gradient_max=float(np.max(abs(derivative))),
        observed_dimensionless_radial_Hessian_max=float(np.max(abs(second))),
        interpretation='Distances in units eta; slopes multiply eta sqrt(lambda amin/(chi M)). Old slope1/16 fails claimed1/1024 exponent on the first derivative shell; repaired slope1/64 with start1/8 supplies at least1/512.')



def joint_prepared_report(cfg: PhiConfig, size=6, exponent=2048, cutoff_bits=512, clock=4096.) -> dict[str,Any]:
    """Source-bound full-normal packet and explicit local-well comparison prices.

    The comparison is to the exact charged/EM Hamiltonian at the displayed
    mean, NOT to RPA or a continuum/Compton phase. The microscopic box is fixed.
    Scalar well ground energy is only an unobservable common reference phase.
    """
    start = time.monotonic()
    centre = global_centre_projection(cfg,size)
    if not centre['global_centre_admissible']:
        raise ArithmeticError('No positive filled-heavy Gauss packet on this odd box')
    quantum_mean = colour_zero_point_quantum_mean(cfg,size,exponent)
    slow = precise_uniform_normal_packet(cfg,size,quantum_mean)
    g = np.asarray(quantum_mean['G_float'])
    frame = np.sqrt(np.diag(g))[:,None,None]*SELECTED
    sites = periodic_grid(size)
    band = models(cfg)[0][1]
    x = coefficients(band,2*np.pi*sites/(size*cfg.a))
    data = dict(size=size,sites=sites,frame=frame,G=g,basis=matrix_unit_basis(),
        parts=[(1.,band,x,'neutral heavy -Lambda')])
    unit = replace(cfg,Lambda=1.)
    coefficients_bound = 0.
    metric_trace = 0.
    Ward = 0.
    rows = []
    zero = None
    for q in sites:
        block = joint_native_block(data,unit,q)
        remainder = block['full']-.2*block['alignment_unit']-.35*block['colour_unit']
        norm = float(np.linalg.norm(remainder,2))
        coefficients_bound = max(coefficients_bound,norm)
        metric_trace += float(np.trace(block['quantum_metric']).real)
        Ward = max(Ward,max(r['relative'] for r in block['vertex_Ward']))
        rows.append(dict(q=q.tolist(),actual_heavy_quartic_remainder_norm=norm,
            actual_heavy_fibre_metric_trace=float(np.trace(block['quantum_metric']).real),
            relative_native_Ward=max(r['relative'] for r in block['vertex_Ward']),
            relative_Hessian_gauge_residual=block['gauge_Hessian_residual']/
                (1+np.linalg.norm(block['full'])*np.linalg.norm(block['gauge'])),
            omitted_contact_gauge_defect=block['omitted_contact_gauge_defect']))
        if not any(q):
            zero = block
    if zero is None:
        raise ArithmeticError('Actual finite box omitted its zero Fourier block')
    normal = natural_gauss_chart(zero)['normal']
    native_eigenvalues = np.linalg.eigvalsh(normal.conj().T@zero['full'][:60,:60]@normal)
    precise_eigenvalues = np.asarray([float(z) for z in slow['eigenvalues']])
    discrepancy = float(np.max(np.abs(native_eigenvalues-precise_eigenvalues)))
    if discrepancy>1e-9*(1+np.max(np.abs(precise_eigenvalues))):
        raise ArithmeticError('Independent high precision normal Hessian disagrees with native vertex/contact')
    control_cfg = replace(cfg,Lambda=float(2**40))
    control_data = uniform(control_cfg,4)
    control_blocks = [joint_native_block(control_data,control_cfg,np.asarray(q))
        for q in ((0,0,0),(1,0,0),(1,2,1))]
    geometry_control = canonical_geometry_report(control_cfg,4,control_data['frame'],control_blocks)
    Casimir_control = colour_link_casimir_control(control_cfg,control_blocks[0],4**3)
    EM_control = noncompact_EM_Gauss_control(size)
    cut_support_control = flat_cut_ramp_support_control()
    joint_descent = joint_gauge_descent_control(control_data['frame'])
    # Actual literal path coefficients determine all operator caps below.
    def path_cap(model):
        onsite,paths = scalar_paths(model)
        return abs(float(onsite))+2*sum(float(np.sum(np.abs(s))) for _,s in paths)
    heavy_cap = path_cap(band)
    light_cap = sum(load*path_cap(model) for load,model,role in models(cfg)
        if not role.startswith('neutral'))
    with mp.workdps(200):
        scale = mp.mpf(2)**exponent
        eta = mp.mpf(2)**(-cutoff_bits)
        chi = scale**(-mp.mpf(1)/2)
        volume = size**3
        dimension = 90*volume
        fast_dimension = dimension-45
        g0,gs = mp.mpf(quantum_mean['g0_decimal']),mp.mpf(quantum_mean['gs_decimal'])
        rho = mp.sqrt(g0+3*gs)
        orbit_eigenvalues = [(b0*g0+bs*gs)/2 for b0,bs,_ in orbit_categories()]
        rmin,rmax = min(orbit_eigenvalues),max(orbit_eigenvalues)
        a0 = mp.mpf(slow['minimum_stiffness'])
        a1 = mp.mpf(slow['maximum_stiffness'])
        hfast = min(mp.mpf(1)/5*min(4*sum(mp.sin(mp.pi*int(z)/size)**2 for z in q)
            for q in sites if any(q)),rmin/5)
        omega_min = mp.sqrt(3*rmin/25)
        kmax = 1+12/rmin
        static_cap = mp.mpf(coefficients_bound)*2
        normalized_perturbation = scale**(-mp.mpf(1)/4)*static_cap/hfast
        schur_denominator = 3*hfast/5-scale**(-mp.mpf(1)/4)*(a1+static_cap)
        if schur_denominator<=0:
            raise ArithmeticError('Actual q0 fast block is not separated from the slow spectrum')
        schur = static_cap**2/schur_denominator
        if normalized_perturbation>=mp.mpf('0.01') or scale**(-mp.mpf(1)/4)*schur>a0/4:
            raise ArithmeticError('Material hierarchy does not control full fast/slow harmonic blocks')
        coercivity_margin = hfast/2-scale**(-mp.mpf(1)/4)*(
            static_cap+4*static_cap**2/a0)
        if coercivity_margin<=0:
            raise ArithmeticError('Young inequality does not protect separate slow/fast potential masses')
        if scale**(mp.mpf(1)/4)*hfast<a0:
            raise ArithmeticError('Fast quadratic mass does not dominate the scalar radial mass')
        gap0 = scale**(mp.mpf(1)/4)*mp.sqrt(a0/2)
        zpe_interval = fast_zero_point_interval(quantum_mean,size,200)
        zpe_upper = Fraction(int(zpe_interval['upper_exact']['numerator']),
            int(zpe_interval['upper_exact']['denominator']))
        # This is a positive bound ONLY, not a numerical common phase. In the
        # excitation/excess inequalities the SAME exact Efast cancels first.
        fast_energy = 2*scale**(mp.mpf(3)/8)*volume*(
            mp.mpf(zpe_upper.numerator)/zpe_upper.denominator)
        slow_energy = 45*scale**(mp.mpf(1)/4)*mp.sqrt(static_cap)
        fast_variance_max = (mp.mpf(3)/5+12/rmin)/(2*mp.sqrt(3*rmin/25+mp.mpf(12)/5))
        slow_variance_max = 1/(2*mp.sqrt(a0))
        mean_radius2 = fast_dimension*fast_variance_max*scale**(-mp.mpf(7)/8)+45*slow_variance_max*scale**(-mp.mpf(3)/4)
        cut_norm_lower = 1-4*mean_radius2/eta**2
        if cut_norm_lower<=mp.mpf('0.99'):
            raise ArithmeticError('Actual full Gaussian is not supported inside its Gauss chart')
        # Analytic complex ball: A=O0^*O(Phi), ||delta O||<=2||delta C||.
        analytic_radius = min(mp.mpf('0.001'),rmin/(16*mp.sqrt(rmax)))
        metric_cap = 4+512/rmin
        metric_first = 8*metric_cap/analytic_radius
        log_measure_first = 8/analytic_radius
        log_measure_second = 32/analytic_radius**2
        geometric_scalar = chi*(dimension**2*(metric_first*log_measure_first+
            metric_cap*log_measure_second)/4+
            dimension*metric_cap*log_measure_first**2/8)
        electric_metric_relative = metric_first*eta/(mp.mpf(3)/5)
        heavy_gap = 2*mp.mpf(cfg.heavy_m)*mp.sqrt(g0)
        K1 = 8*mp.mpf(heavy_cap)*(1+rho)
        K2 = 16*mp.mpf(heavy_cap)*(1+rho)
        K3 = 32*mp.mpf(heavy_cap)*(1+rho)
        if heavy_gap<=4*K1*eta:
            raise ArithmeticError('The actual heavy filled cluster is not gapped throughout the chart')
        occupied = 2*volume
        third_heavy = occupied*(128*K1**3/heavy_gap**2+32*K1*K2/heavy_gap+K3)
        third_quartic = 24*(2/mp.mpf(cfg.a))*(rho+1)
        potential_relative = eta*(8*(third_heavy+third_quartic)/a0+
            256*(1+rho)**2/hfast+4096/rmin)
        rounded0 = mp.mpf(float(np.sqrt(g[0,0])))**2
        roundedS = mp.mpf(float(np.sqrt(g[1,1])))**2
        heavy_rows = mp_uniform_coefficients(band,size)
        rounded_radii = [mp.sqrt(rounded0*row[0]**2+roundedS*sum(z*z for z in row[1:]))
            for row in heavy_rows]
        rounded_f0 = 2/mp.mpf(cfg.a)*rounded0-sum(
            row[0]**2/r for row,r in zip(heavy_rows,rounded_radii))/volume
        rounded_fs = 6/mp.mpf(cfg.a)*roundedS-sum(
            sum(z*z for z in row[1:])/r for row,r in zip(heavy_rows,rounded_radii))/volume
        rounded_force = 2*mp.sqrt(rounded0*rounded_f0**2+roundedS*rounded_fs**2/3)
        rounded_action_lower = (scale**(mp.mpf(5)/8)*mp.sqrt(volume)*rounded_force/
            (2*mp.sqrt(2*mp.sqrt(a1))))
        form_relative = electric_metric_relative+potential_relative
        light_lipschitz = 8*mp.sqrt(volume)*mp.mpf(light_cap)*(1+rho)
        stationary_error = mp.mpf(quantum_mean['full_normalized_uniform_gradient_decimal'])
        linear_force = (4*rho*mp.sqrt(volume)*scale**(mp.mpf(3)/8)*sum(
            abs(mp.mpf(z)) for z in quantum_mean['fast_zero_point_gradient_decimal'])+
            light_lipschitz+4*scale*rho*mp.sqrt(volume)*stationary_error)
        harmonic_energy_excess = scale**(mp.mpf(1)/8)*fast_dimension*static_cap*(
            fast_variance_max+kmax/omega_min)+45*schur/mp.sqrt(a0)
        trial_energy = fast_energy+slow_energy+harmonic_energy_excess
        normalization_excess = (1/cut_norm_lower-1)*trial_energy
        cut_derivative_energy = 8*chi*metric_cap*mean_radius2/eta**4
        cut_energy_excess = normalization_excess+2*mp.sqrt(trial_energy*cut_derivative_energy)+cut_derivative_energy
        shape_price = mp.mpf(10)**(-slow['precision']+40)*dimension**2*trial_energy
        scalar_energy_perturbation = form_relative*(trial_energy+gap0)+linear_force*eta+geometric_scalar
        baseline_excess = harmonic_energy_excess+cut_energy_excess+shape_price
        gap = gap0-4*scalar_energy_perturbation-2*baseline_excess
        if gap<gap0/2:
            raise ArithmeticError('Actual full metric/potential form sandwich does not protect the well gap')
        energy_excess = harmonic_energy_excess+cut_energy_excess+shape_price+2*scalar_energy_perturbation
        spectral_distance = mp.sqrt(2*energy_excess/gap)
        well_energy_upper = trial_energy+cut_energy_excess+shape_price+scalar_energy_perturbation
        weighted_energy_budget = well_energy_upper+linear_force*eta+geometric_scalar
        barrier = scale*a0*eta**2/1024-linear_force*eta-geometric_scalar
        shell_potential_lower = (1-form_relative)*scale*a0*eta**2/256-linear_force*eta-geometric_scalar
        ramp_slope = eta*mp.sqrt(scale*a0/(chi*metric_cap))/64
        ramp_kinetic_price = chi*metric_cap*ramp_slope**2/2
        if (form_relative>mp.mpf('.01') or barrier<16*well_energy_upper or
                ramp_kinetic_price>shell_potential_lower/8 or
                shell_potential_lower-well_energy_upper-ramp_kinetic_price<barrier/2):
            raise ArithmeticError('Repaired Agmon ramp does not lie below the actual well barrier')
        Agmon = eta**2*mp.sqrt(scale*a0/(chi*metric_cap))/1024
        weighted_norm2 = 1+2*weighted_energy_budget/barrier
        weighted_gradient2 = 4*weighted_energy_budget/(chi*mp.mpf('0.6'))+2*ramp_slope**2*weighted_norm2
        boundary_prefactor = 2*(chi*metric_cap*8/eta*mp.sqrt(weighted_gradient2)+
            chi/2*(dimension**2*metric_first*8/eta+
                dimension*metric_cap*96/eta**2)*mp.sqrt(weighted_norm2))
        boundary_log2_upper = mp.log(boundary_prefactor,2)-Agmon/mp.log(2)
        if boundary_log2_upper>-4096:
            raise ArithmeticError('Full chart boundary action has not been priced')
        # Normalized gapped Slater section: wedge(P(C,U) occupied columns).
        D1 = 4*mp.sqrt(dimension)*occupied*K1/heavy_gap
        D2 = dimension*(64*occupied**2*K1**2/heavy_gap**2+16*occupied*K2/heavy_gap)
        derivative_norm = mp.sqrt(8*weighted_energy_budget/(chi*mp.mpf('0.6')))
        charge_norm = mp.sqrt(volume)*(3+cfg.auxiliary_count)*mp.sqrt(mp.mpf(5)/2)
        fibre_action = chi*(4*metric_cap*derivative_norm*(D1+charge_norm/rmin)+
            metric_cap*D2+dimension*metric_first*(D1+charge_norm)+
            metric_cap*charge_norm**2/rmin)
        charged_action = light_lipschitz*eta
        action = fibre_action+charged_action+mp.mpf(2)**(-4096)
        distance = 2*spectral_distance+mp.mpf(clock)*action
        # A real all-mode electromagnetic input: H2 coexact/harmonic packet.
        # Reference graph norm is preserved by its exact charged/EM unitary.
        em_radius = mp.mpf(1)/8
        em_modes = 2*volume+1
        UElink = mp.mpf(cfg.inertia)/mp.mpf(cfg.a)
        smallest_laplacian = min(4*sum(mp.sin(mp.pi*int(z)/size)**2 for z in q)
            for q in sites if any(q))
        em_graph = UElink*mp.sqrt(mp.mpf(63)*em_modes/2+
            9*em_modes*(em_modes-1))/em_radius**2
        charge_cap = 2*(1+cfg.auxiliary_count)+4
        Coulomb_graph = UElink*volume*charge_cap**2/smallest_laplacian
        charged_graph = 8*volume*mp.mpf(light_cap)*rho
        reference_graph = em_graph+Coulomb_graph+charged_graph
        quartic_coercivity = scale/(8*mp.mpf(cfg.a)*volume)
        global_matter_cap = 8*mp.sqrt(volume)*(scale*mp.mpf(heavy_cap)+mp.mpf(light_cap))
        lower_energy_shift = 3*global_matter_cap**(mp.mpf(4)/3)/(
            4**(mp.mpf(4)/3)*quartic_coercivity**(mp.mpf(1)/3))
        packet_rho_cap = mp.sqrt(volume)*rho+eta
        inverse_fast_covariance = 2*mp.sqrt(3*rmax/25+mp.mpf(63)/25)/(mp.mpf(3)/5)
        kinetic_graph = metric_cap*dimension*(
            scale**(mp.mpf(3)/8)*(1+inverse_fast_covariance)+
            scale**(mp.mpf(1)/4)*(1+2*mp.sqrt(a1)))+16*metric_cap*dimension*chi/eta**2
        potential_graph = (scale/mp.mpf(cfg.a)*volume*(rho+eta)**4+
            8*scale*volume*mp.mpf(heavy_cap)*(rho+eta)+
            128*volume*scale**(mp.mpf(5)/4)*(1+rho)**2*eta**2)
        graph_cap = kinetic_graph+potential_graph+reference_graph+action
        difference_graph = 2*(graph_cap+lower_energy_shift)
        weighted_Phi_error = (difference_graph/quartic_coercivity)**(mp.mpf(1)/4)*distance**(mp.mpf(3)/4)
        current_cap = 16*mp.mpf(cfg.a)*mp.sqrt(volume)*mp.mpf(light_cap)
        contact_cap = 32*mp.mpf(cfg.a)**2*mp.sqrt(volume)*mp.mpf(light_cap)
        current_action_error = current_cap*weighted_Phi_error
        contact_action_error = contact_cap*weighted_Phi_error
        current_expectation_error = 2*current_action_error+current_cap*(1+rho)*eta
        contact_expectation_error = 2*contact_action_error+contact_cap*(1+rho)*eta
        fast_rows = []
        for q in sites:
            lap = 4*sum(mp.sin(mp.pi*int(z)/size)**2 for z in q)
            coefficients_rows = []
            for (b0,bs,mult),r in zip(orbit_categories(),orbit_eigenvalues):
                transverse = mp.mpf(3)/(10*mp.sqrt(3*r/25+21*lap/100))
                longitudinal = (mp.mpf(3)/5+lap/r)/(2*mp.sqrt(3*r/25+lap/5))
                coefficients_rows.append(dict(category=[b0,bs,mult],r=mp.nstr(r,200),
                    transverse_scaled_variance=mp.nstr(transverse,200),
                    longitudinal_scaled_variance=mp.nstr(longitudinal,200)))
            fast_rows.append(dict(q=q.tolist(),lap=mp.nstr(lap,200),
                Phi_scaled_variance=mp.nstr(1/(2*mp.sqrt(lap/5)),200) if any(q) else None,
                colour=coefficients_rows))
        price = lambda value: mp.nstr(value,60)
        prices = dict(eta=price(eta),principal_metric_cap=price(metric_cap),
            normal_gap_lower=price(gap),fast_energy_upper=price(fast_energy),
            harmonic_reprepare_energy_excess=price(harmonic_energy_excess),
            Haar_FP_scalar_action_upper=price(geometric_scalar),
            nonlinear_form_relative=price(form_relative),
            actual_H2_cut_norm_squared_lower=dict(expression='1-deficit',
                deficit_upper=price(4*mean_radius2/eta**2)),
            cut_energy_excess=price(cut_energy_excess),shape_rounding_energy_excess=price(shape_price),
            mean_force_linear_action_on_chart=price(linear_force*eta),
            rounded_literal_centre_force_action_lower=price(rounded_action_lower),
            well_spectral_state_distance_upper=price(spectral_distance),
            Agmon_exponent_lower=price(Agmon),physical_boundary_action_log2_upper=price(boundary_log2_upper),
            heavy_Berry_Gauss_affine_action_upper=price(fibre_action),
            charged_EM_mean_freezing_action_upper=price(charged_action),
            reference_EM_CAR_graph_norm_upper=price(reference_graph),
            full_Hamiltonian_lower_shift=price(lower_energy_shift),
            quartic_graph_interpolation_coercivity=price(quartic_coercivity),
            difference_graph_norm_upper=price(difference_graph),
            weighted_Phi_action_error=price(weighted_Phi_error),
            all_flavour_current_action_error=price(current_action_error),
            all_flavour_contact_action_error=price(contact_action_error),
            all_flavour_current_expectation_error=price(current_expectation_error),
            all_flavour_contact_expectation_error=price(contact_expectation_error),
            physical_state_distance_at_clock_upper=price(distance),clock=clock)
        derivation = dict(
            coordinates=dict(real_space='45 exact normal Cart coordinates per site, 15 exponential Lie coordinates per oriented link; Fourier transform is real orthogonal and does not alter the radial cutoff',
                chart_radius=price(analytic_radius),well_radius=price(eta),
                flat_cut_inner=price(eta/4),flat_cut_outer=price(eta/2),
                FP='A=O0^T O(Phi), R0=O0^T O0. w_local=prod det(A)/sqrt(det(R0)) times prod det(Jleft). Joint Phi/U gauge tube has residual diagonalZ4 and cover degree4, not4^V. With normalized SU4 Haar and Maurer volume Vol4, w_joint=Vol4^(V-L) w_local/4; cubic L=3V gives Vol4^(-2V)w_local/4. W=w_joint^(-1/2) times matter-equivariant extension. Constant volume/cover factors drop from every log-measure derivative.',
                exact_basis=slow['normal_basis_definition']),
                joint_map='Phi_x=Vx(Cmean+N0Yx)Vx^dagger; Uxy=Vx exp(iTheta_xy.T) Vy^dagger. Y and relativeTheta lie in displayed near-zero chart. Phi stabilizer isZ4 because full-rank Clifford frame generates M4; near-I link support and connectivity force all local centre choices equal.',
                descent_isometry='Psi(F(Y,Theta,V))=R_matter(V)w_joint^(-1/2) f(Y,Theta). Diagonal z acts by z^Ntotal; declared totalNmod4=0 gives single-valued descent. Degree4 change of variables gives dmu_boson=Vol4^(V-L)w_local dYdTheta dHaar(V)/4. Normalized Haar integral1 and unitary R_matter give integral||Psi||^2=integral||f||^2=1. Vol4 is geometrical total Maurer volume, not adjustable input.',
            ellipticity=dict(lower_exact='3/5',
                formula='pPhi=N0 pN+O0 A^(-T)[rho-div(Eleft)-O(Phi)^T N0 pN], Eleft=Jleft^(-T) ptheta. K/chi is I45 plus the positive Gauss-square and (3/5) Jleft^(-1)Jleft^(-T).',
                derivation='For real theta, Jleft=int_0^1 Ad(exp(it theta.T))dt averages orthogonal matrices, hence ||Jleft||<=1. N0 is exactly orthogonal to O0. Thus the homogeneous kinetic form is >=(3/5)(|pN|^2+|ptheta|^2), not a fitted normal metric.',
                complex_cap=price(metric_cap),
                cap_derivation='||delta O||<=2||delta C|| follows adjoint Casimir4. ||R0^-1/2 delta A R0^-1/2||<=2sqrt(rmax)R/rmin<=1/8. Hence ||O0 A^-T||<=2/sqrt(rmin). The six-edge divergence has norm<=sqrt(13), ||Jleft^-1||<=2 and ||O(Phi)^T N0||<=2R. The resulting principal norm is <=4+(208+16R^2)/rmin <=4+512/rmin.',
                metric_Cauchy_first=price(metric_first)),
            measure=dict(log_first_cap=price(log_measure_first),log_second_cap=price(log_measure_second),
                derivation='Per local site plus its three outgoing links, |log(w/w0)|<4: 15[-log(1-1/8)]<2.01 for FP, and the three small Haar determinants contribute <1.99 at R<=.001. Real-space Cauchy estimates on radius R/2 give 8/R and 32/R^2. Fourier coordinates are NOT used to claim a dimension-free bound on the sum log(w).',
                exact_scalar='U_w=chi div(Kbar grad log w)/4+chi (grad log w)^T Kbar grad log w/8, where Kbar=K/chi. Sum at most D^2 first/second-derivative terms and D gradient-square terms gives the displayed scalar cap.'),
            potential=dict(heavy_gap=price(heavy_gap),chart_heavy_gap_lower=price(heavy_gap-2*K1*eta),
                K1=price(K1),K2=price(K2),K3=price(K3),heavy_third_cap=price(third_heavy),
                heavy_derivation="Actual one-link row/column path sum bounds h derivatives by K1,K2,K3. Perturbation <=K1 eta leaves cluster gap >=Delta/2. E'''=Tr(P h'''+2P' h''+P'' h'). With ambient dimension4V=2 nocc, projector differentiation gives nocc[K3+16K1K2/gap+32K1^3/gap^2]. Substituting gap>=Delta/2 gives the recorded128/32 coefficients.",
                quartic_third_cap=price(third_quartic),
                separate_mass_margin=price(coercivity_margin),
                quadratic_lower='Young: 2lambda C |s||f| >=-lambda amin |s|^2/4-4lambda C^2 |f|^2/amin. The recorded positive margin gives Vharm>=lambda amin |s|^2/4+lambda^(5/4)hfast |f|^2/4.',
                heavy_quartic_remainder='Their cubic remainder is <=lambda(Th+Tq) eta (|s|^2+|f|^2)/6. Comparing to the separate quadratic lower bounds is dominated by 8(Th+Tq)eta/amin.',
                alignment_remainder='Only fast coordinates enter the linear covariant difference; the uniform normal displacement has zero linear difference. The nonlinear difference is bounded by4(1+rho)eta |f|, the linear one by6(1+rho)|f|. Thus |Ralign|<=32 J(1+rho)^2 eta |f|^2, dominated relative to the separate fast mass by256(1+rho)^2 eta/hfast. No erroneous J eta/slow-mass bound is used.',
                curl_remainder='Each link occurs in four plaquettes. Expanding the four unitary exponentials on the radius<=.001 ball and summing the cubic remainder bounds it by a constant times lambda^(5/4) eta |theta|^2. Relative to the Higgs mass, the explicit4096 eta/rmin cap dominates this term.'),
            minmax=dict(exact_common='Write E_h for the exact full harmonic ground scalar. Identical exact V(mean), Efast and slow zero-point scalars cancel algebraically before every gap/excess comparison; numerical energies are used only as positive upper bounds.',
                roots='Fast block roots change by at most C lambda^(1/8)/omega_min; q0 slow roots obey the recorded Schur bound. Leading Gaussian energy minus E_h is <=harmonic_reprepare_energy_excess.',
                sandwich='(1-epsilon)Hharm-F eta-U <= Hwell-V(mean) <=(1+epsilon)Hharm+F eta+U on the Dirichlet ball.',
                endpoints='Cut trial upper E0<=E_h+X+P; minmax lower E1>=E_h+gap0-P; X=harmonic excess+cut excess+shape price and P=epsilon(Etrial_upper+gap0)+F eta+U. Therefore gap0-4P-2X is a conservative gap lower, and X+2P bounds trial-ground energy excess. The harmonic Dirichlet ground cutoff shift X is explicitly paid, not silently dropped.',
                spectral_overlap='If excess=X+2P and gap>0, 1-|<trial,ground>|^2<=excess/gap; phase-aligned distance<=sqrt(2 excess/gap).'),
            cutoffs=dict(H2_radial='b=(1-r^2/eta^2)^2_+. b and its first derivative vanish at r=eta. ||grad b||<=4r/eta^2, ||Hess b||<=12/eta^2, |Delta b|<=(4D+8)/eta^2. ||b Gaussian||^2>=1-4 E(r^2)/eta^2. Normalize by the exact positive integral, not by a float rounded to1.',
                flat='u=4r/eta-1; interior1, shell1-(10u^3-15u^4+6u^5), exterior0. For eta/4<=r<=eta/2: |grad cut|<=8/eta, ||Hess cut||<=96/eta^2, |Delta cut|<=(96+32D)/eta^2. This flat cutoff, not zero-extension of the Dirichlet ground, is used for the H2 quasimode.',
                barrier=price(barrier),Agmon=price(Agmon),
                agmon_derivation='Ramp begins eta/8, strictly inside first flat-cut derivatives eta/4. Slope=eta sqrt(lambda amin/(chi M))/64; weight kinetic price=lambda amin eta^2/8192. Actual shell potential lower=(1-epsilon)lambda amin eta^2/256-F eta-U. Price<=this shell lower/8 and shell lower-E_D-price>=recorded weaker barrier/2 are executed guards. This avoids incorrectly claiming price<=(unperturbed barrier-F eta-U)/8 at equality. E_D-V(mean)<=recorded well_energy_upper via normalized cut trial, barrier>16 times that upper. Distance eta/8 gives weight>=eta^2 sqrt(lambda amin/(chi M))/512; retained /1024 exponent is conservative.',
                boundary='Weighted Dirichlet identity gives outside coefficient V-E_D-chi grad(f)^T Kbar grad(f)/2>=barrier/2. Interior potential>=-F eta-U implies weighted norm^2<=1+2 budget/barrier and weighted gradient^2<=4 budget/(chi*.6)+2 slope^2 weighted norm^2. Commutator chi Kbar grad(cut).grad plus chi div(Kbar grad(cut))/2 uses actual8/eta,96/eta^2 constants and D^2 metric derivative terms. Executed boundary prefactor includes normalization factor2 and multiplies exp(-Agmon), not an exponent-only assertion.',
                well_energy_upper=price(well_energy_upper),weighted_energy_budget=price(weighted_energy_budget),
                ramp_slope=price(ramp_slope),ramp_kinetic_price=price(ramp_kinetic_price),
                shell_potential_lower=price(shell_potential_lower),
                weighted_norm_squared_upper=dict(expression='1+excess',
                    excess_upper=price(2*weighted_energy_budget/barrier)),
                weighted_gradient_squared_upper=price(weighted_gradient2),
                actual_boundary_prefactor=price(boundary_prefactor)),
            graphs=dict(coercivity='Quartic >=lambda rhoPhi^4/(4aV)=2c rhoPhi^4. Matter >=-L rhoPhi by actual unitary paths. The retained other half c rho^4-L rho>=-3L^(4/3)/(4^(4/3)c^(1/3))=-B. Positive kinetic/alignment/curl/electric/Coulomb terms give H+B>=c rhoPhi^4.',
                reference_at_all_times='Reference H_ref graph norm is conserved by its exact unitary. Direct unitary path and charged number bounds are uniform in EM coordinates; Phi cut support remains fixed. Kinetic Gaussian/cutoff/fibre derivative bounds and positive-potential path bounds thus bound the full H+B applied to the lifted reference at each time. The same initial full graph norm is conserved along the actual full evolution. Their sum is the recorded difference-graph cap.',
                EM_moments='For normalized 1D (1-(theta/eta)^2)^2_+, <p^2>=3/eta^2 and ||p^2 f||^2=63/(2eta^4). Orthogonal product modes give ||sum p_j^2||=sqrt(63D/2+9D(D-1))/eta^2. These are dimensionless canonical Peierls-phase coordinates, not physical A coordinates.',
                Heinz='rho^4<=c^-1(H+B) in forms. Operator monotonic square root gives rho^2<=c^-1/2(H+B)^1/2. Spectral interpolation then gives ||rho delta||<=c^-1/4 ||delta||^3/4 ||(H+B)delta||^1/4. Both graph norms are priced, not inferred merely from norm closeness.',
                consumer='Peierls phases are unitary for every quantum theta. Actual one/two-link path derivative sums give ||J delta||<=CJ||rho delta|| and ||contact delta||<=CJJ||rho delta||. Cauchy-Schwarz gives the displayed expectation differences.'),
            review_status='Explicit derivations and executed constants for parent mathematical review; not automatic theorem admission')
    microscopic = {key:value for key,value in asdict(cfg).items() if key!='Lambda'}
    microscopic['Lambda_exact'] = str(2**exponent)
    descriptor = dict(config=microscopic,size=size,Lambda_exponent=exponent,cutoff_bits=cutoff_bits,clock=clock,
        bosonic_coefficients='Canonical LAW05 exact fractions J=Lambda^(5/4)/5, colour curl7Lambda^(5/4)/10, colour Hamiltonian3chi E4^2/10 and cometric3chi/5; scalar-path matter coefficients are exact binary64 inputs',
        colour_gauge_section='JOINT Phi/U near-relative-link chart; connected-link residual diagonalZ4, totalNmod4 descent. Normalized SU4 Haar, coverdegree4 and geometrical Maurer-volume constants included in W; no independent local-centre matter projection.',
        scalar_heavy_sector='fixed NH=2V upper-h occupied; minusLambda h, exact actual one-link paths',
        charged_reference='all64 actual CAR flavours, full linear Cartesian one/two-link Peierls h(physical exact mean, Ucolour=I, A quantum); electric plus Coulomb reference retained',
        reference_EM_packet=dict(coexact_harmonic_modes=2*size**3+1,radius='1/8',
            coordinates='dimensionless canonical Peierls phases theta_link=a Aphysical_link; primitive electric UEphys/a ptheta_link^2',
            Hilbert_domain='NONCOMPACT electromagnetic reference: L2(R^(2V+1)), quotient of theta_link in R^(3V) by additive real site gauges, with real-gradient Gauss solution. NOT compact U1/integer electric/Haar branch.',
            Coulomb_solution='Pi_long=D^T(DD^T)^+rho; H_C=UEphys/a rho^T(DD^T)^+rho, total charge0. Spectral norm<=1/[4sin(pi/N)^2] gives the same Coulomb graph cap used here.',
            wavefunction='normalized product (1-(theta_j/etaTheta)^2)^2_+ in real orthonormal coexact+harmonic coordinates',
            charge_sector='all64 filled valence doublets, Ref0/4 or extra active electron plus compensating Ref hole; total U1charge0 and SU4centreclass0'),
        quantum_inputs='hbar=1 canonical Cart/Haar/EM quantization and CAR declared inputs')
    return dict(branch='NEW full-normal canonical Gauss packet local-well comparison; no vacuum or continuum promotion',
        input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        quantum_mean=quantum_mean,slow_packet=slow,fast_packet=fast_rows,
        physical_centre=centre,native_heavy_blocks=rows,actual_native_Ward_max=Ward,
        actual_heavy_metric_trace=metric_trace,independent_MP_native_Hessian_discrepancy=discrepancy,
        prices=prices,derivation_ledger=derivation,
        canonical_geometry_control=dict(microscopic=asdict(control_cfg),size=4,report=geometry_control),
        canonical_one_link_Casimir_control=Casimir_control,
        noncompact_EM_Gauss_control=EM_control,flat_cut_Agmon_support_control=cut_support_control,
        joint_gauge_descent_control=joint_descent,
        normalized_state='All45 uniform slow+90V-45 fast real Fourier Gaussian modes at exact decimal centre, H2 radialcut and exact positive normalization. Joint Phi/U chart with relative links nearI, residual diagonalZ4. W=[Vol4^(-2V)prod(det A/sqrt det R0)prod HaarJac/4]^(-1/2) followed by unitary matter-equivariant extension. TotalNmod4=0 ensures descent; degree4 change of variables proves W isometry.',
        fast_zero_point_interval=zpe_interval,
        common_phase='Exact scalar Dirichlet-well eigenvalue E_D defines exp(-it E_D); no rounded numerical phase is used. V(mean) and the SAME exact algebraic/trigonometric Efast cancel before excitation-energy/excess comparison.',
        fast_covariance='z_fast=Lambda^-7/16 y. Per colour r: transverse/longitudinal projectors of d_axis=1-exp(iq_axis); q0 three identical colour variances. Nonzero-q Phi45 identity covariance; q0 Phi45 belongs slow packet only.',
        comparison_lemmas=[
            'Native harmonic full block has fast-root variation <=C Lambda^1/8 and q0 slow Schur root correction O(1), with displayed actual remainder norm and scale-separated Schur margin.',
            'On the chart ball, exact Gauss/Haar principal metric and scalar measure potential obey displayed Cauchy caps. Heavy Kato cluster third derivative and quartic/colour/alignment Taylor bounds give the displayed quadratic-form sandwich.',
            'Dirichlet well gap >= harmonic gap minus twice both endpoint form-price errors. Trial energy excess over the well ground divided by the gap bounds squared state distance.',
            'The displayed potential barrier and elliptic cometric give Agmon decay; a flat-interior smooth cutoff of the well ground has the displayed physical boundary-action bound.',
            'Physical Gaussian is close to that quasimode. Exact unitarity preserves this preparation error; Duhamel adds actual gapped-heavy fibre, affine matter-Gauss and charged/EM freezing actions.',
            'Retained half-quartic gives H+B>=c rhoPhi^4. Heinz interpolation yields ||rhoPhi deltaPsi||<=c^-1/4 ||deltaPsi||^3/4 ||(H+B)deltaPsi||^1/4. Actual all-mode H2 EM input and CAR/Coulomb caps price graph norm; linear current/contact Peierls-unitary caps then price the consumer.'],
        proof_status='Executed coefficients, normalized packet, native controls and analytic-bound arithmetic; comparison lemmas displayed for parent mathematical review, NOT an exact many-body simulation or an admitted full quantum phase.',
        limitations='Finite even box only, not macrobox/bulk stationarity. No common-cone repair. This is Phi/colour and weighted-current transfer to exact charged/EM reference, not validation of RPA photon vacuum or Compton kinetics.',
        source_at_import=IMPORT_HASHES,
        source_snapshot_at_import=dict(path='induced_phi_backreaction.py',
            sha256=IMPORT_HASHES['induced_phi_backreaction.py'],
            text=IMPORT_BYTES['induced_phi_backreaction.py'].decode('utf8')),
        source_unchanged_since_import={name:IMPORT_HASHES[name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_NAMES},
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform()),
        elapsed_seconds=time.monotonic()-start)


def born_oppenheimer_repair(cfg: PhiConfig, size: int, exponent: int = 816):
    """Finite original-family Lambda repair, with explicitly bounded mean error.

    This is an all-coordinate NORMAL TANGENT state. Colour/Gauss chart kinetic
    prices are not included, and are not claimed to vanish. It supplies the
    independent Phi/sea calculation to that physical completion.
    """
    scale = float(2**exponent)
    heavy_cfg = replace(cfg, Lambda=1., auxiliary_count=0)
    sites = periodic_grid(size)
    p = 2*np.pi*sites/(size*cfg.a)
    heavy_band = models(heavy_cfg)[0][1]
    xh = coefficients(heavy_band,p)
    selected = select_gram(xh,2/cfg.a,0.)
    precision = high_precision_heavy_mean(heavy_band,size,selected['G'])
    g = np.asarray(precision['G'])
    frame = np.sqrt(np.diag(g))[:,None,None]*SELECTED
    basis = matrix_unit_basis()
    h = np.einsum('pm,mij->pij',xh,frame)
    values,vectors = np.linalg.eigh(h)
    data = dict(sites=sites,x=xh,values=values,vectors=vectors)
    elastic = elastic_hessian(frame,basis,2/cfg.a,0.)
    h0 = elastic+fourier_sea_hessian(data,np.zeros(3,dtype=int),size,basis)
    orbit = np.asarray([1j*(b@frame-frame@b) for b in basis])
    orbit = orth(np.einsum('tmij,aji->mat',orbit,basis).real.reshape(60,15)/4)
    normal = null_space(orbit.T)
    normal_eig = np.linalg.eigvalsh(normal.T@h0@normal)
    volume = size**3
    dimension = 60*volume-15
    loads = models(cfg)
    light_gradient = np.zeros((4,4))
    derivative1 = 0.
    derivative2 = 0.
    cubic_sea = 0.
    light_curvature_bound = 0.
    gap_records = []
    for _,band,role in loads:
        x = coefficients(band,p)
        radii = np.sqrt(np.einsum('pi,ij,pj->p',x,g,x))
        onsite,paths = scalar_paths(band)
        cap = 2*(abs(onsite)+2*sum(float(np.linalg.norm(s,1)) for _,s in paths))
        gap = 2*band.m*np.sqrt(g[0,0])
        count = 1 if role.startswith('neutral') else (1 if role.startswith('active') else cfg.auxiliary_count)
        nocc = 2*volume
        # Normalized projected Slater determinant: conservative analytic
        # first/second derivative bounds in its nonvanishing overlap chart.
        derivative1 += 4*nocc*count*cap/gap
        derivative2 += 128*nocc*nocc*count*(cap/gap)**2
        load = scale if role.startswith('neutral') else count
        cubic_sea += 64*nocc*load*cap**3/gap**2
        if not role.startswith('neutral'):
            light_gradient -= count*np.einsum('pi,pj,p->ij',x,x,1/radii)/volume
            light_curvature_bound += 8*nocc*count*cap**2/gap
        gap_records.append(dict(role=role,count=count,normalized_pair_gap=gap,coefficient_cap=cap))
    # Strong convexity gives an EXACT error interval for finite nonzero charged
    # loads which are smaller than binary64 resolution relative to the heavy.
    gram_error = float(np.linalg.norm(light_gradient))/(2*scale/cfg.a)
    minimum = .99*scale*float(normal_eig[0])-light_curvature_bound
    if minimum <= 0:
        raise ArithmeticError('Heavy-only normal state is not stable under charged backreaction')
    # For nonzero q, sea curvature norm is bounded by the same resolvent estimate.
    heavy_gap = gap_records[0]['normalized_pair_gap']
    heavy_cap = gap_records[0]['coefficient_cap']
    negative_heavy_bound = 8*(2*volume)*heavy_cap**2/heavy_gap
    first_lap = 4*np.sin(np.pi/size)**2
    alignment = .2*scale**1.25
    spatial_lower = alignment*first_lap-scale*negative_heavy_bound-light_curvature_bound
    if spatial_lower <= 0:
        raise ArithmeticError('The material Lambda does not dominate the spatial sea bound')
    minimum = min(minimum,spatial_lower)
    maximum = (2*scale/cfg.a)*6*float(np.trace(g))+12*alignment
    sigma_max = float(np.exp((-.5*np.log(scale)-np.log(4.)-np.log(minimum))/4))
    sigma_min = float(np.exp((-.5*np.log(scale)-np.log(4.)-np.log(maximum))/4))
    radius = sigma_max*np.sqrt(2*dimension*np.log(scale))
    gap_chart_ok = all(2*radius*part['coefficient_cap'] < part['normalized_pair_gap']/4
                       for part in gap_records)
    if not gap_chart_ok:
        raise ArithmeticError('The BO packet leaves the proven spectral gap chart')
    r0 = np.sqrt(volume*float(np.trace(g)))
    cubic_quartic = 2*(2*scale/cfg.a)*r0
    moment3 = sigma_max**3*np.sqrt(dimension*(dimension+2)*(dimension+4))
    # Group factors before multiplication: sigma^4 underflows but Lambda*sigma^4 does not.
    quartic_residual = (scale*sigma_max**3/cfg.a)*sigma_max*np.sqrt(
        dimension*(dimension+2)*(dimension+4)*(dimension+6))
    linear_residual = 2*r0*float(np.linalg.norm(light_gradient))*sigma_max*np.sqrt(dimension)
    potential_residual = (cubic_sea+cubic_quartic)*moment3+quartic_residual+linear_residual
    mean_residual = scale*precision['stationary_Gram_error_upper']*(2/cfg.a)*2*r0*sigma_max*np.sqrt(dimension)
    kinetic_residual = scale**-.5*(dimension*derivative1/(2*sigma_min)
                                   +dimension*(derivative2+derivative1**2)/2)
    # Smooth cutoff equal to one inside radius, supported within twice radius.
    # Gaussian tail norm <= exp(log_tail); cutoff derivative constants 10/100.
    log_tail = -.25*(radius/sigma_max)**2+.25*dimension*np.log(2.)
    log_cutoff = (np.log(scale**-.5*(10*np.sqrt(dimension)/(sigma_min*radius)
                        +100*dimension/radius**2))+log_tail)
    cutoff = float(np.exp(log_cutoff)) if log_cutoff > np.log(1e-300) else 1e-300
    residual = 2*(potential_residual+kinetic_residual+cutoff+mean_residual)
    time_window = .01/residual
    q0_maximum = (2*scale/cfg.a)*6*float(np.trace(g))
    sigma_q0_min = float(np.exp((-.5*np.log(scale)-np.log(4.)-np.log(q0_maximum))/4))
    rounded_centre_control = (scale*precision['rounded_centre_normalized_force']
                               *np.sqrt(volume)*sigma_q0_min)
    # Every microscopic Peierls/SU4 multiplier is unitary, even in excited PH
    # states. Entrywise bilinear norm <=1 gives this all-state Phi Lipschitz cap.
    charged_quantum_lipschitz = 4*np.sqrt(volume)*sum(
        record['count']*record['coefficient_cap'] for record in gap_records[1:])
    # H error and current/contact variation from ACTUAL charged scalar paths.
    current_bounds = []
    for _,band,role in loads[1:]:
        _,paths = scalar_paths(band)
        hcap = 2*(abs(scalar_paths(band)[0])+2*sum(float(np.linalg.norm(s,1)) for _,s in paths))
        jcap = 4*sum(float(np.linalg.norm(s,1))*int(np.max(d)) for d,s in paths)
        kcap = 4*sum(float(np.linalg.norm(s,1))*int(np.max(d))**2 for d,s in paths)
        encoding = precision['float_consumer_frame_operator_error']
        current_bounds.append(dict(role=role,onebody_error=hcap*2*radius,
            current_error_link=jcap*2*radius,contact_error_link=kcap*2*radius,
            float_mean_encoding_onebody_error=hcap*encoding,
            float_mean_encoding_current_error_link=jcap*encoding,
            float_mean_encoding_contact_error_link=kcap*encoding))
    return dict(branch='Material original-family Phi repair, finite Lambda=2**'+str(exponent),
        Lambda=scale,size=size,G_centre=g.tolist(),
        charged_Gram_shift_from_exact_heavy_upper=gram_error,
        exact_Gram_error_Frobenius_upper=gram_error+precision['stationary_Gram_error_upper'],
        exact_heavy_mean_precision=precision,
        high_precision_centred_mean_residual=mean_residual,
        ignored_high_precision_control=dict(rounded_centre_force_action_lower=rounded_centre_control,
            statement='The actual nonzero rounded-centre gradient creates a linear Gaussian residual; q0 covariance lower prices it. Binary64 centre cannot be declared stationary at this Lambda.'),
        all_charged_and_photon_state_backreaction=dict(Lipschitz=charged_quantum_lipschitz,
            convex_normal_displacement_upper=2*charged_quantum_lipschitz/minimum,
            energy_change_within_packet_upper=2*radius*charged_quantum_lipschitz,
            scope='All charged PH occupancies and all Peierls/SU4 phases; electric/link Hamiltonian is Phi-independent. Conditional on the reported normal convex chart; no full quantum phase inferred.'),
        occupancy='one neutral heavy and all64 charged filled seas; no quenching',
        state='BO Slater(C) times actual normal-mode Gaussian, projected-determinant smooth chart; Gaussian cutoff within twice radius; normal tangent Phi quantum Hamiltonian',
        normal_dimension=dimension,uniform_SU4_orbit_removed=15,
        gap_descriptors=gap_records,normal_stiffness_lower=minimum,
        spatial_stiffness_lower=spatial_lower,stiffness_upper=maximum,
        sigma_min=sigma_min,sigma_max=sigma_max,support_radius=2*radius,
        spectral_chart_verified=gap_chart_ok,
        potential_Taylor_action_residual=potential_residual,
        actual_BO_kinetic_action_residual=kinetic_residual,
        cutoff_residual_log_upper=log_cutoff,cutoff_residual_representable_upper=cutoff,
        total_normal_tangent_residual_upper=residual,
        finite_T_for_norm_error_0p01=time_window,current_contact_errors=current_bounds,
        finite_T_statement='In the stated normal-tangent Hamiltonian, Duhamel gives norm error<=T*residual. Reference harmonic packet is confined to twice radius up to reported Gaussian cutoff tail, hence actual escaped amplitude<=T*residual+tail. This is NOT yet the full gauge-horizontal Laplace-Beltrami, SU4 link, EM or Root transport bound.',
        physical_completion_debt='Actual full-Gauss chart curvature and colour/link electric action must be added by Field owner; not inferred from this independent normal-tangent calculation.')


def global_centre_projection(cfg: PhiConfig, size: int) -> dict[str,Any]:
    """Actual determinant character of every occupied valence doublet."""
    volume = size**3
    species = 2+cfg.auxiliary_count
    centres = (1.,1j,-1.,-1j)
    # Restrict z I4 to each actual occupied two-plane; its determinant is z²,
    # independent of which heavy/light eigenvectors span that filled plane.
    characters = [complex(np.linalg.det(z*np.eye(2)))**(species*volume) for z in centres]
    projection = sum(characters)/4
    return dict(size=size,V=volume,heavy_occupation=2*volume,
        charged_occupation=2*(1+cfg.auxiliary_count)*volume,reference_centre_class=0,
        characters=[[z.real,z.imag] for z in characters],projection_norm_squared=projection.real,
        imaginary_error=abs(projection.imag),global_centre_admissible=abs(projection-1)<1e-12,
        scope='Necessary global SU4 centre Gauss positive control, not proof of full local Gauss projection or confinement. Odd-volume filled-heavy65-species state is annihilated; an empty-heavy full-Fock upper does not prove its fixed-number vacuum upper.')


def run(cfg: PhiConfig, sizes: tuple[int,...], spatial_sizes: tuple[int,...], orders: tuple[int,...], with_response: bool) -> dict[str,Any]:
    start=time.monotonic()
    centre=[global_centre_projection(cfg,n) for n in spatial_sizes]
    if not all(row['global_centre_admissible'] for row in centre):
        raise ArithmeticError('Physical prepared filled-heavy sea requires an even volume; odd diagnostic means are not Gauss states')
    means=[uniform(cfg,n) for n in sizes]+[uniform(cfg,0,8),uniform(cfg,0,10)]
    data=means[-1]
    report: dict[str,Any]=dict(schema=1, branch='NEW actual common-lattice heavy plus mass-resolved charged Cartesian Phi; original quartic/alignment; no bare magnetic photon',
        microscopic=asdict(cfg), species_count=2+cfg.auxiliary_count,
        heavy_descriptor='One neutral RotationBand(a=common a,m=.2,r=.12,v=.3,improved=False), h multiplied by -Lambda; NOT old heavy a1',
        quantum_inputs='hbar=1 CAR and canonical field quantization declared; no derivation gate',
        uniform=[uniform_report(d,cfg) for d in means],
        input_descriptor=dict(config=asdict(cfg),sizes=sizes,bulk_mean_orders=(8,10),
            spatial_sizes=spatial_sizes,response_orders=orders,with_response=with_response,BO_Lambda_exponent=816),
        source_at_import=IMPORT_HASHES,
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform()))
    noaux=uniform(replace(cfg,auxiliary_count=0),0,10)
    report['ignore_63_aux_control']=dict(G=noaux['G'].tolist(),
        actual_energy_excess=objective(data,cfg,noaux['G'])-data['value'],
        metric_relative_change=float(np.linalg.norm(noaux['G']-data['G'])/np.linalg.norm(data['G'])))
    spatial=[spatial_report(cfg,n) for n in spatial_sizes]
    report['spatial']=spatial
    # Particle-hole double sum is finite actual box, not a surrogate continuum variance.
    packet_data=uniform(cfg,spatial_sizes[-1])
    report['prepared_Phi']=packet_report(cfg,packet_data,spatial[-1])
    report['BO_material_repair']=born_oppenheimer_repair(cfg,spatial_sizes[-1])
    report['physical_centre_Gauss']=dict(actual_even_boxes=centre,
        ignored_odd_volume_adverse_control=global_centre_projection(cfg,3),
        repair='Even physical box, filled NH=2V and all64 charged valence seas retained; full local-Gauss packet owned by Field. Old odd-box Hessians are mathematical diagnostics only.')
    if with_response:
        response=compute_response(response_config(cfg,data,orders))
        # Producer metadata is historical; attach new mean provenance without retagging producer source.
        response['metric']['frame_provenance']='Actual renewed common-lattice weighted10Gram uniform minimizer in induced_phi_backreaction.py; not a full quantum phase'
        response['assumptions']['vacuum']='Actual renewed uniform mean with all mass-resolved filled seas; quantum Phi confinement/dressing debt explicitly separately priced'
        response['assigned_plan_epoch']='beef5f7dadf30f276f87f89547a4ac68b8bf4f8f101abe3ffd89c82a9c8b58ec'
        response['Phi_supplier']=dict(source_at_import=IMPORT_HASHES,
            microscopic=asdict(cfg),uniform=report['uniform'][-1],prepared_Phi=report['prepared_Phi'])
        report['response']=response
        report['Compton']=transfer(response)
        dynamic_model=SameCurrent(response,dynamic=True)
        report['dynamic_Compton_probe']=dynamic_model.observe(.4,1.3,np.ones(3)/np.sqrt(3))
        repair=report['BO_material_repair']
        repair_data=dict(G=np.asarray(repair['G_centre']))
        repair_response=compute_response(response_config(cfg,repair_data,orders))
        repair_response['metric']['frame_provenance']='Even finiteN'+str(spatial_sizes[-1])+' heavy-dominant mean with actual64-charged strong-convex Gram interval, explicit finiteLambda2**816 original-family repair'
        repair_response['Phi_supplier']=repair
        report['BO_response']=repair_response
        report['BO_Compton']=transfer(repair_response)
        reference_path=ROOT/'induced_photon_response_mass_resolved_receipt.json'
        reference_bytes=reference_path.read_bytes()
        reference=json.loads(reference_bytes)
        report['preserved_old_mean_response_comparison']=dict(
            path=str(reference_path),sha256=hashlib.sha256(reference_bytes).hexdigest(),
            scope='Reference and current microscopic descriptors are explicit; spacing and renewed mean may both change. Converged integration orders retained; no velocity or current refit.',
            old_config=reference['config'],new_config=response['config'],
            old_orders=reference['config']['orders'],new_orders=list(orders),
            deltas={key:dict(old=reference[key],new=response[key],
                            absolute=response[key]-reference[key],relative=response[key]/reference[key]-1)
                    for key in ('epsilon','kappa','photon_speed','charged_cone','speed_ratio')})
    report['input_sha256']=hashlib.sha256(json.dumps(report['input_descriptor'],sort_keys=True).encode()).hexdigest()
    report['elapsed_seconds']=time.monotonic()-start
    report['source_unchanged_since_import']={name:IMPORT_HASHES[name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
    return report


def main():
    parser=argparse.ArgumentParser()
    started=time.monotonic()
    parser.add_argument('--sizes',default='9,17,33')
    parser.add_argument('--spatial-sizes',default='4,6')
    parser.add_argument('--orders',default='6,8')
    parser.add_argument('--response',action='store_true')
    parser.add_argument('--supplier-output')
    parser.add_argument('--output',required=True)
    parser.add_argument('--joint-prepared',action='store_true')
    parser.add_argument('--prepared-size',type=int,default=6)
    parser.add_argument('--Lambda-exponent',type=int,default=2048)
    parser.add_argument('--cutoff-bits',type=int,default=512)
    parser.add_argument('--clock',type=float,default=4096.)
    args=parser.parse_args()
    if args.joint_prepared:
        result=joint_prepared_report(PhiConfig(),args.prepared_size,args.Lambda_exponent,
            args.cutoff_bits,args.clock)
        result['input_descriptor']['response_benchmark']=dict(enabled=args.response,
            orders=tuple(map(int,args.orders.split(','))) if args.response else (),
            scope='Bulk-BZ RPA benchmark at finite-box supplied mean, not that box photon vacuum or a macrobox stationary state')
        result['input_sha256']=hashlib.sha256(json.dumps(result['input_descriptor'],sort_keys=True).encode()).hexdigest()
        if args.response:
            response=compute_response(response_config(PhiConfig(),
                dict(G=np.asarray(result['quantum_mean']['G_float'])),
                tuple(map(int,args.orders.split(',')))))
            response['metric']['frame_provenance']='New exact-decimal finite-box full-normal Gauss packet mean; binary64 current encoding; bulk RPA population is explicitly a separate diagnostic'
            response['assumptions']['vacuum']='Conditional full negative-band bulk RPA at supplied mean; Phi/colour finite-box transfer does not admit this photon vacuum'
            response['Phi_supplier']=dict(input_sha256=result['input_sha256'],
                mean=result['quantum_mean'],prices=result['prices'],
                scope=result['limitations'],source_at_import=IMPORT_HASHES)
            result['response']=response
            result['Compton']=transfer(response)
    else:
        result=run(PhiConfig(),tuple(map(int,args.sizes.split(','))),
            tuple(map(int,args.spatial_sizes.split(','))),tuple(map(int,args.orders.split(','))),args.response)
    result['elapsed_seconds']=time.monotonic()-started
    result['source_unchanged_since_import']={name:IMPORT_HASHES[name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.supplier_output:
        Path(args.supplier_output).write_text(json.dumps(result['response'],indent=2,allow_nan=False)+'\n')
    summary=dict(output=args.output,source_at_import=IMPORT_HASHES,
        input_sha256=result['input_sha256'],elapsed_seconds=result['elapsed_seconds'])
    if args.joint_prepared:
        summary['prices']=result['prices']
    else:
        summary.update(g0=result['uniform'][-1]['g0'],gs=result['uniform'][-1]['gs'],
            normal_gaps=[s['normal_gap'] for s in result['spatial']],
            PH_period_action=result['prepared_Phi']['frozen_sea_PH_action_at_one_normal_period'])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
