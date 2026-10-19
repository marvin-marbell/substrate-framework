#!/usr/bin/env python3
"""Actual primitive-coefficient heavy sea on all 1024 centre gauge classes of a cube."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.linalg import block_diag, expm

from full_band_gauss_section import FullBandGaussSection
from full_field_charge_bound import open_box, primitive_edges
from local_orbital_stability import scalar_paths, uniform_data
from rotation_band import HOPPING, MIXING, RotationBand
from sea_selection import matrix_unit_basis

SOURCE_NAMES = ('centre_flux_sea.py', 'centre_flux_sea.md', 'full_band_gauss_section.py',
                'full_field_charge_bound.py', 'local_orbital_stability.py', 'rotation_band.py',
                'sea_selection.py', 'bosonic_statistics.py', 'quantum_su4_gauss.py',
                'pauli_compton.py',
                'quantum_spinor_plaquette.py', 'gauss_charge_intertwiner.py', 'soft_budget.py')


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def check(value, name, tolerance=2e-9):
    if not np.isfinite(value) or value > tolerance:
        raise AssertionError(f'{name}: {value}')


def centre(k):
    return (1., 1j, -1., -1j)[int(k) % 4]


def tree_edges(graph):
    reached, tree = {0}, []
    while len(reached) < graph.volume:
        for e, (x, y) in enumerate(graph.bonds):
            if (x in reached) != (y in reached):
                tree.append(e); reached.update((x, y))
                break
        else:
            raise ValueError('Connected graph required')
    return tuple(tree)


def tree_fix(graph, exponents, tree):
    """k_xy -> k_xy+g_x-g_y; fix root g_0=0 and every tree edge to zero."""
    gauge = {0: 0}
    while len(gauge) < graph.volume:
        for e in tree:
            x, y = graph.bonds[e]
            if x in gauge and y not in gauge:
                gauge[y] = (gauge[x] + int(exponents[e])) % 4
            elif y in gauge and x not in gauge:
                gauge[x] = (gauge[y] - int(exponents[e])) % 4
    return tuple((int(k) + gauge[x] - gauge[y]) % 4
                 for k, (x, y) in zip(exponents, graph.bonds))


def faces(graph):
    index = {site: x for x, site in enumerate(graph.sites)}
    result = []
    for i, j in itertools.combinations(range(3), 2):
        other = next(a for a in range(3) if a not in (i, j))
        for side in (0, 1):
            origin = np.zeros(3, int); origin[other] = side
            ei, ej = np.eye(3, dtype=int)[[i, j]]
            corners = tuple(index[tuple(p)] for p in (origin, origin + ei, origin + ei + ej, origin + ej))
            result.append((i, j, other, side, corners))
    return tuple(result)


def fluxes(graph, exponents):
    directed = {(x, y): int(k) for (x, y), k in zip(graph.bonds, exponents)}
    directed.update({(y, x): -int(k) for (x, y), k in zip(graph.bonds, exponents)})
    return tuple(sum(directed[a, b] for a, b in zip(corners, corners[1:] + corners[:1])) % 4
                 for _, _, _, _, corners in faces(graph))


def onebody(graph, band, fields, links):
    """Full three-axis onsite, every actual directed scalar path and its adjoint.

    Links are genuine SU4 matrices indexed in graph.bonds' positive orientation.
    Endpoint orbital matrices and ordered primitive transport are never replaced
    by scalar hopping magnitudes. Heavy species has no EM path factor.
    """
    onsite, edges = primitive_edges(graph, band)
    fields = np.asarray(fields, complex)
    if fields.shape != (graph.volume, 4, 4, 4):
        raise ValueError('Four Hermitian orbital fields per graph site required')
    transports = {(x, y): np.asarray(u, complex) for (x, y), u in zip(graph.bonds, links)}
    transports.update({(y, x): u.conj().T for (x, y), u in list(transports.items())})
    index = {site: x for x, site in enumerate(graph.sites)}
    h = np.zeros((4 * graph.volume, 4 * graph.volume), complex)
    for x in range(graph.volume):
        h[4*x:4*x+4, 4*x:4*x+4] = onsite * fields[x, 0]
    for edge in edges:
        displacement = np.asarray(edge.displacement)
        axes = np.flatnonzero(displacement)
        if len(axes) != 1:
            raise ValueError('Actual supplier only declares straight primitive paths')
        axis = int(axes[0]); step = int(np.sign(displacement[axis]))
        u, position, previous = np.eye(4, dtype=complex), np.array(graph.sites[edge.x]), edge.x
        for _ in range(abs(int(displacement[axis]))):
            position[axis] += step
            following = index[tuple(position)]
            u = u @ transports[previous, following]
            previous = following
        left = np.einsum('m,mij->ij', edge.coefficients, fields[edge.x])
        right = np.einsum('m,mij->ij', edge.coefficients, fields[edge.y])
        coupling = (left @ u + u @ right) / 2
        sx, sy = slice(4*edge.x, 4*edge.x+4), slice(4*edge.y, 4*edge.y+4)
        h[sx, sy] += coupling; h[sy, sx] += coupling.conj().T
    return h


def heavy_energy(h, scale=1.):
    eigenvalues = np.linalg.eigvalsh(h)
    return -scale * float(eigenvalues[len(eigenvalues)//2:].sum()), eigenvalues


def primitive_pair_control(band, frame):
    """Bind spectral filling to actual 4900-state Pauli words, not ladder stubs.

    The existing pair owner has onsite m+r, whereas a three-axis open graph
    retains m+3r. Correct ONLY this explicit geometry difference in the comparator.
    """
    model = FullBandGaussSection(band, heavy_scale=1.)
    graph = open_box((2, 1, 1)); fields = np.tile(frame, (2, 1, 1, 1))
    correction = 2 * band.r / band.a * block_diag(frame[0], frame[0])
    rows = []
    for exponent in range(4):
        link = centre(exponent) * np.eye(4)
        section = model.state(fields, link, (.137, -.081))
        hp = section.onebody[0]
        observed = -float(np.vdot(section.state, model.lift(hp, 0) @ section.state).real)
        predicted, _ = heavy_energy(hp)
        residual = float(np.linalg.norm(-model.lift(hp, 0) @ section.state - predicted * section.state))
        graph_h = onebody(graph, band, fields, (link,))
        coefficient_error = float(np.linalg.norm(graph_h - hp - correction))
        check(abs(observed - predicted), 'actual primitive occupied pair energy')
        check(residual, 'actual primitive occupied pair eigenvector')
        check(coefficient_error, 'pair and graph coefficient geometry binding')
        rows.append(dict(centre_exponent=exponent, primitive_energy=observed,
                         top_four_prediction=predicted, eigen_residual=residual,
                         three_axis_minus_pair_onsite_coefficient=2*band.r/band.a,
                         graph_coefficient_error=coefficient_error, projection_weight=section.weight))
    return rows


def loop_fourth(h, mass):
    """Analytic fourth coefficient of -Tr sqrt(h(t)^2)/2, h(t)=A beta+tT."""
    beta = np.kron(np.eye(len(h)//4), MIXING)
    t = h - mass * beta
    x = mass * (beta @ t + t @ beta); y = t @ t
    return float((np.trace(y @ y)/(16*mass**3)
                  - np.trace(x@x@y + x@y@x + y@x@x)/(32*mass**5)
                  + 5*np.trace(x@x@x@x)/(256*mass**7)).real)


def magnetic_onebody(band, frame, link_exponents, momentum):
    """Period-two centre background: actual 32-band magnetic Bloch consumer.

    Each U_i(x) is independent of x_i and repeats the cube's positive bond.
    For uniform pi flux this is gauge-equivalent to U_i=(-1)^sum_(j<i)x_j.
    Bloch convention exp(i k dot cell), k conjugate to the two-site supercell.
    """
    graph = open_box((2, 2, 2)); index = {p: x for x, p in enumerate(graph.sites)}
    phases = {}
    for (x, y), exponent in zip(graph.bonds, link_exponents):
        axis = next(i for i in range(3) if graph.sites[x][i] != graph.sites[y][i])
        phases[axis, graph.sites[x]] = centre(exponent)
    onsite, paths = scalar_paths(band)
    h = np.kron(np.eye(8), onsite*frame[0])
    for x, position in enumerate(graph.sites):
        for displacement, coefficients in paths:
            raw = np.array(position) + displacement
            target = tuple(raw % 2); winding = raw // 2
            axis = int(np.flatnonzero(displacement)[0])
            start = list(position); start[axis] = 0
            transport = phases[axis, tuple(start)]
            if int(displacement[axis]) != 1:
                raise ValueError('Magnetic consumer uses the selected unimproved law')
            coupling = transport * np.exp(1j*np.dot(momentum, winding)) * np.einsum('m,mij->ij', coefficients, frame)
            y = index[target]; sx, sy = slice(4*x,4*x+4), slice(4*y,4*y+4)
            h[sx,sy] += coupling; h[sy,sx] += coupling.conj().T
    return h


def cubic_orbit(graph, exponents):
    """All 24 proper cubic rotations, retaining directed link orientation."""
    index = {p: x for x, p in enumerate(graph.sites)}
    result = []
    for permutation in itertools.permutations(range(3)):
        parity = (-1)**sum(permutation[i] > permutation[j] for i in range(3) for j in range(i+1,3))
        for signs in itertools.product((-1,1), repeat=3):
            if parity*np.prod(signs) != 1:
                continue
            image = {x: index[tuple(p[permutation[i]] if signs[i] == 1 else 1-p[permutation[i]] for i in range(3))]
                     for x,p in enumerate(graph.sites)}
            directed = {(image[x],image[y]): k for (x,y),k in zip(graph.bonds,exponents)}
            directed.update({(y,x): -k for (x,y),k in list(directed.items())})
            result.append(tuple(directed[bond] % 4 for bond in graph.bonds))
    return result


def analyse(*, m=.2, r=.12, v=.3, stiffness=2., heavy_scale=1., K4=0.):
    if not all(np.isfinite(x) for x in (m,r,v,stiffness,heavy_scale,K4)) or min(m,stiffness,heavy_scale) <= 0 or min(r,K4) < 0:
        raise ValueError('Positive mass, stiffness, scale; nonnegative r and K4 required')
    band = RotationBand(a=1., m=m, r=r, v=v, improved=False)
    data = uniform_data(band, 3, stiffness, 0., matrix_unit_basis())
    frame = np.asarray(data['frame'], dtype=np.complex128)
    b, s = np.sqrt(np.diag(data['selected']['G']))[[0,1]]
    check(float(np.linalg.norm(frame - np.array([b*MIXING, *(s*a for a in HOPPING)]))), 'shared selected frame')
    graph = open_box((2,2,2)); tree = tree_edges(graph)
    chords = tuple(i for i in range(len(graph.bonds)) if i not in tree)
    fields = np.tile(frame,(8,1,1,1)); onsite,_ = scalar_paths(band)
    mass = onsite*b
    beta = np.kron(np.eye(8),MIXING)
    rows, configurations, fourth = [], [], []
    unit_heavy_energies, bare_costs = [], []
    comparison_scale = max(heavy_scale, K4)
    covariance = bianchi = oriented = 0.; min_gap = float('inf'); hopping_bound = 0.
    gauge = np.array([(3*x+1)%4 for x in range(8)])
    gauge_matrix = np.kron(np.diag([centre(k) for k in gauge]), np.eye(4))
    for labels in itertools.product(range(4),repeat=len(chords)):
        exponents = np.zeros(12,int); exponents[list(chords)] = labels
        links = [centre(k)*np.eye(4) for k in exponents]
        h = onebody(graph,band,fields,links)
        unit_energy,spectrum = heavy_energy(h,1.)
        energy = heavy_scale * unit_energy
        flux = fluxes(graph,exponents)
        changed = tuple((int(k)+gauge[x]-gauge[y])%4 for k,(x,y) in zip(exponents,graph.bonds))
        hc = onebody(graph,band,fields,[centre(k)*np.eye(4) for k in changed])
        covariance = max(covariance,float(np.linalg.norm(hc-gauge_matrix@h@gauge_matrix.conj().T)))
        oriented = max(oriented,float(max(abs((a-b)%4) for a,b in zip(flux,fluxes(graph,changed)))))
        # Face ordering: xy(z0,z1), xz(y0,y1), yz(x0,x1).
        bianchi = max(bianchi,float((flux[5]-flux[4]-flux[3]+flux[2]+flux[1]-flux[0])%4))
        if tree_fix(graph,changed,tree) != tuple(exponents):
            raise AssertionError('Tree gauge canonical representative mismatch')
        check(float(np.linalg.norm(h-h.conj().T)),'whole-H Hermiticity')
        check(abs(float(np.trace(h).real)),'whole-H trace')
        if np.count_nonzero(spectrum > 0) != 16 or np.count_nonzero(spectrum < 0) != 16:
            raise AssertionError('Positive Wilson inertia lost')
        min_gap = min(min_gap,float(np.min(abs(spectrum))))
        hopping_bound = max(hopping_bound,float(np.linalg.norm(h-mass*beta,2)))
        # Re(i**k) is exact on Z4; trig roundoff must not break cubic symmetry.
        bare_cost = sum((0,1,2,1)[k] for k in flux)
        bare = K4 * bare_cost
        unit_heavy_energies.append(unit_energy); bare_costs.append(bare_cost)
        rows.append(dict(chord_exponents=list(labels), flux_exponents=list(flux),
                         heavy_energy=energy, bare_magnetic_energy=float(bare), total_energy=energy+bare,
                         negative_count=16, positive_count=16, onebody_gap=float(np.min(abs(spectrum)))))
        configurations.append(tuple(int(k) for k in exponents)); fourth.append(loop_fourth(h,mass))
    check(covariance,'all-class gauge covariance'); check(bianchi,'oriented cube Bianchi'); check(oriented,'all-class flux gauge invariance')
    if len(set(tuple(row['flux_exponents']) for row in rows)) != 1024:
        raise AssertionError('Six Bianchi-constrained fluxes do not separate all gauge classes')
    check(max(0.,m*b-min_gap),'global analytic positive-mass bound')
    energies = np.array([row['total_energy'] for row in rows])
    unit_heavy = np.asarray(unit_heavy_energies)
    comparison_energies = (heavy_scale / comparison_scale) * unit_heavy + (
        K4 / comparison_scale) * np.asarray(bare_costs)
    best_comparison = float(comparison_energies.min())
    best = float(energies[np.argmin(comparison_energies)])
    tolerance = 1e-10 * abs(best_comparison)
    winners = np.flatnonzero(comparison_energies <= best_comparison + tolerance)
    lookup = {configuration:i for i,configuration in enumerate(configurations)}
    orbit_indices = sorted({lookup[tree_fix(graph,k,tree)] for i in winners for k in cubic_orbit(graph,configurations[i])})
    # K4 plaquette scalar and selected orbital vector transform together under proper rotations.
    check(max(abs(float(comparison_energies[i])-best_comparison) for i in orbit_indices),'proper cubic minimizer orbit')
    adverse_index = 433
    adverse_orbit = [lookup[tree_fix(graph,k,tree)] for k in cubic_orbit(graph,configurations[adverse_index])]
    rotation_error = max(abs(float(energies[i]-energies[adverse_index])) for i in adverse_orbit)
    unit_rotation_error = max(abs(float(unit_heavy[i]-unit_heavy[adverse_index])) for i in adverse_orbit)
    check(unit_rotation_error,'nonflat proper cubic orbit covariance per heavy scale')
    if any(bare_costs[i] != bare_costs[adverse_index] for i in adverse_orbit):
        raise AssertionError('Nonflat proper cubic orbit changed the exact Z4 plaquette cost')
    sectors = {}
    for i,row in enumerate(rows):
        key = tuple(sorted(row['flux_exponents']))
        sectors.setdefault(key,[]).append(i)
    sector_rows = [dict(sorted_flux_exponents=list(key), classes=len(indices),
                        minimum_energy=float(energies[indices].min()), gap_above_global=float(energies[indices].min()-best))
                   for key,indices in sorted(sectors.items())]
    above = comparison_energies[comparison_energies > best_comparison + tolerance]
    heavy_only = np.array([row['heavy_energy'] for row in rows]); flat = float(heavy_only[0])
    pi_index = next(i for i,row in enumerate(rows) if row['flux_exponents'] == [2]*6)
    # Exact fourth-order closed-walk coefficient decomposed into elementary face characters.
    design = np.array([[1.,*[np.cos(np.pi*k/2) for k in row['flux_exponents']]] for row in rows])
    fit,*_ = np.linalg.lstsq(design,np.array(fourth),rcond=None)
    loop_error = float(np.max(abs(design@fit-np.array(fourth))))
    check(loop_error,'analytic fourth-order elementary loop decomposition')
    ratio = hopping_bound/mass
    eta = min(1., .25/ratio) if ratio else 1.
    q = 2*eta*ratio+(eta*ratio)**2
    remainder = 16*mass*q**5/(1-q)  # |binom(1/2,n)| <=1, n>=5, square-root series tail.
    # Noncentral SU4 endpoint covariance rejects replacing Phi Gamma by scalar norms.
    basis = matrix_unit_basis() / np.sqrt(8)
    local_gauge = [expm(1j*np.einsum('a,aij->ij', .13*np.sin(np.arange(15)*.37+x*.29), basis))
                   for x in range(8)]
    rotated_fields = np.array([g@frame@g.conj().T for g in local_gauge])
    rotated_links = [local_gauge[x]@local_gauge[y].conj().T for x,y in graph.bonds]
    flat_h = onebody(graph,band,fields,[np.eye(4)]*12)
    full_gauge = np.asarray(block_diag(*local_gauge), dtype=np.complex128)
    noncentral_covariance = float(np.linalg.norm(
        onebody(graph,band,rotated_fields,rotated_links)-full_gauge@flat_h@full_gauge.conj().T))
    check(noncentral_covariance,'noncentral local SU4 endpoint covariance')
    # Construct the standard pi gauge and bind it to the enumerated unique pi class.
    standard_pi = []
    for x,y in graph.bonds:
        axis = next(i for i in range(3) if graph.sites[x][i] != graph.sites[y][i])
        standard_pi.append(2*sum(graph.sites[x][:axis]) % 4)
    if tree_fix(graph,standard_pi,tree) != configurations[pi_index]:
        raise AssertionError('Uniform pi class disagrees with standard magnetic gauge')
    # Check all oriented period-two plaquettes, including seams, not just open faces.
    bond_lookup = {}
    for (x,y),k in zip(graph.bonds,standard_pi):
        axis = next(i for i in range(3) if graph.sites[x][i] != graph.sites[y][i])
        bond_lookup[axis,graph.sites[x]] = k
    def periodic_link(axis, position):
        p = list(np.asarray(position) % 2); p[axis] = 0
        return bond_lookup[axis,tuple(p)]
    seam_fluxes = []
    for position in graph.sites:
        for i,j in itertools.combinations(range(3),2):
            ei,ej = np.eye(3,dtype=int)[[i,j]]
            seam_fluxes.append((periodic_link(i,position)+periodic_link(j,np.array(position)+ei)
                                -periodic_link(i,np.array(position)+ej)-periodic_link(j,position)) % 4)
    if seam_fluxes != [2]*24:
        raise AssertionError('Uniform pi extension lost flux on a periodic seam')
    site_index = {p:x for x,p in enumerate(graph.sites)}
    translations = []
    for axis in range(3):
        operator = np.zeros((8,8),complex)
        for x,position in enumerate(graph.sites):
            target = list(position); target[axis] = (target[axis]+1) % 2
            operator[x,site_index[(target[0],target[1],target[2])]] = (-1)**sum(position[axis+1:])
        translations.append(np.kron(operator,np.eye(4)))
    pi_zero = magnetic_onebody(band,frame,standard_pi,(0.,0.,0.))
    translation_symmetry = max(float(np.linalg.norm(t@pi_zero-pi_zero@t)) for t in translations)
    translation_projectivity = max(float(np.linalg.norm(translations[i]@translations[j]+translations[j]@translations[i]))
                                   for i,j in itertools.combinations(range(3),2))
    check(translation_symmetry,'actual uniform pi magnetic translation symmetry')
    check(translation_projectivity,'actual uniform pi magnetic translation algebra')
    effective = RotationBand(a=1.,m=m*b,r=r*b,v=v*s,improved=False)
    folding_error = 0.; pi_bands = []
    # Actual finite magnetic consumer for the selected cube minimum, no four-band flat substitution.
    chosen = int(winners[0]); magnetic_rows = []
    for momentum in ((0.,0.,0.),(.19,-.27,.31),(np.pi,0.,0.),(np.pi,np.pi,np.pi)):
        hm = magnetic_onebody(band,frame,configurations[chosen],momentum)
        em,ev = heavy_energy(hm,heavy_scale)
        check(max(0.,m*b-float(np.min(abs(ev)))),'magnetic Bloch positive Wilson bound')
        magnetic_rows.append(dict(supercell_momentum=list(momentum), heavy_energy=em,
                                  eigenvalues=ev.tolist(), gap=float(np.min(abs(ev)))))
        flat_bloch = magnetic_onebody(band,frame,configurations[0],momentum)
        folded = np.sort(np.concatenate([np.linalg.eigvalsh(effective.h((np.array(momentum)+2*np.pi*np.array(n))/2))
                                        for n in itertools.product((0,1),repeat=3)]))
        folding_error = max(folding_error,float(np.linalg.norm(np.linalg.eigvalsh(flat_bloch)-folded)))
        pi_h = magnetic_onebody(band,frame,standard_pi,momentum)
        pi_energy,pi_ev = heavy_energy(pi_h,heavy_scale)
        check(max(0.,m*b-float(np.min(abs(pi_ev)))),'uniform pi magnetic Bloch gap')
        pi_bands.append(dict(supercell_momentum=list(momentum),heavy_energy=pi_energy,
                             eigenvalues=pi_ev.tolist(),gap=float(np.min(abs(pi_ev)))))
    check(folding_error,'flat magnetic consumer equals actual folded four-band spectrum')
    return dict(schema=1,source_sha256=source_hashes(),
                governing_plan=dict(id='issue-222-continuing-backward-20260930',revision='6948b8f0b38c3ffd396e76b82faa1b34fe7ee481587e4f5bd2890d5e1abf37f7',work_item='campaign'),
                parameters=dict(m=m,r=r,v=v,stiffness=stiffness,heavy_scale=heavy_scale,K4=K4,a=1.,improved=False),
                selected_frame=dict(real=frame.real.tolist(),imaginary=frame.imag.tolist()),
                selected_gram_diagonal=np.diag(data['selected']['G']).tolist(),
                selected_stationarity_residual=float(data['selected']['gradient_residual']),
                graph=dict(shape=[2,2,2],sites=graph.sites,bonds=graph.bonds,tree_bond_indices=tree,chord_bond_indices=chords,
                           faces=[dict(axes=[i,j],normal_axis=k,side=side,corners=corners) for i,j,k,side,corners in faces(graph)],
                           directed_primitive_paths=len(primitive_edges(graph,band)[1]),onsite_coefficient=onsite),
                quantum_law='Declared primitive Pauli statistics; frozen selected orbital fields; heavy EM-neutral whole Hamiltonian -Lambda*h; canonical N=16 of32',
                controls=dict(gauge_covariance_residual=covariance,oriented_flux_gauge_residual=oriented,Bianchi_residual=bianchi,
                              distinct_flux_gauge_classes=1024,proper_rotation_orbit_indices=orbit_indices,
                              noncentral_SU4_endpoint_covariance_residual=noncentral_covariance,
                              nonflat_cubic_covariance_residual=rotation_error,
                              nonflat_cubic_covariance_per_heavy_scale=unit_rotation_error,
                              uniform_pi_translation_commutator_residual=translation_symmetry,
                              uniform_pi_translation_anticommutator_residual=translation_projectivity,
                              flat_Bloch_folding_residual=folding_error,uniform_pi_periodic_fluxes=seam_fluxes,
                              primitive_pair=primitive_pair_control(band,frame)),
                global_static_gap=dict(analytic_lower=m*b,all_classes_measured_minimum=min_gap,
                                       half_filled_cluster_separation_lower=2*m*b,proof='Positive Wilson mass Schur complement and inertia; not hopping-norm or eigenvalue-pairing assumption'),
                minimizer_indices=winners.tolist(),minimum_energy=best,
                next_energy_gap=float((above.min()-best_comparison)*comparison_scale) if len(above) else None,
                energy_comparison=dict(scale=comparison_scale,normalized_minimum=best_comparison,
                                       normalized_tolerance=tolerance,
                                       normalized_next_gap=float(above.min()-best_comparison) if len(above) else None,
                                       units='Physical energy = scale times normalized comparison energy; physical gap may underflow binary64'),
                minimizer_link_exponents=[list(configurations[i]) for i in winners],
                minimizer_flux_exponents=[rows[i]['flux_exponents'] for i in winners],flux_sectors=sector_rows,
                comparison=dict(flat_heavy_energy=flat,uniform_pi_heavy_energy=float(heavy_only[pi_index]),
                                uniform_pi_minus_flat=float(heavy_only[pi_index]-flat),
                                uniform_pi_bare_minus_flat=12*K4,
                                pi_flat_crossing_K4_per_heavy_scale=float((unit_heavy[0]-unit_heavy[pi_index])/12)),
                closed_loop_expansion=dict(onsite_mass=mass,hopping_operator_norm_uniform=hopping_bound,
                                          full_hopping_square_root_series_q_bound=2*ratio+ratio**2,
                                          controlled_homotopy_eta=eta,controlled_q_bound=q,
                                          square_root_tail_after_four_powers_upper=remainder,
                                          fourth_order_constant=float(fit[0]),per_oriented_face_cosine_coefficients=fit[1:].tolist(),
                                          fourth_order_decomposition_residual=loop_error,
                                          meaning='Exact t^4 coefficient for h(t)=A beta+t T, not a controlled fourth-order approximation at t=1 unless q<1; tail bound is after powers of X(t), not powers of t'),
                magnetic_consumer=dict(cell_shape=[2,2,2],orbitals=32,chosen_cube_class=chosen,
                                       link_extension='U_i(x)=cube U_i(x with x_i=0), other coordinates mod2',
                                       sampled_bands=magnetic_rows,
                                       uniform_pi_candidate_bands=pi_bands,
                                       uniform_pi_standard_link_exponents=standard_pi,
                                       translation_algebra='Uniform pi: T_i T_j=-T_j T_i; translations combined with centre gauge. General chosen flux: retain period-two supercell, no ordinary four-orbital translation claim',
                                       spatial_rotations='Proper cube rotations preserve minimizer orbit; uniform pi flux invariant modulo4, up to centre gauge. Diagonal orbital vector rotation only; no mechanical half-spin inference'),
                enumeration=rows,
                limits=['1024 finite open-cube comparison; no thermodynamic flux phase or deconfinement theorem',
                        'Frozen field static sea, not full matter-field angular spectrum or quantum link ground',
                        'Light charge and compact photon consumer not integrated; uniform pi invalidates naive ordinary four-band flat momentum map',
                        'Period-two Bloch extension is an actual candidate consumer, not a proved infinite-volume minimizer',
                        'No Lorentz/Dirac continuum, mechanical half-spin, full QED or phase conclusion'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    for name,default in (('m',.2),('r',.12),('v',.3),('stiffness',2.),('heavy-scale',1.),('K4',0.)):
        parser.add_argument('--'+name,type=float,default=default)
    args=parser.parse_args()
    result=analyse(m=args.m,r=args.r,v=args.v,stiffness=args.stiffness,heavy_scale=args.heavy_scale,K4=args.K4)
    payload=json.dumps(result,indent=2,allow_nan=False)
    args.output.write_text(payload+'\n')
    print(json.dumps({key:result[key] for key in ('source_sha256','minimum_energy','next_energy_gap','energy_comparison','minimizer_flux_exponents','comparison','global_static_gap','closed_loop_expansion','controls','limits')},indent=2))


if __name__ == '__main__':
    main()
