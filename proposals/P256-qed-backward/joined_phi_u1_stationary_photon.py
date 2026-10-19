"""N4 Bloch0 onsite-reference ring and interacting electric-loop compression.

No Maxwell term, vacuum adoption, pole fit, or initial-time Taylor inference.
The finite compression is a Rayleigh-Ritz diagnostic, not a full-Phi ground.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import permutations
import json
from pathlib import Path
import time
import zipfile

import induced_phi_nonuniform_field_car as old
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PLAN = 'issue-222-induced-photon-delta13-continuation-20261008'
ASSIGNED_REVISION = '3a1f07a384a7208fe0056b4d597a59b92bca6314d1746b9fa85a6b3fc7717721'
BOOKKEEPING_REVISION = '9ea88239ef1b96d1414978150fd9e2c919e29de24d91834808068dfd13a33340'
SOURCE_BYTES = {**old.CAPTURED, Path(__file__).name: Path(__file__).read_bytes()}


def car(mask: int, create: int, annihilate: int) -> tuple[int, int] | None:
    if not (mask >> annihilate) & 1:
        return None
    sign = -1 if (mask & ((1 << annihilate)-1)).bit_count() % 2 else 1
    mask ^= 1 << annihilate
    if (mask >> create) & 1:
        return None
    sign *= -1 if (mask & ((1 << create)-1)).bit_count() % 2 else 1
    return mask | (1 << create), sign


def reference() -> int:
    return sum(1 << (4*x+s) for x in range(4) for s in (0, 1))


def density(mask: int) -> np.ndarray:
    return np.array([((mask >> (4*x)) & 15).bit_count()-2 for x in range(4)], int)


def loop(data: dict, lengths: tuple[int, int], base: tuple[int, int, int]) -> dict:
    native = data['native']
    sites = native['sites']
    offsets = [(0, 0, 0), (lengths[0], 0, 0), (lengths[0], lengths[1], 0), (0, lengths[1], 0)]
    vertices = [int(np.flatnonzero(np.all(sites == (np.array(base)+v) % 4, axis=1))[0]) for v in offsets]
    rows = []
    for profile in data['profiles'][1:]:
        onsite, rotation = np.linalg.eigh(float(profile['onsite'])*native['frame'][0])
        if not (onsite[1] < 0 < onsite[2]):
            raise ArithmeticError('Legal negative onsite reference must have rank2')
        paths = old.native_paths(profile, native)
        hops, edges, seams = [], [], []
        for j in range(4):
            x, y = vertices[j], vertices[(j+1) % 4]
            candidates = [(p, 1) for p in paths if p['source'] == x and p['target'] == y]
            candidates += [(p, -1) for p in paths if p['source'] == y and p['target'] == x]
            # On N4 a length2 path has two native arcs. Choose the geometrical
            # rectangle arc, keeping its actual AP seam; the other arc is not
            # silently merged into this reduced loop.
            direction = 1 if j < 2 else -1
            candidates = [(p, s) for p, s in candidates if s == direction]
            if len(candidates) != 1:
                raise ArithmeticError('Unique declared native rectangle arc required')
            p, sign = candidates[0]
            matrix = p['seam']*np.einsum('m,mij->ij', p['coefficients'], native['frame'])
            if sign == -1:
                matrix = matrix.conj().T
            hops.append(rotation.conj().T@matrix@rotation)
            flux = np.zeros(192)
            flux[list(p['edges'])] = sign
            edges.append(flux)
            seams.append(int(p['seam']))
        edge_vectors = np.asarray(edges)
        rows.append(dict(load=int(profile['load']), onsite=onsite, hops=hops,
                         modes=edge_vectors@native['Q'], edge_vectors=edge_vectors, seams=seams))
    incidence = np.zeros((4, 4), int)
    for j in range(4):
        incidence[j, j] = 1
        incidence[(j+1) % 4, j] = -1
    return dict(vertices=vertices, incidence=incidence, species=rows,
                coulomb=native['Lplus'][np.ix_(vertices, vertices)], lengths=lengths, base=base)


def terms(item: dict, edge: int) -> list[tuple[int, int, complex]]:
    return [(4*edge+i, 4*((edge+1) % 4)+j, complex(item['hops'][edge][i, j]))
            for i in range(4) for j in range(4) if abs(item['hops'][edge][i, j]) > 1e-13]


def energy(mask: int, flux: np.ndarray, item: dict, geometry: dict,
           omit: str = '') -> tuple[float, float, float]:
    onsite = sum(item['onsite'][s]*(int(bool(mask & (1 << (4*x+s))))-int(s < 2))
                 for x in range(4) for s in range(4))
    rho = density(mask)
    electric = 64*float(np.linalg.norm(flux@item['modes'])**2)
    coulomb = 64*float(rho@geometry['coulomb']@rho)
    return float(onsite), 0. if omit == 'HE' else electric, 0. if omit == 'HC' else coulomb


def ring(item: dict, geometry: dict, omit: str = '', incoming_winding: int = 0) -> dict:
    total = 0j
    min_gap = float('inf')
    max_covariance = 0.
    witness = None
    for order in permutations(range(4)):
        amplitudes = {reference(): 1.+0j}
        flux = np.full(4, incoming_winding, float)
        for depth, edge in enumerate(order):
            flux[edge] += 1
            updated: dict[int, complex] = {}
            for mask, amp in amplitudes.items():
                for create, annihilate, coefficient in terms(item, edge):
                    transition = car(mask, create, annihilate)
                    if transition is None:
                        continue
                    result, sign = transition
                    updated[result] = updated.get(result, 0j)+amp*coefficient*(1 if omit == 'CAR' else sign)
            if depth < 3:
                for mask in list(updated):
                    pieces = energy(mask, flux, item, geometry, omit)
                    gap = sum(pieces)
                    if gap <= 0:
                        raise ArithmeticError('Virtual denominator is not positive')
                    min_gap = min(min_gap, gap)
                    updated[mask] /= -gap
                    individual = sum(energy(reference(), np.eye(4)[e], item, geometry)[1] for e in order[:depth+1])
                    individual_coulomb = sum(64*float(geometry['incidence'][:, e]@geometry['coulomb']@geometry['incidence'][:, e]) for e in order[:depth+1])
                    coulomb_cross = pieces[2]-individual_coulomb
                    covariance = pieces[1]-individual
                    if abs(covariance) > max_covariance:
                        max_covariance = abs(covariance)
                        witness = dict(order=order, depth=depth+1, rho=density(mask).tolist(),
                                       flux=flux.tolist(), onsite=pieces[0], HE=pieces[1], HC=pieces[2],
                                       electric_cross_term=covariance, Coulomb_cross_term=coulomb_cross)
            amplitudes = updated
        total += amplitudes.get(reference(), 0j)
    return dict(coefficient_real=float(total.real), coefficient_imag=float(total.imag),
                incoming_winding=incoming_winding, Feshbach_energy=0,
                minimum_virtual_gap=min_gap if np.isfinite(min_gap) else None, exact_configuration_cross_witness=witness,
                maximum_electric_cross_abs=max_covariance)


def flux_for(mask: int, winding: int, incidence: np.ndarray) -> np.ndarray:
    # Fix fourth edge; the first three are an integer spanning-tree solution.
    rho = density(mask)
    k = np.zeros(4, int)
    k[3] = winding
    k[:3] = np.rint(np.linalg.lstsq(incidence[:, :3], rho-incidence[:, 3]*winding, rcond=None)[0]).astype(int)
    if not np.array_equal(incidence@k, rho):
        raise ArithmeticError('Exact incidence/charge relation failed')
    return k


def reduced(geometry: dict, native: dict, winding_cut: int, omit: str = '') -> dict:
    # All onsite spin/CAR channels retained. One PH in either active or one
    # symmetric auxiliary flavour, never two simultaneous PHs; load63 appears
    # exactly as sqrt63 on vacuum transitions. Spectator occupation is2/site.
    ref = reference()
    masks_set: set[int] = set()
    for x in range(4):
        for y in range(4):
            for s in (2, 3):
                for t in (0, 1):
                    transition = car(ref, 4*x+s, 4*y+t)
                    if transition is None:
                        raise ArithmeticError('Occupied-to-empty onsite CAR transition missing')
                    masks_set.add(transition[0])
    masks = sorted(masks_set)
    basis = [(-1, ref, w) for w in range(-winding_cut, winding_cut+1)]
    basis += [(flavour, mask, w) for flavour in (0, 1) for mask in masks
              for w in range(-winding_cut, winding_cut+1)]
    lookup = {state: i for i, state in enumerate(basis)}
    n = len(basis)
    h = np.zeros((n, n), complex)
    q_indices = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    currents = [np.zeros_like(h) for _ in q_indices]
    contacts = [np.zeros_like(h) for _ in q_indices]
    klist = [flux_for(mask, w, geometry['incidence']) for _, mask, w in basis]
    field_phases = []
    for qi in q_indices:
        field = np.zeros(192, complex)
        q = 2*np.pi*np.array(qi)/4
        for e in range(192):
            if e % 3 == 1:
                field[e] = np.exp(1j*native['sites'][e//3]@q)/8
        field_phases.append(geometry['species'][0]['edge_vectors']@field)
    for column, (flavour, mask, winding) in enumerate(basis):
        k = klist[column]
        item = geometry['species'][max(flavour, 0)]
        h[column, column] = sum(energy(mask, k, item, geometry, omit))
        for species in ((0, 1) if flavour == -1 else (flavour,)):
            item = geometry['species'][species]
            for e in range(4):
                for direction in (1, -1):
                    for create, annihilate, coefficient in terms(item, e):
                        if direction == -1:
                            create, annihilate = annihilate, create
                            coefficient = coefficient.conjugate()
                        transition = car(mask, create, annihilate)
                        if transition is None:
                            continue
                        new_mask, sign = transition
                        new_k = k.copy()
                        new_k[e] += direction
                        new_flavour = -1 if new_mask == ref else species
                        state = (new_flavour, new_mask, int(new_k[3]))
                        row = lookup.get(state)
                        if row is None or not np.array_equal(klist[row], new_k):
                            continue
                        multiplicity = np.sqrt(item['load']) if flavour == -1 or new_flavour == -1 else 1.
                        value = sign*coefficient*multiplicity
                        h[row, column] += value
                        for j, phase in enumerate(field_phases):
                            vertex = direction*phase[e]
                            currents[j][row, column] += 1j*vertex*value
                            contacts[j][row, column] -= abs(vertex)**2*value
    hermitian = float(np.max(np.abs(h-h.conj().T)))
    if hermitian > 1e-10:
        raise ArithmeticError('Reduced operator is not Hermitian')
    values, vectors = eigh(h)
    ground = vectors[:, 0]
    gaps = values[1:]-values[0]
    residual = float(np.linalg.norm(h@ground-values[0]*ground))
    wilson = np.zeros_like(h)
    for column, (flavour, mask, w) in enumerate(basis):
        row = lookup.get((flavour, mask, w+1))
        if row is not None:
            wilson[row, column] = 1
    spectra = []
    for qi, current, contact in zip(q_indices, currents, contacts):
        plus = vectors[:, 1:].conj().T@current@ground
        minus = vectors[:, 1:].conj().T@current.conj().T@ground
        wp, wm = abs(plus)**2, abs(minus)**2
        diamagnetic = complex(np.vdot(ground, contact@ground))
        static = float(np.real(diamagnetic)-np.sum((wp+wm)/gaps))
        # H(A) is translation-conjugate on each finite Fourier set. This static
        # cancellation is a control, never a claim of a photon pole.
        if abs(static) > 2e-9:
            raise ArithmeticError('Current/contact ground resolvent assertion failed')
        active = np.flatnonzero(wp+wm > 1e-13)
        spectra.append(dict(spatial_q_integer=qi, spatial_q=[2*np.pi*v/4 for v in qi],
                            both_channel_weight_plus=float(wp.sum()), both_channel_weight_minus=float(wm.sum()),
                            contact_real=float(diamagnetic.real), static_contact_minus_resolvent=static,
                            omitted_contact_wrong_static=-float(np.sum((wp+wm)/gaps)),
                            lowest_current_weighted_gap=float(gaps[active[0]]) if len(active) else None,
                            poles=[dict(gap=float(gaps[i]), plus=float(wp[i]), minus=float(wm[i])) for i in active],
                            resolvent_imaginary_frequency=[dict(nu=nu,
                                real=float(np.real(diamagnetic)-np.sum((wp+wm)*gaps/(gaps*gaps+nu*nu))))
                                for nu in (0., .1, 1., 10., 100.)]))
    if omit == '' and max(abs(s['omitted_contact_wrong_static']) for s in spectra) < 1e-7:
        raise ArithmeticError('Omitted-contact control did not reach actual spectral assertion')
    ws = abs(vectors[:, 1:].conj().T@wilson@ground)**2
    active_w = np.flatnonzero(ws > 1e-13)
    electric_weights = np.array([64*np.linalg.norm(k@geometry['species'][0]['modes'])**2 for k in klist])
    return dict(winding_cut=winding_cut, dimension=n, ground_energy_shift=float(values[0]),
                ground_residual=residual, minimum_excitation_gap=float(gaps[0]),
                electric_energy_expectation=float(np.dot(abs(ground)**2, electric_weights)),
                reference_zero_flux_overlap_squared=float(abs(ground[lookup[(-1, ref, 0)]])**2),
                hermiticity_defect=hermitian, spectra=spectra,
                wilson_loop_expectation_real=float(np.vdot(ground, wilson@ground).real),
                wilson_lowest_weighted_gap=float(gaps[active_w[0]]) if len(active_w) else None,
                wilson_spectral_weight=float(ws.sum()))


def run() -> dict:
    started = time.monotonic()
    data = old.load_mean(g_override=[Fraction('0.28917434559291342'), *[Fraction('0.10475219290309079')]*3],
                         mean_descriptor=dict(law='Current fixed N4 root display; onsite-reference Bloch0 probe, not root rerun or adopted ground'))
    native = data['native']
    if (native['N'], native['V'], native['a'], native['Q'].shape) != (4, 64, 1/64, (192, 129)):
        raise ArithmeticError('Current N4 quotient law mismatch')
    loops = []
    for lengths in ((1, 1), (2, 1), (1, 2), (2, 2)):
        geometry = loop(data, lengths, (0, 0, 0))
        species = []
        for item in geometry['species']:
            actual = ring(item, geometry)
            wrong_he, wrong_hc, wrong_car = [ring(item, geometry, omission) for omission in ('HE', 'HC', 'CAR')]
            species.append(dict(load=item['load'], onsite_eigenvalues=item['onsite'].tolist(),
                                seams=item['seams'], actual=actual, omitted_HE=wrong_he,
                                omitted_HC=wrong_hc, omitted_CAR_sign=wrong_car))
            species[-1]['incoming_cycle_kernel'] = ring(item, geometry, incoming_winding=1)
        coefficient = sum(s['load']*s['actual']['coefficient_real'] for s in species)
        wrong_he = sum(s['load']*s['omitted_HE']['coefficient_real'] for s in species)
        wrong_hc = sum(s['load']*s['omitted_HC']['coefficient_real'] for s in species)
        control_has_channel = any(abs(s['actual']['coefficient_real']) > 1e-14 for s in species)
        if control_has_channel and (abs(wrong_he-coefficient) < 1e-12 or abs(wrong_hc-coefficient) < 1e-12):
            raise ArithmeticError('Actual nonzero ring HE/HC control failed to discriminate')
        wrong_car = sum(s['load']*s['omitted_CAR_sign']['coefficient_real'] for s in species)
        incoming_coefficient = sum(s['load']*s['incoming_cycle_kernel']['coefficient_real'] for s in species)
        if control_has_channel and (coefficient*wrong_car >= 0 or abs(incoming_coefficient-coefficient) < 1e-12):
            raise ArithmeticError('Actual CAR-sign or incoming-electric-momentum discriminator failed')
        cycle = geometry['species'][0]['modes'].sum(axis=0)
        electric_gap = 64*float(cycle@cycle)
        if np.linalg.norm(native['D']@geometry['species'][0]['edge_vectors'].sum(axis=0)) > 1e-9:
            raise ArithmeticError('Native rectangle not gauge closed')
        translated = loop(data, lengths, (3, 3, 0))
        translated_coefficient = sum(item['load']*ring(item, translated)['coefficient_real'] for item in translated['species'])
        if abs(translated_coefficient-coefficient) > 1e-10:
            raise ArithmeticError('AP seam translated closed-loop coefficient mismatch')
        loops.append(dict(lengths=lengths, vertices=geometry['vertices'], species=species,
                          nonzero_ring_control_available=control_has_channel,
                          spin_selection='All-length2 Gamma0 paths annihilate this filled negative-onsite reference; zero2x2 channel is a measured selection rule, not an omission-control pass.' if not control_has_channel else 'Nonzero full spin/CAR loop',
                          all_charged_ring_coefficient=coefficient, omitted_HE_coefficient=wrong_he,
                          omitted_HC_coefficient=wrong_hc, AP_translated_coefficient=translated_coefficient,
                          cycle_electric_gap=electric_gap, abs_ring_to_electric_gap=abs(coefficient)/electric_gap,
                          zero_flux_formal_curvature=-2*coefficient,
                          incoming_cycle_to_two_cycles_kernel=incoming_coefficient,
                          omitted_CAR_sign_coefficient=wrong_car,
                          scope='Fourth-order zero-energy Feshbach offdiagonal from electric0 to cycle. NOT a translation-invariant potential coefficient: denominators depend on incoming Pi.'))
    curvature = {tuple(row['lengths']): row['zero_flux_formal_curvature'] for row in loops}
    formal_stiffness = curvature[(1, 1)]+4*curvature[(2, 1)]+4*curvature[(1, 2)]
    formal_spatial = [dict(q=q, transverse_y_curvature=
                          curvature[(1, 1)]*4*np.sin(q/2)**2+
                          curvature[(2, 1)]*4*np.sin(q)**2+
                          curvature[(1, 2)]*16*np.sin(q/2)**2)
                      for q in (0., np.pi/2, np.pi)]
    geometry = loop(data, (2, 1), (0, 0, 0))
    # Materially different representation after the small-ring/strong-electric
    # route fails its required weak-field regime: diagonalize the interacting
    # collective PH plus electric-loop Fourier compression, no ring expansion.
    exact_compressions = [reduced(geometry, native, cut) for cut in (1, 2)]
    wrong_he = reduced(geometry, native, 1, 'HE')
    wrong_hc = reduced(geometry, native, 1, 'HC')
    actual = exact_compressions[0]
    if abs(wrong_he['ground_energy_shift']-actual['ground_energy_shift']) < 1e-5 or abs(wrong_hc['ground_energy_shift']-actual['ground_energy_shift']) < 1e-5:
        raise ArithmeticError('Reduced interacting HE/HC ground control was blind')
    input_names = [Path(data['descriptor'][key]['path']).name for key in ('root_receipt', 'full_receipt')]
    inputs = {name: (ROOT/name).read_bytes() for name in input_names}
    source_rows = {name: dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), unchanged=(ROOT/name).read_bytes() == raw)
                   for name, raw in SOURCE_BYTES.items()}
    if not all(row['unchanged'] for row in source_rows.values()):
        raise RuntimeError('Captured source epoch changed')
    return dict(plan=PLAN, assignment_revision=ASSIGNED_REVISION, bookkeeping_revision=BOOKKEEPING_REVISION,
                pinned_template_revision='89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',
                exact_law=dict(N=4, V=64, a='1/64', HE=64, HC=64, diffusion=128, EM_real_dimensions=129,
                    charged_masses=['1/5', '2'], charged_loads=[1, 63], Ref=4, background_per_site=128,
                    no_bare_magnetic_term=True, native='Original rational one/two coefficients, spin4, CAR, AP in all axes',
                    fixed_mean=data['descriptor']),
                reference=dict(state='At all64 sites each of64 charged copies fills negative onsite Gamma0 rank2; Ref4, rho=0; constant normalized configuration Floquet Bloch0 fiber wavefunction, Pi0.',
                    neutral='One heavy neutral copy remains spectator; no Lambda adopted as charged scale or photon clock.',
                    fiber='Direct integral of original noncompact cover; character0 for exact2pi integer cycle period lattice. Fourier p=Q^T integer native path words. Not a compactification of original whole Hilbert space.',
                    observables='Periodic Peierls currents, electric Pi and Wilson loops preserve fiber. Theta and unrestricted field CCR do not preserve one fiber.',
                    configuration_Bloch_momentum=[0]*129, spatial_q='2pi integer/4 in current vertex, NOT configuration Bloch momentum'),
                derivation=dict(ring='Sum24 orderings with exact16-mode CAR; each intermediate charge configuration rho priced with full N4 Lplus and cumulative Pi, including all cross terms. Three negative resolvents give signed4th-order amplitude. Closed square return requires all4 hops same flavor; no differently colored open subcycle can return every flavor occupancy, hence coefficient Kactive+63Kaux.',
                    diagonal_prices='Delta onsite +64||sum directed native Q path||^2+64 rho^T Lplus rho; no one-hop selfprice approximation.',
                    formal_low_q='For the conditional Pi-independent potential2K_L cos(C_L Theta), curvature=-2K_L. Transverse y spatial qx gives unit-square4sin(q/2)^2,2x1 rectangle4sin(q)^2,1x2 rectangle16sin(q/2)^2. Thus B_formal=-2K11-8K21-8K12 and omega(q)^2=128*B(q). Actual incoming-Pi dependence is explicitly measured, so this local potential/weak-field premise is invalid as a supplier. No dispersion/speed adopted.',
                    resolvent='Finite Hermitian PHP exactly diagonalized; D_q=<contact_q,-q>; chi_q(i nu)=D_q-sum_n[(|<n|J_q|g>|^2+|<n|J_-q|g>|^2)Delta_n/(Delta_n^2+nu^2)]. Both spectral channels from exact interacting compressed eigenstates, not free Wick or Taylor moments.',
                    static_translation='External A is configuration translation of dynamical Theta: finite Fourier compression obeys H(A)=U(A)H(0)U(A)^dag. Current/contact cancel static twist; this is not magnetic stiffness or photon proof.'),
                conditional_magnetic_diagnostic=dict(formal_long_wavelength_stiffness=formal_stiffness,
                    formal_electric_to_magnetic_ratio=128/formal_stiffness,
                    formal_speed_squared=128*formal_stiffness, spatial_curvature=formal_spatial,
                    status='Derived counterfactual local-potential Hessian only. Actual measured incoming-cycle Feshbach coefficient differs, invalidating Pi-independent magnetic stiffness adoption; no actual full-ground magnetic stiffness is claimed.'),
                ring_results=loops, materially_different_route=dict(construction='Exact interacting collective-one-PH Fourier loop compression on native2x1 rectangle. All4 spin modes per vertex, active and symmetric63 auxiliary copies with exact sqrt63 vacuum vertices; all other sites/copies onsite spectators. Three/five electric winding values. Independent variational ground per cut, not strong-coupling ring coefficient adoption.',
                    scope='Reduced fixed-body PHP only: excludes simultaneous multiple PHs (including active-aux covariance), other native paths/vertices, unrestricted winding, neutral/Phi/Haar fluctuations. Full Coulomb on every retained state; not global N4/Phi spectrum.',
                    results=exact_compressions, omitted_HE=wrong_he, omitted_HC=wrong_hc),
                wrong_controls=dict(assertion='Nonzero1x1/2x1 ring coefficients must differ from omittedHE/HC; zero2x2 spin-blocked ring has no control channel. Finite ground must differ from omittedHE/HC; static exact spectral contact-resolvent cancellation must hold and omitted-contact must break it.',
                    passed=True, AP_translation='Native seam-exact translated loops matched; noncontractible AP topology not inferred from these contractible rectangles.'),
                conclusion=dict(stage='New ground-directed operator-bound finite spectral supplier; no full photon admission',
                    first_route_failed='The electric-loop jump costs hundreds while signed fourth-order coupling is tiny; incoming-cycle kernel changes substantially, so no Pi-independent magnetic potential is supplied. Small loop/electric ratio alone does not control collective64-flavour hopping or prove confinement. Positive conditional curvature cannot establish a photon branch.',
                    second_route='Actual reduced variational ground and discrete current/Wilson spectra executed. Finite oscillator/gapped diagnostic; no massless branch, confinement theorem, or commoncone asserted.',
                    open_debt=['Control multi-PH and all native path enlargement with full charged covariance.',
                               'Control unrestricted electric winding, fullPhi/Haar and neutral heavy ground embedding.',
                               'Construct thermodynamic generalized stationary state and actual spatial-q pole/residue from same ground; N4 has no q->0 sequence.',
                               'Derive photon/electron common cone and Compton consumer without clock/source/parameter adoption.'],
                    next_constructive_step='Enlarge collective interacting electric-Fourier cluster by two simultaneous PH sectors, retaining actual active-aux/allaux charge covariance; use rigorous Ritz/Schur tail brackets before spatial N enlargement.'),
                sources=source_rows, inputs={name:dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)) for name, raw in inputs.items()},
                numerical_status='Float64 finite Hermitian diagonalization and perturbative coefficients, not interval proof.',
                elapsed_seconds=time.monotonic()-started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    archive = args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, raw in SOURCE_BYTES.items():
            bundle.writestr(name, raw)
        for name in result['inputs']:
            bundle.writestr(name, (ROOT/name).read_bytes())
        bundle.write(args.output, args.output.name)
    print(json.dumps(dict(receipt=str(args.output), runtime_archive=str(archive),
                         ground_shifts=[r['ground_energy_shift'] for r in result['materially_different_route']['results']],
                         gaps=[r['minimum_excitation_gap'] for r in result['materially_different_route']['results']],
                         rings=[r['all_charged_ring_coefficient'] for r in result['ring_results']],
                         elapsed_seconds=result['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
