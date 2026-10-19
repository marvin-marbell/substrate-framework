"""First same-H neutral field-energy discriminator, then finite-law audit."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln, jv
from rich_neutral_mott_gauge_20261015_model import NeutralCube, LAW_ID

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_neutral_mott_gauge_20261015_'


def save(name, data):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(data, indent=2)+'\n')


def chebyshev(H, state, duration, tolerance=2e-12):
    diagonal = H.diagonal()
    row_radius = np.asarray(abs(H).sum(axis=1)).ravel()-abs(diagonal)
    low, high = float(np.min(diagonal-row_radius)), float(np.max(diagonal+row_radius))
    center, radius = (low+high)/2, (high-low)/2
    z = radius*duration
    degree = max(1, math.ceil(z))
    while True:
        n = degree+1
        ratio = z/(2*(n+1))
        bound = math.log(2)+n*math.log(z/2)-float(gammaln(n+1))+z*z/(4*(n+1))-math.log1p(-ratio)
        if bound <= math.log(tolerance):
            break
        degree += 1
    scaled = (H-center*sparse.eye(H.shape[0], format='csr'))/radius
    previous, current = state, scaled @ state
    result = jv(0, z)*previous+2*(-1j)*jv(1, z)*current
    for n in range(2, degree+1):
        following = 2*(scaled @ current)-previous
        result += 2*((-1j)**n)*jv(n, z)*following
        previous, current = current, following
    return np.exp(-1j*center*duration)*result, dict(enclosure=[low, high], degree=degree, analytic_tail_excluding_roundoff=math.exp(bound))


def delta(row, key):
    return np.asarray(row[key][1])-np.asarray(row[key][0])


def main():
    data = json.loads((ROOT/sys.argv[1]).read_text())
    started = time.monotonic()
    model = NeutralCube()
    print(json.dumps(dict(stage='new_H_constructed', dimension=model.dimension, H_nnz=model.H.nnz)), flush=True)
    seed = model.seed()
    pulse = expm_multiply(-1j*data['alpha']*model.source, seed, traceA=0)
    initial = np.column_stack((seed, pulse))
    state = initial.copy()
    times = np.linspace(0, data['duration'], data['intervals']+1)
    records = []
    dt = float(times[1]-times[0])
    for k, instant in enumerate(times):
        record = model.measure(state)
        record['time'] = float(instant)
        records.append(record)
        print(json.dumps(dict(stage='physical_sample', time=float(instant), ER=record['ER'], HR=record['HR'], right_HU=record['right_HU'])), flush=True)
        if k < len(times)-1:
            state = expm_multiply(-1j*dt*model.H, state, traceA=-1j*dt*float(model.H.diagonal().sum()))
    de = [float(delta(r, 'ER')) for r in records]
    dh = [float(delta(r, 'HR')) for r in records]
    provisional = dict(kind='FIRST_ACTUAL_HALF_FILLED_UNIFORM_QLINK_NEUTRAL_ENERGY_DISCRIMINATOR', law_id=LAW_ID,
                       dimension=model.dimension, input=data, preparation='Four newly prepared nearest-neighbour spin singlets, one electron per vertex and E0; C-even unitary exp(-i alpha Re Wleft). Nonstationary, not a ground state.',
                       delta_ER=de, delta_HR=dh, times=times.tolist(),
                       endpoint_right_HU_delta=float(delta(records[-1], 'right_HU')),
                       endpoint_right_charge_variance_delta=delta(records[-1], 'charge_variance')[model.right_sites].tolist(),
                       endpoint_ER_right_charge_free_delta=float(delta(records[-1], 'ER_right_charge_free')),
                       endpoint_ER_global_charge_free_delta=float(delta(records[-1], 'ER_global_charge_free')),
                       work=float(delta(records[0], 'H')),
                       decision='Finite neutral field-bright response with dynamical particle-hole fluctuations; mixed gauge/matter transport candidate, not virtual-only radiation. Exact locality/control and numerical resolution appended after this first discriminator.',
                       no_promotion='No massless, two-photon, helicity, speed, continuum, rotor convergence, stationary Mott phase, CAR36 or Nature claim.')
    save('provisional_synthesis', provisional)
    print(json.dumps(provisional), flush=True)
    algebra = model.algebra()
    independent, numerical = chebyshev(model.H, initial, data['duration'])
    numerical['state_endpoint_l2_difference'] = np.linalg.norm(independent-state, axis=0).tolist()
    independent_measure = model.measure(independent)
    numerical['delta_ER_method_difference'] = float(abs(delta(independent_measure, 'ER')-delta(records[-1], 'ER')))
    numerical['delta_HR_method_difference'] = float(abs(delta(independent_measure, 'HR')-delta(records[-1], 'HR')))
    numerical['max_norm_drift'] = max(float(np.max(abs(np.asarray(r['norm'])-1))) for r in records)
    numerical['max_energy_drift'] = max(float(np.max(abs(np.asarray(r['H'])-records[0]['H']))) for r in records)
    # Exact initial separability witness in the unprojected matter/chord embedding.
    embedded = np.zeros((len(model.fock)*model.nflux, 2), dtype=complex)
    embedded[model.old_indices] = initial
    seedwave = embedded[:, 0].reshape(len(model.fock), model.nflux)
    pulsewave = embedded[:, 1].reshape(len(model.fock), model.nflux)
    matter = seedwave[:, model.mi[(0,)*5]]
    field = matter.conj() @ pulsewave
    preparation = dict(C_seed_residual=float(np.linalg.norm(model.C @ seed-seed)), C_pulse_residual=float(np.linalg.norm(model.C @ pulse-pulse)),
                       initial_matter_unchanged_factorization_residual=float(np.linalg.norm(pulsewave-matter[:, None]*field[None, :])),
                       pulse_norm=float(np.linalg.norm(pulse)), seed_nonzero_components=int(np.count_nonzero(seed)),
                       seed_doublons=records[0]['doublons'], seed_holons=records[0]['holons'], seed_H_variance=records[0]['H_variance'],
                       initial_ER_delta=float(delta(records[0], 'ER')), initial_HR_delta=float(delta(records[0], 'HR')),
                       conditioning='None: genuine unitary field pulse, raw alpha disclosed. No postselection or renormalized odd-source strength.')
    controls = dict(hopping_off=dict(H='Hoff=HE+4D', receiver='HRoff=weighted HE+4Dright',
                    exact_zero_excess=all(algebra[k] == 0 for k in ('off_ER', 'off_HRoff', 'source_ER', 'source_HRoff')),
                    proof='Both receiver energies conserved and commute with source; same initial pair, no re-preparation.'),
                    path_cut=dict(H='Hcut removes only four x-bridge CAR U hops', receiver='HRcut=HRoff+right-face hopping',
                    exact_zero_excess=all(algebra[k] == 0 for k in ('cut_factorization', 'cut_left_ER', 'source_HRcut', 'source_ER')),
                    proof='[HLcut,HRcut]=[HLcut,ER]=[source,HRcut]=[source,ER]=0. Unitary source commutes with entire right trajectory. Physical local-link projection, not old unbounded-tree assumption.'))
    result = dict(kind='NEW_UNIFORM_PHYSICAL_EDGE_QLINK_HALF_FILLED_RESULT', law_id=LAW_ID, input=data,
                  governor=dict(plan_id='issue222-rich-backward-candidates-20261014', launch_revision='a9f3929e495be48d89cc15bd969c7641d83a685517a84b32b20f5d00a9ba638b', template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe', work_item='photon-candidate', requirements=['outcome', 'verification']),
                  geometry=dict(positions=model.positions.tolist(), edges=model.edges, tree=model.tree, chords=model.chords, incidence=model.incidence.tolist(), source_cycle=model.left_cycle.tolist(), left_edges=model.left_edges.tolist(), right_edges=model.right_edges.tolist(), bridges=model.bridges.tolist(), background=[1]*8),
                  Hilbert=dict(Nup=4, Ndown=4, full_fock_count=len(model.fock), chord_candidate_count=len(model.fock)*model.nflux, accepted_dimension=model.dimension, physical_E_values=[-1, 0, 1], all_dynamic_charge_configurations=True, H_nnz=model.H.nnz),
                  preparation=preparation, discriminator=provisional, algebra=algebra, controls=controls, numerical=numerical, samples=records,
                  semantics=dict(columns=['untouched_baseline', 'unitary_field_pulse'], scalar_differences='pulse minus baseline',
                    charge_variance='<(n_i-1)^2>; actual C symmetry fixes <n_i>=1, but NOT doublon/holon populations',
                    ER_charge_free='UNNORMALIZED expectation ER times indicated no-charge projector at that time; not a history/virtuality discriminator, not a postselected propagation experiment',
                    sym_EJ_power='1/2 <E_e J_e+J_e E_e>=d<E_e^2/2>/dt; unlike <E><J>, may be nonzero in C-even state',
                    right_J2='<J_e^2> for four right-face links; positive equal-time current correlation, not mean current',
                    powerR='i<[H,HR]>=-2 Im <H psi|HR psi>',
                    energy_ownership='HR: x-endpoint weights for electric and hopping edges, x=1 sites own Hubbard; HL=H-HR; bridge half ownership',
                    limitations='Unconditional finite prepared-state energy redistribution with real fluctuating charges. Does not establish quasiparticle poles, pure virtual radiation, Mott phase, or polarization.'),
                  wall_seconds=time.monotonic()-started)
    save('actual_result', result)
    np.savez_compressed(ROOT/(PREFIX+'actual_states.npz'), initial=initial, endpoint=state, independent_endpoint=independent,
                        old_indices=model.old_indices, fock=model.fock, chord_flux=model.flux, electric=model.electric)
    print(json.dumps(dict(completed=True, wall_seconds=result['wall_seconds'], algebra=algebra, controls=controls, numerical=numerical)), flush=True)


if __name__ == '__main__':
    main()
