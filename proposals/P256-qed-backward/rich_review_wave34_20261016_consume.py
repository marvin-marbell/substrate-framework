"""New full-H graded dressed-source/two-cycle-leg/adjoint-source consumer."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
from rich_local_frame_carrier_20261015_model import Model, INCIDENCE
from rich_dressed_local_carrier_20261015_source import cool, energy
from rich_review_wave31_20261015_krylov import propagate

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave34_20261016_'
REPORTS: list[dict[str, Any]] = []


def pack(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return dict(real=value.real.tolist(), imag=value.imag.tolist()) if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def dump(suffix: str, value: Any) -> None:
    (ROOT/(P+suffix+'.json')).write_text(json.dumps(pack(value), indent=2)+'\n')


def evolve(h: Any, vector: Any, dt: float, label: str) -> Any:
    if abs(dt) < 1e-15:
        return vector.copy()
    norm = float(np.linalg.norm(vector))
    if norm < 1e-25:
        return vector.copy()
    states, report = propagate(h, vector/norm, [dt], 24, 18)
    REPORTS.append(dict(label=label, input_norm=norm, **report))
    return norm*states[0]


def filtered(h: Any, vector: Any, tau: float, shift: float, label: str) -> Any:
    states, gains, report = cool(h, vector, [tau], 40, 30, shift)
    REPORTS.append(dict(label=label, **report))
    return np.exp(gains[0])*states[0]


def primitives(model: Any, h: Any, currents: Any, vector: Any) -> dict:
    exp = lambda a: float(np.vdot(vector, a@vector).real)
    return dict(n=[exp(a) for a in model.n], E=[exp(a) for a in model.E],
                currents=[exp(a) for a in currents], S=[[exp(a) for a in site] for site in model.S_local],
                material_L=[exp(a) for a in model.material_L], energy=energy(h, vector))


def neutral_moments(model: Any, h: Any, neutral: Any, times: list[float], label: str) -> tuple[dict, dict]:
    _, currents, contacts = model.law()
    ops = dict(E=sum(model.E)/np.sqrt(3), J=sum(currents)/np.sqrt(3), K=sum(contacts)/3)
    states = {t: evolve(h, neutral, t, label+':neutral:'+str(t)) for t in times}
    means = {str(t): {name: np.vdot(states[t], op@states[t]) for name, op in ops.items()} for t in times}
    pairs = {}
    for early, late in [(0., 0.), (.12, .26)]:
        for first, second in [('J', 'E'), ('E', 'J')]:
            work = evolve(h, ops[first]@states[early], late-early, label+':neutralpair:'+first)
            pairs[f'{early}:{late}:{first}:{second}'] = np.vdot(states[late], ops[second]@work)
    return dict(means=means, pairs=pairs, energies={str(t): energy(h, v) for t, v in states.items()}), states


def response(model: Any, hn: Any, neutral_model: Any, neutral: Any, neutral_states: dict,
             moments: dict, shift: float, tau: float, branch: str, baseline: str) -> dict:
    h, currents, contacts = model.law()
    ops = dict(E=sum(model.E)/np.sqrt(3), J=sum(currents)/np.sqrt(3), K=sum(contacts)/3)
    out: dict[str, Any] = dict(source_site=0, receiver_site=1, index_order=['orbital0_spin0', 'orbital0_spin1', 'orbital1_spin0', 'orbital1_spin1'], schedules={})
    sources = []
    bare_sources = []
    for orbital in range(2):
        for spin in range(2):
            b = model.source_from(neutral_model, 0, branch, spin, orbital)
            bare = b@neutral
            bare_sources.append(bare)
            sources.append(filtered(h, bare, tau, shift, f'{branch}:{baseline}:source:{orbital}:{spin}'))
    out['source_bare_weights'] = [float(np.vdot(v, v).real) for v in bare_sources]
    out['source_filtered_raw_weights'] = [float(np.vdot(v, v).real) for v in sources]
    lower = model.J[0]-1j*model.J[1]
    out['actual_source_spin'] = [dict(orbital=o, Jminus_B0_plus_B1_residual=float(np.linalg.norm(lower@sources[2*o]+sources[2*o+1])),
        Jz_B0_residual=float(np.linalg.norm(model.J[2]@sources[2*o]-.5*sources[2*o])),
        Jhalf_residual=float(np.linalg.norm(sum(a@(a@sources[2*o]) for a in model.J)-.75*sources[2*o]))) for o in range(2)]
    for early, late, total in [(0., 0., 0.), (.12, .26, .42)]:
        receivers = []
        receiver_weights = []
        for orbital in range(2):
            for spin in range(2):
                # This is F_N B_receiver U6(T)|Omega>, hence its bra is the ACTUAL Bdag F_N adjoint.
                bare = model.source_from(neutral_model, 1, branch, spin, orbital)@neutral_states[total]
                receivers.append(filtered(h, bare, tau, shift, f'{branch}:{baseline}:receiver:{total}:{orbital}:{spin}'))
                receiver_weights.append(float(np.vdot(receivers[-1], receivers[-1]).real))
        bra = np.asarray(receivers).conj()
        pair = np.empty((4, 4), complex)
        singles = {name: np.empty((4, 4), complex) for name in ['early_E', 'early_J', 'late_E', 'late_J', 'contact']}
        fours = {order: np.empty((4, 4), complex) for order in ['J_E', 'E_J']}
        for column, src in enumerate(sources):
            at_early = evolve(h, src, early, 'charged:early')
            at_late = evolve(h, at_early, late-early, 'charged:late')
            pair[:, column] = bra@evolve(h, at_late, total-late, 'charged:pair')
            for name in ['E', 'J']:
                insertion = evolve(h, ops[name]@at_early, late-early, 'charged:insert:'+name)
                singles['early_'+name][:, column] = bra@evolve(h, insertion, total-late, 'charged:3early')
                singles['late_'+name][:, column] = bra@evolve(h, ops[name]@at_late, total-late, 'charged:3late')
                other = 'J' if name == 'E' else 'E'
                fours[name+'_'+other][:, column] = bra@evolve(h, ops[other]@insertion, total-late, 'charged:4')
            singles['contact'][:, column] = bra@evolve(h, ops['K']@at_late, total-late, 'charged:contact')
        schedule = dict(times=[0, early, late, total], receiver_filtered_raw_weights=receiver_weights, source_pair=pair, three_point=singles, orders={})
        for first, second in [('J', 'E'), ('E', 'J')]:
            key = first+'_'+second
            mfirst = moments['means'][str(early)][first]
            msecond = moments['means'][str(late)][second]
            mfields = moments['pairs'][f'{early}:{late}:{first}:{second}']
            disconnected = pair*mfields+singles['early_'+first]*msecond+singles['late_'+second]*mfirst-2*pair*mfirst*msecond
            connected = fours[key]-disconnected
            schedule['orders'][key] = dict(raw_four=fours[key], disconnected=disconnected, connected=connected,
                neutral_one_point=[mfirst, msecond], neutral_two_point=mfields,
                Frobenius_connected=float(np.linalg.norm(connected)), Frobenius_raw=float(np.linalg.norm(fours[key])),
                spin_offdiagonal_max=float(max(abs(connected[i,j]) for i in range(4) for j in range(4) if i%2 != j%2)),
                spin_diagonal_difference_max=float(max(abs(connected[2*i,2*j]-connected[2*i+1,2*j+1]) for i in range(2) for j in range(2))))
        schedule['contact_connected'] = singles['contact']-pair*moments['means'][str(late)]['K']
        out['schedules'][str(total)] = schedule
        dump('partial_result', dict(branch=branch, baseline=baseline, completed_total=total, response=out))
    # Bare preparation is compared only to its OWN t0; no intermediate source projection.
    axis = np.array([-.36, .48, .8])
    generator = sum(axis[a]*model.architecture['one_spin'][a][:2, :2] for a in range(3))
    _, eigenvectors = np.linalg.eigh(generator)
    c = eigenvectors[:, -1]
    dressed = c[0]*sources[0]+c[1]*sources[1]
    dressed /= np.linalg.norm(dressed)
    bare = c[0]*bare_sources[0]+c[1]*bare_sources[1]
    bare /= np.linalg.norm(bare)
    out['controls'] = {}
    for name, variant, initial in [('full', 'full', dressed), ('path_cut', 'path_cut', dressed), ('vector_off', 'vector_off', dressed),
                                    ('cycle_field_pulse', 'full', np.exp(.31j*np.asarray(ops['E'].diagonal()))*dressed), ('bare_own_t0', 'full', bare)]:
        hc, jc, _ = model.law(variant)
        states, report = propagate(hc, initial, [0., .105, .21, .315, .42], 28, 20)
        REPORTS.append(dict(label=branch+':'+baseline+':control:'+name, **report))
        observations = [primitives(model, hc, jc, v) for v in states]
        dn = np.array(observations[-1]['n'])-observations[0]['n']
        de = np.array(observations[-1]['E'])-observations[0]['E']
        current_integral = np.trapezoid(np.array([v['currents'] for v in observations]), x=[0., .105, .21, .315, .42], axis=0)
        continuity = []
        for v in range(3):
            continuity.append(float(np.linalg.norm(1j*(hc@(model.n[v]@initial)-model.n[v]@(hc@initial))-sum(INCIDENCE[v,e]*(jc[e]@initial) for e in range(3)))))
        out['controls'][name] = dict(variant=variant, times=[0., .105, .21, .315, .42], observations=observations,
            delta_n=dn, delta_E=de, integrated_current_trapezoid=current_integral,
            exact_Gauss_count_transfer_residual=dn+INCIDENCE@de,
            count_minus_integrated_current=dn-INCIDENCE@current_integral,
            continuity_operator_on_initial_residual=continuity)
        del hc, jc, states
    eps = 1e-4
    # A genuine cycle vector-potential derivative; all three links, not a bare un-gauged hop.
    forward = [a+b for a,b in zip(model.scalar_transport, model.vector_transport)]
    hp = model.Hstatic-sum(np.exp(-1j*eps/np.sqrt(3))*a+np.exp(1j*eps/np.sqrt(3))*a.conj().T for a in forward)
    hm = model.Hstatic-sum(np.exp(1j*eps/np.sqrt(3))*a+np.exp(-1j*eps/np.sqrt(3))*a.conj().T for a in forward)
    out['own_H_cycle_probe'] = dict(epsilon=eps,
        first_derivative_plus_J_state_residual=float(np.linalg.norm((hp@dressed-hm@dressed)/(2*eps)+ops['J']@dressed)),
        second_derivative_minus_K_state_residual=float(np.linalg.norm((hp@dressed+hm@dressed-2*(h@dressed))/eps**2-ops['K']@dressed)),
        Edot_plus_J_state_residual=float(np.linalg.norm(1j*(h@(ops['E']@dressed)-ops['E']@(h@dressed))+ops['J']@dressed)))
    return out


def main() -> None:
    start = time.monotonic()
    config = json.loads((ROOT/(P+'input.json')).read_text())
    for name, expected in config['expected_input_SHA256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen input changed: '+name)
    interface = json.loads((ROOT/config['interface_file']).read_text())
    nm = Model(6, interface['parameters'])
    hn, _, _ = nm.law()
    with np.load(ROOT/interface['state_file']) as saved:
        neutral = saved['neutral'].copy()
    matched_raw = filtered(hn, neutral, .35, interface['filter_shift'], 'matched_neutral')
    matched = matched_raw/np.linalg.norm(matched_raw)
    result: dict[str, Any] = dict(kind='ACTUAL_NEW_NONFORWARD_TWO_CYCLE_LEG_GRADED_CONNECTED_RESPONSE', input=config,
        original_neutral=energy(hn, neutral), matched_neutral=energy(hn, matched),
        matched_ray_distance=float(np.linalg.norm(matched-neutral)), branches={}, neutral={})
    backgrounds = dict(original=neutral, matched=matched)
    neutral_data = {}
    for name, state in backgrounds.items():
        moments, states = neutral_moments(nm, hn, state, [0., .12, .26, .42], name)
        result['neutral'][name] = moments
        neutral_data[name] = (moments, states)
    for branch, number in [('addition', 7), ('removal', 5)]:
        model = Model(number, interface['parameters'])
        result['branches'][branch] = {}
        for baseline, state in backgrounds.items():
            moments, states = neutral_data[baseline]
            result['branches'][branch][baseline] = response(model, hn, nm, state, states, moments,
                interface['filter_shift'], .35, branch, baseline)
            dump('partial_result', result)
            print(json.dumps(dict(branch=branch, baseline=baseline, elapsed=time.monotonic()-start)), flush=True)
        del model
        gc.collect()
    result['numerics'] = REPORTS
    result['elapsed_seconds'] = time.monotonic()-start
    dump('result', result)
    print(json.dumps(dict(complete=True, elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
