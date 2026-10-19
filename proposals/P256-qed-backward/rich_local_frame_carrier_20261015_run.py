"""NEW owning source propagation with actual nonstationary neutral baseline.

Full Hilbert space; scipy sparse expm_multiply, no ground/pole projection.
Scalar-only rich law is a material attempted route, not a Mott replacement.
"""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace
import zipfile
import numpy as np
from scipy.sparse.linalg import expm_multiply
from rich_local_frame_carrier_20261015_model import Model, Sector, INCIDENCE

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_local_frame_carrier_20261015_'
TIMES = [0., .3, .6]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(data, indent=2)+'\n')


def mean(psi, op):
    return float(np.vdot(psi, op@psi).real)


def observe(model, h, currents, psi):
    s = model.sector
    probability = np.abs(psi.reshape(model.shape))**2
    gauge_probability = probability.sum(axis=(1, 2, 3, 4))
    n = [mean(psi, a) for a in model.n]
    spin2 = [mean(psi, a) for a in model.S2_local]
    r2 = [mean(psi, a) for a in model.r2_local]
    hpsi = h@psi
    spin = [[mean(psi, a) for a in site] for site in model.S_local]
    spin_norm = [float(np.linalg.norm(site)) for site in spin]
    current = [mean(psi, a) for a in currents]
    charge_rate = [float(-2*np.vdot(hpsi, a@psi).imag) for a in model.Q]
    continuity = np.array(charge_rate)+INCIDENCE@current
    total_torque = [float(-2*np.vdot(hpsi, a@psi).imag) for a in model.J]
    local_spin_rate = [[float(-2*np.vdot(hpsi, a@psi).imag) for a in site] for site in model.S_local]
    electric_rate = [float(-2*np.vdot(hpsi, a@psi).imag) for a in model.E]
    ns_covariance = [mean(psi, model.n[v]@model.S2_local[v])-n[v]*spin2[v] for v in range(3)]
    nr_covariance = [mean(psi, model.n[v]@model.r2_local[v])-n[v]*r2[v] for v in range(3)]
    local_count = [[float(gauge_probability@(s.density[:, v] == count)) for count in range(5)] for v in range(3)]
    negative = [mean(psi, model.root_band_count(v, -1)) for v in range(3)]
    positive = [n[v]-negative[v] for v in range(3)]
    radial_marginals = [probability.sum(axis=tuple(a for a in range(5) if a != v+2)).tolist() for v in range(3)]
    return dict(norm=float(np.vdot(psi, psi).real), energy=float(np.vdot(psi, hpsi).real),
        n=n, Q=[2-a for a in n], S_local=spin, S_local_vector_norm=spin_norm,
        S2_local=spin2, r2_local=r2, E=[mean(psi, a) for a in model.E],
        electric_energy=mean(psi, model.HE), Wilson_sine=mean(psi, model.F),
        Wilson_cosine=mean(psi, model.Wcos), root_energy=mean(psi, model.Hroot),
        local_count_PVM=local_count, radial_node_PVM=radial_marginals,
        n_S2_connected=ns_covariance, n_r2_connected=nr_covariance,
        negative_root_occupancy=negative, positive_root_occupancy=positive,
        material_L=[mean(psi, a) for a in model.material_L], J=[mean(psi, a) for a in model.J],
        current=current, charge_rate=charge_rate, local_spin_rate=local_spin_rate,
        charge_continuity_max=float(np.max(np.abs(continuity))),
        electric_current_max=float(np.max(np.abs(np.array(electric_rate)+current))),
        total_Noether_torque_max=float(np.max(np.abs(total_torque))),
        electric_cap_shell=float(gauge_probability@np.any(abs(s.electric) == s.cap, axis=1)))


def propagate(model, psi, variant, name, outputs, checkpoints):
    h, currents, _ = model.law(variant)
    initial = psi.copy()
    energy0 = mean(psi, h)
    trace = h.diagonal().sum()
    records = []
    for index, t in enumerate(TIMES):
        if index:
            dt = t-TIMES[index-1]
            psi = expm_multiply(-1j*dt*h, psi, traceA=-1j*dt*trace)
        record = observe(model, h, currents, psi); record['time'] = t
        records.append(record)
        print(json.dumps(dict(case=name, time=t, n=record['n'], S2=record['S2_local'], r2=record['r2_local'],
                              current=record['current'], norm=record['norm'])), flush=True)
        if index == len(TIMES)-1:
            checkpoints[name+'_final'] = psi
    # Direct .6 versus two .3 steps independently exercises adaptive exponential.
    fine = expm_multiply(-.6j*h, initial, traceA=-.6j*trace)
    outputs[name] = dict(variant=variant, dimension=model.dimension, basis_SHA256=model.basis_hash,
        records=records, endpoint_direct_vs_two_half_norm=float(np.linalg.norm(fine-psi)),
        energy_drift_max=max(abs(r['energy']-energy0) for r in records),
        norm_error_max=max(abs(r['norm']-1) for r in records))
    del h, currents, fine
    gc.collect()
    return records


def difference(charged, neutral):
    keys = ['n', 'Q', 'S2_local', 'r2_local', 'E', 'S_local', 'material_L', 'J',
            'negative_root_occupancy', 'positive_root_occupancy']
    return [dict(time=c['time'], **{key:(np.array(c[key])-np.array(n[key])).tolist() for key in keys})
            for c, n in zip(charged, neutral)]


def main():
    start = time.monotonic()
    interface = json.loads((ROOT/(PREFIX+'interface.json')).read_text())
    params = interface['parameters']
    expected = interface['source_SHA256'][PREFIX+'model.py']
    if sha(ROOT/(PREFIX+'model.py')) != expected:
        raise RuntimeError('Owning model changed after reference publication')
    if sha(ROOT/(PREFIX+'neutral_reference.npz')) != interface['neutral_reference']['SHA256']:
        raise RuntimeError('Reference bytes changed')
    sources = {name:sha(ROOT/name) for name in [PREFIX+'run.py', PREFIX+'model.py', PREFIX+'input.json', PREFIX+'interface.json', PREFIX+'equations.json']}
    save('run_start', dict(source_SHA256=sources, reference_SHA256=interface['neutral_reference']['SHA256'], times=TIMES))
    with zipfile.ZipFile(ROOT/(PREFIX+'run_source_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
        for name in sources:
            capsule.write(ROOT/name, name)
    reference = np.load(ROOT/(PREFIX+'neutral_reference.npz'))['psi']
    outputs, checkpoints, baselines = {}, {}, {}
    neutral = Model(6, params)
    for variant in ['vector_off', 'full']:
        baselines[variant] = propagate(neutral, reference.copy(), variant, 'neutral_'+variant, outputs, checkpoints)
    del neutral
    gc.collect()
    ref_api = SimpleNamespace(sector=Sector(6, params['electric_cap']))
    for branch, number in [('addition', 7), ('removal', 5)]:
        model = Model(number, params)
        b = model.source_from(ref_api, 0, branch, 0, 0)
        raw = b@reference
        brightness = float(np.vdot(raw, raw).real)
        psi = raw/np.sqrt(brightness)
        checkpoints[branch+'_t0'] = psi
        outputs[branch+'_source'] = dict(branch=branch, number=number, site=0, orbital=0, component=0,
                                         raw_strength=brightness, reference_SHA256=interface['neutral_reference']['SHA256'])
        for variant in ['vector_off', 'full']:
            records = propagate(model, psi.copy(), variant, branch+'_'+variant, outputs, checkpoints)
            outputs[branch+'_'+variant]['absolute_time_excess'] = difference(records, baselines[variant])
            if branch == 'addition' and variant == 'full':
                save('first_synthesis', dict(kind='FIRST_ACTUAL_RICH_LOCAL_SOURCE_NOT_PARTICLE_ADMISSION',
                    assumptions=PREFIX+'equations.json', model_SHA256=expected,
                    reference_SHA256=interface['neutral_reference']['SHA256'],
                    raw_source_strength=brightness, dimension=model.dimension,
                    actual_full=outputs[branch+'_full'], actual_scalar_attempt=outputs[branch+'_vector_off'],
                    finding='Local physical-frame and local norm evidence are measured against evolving own neutral; scalar rich attempt is not an electron and full result must decide co-motion.'))
        if branch == 'addition':
            pulse = np.exp(.7j*model.r2_local[0].diagonal())*psi
            records = propagate(model, pulse, 'full', 'addition_radial_pulse', outputs, checkpoints)
            outputs['addition_radial_pulse']['same_H_minus_unpulsed'] = difference(records, outputs['addition_full']['records'])
            outputs['addition_radial_pulse']['pulse'] = 'exp(.7 i r0²), field-only; exact t0 n/S/r² unchanged, local radial conjugate kick'
        del model, b, raw, psi
        gc.collect()
    save('result', dict(kind='ACTUAL_OWNING_FULL_CAR12_LOCAL_FIELDS_SOURCE_RESPONSE', times=TIMES,
        parameters=params, model_SHA256=expected, reference_SHA256=interface['neutral_reference']['SHA256'],
        cases=outputs, absolute_time_neutral_evolved=True,
        scope='Finite local/radial/material/electric regulator; all fixedN CAR/root branches retained; no ground or freeCOM/particle theorem'))
    np.savez_compressed(ROOT/(PREFIX+'states.npz'), **checkpoints)
    save('run_exit', dict(natural_exit=True, exit_code=0, elapsed_seconds=time.monotonic()-start,
                          max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          source_SHA256=sources, result_SHA256=sha(ROOT/(PREFIX+'result.json')),
                          state_SHA256=sha(ROOT/(PREFIX+'states.npz'))))
    print(json.dumps(dict(done=True, cases=list(outputs), elapsed=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
