"""Actually construct a new cooled neutral and full-H dressed charge doublets."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from rich_dressed_local_carrier_20261015_source import Model, prepared_reference, cool, energy, expectation, spin_down, component_source

ROOT = Path(__file__).resolve().parent
P = 'rich_dressed_local_carrier_20261015_'


def dump(suffix, value):
    (ROOT/(P+suffix+'.json')).write_text(json.dumps(value, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def primitive(model, state):
    density = abs(state.reshape(model.shape))**2
    gauge = density.sum(axis=(1, 2, 3, 4))
    radial = density.sum(axis=(0, 1))
    return dict(n=[expectation(state,a) for a in model.n],
        local_S2=[expectation(state,a) for a in model.S2_local],
        local_r2=[expectation(state,a) for a in model.r2_local],
        S=[[expectation(state,a) for a in site] for site in model.S_local],
        material_L=[expectation(state,a) for a in model.material_L],
        J=[expectation(state,a) for a in model.J],
        E=[expectation(state,a) for a in model.E],
        any_Ecap_shell=float(gauge[np.any(abs(model.sector.electric)==2, axis=1)].sum()),
        outer_radial_population=[float(np.take(radial, -1, axis=v).sum()) for v in range(3)])


def main():
    start = time.monotonic()
    config = json.loads((ROOT/(P+'input.json')).read_text())
    neutral_model = Model(6)
    h, _, _ = neutral_model.law()
    seed = prepared_reference(neutral_model)
    seed_measure = energy(h,seed)
    states, gains, report = cool(h, seed, config['neutral_taus'], config['neutral_dimension'],
                                  config['neutral_coarse_dimension'], seed_measure['energy'])
    neutral = states[-1]
    saved = dict(neutral=neutral, neutral_seed=seed)
    reference = dict(seed=seed_measure, cooling=report, selected_tau=config['neutral_taus'][-1],
                     selected=energy(h,neutral), primitive=primitive(neutral_model,neutral),
                     total_J_generator_residual=[float(np.linalg.norm(a@neutral)) for a in neutral_model.J],
                     basis_hash=neutral_model.basis_hash, status='NEW conditional cooled input; NOT ground theorem or stationary assertion')
    dump('neutral_result', reference)
    print(json.dumps(dict(stage='actual_new_neutral',result=reference)), flush=True)
    del h, states, seed
    gc.collect()
    interface = dict(kind='ACTUAL_SAME_H_FULL_SPACE_QUANTUM_ENERGY_FILTER_CLOUD',
        model_module='rich_local_frame_carrier_20261015_model', source_module=P+'source',
        parameters=neutral_model.parameters, neutral=reference, branches={},
        source_site=0, orbital=0, selected_source_tau=config['source_tau'],
        filter='A_alpha=exp[-tau(H_N-E_Omega)] B_alpha; common shift E_Omega=<Omega|H6|Omega>',
        filter_shift=reference['selected']['energy'],
        scope='Finite source preparation; full-H future evolution required. No measured electron or continuous stability.',
        domain='UNCHANGED positive u=r^(3/2)f radial grid; inherited kinetic/inverse-r2/precut products; no p_r translation',
        normalization='NPZ selected component arrays are normalized with positive real factors only. Multiply each by exp(component_log_raw_norm) for exact coherent unnormalized filter output; norm squared is filter strength, not CAR brightness or event probability.',
        symmetry='Cooling full rotational-scalar H preserves neutral singlet; source1 computed by total-J lowering divided by MEASURED literal-source coefficient. Initial failed sign convention preserved separately; no density assignment.',
        source_sha256={name:digest(ROOT/name) for name in config['source_files']})
    for branch, number in [('addition',7),('removal',5)]:
        model = Model(number)
        h, _, _ = model.law()
        raw = component_source(model, neutral_model, neutral, 0, branch=branch)
        raw_partner = component_source(model, neutral_model, neutral, 1, branch=branch)
        lowered = spin_down(model,raw)
        partner_coefficient = np.vdot(raw_partner,lowered)/np.vdot(raw_partner,raw_partner)
        partner_residual = float(np.linalg.norm(lowered/partner_coefficient-raw_partner))
        uncorrected_residual = float(np.linalg.norm(lowered-raw_partner))
        taus = config['source_taus']
        dressed, logs, details = cool(h, raw, taus, config['source_dimension'], config['source_coarse_dimension'], interface['filter_shift'])
        selected = taus.index(config['source_tau'])
        up = dressed[selected]
        down_raw = spin_down(model,up)/partner_coefficient
        down_norm = float(np.linalg.norm(down_raw))
        down = down_raw/down_norm
        saved[branch+'_component0'] = up
        saved[branch+'_component1'] = down
        saved[branch+'_bare_component0'] = raw/np.linalg.norm(raw)
        for tau, state in zip(taus,dressed):
            saved[branch+'_tau'+str(tau).replace('.','p')] = state
        branch_report = dict(number=number, dimension=model.dimension, shape=model.shape,
            basis_hash=model.basis_hash, physical_source_strength=[float(np.vdot(raw,raw).real),float(np.vdot(raw_partner,raw_partner).real)],
            bare_partner_lowering_residual=partner_residual, selected_partner_norm=down_norm,
            measured_lowering_coefficient=[float(partner_coefficient.real),float(partner_coefficient.imag)],
            uncorrected_lowering_residual=uncorrected_residual,
            component_log_raw_norm=[logs[selected],logs[selected]+float(np.log(down_norm))],
            filtered_source_strength=[float(np.exp(2*logs[selected])),float(np.exp(2*logs[selected])*down_norm**2)],
            cooling=details, selected_initial=primitive(model,up), selected_energy=energy(h,up),
            bare_initial=primitive(model,raw/np.linalg.norm(raw)), bare_energy=energy(h,raw/np.linalg.norm(raw)),
            filter_overlap_with_bare=float(abs(np.vdot(raw/np.linalg.norm(raw),up))**2),
            total_Jhalf_residual=float(np.linalg.norm(sum(a@(a@up) for a in model.J)-.75*up)))
        interface['branches'][branch] = branch_report
        dump('preparation_partial', interface)
        print(json.dumps(dict(stage=branch,result=branch_report)), flush=True)
        del h, model, dressed, up, down, raw, raw_partner, down_raw
        gc.collect()
    np.savez_compressed(ROOT/(P+'states.npz'), **saved)
    interface['state_file'] = P+'states.npz'
    interface['state_SHA256'] = digest(ROOT/interface['state_file'])
    interface['state_keys'] = list(saved)
    interface['elapsed_seconds'] = time.monotonic()-start
    dump('interface', interface)
    dump('preparation_result', interface)
    print(json.dumps(dict(stage='FROZEN_REAL_INPUT_READY', state_SHA256=interface['state_SHA256'], elapsed_seconds=interface['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
