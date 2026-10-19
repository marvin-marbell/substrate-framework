"""Bounded physics handoff; raw MPS/current images remain separate artifacts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_recoil_current_wave_20261016_'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.array(a)-b))


def main() -> None:
    stamp = sys.argv[1]
    stem = PREFIX+stamp+'_'
    result = json.loads((ROOT/(stem+'result.json')).read_text())
    exit_data = json.loads((ROOT/(stem+'exit.json')).read_text())
    emission = json.loads((ROOT/(PREFIX+sys.argv[2]+'_emission_result.json')).read_text())
    typing = json.loads((ROOT/(PREFIX+'language_final.json')).read_text())
    source_check = json.loads((ROOT/(PREFIX+'source1_source_check_result.json')).read_text())
    seed_details = json.loads((ROOT/(stem+'seed_details_result.json')).read_text())
    field_receivers = json.loads((ROOT/(stem+'field_receivers_result.json')).read_text())
    cases = result['cases']
    required = {f'N{number}_{protocol}' for number in [54, 55, 53] for protocol in ['baseline', 'xcos', 'ycos']}
    if set(cases) != required or exit_data['returncode'] != 0 or set(emission['cases']) != required or len(seed_details) != 25 or len(field_receivers['cases']) != 9:
        raise ValueError('Incomplete physical construction is not a delivery')
    lines = [
        'NEW FULL108-CAR CURRENT-DRESSED RECOIL WAVE: finite joined physical construction, not an independent review.',
        'Same chosen3^3 law/couplings/CAR/Clifford/spin/scalar data; no Maxwell magnetic, Pauli, new free-Dirac or auxiliary COM sector.',
        'Full108 fermionic occupation sites plus scalar/EM75-state site; no hard local Fock/particle-hole cap or old27/162 Gaussian-source-span closure.',
        'N and actual physical C3z are explicit tensor charges. Three total-C3 components remain a coherent spatial packet, including which(kx,ky)block carries the canonical addition/removal.',
        'Fresh reference Slater is a PREPARED neutral, not a coupled ground. Matched N54 propagates under the new coupled law at the same absolute clock for every protocol.',
        'Actual original H and four signed before-cut Q_counter J images supply a real .01 physical current-Galerkin evolution, then occupied-Schmidt two-site TDVP for .095+.095; no zero padding or extra QJ interaction term.',
        'Both transverse finite projected-Q-only preparations are labelled finite protocols: NOT full-W, pure photons or asymptotic states.',
        'J=-d_A H and contact=d_A d_A H share the owning Peierls law. Current is neither momentum nor a unitary momentum shift.',
        'QJ and JQ are distinct before-cut products; shell3 is the exact sufficient intermediate for Q times the full Peierls current compressed to shell2.',
        'Model units: hbar=lattice spacing=1; momentum labels2pi*r/3 are modular, densities are particle counts, spin is in hbar units. Raw J/QJ Gram weights have squared current/coordinate-current units, NOT calibrated emission probabilities or density contrasts.',
        f'Natural physical exit={exit_data["returncode"]}, signal={exit_data["termination_signal"]}; elapsed={result["elapsed_seconds"]:.6g}s; peakKiB={result["peak_resident_KiB"]}.',
        f'Own literal source consumer: {source_check["tested_complex_source_actions"]} complex CAR/JW actions, maxerror={source_check["maximum_complex_action_error"]:.4g}; raw source sums use ACTUAL coefficient norm times own reference norm, not an authored target.',
        'Earlier actual1/2 were intentionally cancelled encoding epochs, not OOM/physics failures. Natural exit1 epochs3(layout),4(raw operator API/invalid zero-Schmidt padding),5(charged-source charge inference) remain preserved; partial neutral5 data are not relabelled the completed charged construction.',
    ]
    for number in [54, 55, 53]:
        for protocol in ['baseline', 'xcos', 'ycos']:
            case = cases[f'N{number}_{protocol}']
            first, final = case['observations'][0], case['observations'][-1]
            control0, control1 = case['frozen_interaction_control'][0], case['frozen_interaction_control'][-1]
            lines.append(f'N{number}/{protocol}: rawsource={case["raw_source_norm2"]:.12g}; dm={distance(final["momentum"]["matter"],first["momentum"]["matter"]):.7g}, df={distance(final["momentum"]["field"],first["momentum"]["field"]):.7g}, dtotal={distance(final["momentum"]["total"],first["momentum"]["total"]):.3g}; frozen dm/df={distance(control1["momentum"]["matter"],control0["momentum"]["matter"]):.3g}/{distance(control1["momentum"]["field"],control0["momentum"]["field"]):.3g}.')
            lines.append(f'  spin0={first["primitive_frame_spin"]}; spinT={final["primitive_frame_spin"]}; sea P0/P1/P>=2={final["reference_depletion_P0_P1_Pge2"]}; dE={final["physical_energy"]-first["physical_energy"]:.7g}; maxbond={final["maximum_bond"]}.')
    for number in [55, 53]:
        baseline = cases[f'N{number}_baseline']['observations']
        neutral_base = cases['N54_baseline']['observations']
        for protocol in ['xcos', 'ycos']:
            drive = cases[f'N{number}_{protocol}']['observations']
            neutral = cases[f'N54_{protocol}']['observations']
            excess_delta = (np.array(drive[-1]['density'])-neutral[-1]['density'])-(np.array(baseline[-1]['density'])-neutral_base[-1]['density'])
            lines.append(f'N{number}/{protocol} paired: t0 body-density change={distance(drive[0]["density"],baseline[0]["density"]):.3g}; final body change={distance(drive[-1]["density"],baseline[-1]["density"]):.7g}; propagated-neutral-subtracted excess-response L2={np.linalg.norm(excess_delta):.7g}.')
            lines.append(f'  currentsT={drive[-1]["currents"]}; same-law contact, spin-resolved current form factors and both equal-time QJ/JQ orders are retained in raw results.')
    for protocol, axis, receiver in [('xcos', 0, 2), ('ycos', 2, 0)]:
        contrasts = field_receivers['pulse_contrasts'][f'N55_{protocol}']
        first, final = contrasts[0], contrasts[-1]
        lines.append(f'FIELD/{protocol}: initial drive Pi={first["pulse_minus_baseline"]["Pi"][axis]:.9g}, other-T Pi={first["pulse_minus_baseline"]["Pi"][receiver]:.3g}; own QQ change={first["pulse_minus_baseline"]["QQ"][axis][axis]:.8g}; both signed field-C3 populations already={cases[f"N55_{protocol}"]["observations"][0]["momentum"]["field"][1:]} at preparation, not later emission.')
        lines.append(f'  Other-T Pi atT: coupled pulse contrast={final["pulse_minus_baseline"]["Pi"][receiver]:.8g}, frozen={final["frozen_pulse_minus_baseline"]["Pi"][receiver]:.3g}; frozen other-T Pi2 contrast={final["frozen_pulse_minus_baseline"]["PiPi"][receiver][receiver]:.8g}. All N55/N53/matched-neutral before-cut covariance receivers retained; finite-Q protocol is not relabelled full-W.')
    for number in [55, 53]:
        endpoints = emission['cases'][f'N{number}_baseline']['endpoints']
        for index, final in enumerate(endpoints[-1]['coupled']):
            initial = endpoints[0]['coupled'][index]
            difference = [a-b for a, b in zip(final['QJ'], initial['QJ'])]
            neutral_excess = endpoints[-1]['charge_minus_propagated_matched_neutral'][index]['QJ']
            frozen_difference = endpoints[-1]['coupled_minus_frozen_original_law_probe'][index]['QJ']
            protocol = 'xcos' if final['polarization'] == 0 else 'ycos'
            pulse_excess = emission['pulse_contrasts'][f'N{number}_{protocol}'][-1]['charge_excess_pulse_minus_baseline'][index]['QJ']
            lines.append(f'N{number}/T{final["polarization"]}/sign{final["profile_sign"]}: JOINT J grade={final["actual_current_C3_charge"]}; J(T)={final["J"]}; delta<QJ>={difference}; <QJ>minus matched neutral={neutral_excess}; coupled-minus-original-law-frozen-probe={frozen_difference}; paired excess pulse delta<QJ>={pulse_excess}.')
    initial_metrics = emission['completed_initial_N55_expanded_receiver_metrics']
    for family in ['J', 'QJ']:
        gram = initial_metrics['coherent_cross_vertex_Gram'][family]
        lines.append(f'EXERCISED N55-clock0 {family} receiver Gram diagonal weights={[gram[i][i][0] for i in range(4)]}; full complex4x4 and conditional recoil/spin/depletion retained. These are raw model-unit insertion weights, not emission probabilities.')
    maximum_truncation = max(float(step[r]['truncation_eps']) for case in cases.values() for step in case['evolution'] for r in step)
    lines += [
        'Emission image norms/form factors are not transition probabilities. J shifts total C3 by its measured finite-group character; counterfield QJ mixes no total-C3 sectors, but nonunitary conditioning can reweight their norms.',
        'Exact source+MPO primitive-image networks are retained separately from committed expanded MPS layers. Evolved signed insertion AMPLITUDES above were actually contracted; optional uncontracted evolved-image norm/Gram/spin/depletion catalogue is explicitly UNEXERCISED, not guessed or approved. Completed N55-clock0 expanded receiver metrics and all25 raw six-direction metrics retain their exact scopes.',
        'Frozen controls propagate their OWN generator from the SAME clock0 rays, including the first .01; real body hopping/convection persists but scalar/EM feedback is removed. Original coupled current is not renamed a frozen-H field-source derivative.',
        'n_site/N is inclusive particle POVM; signed propagated-neutral-subtracted excess is not a probability or an excitation-position PVM.',
        f'Maximum accumulated TDVP SVD epsilon={maximum_truncation:.8g}; numerical current-space thresholds, initial-ray overlaps, bond ranks and physical energy changes are retained, not replaced by norm conservation.',
        f'All25 raw current-sector phase/Gram/H/vector records retained; minimum raw metric eigenvalue={min(min(v["raw_Gram_eigenvalues"]) for v in seed_details.values()):.8g}. No state channel is discarded into the old canonical span.',
        'emit1 export was SIGKILL(-9), not natural; OOM remains suspected, not established. emit2 optional expanded catalogue was stopped on parent direction, with completed layers immutable. Final central export uses existing actual6 contractions/exact primitive networks; no new dynamics, preparation or Hamiltonian.',
        'No field/bond/time/continuum convergence or full-H/global-ground/domain claim. One q/finiteC3, primitive fixed spin, supplied CAR/charge/Clifford/scalar and absent nonuniform longitudinal sector remain explicit.',
        'No free/continuous COM, Noether-derived halfspin, photon pole/on-shell radiation, Lorentz/inertial mass, proton or Nature claim.',
        f'Final selected typing: exit={typing["natural_exit"]}, summary={typing["diagnostics"]["summary"]}; real rooted LSP exit={typing["rooted_LSP"]["natural_exit"]}.',
        'No old/nativeCommon/47/OLD48 evolved state, solved operator or physical result is spliced into this construction. Selected immutable old translation-law constructors are construction input, not independent validation.',
    ]
    if len(lines) > 80:
        raise ValueError(f'Physics synthesis exceeds80 lines: {len(lines)}')
    synthesis = ROOT/(PREFIX+'synthesis.txt')
    synthesis.write_text('\n'.join(lines)+'\n')
    interface = dict(kind='FULL_CAR_CURRENT_DRESSED_RECOIL_CONSTRUCTION_INTERFACE', root=str(ROOT),
        entrypoint=PREFIX+'capture.py <newstamp>', law='rich_spatial_translation_20261015_actual3_input.json',
        captured_physical_plan_revision='813490dc61fdae6c7b9b049ec7f59e2f9fc63415db5461f0f80e39ffecd139e4',
        current_parent_plan_revision='b0f0f4f1d68cb97be000731234587faa9fa094290dc87d837f9cf62bbf2581fc',
        APIs=dict(Law='Full108-CAR+scalar/EM MPS and owning same-law Hamiltonian/current/contact/counterfield-emission operators.',
            prepared='Canonical N54/N55/N53 preparation with coherent total-C3 completion of the finite field-only protocol.',
            prepare_current_space='Real same-H physical short evolution in psi,Hpsi,four full before-cut QJpsi, actual nonorthogonal metric, then occupied-Schmidt full-Fock TDVP; absolute clock and raw phase data explicit.',
            CoherentWave='Exact dense tensor contractions of the coherent C3 sum, including inter-(kx,ky) inclusive density/spin.',
            emission='Existing actual6 signed J/QJ/JQ contractions with propagated matched-neutral and original-law frozen-state probes; exact primitive TN images. Expanded receiver norms/Grams/spin are exercised only in separately completed records, not inferred from descriptors.'),
        scope='Finite regulator and adaptive numerical projection, not old finiteH equality or Nature proof.',
        source_sha256={p.name: sha(p) for p in sorted(ROOT.glob(PREFIX+'*.py'))})
    (ROOT/(PREFIX+'interface.json')).write_text(json.dumps(interface, indent=2)+'\n')
    capsule = ROOT/(PREFIX+'delivery_capsule.zip')
    manifest_path = ROOT/(PREFIX+'manifest.json')
    receipt_path = ROOT/(PREFIX+'delivery_receipt.json')
    paths = [p for p in sorted(ROOT.glob(PREFIX+'*')) if p.is_file() and p not in [capsule, manifest_path, receipt_path]]
    manifest = dict(files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths}, synthesis_lines=len(lines),
        physical_exit=exit_data, source_and_scope='All executed epochs retain their own immutable input/source/exit captures. Current source hashes do not relabel earlier computations.')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(capsule, 'w', compression=zipfile.ZIP_STORED) as archive:
        for p in paths+[manifest_path]:
            archive.write(p, p.name)
    receipt = dict(capsule=capsule.name, sha256=sha(capsule), bytes=capsule.stat().st_size,
                   manifest_sha256=sha(manifest_path), synthesis_lines=len(lines))
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
