"""Bind actual owning observations, failures and finite scope to exact artifacts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from rich_local_frame_carrier_20261015_model import architecture

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_local_frame_carrier_20261015_'


def load(name):
    return json.loads((ROOT/(PREFIX+name+'.json')).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (ROOT/(PREFIX+name+'.json')).write_text(json.dumps(value, indent=2)+'\n')


def main():
    result, interface = load('result'), load('interface')
    cases = result['cases']
    operators = architecture()
    gamma = [operators['gamma0']]+operators['gamma']
    one_spin = operators['one_spin']
    tangent = [[1j*(s@g-g@s) for g in gamma] for s in one_spin]
    metric = dict(radial_trace=float(sum(np.trace(g@g).real for g in gamma)),
                  angular_trace=[[float(sum(np.trace(a@b).real for a, b in zip(tangent[i], tangent[j]))) for j in range(3)] for i in range(3)],
                  radial_angular_cross=[float(sum(np.trace(g@a).real for g, a in zip(gamma, direction))) for direction in tangent],
                  explanation='Actual constitutive matrix trace at r=1: radial16, angular8delta, cross0; hence I_v(r)=M_rad r²/2. Independent cone fields, not a global density split.')
    findings = {}
    for branch in ['addition', 'removal']:
        findings[branch] = {}
        for variant in ['vector_off', 'full']:
            case = cases[branch+'_'+variant]
            d0, de = case['absolute_time_excess'][0], case['absolute_time_excess'][-1]
            end = case['records'][-1]
            dn, dz = np.array(de['n']), np.array(de['S_local'])[:, 2]
            signed_number = 1 if branch == 'addition' else -1
            # This signed expectation identity is NOT a particle-position PVM.
            locked = .5*signed_number*dn
            findings[branch][variant] = dict(
                source_raw_strength=cases[branch+'_source']['raw_strength'],
                initial_excess=d0, final_absolute_time_excess=de,
                source_site_signed_charge_departure=float(1-signed_number*dn[0]),
                local_spin_z_minus_signed_half_excess_n=(dz-locked).tolist(),
                local_spin_charge_lock_L2=float(np.linalg.norm(dz-locked)),
                final_signed_r2_excess_norm=float(np.linalg.norm(de['r2_local'])),
                final_material_L_excess=de['material_L'],
                final_reference_positive_orbital_count=float(sum(cases['neutral_'+variant]['records'][-1]['positive_root_occupancy'])),
                local_count_PVM=end['local_count_PVM'],
                local_count_spin_connected=end['n_S2_connected'],
                local_count_norm_connected=end['n_r2_connected'],
                absolute_time_baseline=True)
    pulse = cases['addition_radial_pulse']
    interventions = dict(radial_kick='exp(.7 i r0²), same full H and source; initial count, spin and r² unchanged',
                         initial_difference=pulse['same_H_minus_unpulsed'][0],
                         final_difference=pulse['same_H_minus_unpulsed'][-1],
                         final_charge_difference_L2=float(np.linalg.norm(pulse['same_H_minus_unpulsed'][-1]['n'])),
                         final_norm_difference_L2=float(np.linalg.norm(pulse['same_H_minus_unpulsed'][-1]['r2_local'])))
    actual = [case for case in cases.values() if 'records' in case]
    resolution = dict(max_norm_error=max(c['norm_error_max'] for c in actual),
                      max_energy_drift=max(c['energy_drift_max'] for c in actual),
                      max_direct_vs_two_half_norm=max(c['endpoint_direct_vs_two_half_norm'] for c in actual),
                      max_charge_continuity=max(r['charge_continuity_max'] for c in actual for r in c['records']),
                      max_electric_current_error=max(r['electric_current_max'] for c in actual for r in c['records']),
                      max_total_Noether_torque=max(r['total_Noether_torque_max'] for c in actual for r in c['records']),
                      max_electric_cap_shell=max(r['electric_cap_shell'] for c in actual for r in c['records']),
                      qualification='Adaptive sparse exponential direct .6 vs two .3 endpoints; no claim of radial/material/electric continuum convergence. Both paths use full unprojected finite H.')
    synthesis = dict(kind='EXECUTED_RICH_LOCAL_NOETHER_NORM_AND_CHARGE_RESULT_NOT_ADMISSION',
        question='Does local physical matrix-frame angular momentum/norm dressing co-move with a true rich charged-source response?',
        answer='Actual local LEFT spin, quantum norm and charge respond in one fullCAR12 law. A rigid one-profile body/electron interpretation must confront the measured profile differences, dynamical interband sea, radial feedback and separate material angular momentum. This finite propagation is not an autonomous stable particle proof.',
        assumptions=load('equations'), frozen_model_SHA256=result['model_SHA256'],
        reference=interface['neutral_reference'], primitive_provenance=interface['primitive_provenance'], actual_matrix_orbit_metric=metric,
        input='Prepared occupied negative-orbital neutral INPUT, NOT an interacting vacuum theorem; actual separately evolving absolute-time neutral baseline for every variant.',
        source='Local physical D(g_v)c, exact CAR and real boundary Wilson tree, spin component0 orbital0 at site0, both raw signed strengths preserved.',
        findings=findings, same_H_field_feedback=interventions, resolution=resolution,
        material_failed_route=dict(
            attempted='Actual rich scalar-only transport (vector_off) with all4CAR/site, independent quantum local norms, local frames, full Gauss and material factors retained.',
            failure='Scalar total-frame torque0 does not fail local spin transport: scalar charge/spin can co-move. It does NOT produce a rigid co-moving norm cloud or derive unique autonomous electron dynamics. The norm starts unshifted and relaxes/responds rather than being inserted as a co-moving particle envelope.',
            material_change='Turn on noncommuting relative-frame/vector transport in the SAME owning primitive law and execute both signed branches anew; actual material torque/interband dynamics retained, no old result splice.',
            result='Full law is a conditional local dressed-source response; source spin/dressing versus separated collective responses is a measured finite outcome, not permission to force an electron label.'),
        limitations=[
            'Product local physical frames/right gauging, constitutive Clifford orbit and primitive CAR/statistics are explicit assumptions; no emergence or full-matrix ordering proof.',
            'Localr3 finite difference with actual inherited measure/quantum potential, materiall0+l1 and Ecap2 are finite regulators; report shell/node weights, not continuum convergence.',
            'Signed excess n/Q/S/r² are expectations, NOT added-particle position PVM. Actual local count/spin PVMs and correlations supplement them.',
            'Initial source/reference correlations are not transport. Own t0 changes and evolving neutral baseline are required and retained.',
            'r² is matrix norm, not spatial radius; local response is not selfbinding/freeCOM/Lorentz.',
            'No global-vacuum, measured positron, Nature electron/proton, SU3quark or Maxwell/massless3Dphoton claim.'
        ],
        exits=dict(reference=load('reference_exit'), scientific_run=load('run_exit'), process=load('process_exit')),
        independent_consumer='LocalCarrierReview31 owns separate raw inspection and NEW field intervention/trajectory on this declared shared input; no claim of independent ground solve.',
        final_diagnostics='language_final.json is selected owned-file evidence only; no ancestor/project-clean claim')
    save('synthesis', synthesis)
    selected = sorted(p for p in ROOT.glob(PREFIX+'*') if p.is_file() and p.name not in [PREFIX+'manifest.json', PREFIX+'delivery_capsule.zip', PREFIX+'delivery_receipt.json'])
    manifest = dict(kind='EXACT_OWNING_LOCAL_CARRIER_DELIVERY', files={p.name:dict(SHA256=sha(p), bytes=p.stat().st_size) for p in selected},
                    reference_SHA256=interface['neutral_reference']['SHA256'], frozen_model_SHA256=result['model_SHA256'])
    save('manifest', manifest)
    with zipfile.ZipFile(ROOT/(PREFIX+'delivery_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as capsule:
        for path in selected+[ROOT/(PREFIX+'manifest.json')]:
            capsule.write(path, path.name)
    save('delivery_receipt', dict(capsule=PREFIX+'delivery_capsule.zip', SHA256=sha(ROOT/(PREFIX+'delivery_capsule.zip')),
                                  bytes=(ROOT/(PREFIX+'delivery_capsule.zip')).stat().st_size,
                                  manifest_SHA256=sha(ROOT/(PREFIX+'manifest.json'))))
    print(json.dumps(dict(findings=findings, feedback=interventions, resolution=resolution)), flush=True)


if __name__ == '__main__':
    main()
