"""Publish only this own bounded scientific review, never other running epochs."""
from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent;P='rich_common_quantum_em_review_20261016_'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def final_response(result,sector,mode,protocol):
    return result['sectors'][sector][mode]['same_age_responses'][protocol][-1]['response']


def retention(support):
    return [dict(backward=v['backward'],retained_fraction=None if v['selected_body_full_flux_virtual_norm2']==0 else v['retained_native_hop_image_norm2']/v['selected_body_full_flux_virtual_norm2'],retained_norm2=v['retained_native_hop_image_norm2'],raw_selected_body_norm2=v['selected_body_full_flux_virtual_norm2']) for v in support['selected_state_virtual_legs'] if v['edge']==2 and v['integer_charge']==-3]


def main():
    attempt=sys.argv[1];stem=P+attempt
    result=json.loads((ROOT/(stem+'_result.json')).read_text());exit_record=json.loads((ROOT/(stem+'_exit.json')).read_text());language=json.loads((ROOT/(P+'language_final.json')).read_text())
    modes=('full_quantum_connection','active_transverse_identity');own={}
    for mode in modes:
        own[mode]={}
        for protocol in ('field_kick','white_pulse'):
            pair=result['sectors']['pair'][mode]['same_age_responses'][protocol]
            neutral=result['sectors']['neutral'][mode]['same_age_responses'][protocol]
            double=result['added_source_double_contrast'][mode][protocol][-1]['response']
            own[mode][protocol]=dict(whole_pair=pair[-1]['response'],whole_neutral=neutral[-1]['response'],added_source_double_contrast=double,pair_initial_response=pair[0]['response'],pair_temporal_current_response={s:pair[-1]['response']['physical_current'][s]-pair[0]['response']['physical_current'][s] for s in ('quark','white')},meaning='Same-age perturbed-minus-baseline. Added-source contrast subtracts the separate conditional neutral response, NOT legal single-B and single-L sectors or a connected B/L cumulant/scattering amplitude.')
    verdict=dict(kind='INDEPENDENT_BOUNDED_COMMON_RECIPROCAL_GAUSS_COVER_REVIEW',physical_result=result,own_response_interpretation=own,initial_white_e2_virtual_retention=retention(result['initial_native_support']),final_white_e2_virtual_retention=retention(result['final_native_support']),exit=exit_record,typing=dict(natural_exit=language['natural_exit'],summary=language['diagnostics'].get('summary')),positive_mechanism='Live universal quantum charge phases in BOTH scalar/vector rich hops permit reciprocal material white-source/field and field/matter response in one selected quantum law. The active transverse-identity comparison is not sourceOFF; other rich channels remain.',scope='Frozen first-law only. Fresh OWN native orthonormal states and coupled preparation; no author stationary/evolved array or new loop-rich state/law splice. N32/J0 and N36/J1 selected bodies, Phi7/R10/PW19 and85 fluxes are a fundamental finite-matrix hypothesis, not full CAR64 derivation.',nonclaims=['Connected B/L cumulant or scattering','Isolated legal charged controls on closed torus','Natural primitive charges/spins/gauging','Lab rotating-density minimal EM or field spin1 Noether','Photon modes/dispersion, binding/mass/lifetime, full-H closure or convergence'])
    geometry=json.loads(Path(result['config']['pair_prefix']+'_geometry.json').read_text())
    direct_force={sector:(np.asarray(geometry['matter_profiles'][2])*result['sectors'][sector]['full_quantum_connection']['same_age_responses']['white_pulse'][0]['response']['physical_current']['white']).tolist() for sector in ('neutral','pair')}
    verdict['direct_white_t0_physical_generator_force_from_same_state_current']=dict(values=direct_force,relation='Full connection: delta d(m/3)/dt = t_edge2 delta Jwhite, from the actual changed-source current and same-H Ward relation. For integer m force multiply by3. Not applicable to the transverse-identity controlled H; later finite response is not decomposed into exclusively direct and indirect paths.')
    (ROOT/(P+'verdict.json')).write_text(json.dumps(verdict,indent=2)+'\n')
    cfg=result['config'];checks=result['physical_checks'];source=result['source'];identities=result['operator_identities'];lines=[
        'INDEPENDENT COMMON QUANTUM CHARGE-FIELD REVIEW — bounded positive mechanism, not particle/photon closure.',
        'FIRST frozen dynamics2 law only; no47/old48/native/loop-rich state or Hamiltonian splice.',
        'Own source/input/exit capsules and own raw arrays are separate; author stationary/evolved arrays/results are NOT primary proof.',
        'KINEMATICS / PREPARATION',
        'Fundamental selected CAR64/Phi7/R10/PW19 matrix law: neutral258(N32,J0), pair5318(N36,J1M1); SEVERE matter cuts, not unprojected full-CAR closure.',
        'Two compact material Cartan-cover coordinates, not independent integer edge rotors: E_Q=(D^T Lplus Nint+t Gt^-1 m)/3, Gt=8I; only2of21 charge-cycle profiles.',
        'All85 signed |m|<=6 with even my+mz retained. Other disconnected cover sectors are not claimed absent or heavier; spatial Z2 cover kernel is not colour/charge Z3.',
        f"Own native seed index {result['neutral_seed_index']}, counts {result['seed_counts']}; own Phi-only seed and coherent flux packet, then actual coupled preparation t={cfg['preparation_time']}; NOT a coupled vacuum/ground claim.",
        f"Canonical full Bdag0 W(path2->0->1->2->0) Ldag2, L FIRST, Wshift{source['Wilson_shift']}; raw fullCAR strength {source['full_CAR_Gram_strength_NOT_probability']:.12g}, body {source['body_before_flux_strength']:.12g}, retained body+flux {source['retained_body_and_flux_strength']:.12g}, fraction {source['retained_fraction']:.12g}.",
        f"Own body loss {source['body_lost_strength']:.6g}, raw flux loss {source['raw_flux_lost_strength']:.6g}; raw strengths are NOT probabilities. Source charge Ward {source['source_charge_Ward_error']:.3g}; Gauss basis errors {result['Gauss_basis_errors']}.",
        'B(upupdown colour-singlet) and charged DOWN-WHITE L are distinct halfspin operators; combined source J1M1, not one halfspin particle or C partners.',
        'OPERATOR / FIELD SUPPORT',
        f"Actual physical-generator/current/charge/source/contact derivative errors {identities['errors']}.",
        f"Physical G=m/3, not photon number/oscillator momentum. Wrong m-vs-m/3 errors {identities['WRONG_unnormalized_m_vs_m_over3_relative_errors']} reached a discriminating physical assertion.",
        f"Isolated charge+1 closed-torus Gauss residual norm {identities['isolated_physical_charge1_Gauss_residual_norm']:.6g}; no imported single-B/single-L four-sector subtraction.",
        f"Direct primitive-harmonic support {[dict(n=r['integer_charge'],edge=r['edge'],shift=r['shift'],rank=r['retained_shift_rank']) for r in result['initial_native_support']['primitive_harmonics']]}; Wrank {result['initial_native_support']['Wilson_rank']}.",
        f"ACTUAL selected-state white e2 forward/back virtual retention at input {retention(result['initial_native_support'])}; at final baseline {retention(result['final_native_support'])}.",
        'Thus direct charge-3 legs are tested, not inferred from Ward identities, active source, or clamped nonzero loops. Raw selected-body/uncut-flux norms and retained images are distinct; loss is NOT an observable error bar.',
        'FRESH SAME-H RECIPROCAL EXECUTION',
        f"Own times {cfg['times']}; white species source phase theta={cfg['theta']} on bond2 held ON, quark receiver bond0; each H has its own same-age baseline and actual-source observer.",
        f"Own y kick exp(i {cfg['physical_generator_kick_beta']}*m_y/3) translates canonical a_y by {cfg['physical_generator_kick_beta']/3}; NOT the author's exp(i .3*m_y) preparation or an electric-momentum kick.",
    ]
    for mode in modes:
        activation={sector:result['sectors'][sector][mode]['active_white_source_norm'] for sector in ('neutral','pair')}
        lines.append(f"{mode}: actual white-source activation norms {activation}; hopping/source remain active.")
        for protocol in ('field_kick','white_pulse'):
            r=own[mode][protocol];p=r['whole_pair'];d=r['added_source_double_contrast']
            lines.append(f"  {protocol} WHOLEPAIR deltaG={p['physical_transverse_generator']}, quark deltaQ={p['quark_Q_site']}, currents={p['physical_current']}, deltaPhi={p['Phi']:.6g}.")
            lines.append(f"  ADDED-source doublecontrast deltaG={d['physical_transverse_generator']}, quark deltaQ={d['quark_Q_site']}, currents={d['physical_current']}; pair temporal-current change after subtracting t0 preparation/contact={r['pair_temporal_current_response']}.")
    lines.extend([
        f"Reciprocal physical checks {checks}.",
        f"Direct t0 white contribution predicted by measured source-current change and SAME-H Ward: delta d(m/3)/dt={direct_force}. This quantifies only the direct initial leg; later finite response can also include other rich paths.",
        'Active control changes ONLY transverse U factors to identity, retaining all rich matter hops, Gauss longitudinal and field kinetic terms; CONTROL observers use decoupled=True too.',
        'White-source->G and field-kick->matter nulls in this active control isolate live-connection participation. White->quark residual through other rich channels is NOT all EM; field loop observables can still change under a field kick.',
        'Whole-pair response is not pair-minus-neutral added-source contrast; neither is a genuinely connected B/L cumulant or two-particle scattering/S matrix.',
        'POSITIVE CONSTRUCTION / LIMITS',
        'This is actual reciprocal coupling among occupied rich matter, quantum Phi/rotor/colour and a live universal charge connection, rather than pointcharge/valence-only source relabelling.',
        'Material next construction: carry the demonstrated scalar/vector charge phases, legal complete source and direct charge-3 leg support into the separately evolving loop-rich body; preserve selected-state virtual-loss readout, not premature coarse/rigid-profile gates.',
        'Primitive charges/spins, gauging, point-density/inertia and coupling constants remain inputs. Material current is not full lab rotating-charge EM, and two Cartan modes are not photon branches.',
        'No binding/mass/lifetime, freeCOM/Lorentz dispersion, physical charge derivation, full-H closure or continuum convergence follows from reciprocal finite response.',
        f"Natural exit {exit_record['natural_exit']}, elapsed {exit_record['elapsed_seconds']:.6g}s; own source/bundle/input unchanged flags retained. Selected diagnostics exit {language['natural_exit']}, {language['diagnostics'].get('summary')}; rooted LSP exit {language['rooted_LSP']['natural_exit']}.",
    ])
    assert len(lines)<=80,len(lines)
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    excluded={P+'capsule.zip',P+'manifest.json',P+'capsule_receipt.json'}
    files=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in excluded)
    manifest={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in files}
    (ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(P+'capsule.zip')
    with zipfile.ZipFile(capsule,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        z.write(ROOT/(P+'manifest.json'),P+'manifest.json')
    receipt=dict(capsule=capsule.name,sha256=digest(capsule),bytes=capsule.stat().st_size,synthesis_lines=len(lines),physical_natural_exit=exit_record['natural_exit'],typing_natural_exit=language['natural_exit'],scope=verdict['scope'])
    (ROOT/(P+'capsule_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
