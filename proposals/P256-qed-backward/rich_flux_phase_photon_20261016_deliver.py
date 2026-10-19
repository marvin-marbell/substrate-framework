"""Compact physical synthesis; separately preserve complete raw-array evidence."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
P='rich_flux_phase_photon_20261016_'


def read(name):return json.loads((ROOT/(P+name+'.json')).read_text())


def main():
    result=read('result');completion=read('completion');exit_record=read('exit');language=read('language_final')
    states=result['results'];lines=[
        'Intrinsic pi-flux joint-state attempt on fixed actual V64 quantum-link Hubbard H',
        'Scope: provisional known flux-phase mechanism, not a new law/novelty or claimed photon.',
        'Fixed: 4x4x4, 64 sites, 192 distinct positive arcs; Nup=Ndown32; local Fock4/link3;',
        'ions1, t1/U4/I16/Ecap1, external H phase EXACTLY0, no magnetic/Maxwell term.',
        'State: legal Gauss projector times correlated determinant and active quantum-electric envelope,',
        'times exp(i pi codes.E), codes=(0,x,x+y). Matching signed-h orbitals change the fermionic STATE.',
        'This does not insert Peierls phases into production H or compare relabelled external-H spectra.',
        'Actual incidence checks: 192/192 contractible pi plaquettes, 48/48 trivial L4 winding phases.',
        'Each pi state includes full64-mode flux-adapted pair orbitals or a material AF-projector x-dimer.',
        'The AF dimer breaks singlet/cubic symmetry and is an ordered fixed-Sz0 alternative, not a photon vacuum.',
        'All original legal nearest-hop Fock/link virtuals retained; no pair-count restriction or frozen field.',
        'Positive nodal guide is not a state coefficient; absolute Hpsi/w and |Hpsi|^2/w retain discrete nodes.',
    ]
    ring=result['ring'];keys=[k for k in ring if k!='meaning']
    lines.append('Exact correlated4-site U4 clamped ring: E(0)=%.9g, E(pi)=%.9g (36-dimensional full sector).'%(ring[keys[0]]['ground'],ring[keys[1]]['ground']))
    lines.append('Ring tests fermionic loop preference ONLY; qutrit U is nonunitary/nilpotent, not a cnumber phase.')
    for name,state in states.items():
        q=state['quality'];chains=state['chains'];d=state['diagnostics']
        lines.append('%s: E=%.8g +/-%.3g blockSE; VarH=%.8g +/-%.3g; RMS=%.6g; Gaussmax=%d.'%(name,q['energy'],q['energy_blockSE'],q['full_H_variance'],q['variance_blockSE'],q['residual'],state['maxGauss']))
        lines.append('  %d samples; importanceESS=%.5g; chain energy tau=%s; autocorrESS=%s; doublon observed=%s.'%(q['samples'],q['importance_ESS'],','.join('%.4g'%c['tau'] for c in chains),','.join('%.4g'%c['autocorrelation_ESS'] for c in chains),q['doublon_observed_range']))
        lines.append('  Independent-chain E: %s; acceptance counts %s.'%(','.join('%.7g'%c['energy'] for c in chains),str([c['accepted'] for c in chains])))
        lines.append('  CAR Python-int128 edges=%d maxerr=%.3g; real-source J/contact FD relative errors %.3g/%.3g.'%(d['CAR_128mode_Python_integer_edges'],d['CAR_max_error'],d['current_first_relative'],d['contact_second_relative']))
        lines.append('  Exact-node adverse consumer: psi2=%.3g, Hpsi2=%.5g, guide=%.5g, H2/guide=%.5g.'%(d['nodal_psi_squared'],d['nodal_Hpsi_squared'],d['nodal_positive_guide'],d['nodal_H2_over_guide']))
        for r in state['responses']:
            lines.append('  q=%s 2T+rho +/-q rank%d; J outside fractions=%s; sample |Gdot|=%.4g, sample connection=%.4g.'%(r['q'],r['rank'],','.join('%.4g'%f for f in r['current_outside_fraction']),r['metric_derivative_norm'],r['moving_connection_norm']))
            lines.append('    contact Re diag=%s; frozen static contact-chi Re diag=%s.'%(','.join('%.5g'%r['contact_real'][i][i] for i in range(2)),','.join('%.5g'%r['frozen_static_twist_real'][i][i] for i in range(2))))
        if state['zero_phase_same_pi_orbitals'] is not None:
            control=state['zero_phase_same_pi_orbitals']
            lines.append('  Same pi orbitals/guide, phase removed control E=%.7g, VarH=%.7g; not a refitted state.'%(control['energy'],control['full_H_variance']))
    for order in ('pair','af_dimer'):
        zero=states['zero_'+order]['quality'];pi=states['pi_'+order]['quality']
        lines.append('Measured pi-minus-zero %s: dE=%.7g, dVar=%.7g; finite-chain differences, not converged variational ranking.'%(order,pi['energy']-zero['energy'],pi['full_H_variance']-zero['full_H_variance']))
    loop_result=read('quantum_loops');loop_exit=read('quantum_loops_exit');source_review=read('source_semantics_review')
    for name,loop in loop_result['results'].items():
        lines.append('Actual nonunitary qutrit loop %s: plaquette=%.7g +/-%.3g, winding=%.7g +/-%.3g blockSE; fresh768 samples.'%(name,loop['plaquette'],loop['blockSE'][0],loop['winding'],loop['blockSE'][1]))
    lines.append('Loop witness natural exit=%d; U-product nilpotent boundary zeros retained, NOT a cnumber substitution.'%loop_exit['natural_exit'])
    for name in ('zero_pair','pi_pair'):
        q=states[name]['quality'];potential=4*q['doublons']+q['field_E2']/32.
        lines.append('%s energy decomposition: onsite+electric=%.7g, hopping=%.7g (same measured state/H).'%(name,potential,q['energy']-potential))
    lines.append('Actual [H,F] identity relative %.3g; WRONG fullH-minus-E source discrepancy %.5g.'%(source_review['actual_field_commutator_identity_relative'],source_review['wrong_fullH_minus_energy_source_relative']))
    lines.extend([
        'Both initial pair and the physically material AF/order alternative were actually run, with fixed input constants.',
        'All states remain nonstationary on measured full H; no stationary photon pole or demonstrated vacuum follows.',
        'K=<B Omega|[H,B]Omega> from HB-BHpsi, NOT full-H Kfull; both saved, neither conflated.',
        'Gdot=i(K-Kdag); exact initial Gdot/connection=0 by real-H/real-state diagonal-observable symmetry.',
        'Reported sample Gdot/connection are MC drift, NOT demonstrated metric motion; later-time metric not fixed.',
        'Earlier full-H-metric interpretation explicitly superseded; exact source/exit retained in superseded capsule.',
        'Saved +/-q current/contact frozen-time resolvents are coupled diagnostics, NOT stationary Kubo/Lehmann dynamics.',
        'Large current outside E/rho span can be UV pairs, not a forced closure criterion or a law failure.',
        'No reducible contact-minus-current kernel is fed through Dyson again; joint static cancellation is not absence.',
        'Known magnetic-translation Clifford analogy would be BAND/pseudospin, not physical Noether spin or Nature electron.',
        'Uncertainty: consecutive-block SE and positive-sequence weighted-energy tau are exploratory; no mixing proof.',
        'Only two finite chains/state; importance ESS is NOT autocorrelation ESS; q_min=pi/2, no continuum/low-q claim.',
        'Metropolis proposals are symmetric legal hop/closed-loop/spin swaps; guide detailed balance derived in equations.json.',
        'One initial compiler failure (Numba int dot) is preserved with exact exit1/source in failed_compile_capsule.zip.',
        'Science natural exit=%d; full completion=%s; source unchanged=%s.'%(exit_record['natural_exit'],completion['natural_scientific_completion'],exit_record['sources_unchanged']),
        'Final selected Pyright exit=%d, errors=%d; rooted LSP exit=%d, positive AST and full diagnostics retained.'%(language['natural_exit'],language['diagnostics']['summary']['errorCount'],language['rooted_LSP']['natural_exit']),
        'Evidence: input/interface/equations, all per-state raw arrays, response matrices, ring/geometry, exact exit, source capsule.',
    ])
    assert len(lines)<=80
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    summary=dict(scope='ACTUAL_V64_INTRINSIC_PI_STATE_AND_COUPLED_2T_CURRENT_DIAGNOSTIC_NO_PHOTON_CLAIM',states={name:dict(quality=s['quality'],maxGauss=s['maxGauss']) for name,s in states.items()},natural_exit=exit_record['natural_exit'],completion=completion,language_summary=language['diagnostics']['summary'],summary_lines=len(lines))
    (ROOT/(P+'synthesis.json')).write_text(json.dumps(summary,indent=2)+'\n')
    files=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in (P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'))
    dependencies=[ROOT/'rich_bulk_photon_20261016_state.py',ROOT/'rich_bulk_photon_20261016_run.py',ROOT/'rich_bulk_photon_20261016_language.py']
    files+=dependencies
    manifest=dict(files={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files},source_sha256=exit_record['source_sha256'],science_natural_exit=exit_record['natural_exit'])
    (ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[ROOT/(P+'manifest.json')]:archive.write(path,path.name)
    receipt=dict(capsule=capsule.name,sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),bytes=capsule.stat().st_size,science_natural_exit=exit_record['natural_exit'],summary_lines=len(lines))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
