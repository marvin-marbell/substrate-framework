"""Bound exact leading mechanism, preserve all phases and original source epochs."""
from pathlib import Path
import json,zipfile
import numpy as np
from rich_full108_recoil_review_20261017_capture import digest
from rich_full108_recoil_review_20261017_analysis import cx

ROOT=Path(__file__).resolve().parent;P='rich_full108_recoil_review_20261017_'


def main():
    analysis=json.loads((ROOT/(P+'analysis1_result.json')).read_text());physics=analysis['physical_result_with_complex_sector_metrics'];summary=analysis['cases_summary'];language=json.loads((ROOT/(P+'language_final.json')).read_text());exits={name:json.loads((ROOT/(P+name+'_exit.json')).read_text()) for name in ('wick1','analysis1')};frozen=physics['config']['frozen']
    verdict=dict(kind='INDEPENDENT_POSITIVE_EXACT_SELECTED_ORDER_FULL108_QUANTUM_RECOIL_NOT_FINITE_TIME_APPROVAL',physical_assertions=physics['physics_assertions'],physical_epoch='wick1',analysis_epoch='analysis1',natural_exits=exits,scope='Same full108CAR/75 boson primitive law. Fresh own Slater/source/field packets; exact O(t²) modular recoil and O(t) signed current response. Not author t=.2 magnitudes or solver approval.',important_order_finding=analysis['scope_correction'],phase_export_repair=analysis['phase_export_repair'],typing=dict(natural_exit=language['natural_exit'],summary=language['diagnostics'].get('summary')),limits=physics['limits'])
    (ROOT/(P+'verdict.json')).write_text(json.dumps(verdict,indent=2)+'\n')
    lines=[
        'INDEPENDENT FULL108 RECOIL: positive exact LEADING quantum mechanism, NOT finite-time/particle/photon approval.',
        'Same3^3 lattice108CAR x75boson (Phi5 x4real-mode total-Fock-shell<=2); all108 bands/fullFock virtual contractions retained, no local/pair cap or old27/162 source-span closure.',
        'Primary inputs ONLY frozen pure single_particle/shell_fields definitions. Native12.622GB capsule e3e1c4df95c45abefb96c8ef6c462ac9daeba36006b73a2646b38c2f05bd260d streamed-hashed; ONLY3 tiny source/input members read, no author Model/state/evolved output loaded.',
        'Actual6 model72e5... versus emit3 modelc966... are distinct source epochs: selected primitive function bodies equal after type/doc normalization, all7 old pure dependencies identical. No blanket unchanged-source claim.',
        'OWN METHOD / STATE: normalized54-valence Slater from fixed reference operator, own complex3D canonical packet/generic primitive spinor, own Phi Gaussian center1.05 width.31; input is PREPARED, not a coupled ground.',
        'Own source norm² addition=.42293735059806153, removal=.5770626494019386; sum1.0000000000000002 equals coefficient norm1. Raw source weights are not emission probabilities.',
        'Full-Fock Wick2=<A><B>+Tr(ABrho)-Tr(ArhoBrho); twisted determinant contracts ACTUAL C3. No single-particle/mean-current closure. Exact5CAR sign/order fixture error3.71e-14 is algebra ONLY, not the108-mode physical proof.',
        'Observable(t)=c0+c1*t+c2*t²+O(t³). Recoil/field receiver coefficients exact throughc2; signed J/QJ/JQ current coefficients throughc1. No chosen finite-time magnitude or remainder bound inferred.',
        'JOINT TRANSLATION, NOT CURRENT=Pcm',
        'Independent geometric z-shift versus actual orbital translation error1.04e-14; fullFock positive-HS joint-H covariance residual3.1193e-16.',
        'BothT signed J grades=(1,2,1,2), joint character residual<=9.33e-16. Demanded MATTER-only current grade fails relative.491910; current is not a unitary momentum shift.',
        'WRONG inverse field character rejects physical joint-H covariance relative.01591848. Sector-weight conservation alone was NOT the covariance proof.',
        'Raw joint(matter,field) modular-C3 coefficients retain correlation and three coherent total sectors. Marginal principal momenta2pi*r/3 need NOT add as continuum momenta.',
        'OWN COUPLED RECOIL / MATCHED NEUTRAL: vectors below are c2 for [r0,r+1,r-1], NOT finite probabilities or emissions.',
    ]
    for number in (55,53):
        for pol,name in enumerate(('xcos','ycos')):
            key=f'N{number}_{name}';row=summary[key];pair=physics['paired_propagating_neutral_time_coefficient_contrasts'][key];delta=np.asarray(pair['dynamic_neutral_subtracted_recoil_coefficients']);current=cx(pair['dynamic_neutral_subtracted_signed_current_coefficients']['J']);driven=current[2*pol:2*pol+2,1]
            lines.append(f"{key}: matterc2={np.round(row['matter_modular_recoil_time_coefficient2'],10).tolist()}, fieldc2={np.round(row['field_modular_recoil_time_coefficient2'],10).tolist()}; frozen maxderiv={row['frozen_joint_time_derivative_max']:.3g}.")
            lines.append(f"  Evolving-neutral SUBTRACTED pulse excess: matterc2L2={np.linalg.norm(delta[:,:,2].sum(axis=1)):.9g}, fieldc2L2={np.linalg.norm(delta[:,:,2].sum(axis=0)):.9g}; driven signedJ c1(+,-)={[ [float(z.real),float(z.imag)] for z in driven]}.")
    lines.extend([
        'Neutral time coefficients use SAME coupled H and clock, not a static reference subtraction. Initial, baseline, pulse, frozen-original-law probes and paired-neutral contrasts remain separate.',
        'Own incoming pulses exp(i*.23*P Qx/y_cos P), NOT full-W/pure/asymptotic photon preparations; initial signed field populations already[.97406964123,.012965179385,.012965179385], before any response.',
    ])
    for pol,name in enumerate(('xcos','ycos')):
        case=physics['cases']['N55_'+name];baseline=physics['cases']['N55_baseline'];other=2*(1-pol);label=f'Pi{other}Pi{other}';c=cx(case['field_means_covariance_receivers'][label]['coupled']);b=cx(baseline['field_means_covariance_receivers'][label]['coupled']);f=cx(case['field_means_covariance_receivers'][label]['frozen_own_generator']);fb=cx(baseline['field_means_covariance_receivers'][label]['frozen_own_generator']);energy=cx(case['full_H_energy'])-cx(baseline['full_H_energy']);initial=cx(case['field_means_covariance_receivers'][f'Pi{2*pol}']['coupled'])[0]
        lines.append(f"{name}: initial drivenPi={initial.real:.12g}, same-H kickwork={energy.real:.12g}; other-T Pi² pulse c2 coupled={float((c-b)[2].real):.12g}, FROZEN={float((f-fb)[2].real):.12g}. Nonzero frozen receiver covariance is not emission/purity.")
    lines.extend([
        'Frozen generator retains real body hopping/convection (density-rate L2 .0496302 addition/.0577736 removal), free scalar/EM; not all-physics-OFF. Original coupled-current probe on frozen state is NOT frozen-H own zero field-source derivative.',
        'Both signed J and QJ raw Gram weights/complex phases retained in actual source states. Example N55/xcos J diagonals~[1.5706473,1.5706462,1.5695352,1.5695352], QJ~[.912201,.912172,.914601,.914601]. Units current²/(coordinate*current)², NOT scattering probabilities.',
        'Omitting connected fullFock virtual contractions loses~1.23 of those rawJ weights. ALL4 signed directions and all Gram eigenvalues kept, including nulls; nonunitary insertion can reweight coherent total-C3 sectors.',
        'BEFORE-CUT ORDER FINDING / WRONG PHYSICAL CASE',
        'QJ and JQ are different written words but in THIS coordinate-only Peierls law Q commutes with uncutJ. Their proper before-cut coefficient difference is3.20e-16; do NOT demand a nonzero difference as evidence.',
        'On own N55/N53 rays, exact before-cut(QJ-JQ) image norm²<=2.75e-32. WRONG middle-P2 product manufactures norm² .0005444..0005554 in ALL4 signed channels; missing shell3 reaches the physical assertion.',
        'Shell3 is sufficient for Q times full Peierls J compressed to shell2; this does not establish general field-cut convergence or equality of every current law containing canonical Pi.',
        f"PHASE EXPORT REPAIR: wick1 accidentally took.real of sector GRAM matrices; complex characters/fullGrams/physical coefficients were preserved. analysis1 reconstructs complex sectors from OWN saved characters, no nine-case dynamics rerun. Lost imaginary part up to {analysis['phase_export_repair']['max_previously_unexported_sector_Gram_imaginary_part']:.12g}; repaired sum residual {analysis['phase_export_repair']['corrected_complex_sector_sum_to_full_Gram_residual']:.3g}.",
        'Own full-H energy/H²/variance include ALL cross terms; no sum-of-image-squares or positive Gram is interpreted as particle probability/mass.',
        f"EPOCHS: wick1 natural{exits['wick1']['natural_exit']}/{exits['wick1']['elapsed_seconds']:.6f}s; analysis1 natural{exits['analysis1']['natural_exit']}/{exits['analysis1']['elapsed_seconds']:.6f}s, source/frozen guards allTRUE. Raw own Slater/field/operators in wick1_raw.npz; original physics and corrected analysis source/input receipts separate.",
        f"Final selected diagnostics natural{language['natural_exit']}: {language['diagnostics'].get('summary')}; actual rootedLSP exit{language['rooted_LSP']['natural_exit']}. Exact residuals retained, not producer typing borrowed.",
        'LIMITS: own evidence is exact selected-order recoil/current response, NOT author actual6 t=.2 magnitudes, optional image catalogue or TDVP convergence. Author epsilon4.51e-5/energy drift5.35e-5 are not observable error bars.',
        'N53 already has ONE valence hole; P(holes>=2) is not two newly created pairs. n_site/N inclusive particle POVM is not excitation-position PVM.',
        'Supplied primitive spin/CAR/Clifford/scalar, oneq/finiteC3/75field, absent nonuniform longitudinal sector and no Maxwell magnetic term remain. No Noether-derived spin, freeCOM, photon pole/on-shell radiation, Lorentz/inertial mass, proton or Nature claim.',
        'Next material link: carry measured current/field recoil into the already-open genuinely local colour-flux/deformation/body construction with its own longitudinal and Noether/response closure. Exploration is not gated by this review.',
        'This full108 law/state is DISTINCT from Common native, body408, original47/cc6 and OLD48/CAR8; no borrowed state, result splice or plot collage. Original47 transport-first review continues independently.',
    ])
    assert len(lines)<=80,len(lines)
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    interface=dict(kind=verdict['kind'],source_capsule_sha256=frozen['capsule_sha256'],pure_source_snapshot_sha256=frozen['code_snapshot_sha256'],physics_result=P+'wick1_result.json',phase_complete_primary_result=P+'analysis1_result.json',raw_arrays=P+'wick1_raw.npz',synthesis=P+'synthesis.txt',natural_exits={name:value['natural_exit'] for name,value in exits.items()},acceptance='Exact full108 selected-order quantum recoil/current mechanism only; no finite-time magnitudes/solver/photon/particle approval.',field_and_source_semantics='Finite P-Q pulse. Raw source/Gram weights and complex sectors retained, matched neutral time-dependent. Before-cut QJ=JQ in this coordinate-only law, while middle-cut products spuriously differ.',capsule=P+'capsule.zip',capsule_receipt=P+'capsule_receipt.json',typing=verdict['typing'])
    (ROOT/(P+'interface.json')).write_text(json.dumps(interface,indent=2)+'\n')
    excluded={P+'capsule.zip',P+'manifest.json',P+'capsule_receipt.json'};files=sorted(path for path in ROOT.glob(P+'*') if path.is_file() and path.name not in excluded);manifest={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in files};(ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(P+'capsule.zip')
    with zipfile.ZipFile(capsule,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.name)
        archive.write(ROOT/(P+'manifest.json'),P+'manifest.json')
    final=dict(capsule=capsule.name,sha256=digest(capsule),bytes=capsule.stat().st_size,synthesis_lines=len(lines),physical_natural_exit=exits['wick1']['natural_exit'],analysis_natural_exit=exits['analysis1']['natural_exit'],final_typing_natural_exit=language['natural_exit'])
    (ROOT/(P+'capsule_receipt.json')).write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)

if __name__=='__main__':main()
