"""Completed physical candidate delivery. No producer replay or new cases."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
import zipfile
ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_composite_local_flux_binding_20261017_'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> Any:
    return json.loads((ROOT/(PREFIX+name)).read_text())


def main() -> None:
    actual_stamp, control_stamp = sys.argv[1:3]
    actual, control = load(actual_stamp+'_result.json'), load(control_stamp+'_result.json')
    exits = {stamp:load(stamp+'_exit.json') for stamp in [actual_stamp, control_stamp]}
    if any(row['returncode'] != 0 or not row['natural_exit'] for row in exits.values()):
        raise ValueError('Physical cases must be completed before delivery')
    typing = load('language_final.json')
    image = load(actual_stamp+'_canonical_source_image.json')
    ground, off = actual['ground_records'][0], control['ground_records'][0]
    rotation = actual['ground_block_joint_rotation']
    full_comparison = actual['finite_same_H_comparisons']['no_zero_pair_compact_support']
    off_comparison = control['finite_same_H_comparisons']['no_zero_pair_compact_support']
    final = actual['compact_dressed_source_trajectory'][-1]
    original = actual['compact_dressed_source_trajectory'][0]
    lines = [
        'NEW LOCAL NON-ABELIAN uud BINDING CANDIDATE: executed finite quantum variational construction, NOT a proton/QCD/Nature claim.',
        'Backward target: extended moving charge1/halfspin odd three-quark object. Selected mechanism: cooperative LOCAL trace-matrix clouds plus dynamical SU3 electric flux; actual virtual constituent hops transport it.',
        'Genuine3D regular tetrahedron,4 equivalent free-centre vertices/all6 hopping edges. No external one-body trap, rigid cluster projection, whole-trimer hop or exact COM/internal tensor factorization. Finite graph, not bulk translation/dispersion.',
        'Law FIRST: SU3 L2 matrix links/U_ab/electric Casimirs/left-right Gauss generators; shared72-mode CAR u,d,anti-u; independent quantum Phi_s=X_s I3 fields. Charges2/3,-1/3,-2/3, masses and primitive Pauli spins are explicit NEW law inputs, not Nature-derived.',
        'Units: hbar=a=omega_reference=1; Q in the input e-unit, energies in the chosen model units, primitive S2 in hbar2, current in input charge*a/model-time. Supplied mass6 is a rest COST, not a measured inertial/Lorentz mass.',
        'Approximation NEXT: covariant PW1+3+bar3 link projector (19states), actual epsilon/delta matrix elements and noncommuting generators retained. PUP is nonunitary; excluded6/8/higher representations are real debt, not replaced by a Z3 clock.',
        'Local field: omega bdagb+lambda X4-g Ntotal X plus edge gradients. Conditional2 vibronic states/load0..18 from bare32 oscillator basis; X2/X4 products BEFORE primitive cut. No global breather, imported clamped curve or fitted bare pair potential.',
        'Literal identical-u CAR/colour antisymmetry is preserved; not three artificially distinct flavours. Local spin-colour-singlet u/anti-u creation followed by ACTUAL quark/antiquark SU3 hopping generates mixed local matter+flux channels, not an inert singlet sea ladder.',
        'Heavy anti-u, pair vertex and spin-orbit hopping are explicit hypotheses; no d-antiquark band in this new regime. Ordinary H-word depth2 permits2 extra pairs, not a microscopic one-pair cap; higher virtual response remains outside the ansatz.',
        f'Fixed chosen parameters={actual["parameters"]}. Bare plaquette coefficient0 is the declared pure-electric strong-coupling regime; real Wilson-loop source directions and virtual non-Abelian constituent loops remain.',
        'H/current-generated compact/separated/Wilson sources resolved by physical net-flavour/anti-number channels; actual complex Gram/phase/source weights retained. This is a quantum relative-state variational approximation, not fullFock/full-H closure or a globalground theorem.',
        f'Exact streamed weak-FORM contraction: same original6,792,519 primitive keys/rank{actual["rank"]}; weak means matrix elements, NOT weakened H or weak coupling. Original coefficients/tiny values/channels preserved; no primitive-union sparse H allocation.',
        f'Native literal coefficient reconstruction on{actual["native_encoding"]["selected_primitive_dimension"]} diagnostic keys: maxerror={actual["native_encoding"]["max_literal_coefficient_error"]:.4g}. Diagnostic selection is not a state cut. Actual Gram error={actual["metric_error"]:.4g}.',
        f'All32 source/H Gauss maxnorm2={actual["first_H_max_all32_Gauss2"]:.5g}; invariant words, Wilson loops and number projections preserve local/global Gauss. Total colour singlet, no external coloured fragments or B-L atom.',
        'Completed local quantum-load layer: E3+E0-E2-E1=-1.8191048521 and E3+2E0-3E1=-2.9363457784 BEFORE edge gradients/hopping/sea mixing. Cooperative-cloud result, not alone full baryon binding or a breakup threshold.',
        f'Variational E_min={actual["eigenvalues"][0]:.12g}; SAME-H no-zero-pair-compact-support minimum={full_comparison["energy"]:.12g}, finite cost={full_comparison["cost"]:.9g}.',
        'That comparison includes pair+single, three-separated and ALL generated sea/cloud/colour channels; legal gauge-invariant states. It is a constrained finite relative criterion, NOT an asymptotic threshold, coloured isolated-single energy or pole/lifetime proof.',
        f'Three-distinct-position/zero-extra-pair SAME-H minimum={actual["finite_same_H_comparisons"]["three_separated_zero_pair"]["energy"]:.12g}, cost={actual["finite_same_H_comparisons"]["three_separated_zero_pair"]["cost"]:.9g}; this narrower comparison is explicitly separate.',
        f'Selected ground vector: Q={ground["total_charge"]:.12g}, P0pair={ground["zero_pair_norm"]:.9g}, Pcompact|0pair={ground["conditional_compact_probability"]:.9g}; conditional zero-pair relative RMS={math.sqrt(ground["zero_pair_weighted_pair_radius2"]/ground["zero_pair_norm"]):.9g}a, nonzero/extended.',
        f'Uniform lab charge density={ground["charge_density"]} does NOT establish binding. Actual relative correlations do: compact receiver weight={ground["zero_pair_compact_weight"]:.9g}; three-separated weight={ground["zero_pair_all_separated_weight"]:.9g}.',
        f'Quantum dressing: mean Phi={ground["quantum_phi_mean"]}; electric Casimir={ground["electric_Casimir"]:.9g}; Nanti={sum(ground["antiparticle_density"]):.9g}; P>=2 extra pairs={ground["at_least_two_sea_pairs"]:.9g}; mixed screening-pattern weight={ground["mixed_screening_pattern_weight"]:.9g}.',
        'The mixed-pattern diagnostic is not the entire non-singlet-sea probability. Sea counts are ABSOLUTE occupations, not a propagated-neutral subtraction or tagged-added-quark position PVM. No additional neutral case was run.',
        f'Lowest block dimension={actual["ground_block_degeneracy"]}; primitive spin2 complex readouts={actual["ground_block_primitive_spin2"]}. SOC does not conserve primitive S2; no AF J was inserted to choose spinhalf.',
        f'Actual JOINT tetrahedral position/spin/link rotation character={rotation["character"]}; block unitarity error={rotation["ground_block_unitarity_error"]:.4g}, R3+I error={rotation["triple_rotation_minus_identity_error"]:.4g}.',
        'This is the finite spinorial doublet (character1), not merely internal spin SO3 and not an exact continuum J theorem. Odd-CAR/primitive spin inputs remain assumptions, not a Noether derivation of spin/statistics.',
        f'Active mediator-OFF SAME SU3/PW19/CAR hopping/SOC construction: g=eta=kE=0, own reference/cloud source/ansatz; E={control["eigenvalues"][0]:.12g}, Pcompact|0pair={off["conditional_compact_probability"]:.9g}, relative RMS={math.sqrt(off["zero_pair_weighted_pair_radius2"]/off["zero_pair_norm"]):.9g}a.',
        f'OFF same-law-within-control noncompact cost={off_comparison["cost"]:.9g} (NOT0: finite kinetic/projection costs exist). OFF is a DIFFERENT-parameter diagnostic, not a fragmentation threshold or proof merely from E_full<E_OFF.',
        'The joint mediator mechanism survives this finite discriminator: markedly tighter relative state and retained/relocating compact packet with active hopping. No separate necessity attribution to g versus colour flux is claimed.',
        'Compact receiver C_s=Pi(Nanti=0)*Pi(all3 quarks at vertex s), identity on ALL spin/colour/flux/cloud states. It is a readout, not a Hamiltonian restriction, selected source profile or cluster-by-definition approximation.',
        'Raw R_s=<C_s>, R=sum_s R_s and P0=<Pi(Nanti=0)> use the normalized whole state; reported ground Pcompact|0pair=R/P0 is explicitly conditional. Raw transported weight is sum_{s!=initial0}R_s, NOT a tagged-added-quark probability.',
    ]
    for row, comparison in zip(actual['compact_dressed_source_trajectory'], control['compact_dressed_source_trajectory']):
        lines.append(f't={row["absolute_time"]:g}: compact zero-pair receiver full/OFF={row["zero_pair_compact_weight"]:.9g}/{comparison["zero_pair_compact_weight"]:.9g}; full compact-centre weights={row["compact_centre_weights"]}; owning J={row["owning_uniform_current"]}.')
    lines += [
        f'AtT={final["absolute_time"]:g}, compact weight on OTHER3 centres={sum(final["compact_centre_weights"][1:]):.9g}, versus clock0~0. Transport is from constituent hopping/virtual flux-cloud states; no imposed whole-trimer kinetic term.',
        f'That transported weight conditioned on a COMPACT outcome is {sum(final["compact_centre_weights"][1:])/final["zero_pair_compact_weight"]:.9g}; its denominator is R(T)={final["zero_pair_compact_weight"]:.9g}, distinct from normalization by P0 or whole-state norm.',
        f'Field feedback on that SAME state: Phi clock0={original["quantum_phi_mean"]}, Phi(T)={final["quantum_phi_mean"]}; time coordinates/complex phases/full source basis retained, not a collage of separately solved backgrounds.',
        'EM is an EXTERNAL probe only: J=-d_AH/contact=d_A2H from constituent Peierls phases and actual nonorthogonal charge-channel metric. Not dynamical EM/photon dressing, momentum=current, or a replacement whole-trimer current.',
        f'Canonical bare Bdag on prepared empty-sea field reference: full norm2={image["canonical_bare_full_norm2"]:.12g}; local2-field projection={actual["canonical_bare_source_local_projection_norm2"]:.9g}; Galerkin projection={actual["canonical_bare_source_Galerkin_projection_norm2"]:.9g}.',
        f'Missing bare-canonical complement = full norm minus retained Galerkin norm = {image["canonical_bare_full_norm2"]-actual["canonical_bare_source_Galerkin_projection_norm2"]:.12g}; it is retained as explicit exterior/ansatz debt, not renormalized away.',
        'Large missing bare-source strength is real ansatz debt, NOT absent physical strength, a complete emission probability or pole residue. Full finite primitive tensor image/coefficients frozen separately. DRESSED clock0 source is a DISTINCT preparation.',
        'Auxiliary compact enhancement in rawJSON uses an inclusive-marginal iid reference, not a universal fermionic connected cumulant or independent binding proof. Core conclusions above use literal relative receiver weights/RMS and same-H comparisons.',
        f'Numerical scope: raw H antihermiticity={actual["raw_H_antihermiticity"]:.5g}; raw matrix saved, tiny Hermitian accumulation average disclosed. Bare-top indicator={ground["local_basis_top_indicator"]:.5g}, NOT a field/word/PW19/volume convergence bound.',
        'Adverse history: control1 SIGKILL9, OOM only suspected; last OBSERVED row2,770,000/2,774,784. Allocation-boundary location is inference, not established killer evidence. No killer-confirmation replay.',
        'actual1 deliberately harness-cancelled; valid local/Gauss PARTIAL layers stay distinct from binding. actual2 natural1 in Numba index-type compilation BEFORE physical basis; fixed signed int64 representation, no new physics/cap.',
        f'Natural physical exits: {actual_stamp}=0/{exits[actual_stamp]["elapsed_seconds"]:.3f}s, {control_stamp}=0/{exits[control_stamp]["elapsed_seconds"]:.3f}s; peakKiB={actual["peak_KiB"]}/{control["peak_KiB"]}.',
        f'Final selected typing exit={typing["natural_exit"]}, summary={typing["diagnostics"]["summary"]}; rooted actual LSP exit={typing["rooted_LSP"]["natural_exit"]}. Diagnostics are source evidence, not physical approval.',
        'No full-H/globalground/domain/continuum/word-order/link/field convergence, asymptotic confinement/pole/lifetime, Lorentz/inertial mass, Nature antiparticle/proton, measured parameter fit or same-law electron/photon join claim.',
        'Old recoil capsule e3e1c4df/source files remain immutable. Centre-only cube proposals stay UNEXECUTED controls, not rich suppliers. No old/nativeCommon/47 state, solved background, public/plan/graph update or clamped energy curve is imported.',
    ]
    if len(lines) > 80:
        raise ValueError('Physical synthesis exceeds80 lines')
    (ROOT/(PREFIX+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    interface = dict(kind='COMPLETED_FINITE_NONABELIAN_LOCAL_COMPOSITE_CANDIDATE', root=str(ROOT),
        law=PREFIX+'model.py', primitive_coefficients=[PREFIX+stamp+'_frozen_coefficients.npz' for stamp in [actual_stamp, control_stamp]],
        current_entrypoint=PREFIX+'streamed.py', capture=PREFIX+'stream_capture.py', native=PREFIX+'native.py',
        source_generator=PREFIX+'execute.py:source_words ONLY; old full-union producer is preserved historical source, not the current entrypoint',
        arrays='keys_lo/high uint64 + basis_indptr/indices/values define EXACT original canonical primitive embedding; source_checkpoint/spectrum/phase_and_source preserve complex metrics, operator matrices and coherent states.',
        selected_cases=[actual_stamp, control_stamp], scope='Finite SU3/local quantum field/sea operator-orbit candidate. No scalar gauge replacement, new state cap, continuum/proton approval or hidden source normalization.',
        source_sha256={path.name:sha(path) for path in sorted(ROOT.glob(PREFIX+'*.py'))})
    (ROOT/(PREFIX+'interface.json')).write_text(json.dumps(interface, indent=2)+'\n')
    capsule = ROOT/(PREFIX+'delivery_capsule.zip'); manifest_path = ROOT/(PREFIX+'manifest.json'); receipt_path = ROOT/(PREFIX+'delivery_receipt.json')
    files = [path for path in sorted(ROOT.glob(PREFIX+'*')) if path.is_file() and path not in [capsule, manifest_path, receipt_path]]
    manifest = dict(kind='IMMUTABLE_SOURCE_INPUT_RUNTIME_STATE_METRIC_CAPSULE', files={path.name:dict(bytes=path.stat().st_size, sha256=sha(path)) for path in files})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(capsule, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in files+[manifest_path]:
            archive.write(path, path.name)
    receipt = dict(capsule=capsule.name, sha256=sha(capsule), bytes=capsule.stat().st_size, manifest_sha256=sha(manifest_path), synthesis_lines=len(lines))
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n'); print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
