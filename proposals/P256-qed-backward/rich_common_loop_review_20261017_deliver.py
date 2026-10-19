"""Scoped independent quantum-loop verdict; no photon/ground approval replay."""
from pathlib import Path
import hashlib,json,sys,zipfile

ROOT=Path(__file__).resolve().parent;P='rich_common_loop_review_20261017_'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    attempt=sys.argv[1];stem=P+attempt;result=json.loads((ROOT/(stem+'_result.json')).read_text());receipt=json.loads((ROOT/(stem+'_exit.json')).read_text());language=json.loads((ROOT/(P+'language_final.json')).read_text())
    failures={name:json.loads((ROOT/(P+name+'_assessment.json')).read_text()) for name in ('literal1','literal2')}
    verdict=dict(kind='BOUNDED_INDEPENDENT_POSITIVE_QUANTUM_LOOP_SURVIVAL_WITH_SCOPE',result=result,repair_epoch_exit=receipt,failed_epoch_scopes=failures,typing=dict(natural_exit=language['natural_exit'],summary=language['diagnostics'].get('summary')),interpretation='Genuine literal nonbacktracking charged-quark and white hopping-word actions survive on OWN Gauss-valid native quantum packets with every field projector. Net-only shortcut fails on a deliberate boundary counterpacket. Raw energy-power norms and image-weighted retention are not state probabilities, effective induced energies, persistent currents or photons.',author_provenance_only='D75C reports its own13949s loop-rich reference/response, purity.441737 and broad24-window results. No author vector re-contraction or independent ground search performed here. Those author claims are not replaced by our fresh-packet action proof.',next_construction='Carry literal charged CAR paths, quantum local Phi/R/colour, field-projector/virtual support and current normalization into the already-chosen connected spatial256CAR/all96hops/all81cycles response construction. Exploration is not gated by this review.')
    (ROOT/(P+'verdict.json')).write_text(json.dumps(verdict,indent=2)+'\n')
    lines=[
        'INDEPENDENT LOOP-RICH QUANTUM WORD REVIEW — positive finite mechanism, not photons/ground closure.',
        'DISTINCT loop-rich body4407 tensorPhi7 tensor85field =2,622,165 dimensions; NOT first-C1 pair/e740 or old47 state/law splice.',
        'Frozen native capsule eeae0c720a8ac112a7f6b9931da1a8cf934371edf649ecee8eb88e679a5cec36; only operator definitions/matrices used, no author solved/evolved vectors.',
        'Native body is a fundamental selected proper-J0/CAR64/Phi7/R10/PW19 law, NOT unprojected all-CAR/all-virtual closure.',
        'PhysicalQ=Nint/3, transverseG=m/3; E_Q=(D^T Lplus Nint+t Gt^-1m)/3. Real Gauss-reduced edge flow, not independent integer link rotors.',
        'All85 signed |m|<=6 even(my+mz) fluxes remain; only2of21 charge-cycle profiles. Other cover sectors and lab spatial-field Noether remain unpaid.',
        f"Own Gauss basis residual {result['Gauss_integer_basis_residual']:.3g}. Fresh packets use actual native orthonormal matter/Phi/R/colour coordinates, not source-matrix copies or a clamped phase.",
        'OWN INPUTS / EXACT APPLICATION',
    ]
    for name,stats in result['packet_stats'].items():
        lines.append(f"{name}: norm²={stats['norm2']:.12g}, fieldcoordinates={stats['field_coordinates']}, weights={stats['field_weights']}, boundary={stats['field_boundary_weight']:.6g}, purity={stats['reduced_transverse_purity']:.12g}; physical siteQ={stats['physical_Q_site']}.")
    lines.extend([
        'Input entanglement was CONSTRUCTED (purity.52); not claimed dynamically generated or an independently solved vacuum. Reducedm0 is not full electricfield0 because longitudinal Gauss follows quantum charge.',
        'Triangle chronology01->12->20 and square01->12->23->30 are literal nonbacktracking words. Every native one-hop c0(vector)+cPhi*Phi(scalar), CAR/R/colour coefficient and intermediate fieldP is applied.',
        'Independent sparse field-support propagation agrees with native full85flux tensor action. Uncut-FLUX reference uses the SAME selected-body/Phi/R/colour word; it does NOT remove body projection or add missing full-CAR virtuals.',
        'ACTUAL OWN CIRCUITS: rawuncut / every-P norm²; energy units; image-retention versus bare input-state survival are separate.',
    ])
    for packet,rows in result['circuits'].items():
        for row in rows:
            lines.append(f"{packet} {row['root']} {row['path']}: raw={row['raw_selected_body_uncut_flux_word_norm2']:.12g}, actual={row['actual_every_intermediate_P_norm2']:.12g} {row['norm_squared_units']}, image-ret={row['operator_image_retention_fraction']:.12g}, input-survival={row['bare_input_state_survival_weight']:.12g}; native/independent error={row['independent_literal_vector_relative_error']:.3g}.")
    lines.append('COUNTERCASES / SEMANTICS')
    for row in result['circuits']['entangled_boundary']:
        lines.append(f"{row['root']} {row['path']}: WRONG net-only norm²={row['net_shift_only_norm2_NOT_actual_word']:.12g}, relativevectorerror={row['WRONG_net_only_relative_vector_error']:.12g}; meanLoop={row['complex_loop_mean']}, HermitianRe/Im variances={row['Hermitian_Re_loop_variance']:.12g}/{row['Hermitian_Im_loop_variance']:.12g}, Robertsonbound={row['Robertson_standard_deviation_product_lower_bound']:.12g}.")
    lines.extend([
        'Boundary(6,-4) square: final netshift can lie inside while FIRST primitive step lies outside. Square net-only shortcut is therefore wrong; triangle is an explicit equality control in this geometry, not a universal shortcut ban.',
        'Bare input survival.6 does not equal energy-weighted image retention: high-Phi boundary component can dominate raw word strength. Cut loss is state/operator-specific, NOT an observable error bar or particle probability.',
        'Triangle raw norm² has energy^6 units and square energy^8. They are not bounded by1, not Born probabilities, not induced magnetic energies; no virtual energy denominators/resolvent were evaluated.',
        'Nonzero action with zero loop mean demonstrates availability of a virtual quantum word, NOT persistent circulating order. Re/Im Hermitian quadrature variances require BOTH L and Ldagger; raw normL² is not a generic uncertainty.',
    ])
    for car in result['independent_CAR64_exchange_fixtures']:
        lines.append(f"Independent CAR64 {car['root']} exchange: amplitude {car['exact_amplitude']}, frozenprimitive {car['frozen_primitive_amplitude']}, WRONG signless {car['WRONG_signless_amplitude']}; {car['chronology']}. Same determinant returns; not a substitute for native primary path ordering.")
    lines.extend([
        'EXECUTION EPOCHS / PHYSICAL SCOPE',
        'literal1 natural1: emitted NumPy-int metadata JSON failed after first native circuit; no physics failure. Fixed exact emitted coordinate types only.',
        'literal2 natural1: ALL8 own native/dictionary loop, projection, adjoint/quadrature and square-shortcut assertions completed/saved; later separate spin-flip CAR fixture had an invalid universal-sign expectation.',
        f"{attempt} repair natural exit {receipt['natural_exit']}: corrected same-determinant CAR exchange and Gauss receipt; unaffected OWN8 loop layers reused under original hashes, NOT re-labelled as a natural0 full-loop rerun. Original source/input/exit/raw failures remain immutable.",
        f"Own quantum raw arrays: {result['own_quantum_word_raw_file']}, SHA {result['own_quantum_word_raw_sha256']}; repaired epoch/source/input receipts separate. All source/frozen/input flags retained.",
        'Positive finite result: charged quark AND charged-white nonbacktracking circuits survive the actual quantum field with every intermediateP, rather than only a clamped Wilson number.',
        'This review does NOT reproduce/approve the author13949s ground solve, purity.4417, full-H moments or broad24 supported Ritz nodes. Optional author-vector recontraction was NOT used.',
        'Positive Gram, Ritz weight, phase-POVM contrast or clamped curvature is not a photon pole/propagation/continuum or full-H closure proof. Historical source-selected spaces are not assumed nested.',
        'Meaningful next construction: retain these literal rich charged paths/projectors/support diagnostics in connected256CAR/localPhi-R-colour/all96hops/all81cycles response; exploration proceeds without this being a permission gate.',
        f"Final selected diagnostics exit {language['natural_exit']}, {language['diagnostics'].get('summary')}; actual rooted LSP exit {language['rooted_LSP']['natural_exit']}. Exact residual diagnostics retained.",
    ])
    assert len(lines)<=80,len(lines)
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    exclude={P+'capsule.zip',P+'manifest.json',P+'capsule_receipt.json'}
    files=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in exclude)
    manifest={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in files};(ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(P+'capsule.zip')
    with zipfile.ZipFile(capsule,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        z.write(ROOT/(P+'manifest.json'),P+'manifest.json')
    final=dict(capsule=capsule.name,sha256=digest(capsule),bytes=capsule.stat().st_size,synthesis_lines=len(lines),repaired_stage_natural_exit=receipt['natural_exit'],original_quantum_word_epoch=result['own_quantum_word_epoch'],original_quantum_word_exit=1,typing_natural_exit=language['natural_exit'],meaning='Complete scientific layers with honest technical failure/repair provenance; no author state primary, no ground/photon approval replay.')
    (ROOT/(P+'capsule_receipt.json')).write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)

if __name__=='__main__':main()
