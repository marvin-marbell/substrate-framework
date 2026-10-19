"""Assemble bounded physical synthesis, raw-array manifest and exact capsules."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_review_wave36_20261016_'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    initial=json.loads((ROOT/(P+'result.json')).read_text());endpoint=json.loads((ROOT/(P+'endpoint_result.json')).read_text());metric=json.loads((ROOT/(P+'metric_result.json')).read_text());language=json.loads((ROOT/(P+'language_final.json')).read_text())
    source=initial['source'];rich=endpoint['projections']['rich_Phi_R_U'];extra=endpoint['projections']['material_Phi3_extension'];core=endpoint['unchanged_source_quark_core_ALL_fields']
    exits={label:json.loads((ROOT/(P+label+'_exit.json')).read_text()) for label in ('actual1','metric1','endpoint2')}
    lines=[
        'Review36: actual quantum-field dressed source/EOM/current mechanism; bounded independent verdict.',
        'Only rich_review_wave36_20261016_* changed. Active model/dressing owner CommonColourLeptonRoot supplied frozen interfaces.',
        'Shared input is its actual one-hop quantum Omega, NOT an independently solved full-law ground.',
        'Root /home/administrator/substrate-222/proposals/P256-qed-backward; baseline interpreter /home/administrator/substrate-framework/.venv/bin/python.',
        'Original neutral_basis/coupling/states filenames, every dependency SHA and exact command are pinned in each *_input.json/source_capsule.zip.',
        'Independent raw review sums full-H component Grams: charged residual²34.47743587635, neutral34.86927772549;',
        'RMS5.87174896231 and5.90502139924. No subtraction, lifetime/mass, stationary vacuum or full-H closure inferred.',
        'Bare unmasked EOM [H,Bdag]Omega independently gives norm²2.98059545912, G=.855952013215, mean frequency.284946248439;',
        'source-orthogonal RMS1.84418135938 is a different quantity from the full-H source variance.',
        'NEW selected quantum source Phi Bdag includes [Hphi,Phi]Bdag BEFORE contraction, not just Phi[H,Bdag].',
        f'Its raw source G={source["raw_source_G"][1][1]:.12g}; full EOM norm²={source["raw_EOM_norms2"][1]:.12g}.',
        f'Actual momentum derivative norm²={source["Phi_derivative_norm2"]:.12g}; independently dropped derivative fails with that exact nonzero error.',
        f'Correct product-rule difference norm²={source["correct_product_derivative_error_norm2"]:.3g}; independent -iP/(metric*volume) matrix error={metric["Hphi_Phi_vs_primitive_minus_iP_error"]:.3g}.',
        'Selected Phi commutes exactly with Hrot, SU3 electric, plaquette and hopping coordinate factors; those zero derivatives are NOT field freezing.',
        'All actual Phi7/R10/PW19, occupied white/quark sea and generated local CAR components remain in the EOM.',
        f'Same primitive fractional-charge chord Peierls dressed-current norm²={source["current_norms2"][1]:.12g}; contact norm²={source["contact_norms2"][1]:.12g}.',
        f'Independent finite-theta derivative errors norm²={source["dressed_current_finite_difference_error_norm2"]:.3g}/{source["dressed_contact_finite_difference_error_norm2"]:.3g}.',
        'These are actual source/current/contact responses to dimensionless material chord theta, not lab3D photons or a baryon-hop charge substitution.',
        f'Original instant has Gdot norm{source["Gdot_norm"]:.3g}: nonstationary full sea need not change this chosen metric at every instant.',
        'NEW explicit unitary exp(i*.37*Phi)Omega supplies a momentum-displaced quantum hot sea, not a replacement ground.',
        f'Actual Gdot=i(K-Kdag) norm{metric["Gdot_norm"]:.12g}, offdiag{metric["Gdot_real"][0][1]:.12g}; independent scalar-product rule error{metric["independent_product_derivative_error"]:.3g}.',
        f'Actual sqrtG Sylvester connection norm{metric["actual_sqrtG_connection_norm"]:.12g}; correct moving-generator Hermiticity defect{metric["correct_moving_generator_nonhermiticity"]:.3g}.',
        f'Dropping connection gives defect{metric["wrong_no_connection_nonhermiticity"]:.12g}; fixed-G Hermitization discards metric flux{metric["wrong_fixed_G_Hermitization_discarded_metric_flux"]:.12g}.',
        'Thus the independent wrong derivative AND bad moving-metric cases reach and fail this physical consumer; no fake fixed-G unitary motion.',
        f'NEW owner-aligned field endpoints B,PhiB,(sigma.Rd)B,TrU B capture{rich["raw_captured"]:.12g} of raw full K3{endpoint["full_K3_norm2"]:.12g};',
        f'raw rest{rich["raw_rest"]:.12g}; FULL captured fraction{rich["captured_fraction_of_FULL_K3"]:.12g}, metric rank20/20.',
        f'Unchanged-source-quark-core ALL-fields subset{core:.12g}; rich endpoints capture{rich["raw_captured"]/core:.12g} OF THAT SUBSET, not of whole baryon.',
        f'Bare Omega-return is only1.04096e-22; quantum field dressing therefore materially opens that endpoint channel.',
        f'Phi³ extension capture{extra["raw_captured"]:.12g}, raw rest{extra["raw_rest"]:.12g}, core-subset capture{extra["raw_captured"]/core:.12g}.',
        f'Changed-quark-core/sea raw weight{endpoint["changed_quark_core_raw_weight"]:.12g} remains ALLOWED, not rejected and not automatically a baryon.',
        'Actual1 endpoint-only result used transposed creation index/ReTrU; preserved but SUPERSEDED by endpoint2 sigma_(mu,m)/complexTrU.',
        'Actual1 Phi EOM/current results are unaffected; exact historical bytes remain in its capsule, never relabelled.',
        'Predeclared d=geometry.axes[0] tree01. K3 uses actual C0-conditioned source, not unmasked EOM source; their strengths differ.',
        'K3 squared image has energy^6 units and t^6/36 coefficient; not a finite-time probability/mobility/binding/lifetime.',
        'Constructive next source: actual outgoing constituent-replacement/sea-hole monomials from unmasked [H,Bdag], plus Pphi Bdag;',
        'keep their quantum Phi/R/Wilson coefficients and CAR signs. Higher scalar powers alone cannot reach changed-quark-core weight439.1932.',
        'R/U probes here are ENDPOINTS ONLY. Their EOM must retain Hrot/SU3electric/Noether and finite-cut shared-field commutators; owner retains that model work.',
        'Chosen finite J=L_left+S_CAR gives halfspin for B/PhiB; this is not derivation of primitive spin from a field-only physical Noether origin.',
        'No rigid core, Omega return, count mask, two-level truncation or Dirac proof used as particle definition or permission gate.',
        'Old tetra_poles1 and correlated_onehop_poles1 explicitly CANCELLED: SIGTERM15/rc-15, natural_exit=false; partial checkpoints preserved, never restarted.',
    ]
    for label,record in exits.items():lines.append(f'Own {label}: exact child rc={record["returncode"]}, natural={record["natural_exit"]}, capture_seconds={record["elapsed_seconds"]:.6f}, child_RSS_KiB={record["child_peak_RSS_KiB"]}, source/input unchanged={record["source_input_unchanged"]}.')
    summary=language['diagnostics']['summary'];lines.append(f'Final selected Python diagnostics: exit{language["diagnostic_exit"]}, {summary.get("errorCount")} errors/{summary.get("warningCount")} warnings; rooted LSP exit{language["rooted_LSP"]["natural_exit"]}; AST receipts retained. No all-clean claim.')
    lines.append('Raw arrays are separate *_raw.npz; all source/input/exit bytes and this <=80-line synthesis are carried by delivery_capsule.zip and manifest.json.')
    if len(lines)>80:raise ValueError('Physical synthesis exceeded requested line bound')
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    excluded={P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'}
    files=sorted(path for path in ROOT.glob(P+'*') if path.is_file() and path.name not in excluded)
    manifest=dict(kind='REVIEW36_EXACT_CAPSULE_AND_SEPARATE_RAW_ARRAYS',governing_plan='issue222-rich-backward-candidates-20261014',revision='9391f50740632658dd89f414a8e25e6d510fce4416ca8ddc054f56229fa59f47',files={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in files},scientific_exits={name:dict(returncode=r['returncode'],natural_exit=r['natural_exit'],source_input_unchanged=r['source_input_unchanged']) for name,r in exits.items()},supersession=endpoint['supersession'],synthesis_lines=len(lines))
    manifest_path=ROOT/(P+'manifest.json');manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[manifest_path]:z.write(path,path.name)
    receipt=dict(capsule=capsule.name,sha256=digest(capsule),bytes=capsule.stat().st_size,synthesis=P+'synthesis.txt',synthesis_lines=len(lines))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
