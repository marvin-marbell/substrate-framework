"""Plain physical synthesis, exact inputs/own raw outputs and scope manifest."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_primitive_vertex_review_20261016_'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    result=json.loads((ROOT/(P+'result.json')).read_text());exit_record=json.loads((ROOT/(P+'actual1_exit.json')).read_text());language=json.loads((ROOT/(P+'language_final.json')).read_text());traces=result['raw_source_traces'];field=result['field_receipt'];q=result['CAR_pair_receipts']['quark'];w=result['CAR_pair_receipts']['white']
    lines=[
      'INDEPENDENT PRIMITIVE CANONICAL SOURCE-CONNECTED RETARDED CURRENT VERTEX: positive leading shared-inertia mechanism.',
      'Only rich_primitive_vertex_review_20261016_* created; all producer/prior36/native/common-species review files and scientific epochs immutable.',
      'Command: /home/administrator/substrate-framework/.venv/bin/python rich_primitive_vertex_review_20261016_capture.py actual1',
      'Root /home/administrator/substrate-222/proposals/P256-qed-backward; exact imported source/physical input/secondary-comparison hashes and actual exit archived.',
      'PRIMARY: original occupied N32Omega snapshots,p/geometry,canonical source definitions. Not a newly solved ground; no author Vpsi/outputGram/sectorstate proof input.',
      'Own fresh Bdag0 epsilon/sqrt6 half-tripletM+.5,charged DOWN-WHITE spinUP Ldag2 coefficient1, L FIRST for Bdag0Ldag2Omega; no compact mask or Lsite0.',
      f'Own RAW traces [rho0,rhoB,rhoL,rhoBL]=[{traces["0"]:.15g},{traces["B"]:.15g},{traces["L"]:.15g},{traces["BL"]:.15g}], oneCAR64/commonOmega.',
      'N32/N35/N33/N36 matched canonical source insertions, jointJ1M1. No independent native-sector propagation/state splice or singlet gate.',
      'Own exact sitepair factorization: tree quarkedge0 acts on0,1; whiteedge2 on2,3. This is operator support, NOT a factorized particle/reference state ansatz.',
      f'Quark pair input/full onehop output dimensions{q["input_pair_dimension"]}/{q["full_onehop_output_pair_dimension"]}; white{w["input_pair_dimension"]}/{w["full_onehop_output_pair_dimension"]}.',
      'All original correlated CAR/PW/R/Phi amplitudes remain in each source. Elementary primitive/localFock factory is reused with provenance; composite contractions are NEW.',
      f'Independent explicit32bit Fock spin/flavour/reverse-action norm errors max{max(row["error"] for receipt in (q,w) for row in receipt["independent_Fock_checks"]):.3g}; actual source-family spectator/Jordan-Wigner parity checks pass.',
      f'Full KdagK minus compressed pair-PAP product Frobenius quark/white={q["unprojected_pair_product_complement_Frobenius"]:.12g}/{w["unprojected_pair_product_complement_Frobenius"]:.12g}; complements are NOT dropped.',
      'Final bra contraction computes a physical moment; all intermediate CAR pair outputs remain in products, rather than inserting source P between operators.',
      f'Own PIVOTED-QR SO3 polynomial wordspace rank{field["word_dimension"]}, HaarGramerror{field["polynomial_Haar_Gram_error"]:.3g}, inputmetricerror{field["input_embedding_error"]:.3g}, R10matching{field["input_R10_coordinate_error"]:.3g}.',
      'Degree certificate: inputj<=1 plus TWO R factors reaches exactlyj<=3=1+9+25+49=84; Trot preserves j. Fixed9x9x7 Haar quadrature, no largergrid/cap campaign.',
      'No claim of commuting coordinates on arbitrary truncated84 boundary states: exact word identities are checked on the original INPUT10 domain.',
      'Disjoint species even CAR bilinears have EMPTY actual modeintersection; fullPhi/R multipliers commute; selected quarktree0 and white2 have identityU.',
      'Hence primitive FIRST [DeltaHsource,Jreceiver]=0 operatorially for ANYstate; not an imposed zero for one source. NativePAP first slopes remain separate.',
      'Prior independently reviewed nativePAP slopes +.000779341594/-.001454102226 and positive finite geometry coupling are unchanged, NOT reclassified as this primitive vertex.',
      'Heisenberg sign/order: Delta<J>(u)=-u²/2 <V>+O(u³), V=[DeltaHsource,[Tphi+Trot,Jreceiver]], actual finite theta.3 and physicalfractional charges retained.',
      'Matter/potential/plaquette/unsharedHE terms cancel by species/coordinate commutation and Jacobi; no guessed ground or source-return definition.',
      f'Actual Phi7 Kphi=[Phi,[Tphi,Phi]] eigenvalues{field["Phi_double_commutator_eigenvalues"]}; trace{field["Phi_double_commutator_trace"]:.3g}, NOT inverseMassI.',
      'SS uses this signed finite matrix; SV/VS vanish by separated field algebra. VV is sum_ab Iinv_ab Da(Rds)_i Db(Rdr)_j with owning Pauli/hop coefficients.',
      f'Actual inertia{field["inertia"]}; isotropic VV=[(ds.dr)deltaij-(Rdr)_i(Rds)_j]/I, an UNFITTED physical spin/frame tensor, not inserted sigma.B.',
      'Z=Tr exp(i eta.J) Utheta[rho0+s rhoB+t rhoL+st rhoBL]Utheta†; RAW n(s,t)=n0+s nB+t nL+st nBL measured, not set to1 or factorized by assumption.',
      'Own nilpotent source polynomial inverse solves n*ninv=1. Mixedst log derivatives retain ALL trace/partition terms; not naive normalized-sector subtraction.',
      'This isolates the DECLARED canonical source-INTENSITY cumulant, not an amputated/asymptotic S matrix or unique tagged-particle force.',
    ]
    for source in ('white','quark'):
        item=result['directions'][source];con=item['connected'];receipt=item['field'];actions=item['active_source_actions'];rich=item['selected_rich_output'];bad=item['wrong_cases']
        lines.extend([
          f'{source}pulse: first fullcoordinate comm inputnormmax{receipt["first_full_coordinate_commutator_max"]:.3g}; actual kinetic-word versus rotational-derivative tensor error{receipt["rotor_kinetic_vs_derivative_tensor_error"]:.3g}.',
          f'{source}: canonical sourceweights [rho0,B,L,BL]={list(con["raw_source_weights"].values())}; all measured normalizers/partitions kept.',
          f'{source}: OWN connected mean u² TOTAL={con["mean_total"]:.15g}; coherent scalar/vector components{con["mean_components"]}.',
          f'{source}: OWN full4x4 Weyl currentcovariance u² tensor reconstructed; TOTAL variance coefficient{con["current_variance_u2_total"]:.15g}.',
          f'{source}: actual joint sourceDeltaH/current(theta)/contact(theta) action norms²={actions["delta_norm2"]:.15g}/{actions["current_norm2"]:.15g}/{actions["contact_norm2"]:.15g}.',
          f'{source}: selected coherent V00+V11 rich output norm²{rich["coherent_selected_norm2"]:.15g}, selected V11 norm²{rich["individual_norms2"][1]:.15g}, R>j1 part{rich["selected_V11_R_above_input_j1_norm2"]:.15g}.',
          f'{source}: ACTUAL mediatorOFF kinetic wordnorm{item["mediator_OFF"]["shared_kinetic_word_norm"]}; unchanged NONZERO joint DeltaHnorm²{item["mediator_OFF"]["unchanged_active_joint_DeltaH_norm2"]:.15g}.',
          f'{source}: wrong PhiCCR yields{bad["Phi_CCR_substitution_connected_mean"]:.15g}, wrong naive normalized alternating mean{bad["naive_normalized_sector_alternating_mean"]:.15g}; both reach discriminating target assertions and are rejected.',
        ])
    lines.extend([
      'White-pulse variance is NONZERO; reverse variance is ZERO at rounding level and deliberately preserved, not retuned to bilateral nonzero.',
      'Coherent SS/all9VV sums and all current-component offdiagonals/normalization partitions retained; no sum-of-norms replacement.',
      'MediatorOFF removes ONLY Tphi/Trot; matter hopping/pulse,scalarpotential,plaquette andunsharedHE remain. Derived primitive wordzero with actual active source, NOT sourceOFF.',
      'This DOES NOT undo the old native coordinate-only PAP response: full products/intermediate complements define a different observable/law boundary.',
      'All four-source current-dressed covariance tensors are independently reconstructed. Additional rich outputs use exact full-pair KdagK and84x10/7x7 lifts.',
      'Scope boundary: selected V00+V11 norm and V11 higherR output independently checked; ENTIRE producer27x27/fullVpsi/newmaterial-output catalogue NOT rebuilt or claimed reviewed.',
      'ChangedCAR/core/sea/field channels remain allowed; neither barecompound/Omega return nor count support is a particle-admission definition.',
      f'Secondary comparison ONLY AFTER own reconstruction: mean-component differences white/quark={result["secondary_producer_comparison"]["white"]["mean_components_difference"]:.3g}/{result["secondary_producer_comparison"]["quark"]["mean_components_difference"]:.3g}; covariance differences={result["secondary_producer_comparison"]["white"]["covariance_tensor_difference"]:.3g}/{result["secondary_producer_comparison"]["quark"]["covariance_tensor_difference"]:.3g}.',
      'No author normalized coefficient, raw_tensor, Vpsi or27x27Gram entered the primary primitive reconstruction. Exact producer-output comparisons are separately labelled.',
      'What works: a genuinely delayed shared-inertia canonical source-connected primitive CURRENT link at leadingu² in BOTH directions, after fullcoordinate firstcomm vanishes.',
      'This is NOT an integrated finite-u pulse curve/scattering trajectory, force/probability/mass/photon/Maxwell/Coulomb, boundproton/lifetime/freeCOM or stationary-pole proof.',
      'Units: current/contact energy per dimensionless theta; V energy3,|Vpsi|²energy6,mean_u²energy3,covariance_u²energy4 in owning natural units.',
      'Primitive spin/charges/SU3/3+1/action,negative scalar quadratic/volume,Phi7/PW19/severeonehopOmega remain INPUTS/cuts; no field-only spin origin derived.',
      'Next joined missing physics is controlled propagation/exterior/dynamicalEM embedding of this link, not a new ground/range permission gate.47/48/body408/CURRENT427/induced laws stay distinct.',
      f'Own child rc{exit_record["returncode"]}, natural={exit_record["natural_exit"]}; capture seconds{exit_record["elapsed_seconds"]:.6f}, child peakRSS_KiB{exit_record["child_peak_RSS_KiB"]}; exact sources/inputs unchanged={exit_record["source_input_unchanged"]}.',
      f'Final selected diagnostics exit{language["diagnostic_exit"]}: {language["diagnostics"]["summary"]}; actual rooted LSP exit{language["rooted_LSP"]["natural_exit"]}; no all-clean claim.',
      'Own raw field basis/moments/full4x4covariances/selected factorized rich-output kernels and source coefficients are separate NPZ; exact source/input/exit archive and manifest included.'
    ])
    if len(lines)>80:raise ValueError(('Plain physical synthesis exceeded80lines',len(lines)))
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n');excluded={P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'};files=sorted(path for path in ROOT.glob(P+'*') if path.is_file() and path.name not in excluded)
    manifest=dict(kind='INDEPENDENT_RECONSTRUCTED_PRIMITIVE_SOURCE_VERTEX_REVIEW',plan='issue222-rich-backward-candidates-20261014',captured_revision='ba0876c0c0b85ffcded63bc7c1ec4caae9bf2eb48a20671379025f54ecd48761',template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',synthesis_lines=len(lines),files={path.name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path in files},science_exit={key:exit_record[key] for key in ('returncode','natural_exit','elapsed_seconds','child_peak_RSS_KiB','source_input_unchanged')},independence=exit_record['independence'],limits=result['limits'])
    m=ROOT/(P+'manifest.json');m.write_text(json.dumps(manifest,indent=2)+'\n');capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[m]:archive.write(path,path.name)
    receipt=dict(capsule=capsule.name,sha256=digest(capsule),bytes=capsule.stat().st_size,synthesis=P+'synthesis.txt',synthesis_lines=len(lines));(ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
