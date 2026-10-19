"""Once-stable selected diagnostics, exact binding, synthesis and replay capsule."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED
import numpy as np
from scipy import sparse
from rich_review_wave12_20261015_diagnose import language_server
from rich_spatial_colour_em_20261015_snapshot import factor

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_review_wave24_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(suffix, value):
    (ROOT/(PREFIX+suffix)).write_text(json.dumps(value, indent=2)+'\n')


def hermiticity(request):
    """Tensor-factor Hermiticity/skew norm witnesses, without a dense H."""
    metadata = json.loads((ROOT/request['state_json']).read_text())
    data = np.load(ROOT/request['state_npz'])
    scalar = dict(grid=data['scalar_grid'], H=data['scalar_H'])
    h = factor(data, 'baryon_H', metadata['sectors']['baryon']['matter_dimension'], scalar, 70)
    remaining = set(range(len(h.terms)))
    singles = []
    pairs = []
    scalar_skew = float(np.linalg.norm(scalar['H']-scalar['H'].conj().T))
    skew_bound = scalar_skew
    while remaining:
        index = min(remaining)
        a, b, phi = h.terms[index]
        skew_a = float(sparse.linalg.norm(a-a.conj().T))
        skew_b = float(sparse.linalg.norm(b-b.conj().T))
        norm_a = float(sparse.linalg.norm(a))
        norm_b = float(sparse.linalg.norm(b))
        phi_norm = float(max(abs(scalar['grid']))) if phi else 1.
        if skew_a < 1e-12*max(1.,norm_a) and skew_b < 1e-12*max(1.,norm_b):
            contribution = phi_norm*(skew_a*norm_b+norm_a*skew_b)
            skew_bound += contribution
            singles.append(dict(index=index, matter_skew=skew_a, rq_skew=skew_b, tensor_skew_norm_upper_bound=contribution))
            remaining.remove(index)
            continue
        for other in sorted(remaining-{index}):
            c, d, other_phi = h.terms[other]
            if phi != other_phi:
                continue
            error_a = float(sparse.linalg.norm(c-a.conj().T))
            error_b = float(sparse.linalg.norm(d-b.conj().T))
            if max(error_a, error_b) < 1e-12:
                pairs.append(dict(indices=[index, other], matter_adjoint_error=error_a, rq_adjoint_error=error_b))
                skew_bound += 2*phi_norm*(error_a*float(sparse.linalg.norm(d))+norm_a*error_b)
                remaining.remove(index)
                remaining.remove(other)
                break
        else:
            raise RuntimeError('No Hermitian single or adjoint pair for actual H term')
    return dict(term_count=len(h.terms), scalar_skew=scalar_skew,
                hermitian_single_terms=singles, adjoint_pairs=pairs,
                full_H_skew_spectral_norm_upper_bound=skew_bound,
                finite_H_vs_Hermitian_part_state_error_upper_bound=float(np.expm1(request['time']*skew_bound/2)),
                failed_initial_packaging_attempt='The first factor audit used absolute1e-12 and stopped on bare matter term skew5.844842e-12 (relative about2.5e-14). No physics evolution or diagnostics reran. Replaced that binary tolerance with explicit tensor skew bound; no author H change.',
                scope='Intended projected H is self-adjoint by Hermitian singles/adjoint pairs. Actual floating snapshot has displayed tiny skew; factor Frobenius norms bound its spectral skew and exp(t*skew/2)-1 bounds state departure from its Hermitian part. This numerical norm accounting and Taylor defect estimate are not interval floating-point arithmetic certificates.')


def bind(receipt, request):
    owner = 'rich_spatial_colour_em_20261015_'
    manifest = json.loads((ROOT/(owner+'manifest.json')).read_text())
    extra = [owner+x for x in ['synthesis.json','interface.json','equations.json','manifest.json',
                               'b_event.py','b_control.py','b_event1_input.json','b_event1_result.json',
                               'b_event1_interruption.json','b_control1_input.json','b_control1_result.json',
                               'b_control1_exit.json']]
    extra += ['rich_review_wave19_20261015_synthesis.json']
    selected = sorted(set(extra) | set(receipt['source_input_sha256_before']))
    records = {}
    with ZipFile(ROOT/(owner+'handoff.zip')) as archive:
        membership = set(archive.namelist())
        for name in selected:
            actual = sha(ROOT/name)
            published = manifest['files'].get(name, {}).get('sha256')
            archive_digest = hashlib.sha256(archive.read(name)).hexdigest() if name in membership and not name.endswith('.npz') else None
            records[name] = dict(sha256=actual, author_manifest_sha256=published,
                                 manifest_match=None if published is None else actual==published,
                                 archive_member_sha256=archive_digest,
                                 archive_match=None if archive_digest is None else actual==archive_digest)
    handoff = sha(ROOT/(owner+'handoff.zip'))
    expected = '027b26a41733d1bae9828f29c8db710a9577ca99158f701e503f01312e9e1ee9'
    if handoff != expected or any(r['manifest_match'] is False or r['archive_match'] is False for r in records.values()):
        raise RuntimeError('Selected target byte binding failed')
    positive = json.loads((ROOT/(owner+'b_event1_interruption.json')).read_text())
    control = json.loads((ROOT/(owner+'b_control1_exit.json')).read_text())
    control_input = json.loads((ROOT/(owner+'b_control1_input.json')).read_text())
    positive_input = json.loads((ROOT/(owner+'b_event1_input.json')).read_text())
    return dict(handoff_sha256=handoff, handoff_matches_assignment=True, selected=records,
                scope='New event/control, governing target documents and independently imported exact H/source dependency closure only; no whole-old-campaign replay.',
                author_positive_receipt=dict(natural_completion=positive['natural_completion'], observed_returncode=positive['observed_returncode'],
                                             stage=positive['last_committed_stage'], completed_times=positive['completed_times'],
                                             immutable=positive['source_inputs_unchanged']),
                author_control_receipt=dict(natural_returncode=control['natural_exit'], immutable=control['immutable_during_run'],
                                            original_t0_from_saved_initial=True, time=control_input['time']),
                author_input_relation=dict(independent_same_snapshot=all(request[key]==positive_input[key]==control_input[key] for key in ['state_json','state_npz']),
                                           independent_new_time=request['time'], author_positive_time=1.,
                                           independent_new_pulse=request['drive_profile'], author_pulse=positive_input['drive_profile']),
                actual_H_hermiticity=hermiticity(request))


def diagnose():
    selected = [PREFIX+suffix+'.py' for suffix in ['consumer','capture','finalize']]
    version = subprocess.run(['/home/administrator/.local/bin/pyright','--version'], capture_output=True, text=True, check=True).stdout.strip()
    command = ['/home/administrator/.local/bin/pyright','--pythonpath',PYTHON,'--outputjson',*selected]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    checker = json.loads(done.stdout)
    lsp = language_server(selected)
    structures = {name:[dict(name=node.name,line=node.lineno) for node in ast.walk(ast.parse((ROOT/name).read_text()))
                        if isinstance(node,(ast.ClassDef,ast.FunctionDef))] for name in selected}
    result = dict(kind='ONCE_FINAL_THREE_NEW_FILES_NOT_PROJECT_CLEAN', version=version, command=command,
                  natural_process_exit=True, returncode=done.returncode, summary=checker.get('summary'),
                  errors_by_file=dict(Counter(Path(d['file']).name for d in checker.get('generalDiagnostics',[]) if d['severity']=='error')),
                  checker=checker, stderr=done.stderr, rooted_LSP=lsp, positive_AST=structures,
                  pre_edit='Installed rooted Pyright direct LSP symbols+diagnostics on b_control.py exited0; positive ast_grep assemble declaration exercised before writes. xd://lsp unmounted, no installation or configuration change.',
                  scope='Only three new owned Python files explicitly selected; imported helper historical errors are not declared clean, repaired or rechecked as an old seventeen-file backlog.')
    save('language_final.json', result)
    return dict(version=version, summary=result['summary'], errors_by_file=result['errors_by_file'],
                LSP_exit=lsp['server_exit'], artifact=PREFIX+'language_final.json')


def run():
    request = json.loads((ROOT/(PREFIX+'input.json')).read_text())
    observation = json.loads((ROOT/(PREFIX+'provisional.json')).read_text())
    receipt = json.loads((ROOT/(PREFIX+'exit.json')).read_text())
    binding = bind(receipt, request)
    save('binding.json', binding)
    diagnostics = diagnose()
    truncation = {}
    for arm in ['baseline','pulse']:
        rows = observation['propagation'][arm]
        truncation[arm] = dict(accepted_exact_arithmetic_bound=float(np.sqrt(sum(row.get('terminal_defect_bound',0.)**2 for row in rows))),
                              four_extra_terms_difference=float(np.sqrt(sum(row.get('four_extra_terms_difference',0.)**2 for row in rows))),
                              delivered_state_conservative_bound=float(np.sqrt(sum((row.get('terminal_defect_bound',0.)+row.get('four_extra_terms_difference',0.))**2 for row in rows))),
                              energy_drift=observation[arm]['energy']-observation['initial_'+('baseline' if arm=='baseline' else 'pulse')]['energy'])
        truncation[arm]['finite_H_vs_Hermitian_part_state_error_upper_bound'] = binding['actual_H_hermiticity']['finite_H_vs_Hermitian_part_state_error_upper_bound']
    synthesis = dict(kind='INCREMENTAL_REVIEW24_PRIMARY_PHYSICAL_B_FEEDBACK_NOT_FRAME_SPLICE_OR_ADMISSION',
        governor=dict(plan_id=request['plan_id'], launch_revision=request['launch_revision'], template=request['template'], work_item='independent-review', requirements=['outcome','verification'], plan_writer='Main only'),
        answer='Independent short full-compatible-space physical uud-field consumer is positive: a field-only mixed TE/TM intervention changes whole physical charge PVM probabilities, colour, scalar and reverse-source observables. Actual full transverse-OFF H factorizes and forces zero material pulse excess from the same original t0 state/preparation, including entangled preparation. This supports the finite interaction candidate; no new delivery-blocking physical defect was found in the newly reviewed causal claim.',
        first_discriminator=PREFIX+'first_verdict.json',
        author_report=dict(positive='Frozen t1 endpoints are interrupted-driver evidence, NOT late/anti-B natural completion.',
                           control='b_control1 naturally exited numeric0, immutable inputs, OFF from saved original t0; not off switched at t1.',
                           reported_t1=dict(delta_rho0=0.0003573120299965038, charge_phi_TV=0.0014254268813166506, off_charge_phi_TV=1.622888026403674e-14),
                           independent_protocol_warning='Our t.125 mixed TE/TM result is not a same-protocol rerun or an independent temporal-error certification of author t1. Different signs are not a discrepancy.'),
        unchanged_prior_evidence=dict(artifact='rich_review_wave19_20261015_synthesis.json', sha256=sha(ROOT/'rich_review_wave19_20261015_synthesis.json'),
                                      retained='Original same-H independent reference/all56 physical sources/full six-current response and lepton/C-paired dynamics retain their exact earlier protocol/exit scope; none replayed.',
                                      excluded='REVIEW23 physical-LEFT/r3 FrameField is a different law. No state, spin conclusion or source/gap is imported from it.'),
        independently_observed=dict(protocol=request, declared_reference=observation['reference_input'], source=observation['source'],
                                    shape=observation['shape'], dimension=observation['full_dimension'], excess=observation['excess'],
                                    charge_phi_PVM_TV=observation['charge_PVM']['charge_phi_TV'],
                                    physical_charge_PVM=dict(label_range=[min(observation['charge_PVM']['physical_charge_labels']),max(observation['charge_PVM']['physical_charge_labels'])],
                                                             identity_error=observation['charge_PVM']['identity_error'], idempotence_error=observation['charge_PVM']['idempotence_error_max'], rho0_reconstruction_error=observation['charge_PVM']['physical_rho0_reconstruction_error']),
                                    all_M=observation['all_M'], H_reassembly_error=observation['owning_H_reassembly_relative_error'],
                                    initial_material_pulse_jump_max=max(abs(observation['initial_pulse'][key]-observation['initial_baseline'][key]) for key in ['rho0','C','B','phi','phi2','source_removal_strength']),
                                    norm_errors=[abs(observation[arm]['norm']-1) for arm in ['baseline','pulse']], numerical_accuracy=truncation),
        causal_discriminator=dict(exact_off_material_excess=0., actual_termwise_factorization_max_error=observation['causal_off']['maximum_field_identity_error'],
                                  theorem=observation['causal_off']['theorem'], assumptions=observation['causal_off']['projected_assumptions'],
                                  receipt='Analytic consequence of checked actual projected tensor-H, not an independently run OFF time trajectory; duplicate long OFF arms deliberately replaced.',
                                  source_observable='Source removal is expectation of S Sdag tensor I_field, not event probability, binding or a species projector.',
                                  typing='Pq labels are physical site0 EM charge1 units; h=3q integer labels, multiplet IDs, colour C Casimir and reverse-source capture are distinct. Inclusive field TV may remain nonzero in OFF. Hops-only-off is not this control.'),
        runtime=dict(natural_process_exit=receipt['natural_process_exit'], returncode=receipt['returncode'], interrupted=False, immutable_inputs=receipt['immutable_inputs'],
                     elapsed_seconds=receipt['elapsed_seconds'], peak_RSS_KiB=observation['peak_RSS_KiB'], receipt=PREFIX+'exit.json'),
        reviewed_findings=dict(new_blocking_defects=[], retained_limits='Published finite charge/scalar/angular/field/native regulator and Ward/CCR/current-force/image-support limitations remain in force. Not audited away by this positive event.'),
        diagnostics=diagnostics,
        boundaries='No frequency sweep, all56/current/lepton replay, old late/anti-B completion, H/current-range theorem, binding, COM, photons, continuum Maxwell, global ground, Nature/QCD or issue closure. Frozen primitive source matrix/reference are inputs, not independently derived again. No author edits, graph/plan/public/git changes.',
        artifacts=dict(observation=PREFIX+'provisional.json', states=PREFIX+'states.npz', binding=PREFIX+'binding.json', manifest=PREFIX+'manifest.json', capsule=PREFIX+'capsule.zip', capsule_receipt=PREFIX+'capsule_receipt.json'))
    save('synthesis.json', synthesis)
    owned = sorted(path for path in ROOT.glob(PREFIX+'*') if path.is_file() and path.suffix not in ['.zip'] and path.name not in [PREFIX+'manifest.json',PREFIX+'capsule_receipt.json'])
    frozen_names = set(receipt['source_input_sha256_before'])-set(path.name for path in owned)
    frozen_names.add('rich_review_wave12_20261015_diagnose.py')
    capsule_files = owned+[ROOT/name for name in sorted(frozen_names)]
    manifest = dict(kind='EXACT_REVIEW24_SELF_CONTAINED_RUNTIME_INPUT_CAPSULE_MANIFEST',
                    files={path.name:dict(sha256=sha(path),bytes=path.stat().st_size,role='owned review artifact' if path in owned else 'frozen runtime dependency or explicit input copy; original unchanged') for path in capsule_files},
                    excluded='Original3.9GB author handoff is external SHA-pinned in binding.json. Exact original reference/snapshot NPZ and full local runtime import closure ARE embedded; no evolved author output is an independent input.')
    save('manifest.json', manifest)
    capsule = ROOT/(PREFIX+'capsule.zip')
    with ZipFile(capsule,'w',compression=ZIP_DEFLATED) as archive:
        for path in capsule_files+[ROOT/(PREFIX+'manifest.json')]:
            archive.write(path,arcname=path.name,compress_type=ZIP_STORED if path.suffix=='.npz' else ZIP_DEFLATED)
    save('capsule_receipt.json',dict(capsule=capsule.name,sha256=sha(capsule),bytes=capsule.stat().st_size,
                                    manifest_sha256=sha(ROOT/(PREFIX+'manifest.json')),synthesis_sha256=sha(ROOT/(PREFIX+'synthesis.json'))))
    print(json.dumps(dict(verdict=synthesis['answer'],diagnostics=diagnostics,numerical_accuracy=truncation,capsule_sha256=sha(capsule))),flush=True)


if __name__ == '__main__':
    run()
