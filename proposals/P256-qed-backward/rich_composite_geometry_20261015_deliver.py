"""Deliver completed actual model/derivative capsules, not a completion claim.
Requires natural successful scientific receipts; archives are streamed Zip64.
Historical failed/interrupted runs remain distinct external provenance, not poles.
"""
from pathlib import Path
import hashlib,json,resource,shutil,sys,time,zipfile

ROOT=Path(__file__).resolve().parent
PREFIX='rich_composite_geometry_20261015_'


def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def actual_run(name):
    if not name.startswith(PREFIX) or '/' in name or '\\' in name:raise ValueError('Owned actual run prefix required')
    receipt=json.loads((ROOT/(name+'_exit.json')).read_text());result=json.loads((ROOT/(name+'_result.json')).read_text())
    if receipt['returncode']!=0 or not receipt['natural_exit']:raise ValueError('Required scientific result is not an actual natural0: '+name)
    if receipt['source_input_before']!=receipt['source_input_after']:raise ValueError('Captured scientific epoch changed: '+name)
    return receipt,result


def main():
    config=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);started=time.time();model_name=config['model_run'];field_name=config['field_eom_run'];model_exit,model=actual_run(model_name);field_exit,field=actual_run(field_name)
    core=[model_name,field_name,PREFIX+'correlated_cloud_variational1',PREFIX+'correlated_cloud_residual1',PREFIX+'correlated_cloud_transport1',PREFIX+'correlated_source_eom1',PREFIX+'correlated_cloud_parity1',PREFIX+'correlated_leading1',PREFIX+'tetra_free1',PREFIX+'tetra_free_counts1']
    frozen=[]
    for run in dict.fromkeys(core):
        receipt,observed=actual_run(run);capsule=ROOT/(run+'_capsule.zip');frozen.append({'run':run,'returncode':receipt['returncode'],'natural_exit':receipt['natural_exit'],'child_peak_RSS_KiB':receipt.get('child_peak_RSS_KiB'),'self_peak_RSS_KiB_from_result':observed.get('resource',{}).get('peak_RSS_KiB'),'RSS_semantics':'Sole-child receipt and scientific self-resource are separately labelled; absent remainsUNKNOWN, never backfilled by inference.','capsule':capsule.name,'bytes':capsule.stat().st_size,'SHA256':sha(capsule)})
    historical=[]
    for suffix in ('tetra_charged1','tetra_charged2','tetra_poles1','correlated_onehop_poles1','correlated_dressing1','correlated_dressing2'):
        run=PREFIX+suffix;receipt_path=ROOT/(run+'_exit.json');receipt=json.loads(receipt_path.read_text());capsule=ROOT/(run+'_capsule.zip')
        historical.append({'run':run,'returncode':receipt['returncode'],'natural_exit':receipt['natural_exit'],'signal':receipt.get('signal'),'peak_RSS_KiB':receipt.get('child_peak_RSS_KiB'),'capsule':capsule.name,'bytes':capsule.stat().st_size,'SHA256':sha(capsule),'scope':'Historical failed constructor or explicit supersession, NOT completed pole/motion/particle evidence. Exact large capsule preserved externally rather than duplicated in this bundle.'})
    sectors={}
    for key,sector in model['sectors'].items():
        sectors[key]={name:sector[name] for name in ('body_dimension','full_independent_Phi_dimension','metric','minimum_energy','gap_from_OWN_selected_reference','low_poles','generalized_residuals','observed_ground_cluster_members','observed_ground_cluster_closed_at_lowpole_boundary','ground','channels','sources','compact0_conditioning_strength','finite_prepared_events','electric_off','original_Peierls_current','original_Peierls_contact','Ward_after_ALL_MGS_and_actual_metric','material_EM_pulse_events','same_age_no_EM_events','field_only_kick_events')}
    synthesis={'kind':'ACTUAL_FINITE_CONDITIONAL_RICH_COMPOSITE_FIELD_MODEL_HANDOFF_NOT_WHOLE_ISSUE_CLOSURE','model_run':model_name,'field_source_EOM_run':field_name,'basis':model['basis'],'assembly':model['assembly'],'metric':model['metric'],'sectors':sectors,'threehalf_minus_half':model['threehalf_minus_half'],
        'actual_unmasked_field_source_derivatives':field,'completed_run_capsules':frozen,'historical_adverse_and_interrupted':historical,
        'common_law':'Same actual occupiedCAR64/Phi7/R10/PW19 geometry, physical microscopicQ/current/contact/C/source. Full independentPhi factor; rich R/complexTrU/primitive[H,B] sea/core sources, not algebra-only/fixedPhi or inert-valence trimer. No other-owner Hamiltonian/state/energy splice.',
        'operator_and_metric':'Literal operator lifts retained; source Gdot=i(K-Kdag) and moving connection are distinct from fixed finite weakH/G dynamics. No source-EOM K Hermitization or RMS subtraction as mass/decay. All actual field/Noether/SU3/shared-hop derivatives in the selected derivative consumer, not inferred from an endpoint overlap.',
        'units':'Natural input energy/field units only, no MeV fit. Bare/vector/TrU source norms and residues use their literal operator normalization; D=[H,B] norm/residue carry an EXTRA energy². Pphi source has momentum² strength. Source frequencies/cloud RMS have energy units; K3 squared image has energy^6 and t6 coefficient, NOT finite probability. Conditional residue fractions and normalized state count PVM probabilities remain separate.',
        'readout_semantics':'C/F are whole-sea regional quark occupation channels, not tagged-added-particle positions, Q_em, colour8-vector/Casimir or free breakup thresholds. Changed quark-core/sea/field wakes allowed; neither rigidOmega return nor bare-core capture defines a particle. Pinned source0/timegrid/receivers unchanged.',
        'candidate_status':'ACTUAL mobile charged spin-half THREE-QUARK CANDIDATE/component in this finite native rich-body model, with changed cores/sea/field wakes allowed; NOT a bound proton. Negative own-reference offsets are not positive gaps/masses. Compact-site probabilities differ from net physical Qsite expectations. ELECTRIC-OFF removes only electric term, not all colour interactions. Original source0/preparation/receivers preserved.',
        'Noether_and_inputs':model['Noether']+' Primitive CAR spin/selected3+1 split/flavour/charges/gauging, negative scalar quadratic/shape potential, scalar volume and pointmass density are INPUTS, not derived rank/splitting/fermion-origin theorems.',
        'limits':model['limits']+' OWN selected neutral reference remains hot under full primitiveH; actual charged offsets BOTH NEGATIVE, no recenter/positive-gap/stationary-vacuum/mass inference. New rich-state full-primitive residual/lifetime NOT measured; oldW residual is NOT inherited. R/PW/body-source/sea cuts severe. Electric-OFF is NOT allcolourOFF or a nonbinding theorem. Material continuation has not silently inherited regular spectra. No photons/COM, lab3D/mechanicalEM, proton or whole-issue closure; no independent anti-rich pole/lepton calculation claimed by this composite consumer.',
        'independent_review':'Review36 frozen capsule274a89d0581f3bba17c134121faf451bceba3612a405f0b64e310836806a2f95 independently exercises prior B/Phi/metric/corrected endpoint data only. It did NOT exercise this newly published source/model/derivative helper; its state/control is not spliced into model.',
        'verification':'Actual scoped LSP/AST and captured scientific consumers only; no build/test/lint/formatter run or all-clean typecheck claim. Main owns integration/review/plan.'}
    synthesis_path=Path(str(output)+'_synthesis.json');manifest_path=Path(str(output)+'_manifest.json');archive_path=Path(str(output)+'_capsule.zip');receipt_path=Path(str(output)+'_receipt.json');native_path=Path(str(output)+'_native_review.zip');native_manifest_path=Path(str(output)+'_native_review_manifest.json')
    for path in (synthesis_path,manifest_path,archive_path,receipt_path,native_path,native_manifest_path):
        if path.exists():raise FileExistsError('Never overwrite delivery evidence: '+str(path))
    native_names=[model_name+suffix for suffix in ('_body_basis.pkl','_J1_lowstates.npz','_J3_lowstates.npz','_assembly_receipt.json','_result.json','_input.json','_exit.json')]
    native_names.extend(name for name in model_exit['files'] if name.startswith(model_name+('_op_')) or name.startswith(model_name+('_raw_')))
    native_names.extend(name for name in model_exit['source_input_before'] if name.endswith('.py'))
    native_names=list(dict.fromkeys(native_names));captured_hashes={**model_exit['source_input_before'],**{name:item['sha256'] for name,item in model_exit['files'].items()},model_name+'_exit.json':sha(ROOT/(model_name+'_exit.json'))}
    with zipfile.ZipFile(ROOT/(model_name+'_capsule.zip')) as source:
        native_manifest={'kind':'FROZEN_NATIVE_H_METRIC_SOURCE_PVM_REVIEW_WITHOUT_PRODUCER_REPLAY','model_run':model_name,'members':{name:{'bytes':source.getinfo(name).file_size,'SHA256':captured_hashes[name]} for name in native_names},'source_epoch':'EXACT captured model epoch copied from its completed capsule, NOT current working-tree substitute.','usage':'body_basis.pkl contains p/labels/probes coordinates; loading it does NOT load1328blocks. Exact op/raw NPZ and full-coordinate states/prepared vectors included. Fresh state/evolution can use actual Gram/Phi weakH directly; no assemble/construct/eigensolve required. Full1328raw embedding blocks remain in full scientific capsule if needed for primitive-action review. See native_review_interface.json for exact coordinate/current/PVM/source definitions.'}
        native_manifest_path.write_text(json.dumps(native_manifest,indent=2)+'\n')
        with zipfile.ZipFile(native_path,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as native:
            for name in native_names:
                info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED
                with source.open(name) as incoming,native.open(info,'w',force_zip64=True) as outgoing:shutil.copyfileobj(incoming,outgoing,1048576)
            native.write(native_manifest_path,native_manifest_path.name)
    synthesis['native_review']={'capsule':native_path.name,'bytes':native_path.stat().st_size,'SHA256':sha(native_path),'manifest':native_manifest_path.name,'physical_model_change':'NONE; streamed exact captured artifacts, no producer replay or1328-block reconstruction.'}
    synthesis_path.write_text(json.dumps(synthesis,indent=2)+'\n');members=[synthesis_path,Path(__file__),Path(sys.argv[1]),native_path,native_manifest_path]
    for item in frozen:members.append(ROOT/item['capsule'])
    for suffix in ('equations.json','interface.json','dressing_interface.json','dressed_eom_interface.json','native_review_interface.json','provisional_synthesis1.json','adverse.json','environment.json','language_scoped.json','language_leading.json','language_cloud.json','language_range.json','language_parity.json','language_transport.json','language_source_eom.json','language_dressing.json','language_disk_eom.json','language_atomic.json','language_delivery.json','language_delivery2.json'):
        members.append(ROOT/(PREFIX+suffix))
    for item in historical:
        members.append(ROOT/(item['run']+'_exit.json'))
        decision=ROOT/(item['run']+'_termination_decision.json')
        if decision.exists():members.append(decision)
    manifest={'kind':'EXACT_STREAMED_SCIENTIFIC_DELIVERY_MANIFEST','members':{path.name:{'bytes':path.stat().st_size,'SHA256':sha(path)} for path in dict.fromkeys(members)},'historical_external_capsules':historical,'source_closure':'Each completed actual scientific capsule embeds its own exact executable/helper/input/output epoch. Disk body blocks are flat owned-prefix outputs inside its capsule; relocate manager on extraction and explicitly pin every block on resume. No all-old-producer replay needed.'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');members.append(manifest_path)
    with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
        for path in dict.fromkeys(members):archive.write(path,path.name,compress_type=zipfile.ZIP_STORED if path.suffix=='.zip' else zipfile.ZIP_DEFLATED)
    receipt={'kind':'ACTUAL_DELIVERY_NOT_A_NEW_PHYSICS_RESULT','capsule':str(archive_path),'bytes':archive_path.stat().st_size,'SHA256':sha(archive_path),'synthesis':str(synthesis_path),'manifest_SHA256':sha(manifest_path),'elapsed_seconds':time.time()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_returncode':model_exit['returncode'],'field_source_returncode':field_exit['returncode']}
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
