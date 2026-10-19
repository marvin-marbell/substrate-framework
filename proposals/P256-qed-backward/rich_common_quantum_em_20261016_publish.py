"""Evidence-only handoff: actual completed runs, native inputs, <=80-line physics.
Does not run a solver, test, old producer, or substitute missing model output.
"""
from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np


def load(path):return json.loads(Path(path).read_text())


def digest(path):
    result=hashlib.sha256()
    with open(path,'rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):result.update(block)
    return result.hexdigest()


def amplitude(rows,key):return float(max(np.max(abs(np.asarray(row[key],float)))for row in rows))


def current_amplitude(rows,species):return float(max(abs(row['physical_current'][species])for row in rows))


def metrics(config):
    d=load(config['dynamics']+'_result.json');s=load(config['stationary']+'_result.json');f=load(config['fieldreadout']+'_result.json');phase=load(config['phase']+'_result.json')['phase_discriminator']['phase_landscape'];spectrum=load(config['spectrum']+'_result.json')['source_supported_collective_spectrum'];contrast=d['added_neutral_pair_source_contrasts'];full='full_live_quantum_links';control='TRANSVERSE_link_identity_active_matter_source_control';response={}
    for law in(full,control):
        response[law]={}
        for kind in('field_kick_y','field_kick_z','white_pulse'):
            rows=contrast[law]['reciprocal_responses'][kind];response[law][kind]={'added_pair_quark_current_amplitude':current_amplitude(rows,'quark'),'added_pair_white_current_amplitude':current_amplitude(rows,'white'),'added_pair_Q_site_amplitude':amplitude(rows,'Q_site'),'added_pair_transverse_physical_generator_amplitude':amplitude(rows,'flux_mean')/3,'added_pair_transverse_second_moment_amplitude':amplitude(rows,'flux_second')}
    norm=max(abs(event['norm']-1)for sector in d['sectors'].values()for law in sector.values()for label in('baseline','field_kick_y','field_kick_z','white_pulse')for event in law[label]);activation={sector:{law:value['external_white_source_activation_state_norm']for law,value in laws.items()}for sector,laws in d['sectors'].items()};energy_drift=max(max(abs(event['own_H_preparation_or_evolution_energy']-events[0]['own_H_preparation_or_evolution_energy'])for event in events)for laws in d['sectors'].values()for law in laws.values()for label in('baseline','field_kick_y','field_kick_z','white_pulse')for events in[law[label]])
    with np.load(config['stationary']+'_collective_response.npz')as data:
        weight=np.diag(data['complete_equal_time_Gram'])[:2].real;first=np.diag(data['complete_energy_weighted_Gram'])[:2].real
    centroids=(first/weight).tolist();rms=(np.asarray([row['force_state_norm']for row in s['field_force_current_Ward']])/np.sqrt(weight)).tolist()
    whole={law:{kind:{'time':value[-1]['time'],'quark_current_delta':value[-1]['physical_current']['quark'],'white_current_delta':value[-1]['physical_current']['white'],'Q_site_delta':value[-1]['Q_site'],'transverse_integer_flux_delta':value[-1]['flux_mean']}for kind,value in data['same_age_responses'].items()}for law,data in d['sectors']['pair'].items()}
    colour={law:{kind:float(max(abs(row['colour_ReTrU'])for row in rows))for kind,rows in value.items()}for law,value in f['added_neutral_pair_source_response'].items()};return{'source':d['source'],'stationary':s,'responses':response,'source_activation_norms':activation,'maximum_evolution_norm_drift':norm,'maximum_fixed_H_energy_drift':energy_drift,'colour_loop_source_responses':colour,'phase_landscape_energy_width':max(x['own_clamped_energy']for x in phase)-min(x['own_clamped_energy']for x in phase),'phase_landscape_grid_minimum':min(phase,key=lambda x:x['own_clamped_energy']),'source_supported_spectrum':spectrum,'complete_full_H_electric_source_centroid':centroids,'complete_full_H_electric_source_rms_frequency':rms,'whole_PAIR_last_time_responses':whole}


def native_archive(config,out):
    used=set();manifest={};archive=zipfile.ZipFile(out+'_native.zip','w',allowZip64=True)
    def add(name,payload):
        if name in used:return
        used.add(name);binary=name.endswith(('.npz','.zip','.pkl'));info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_STORED if binary else zipfile.ZIP_DEFLATED;archive.writestr(info,payload);manifest[name]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
    for path in sorted(Path.cwd().glob('rich_common_quantum_em_20261016_*.py')):add(path.name,path.read_bytes())
    for bundle in config['native_input_bundles']:
        with zipfile.ZipFile(bundle)as source:
            for info in source.infolist():
                name=Path(info.filename).name
                if not info.is_dir()and name not in used:add(name,source.read(info))
    for pattern in config['native_patterns']:
        paths=sorted(Path.cwd().glob(pattern))
        if not paths:raise FileNotFoundError('Missing actual native output '+pattern)
        for path in paths:
            if path.is_file()and path.name not in used:add(path.name,path.read_bytes())
    archive.close();Path(out+'_native_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return digest(out+'_native.zip'),digest(out+'_native_manifest.json')


def main():
    config=load(sys.argv[1]);out=config['out'];runs={}
    for prefix in config['required_natural_runs']:
        receipt=load(prefix+'_exit.json')
        if not receipt['natural_zero']or receipt['input_epoch_changed']:raise ValueError('Missing valid unchanged-epoch completed consumer '+prefix)
        runs[prefix]=receipt
    preserved={prefix:load(prefix+'_exit.json')for prefix in config['preserved_runs']};m=metrics(config);loop=load(config['loopsea_result']);full='full_live_quantum_links';control='TRANSVERSE_link_identity_active_matter_source_control';r=m['responses'];s=m['stationary'];src=m['source'];stiff=s['clamped_matter_induced_stiffness'];loop_phase=loop['phase_discriminator']['phase_landscape'];loop_width=max(x['own_clamped_energy']for x in loop_phase)-min(x['own_clamped_energy']for x in loop_phase)
    quantum=load(config['loopquantum']+'_result.json')
    harmonic=load(config['harmonics']+'_result.json');quantum_support=load(config['harmonicloops']+'_result.json');m['actual_C1_selected_state_harmonic_support']=harmonic;m['actual_loop_rich_quantum_harmonic_and_circuit_support']=quantum_support;Path(out+'_metrics.json').write_text(json.dumps(m,indent=2)+'\n')
    white_late=[leg['retained_fraction_of_virtual_leg']for state in harmonic['selected_states']if state['state_label'].startswith('C1_pair_')and state['state_label'].endswith('_t1')for leg in state['direct_harmonics']if leg['root']=='charged_white_nminus3'and leg['direction']==1]
    quantum_words=quantum_support['actual_quantum_nonbacktracking_circuits']
    whole=m['whole_PAIR_last_time_responses'][full]
    loops=loop['phase_discriminator']['nonbacktracking_operator_support']['actual_directed_loop_support']
    lines=[
      'NEW common rich matter + live quantum charge-link candidate; finite conditional model, NOT completed photons/protons.',
      'One CAR64 occupied N32 reference and N36 neutral Bdag0 Wilson Ldag2 pair, shared Phi7/R10/PW19 and live charge field.',
      'B is canonical colour-singlet uud spin-half source; L is charged DOWN-WHITE spin-half. Distinct opposite-charge operators, NOT C partners.',
      'Pair source has total Noether J1,M1, not spinhalf. No isolated charged pole/C-conjugate N28 sector on this closed Gauss model.',
      'Root3+1/flavours/spins/charges/gauging, negative scalar quadratic, point-density/inertia and new I_E=1 are INPUTS.',
      'Classical common link U8=diag(e^(2ia)V,1,e^(-ia)V,e^(-3ia)); charge trace0, traceQint²24. SU3xU1 Z3 cover not quotiented.',
      'Independent colour1.8 and charge I_E1 kept; no coupling unification/measuredcharge or SU8 gauge theory derived from embedding.',
      'Tetrahedral matter in 3D FCC torus: original4bonds plus physical02 colour-tree-route chord, unchanged g=.65,v=.6.',
      'Two actual finite-q sine-link coordinates at qX=(sqrt2*pi,0,0); Dt=0, tTt=8I; cycle/profile matrix[(0,-2),(-1,-1)] rank2.',
      'Gauss-reduced cover metric gives H_E=Nint^T Lplus Nint/2+(my²+mz²)/16, physicalQ=Nint/3 and physical electricflux=Eint/3.',
      'Edge E is real projected flow, NOT independent integer edge-flux rotors. Spatial cover-map kernel Z2 differs from colour/charge Z3.',
      'Reference exact even(my+mz) sector has all85 signed fluxes |m|<=6. Other cover sectors not claimed absent/higher.',
      'Both scalar gPhi and vector iv sigma.Rd hopping carry SAME quantum integer-charge Peierls factors (2,-1,0,-3).',
      'No Maxwell B², fitted photon bath/omega(q), added fermion mass, phase recentering, spin penalty, or reducible-Dyson join.',
      f'Fresh selected neutral stationary E={s["own_selected_sector_reference_energy"]:.12g}; actual Gauss localQ variances and transverse purity={s["reduced_transverse_purity"]:.9g}.',
      'This is a genuinely charge-fluctuating occupied sea with correlated Gauss flux, not matter tensor zero electricfield.',
      f'Field-force/current Ward max={max(x["residual"]for x in s["field_force_current_Ward"]):.4g}; reference flux-boundary weight={s["reference"]["flux_boundary_weight"]:.4g}.',
      'Complete Bdag-Wilson-Ldag word acts BEFORE final physical/body selection; no projection after individual charged factors.',
      f'Raw CAR source strength={src["raw_full_CAR_source_strength_NOT_probability"]:.12g}; retained both={src["retained_body_AND_flux_strength"]:.12g}; raw strengths NOT probabilities.',
      f'Source retained fraction={src["retained_fraction_of_raw_source_NOT_particle_probability"]:.12g}, raw body loss={src["raw_CAR_body_lost_strength"]:.4g}, raw flux loss={src["raw_CAR_flux_lost_strength"]:.4g}.',
      f'Wilson path2->0->1->2->0 shifts transverse m by(-3,-3); endpoint Gauss residual={src["Wilson_endpoint_integer_Gauss_residual"]:.4g}.',
      'Raw full source blocks/local CAR dictionary/Gram exported; prepared source E is preparation energy, NOT pole/mass/binding.',
      'Actual reciprocal response uses finite white-species diagnostic phase. It is NOT falsely labelled universal labEM.',
      'Universal quantum-field generator kicks probe both charged source components; source/receiver phases differentiated independently.',
      f'WHOLE-PAIR same-age t1 current changes: field-y(q,w)=({whole["field_kick_y"]["quark_current_delta"]:.8g},{whole["field_kick_y"]["white_current_delta"]:.8g}), field-z=({whole["field_kick_z"]["quark_current_delta"]:.8g},{whole["field_kick_z"]["white_current_delta"]:.8g}); NOT added-particle contrasts.',
      'Responses below are [PAIR perturbed-PAIR same-age baseline]-[NEUTRAL perturbed-NEUTRAL same-age baseline], no charged four-sector splice.',
      f'White pulse -> transverse physical-generator source response max={r[full]["white_pulse"]["added_pair_transverse_physical_generator_amplitude"]:.8g}; active control={r[control]["white_pulse"]["added_pair_transverse_physical_generator_amplitude"]:.4g}.',
      f'Field-y kick -> source quark/white current max={r[full]["field_kick_y"]["added_pair_quark_current_amplitude"]:.8g}/{r[full]["field_kick_y"]["added_pair_white_current_amplitude"]:.8g}.',
      f'Field-z kick -> source quark/white current max={r[full]["field_kick_z"]["added_pair_quark_current_amplitude"]:.8g}/{r[full]["field_kick_z"]["added_pair_white_current_amplitude"]:.8g}.',
      f'Field-y/z kick -> source Qsite max={r[full]["field_kick_y"]["added_pair_Q_site_amplitude"]:.8g}/{r[full]["field_kick_z"]["added_pair_Q_site_amplitude"]:.8g}.',
      'Control removes ONLY transverse quantum link factors, keeping ALL matter hops, Gauss longitudinal and free field kinetic, with own current/contact.',
      f'Control field-y source current max(q,w)={r[control]["field_kick_y"]["added_pair_quark_current_amplitude"]:.4g}/{r[control]["field_kick_y"]["added_pair_white_current_amplitude"]:.4g}; minimum control source activation={min(value[control]for value in m["source_activation_norms"].values()):.8g}.',
      f'Actual source-colour ReTrU response to field-y/z={m["colour_loop_source_responses"][full]["field_kick_y"]:.8g}/{m["colour_loop_source_responses"][full]["field_kick_z"]:.8g}; Phi/frame/colour energies separately exported.',
      f'Max actual norm/constant-H energy drift={m["maximum_evolution_norm_drift"]:.4g}/{m["maximum_fixed_H_energy_drift"]:.4g}; this numerical unitarity is NOT a particle lifetime proof.',
      f'All1806 finite-matter virtuals give phase0 stiffness eigenvalues{stiff["total_eigenvalues"]}; NOT an inserted magnetic term.',
      f'Nonresonant5x5 actual holonomy energy width={m["phase_landscape_energy_width"]:.8g}; sampled minimum is shifted, not forced phase0/global theorem.',
      f'Actual joint-state phase-POVM max/min={m["source_supported_spectrum"]["actual_joint_state_phase_POVM"]["marginal_phase_max_min_ratio"]:.9g}; no quantum vacuum recentering.',
      'Lowest20 have little electric-generator weight: spectral WINDOW finding, NOT absence/mode-count theorem.',
      f'Exact full-H electric-source centroid frequencies={m["complete_full_H_electric_source_centroid"]}, RMS={m["complete_full_H_electric_source_rms_frequency"]}; not an excited-state-window estimate.',
      'Source-resolved full-H96-column Krylov locates higher mixed matter/electric weight; broad Ritz residuals are NOT resolved photon poles.',
      'Bare rotor loop gaps are explicit comparators, not photons. Positive curvature/two coordinates alone would not prove photons either.',
      'Measured flux-cut coordinate commutators are nonzero and separately retained; tiny boundary probability is not a full-cut theorem.',
      f'Actual primitive n2/-1/-3 field legs survive. White rank50/85; neutral virtual-leg fraction .99999999990, initial pair forward .97338333, backward .99999684; raw energy² strengths, NOT particle probabilities.',
      f'Actual paired t1 forward-white retained virtual-leg fractions {min(white_late):.9g}..{max(white_late):.9g}; up to{1-min(white_late):.5g} lost. Tiny neutral/source boundary weight does NOT establish virtual-field convergence.',
      f'Direct t0 white-source force identity residual={harmonic["direct_white_initial_force_from_completed_data"]["pair"]["residual"]:.4g}; direct charged-white leg, not only later indirect Phi/quark response.',
      f'Material fork actually TESTED: occupied-sea intermediates complete NONBACKTRACKING01->12->20 triangle and01->12->23->30 square, not two-edge backtracks; body={loop["model"]["body_dimension"]}.',
      f'Quark triangle/square norm²={loops[0]["inclusive_operator_output_norm2_NOT_probability"]:.7g}/{loops[1]["inclusive_operator_output_norm2_NOT_probability"]:.7g}, white={loops[2]["inclusive_operator_output_norm2_NOT_probability"]:.7g}/{loops[3]["inclusive_operator_output_norm2_NOT_probability"]:.7g}; energy powers, NOT probabilities.',
      f'Fork holonomy width={loop_width:.9g}, phase0 stiffness eigenvalues={loop["phase_discriminator"]["phase0_stiffness"]["eigenvalues"]}; own virtual response, NEVER first-law pair-state splice.',
      f'Loop-rich FULL live quantum reference independently computed: E={quantum["own_selected_neutral_quantum_energy"]:.12g}, transverse purity={quantum["reduced_transverse_purity"]:.9g}.',
      f'Its actual phase-POVM max/min={quantum["supported_collective_response"]["actual_joint_state_phase_POVM"]["marginal_phase_max_min_ratio"]:.9g}; full-H electric Gram eigenvalues={quantum["complete_full_H_electric_Gram_eigenvalues"]}.',
      'Loop-rich quantum H contains original live matter/field operators, NOT inserted BO potential; supported24-window response is not a photon proof or first-law paired-state splice.',
      f'Actual loop-rich QUANTUM every-intermediate-cut strengths (quarkTriangle,quarkSquare,whiteTriangle,whiteSquare)={[word["ACTUAL_every_intermediate_field_projector_output_norm2"]for word in quantum_words]}; raw energy powers, NOT probabilities.',
      f'Actual quantum loop retained fractions={[word["actual_retained_fraction"]for word in quantum_words]}; net-shift-only shortcut is separately measured, NOT substituted for primitive operator words.',
      f'Loop-rich reference boundary={quantum["reference"]["flux_boundary_weight"]:.6g}; coordinate-commutator cut defects={[x["matter_coordinate_commutator_boundary_defect_norm"]for x in quantum["flux_cut_coordinate_defects"]]}, electric terms={[x["electric_kinetic_coordinate_commutator_norm"]for x in quantum["flux_cut_coordinate_defects"]]}. No field-cut convergence claimed.',
      'Historical source-selected spaces not assumed nested or variational ordering repair. Missing particle/photon outcomes remain open.',
      'SEVERE: selected-body CAR sea is not full Dirac/all-pair closure; Phi7/R10/PW19 and only2of21 spatial charge cycles; selected J sectors.',
      'Current and q are MATERIAL; lab rotating-density minimal coupling/full field spatial Noether/freeCOM are not silently claimed.',
      'Qsite/count and compactquark masks are not taggedparticle/capture probabilities; no bound proton/QCD, mass, lifetime or Nature claim.',
      'Preserved filename-collision natural0, real-estimator casting natural1 and legacy-CSR schema natural1 are technical histories, NOT physical model failures.',
      'Actual completed natural-exit receipts, raw arrays, source/input epochs, code and exact interfaces are packaged; no old solved state/energy/dispersion reused.',
      'Equations/interface: rich_common_quantum_em_20261016_equations.json / rich_common_quantum_em_20261016_interface.json.'
    ]
    if len(lines)>74:raise ValueError('Physics synthesis exceeds reserved80line envelope')
    Path(out+'_physics.txt').write_text('\n'.join(lines)+'\n');native_hash,manifest_hash=native_archive(config,out);evidence={'valid_runs':runs,'preserved_runs':preserved,'native_sha256':native_hash,'native_manifest_sha256':manifest_hash};Path(out+'_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    capsule=out+'_capsule.zip'
    with zipfile.ZipFile(capsule,'w',compression=zipfile.ZIP_STORED,allowZip64=True)as archive:
        for path in[Path(out+'_native.zip'),Path(out+'_native_manifest.json'),Path(out+'_metrics.json'),Path(out+'_physics.txt'),Path(out+'_evidence.json'),Path(sys.argv[1]),Path(__file__)]:archive.write(path,path.name)
        for prefix in config['required_natural_runs']+config['preserved_runs']:
            path=Path(prefix+'_capsule.zip');archive.write(path,path.name)
    capsule_hash=digest(capsule);receipt={'capsule':capsule,'capsule_sha256':capsule_hash,'native':out+'_native.zip','native_sha256':native_hash,'native_manifest_sha256':manifest_hash,'physics_lines':len(lines),'natural_runs':len(runs),'preserved_runs':len(preserved)};Path(out+'_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');final=lines+['Native executable capsule SHA256 '+native_hash,'Full provenance capsule SHA256 '+capsule_hash,'Native manifest SHA256 '+manifest_hash];Path(out+'_synthesis.txt').write_text('\n'.join(final)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
