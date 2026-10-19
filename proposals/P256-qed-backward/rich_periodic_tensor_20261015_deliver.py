"""Package observed tensor physics, limitations, exact source/exit/raw arrays."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
P='rich_periodic_tensor_20261015_'


def load(s):
    return json.loads((ROOT/(P+s+'.json')).read_text())


if __name__=='__main__':
    first=load('result'); second=load('loopgas_result')
    states=first['state_comparison']+[second['state']]
    for state in states:
        state['electric2_direction_sums_xyz']=np.asarray(state['electric2']).reshape(8,3).sum(axis=0).tolist()
        state['mean_charge_pair_rho2']=float(np.mean(state['rho2']))
    support={}
    for tag in ['chi32','chi64','loopgas']:
        data=load(tag+'_samples')
        support[tag]={'outside_old_support_draws':data['outside_old_support_count'],'draws':len(data['draws']),'max_nonzero_links':max(x['nonzero_links'] for x in data['draws']),'max_Gauss_error':max(x['Gauss_max'] for x in data['draws'])}
    raw_summaries=[]
    for q in ['100','110']:
        with np.load(ROOT/(P+'q'+q+'_response.npz')) as a:
            omega=a['omega']; overlap=a['source_overlaps']
            groups=[]
            unassigned=set(range(len(omega)))
            while unassigned:
                i=min(unassigned); group=sorted(j for j in unassigned if abs(omega[j]-omega[i])<1e-5)
                unassigned.difference_update(group)
                W=overlap[group].conj().T@overlap[group]
                ev=np.linalg.eigvalsh(W)
                groups.append({'omega_min':float(min(omega[group])),'omega_max':float(max(omega[group])),'multiplicity':len(group),'joint_weight_rank':int(sum(ev>max(1e-13,1e-9*max(ev)))),'field_T1_weight':float(W[0,0].real),'field_T2_weight':float(W[1,1].real),'current_T1_weight':float(W[2,2].real),'current_T2_weight':float(W[3,3].real)})
            raw_summaries.append({'q':q,'groups':groups,'contact_real':a['contact'].real.tolist(),'contact_imag':a['contact'].imag.tolist(),'Wilson_covariance_real':a['winding_plaquette_joint_covariance'].real.tolist(),'Wilson_covariance_imag':a['winding_plaquette_joint_covariance'].imag.tolist()})
    exits={p.name:json.loads(p.read_text()) for p in ROOT.glob(P+'*exit.json')}
    synthesis={'kind':'MATERIAL_GAUGE_EXACT_TENSOR_PHYSICS_NOT_PHOTON_ESTABLISHMENT','decision':'Real32-node gauge-exact tensor states and same-H response completed; local entanglement plus H-image sectors are materially outside old selected support. Full-H residual, finite bond anisotropy and L2 winding contamination prohibit exact stationary/massless/bulk-photon interpretation. These limitations do NOT diagnose all same-H states or the S1 law as impossible.','state_comparison':states,'outside_old_support_witness':support,'latest_response':second['responses'],'source_groups_from_raw_arrays':raw_summaries,'first_construction_response':first['responses'],'own_H_controls':first['controls'],'checkpoint':'attempt2_tensor_checkpoint_capsule.zip preserves first real H-image MPS checkpoint; final actual_tensor_checkpoint reports PHYSICAL charge8.','attempts':exits,'corrections':['attempt1: Site.add_op uses permute_dense, not permute.','attempt2: finite32-node packaging hit default max18-sites-per-ring guard, preserved exit1; threshold32 declares actual network rather than deleting wrapping arcs.','attempt3: bond-limited DMRG hit default truncation guard. Later explicit exploratory max_trunc_err1 is NOT an accuracy pass; actual discarded fractions and full-H variances retained.','attempt4: exactlyzero cross-Cartesian contact had empty MPO charge graph; scalar0 represents mathematicallyzero mixed derivative, diagonal contacts and response remain actual contractions.','TeNPy get_total_charge default includes boundary gauge; physical-only target8 reporting is used thereafter. Earlier0 is not Gauss leakage.'],'limitations':['Bond-limited state energy/variance compared honestly with old selected reference; no claim lower energy unless observed.','Finite source-inserted tensor Ritz response includes compression losses and onlyfour source directions, not an exact full-H stationary Lehmann spectrum.','Actual independent transverse directions are measured atq100/q110; spectral groups/source ranks are not assigned particle species.','Two-link noncontractible periodic arcs remain. Separate sine-Wilson winding versus plaquette/current covariances are equal-time, not a winding-free bulk dispersion or dynamical Wilson pole proof.','L2 onlypi momenta cannot establish lowq masslessness, propagation, continuum Coulomb or Nature photons.','No magneticB,cosine,MaxwellD,hidden opposite-spin charge or fitted Hamiltonian constants.'], 'language_evidence':P+'language_final.json','governor':first['input']['governor']}
    synthesis['rejected_response_capsule']=P+'rejected_mixed_norm_response_capsule.zip'
    synthesis['response_correction']=load('response_correction')
    synthesis['actual_tensor_norm_benchmark']=first['initial']['actual_tensor_H_image_norm_benchmark']
    synthesis['second_tensor_norm_benchmark']=load('loopgas_tensor_norm_benchmark')
    synthesis['historical_comparison']={'ancestor_reported_energy':-11.65032,'ancestor_reported_full_H_residual':2.92448,'scope':'The tested small-bond tensor states have broader coherent support but worse energy/residual than the reported old selected variational state. Thus this is an actual tensor-state construction and finite-response artifact, NOT improved vacuum quality or a diagnosis of S1 phase/law. More support alone does not prove a better state.'}
    synthesis['corrections'].append('attempt5 and initial loopgas natural0 responses REJECTED: mixed MPOEnvironment excluded MPS.norm factors whereas Gram overlaps included them. Exact source capsules and rejected raw arrays preserved. Final helper restores norm factors; actual full-H image scaling benchmark discriminates the old bug. attempt6 benchmark itself initially discarded its image norm with default canonical_form; renormalize=False fixes benchmark norm preservation, with natural exit1 retained.')
    synthesis['science_exit_classification']={'attempt5':'natural0 but response rejected by actual finite-H norm contradiction; no photon claim','attempt7':'final first construction, see exact science exit','loopgas_exit':'final second construction, see exact science exit; earlier natural0 rejected response is inside rejected capsule'}
    synthesis['finite_bond_anisotropy']='Directional E2 sums are strongly unequal in these finite-bond, ordered-chain states. Transverse nondegeneracy is not evidence of distinct photons; this ansatz has not restored cubic symmetry.'
    synthesis['sweep_count_semantics']='Configured TeNPy max_sweeps options3/4 yield actual recorded sweep histories4/5 (including initial indexed sweep); raw histories, not configured labels, own observed counts.'
    synthesis['decision']='Actual32-node block-symmetric MPS/MPO and a materially different coherent winding-gas state were evolved with the unchanged24-link Hamiltonian, with NEW outside-old-support weights and joint transverse finite response. These tested states are not better stationary vacua than the reported old selected reference. Full-H residual, source compression, finite-bond anisotropy and unseparated finiteL2 winding effects prevent promotion to propagating/massless photons; they do NOT establish a no-photon or S1-law failure theorem.'
    synthesis['Wilson_insertion_limit']='Separate winding/plaquette covariances are recorded, but plaquette insertion retains only about36-37percent of its source norm2 at bond128; therefore their relative projected amplitudes cannot quantitatively establish winding dominance. They are finite equal-time diagnostic projections, not dynamical or bulk separation. err.eps is the library accumulated truncation metric, distinct from exact-vs-compressed norm2 retention.'
    synthesis['language_diagnostics']={'result_file':P+'language_final.json','summary':load('language_final')['diagnostics']['summary'],'scope':'Actual rooted LSP symbols/definition and positive Python AST succeeded. Selected Pyright residuals are reported honestly; no clean-typecheck claim. Scientific natural exits and numerical benchmarks are separate evidence.'}
    (ROOT/(P+'synthesis.json')).write_text(json.dumps(synthesis,indent=2)+'\n')
    files=[p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in [P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json']]
    manifest={'kind':'EXACT_BYTES_TENSOR_SCIENCE_DELIVERY','files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},'excluded_private_env':'not packaged; physics-tenpy1.1.1,numpy2.5.4,scipy1.18.1,Python3.12 in prefix-owned isolated env; active runtime untouched','pure_ancestor_dependency_sha256':hashlib.sha256((ROOT/'rich_periodic_photon_20261015_model.py').read_bytes()).hexdigest()}
    (ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+'delivery_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/(P+'manifest.json')]:z.write(p,p.name)
    capsule=ROOT/(P+'delivery_capsule.zip')
    receipt={'capsule':capsule.name,'bytes':capsule.stat().st_size,'sha256':hashlib.sha256(capsule.read_bytes()).hexdigest(),'synthesis':P+'synthesis.json'}
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
