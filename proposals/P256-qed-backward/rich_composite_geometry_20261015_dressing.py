"""NEW rich quark/sea/field-source model, not more native catalogues.
Independent FULLPhi7 plus actual B/R/TrU/[H,B] occupied source clouds; sparse
count-gauge closure, full primitive weak-H/G and original source/current/C law.
"""
from pathlib import Path
import gc,json,resource,sys,time
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_compile import save
from rich_composite_geometry_20261015_cloud_probe import neutral_state
from rich_composite_geometry_20261015_dressing_basis import construct,source_coordinates
from rich_composite_geometry_20261015_dressing_operators import assemble
from rich_composite_geometry_20261015_dressing_science import analyze


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    if 'body_basis_snapshot' in config:
        data=load(config['body_basis_snapshot']);p=data['p'];primitive=data['primitive'];storage=data['storage'];labels=data['labels'];probes=data['probes'];generators=data['generators'];basis=data['basis_receipt'];reference_energy=data['reference_energy'];del data
        storage.relocate(Path(config['body_basis_snapshot']).resolve().parent)
        captured={Path(value).name for key,value in config.items() if key.endswith('_snapshot')}
        if not set(storage.paths.values())<=captured:raise ValueError('A resumed disk-body input must explicitly pin EVERY flat body block snapshot')
    else:
        data=load(config['neutral_basis_snapshot']);primitive=primitive_for(data);p=data['p'];coupling=load(config['neutral_coupling_snapshot']);eigen=np.load(config['neutral_states_snapshot'])
        omega=neutral_state(data,coupling,eigen['states'][:,0]);reference_energy=float(eigen['energies'][0]);del data,coupling,eigen;gc.collect();scalar,rotor=model.field_and_rotor(p)
        storage,labels,raw_probes,generators,basis=construct(primitive,omega,config,scalar,rotor,out);probes=source_coordinates(storage,raw_probes,len(labels),p['scalar_points']);del raw_probes,omega;gc.collect()
        save(out+'_body_basis.pkl',{'p':p,'primitive':primitive,'storage':storage,'labels':labels,'probes':probes,'generators':generators,'basis_receipt':basis,'reference_energy':reference_energy})
    scalar,rotor=model.field_and_rotor(p);operators,assembly=assemble(primitive,storage,labels,rotor,out);local_receipt=primitive.local.receipt();del storage,primitive;gc.collect()
    gram=operators['Gram'];metric_error=gram-sparse.eye(len(labels),format='csr');metric={'body_Gram_minus_I_Frobenius':float(np.linalg.norm(metric_error.data)),
        'count_metric_commutator':'Exactly within fixed physical16-count blocks by construction, with ALL actual Gram entries retained. Current/Ward measured after actual generalized metric operations; no assumedI.',
        'meaning':'Numerical body isometry plus FULL independentPhi7; not full-H range closure or all-pair/globalvacuum proof.'}
    result={'kind':'NEW_RICH_OPERATOR_DRESSED_OCCUPIED_CAR64_FIELD_COMPOSITE_CONDITIONAL_MODEL','parameters':p,'basis':basis,'assembly':assembly,'metric':metric,'local_receipt':local_receipt,
        'Noether':'Proper source tensors rankj on actual J0Omega, including vector sum_mu,i B_mu(2J_i)_(mu,m)R_i.d with creation index(mu,m), scalarcomplexTrU, and D=[H0,B] rankj. Spin-summed physical-count closure commutes J=Lleft+SCAR. HighestM representatives, both half copies and threehalf, no fitted S2/spin/baryon-hop penalty. Not full continuum translation/COM/Maxwell angular Noether.',
        'source_extension':'Each body column stores its actual operator lift to |phi_out><phi_in| P_counts G_source. D is literal UNMASKED [H0,B] cubic replacement/field operator at zero externalEM, not a state-only H-E0 extension. Count pieces complete the state gauge orbit, not a particle definition or an EOM leakage gate. Later [H,Oeff] MUST include [H,Pcounts]/field derivatives, never infer a unique operator from the state alone.',
        'quantum_field_scope':'ALL7 independentPhi output modes and ALL7 input slices of rich source/sea/R/PW profiles. PhiB/PphiB states included via full matrix-unit factor, NOT twolevel/lowband scalar. Full Hphi and mechanical/electric/plaquette/hop actions on actual R/U fields precede target projection; shared field coefficient/rest-of-sea dynamics retained. Additional severe source/sea/body-R/PW rank and anchor cuts explicit; primitiveR10/PW19 coordinate cuts remain.',
        'EOM_metric_distinction':'This is a fixed finite charged Hilbert/Galerkin weak-H/G model with actual Hermitian primitive-H matrix elements, NOT silently Hermitized source-EOM K. Independent evolving-sea operator EOM has Gdot=i(K-Kdag) and requires its connection; bare/Phi operator evidence and reviewer endpoints remain separate. No claimed generic dressed-R/U EOM/lifetime closure from this model.',
        'common_reference':'OWN selected correlated onehop quantum reference only, known full-H RMS5.905. This new charged representation does not repair/prove a global stationary vacuum. Same full primitive H and microscopic physicalQ source/contact/C; no inherited ColorSea/other owner spectrum or additive mass/recenter. Negative scalar quadratic/potential, charges/gauging/3+1 split, field volume and pointmass density are INPUTS.',
        'limits':'Conditional finite geometry/source family. C/F occupation channels not free/asymptotic breakup thresholds, tagged-added-particle positions, EMcharge or colourCasimir. Changed quark-core/sea/field channels allowed, rigidOmega return is NOT particle definition. Actual source strengths/residues distinct from probabilities. No proton/measured mass/Lorentz/freeCOM/photon/globalspin/whole-issue claim; no native catalogue restart.'}
    Path(out+'_model_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n');result.update(analyze(operators,labels,probes,scalar,p,reference_energy,config,out));result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
