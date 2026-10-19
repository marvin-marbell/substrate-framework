"""Compact physical synthesis, exact lineage manifest and private evidence capsule."""
import hashlib
import json
from pathlib import Path
import platform
import zipfile
import numpy as np
import scipy
import numba
ROOT=Path(__file__).resolve().parent
P='rich_gauss_projected_photon_20261016_'


def load(name):return json.loads((ROOT/(P+name+'.json')).read_text())


def main():
    result=load('result');exact=load('exact_result');cor=load('correlation_result');language=load('language_final')
    first=result['first']['quality'];second=result['second']['quality'];responses=result['responses']
    loops={n:[x['value'] for x in cor['loop_coherence'] if x['length']==n] for n in (2,4)}
    exits={label:load(label+'_exit') for label in ('attempt1','attempt2','attempt3','attempt4','attempt5','exact1','exact2','observe1','observe2')}
    lines=[
      'Issue222 finite coherent gauge-projected photon candidate; 2026-10-16.',
      'Law unchanged: periodic2x2x2,8 fullCAR4 sites,24 DISTINCT link3 arcs,Nup=Ndown4,ions1,t1,U4,I16,Ecap1.',
      'No bare magnetic/Maxwell term, fixed continuum kernel, contact fit or configuration-list state.',
      'State amplitude delta_G detF(up,down)*CAR_interleaving_sign*exp(-gD-a sumE2-b sum(Ea+Eb)2).',
      'F is cubic closed-shell singlet pair matrix; all0..4 particle-hole pairs and all legal field configurations retained.',
      'Symmetry-imposed determinant nodes are physical: full local support does not mean every global coefficient nonzero.',
      'Exact cube contraction sums4900 matter configs,1107 rho patterns,3125 free netflows per rho.',
      'Seven tree flows solve Gauss exactly;12 pair kernels each sum9 physical twoarc states; no3^24 enumeration.',
      f"Base Slater exact E={exact['results'][0]['energy']:.12f}; norm={exact['results'][0]['norm']:.8f}.",
      f"Exact base0..4pair probabilities={exact['results'][0]['pair_sector_weights']}.",
      f"Undressed genuine pair/correlatedfield exact E={exact['results'][1]['energy']:.12f}: by itself WORSE.",
      'Material second change: [1-.10(H+10)] on g.45,eta.08,a.15,b.08 pair/coherent state, actual fullH CAR image.',
      f"Base16000sample E={first['energy']:.8f} blockSE={first['energy_block_SE']:.6f}; fullHres estimator={first['full_H_residual_estimator']:.6f}.",
      f"Dressed16000sample E={second['energy']:.8f} blockSE={second['energy_block_SE']:.6f}; fullHres estimator={second['full_H_residual_estimator']:.6f}.",
      f"Dressedres2={second['full_H_residual_squared_estimator']:.6f} blockSE={second['residual_squared_block_SE']:.6f}; genuine improvement not wider-support proxy.",
      'OldCI E-11.65032/res2.92448 and tensorloopgas E-10.75589/res3.41638 are frozen contextual references, not replayed inputs.',
      'Four independent seeds713,1734,2755,3776; burn10000,thin15,4000samples each; epsilon.2 positive guide.',
      'Guide=abs(psi)^2+.04 exp(-2gD)w(E)^2 samples exact nodes; numerator |Hpsi|2 includes omitted-nodal support.',
      'No claimed exact dressed norm/H2; blockSE is not an autocorrelation proof and finite chains can retain mixing bias.',
      f"Dressed doublons={second['doublons']:.6f},sumE2={second['electric_squared']:.6f},4pairweight={second['four_pair_weight']:.7f}.",
      f"MaxGauss sample residual={result['second']['sample_max_Gauss']}; connected cov(D,sumE2)={cor['doublon_E2_connected']:.6f}.",
      'Symmetric fixed primitive hops + opposite-spin swaps + plaquettes AND torus windings; invalid proposals self-reject.',
      f"Independent fullCAR/Gauss twovertex/twoarc benchmark dim10: matrix error={result['small_benchmark']['full_tensor_vs_CAR_entry_error']}, wrongunsigned error2.",
      f"Small fixedproposal detailedbalance error={result['small_benchmark']['detailed_balance_error']:.3g}; nodal localvar0 but actualres2=4.",
      f"Selected amplitude translations/axispermutations incl CAR sign relativeerror<={max(x['relative_to_max_amplitude'] for x in cor['symmetry'].values()):.3g}.",
      f"Selected S+ annihilation relativeerror={cor['spin_raising_relative']:.3g}: actual singlet probe,102raw states.",
      f"Full cycle dimension17; all fieldcov eigenvalues positive [{min(cor['cycle_field_eigenvalues']):.7f},{max(cor['cycle_field_eigenvalues']):.7f}], NOT17 photons.",
      f"Actual coherent Wilson cos insertions: length2 mean={np.mean(loops[2]):.6f}; length4 mean={np.mean(loops[4]):.6f}; neither is a magnetic Hterm.",
      'NEW normalized sources F=cE/16,J=-i[H,cE], [H,F]=iJ/16; correct24arc wrap phases and transverse incidence.',
      'Response each3200thinned samples; full connected4source Gram/H with field/current offdiagonal complex weights.',
      f"q100 Ritz shifted energies={responses[0]['ritz_shifted_energies']}; contactdiag={np.diag(responses[0]['contact']).tolist()}.",
      f"q110 Ritz shifted energies={responses[1]['ritz_shifted_energies']}; contactdiag={np.diag(responses[1]['contact']).tolist()}.",
      f"q100 fieldGramdiag={np.diag(responses[0]['field_gram']).tolist()}; q110={np.diag(responses[1]['field_gram']).tolist()}.",
      'q110 unequal T formfactors are a lattice discriminator, not isotropic continuum-photon confirmation.',
      f"Unsymmetrized shiftedH errors q100={responses[0]['unhermitized_shifted_H_error']:.5f},q110={responses[1]['unhermitized_shifted_H_error']:.5f}; retained as sampling limitations.",
      'Correct mixed susceptibility: W/(omega-z)+WT/(omega+z),z=externalfrequency+.1i; positive staticchi for coupling-fO.',
      'Actual twist response=C-chiJJ for H(A)=H+AJ+.5ACA; contact separately retained, never counted as interaction twice.',
      'Attempt3 scalar mixedweight kernel and opposite twist sign SUPERSEDED; original arrays frozen in dedicated zip.',
      'Finite4source Ritz response is NOT stationary fullH Lehmann poles: residual1.34 remains substantial.',
      'No derived massless law, continuum/regulator removal, unique photon, or full particle content claim.',
      'Independent Review33 used own two fresh chains and ownCAR current/H2: E-12.16449/-12.07675,var1.80562/1.66451.',
      'Review33 bounded findings support improvement; its evolved arrays are NOT producer inputs or part of this state.',
      'Amplitude/state interface and selected parameter file are directly callable; field arrays stored only as measurement records.',
      'Executable source/input/results/naturalexits/raw response arrays/sourcecapsules/manifest are all prefix owned.',
      f"Actual science natural exits={ {label:entry['science_child_natural_exit'] for label,entry in exits.items()} }.",
      'First failed Numba generator closure; second failed float(bool); no runtime/environment downgrade; failure evidence retained.',
      f"Final selectedPyright exit={language['natural_exit']},summary={language['diagnostics'].get('summary')}; rootedLSP exit={language['rooted_LSP']['natural_exit']}.",
      'Actual rootedLSP symbols/definition and positiveAST exercised; diagnostic JSON contains exact filenames and every finding.',
      'Reproduce: baselinePython rich_gauss_projected_photon_20261016_capture.py chosenlabel; exactlabel exact; observelabel observe.',
      'Durable shared plan remains parent owned; no ancestor edits, graph promotion, public publication or permission gate.'
    ]
    assert len(lines)<=80
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    summary=dict(kind='ACTUAL_FINITE_COHERENT_GAUSS_STATE_AND_NONSTATIONARY_RESPONSE',selected=load('selected_state'),
                 exact_base_results=exact,quality=dict(base=first,dressed=second),responses=responses,correlation_summary=cor,
                 language_summary=language['diagnostics'].get('summary'),actual_science_exits={k:v['science_child_natural_exit'] for k,v in exits.items()},
                 synthesis_lines=len(lines),governor=dict(plan='issue222-rich-backward-candidates-20261014',revision='58589f21e86de29c925ce31215357cb25d8cde838ffebb9d53d8a38bc3f478d9',work_item='photon-candidate'))
    (ROOT/(P+'synthesis.json')).write_text(json.dumps(summary,indent=2)+'\n')
    excluded={P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'}
    paths=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in excluded)
    paths.append(ROOT/'rich_periodic_photon_20261015_model.py')
    manifest=dict(files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths},
                  environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__),
                  lineage='Only frozen pure geometry/source convention imported. No ancestor CLI or evolved output replay. Own selected raw arrays consumed only for declared measurement.',
                  reproduction_commands=['baselinePython '+P+'capture.py reproduction','baselinePython '+P+'capture.py exact_reproduction exact','baselinePython '+P+'capture.py correlation_reproduction observe'])
    manifestpath=ROOT/(P+'manifest.json');manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths+[manifestpath]:z.write(path,path.name)
    receipt=dict(capsule=capsule.name,sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),bytes=capsule.stat().st_size,
                 files=len(paths)+1,synthesis_lines=len(lines),science_exits=summary['actual_science_exits'])
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
