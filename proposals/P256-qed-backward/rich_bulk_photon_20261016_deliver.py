"""Bounded physical synthesis and exact prefix-owned artifact capsule."""
import hashlib
import json
from pathlib import Path
import platform
import zipfile
import numpy as np
import numba
ROOT=Path(__file__).resolve().parent
P='rich_bulk_photon_20261016_'


def load(name):return json.loads((ROOT/(P+name+'.json')).read_text())


def main():
    primary=load('result')['results'];fork=load('dimer_result')['results'];results=dict(primary,**fork)
    language=load('language_final');exits={name:load(name+'_exit')['natural_exit'] for name in ('attempt1','attempt2','dimer1','uncertainty')}
    uncertainty=load('response_uncertainty')
    selected=min(('singlet','af_dimer'),key=lambda k:results[k]['quality']['energy'])
    selected_input=P+('dimer_input.json' if selected=='af_dimer' else 'input.json')
    selection=dict(family=selected,state_file=P+selected+'_state.npz',input_file=selected_input,parameters=json.loads((ROOT/selected_input).read_text())['state'],quality=results[selected]['quality'],interpretation='Lower measured energy of singlet and material fullrank AF-dimer trials; initial AF projection guide ESS too small for ranking; not a solved ground or photon')
    (ROOT/(P+'delivery_selected_state.json')).write_text(json.dumps(selection,indent=2)+'\n')
    lines=[
      'Issue222 actual L4 coherent matter/field response; finite failed-photon fork, 2026-10-16.',
      'Same law: 4x4x4 torus,64CAR4sites,192distinct link3arcs,Nup=Ndown32,ions1,t1/U4/I16/Ecap1.',
      'No bare Maxwell/magnetic term, frozen field, fitted light speed or virtual-pair-count restriction.',
      'Exact Gauss uses64component incidence vectors; no overflowing integer charge encoding.',
      'State=Gauss*CARparity*det F(up,down)*exp[-.45D-.15sumE2-.08sum(Ee-E_next)^2].',
      'Linked local factors repeat at every cell/link; NOT the L2 fixeddegree global(1-tauH) trick.',
      'Number-projected determinant has unrestricted pair sectors; no thermodynamic factorization theorem claimed.',
      'Singlet BCS delta.8 treats all20zero-shell modes identically f1; no arbitrary cubic half-shell filling.',
      'Pair/orbital kernels are STATE hypotheses, not physical condensates or new Hamiltonian terms.',
      'Material fork1: AF occupied-projector BCS kernel; pathological low-overlap guide retained, not a physical no-go.',
      'Material fork2: F=(Pup+.08Qup)T_x, a fullrank oriented AF-dimer state, not singlet/cubic vacuum.',
      'Every estimator sums actual CAR hops and clamped E moves;32x32inverse row/column ratios per hop.',
      'Illconditioned minors use direct neighbor determinants, never zero psi times an undefined ratio.',
      'Guide=envelope2*(detF2+.04detG2+1e-24), including exact state nodes; guide is not the physical state.',
      'Absolute fullH2 retains nodal support; local-energy variance alone is not used.',
      'Two independent chains per state,1500samples each,burn30000,thin80; checkpoints are separate64sample runs.',
      '30consecutiveblockSE is exploratory; importanceESS is NOT autocorrelationESS or a convergence proof.'
    ]
    for name in ('singlet','antiferromagnetic','af_dimer'):
        r=results[name];v=r['quality']
        lines.extend([f'{name}: E={v["energy"]:.7f} +/- {v["energy_blockSE"]:.5f}; E/site={v["energy_per_site"]:.7f}; fullHvar={v["full_H_variance"]:.5f}.',f'{name}: residual={v["residual"]:.5f}; importanceESS={v["importance_ESS"]:.1f}/3000; D={v["doublons"]:.4f}; sumE2={v["field_E2"]:.4f}; Gaussmax={r["maxGauss"]}.',f'{name}: observedD={v["doublon_observed_range"]}, not a paircut; unstableminor={v["unstable_inverse_samples"]}, retained H2={v["unstable_H2_contribution"]:.6f}.'])
    lines.extend([
      'Initial AF projection rows above are retained raw estimates, NOT reliable physics: importanceESS~5/3000.',
      'AF-dimer lowers measured energy versus singlet but worsens fullHvariance; not a stationarity improvement.',
      'Nonzero sin(k) geometry is now present; twoT measured at q=(pi/2,0,0) and q=(pi/2,pi/2,0).',
      'Actual midpoint/incidence-normalized F=cE/16,J=-i[H,cE]; HF=F H+iJ/16, fermionic rho response included.',
      'Spectral basis=[F1+,F2+,rho+,F1-,F2-,rho-]psi with exact allhop Himages, not named equal-time polarizations.',
      'J is independently evaluated; its measured component outside this source space is retained, never forced P=J.',
      'qminus F/rho=conj(qplus), but Jminus=-conj(Jplus); both momentum/frequency matrix weights retained.',
      'chi=Wplus/(omega-z)+Wminus_transpose/(omega+z); twist diagnostic=C-chi, no contact doublecount.',
      f'Selected finite trial={selected}; lower measured energy, not stationary vacuum or ground-state admission.'
    ])
    for iq,r in enumerate(results[selected]['responses']):
        lines.extend([f'Selected q{iq}: shifted source-Ritz energies='+','.join(f'{x:.6f}' for x in r['shifted_variational_energies'])+'.',f'Selected q{iq}: fullH mode-residual2 range={min(r["fullH_mode_residual_squared"]):.4f}..{max(r["fullH_mode_residual_squared"]):.4f}; antiHermitianMCdefect={r["shiftedH_antihermitian_relative"]:.5f}.',f'Selected q{iq}: Jnorm2T={r["current_norms"][:2]}; outsidefraction2T={r["current_outside_basis_fraction"][:2]}.',f'Selected q{iq}: contact2T={[r["contact_real"][j][j] for j in range(2)]}; negative shifted modes={r["negative_shifted_modes"]}.'])
        errors=uncertainty['results'][selected][iq]['sorted_shifted_Ritz']['bootstrap_std']
        lines.append(f'Selected q{iq}: exploratory sorted-gap blockbootstrapSE='+','.join(f'{x:.4f}' for x in errors)+'.')
    lines.extend([
      'These nonstationary finite variational modes have measured leakage/residual; they are NOT fullH Lehmann poles.',
      'Negative shifted estimates and large source residual prevent stable closedtwoT inference, not prove instability.',
      '120resample/30block bootstrap does not establish mixing; sorted Ritz mode identities can exchange.',
      'Current outside the E/rho source span can be physical; leakage rejects closure, NOT the state or primitive law.',
      'A coherent V64 quantum state and actual coupled field/density/current variational mechanism were executed.',
      'This is a measured failed/limited physical fork, NOT failure of the primitive law or all quantum current.',
      'No lowq massless dispersion, regulator removal, thermodynamic photon, species uniqueness or Nature claim.',
      'Plaquettes and length4windings remain allowed; finite chains do not prove global topological mixing.',
      f'Exact natural science exits={exits}; attempt1 Numba abs(array) frontend failure retained, no runtime downgrade.',
      'Successful science source/input hashes and unchanged-source flags are in exact exit records/sourcecapsules.',
      f'Final selected Pyright exit={language["natural_exit"]}, summary={language["diagnostics"]["summary"]}; rootedLSPexit={language["rooted_LSP"]["natural_exit"]}.',
      'Rooted LSP symbols/definition and positive Python AST exercised; exact filenames/findings retained.',
      'Analytic determinant nodes may appear as tiny floating minors; unstableminor is NOT an exact-node count.',
      'Raw configs, Hpsi,Hbasis, E/J/contact Gram, q+/- weights, frequency arrays separate in prefix NPZ files.',
      'Reproduce: baselinePython capture.py label; materialfork: capture.py label rich_bulk_photon_20261016_dimer_input.json.',
      'Parent owns plan/synthesis; no ancestor edit, producer replay, graph/public/settings/push/merge or permission gate.'
    ])
    assert len(lines)<=80
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    (ROOT/(P+'synthesis.json')).write_text(json.dumps(dict(selected=selection,results=results,natural_science_exits=exits,unpaid_limits=['nonstationarity','large fullH residual','current leakage','mixing','finiteV','Ecap regulator','no massless or Nature claim']),indent=2)+'\n')
    excluded={P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'}
    paths=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in excluded)
    entries={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
    manifest=dict(prefix=P,contract='38C0',plan='issue222-rich-backward-candidates-20261014',captured_revision='1ffd4101d044d4df2235654940f9cd7d6a60a0f5f20adff2af630ca39fe21f03',work_item='photon-candidate',python=platform.python_version(),numpy=np.__version__,numba=numba.__version__,files=entries,selected=selected,natural_exits=exits)
    mpath=ROOT/(P+'manifest.json');mpath.write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths+[mpath]:z.write(p,p.name)
    receipt=dict(capsule=capsule.name,bytes=capsule.stat().st_size,sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),manifest=mpath.name,summary_lines=len(lines),selected=selected)
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
