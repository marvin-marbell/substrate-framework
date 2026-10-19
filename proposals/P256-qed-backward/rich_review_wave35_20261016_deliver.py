"""Assemble observed own physics, raw-array diagnostics and exact source/exit capsule."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import sys
import numba
from typing import Any
from rich_review_wave35_20261016_consume import logguide
ROOT=Path(__file__).resolve().parent
P='rich_review_wave35_20261016_'


def load(name):return json.loads((ROOT/(P+name)).read_text())


def compact(values):return '/'.join(f'{v:.6g}' for v in values)


def main():
    singlet=load('findings.json');af=load('AF_findings.json');dimer=load('dimer_findings.json');control=load('controls_findings.json');geometry=load('geometry_findings.json');clamped=load('clamped_findings.json')
    derived=[]
    for family,data in (('singlet',singlet),('AF_projection_unusable',af),('af_dimer',dimer)):
        arrays=np.load(ROOT/(P+dict(singlet='declared_input_arrays.npz',AF_projection_unusable='AF_declared_input_arrays.npz',af_dimer='dimer_declared_input_arrays.npz')[family]))
        config=data['input']
        for chain in data['chains']:
            raw=np.load(ROOT/(P+f"chain{chain['seed']}_raw.npz"));up=raw['up'];down=raw['down'];E=raw['E'];v=raw['vectors'];w=raw['normalizer'];norm=w.sum()
            D=np.array([len(np.intersect1d(u,d)) for u,d in zip(up,down)])
            # Operator adjoint J(-q)=J(q)^dagger is NOT coordinate-vector conjugation:
            # on real psi, imaginary CAR current gives J(-q)psi=-conj(J(q)psi).
            residual=dict(family=family,seed=chain['seed'],D_observed_min=int(D.min()),D_observed_max=int(D.max()),Ecap_occupancy=float(np.einsum('n,ne->',w,E*E)/(192*norm)),q_minus_field_reality_residual=float(np.max(abs(v[:,:2,:2].conj()-v[:,2:,:2]))),q_minus_current_time_reversal_residual=float(np.max(abs(v[:,:2,2:].conj()+v[:,2:,2:]))))
            sqrtguide=np.exp(.5*np.array([logguide(u,d,field,arrays['F'],arrays['G'],config['params'],config['epsilon'],config['floor']) for u,d,field in zip(up,down,E)]))
            hscale=(np.abs(raw['Hpsi'])+np.abs(raw['psi']))*sqrtguide+1e-100
            residual.update(continuity_relative_max=float(np.max(raw['checks'][:,1]/hscale)),wrong_sign_continuity_relative_max=float(np.max(raw['checks'][:,2]/hscale)),HF_commutator_relative_max=float(np.max(raw['checks'][:,3]/hscale)))
            derived.append(residual)
    (ROOT/(P+'derived_diagnostics.json')).write_text(json.dumps(derived,indent=2)+'\n')
    lines=[
        'REVIEW35: OWN V64 MATTER/FIELD PHYSICS, NOT PHOTON OR GROUND-STATE ADMISSION',
        'Contract38C0; shared plan issue222-rich-backward-candidates-20261014 captured1ffd4101.',
        'Verdict: L2 alias resolved; real quantum currents survive, but no stationary photon pole established.',
        'Own independent canonical-CAR operator contractions and fresh chains, not producer CLI/trajectories.',
        'Declared owner pair kernels are amplitude INPUTS, not independently solved vacua.',
        '',
        'GEOMETRY AND PLAUSIBLE WRONG IMPLEMENTATIONS',
        'L4 has64sites,192distinct positive arcs,48wrap arcs; incidence rank63/cycle rank129.',
        'Cycle/source/polarization counts are geometry, never particle or species counts.',
        '128-mode CAR uses Python integers with explicit int indices; actual orbital127->31 hop survives, uint64 mask fails.',
        'That high-index hop preserves neutral Gauss; pair determinant and hop signs share the interleaved ordering.',
        'Physical +half-link midpoints give two-T divergence<=1.2e-16 and q-minus conjugation0.',
        'Actual clamped one-body CAR current Frobenius L2/L4='+compact([x['current_frobenius'] for x in clamped])+'.',
        'Actual full quantum CAR/Gauss ring ground J-norm2 L2/L4='+compact([x['current_ground_norm2'] for x in geometry['rings']])+'.',
        'Thus L2 sin(k)=0 is NOT actual quantum-U current zero; L4 adds +/-pi/2 band momenta.',
        'Exact ring continuity residual0; wrong source sign and omitted JW both fail actual operators.',
        'Separate complex positive/negative mixed residues matter: scalar shortcut error~.019 at L4.',
        'Caught owner incidence reversed relative to primitive joint hop; owner fixed before selected measurements.',
        'V64 direct determinants vs inverse sums: max H/J/contact relative errors='+compact([max(x[k] for x in control['direct_determinant_controls']) for k in ('H_relative','J_relative','contact_relative')])+'.',
        'Actual nonzero-config opposite J differs byrelative2; dropping CAR parity changes H byup torelative30.',
        'Exact V64 dimer-node control psi0,Hpsi2: positive guide retainsH2=4; null-guide omission loses4.',
        'All-Neel is a genuine current blindpoint; initial control failure retained, nonzero configurations used.',
        '',
        'ACTUAL SAME-LAW SINGLET STATE: TWO FRESH POSITIVE-GUIDE CHAINS',
        'Primitive t1/U4/I16/Ecap1, neutralN64 withNup=down32; no magnetic/B/Maxwell term added.',
        'Number-projected symmetric BCS pair determinant; all20 zero-shell modes f=1, no arbitrary filled subset.',
        'delta.8,g.45,a.15,b.08 are state parameters; local extensive Jastrow, no global one-H-image dressing.',
        'Pairing orbital parameter does not demonstrate a physical superconducting phase.',
    ]
    for family,data in (('Singlet',singlet),('AF-dimer',dimer)):
        cs=data['chains'];lines.extend([
            family+' seeds='+','.join(str(c['seed']) for c in cs)+';800samples/chain,burn16000,thin80.',
            family+' E='+compact([c['energy'] for c in cs])+'; blockSE='+compact([c['energy_block_SE'] for c in cs])+'.',
            family+' E/64='+compact([c['energy_per_cell'] for c in cs])+'; full-H variance='+compact([c['full_H_variance'] for c in cs])+'.',
            family+' importanceESS='+compact([c['importance_ESS'] for c in cs])+' of800; NOT autocorrelation-corrected ESS.',
        ])
        for k,label in ((0,'q(pi/2,0,0)'),(1,'q(pi/2,pi/2,0)')):
            lines.append(family+' '+label+' two-T field insertion Ritz='+ ';'.join(compact(c['q'][k]['field_source_Rayleigh_ritz']) for c in cs)+'.')
            lines.append(family+' '+label+' current outside E-source span fractions='+ ';'.join(compact(c['q'][k]['current_outside_field_fraction']) for c in cs)+'.')
    lines.extend([
        'These Ritz numbers are nonstationary variational source-insertion energies, NOT frequencies/poles.',
        'Negative/noisy chain-dependent insertion energies cannot be promoted to dispersion or law failure.',
        'Full current remains largely outside E span: scalar closure P=J would discard real source weight.',
        'Actual H Fpsi=F Hpsi+i Jpsi/16 checked against independently weighted neighbor F insertions.',
        'q-minus operator is adjoint; on real psi, field vector conjugates but current vector MINUS-conjugates.',
        'Physical continuity uses actual arc quantum currents and density commutator, not clamped sin(k).',
        'All saved Gauss residuals0; selected amplitudes agree direct canonical wrapper to reported roundoff.',
        'No literal floating-zero nodes sampled in singlet; analytic symmetry nodes remain, handled by positive guide.',
        'The guide has an actual constant positive floor; H2 is full |Hpsi|2/guide, never support-only local variance.',
        'All plaquettes and length4windings proposed symmetrically; accepted counts/raw configs retained.',
        'This is not a proof of full Markov-space mixing or all0..32 doublon sectors observed.',
        'AF projection fork unusable: own ESS='+compact([c['importance_ESS'] for c in af['chains']])+' of800, ill-conditioned minors='+','.join(str(c['unstable_determinants']) for c in af['chains'])+'.',
        'Its energy/Ritz arrays are preserved but NOT interpreted physically; material af_dimer fork actually executed.',
        'AF-dimer is oriented, noncubic fixedSz0 trial state; no total-spin-singlet or emergent isotropy claim.',
        'Ecap1 is a hard quantum-link regulator with boundary occupancy in derived diagnostics; no cutoff convergence.',
        'No full-H real-time propagation, ground theorem, continuum photon, Nature claim or independent charged-body join.',
        '',
        'REPRODUCIBILITY AND SCOPE',
        'Science exits0: geometry1,clamped1,singlet1,AF1,dimer1,controls3,highindex1; failures kept separately.',
        'controls1 current-blindpoint and controls2 JSON-int64 failures retained with exact exits and source capsules.',
        'Producer source changed during earlier runs; launch capsules and actual loaded F/G arrays preserved.',
        'Dimer consumes frozen declared F input; independent guide perturbation, seeds and operators remain ours.',
        'Rooted actual LSP symbols/definition/diagnostics and positive Python AST recorded; final selected diagnostics separate.',
        'Manifest/capsule include every own source,input,stdout/stderr,exit,failed attempt and raw physics array.',
    ])
    assert len(lines)<=80,len(lines)
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    paths=sorted(p for p in ROOT.glob(P+'*') if p.is_file() and p.name not in (P+'manifest.json',P+'delivery_capsule.zip',P+'delivery_receipt.json'))
    manifest: dict[str,Any]
    manifest=dict(kind='INDEPENDENT_FINITE_V64_REVIEW35_DELIVERY',summary_lines=len(lines),files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths})
    manifest['environment']=dict(python=sys.version,numpy=np.__version__,numba=numba.__version__,interpreter=sys.executable)
    mp=ROOT/(P+'manifest.json');mp.write_text(json.dumps(manifest,indent=2)+'\n')
    capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths+[mp]:z.write(p,p.name)
    receipt=dict(capsule=capsule.name,sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),bytes=capsule.stat().st_size,summary_lines=len(lines),files=len(paths)+1)
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
