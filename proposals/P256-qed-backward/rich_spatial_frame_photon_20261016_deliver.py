"""Bounded physics synthesis, immutable attempts, raw arrays and source capsule."""
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any
ROOT=Path(__file__).resolve().parent
P='rich_spatial_frame_photon_20261016_'


def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    actual=json.loads((ROOT/(P+'actual1_result.json')).read_text())
    paired=json.loads((ROOT/(P+'paired_result.json')).read_text())
    images=json.loads((ROOT/(P+'images_result.json')).read_text())
    language=json.loads((ROOT/(P+'language_final.json')).read_text())
    contrast=json.loads((ROOT/(P+'contrast_result.json')).read_text())
    cubature=json.loads((ROOT/(P+'quadcheck_result.json')).read_text())
    if set(images)!={'neutral','addition','removal'}:raise ValueError('incomplete image delivery')
    lines=['48 ACTUAL NEW SPATIAL FINITE-q / FULL QUANTUM GLOBAL-FRAME CAVITY CASE',
    'LAW: CAR8 complete N3/N4/N5, quantum SU2 g, material SO3 R and positive matrixnorm r.',
    'Positions x_s=R(site_s-.5), all8 microsites; ions rotate too, Q_s=ions_s-n_s.',
    'Eight real canonical a/Pi: qin=(0,0,.7), qout=(.6,0,.2), two T and cos/sin each.',
    'Hv=HE+Hr+Hg+root*r*beta+Hhop[a]+sum(L_R-a.Lambda)^2/(2IR)+sum Pi²/(2If).',
    'IR3, If2; no Maxwell B², supplied omega²a², Pauli B, bare Dirac or classical-only field.',
    'Exact straight plane-path Peierls/Hermite displacement BEFORE R/field compression.',
    'Lambda=sum Q x cross f, including ionic convection; Lambda², Pi² and contacts before cut.',
    'PZW: F=int0^xs f.dl, P=sum QF, U=exp(i a.P); Udag Hv U has Pi+P,',
    'edge residual phi-(Fj-Fi) and mechanical residual Lambda-d_R P. Not all finite-q A is gauge.',
    'q0: residuals vanish, H40+sum(Pi_l+e_l.D_Q)²/(2If);44 dipole mechanism, not identical8-mode/isotropic3-mode spectra.',
    f"Pure shared kernel current SHA {sha(ROOT/(P+'kernel.py'))}; early d868d38 revision physically identical.",
    '47 imports pure geometry/Hermite/cubature only;48 imports47 exact Gaunt helper only, never its state/H.',
    'New model full Fock sectors, angular complete multiplets, radial5; all8 field coordinates active.',
    'Primary finite total field occupation<=1 is SEVERE regulator; oscillator basis is not oscillator law.',
    'Corrected independent24alpha(2pi)/GL16beta/24gamma integration retains the SAME R multiplets.',
    'Old Band40 grid aliases finite-q source products: coarse runs/partialfield2 archived, not promoted.',
    f"Selected actual source blocks24vs32 error<={max(v['contact24vs32'] for v in cubature.values()):.3g}; Gaunt is exact retained-product factorization.",
    'Fresh neutral staggered/Haar g,R/cone Gaussian/field vacuum, propagated independently.',
    'Physical root-Wilson source exp(-i a.F_s) B40; all8 source weights/normalization retained.',
    'V=d_A Hv(a+A) at FIXED canonical a,Pi,L_R; current=-V, full Peierls+mechanical K.',
    'Current/electric both time orders, retarded2Im C, instantaneous contact on same source.',
    'Finite field-only unitary exp(i eps PaP) used; P exp(i eps a) P differs and discarded norm reported.',
    'Receiver E=sum f_l(x)Pi_l at origin and (.8,-.4,1.1); no bulk/on-shell claim.',
    'Fixed finite q set not SO3 closed: no exact totalJ; source/profile/receiver covariance family only.',
    'Lg is measured physical LEFT frame generator, not assigned extra spin or local spin PVM.']
    for name,b in actual['branches'].items():
        geom=b['geometry'];lines.extend([f"PRIMARY {name}: dimension{b['dimension']}, source weight sum{b['source_weight_sum']:.9g}; cost{b['source_energy_cost']:.9g} NOT inertial mass.",f" t=.08 loop<W>real={geom['actual_Wilson_real']:.9g}, wrong pure-gauge=1; norm defect{geom['physical_Wilson_defect']:.9g}.",f" actual loop flux RMS{geom['loop_flux_rms']:.9g}, residual R connection RMS{geom['residual_connection_rms']:.9g}.",f" source full-H connection{b['source_connections']['full_H']:.9g}; full r force{b['source_radial_force']:.9g}."])
    for label in ['paired','R2','field2','q0']:
        path=ROOT/(P+label+'_result.json')
        data=json.loads(path.read_text())
        if set(data['branches'])!={'addition','removal'}:raise ValueError('incomplete '+label)
        lines.append(f"FRESH {label}, t={data['t']}: neutral propagated; independent same-law charged packets.")
        for name,b in data['branches'].items():
            for ch,v in b['protocols'].items():lines.append(f" {name} {ch}: delta Eout={v['qout_feedback']:.8g}; hopOFF={v['hopping_off_feedback']:.8g}; fieldDecoupledOFF={v['bare_control_feedback']:.8g}.")
    lines.extend(['Hopping OFF retains real rotating-charge convection; probes are calibrated to each own H.',
    'FIELD-DECOUPLED/HOPPING-DISABLED: R kinetic and rotating charges persist; zero d_AH is decoupling, NOT charge-position freeze/current continuity.',
    'Paired0->6 and2->4 are copolarized nonparallel channels; primary0->4 is crosspolarized.',
    'Source E/g/r/root/R/hopping/field commutator images, spin-current legs and contact retained in paired rawarrays.'])
    debt=paired['branches']['addition']['canonical_debt']['0']
    lines.append(f"CAUTION cut1: norm Pi-dot-minus-source {debt['Pi_dot_minus_source_norm']:.7g}, a-dot-minus-Pi/I {debt['a_dot_minus_Pi_over_I_norm']:.7g}.")
    lines.append('Raw nonforward response is DOMINATED by field-cut echo; paired fullminusOFF is the matter-coupled contrast, not a photon.')
    for label,data in contrast.items():
        v=data['branches']['addition']['0']['charged_minus_neutral_full_minus_OFF']['Pi'][6]
        lines.append(f"MATCHED {label}: addition-minus-neutral temporal fullminusOFF Pi_out6={v:.8g}; all8Pi/a/n/LEFT retained.")
    qsignal=contrast['paired']['branches']['addition']['0']['charged_minus_neutral_full_minus_OFF']['Pi'][6]-contrast['q0']['branches']['addition']['0']['charged_minus_neutral_full_minus_OFF']['Pi'][6]
    lines.append(f"MATCHED q1-minus-q0 calibrated source increment {qsignal:.8g}; reducible finite response, NOT irreducible stiffness/pole.")
    for name,r in images.items():lines.append(f"IMAGE {name} paired t={r['time']}: expanded omitted norm lowerbound{r['omitted_image_lower_bound']:.8g}; compression discrepancy{r['compression_identity_error']:.3g}.")
    lines.extend(['Finite-q Peierls has infinite R/field image: expanded-shell norms are lower bounds, NOT full exterior equality.',
    'Exact unbounded Hermite/plane generating functions and primitive products retained; no convergence assertion.',
    'Full field-cut/R-cut controls and continuum kick comparator expose regulator gaps/echoes; do not label these photons.',
    f"Final selected typing exit{language['natural_exit']}, {language['diagnostics']['summary']}; rooted LSP exit{language['rooted_LSP']['natural_exit']}.",
    'Attempt1 real-input/complex-output casting failure retained with source capsule; corrected dtype only.',
    'Initial34 selected typing errors/diagnostics preserved; type-only shared-source changes explicitly signaled to47/49.',
    'Coarse image compression error .05335 exposed cubature defect; corrected images use same24-node-family H.',
    'Field2 SIGKILL-9 retained (OOM only suspected); exact full-product contraction removed3.284GB temporary, no state cut.',
    'All natural exits, exact source capsules, inputs, raw arrays, failed attempts and final hashes are manifest-bound.',
    'No equilibrium, stable electron, Lorentz/inertial mass, bulk/on-shell photon, local Noether transport or Nature claim.'])
    if len(lines)>80:raise ValueError('physics synthesis exceeds80 lines')
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    attempts={f.name:json.loads(f.read_text()) for f in [*ROOT.glob(P+'*_exit.json'),*ROOT.glob(P+'*_termination.json')]}
    (ROOT/(P+'attempts.json')).write_text(json.dumps(attempts,indent=2)+'\n')
    interface: dict[str,Any]
    interface=dict(kernel=dict(file=P+'kernel.py',sha256=sha(ROOT/(P+'kernel.py'))),fields=dict(file=P+'fields.py',sha256=sha(ROOT/(P+'fields.py'))),model=dict(file=P+'model.py',sha256=sha(ROOT/(P+'model.py'))),position_semantics='kernel geometry uses physical LEFT delta x=axis cross x, not neutral texture derivative',state_ownership='48 own actual global spatial law/state;47 independent combined consumer')
    interface['quadrature']=dict(file=P+'quadrature.py',sha256=sha(ROOT/(P+'quadrature.py')),azimuth=24,polar=16)
    interface['pure_Gaunt_dependency']=dict(file='rich_local_frame_spatial_carrier_20261016_angular.py',sha256=sha(ROOT/'rich_local_frame_spatial_carrier_20261016_angular.py'),states_imported=False)
    (ROOT/(P+'interface.json')).write_text(json.dumps(interface,indent=2)+'\n')
    files=sorted(f for f in ROOT.glob(P+'*') if f.is_file() and not f.name.endswith(('_delivery_capsule.zip','_manifest.json','_delivery_receipt.json')))
    files.extend([ROOT/'rich_gauge_band_frame_20261016_model.py',ROOT/'rich_coupled_frame_photon_20261016_language.py',ROOT/'rich_local_frame_spatial_carrier_20261016_angular.py',ROOT/'rich_frame_gauss_20261014_model.py'])
    manifest=dict(kind='SPATIAL48_ACTUAL_AND_PURE_KERNEL_DELIVERY',files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files},synthesis_lines=len(lines),limitations='finite q set/cavity/angular/radial/field regulator; no convergence/on-shell/equilibrium claim')
    mp=ROOT/(P+'manifest.json');mp.write_text(json.dumps(manifest,indent=2)+'\n')
    zp=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files+[mp]:z.write(f,f.name,compress_type=zipfile.ZIP_STORED if f.suffix in ('.npz','.zip') else zipfile.ZIP_DEFLATED)
    receipt=dict(capsule=zp.name,sha256=sha(zp),bytes=zp.stat().st_size,manifest_sha256=sha(mp),kernel_sha256=interface['kernel']['sha256'],synthesis_lines=len(lines))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
