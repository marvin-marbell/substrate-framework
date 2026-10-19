"""Evidence-bound <=80-line physical synthesis; no inferred convergence or poles."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_picture_20261016_'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load(name):return json.loads((ROOT/(P+name+'.json')).read_text())


def main():
    a=load('field1_result');b=load('field2_result');bare=load('bare_result');source=load('source_probe_result');matched=load('matched_ray_result');analysis=load('analysis_result');images=load('images_result');language=load('language_final')
    for data in (a,b):
        if set(data['cases'])!={'neutral','addition','removal'} or set(data['charged_minus_neutral'])!={'addition','removal'}:raise ValueError('incomplete physical case')
    if len(images)!=6 or any(set(v)!={'field','angular'} for v in images.values()):raise ValueError('incomplete primitive images')
    lines=['EXACT UNBOUNDED ELECTRIC INTERACTION PICTURE: ACTUAL SAME-H QUANTUM CASE',
    'PHYSICS UNCHANGED: frozen48 CAR8 full N4/N3/N5, quantum g/R/positive r5, IR3/If2.',
    'Eight real noncompact modes: nonparallel qin/qout, both T and cos/sin partners; all active.',
    'Same rotating ions/microsites/net Q=ions-fermions, actual straight Peierls/Wilson paths.',
    'No bare Maxwell B², omega²a², Pauli B, new couplings or body/field freeze.',
    'H0=sumPi²/(2If) on full L2; U0=exp(-itH0), protocol basis U0(t)W|initial Hermite label>.',
    'Initial label cuts1/2 are9/45 coefficient functions, NOT hard physical photon occupations at t.',
    'H_I^W=Wdag U0dag(H-H0)U0W; b=a+t(Pi+impulse)/If; Pi_I=Pi+impulse.',
    'Exact Weyl alpha=(-t/If+i)c/sqrt2 and BCH phase exp[i(t/If)c.impulse] BEFORE projection.',
    'All Lambda² and b_l b_m primitives/products precede projection; no square of cut a/Pi.',
    'H0 remains in basis time derivative and full physical H energy=H_I+H0; H_I energy not conserved.',
    'Actual source/current/contact/receiver operators use ABSOLUTE time and own protocol basis.',
    'Different-protocol overlaps use full Wleftdag Wright, not coefficient-vdot amplitudes.',
    'Fresh UNKICKED initial source rays equal old48 rays at each respective initial cut; no old state replay.',
    'New main fullW kick is a DIFFERENT physical pulse ray from old exp(i eps PaP); keep attribution separate.',
    'FIELD-DECOUPLED/HOPPING-DISABLED control retains R kinetic and rotating charges, not position freeze.',
    'Its source derivative0 follows decoupling, not a physical lab continuity/current-zero claim.']
    for cut,r in bare.items():
        lines.append(f"FREE cut{cut}: actual HG Gram{r['Gram_error']:.3g}, Weyl quad{r['Weyl_quadrature_error']:.3g}, overlap{r['protocol_overlap_error']:.3g}.")
        lines.append(f" fullkick work error{r['kick_work_error']:.3g}, otherPi invariance{r['other_mode_Pi_initial_invariance']:.3g}; wrongcut kick crossPi{r['wrong_cut_kick_other_mode_change']:.6g}.")
        lines.append(f" wrongcut freePi drift{r['wrong_cut_free_Pi6_drift']:.6g}, wronguntransformed q6 error{r['wrong_untransformed_q6_error']:.6g}.")
    for cut,r in analysis['free_full_covariance'].items():lines.append(f"FREE full8x8 covariance cut{cut}: aa{r['full_aa_covariance_quadrature_error']:.3g}, PiPi{r['full_PiPi_covariance_quadrature_error']:.3g}, sym aPi{r['full_sym_aPi_covariance_quadrature_error']:.3g}.")
    lines.append(f"SOURCE absolute-time fullH0 commutator error<={max(r['time_derivative_full_H0_commutator_error'] for r in source.values()):.3g}; wrongcut error>={min(r['WRONG_cut_H0_commutator_error'] for r in source.values()):.3g}.")
    lines.append(f"SOURCE physicalR derivative error<={max(max(r['R_derivative_errors']) for r in source.values()):.3g}; wrongtime0 source fails.")
    lines.extend(['THREE CLAIMS: old finite-PaP protocol/calibrated contrast remains valid within its regulated model.',
    f" ExactU0 on SAME old finitekick ray: Piout initial{matched['initial_Pi6_finite_kick_shift']:.8g}; coupled temporal{matched['old_ray_temporal_full_minus_decoupled']['Pi'][6]:.8g} (old hardcut~1.3314684e-4).",
    f" NewfullW ray: initialcrossPi{matched['initial_Pi6_full_W_shift']:.3g}, coupled temporal{matched['full_W_temporal_full_minus_decoupled']['Pi'][6]:.8g}.",
    'On this matched CHARGED ray, large reduction is mainly changed physical preparation, NOT exact free propagation alone.',
    'Smaller genuine fullW coupling is not failure; bare temporal echo disappears exactly without deleting H0.'])
    for cut,data in [(1,a),(2,b)]:
        lines.append(f"ACTUAL cut{cut}: t=.04 dt=.02, fresh neutral/bothcharges/bothT; fullW field-only kick.23.")
        for name,c in data['cases'].items():
            lines.append(f" {name} dim{c['dimension']}: sameH(t0) error{c['t0_same_physical_H_action_error']:.3g}; physical energy drift{c['baseline']['physical_energy']-c['initial']['physical_energy']:.6g}.")
        for name in ['addition','removal']:
            for axis,out in [('0',6),('2',4)]:
                v=data['charged_minus_neutral'][name][axis]['Pi'][out];off=data['cases'][name]['protocols'][axis]['decoupled_temporal']['Pi'][out]
                lines.append(f" {name} {axis}->{out}: temporal charged-minus-neutral/fullminusdecoupled Pi={v:.9g}; OFFPi={off:.3g}.")
    dt=a['cases']['addition']['physical_dt_comparison']
    lines.append(f"PHYSICAL dt.02->.01: Piout response change{dt['Pi_out6_change']:.6g}, Jout change{dt['current_out6_change']:.6g}, density max{dt['density_max_change']:.6g}.")
    lines.append('Both current/electric time orders, retarded2Im C, full contact and spin-current legs retained at absolute0/.04 (cut1).')
    for name,r in analysis['initial_cut_ray_comparison'].items():lines.append(f"INITIAL RAY cut1vs2 {name}: squaredoverlap{r['input']['squared_overlap']:.9g}; source weights/tails kept distinct from evolution.")
    for name,r in images.items():
        lines.append(f"IMAGE {name}: HI outgoing field>={r['field']['HI_outgoing_lower_bound']:.6g}, angular>={r['angular']['HI_outgoing_lower_bound']:.6g}; HI compression{max(r[k]['HI_compression_error'] for k in r):.3g}.")
    lines.extend(['Full-H image, H0 basis-connection image and HI residual stored SEPARATELY: H0 outside labels is NOT cut error.',
    'Streamed full Weyl generating functions retained; finite shells are lower bounds, not full infinite exterior equality.',
    'Interacting canonical-debt probes retained; bare echo absence does NOT establish interacting canonical closure.',
    'Field1/2 response and initial-state differences are observed, not a promised convergence improvement.',
    f"Final selected Pyright exit{language['natural_exit']}: {language['diagnostics']['summary']}; rooted LSP exit{language['rooted_LSP']['natural_exit']} plus positive AST.",
    'Natural exits/source capsules/failures/raw arrays/initial source weights and physical detector conventions manifest-bound.',
    'No irreducible stiffness, photon pole, fullH/ground/thermal convergence, Lorentz mass/g2/positron/Nature claim.'])
    if len(lines)>80:raise ValueError(f'synthesis length {len(lines)} exceeds80')
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    (ROOT/(P+'attempts.json')).write_text(json.dumps({f.name:json.loads(f.read_text()) for f in ROOT.glob(P+'*_exit.json')},indent=2)+'\n')
    interface=dict(fields=dict(file=P+'fields.py',sha256=sha(ROOT/(P+'fields.py'))),model=dict(file=P+'model.py',sha256=sha(ROOT/(P+'model.py'))),protocol='U0(t)W fullL2; apply=HI notphysicalenergy; alloperatorsabsolutetime; physicalenergy addsH0; truebasisoverlap required',frozen_H=dict(file='rich_spatial_frame_photon_20261016_model.py',sha256=sha(ROOT/'rich_spatial_frame_photon_20261016_model.py')))
    (ROOT/(P+'interface.json')).write_text(json.dumps(interface,indent=2)+'\n')
    files=sorted(f for f in ROOT.glob(P+'*') if f.is_file() and not f.name.endswith(('_manifest.json','_delivery_capsule.zip','_delivery_receipt.json')))
    files+=sorted(ROOT.glob('rich_spatial_frame_photon_20261016_*.py'))+[ROOT/'rich_gauge_band_frame_20261016_model.py',ROOT/'rich_local_frame_spatial_carrier_20261016_angular.py',ROOT/'rich_frame_gauss_20261014_model.py',ROOT/'rich_coupled_frame_photon_20261016_language.py']
    manifest=dict(kind='SAME_H_EXACT_UNBOUNDED_ELECTRIC_IP_ACTUAL_PHYSICS',synthesis_lines=len(lines),files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files})
    mp=ROOT/(P+'manifest.json');mp.write_text(json.dumps(manifest,indent=2)+'\n');zp=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for file in files+[mp]:z.write(file,file.name,compress_type=zipfile.ZIP_STORED if file.suffix in ('.npz','.zip') else zipfile.ZIP_DEFLATED)
    receipt=dict(capsule=zp.name,sha256=sha(zp),bytes=zp.stat().st_size,manifest_sha256=sha(mp),synthesis_lines=len(lines))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
