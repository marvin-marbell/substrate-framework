"""Bounded physics synthesis, immutable raw artifacts, hashes and honest limits."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
P='rich_local_frame_spatial_carrier_20261016_'


def read(name):return json.loads((ROOT/(P+name+'.json')).read_text())


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    result=read('result');analysis=read('analysis');language=read('language_final');build=read('build')
    add=result['trajectories']['addition'];rem=result['trajectories']['removal'];neutral=result['trajectories']['neutral']
    exits={f.name:json.loads(f.read_text()) for f in ROOT.glob(P+'*_exit.json')}
    def one(value):return json.dumps(value,separators=(',',':'))
    lines=[
      'NEW FOUR-CELL LOCAL NOETHER CARRIER + FINITE-q QUANTUM FIELD CONSUMER (2026-10-16)',
      'One fresh law/state; no old evolved state, global-uniform control polishing or imported48 spectrum.',
      'Physical x_s(R)=R[centre_v+.24(cube_s-.5)] for all32 microsites; four noncoplanar cells.',
      'Full CAR32 binary JW representation: N16/N15/N17 dimensions 601080390/565722720/565722720 before fields.',
      'Four INDEPENDENT g_v and positive r_v. Local central Z2 sewing ties (-1)^Nv to halfinteger frame parity.',
      'Primitive c scalars have NO assigned spin. Actual raw g LEFT differentials carry source halfspin.',
      'Polarized all-band Dband(g)c source uses the geometry-derived pi-Gamma algebra, not cube index as spin.',
      'Actual material R remains active: l0,1,2 multiplets; g j0,.5,1,1.5,2 complete multiplets; r3 Friedrichs/r^3dr.',
      'r is matrix amplitude, not particle radius. Cone/frame force and beta*r feedback are retained.',
      'Every microsite shell a=.08 and integer ionic charge retained; grounded-sphere Green Coulomb B20 solves longitudinal Gauss.',
      'Transverse profiles include the exact shell sinc(qa) form factor; boundary induced charge=-bulk net charge.',
      'Frozen48 pure kernel supplies exact finite-q paths and rotating-charge connection; no48 solved state used.',
      'Finite-q material cubature bound to48:24×GL16×24; exact retained-product Gaunt acceleration, no extra state/profile truncation.',
      'H compiled as exact lazy bounded-channel MPO SUM. Full graph dense upper bytes='+str(build['full_graph_storage_audit']['dense_upper_total_bytes'])+'; never allocated giant fullW.',
      'Evolution is symmetric grouped RK2 POLYNOMIAL splitting, not exact exponential; centering phases restored in physical tensors, norm metadata realpositive.',
      'Eight real fields: two polarizations*cos/sin at qin=(0,0,.7), qout=(.6,0,.2).',
      'No supplied Maxwell magnetic term, oscillator frequency, Pauli-B term, free-spin CAR projection or virtual-Fock removal.',
      'Peierls Hermite exponential and mechanical {L,Lambda}+Lambda^2 products evaluated BEFORE angular/field cut.',
      'Physical lab J=-dH/da includes ALL internal/intercell paths and ionic convection/contact; scalar edge twist is distinct.',
      'Regulator field total occupation<=1 has9 states; raw Q²/Pi² retain external intermediary occupation.',
      'This is a finite quantum-field consumer, NOT an emergent/on-shell/bulk photon or canonical cutoff closure claim.',
      'MPS mixed H includes norm metadata; no variance inferred from compressed image norm.',
      'Fresh neutral sea and both charged branches all evolved with SAME full H; neutral background not frozen.',
      'Source raw norms (not Born probabilities): '+one(result['source_raw_norm2']),
      'Times: '+one(result['times']),
      'Exact regulated source strength versus retained compressed source norm and eps: '+one(result['source_quality']),
      'Addition signed excess n_cells t0/final: '+one([add['excess_matched_background'][k]['n_cells'] for k in (0,-1)]),
      'Addition local LEFT signed excess t0/final: '+one([add['excess_matched_background'][k]['local_LEFT'] for k in (0,-1)]),
      'Removal signed excess n_cells t0/final: '+one([rem['excess_matched_background'][k]['n_cells'] for k in (0,-1)]),
      'Removal local LEFT signed excess t0/final: '+one([rem['excess_matched_background'][k]['local_LEFT'] for k in (0,-1)]),
      'Local Casimir addition t0/final: '+one([analysis['extra']['addition'][k]['local_LEFT_Casimir'] for k in (0,-1)]),
      'Carrier-to-field signed excess Q final: '+one(result['carrier_to_field'][-1]['field_Q']),
      'Field-only kick SAME-H carrier delta n final: '+one(result['field_to_carrier'][-1]['n_cells']),
      'Field-only kick SAME-H local LEFT delta final: '+one(result['field_to_carrier'][-1]['local_LEFT']),
      'Field-only kick SAME-H Q delta final: '+one(result['field_to_carrier'][-1]['field_Q']),
      'Kick is exp(i .28 P a0 P) in finite field space, NOT an exact unbounded single-mode Weyl preparation.',
      'Actual initial Pi kick-minus-charged-base all8modes: '+one((np.asarray(analysis['extra']['addition_field_kick'][0]['field_Pi'])-np.asarray(analysis['extra']['addition'][0]['field_Pi'])).tolist()),
      'Field-decoupled/hopping-OFF kick-to-carrier: '+one(analysis['field_decoupled_hopping_OFF']['carrier_field_kick_difference']),
      'Control RETAINS material R kinetic: physical x(R) still moves; zero d_AHcontrol is not a lab material-current-zero claim.',
      'Dynamical field kick response t-t0 FULL-minus-field-decoupled-control: '+one(analysis['dynamical_field_kick_calibrated']),
      'Dynamical carrier-to-field t-t0 FULL-minus-field-decoupled-control: '+one(analysis['dynamical_carrier_to_field_calibrated']),
      'Raw nonforward field means are NOT photon evidence: total-occupation cutoff can produce OFF regulator echo.',
      'Two nonparallel-q physical source joint Gram and mixed contact: '+one(analysis['physical_source_joint']),
      'Actual projected canonical source debts: '+one({a:{key:row[key] for key in ('actual_dPi_dt','raw_physical_source_expectation','dPi_minus_J','actual_dQ_dt','raw_Pi_over_M','dQ_minus_Pi_over_M')} for a,row in analysis['finite_cut_canonical_debt'].items()}),
      'Actual source-image mixed norm/phase scaling errors: '+one({key:analysis['mixed_norm_metric'][key] for key in ('scaling_error','phase_covariance_error','overlap_phase_error')}),
      'Bond12/20 same-input state distance²: '+str(result['same_input_chi12_20']['distance_squared']),
      'Actual dt.04/.02 local charge/LEFT/field differences: '+one(result['same_input_dt004_002']['physical_observable_differences']),
      'dt.04/.02 raw restored-phase versus phase-aligned state distance²: '+one([result['same_input_dt004_002']['raw_phase_restored_distance_squared'],result['same_input_dt004_002']['phase_aligned_distance_squared']]),
      'Actual outgoing individual g/field image norms: '+one(add['outgoing']),
      'Exterior total-H norm is NOT asserted; individual source/multiplier images are not a full complement theorem.',
      'All disclosed MPS image discards and RK2 normalization defects remain in raw result, not declared negligible.',
      'Local r2 evolved background/final addition: '+one([neutral['measurements'][-1]['local_r2'],add['measurements'][-1]['local_r2']]),
      'Material LEFT/Casimir, radial force and all eight Pi/Q² are retained in analysis.json.',
      'Rotated lattice+source FAMILY is physically covariant; fixed finite lab-mode cavity not a closed SO3 representation.',
      'Central sewing/quantization is explicit quantum/topological input, not classical Euler-derived QM.',
      'No Nature electron, stable pole, C/positron, Lorentz/inertial-mass, physical charge or QED claim.',
      'Natural exits: '+one(exits),
      'Selected Pyright summary: '+one(language['diagnostics']['summary']),
      'Rooted LSP exit='+str(language['rooted_LSP']['natural_exit'])+'; positive AST and exact selected source hashes retained.',
      'Raw states/operators are separate pickle/NPZ artifacts. Pickles are trusted own-run data, not portable untrusted interchange.',
      'Full derivation/units/position/current/metric semantics: equations.json. Capsule manifest binds every included byte.'
    ]
    assert len(lines)<=80
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    checkpoint=ROOT/(P+'build_checkpoint.pkl')
    if checkpoint.exists():checkpoint.unlink()
    excluded={P+'manifest.json',P+'delivery_receipt.json',P+'delivery_capsule.zip'}
    files=[f for f in sorted(ROOT.glob(P+'*')) if f.is_file() and f.name not in excluded]
    files += [ROOT/'rich_gauge_band_frame_20261016_model.py',ROOT/'rich_spatial_frame_photon_20261016_kernel.py',ROOT/'rich_spatial_frame_photon_20261016_quadrature.py',ROOT/'rich_spatial_frame_photon_20261016_fields.py',ROOT/'rich_frame_gauss_20261014_model.py']
    manifest={f.name:dict(bytes=f.stat().st_size,sha256=digest(f)) for f in files}
    (ROOT/(P+'manifest.json')).write_text(json.dumps(dict(files=manifest,law=result['law'],synthesis_lines=len(lines),limits=result['limits']),indent=2)+'\n')
    capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.name)
        z.write(ROOT/(P+'manifest.json'),P+'manifest.json')
    receipt=dict(capsule=str(capsule),bytes=capsule.stat().st_size,sha256=digest(capsule),synthesis_lines=len(lines),manifest_sha256=digest(ROOT/(P+'manifest.json')))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
