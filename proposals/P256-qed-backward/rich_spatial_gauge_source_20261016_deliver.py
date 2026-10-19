"""Outcome-only compact synthesis and revision-bound delivery capsule."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
P='rich_spatial_gauge_source_20261016_'


def load(s):
    return json.loads((ROOT/(P+s+'.json')).read_text())


def main():
    r=load('result');ex=load('exit')
    if ex['returncode']!=0 or not ex['source_unchanged'] or not ex['frozen_inputs_unchanged']:
        raise ValueError('No successful exact-source experiment')
    lines=[
        'ACTUAL DECLARED SPATIAL EM SOURCE FAMILY; not Nature particle/photon admission.',
        'Frozen physical CAR12/radial3/material l0+l1/literal Gauss H0 and dressed neutral retained.',
        'MODEL HYPOTHESIS: positions x_v=R VERTICES_v; net Q_v=2-n_v including rotating ions.',
        'r_v are matrix amplitudes, not spatial radii. Actual I=4 inverse metric; no new mass.',
        'L_rot=I omega²/2+omega dot Lambda[A]; Lambda=sum Q x cross A(x).',
        'H[A]=H0+Peierls correction-{L,Lambda}/(2I)+Lambda²/(2I).',
        'All coordinate products made BEFORE material cut. P Lambda² P != (P Lambda P)².',
        'All scalar/vector hopping, r, gamma, finite U, electric/Gauss degrees remain active.',
        'Weak spatial source J[a]=-dH/dalpha; contact=d²H/dalpha², includes hopping+mechanics.',
        'Uniform B uses A=B cross x/2; magnetic moment=-dH/dB; constant A probes lab current.',
        'This completion is NOT uniquely forced by H0 and does not invalidate old graph chargeflow.',
        'No Pauli spin-B insertion, magnetic restorer, log(nilpotent U), or borrowed Pi propagator.',
        'New .35 matched-neutral filter, new .35 full-H charged source filter, root0 both LEFT components.',
        'Raw filter strengths are not Born probabilities. Both original/matched neutral baselines measured.',
        'Full doublet matrices and actual S,L_material,J axes are in raw result; no presumed spin labels.',
        'Prepared energy slopes below are NOT eigenstate Zeeman levels, mass or g2.',
    ]
    for branch,b in r['branches'].items():
        z=b['axes']['z'];a=z['finite_time_B08_t018']
        lines += [
            f"{branch}: dim={b['dimension']} netQ={b['charge']:.10g} raw strengths={b['raw_filter_weights']}",
            f"{branch}: Jminus overlap={b['lowering_overlap']} residual={b['lowering_residual']:.3g}",
            f"{branch}: mu_z real={z['moment']['real']} imag={z['moment']['imag']}",
            f"{branch}: hop_mu_z={z['hopping_moment']['real']}; convective_mu_z={z['convective_moment']['real']}",
            f"{branch}: C_Bz={z['contact']['real']}; wrong compressed={z['wrong_compressed_contact']['real']}",
            f"{branch}: ownH zero={z['zero_H_action_error']:.3g}, dH={z['own_H_first_derivative_error']:.3g}, d2H={z['own_H_contact_error']:.3g}",
            f"{branch}: contact complement action={z['complete_vs_compressed_contact_action']:.8g}",
            f"{branch}: constant-A direct source norm={z['constant_A_direct_source_norm']:.8g}; cut commutator complement={z['constant_A_fixed_projection_complement_norm']:.8g}",
            f"{branch}: London phase derivative errors={z['London_covariant_derivative_pointwise_errors']}; endpoint errors={z['London_hopping_endpoint_covariance_errors']}",
            f"{branch}: prepared slopes B±.04={z['prepared_energy_slopes']}",
            f"{branch}: actual B=.08,t=.18 ray response={a['ray_distance']:.8g}; deltaS={a['delta_S']}",
            f"{branch}: delta L_material={a['delta_material_L']}; delta zeroB moment={a['magnetic_moment_B_minus_free']:.8g}",
            f"{branch}: propagated B energy/variance={a['B_energy']}; free={a['free_energy']}",
            f"{branch}: B/free coarse-fine={a['propagation'].get('same_state_accuracy_comparison')}/{a['free_propagation'].get('same_state_accuracy_comparison')}",
        ]
    lines += [
        'Gauge covariance is exact in full SO(3)/moving London basis, not claimed in naive fixed finite projection.',
        'Constant-A fullsource versus i[PDP,PHP] complement is reported, not subtracted or fitted.',
        'Pointwise London checks use finite derivative of actual coordinate phase and actual hop endpoint phases.',
        'External B measures this declared source; it is NOT origin or evidence of dynamical photons.',
        'Nonzero spin-odd moment may include material/orbital cloud; not automatically intrinsic spin moment.',
        'Limits: finite Ecap2, radial3, material l0+l1; no regulator convergence, massshell, vacuum/pole theorem.',
        'No dynamic Cartesian Maxwell field, translational completion, photon dispersion or Nature g-factor claimed.',
        f"Actual natural science exit={ex['returncode']}; source unchanged={ex['source_unchanged']}; frozen unchanged={ex['frozen_inputs_unchanged']}; elapsed={ex['elapsed_seconds']:.1f}s.",
        'Exact equations/input/source/raw_arrays/exit/language records and SHA256 manifest are in delivery capsule.',
    ]
    if len(lines)>80:
        raise ValueError('Physical synthesis exceeds contract')
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    names=sorted(p.name for p in ROOT.glob(P+'*') if p.is_file() and not p.name.endswith(('delivery_capsule.zip','manifest.json','delivery_receipt.json')))
    manifest=dict(kind='EXACT_SCOPED_SPATIAL_SOURCE_DELIVERY',files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},summary_lines=len(lines))
    (ROOT/(P+'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(P+'delivery_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for name in names+[P+'manifest.json']:
            z.write(ROOT/name,name)
    receipt=dict(capsule=P+'delivery_capsule.zip',sha256=hashlib.sha256((ROOT/(P+'delivery_capsule.zip')).read_bytes()).hexdigest(),synthesis=P+'synthesis.txt',summary_lines=len(lines))
    (ROOT/(P+'delivery_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
