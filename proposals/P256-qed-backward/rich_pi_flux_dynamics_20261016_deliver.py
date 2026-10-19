"""Physical <=80-line synthesis, raw-array observations, exact manifests and capsule."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent
P='rich_pi_flux_dynamics_20261016_'


def save(name,value):
    (ROOT/(P+name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def fmt(values):return ', '.join(f'{float(x):.7g}' for x in np.asarray(values).ravel())


def main():
    first=json.loads((ROOT/(P+'result.json')).read_text())
    second=json.loads((ROOT/(P+'pair_enriched_result.json')).read_text())
    a=first['joined'];b=second['joined']
    data=np.load(ROOT/(P+'pair_enriched_joined_dynamics.npz'))
    chains=[np.load(ROOT/(P+'pair_enriched_chain'+str(k)+'_dynamics.npz')) for k in range(2)]
    selected=np.array([0,25,50,75,100,150]);observation=dict(times=data['times'][selected].tolist(),receivers='indices x0y,x0z,x1y,x1z,x2y,x2z,x3y,x3z; then q1T0,q1T1,q2T0,q2T1')
    for name in ('field','current','layer_total_energy','matter'):
        joined=data[name][:,1:]-data[name][:,0,None]
        replica=[c[name][:,1:]-c[name][:,0,None] for c in chains]
        se=abs(replica[0]-replica[1])/2
        observation[name+'_response_real']=joined[selected].real.tolist()
        observation[name+'_response_imag']=joined[selected].imag.tolist()
        observation[name+'_two_chain_SE']=se[selected].tolist()
        observation[name+'_max_response_abs_by_channel']=np.max(abs(joined),axis=0).tolist()
        observation[name+'_max_two_chain_SE_by_channel']=np.max(se,axis=0).tolist()
        np.savez_compressed(ROOT/(P+name+'_enriched_response_uncertainty.npz'),times=data['times'],joined_response=joined,chain0_response=replica[0],chain1_response=replica[1],two_chain_SE=se)
    workse=abs(np.array(second['chains'][0]['pulse_total_work'])-second['chains'][1]['pulse_total_work'])/2
    observation['pulse_work_two_chain_SE']=workse.tolist()
    observation['full_H_projection_residual_RMS']=b['projection_residual_RMS_t0']
    save('selected_physical_observations',observation)
    language=json.loads((ROOT/(P+'language_final.json')).read_text())
    exits=[json.loads((ROOT/(P+suffix+'exit.json')).read_text()) for suffix in ('','pair_enriched_')]
    wilson=np.load(ROOT/(P+'joined_wilson_dynamics.npz'))['wilson']
    wresponse=wilson[:,1:]-wilson[:,0,None]
    lines=[
      'Actual coupled pi-state finite-time two-transverse field/current construction; bounded variational result.',
      'Fixed H: V64/I16/t1/U4/Ecap1, full local Fock4/link3, Nup=Ndown32, Gauss legal; no B/Maxwell term.',
      'Input pi-pair state/capsule 410a1543da38e937a319bc4e301b54c3c1356d9077192b8722b3b6f6768beab5.',
      'Two actual material state families were evolved, not only static contact/norm diagnostics.',
      'First38: Omega plus complex field, rho, doublon/nearest-density and actual qutrit Wilson coordinates.',
      'Large first projection residual motivated second76: coherent full64 pair kernels delta .8 AND 1.2.',
      'Delta changes variational STATE, not H. Both pair components retain every legal fermion/link hop.',
      'H B is evaluated absolutely at every sampled configuration and all legal one-hop virtuals, including nodes.',
      'Sampled full-H Galerkin matrix is Hermitian projected; this is NOT exact full-H time evolution.',
      'Nonstationary unpulsed Omega and both pulsed states evolve under the SAME projected whole H.',
      'No frozen hot-state EOM hermitization. Source is [H,O]Omega, never (H-E)O Omega.',
      'Pulses: exp(i .35 loop.E/2) at origin, xy/xz cycles; actual pure-field unitaries, not directed packets.',
      'Initial matter and E probabilities unchanged. Initial hopping/current and Wilson response need not vanish.',
      'Qutrit W products, nilpotent boundary zeros and Wsin/Wcos act before projection; no supplied angle/log(U).',
      'Actual first-family Wcos/Wsin time response maxima by pulse: '+fmt(np.max(abs(wresponse),axis=(0,2))),
      'Both transverse channels at q1=(pi/2,0,0), q2=(pi/2,pi/2,0), q adjoints, both temporal frequency signs saved.',
      'Finite time t=0..1.5 (151 points); frequencies are windowed response, NOT stationary poles.',
      'Each attempt used two independent fresh 768-sample chains, burn18000/thin160; raw states/actions preserved.',
      'First38 rank='+str(a['rank'])+'; E0='+fmt(a['initial_energy'])+'; pulse work='+fmt(a['pulse_total_work']),
      'First38 full-H residual RMS(t0)='+fmt(a['projection_residual_RMS_t0'])+'; max='+fmt(a['projection_residual_RMS_max']),
      'Second76 retained rank='+str(b['rank'])+'; E0='+fmt(b['initial_energy']),
      'Second76 total pulse work='+fmt(b['pulse_total_work'])+'; exploratory two-chain SE='+fmt(workse),
      'This owned excitation energy includes onsite/matter, electric AND hopping interaction, not E2 crest alone.',
      'Second76 full-H residual RMS(t0)='+fmt(b['projection_residual_RMS_t0'])+'; max='+fmt(b['projection_residual_RMS_max']),
      'Residual is direct HBc-BWh a, combines outside-family and sampling error; not a certified trajectory bound.',
      'Sampled whole-H antihermitian relative first/second='+fmt([a['wholeH_sampling_antihermitian_relative'],b['wholeH_sampling_antihermitian_relative']]),
      'Second76 initial state representation squared errors='+fmt(b['initial_representation_relative_squared']),
      'Second76 finite-time q1/q2 field peak response pulse xy='+fmt(np.array(b['field_response_peak'])[0,8:]),
      'Second76 finite-time q1/q2 field peak response pulse xz='+fmt(np.array(b['field_response_peak'])[1,8:]),
      'Corresponding max two-chain field SE xy='+fmt(np.array(observation['field_max_two_chain_SE_by_channel'])[0,8:]),
      'Corresponding max two-chain field SE xz='+fmt(np.array(observation['field_max_two_chain_SE_by_channel'])[1,8:]),
      'Second76 finite-time q1/q2 current peak xy='+fmt(np.array(b['current_response_peak'])[0,8:]),
      'Second76 finite-time q1/q2 current peak xz='+fmt(np.array(b['current_response_peak'])[1,8:]),
      'Initial receiver currents xy, layers0..3(y,z)='+fmt(np.array(b['initial_current_response'])[0,:8]),
      'Initial receiver currents xz, layers0..3(y,z)='+fmt(np.array(b['initial_current_response'])[1,:8]),
      'Final receiver E/16 response xy='+fmt(np.array(b['final_field_response'])[0,:8]),
      'Final receiver E/16 response xz='+fmt(np.array(b['final_field_response'])[1,:8]),
      'Initial owned layer energy response xy='+fmt(b['initial_owned_layer_excitation_energy'][0]),
      'Initial owned layer energy response xz='+fmt(b['initial_owned_layer_excitation_energy'][1]),
      'Final owned layer energy response xy='+fmt(b['final_owned_layer_excitation_energy'][0]),
      'Final owned layer energy response xz='+fmt(b['final_owned_layer_excitation_energy'][1]),
      'Layer responses can be negative (interference/background subtraction); not a positive energy-density packet.',
      'Actual final matter response [D,Eenergy,rhoq1,rhoq2,Dq1,Dq2] xy='+fmt(b['final_matter_response'][0]),
      'Actual final matter response [same] xz='+fmt(b['final_matter_response'][1]),
      'Unpulsed background matter/field changes='+fmt([b['background_matter_change'],b['background_field_change']]),
      'Physical field covariance moving-metric max change='+fmt([b['moving_metric_change']]),
      'Fixed Hilbert coordinates require no moving-basis connection; physical observable covariance is recomputed.',
      'Full-current d<E/16>/dt=-<J>/16 vs projected commutator maximum defect='+fmt([b['full_vs_projected_Ehrenfest_max']]),
      'The Ehrenfest defect remains paid evidence of projection error, not suppressed/retuned into a wave law.',
      'Projected energy/norm algebraic drifts='+fmt([b['energy_drift_max'],b['norm_drift_max']]),
      'Layer sum vs total H error='+fmt([b['layer_energy_sum_error']])+'; projected conservation is NOT full-H accuracy.',
      'All sampled configurations Gauss max=0 in both families; total charge remains zero in this field response.',
      'Positive guides include independent detG plus strict floor and coherent-pair support; exact node psi=0,Hpsi!=0 measured.',
      'Node H2/guide=118.693; first-family proper-commutator relative error <=1.2e-14; wrong(H-E) source differs4.94/23.70.',
      'Importance ESS minima second76='+fmt(b['importance_ESS_min'])+'; not autocorrelation ESS/mixing proof.',
      'Two-chain SE is exploratory, only two replicates. Rare-node/H2 tails and Markov mixing not established.',
      'Physical outcome: NEW actual coupled finite-time variational field/current/matter response in both T channels.',
      'Discriminatory failure: neither finite family establishes a controlled full-H travelling field wave.',
      'Large residual/Ehrenfest defects and chain uncertainty forbid promoting apparent receiver motion into propagation.',
      'A bare cycle-E kick may split/stand; this result is NOT absence of photons and not a photon identification.',
      'No massless/Nature/Lorentz/Poynting/continuum or exact vacuum/pole claim; qmin=pi/2, Ecap1 finite regulator.',
      'First and material enriched science natural exits='+str([e['natural_science_exit'] for e in exits])+'; sources unchanged='+str([e['source_unchanged'] for e in exits]),
      'One owned enriched Numba compile failure retained with exact source/input/exit/stderr in failed_compile_capsule.zip.',
      'Reviewer independently retained earlier bool/closure/abs compiler failures; no runtime downgrade.',
      'Final selected Pyright exit='+str(language['natural_exit'])+', errors='+str(language['diagnostics']['summary'].get('errorCount'))+'; rooted LSP exit='+str(language['rooted_LSP']['natural_exit'])+'.',
      'Evidence: pure interfaces/equations, source/input/exit capsules, actual raw state/H/current arrays and full time traces.',
      'Selected physical arrays/initial tails/work/current and response-specific two-chain errors are separate JSON/NPZ.'
    ]
    assert len(lines)<=80
    (ROOT/(P+'synthesis.txt')).write_text('\n'.join(lines)+'\n')
    save('synthesis',dict(kind='FINITE_TWO_TRANSVERSE_PI_STATE_VARIATIONAL_DYNAMICS_NOT_FULL_H_PHOTON',first_family=a,enriched_family=b,observation=observation,science_exits=exits,physical_summary_lines=len(lines)))
    excluded={P+'delivery_capsule.zip',P+'manifest.json',P+'delivery_receipt.json'}
    files=[p for p in sorted(ROOT.glob(P+'*')) if p.is_file() and p.name not in excluded]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save('manifest',dict(files=hashes,frozen_input_capsule='410a1543da38e937a319bc4e301b54c3c1356d9077192b8722b3b6f6768beab5',science_source_capsules=[e['source_capsule_sha256'] for e in exits]))
    capsule=ROOT/(P+'delivery_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[ROOT/(P+'manifest.json')]:z.write(path,path.name)
    receipt=dict(capsule=capsule.name,sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),bytes=capsule.stat().st_size,summary_lines=len(lines),science_exits=[e['natural_science_exit'] for e in exits])
    save('delivery_receipt',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
