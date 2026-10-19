"""Raw-only changed-layer audit, never a replayed state or independent MC claim."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
P='rich_review_wave43_20261016_'
OWNER='rich_unitary_rotor_photon_20261016_'


def summarize(array):
    blocks=np.array([a.mean(axis=0) for a in np.array_split(array,8)])
    return dict(real=array.mean(axis=0).real.tolist(),imag=array.mean(axis=0).imag.tolist(),block_SEM=(blocks.std(axis=0,ddof=1)/np.sqrt(8)).tolist())


def main():
    owner_exit=json.loads((ROOT/(OWNER+'exit.json')).read_text())
    names=['b0.5_M4_s41101','b1_M4_s41102','b1_M8_s41103','b0.5_M4_s41104','b1_M8_s41105']
    cases={};saved={}
    for name in names:
        path=ROOT/(OWNER+name+'_raw.npz')
        with np.load(path) as data:
            assert all(np.isfinite(data[key]).all() for key in ('EE','JJ','EJ','theta','alpha','HS','logweight'))
            ee=data['EE'];jj=data['JJ'];theta=data['theta'];delta=np.roll(theta,-1,axis=1)-theta
            # Time-independent and reverse-q pairing identities of stored field operators.
            pairing=float(np.max(abs(ee[:,0]-ee[:,2].conjugate())))
            entry=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),samples=len(theta),shape=list(theta.shape),EE=summarize(ee),JJ=summarize(jj),EJ=summarize(data['EJ']),contact=summarize(data['contact']),plaquette=summarize(data['plaquette']),qpair_EE_error=pairing,finite=True,temporal_link_variance_mean=float(np.var(theta,axis=1).mean()))
            cases[name]=entry;saved[name+'_EEblocks']=np.array([a.mean(axis=0) for a in np.array_split(ee,8)]);saved[name+'_JJblocks']=np.array([a.mean(axis=0) for a in np.array_split(jj,8)])
    # Read superseded raw bytes ONLY to audit the changed sampling distribution.
    first=ROOT/(OWNER+'first_attempt_capsule.zip')
    with zipfile.ZipFile(first) as archive:
        with np.load(io.BytesIO(archive.read(OWNER+'b0.5_M4_s41101_raw.npz'))) as data:
            old= data['EE'].mean(axis=0).real
    current=np.array(cases[names[0]]['EE']['real'])
    seed_comparisons={}
    for a,b in ((names[0],names[3]),(names[2],names[4])):
        for observable in ('EE','JJ'):
            x,y=cases[a][observable],cases[b][observable]
            diff=np.array(x['real'])-np.array(y['real']);combined=np.sqrt(np.array(x['block_SEM'])**2+np.array(y['block_SEM'])**2)
            seed_comparisons[a+'__'+b+'_'+observable]=dict(max_standardized_real_difference=float(np.max(abs(diff)/np.maximum(combined,1e-30))),T1_zero_and_midpoint=[float(diff[0,0]),float(diff[0,len(diff[0])//2])])
    result=dict(kind='CHANGED_CORRECTED_V64_THERMAL_RAW_AUDIT_NOT_OWN_STATE_REPLAY',owner_natural_exit=owner_exit,cases=cases,independent_seed_comparisons=seed_comparisons,first_attempt_EE_T1=old[0].tolist(),corrected_EE_T1=current[0].tolist(),classification='Actual corrected finite beta/Trotter full-rotor samples, quantum unequal-imaginary-time E/J response present. Finite chain and seed/Trotter differences remain; no equilibration, analytic continuation, photon or confidence-coverage certification.')
    (ROOT/(P+'corrected_raw_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(P+'corrected_raw_arrays.npz'),**saved)
    print(json.dumps(dict(owner_exit=owner_exit['natural_exit'],cases={n:dict(EE_T1=cases[n]['EE']['real'][0],JJ_T1=cases[n]['JJ']['real'][0]) for n in names},seed_comparisons=seed_comparisons)),flush=True)


if __name__=='__main__':main()
