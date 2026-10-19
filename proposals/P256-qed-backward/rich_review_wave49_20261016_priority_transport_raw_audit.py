"""Independently measure OWN completed carrier tensors; never rerun evolution."""
from pathlib import Path
import json,os,subprocess,sys,time,zipfile
import numpy as np
from scipy.linalg import block_diag
from tenpy.linalg import np_conserved as npc
from tenpy.networks.site import Site
from rich_review_wave49_20261016_priority_raw_audit import state_data,measure,encode,digest

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_';E='transport_priority2';TAG=P+'priority_transport_raw_audit';PY='/home/administrator/substrate-framework/.venv/bin/python'


def worker():
    inputs=json.loads((ROOT/(TAG+'_input.json')).read_text())
    for name,value in inputs['input_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError(('completed own physical input changed',name))
    snapshot=ROOT/(P+'priority_transport2_source_capsule.zip');sys.path.insert(0,str(snapshot))
    import rich_gauge_band_frame_20261016_model as frozen
    if not frozen.__file__.startswith(str(snapshot)):raise ValueError('actual LEFT generator source escaped frozen snapshot')
    js=np.arange(0,2.01,.5);labels=[(j,m,k) for j in js for m in np.arange(-j,j+1) for k in np.arange(-j,j+1)];L=[block_diag(*[-np.kron(frozen.jmat(j)[a].T,np.eye(round(2*j+1))) for j in js]) for a in range(3)]
    ci=npc.ChargeInfo([1,2,2,2,2],['N','sew0','sew1','sew2','sew3']);q=np.zeros((55,5),int);q[:,1]=[round(2*j)%2 for j,m,k in labels];gperm=Site(npc.LegCharge.from_qflat(ci,q),sort_charge=True).perm;rfperm=Site(npc.LegCharge.from_qflat(ci,np.zeros((315,5),int)),sort_charge=True).perm
    preparation=json.loads((ROOT/(P+E+'_preparation.json')).read_text());native=json.loads((ROOT/(P+E+'_carrier_result.json')).read_text());states={};arrays={};error={}
    for age,suffix in (('initial','initial'),('final','full')):
        for kind in ('neutral','charged'):
            key=kind+'_'+age;B,norm,total,record=state_data(ROOT/(P+E+'_'+kind+'_'+suffix+'_stage.json'));value,f=measure(B,norm,total,preparation['field_basis'],L,labels,gperm,rfperm);states[key]=value;arrays[key+'_field_density']=f;reference=native[age][kind]
            value['canonical_field_Pi']=value.pop('canonical_field_Pi_initial')
            error[key]=dict(n_cells=float(np.max(abs(value['n_cells']-reference['n_cells']))),LEFT=float(np.max(abs(value['physical_local_LEFT']-reference['local_LEFT']))),Pi=float(np.max(abs(value['canonical_field_Pi']-reference['physical_field_Pi']))),raw_norm2=abs(value['raw_norm2']-reference['raw_norm2']))
    def excess(age,key):return np.asarray(states['charged_'+age][key])-states['neutral_'+age][key]
    charge=-(excess('final','n_cells')-excess('initial','n_cells'));spin=excess('final','physical_local_LEFT')-excess('initial','physical_local_LEFT');field=excess('final','canonical_field_Pi')-excess('initial','canonical_field_Pi')
    assertions=dict(independent_raw_charge_transport=bool(np.linalg.norm(charge[1:])>1e-12),independent_raw_physical_LEFT_transport=bool(np.linalg.norm(spin[1:])>1e-12),native_observable_agreement=max(v for item in error.values() for v in item.values())<2e-12)
    result=dict(kind='OWN_ACTUAL_FULLCAR32_FINITE_TIME_TRANSPORT_INDEPENDENT_RAW_NETWORK_AUDIT',initial_time=.004,final_time=.008,states=states,excess_charge_change=charge,physical_Noether_density_change=spin,dynamic_canonical_Pi_doublecontrast=field,native_MPO_vs_independent_raw_network_errors=error,physics_assertions=assertions,method='Direct dense raw-MPS left/right environments and native analytic LEFT/Site permutation, every local charge count and before-cut field covariance; NO cached Model allocation, native observable MPO, new dynamics or producer state.',limits='Agreement verifies measurement/normalization/charge-leg interpretation of the stored states. It does NOT resolve the existing chi16/dt/field/R/g/exterior dynamics errors; SVD epsilon sums are not error bars. Field feedback still requires matched free-dressing/full controls.')
    np.savez_compressed(ROOT/(TAG+'_raw.npz'),**arrays);(ROOT/(TAG+'_result.json')).write_text(json.dumps(encode(result),indent=2)+'\n');print(json.dumps(encode(dict(charge_change=charge,LEFT_change=spin,measurement_errors=error,physics_assertions=assertions))),flush=True);assert all(assertions.values()),assertions


def capture():
    files=[ROOT/(P+E+'_preparation.json'),ROOT/(P+E+'_carrier_result.json'),ROOT/(P+'priority_transport2_source_capsule.zip')]
    for age in ('initial','full'):
        for kind in ('neutral','charged'):
            meta=ROOT/(P+E+'_'+kind+'_'+age+'_stage.json');files.extend((meta,ROOT/json.loads(meta.read_text())['raw_file']))
    source=ROOT/(P+'priority_raw_audit.py');inputs=dict(source_sha256=digest(__file__),helper_sha256=digest(source),input_sha256={path.name:digest(path) for path in files},scope='Own completed actual same-H states only; no new evolution, Model clone or convergence approval.')
    (ROOT/(TAG+'_input.json')).write_text(json.dumps(inputs,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(TAG+'_source_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__,Path(__file__).name);archive.write(source,source.name);archive.write(ROOT/(TAG+'_input.json'),TAG+'_input.json')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');started=time.time()
    with (ROOT/(TAG+'_stdout.txt')).open('x') as stdout,(ROOT/(TAG+'_stderr.txt')).open('x') as stderr:child=subprocess.run([PY,__file__,'worker'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    result=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-started,source_unchanged=digest(__file__)==inputs['source_sha256'],helper_unchanged=digest(source)==inputs['helper_sha256'],inputs_unchanged={path.name:digest(path)==inputs['input_sha256'][path.name] for path in files});(ROOT/(TAG+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);sys.exit(child.returncode)

if __name__=='__main__':
    if len(sys.argv)>1:worker()
    else:capture()
