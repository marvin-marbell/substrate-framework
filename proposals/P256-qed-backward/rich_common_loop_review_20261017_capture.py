"""Frozen pure-operator loop review; never opens author states or results."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,zipfile

ROOT=Path(__file__).resolve().parent;P='rich_common_loop_review_20261017_';OLD='rich_common_quantum_em_20261016_'
PY='/home/administrator/substrate-framework/.venv/bin/python'
NATIVE_SHA='eeae0c720a8ac112a7f6b9931da1a8cf934371edf649ecee8eb88e679a5cec36'
MANIFEST_SHA='bf73a2718ca9b8c6a923530aad18b7d7cc48e06dddfd91623fd68aaa535fa699'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def copy_input(archive,name,path,expected):
    if path.exists():
        if digest(path)!=expected:raise ValueError(('frozen input collision',path.name))
    else:
        with archive.open(name) as source,path.open('xb') as target:shutil.copyfileobj(source,target)
    if digest(path)!=expected:raise ValueError(('frozen payload mismatch',name))
    return dict(source_member=name,path=path.name,sha256=expected)


def main():
    attempt=sys.argv[1];stem=P+attempt;native=ROOT/(OLD+'delivery1_native.zip');manifest_path=ROOT/(OLD+'delivery1_native_manifest.json')
    if digest(native)!=NATIVE_SHA or digest(manifest_path)!=MANIFEST_SHA:raise ValueError('frozen native capsule/manifest mismatch')
    manifest=json.loads(manifest_path.read_text());inputs=[];code=ROOT/(P+'frozen_code.zip')
    suffixes=('_body.json','_geometry.json','_static.npz','_vector.npz','_scalar_hop.npz','_scalar.npz','_inversehalf.npz','_chord_vector.npz','_chord_scalar.npz')
    with zipfile.ZipFile(native) as z:
        if not code.exists():
            with zipfile.ZipFile(code,'x',zipfile.ZIP_DEFLATED) as out:
                for name,record in manifest.items():
                    if not name.endswith('.py'):continue
                    payload=z.read(name)
                    if hashlib.sha256(payload).hexdigest()!=record['sha256']:raise ValueError(('frozen source mismatch',name))
                    out.writestr(name,payload)
        with zipfile.ZipFile(code) as frozen:
            for name,record in manifest.items():
                if name.endswith('.py') and hashlib.sha256(frozen.read(name)).hexdigest()!=record['sha256']:raise ValueError(('existing frozen source mismatch',name))
        for suffix in suffixes:
            name=OLD+'loopsea_prepare2'+suffix;inputs.append(copy_input(z,name,ROOT/(P+'frozen'+suffix),manifest[name]['sha256']))
    config=dict(plan_id='issue222-rich-backward-candidates-20261014',captured_revision='5f8d1460d8919b463ee985998615c53b9a15160abaf970bf5eb06f4c50ecfe70',template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='independent-review',kind='DISTINCT_LOOP_RICH_NATIVE_QUANTUM_WORD_REVIEW_NOT_FIRST_C1_OR47',native_capsule_sha256=NATIVE_SHA,native_manifest_sha256=MANIFEST_SHA,frozen_code=str(code),frozen_code_sha256=digest(code),matter_prefix=str(ROOT/(P+'frozen')),operator_inputs=inputs,seed=20261017049,field_coordinates=[[0,0],[6,-4]],Schmidt_weights=[.6,.4],relative_phase=.37,scope='Own fresh native physical quantum packets and literal hop/CAR/field-projector words only; no author solved/evolved state, ground replay, clamped-state shortcut or photon claim.')
    if len(sys.argv)>2:
        prior=P+sys.argv[2]
        config['reuse_OWN_loop_epoch']=prior
        config['reuse_OWN_files']={suffix:digest(ROOT/(prior+suffix)) for suffix in ('_input.json','_exit.json','_partial_result.json','_raw.npz')}
    path=ROOT/(stem+'_input.json');path.write_text(json.dumps(config,indent=2)+'\n');own={p.name:digest(p) for p in ROOT.glob(P+'*.py')}
    with zipfile.ZipFile(ROOT/(stem+'_source_inputs.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for name in own:z.write(ROOT/name,name)
        for item in inputs:z.write(ROOT/item['path'],item['path'])
        z.write(code,code.name);z.write(path,path.name)
        for suffix in config.get('reuse_OWN_files',{}):z.write(ROOT/(config['reuse_OWN_loop_epoch']+suffix),config['reuse_OWN_loop_epoch']+suffix)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');entry='repair.py' if 'reuse_OWN_loop_epoch' in config else 'run.py';command=[PY,str(ROOT/(P+entry)),str(path)]
    start=time.time();child=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    (ROOT/(stem+'_stdout.txt')).write_text(child.stdout);(ROOT/(stem+'_stderr.txt')).write_text(child.stderr)
    result=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-start,own_sources_unchanged={name:digest(ROOT/name)==value for name,value in own.items()},frozen_code_unchanged=digest(code)==config['frozen_code_sha256'],operator_inputs_unchanged={item['path']:digest(ROOT/item['path'])==item['sha256'] for item in inputs},command=command)
    (ROOT/(stem+'_exit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),child.stdout,child.stderr[-3500:],flush=True);sys.exit(child.returncode)

if __name__=='__main__':main()
