"""Exact before/after native construction or consumer capture; no checks/replay."""
import hashlib,json,os,resource,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'
PREFIX='rich_spatial_colour_charge_20261015_'


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def cap():
    n=3800*1024**2;resource.setrlimit(resource.RLIMIT_AS,(n,n))


def main():
    run=sys.argv[1];phase=sys.argv[2];base=ROOT/run;inp=ROOT/(run+'_input.json');p=json.loads(inp.read_text())
    sources=list(ROOT.glob(PREFIX+'*.py'))+[ROOT/x for x in ['rich_spatial_colour_root_20261015_model.py','rich_spatial_colour_root_20261015_calculate.py','rich_common_colour_lepton_20261015_model.py','rich_colour_vector_loop_20261015_model.py','rich_colour_composite_20261014_model.py','rich_quantum_common_field_20261015_sources.py','rich_common_vector_20261015_model.py']]
    sources += [ROOT/x for x in ['rich_spatial_colour_em_20261015_model.py','rich_spatial_colour_em_20261015_fields.py']]
    inputs=[inp]
    if 'body_prefix' in p:
        inputs += [ROOT/(p['body_prefix']+x) for x in ['_body.pkl','_H0.npz','_neutral_state.npz','_input.json','_result.json','_preflight.json']]
    if 'eigenspace_prefix' in p:
        inputs += [ROOT/(p['eigenspace_prefix']+x) for x in ['_states.npz','_input.json','_exit.json']]
    before={x.name:sha(x) for x in sources+inputs};before_path=ROOT/(run+'_before.json');before_path.write_text(json.dumps(before,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(run+'_source_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sources+[inp,before_path]:archive.write(path,path.name)
    out=ROOT/(run+'_stdout.txt');err=ROOT/(run+'_stderr.txt');command=[PYTHON,str(ROOT/(PREFIX+phase+'.py')),str(inp),str(base)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0');start=time.time()
    with out.open('xb') as fo,err.open('xb') as fe:process=subprocess.run(command,cwd=ROOT,env=env,stdout=fo,stderr=fe,check=False,preexec_fn=cap)
    files=list(ROOT.glob(run+'_*'));files=[x for x in files if x.suffix!='.zip']
    after={x.name:sha(x) for x in sources+inputs};exit_path=ROOT/(run+'_exit.json')
    meta=dict(kind='ACTUAL_NEW_NATIVE_CHARGE_ORBIT_NATURAL_CAPTURE_NOT_TEST',returncode=process.returncode,natural_exit=process.returncode>=0,signal=-process.returncode if process.returncode<0 else None,command=command,started_unix=start,finished_unix=time.time(),address_space_limit_MiB=3800,source_input_before=before,source_input_after=after,files={x.name:dict(bytes=x.stat().st_size,sha256=sha(x)) for x in files},scope='Actual new same-law charge-orbit construction/consumer only; exact original physical charge projectors, no old producer replay, H/current patch or science admission.')
    exit_path.write_text(json.dumps(meta,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(run+'_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for path in dict.fromkeys(sources+inputs+files+[exit_path]):archive.write(path,path.name)
    print(json.dumps(dict(run=run,returncode=process.returncode,elapsed=time.time()-start,exit=str(exit_path))))
    sys.exit(process.returncode)

if __name__=='__main__':main()
