"""Unique-prefix finite scientific capture with exact sources/inputs and exit."""
import hashlib,json,os,resource,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PREFIX='rich_composite_geometry_20261015_'
HELPERS=['rich_colour_composite_20261014_model.py','rich_common_colour_lepton_20261015_model.py',
 'rich_spatial_colour_root_20261015_model.py','rich_colour_vector_loop_20261015_model.py',
 'rich_quantum_common_field_20261015_sources.py','rich_common_vector_20261015_model.py']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def limits():resource.setrlimit(resource.RLIMIT_AS,(3800*1048576,3800*1048576))


def main():
    run=sys.argv[1];phase=sys.argv[2]
    if not run.startswith(PREFIX) or any(c in run for c in '/\\'):raise ValueError('Owned new unique prefix required')
    output=ROOT/run;input_path=ROOT/(run+'_input.json')
    if Path(str(output)+'_before.json').exists():raise FileExistsError('Never overwrite an actual capture')
    config=json.loads(input_path.read_text());sources=list(ROOT.glob(PREFIX+'*.py'))+[ROOT/name for name in HELPERS]
    inputs=[input_path]+[ROOT/name for key,name in config.items() if key.endswith('_snapshot')]
    before={p.name:sha(p) for p in sources+inputs}
    Path(str(output)+'_before.json').write_text(json.dumps(before,indent=2)+'\n')
    with zipfile.ZipFile(str(output)+'_source_capsule.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sources+inputs:z.write(p,p.name)
    executable='construct' if phase in ('neutral','charged') else phase
    command=[sys.executable,str(ROOT/(PREFIX+executable+'.py')),str(input_path),str(output)]
    if executable=='construct':command.append(phase)
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.time()
    with open(str(output)+'_stdout.txt','xb') as out,open(str(output)+'_stderr.txt','xb') as err:
        child=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,env=environment,preexec_fn=limits)
    child_rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    after={p.name:sha(p) for p in sources+inputs}
    files=[p for p in ROOT.glob(run+'_*') if p.is_file() and not p.name.endswith(('_capsule.zip','_source_capsule.zip','_exit.json'))]
    receipt={'kind':'ACTUAL_NEW_OCCUPIED_COMPOSITE_GEOMETRY_NATURAL_CAPTURE_NOT_TEST','command':command,
        'returncode':child.returncode,'natural_exit':child.returncode>=0,'signal':-child.returncode if child.returncode<0 else None,
        'started_unix':started,'finished_unix':time.time(),'address_space_limit_MiB':3800,
        'child_peak_RSS_KiB':child_rss,
        'source_input_before':before,'source_input_after':after,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},
        'RSS':'Actual RUSAGE_CHILDREN peak of the sole scientific child in this fresh capture process, also available for failed children; older failure without this field remains UNKNOWN.'}
    exit_path=Path(str(output)+'_exit.json');exit_path.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(str(output)+'_capsule.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in dict.fromkeys(sources+inputs+files+[exit_path]):z.write(p,p.name)
    print(json.dumps({'run':run,'returncode':child.returncode,'elapsed_seconds':time.time()-started,'exit':str(exit_path)}),flush=True)


if __name__=='__main__':main()
