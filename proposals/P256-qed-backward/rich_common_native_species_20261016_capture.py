"""Unique-prefix natural scientific receipts; ancestors read-only, epochs frozen."""
from pathlib import Path
import hashlib,json,os,resource,subprocess,sys,time,zipfile
ROOT=Path(__file__).resolve().parent
PREFIX='rich_common_native_species_20261016_'
HELPERS=['rich_colour_composite_20261014_model.py','rich_common_colour_lepton_20261015_model.py','rich_spatial_colour_root_20261015_model.py','rich_colour_vector_loop_20261015_model.py','rich_quantum_common_field_20261015_sources.py','rich_common_vector_20261015_model.py']


def sha(path):
    value=hashlib.sha256()
    with open(path,'rb')as stream:
        for part in iter(lambda:stream.read(1048576),b''):value.update(part)
    return value.hexdigest()


def limits():resource.setrlimit(resource.RLIMIT_AS,(3800*1048576,3800*1048576))


def main():
    run,phase=sys.argv[1:3]
    if not run.startswith(PREFIX)or'/'in run or'\\'in run:raise ValueError('New owned unique prefix required')
    out=ROOT/run;input_path=ROOT/(run+'_input.json')
    if Path(str(out)+'_before.json').exists():raise FileExistsError('Never overwrite captured source/input/result')
    config=json.loads(input_path.read_text());sources=list(ROOT.glob(PREFIX+'*.py'))+list(ROOT.glob('rich_composite_geometry_20261015_*.py'))+[ROOT/name for name in HELPERS];inputs=[input_path]+[ROOT/value for key,value in config.items()if key.endswith('_snapshot')];before={path.name:sha(path)for path in sources+inputs};Path(str(out)+'_before.json').write_text(json.dumps(before,indent=2)+'\n')
    with zipfile.ZipFile(str(out)+'_source_capsule.zip','x',compression=zipfile.ZIP_DEFLATED)as archive:
        for path in dict.fromkeys(sources+inputs):archive.write(path,path.name)
    command=[sys.executable,str(ROOT/(PREFIX+phase+'.py')),str(input_path),str(out)];started=time.time();environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with open(str(out)+'_stdout.txt','xb')as stdout,open(str(out)+'_stderr.txt','xb')as stderr:child=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,env=environment,preexec_fn=limits)
    after={path.name:sha(path)for path in sources+inputs};files=[path for path in ROOT.glob(run+'_*')if path.is_file()and not path.name.endswith(('_capsule.zip','_source_capsule.zip','_exit.json'))];receipt={'kind':'ACTUAL_COMMON_NATIVE_SPECIES_SCIENTIFIC_CAPTURE','command':command,'returncode':child.returncode,'natural_exit':child.returncode>=0,'signal':-child.returncode if child.returncode<0 else None,'started_unix':started,'finished_unix':time.time(),'child_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'address_space_limit_MiB':3800,'source_input_before':before,'source_input_after':after,'files':{path.name:{'bytes':path.stat().st_size,'SHA256':sha(path)}for path in files}};exit_path=Path(str(out)+'_exit.json');exit_path.write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(str(out)+'_capsule.zip','x',compression=zipfile.ZIP_DEFLATED,allowZip64=True)as archive:
        for path in dict.fromkeys(sources+inputs+files+[exit_path]):archive.write(path,path.name)
    print(json.dumps({'run':run,'returncode':child.returncode,'capture_seconds':time.time()-started,'peak_RSS_KiB':receipt['child_peak_RSS_KiB'],'unchanged':before==after}),flush=True)


if __name__=='__main__':main()
