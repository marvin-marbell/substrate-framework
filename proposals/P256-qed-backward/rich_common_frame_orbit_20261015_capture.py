"""Freeze source/H/state early; actual natural exit and PID-bound exact capsules."""
import hashlib,importlib,importlib.metadata,json,os,platform,resource,subprocess,sys,time,zipfile
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    root=Path(__file__).resolve().parent;prefix='rich_common_frame_orbit_20261015_'+sys.argv[1];inp=root/(prefix+'_input.json');out=root/(prefix+'_result.json')
    sourcecap=root/(prefix+'_source_capsule.zip')
    if sourcecap.exists():raise RuntimeError('Immutable attempt tag already captured; choose a new actual tag, never replay/overwrite')
    parameters=json.loads(inp.read_text())
    environment=dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:environment[key]='1';os.environ[key]='1'
    # Import declarations/pure helpers ONLY to discover the actual complete
    # transitive source graph. No model build, producer/main or old state read.
    for module in ['rich_common_frame_orbit_20261015_frame','rich_common_frame_orbit_20261015_model','rich_common_frame_orbit_20261015_reduce','rich_common_frame_orbit_20261015_calculate']:importlib.import_module(module)
    if parameters.get('representation')=='native_charge_orbit':importlib.import_module('rich_common_frame_orbit_20261015_native')
    paths={Path(module.__file__).name:Path(module.__file__) for name,module in list(sys.modules.items()) if name.startswith('rich_') and getattr(module,'__file__',None) and Path(module.__file__).parent==root}
    for name in ['rich_common_frame_orbit_20261015_capture.py','rich_common_frame_orbit_20261015_equations.json','rich_common_frame_orbit_20261015_interface.json',inp.name]:paths[name]=root/name
    if parameters.get('representation')=='native_charge_orbit':
        body=(root/parameters['native_body']).resolve()
        if body.parent!=root or sha(body)!=parameters['native_body_sha256']:raise ValueError('Untrusted/unpinned native BODY artifact')
        paths[body.name]=body
    frozen={name:path.read_bytes() for name,path in paths.items()};before={name:sha(path) for name,path in paths.items()}
    with zipfile.ZipFile(sourcecap,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in frozen.items():archive.writestr(name,content)
    command=[sys.executable,str(root/'rich_common_frame_orbit_20261015_calculate.py'),str(inp),str(out)];started=time.time();stdout=root/(prefix+'_stdout.txt');stderr=root/(prefix+'_stderr.txt')
    def child_limit():
        if parameters.get('address_space_limit_MiB'):
            bound=parameters['address_space_limit_MiB']*1024**2;resource.setrlimit(resource.RLIMIT_AS,(bound,bound))
    with stdout.open('w') as sout,stderr.open('w') as serr:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=sout,stderr=serr,preexec_fn=child_limit)
        start=dict(kind='NEW_EXACT_CONSUMER_STARTED',command=command,pid=process.pid,started_unix=started,source_before=before,source_capsule_sha256=sha(sourcecap),python=sys.version,executable=sys.executable,packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','sympy']})
        start['address_space_limit_MiB']=parameters.get('address_space_limit_MiB')
        startpath=root/(prefix+'_start.json');startpath.write_text(json.dumps(start,indent=2)+'\n');returncode=process.wait()
    finished=time.time();after={name:sha(path) for name,path in paths.items()}
    suffixes=['_stdout.txt','_stderr.txt','_start.json','_construction.json','_construction.npz','_reference.npz','_source_checkpoint.npz','_result.json','_result.npz']
    outputs=[root/(prefix+suffix) for suffix in suffixes if (root/(prefix+suffix)).exists()]
    receipt=dict(command=command,pid=process.pid,returncode=returncode,natural_exit=returncode>=0,termination_signal=-returncode if returncode<0 else None,started_unix=started,finished_unix=finished,elapsed_seconds=finished-started,peak_child_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,source_before=before,source_after=after,source_unchanged=before==after,source_capsule_sha256=sha(sourcecap),outputs={path.name:sha(path) for path in outputs},platform=platform.platform(),python=sys.version)
    receipt['address_space_limit_MiB']=parameters.get('address_space_limit_MiB')
    exitpath=root/(prefix+'_exit.json');exitpath.write_text(json.dumps(receipt,indent=2)+'\n');capsule=root/(prefix+'_capsule.zip')
    with zipfile.ZipFile(capsule,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in frozen.items():archive.writestr(name,content)
        for path in outputs+[exitpath]:archive.write(path,path.name)
    print(stdout.read_text())
    if stderr.stat().st_size:print(stderr.read_text(),file=sys.stderr)
    print(json.dumps(dict(returncode=returncode,natural_exit=receipt['natural_exit'],pid=process.pid,elapsed_seconds=receipt['elapsed_seconds'],peak_rss_KiB=receipt['peak_child_rss_KiB'],capsule_sha256=sha(capsule))))
    raise SystemExit(returncode)


if __name__=='__main__':main()
