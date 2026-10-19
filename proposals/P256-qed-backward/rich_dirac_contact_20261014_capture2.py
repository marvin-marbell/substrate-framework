"""Exact complete source/input/runtime capsule for NEW vector-field receiver."""
import hashlib,json,os,platform,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PREFIX='rich_dirac_contact_20261014_actual2'
SOURCES=['rich_dirac_contact_20261014_model.py','rich_dirac_contact_20261014_field.py','rich_dirac_contact_20261014_capture2.py','rich_dirac_contact_20261014_input.json','rich_dirac_contact_20261014_calculate.py','rich_spinor_rot_20261014_model.py','rich_mobile_spinor_20261014_model.py']
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    before={a:digest(ROOT/a) for a in SOURCES}
    command=[sys.executable,str(ROOT/SOURCES[1])]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.time(); p=subprocess.run(command,cwd=ROOT,env=env,capture_output=True); elapsed=time.time()-start
    (ROOT/(PREFIX+'_stdout.txt')).write_bytes(p.stdout); (ROOT/(PREFIX+'_stderr.txt')).write_bytes(p.stderr)
    names=[PREFIX+'_stdout.txt',PREFIX+'_stderr.txt']
    if p.returncode==0: names += [PREFIX+'_result.json',PREFIX+'_states.npz']
    receipt={'kind':'NEW_DIRAC_CONTACT_VECTOR_RECEIVER_NATURAL_EXIT','command':command,'cwd':str(ROOT),'natural_exit':True,'returncode':p.returncode,'wall_seconds':elapsed,'python':sys.version,'platform':platform.platform(),'thread_environment':{a:env[a] for a in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},'source_before':before,'source_after':{a:digest(ROOT/a) for a in SOURCES},'outputs':{a:digest(ROOT/a) for a in names}}
    name=PREFIX+'_exit.json'; (ROOT/name).write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(PREFIX+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for a in SOURCES+names+[name]: z.write(ROOT/a,arcname=a)
    print(json.dumps(receipt,indent=2)); print(p.stdout.decode()); print(p.stderr.decode()); raise SystemExit(p.returncode)
if __name__=='__main__': main()
