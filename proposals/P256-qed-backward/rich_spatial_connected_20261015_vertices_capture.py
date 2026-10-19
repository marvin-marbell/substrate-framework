"""Exact before-cut derivative/source-state projection natural capture."""
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time,zipfile
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');result=ROOT/(prefix+'_result.json');config=json.loads(inp.read_text())
    sources=[ROOT/('rich_spatial_connected_20261015_'+s+'.py') for s in ['vertices','vertices_capture','model','calculate','capture']]
    sources += [ROOT/n for n in ['rich_spatial_translation_20261015_model.py','rich_spatial_collective_20261015_model.py','rich_spatial_collective_20261015_calculate.py','rich_spatial_quantum_sea_20261015_model.py','rich_spatial_quantum_sea_20261015_calculate.py','rich_common_colour_lepton_20261015_model.py','rich_colour_composite_20261014_model.py']]
    inputs=[inp]+[ROOT/config[k] for k in ['law_input','primary_result','primary_states','primary_exit','reconstruction']];before={p.name:digest(p) for p in sources+inputs};command=[PYTHON,str(sources[0]),str(inp),str(result)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0');stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt');started=time.time()
    with stdout.open('wb') as so,stderr.open('wb') as se:process=subprocess.run(command,cwd=ROOT,env=env,stdout=so,stderr=se,check=False)
    files=sources+inputs+[stdout,stderr]+[p for p in [result,result.with_suffix('.npz')] if p.exists()];exitpath=ROOT/(prefix+'_exit.json')
    exitpath.write_text(json.dumps(dict(kind='NEW_OWNING_CURRENT_CONTACT_SOURCE_MATRIX_NATURAL_CAPTURE',returncode=process.returncode,natural_exit=True,command=command,cwd=str(ROOT),started_unix=started,finished_unix=time.time(),source_input_state_before=before,source_input_state_after={p.name:digest(p) for p in sources+inputs},captured={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files},scope='Actual owning before-cut current/contact data, never added twice to full-H connected four-point'),indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[exitpath]:z.write(p,p.name)
    print(json.dumps(dict(prefix=prefix,returncode=process.returncode,elapsed=time.time()-started)));sys.exit(process.returncode)


if __name__=='__main__':main()
