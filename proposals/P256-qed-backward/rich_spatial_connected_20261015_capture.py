"""NEW exact-byte cached-owning-H connected consumer natural capture."""
import hashlib,json,os
from pathlib import Path
import subprocess,sys,time,zipfile
ROOT=Path(__file__).resolve().parent
PYTHON='/home/administrator/substrate-framework/.venv/bin/python'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prefix=sys.argv[1];inp=ROOT/(prefix+'_input.json');result=ROOT/(prefix+'_result.json');config=json.loads(inp.read_text())
    sources=[ROOT/('rich_spatial_connected_20261015_'+s+'.py') for s in ['model','calculate','capture']]
    sources += [ROOT/n for n in ['rich_spatial_orthogonal_20261015_model.py','rich_spatial_orthogonal_20261015_tensor.py','rich_spatial_orthogonal_20261015_calculate.py','rich_spatial_orthogonal_20261015_capture.py','rich_spatial_orthogonal_20261015_probe.py','rich_spatial_helicity_20261015_model.py','rich_spatial_translation_20261015_model.py','rich_spatial_collective_20261015_model.py','rich_spatial_collective_20261015_calculate.py','rich_spatial_quantum_sea_20261015_model.py','rich_spatial_quantum_sea_20261015_calculate.py','rich_common_colour_lepton_20261015_model.py','rich_colour_composite_20261014_model.py']]
    inputs=[inp]+[ROOT/config[k] for k in ['law_input','orthogonal_input','orthogonal_result','orthogonal_exit','reconstruction']]
    before={p.name:digest(p) for p in sources+inputs};command=[PYTHON,str(sources[1]),str(inp),str(result)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0');stdout=ROOT/(prefix+'_stdout.txt');stderr=ROOT/(prefix+'_stderr.txt');started=time.time()
    with stdout.open('wb') as so,stderr.open('wb') as se:process=subprocess.run(command,cwd=ROOT,env=env,stdout=so,stderr=se,check=False)
    outputs=[result,result.with_suffix('.npz')]+[ROOT/(prefix+suffix) for suffix in ['_states.json','_states.npz','_checkpoint.json','_checkpoint.npz']]
    files=sources+inputs+[stdout,stderr]+[p for p in outputs if p.exists()];exitpath=ROOT/(prefix+'_exit.json')
    exitpath.write_text(json.dumps(dict(kind='NEW_SAME_H_G_D_CONNECTED_FOUR_POINT_NATURAL_CAPTURE_NOT_REVIEW',returncode=process.returncode,natural_exit=True,command=command,cwd=str(ROOT),started_unix=started,finished_unix=time.time(),source_input_cache_before=before,source_input_cache_after={p.name:digest(p) for p in sources+inputs},captured={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files},environment={k:env[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED']},scope='All24 time orders, canonical addition/removal, owning two-points and raw connected tensor, exact-momentum fiber and own factorized EM-off control; no on-shell photon/electron/cross-section claim'),indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(prefix+'_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[exitpath]:z.write(p,p.name)
    print(json.dumps(dict(prefix=prefix,returncode=process.returncode,elapsed=time.time()-started)));sys.exit(process.returncode)


if __name__=='__main__':main()
