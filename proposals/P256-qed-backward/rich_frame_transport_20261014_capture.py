"""Natural-exit capture for the NEW FrameGauss transport consumer."""
import hashlib, json, platform, subprocess, sys, time, zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
source=root/'rich_frame_transport_20261014_calculate.py'
model=root/'rich_frame_gauss_20261014_model.py'
input_path=root/'rich_frame_transport_20261014_input.json'
result=root/'rich_frame_transport_20261014_result.json'
stdout_path=root/'rich_frame_transport_20261014_stdout.txt'
stderr_path=root/'rich_frame_transport_20261014_stderr.txt'
exit_path=root/'rich_frame_transport_20261014_exit.json'
equations=root/'rich_frame_transport_20261014_equations.json'
interface=root/'rich_frame_transport_20261014_interface.json'
synthesis=root/'rich_frame_transport_20261014_synthesis.json'
capsule=root/'rich_frame_transport_20261014_capsule.zip'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
source_before=sha(source); started=time.time()
proc=subprocess.run([sys.executable,str(source)],cwd=root,capture_output=True)
elapsed=time.time()-started
stdout_path.write_bytes(proc.stdout); stderr_path.write_bytes(proc.stderr)
receipt={'command':[sys.executable,str(source)],'cwd':str(root),'returncode':proc.returncode,'elapsed_seconds':elapsed,
 'python':sys.version,'platform':platform.platform(),'source_sha256_before':source_before,'source_sha256_after':sha(source),
 'model_sha256':sha(model),'input_sha256':sha(input_path),'result_sha256':sha(result) if result.exists() else None,
 'stdout_sha256':sha(stdout_path),'stderr_sha256':sha(stderr_path),'natural_process_exit':True}
exit_path.write_text(json.dumps(receipt,indent=2)+'\n')
with zipfile.ZipFile(capsule,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for path in [Path(__file__),source,model,input_path,result,equations,interface,synthesis,stdout_path,stderr_path,exit_path]: z.write(path,path.name)
print(json.dumps(receipt,indent=2))
raise SystemExit(proc.returncode)
