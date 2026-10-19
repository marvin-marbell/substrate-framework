"""NEW physical protocol consumer after private adjoint-cache naming correction.
Same source law and cuts. Earlier charge2 actual .H actions did not use .T.
"""
from pathlib import Path
from typing import Any
import hashlib,json,resource,sys,time
import numpy as np
from rich_spatial_colour_charge_20261015_snapshot import load
from rich_review_wave18_20261015_law import physical_law
from rich_review_wave18_20261015_charge import evolve,pvm,distance
ROOT=Path(__file__).resolve().parent


def main()->None:
    started=time.monotonic();request:Any=json.loads(Path(sys.argv[1]).read_text());limit=request['memory_limit_MiB']*1024**2;resource.setrlimit(resource.RLIMIT_AS,(limit,limit));body=ROOT/request['body']
    with body.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=request['body_sha256']:raise ValueError('Trusted local numeric snapshot digest mismatch; not unpickled')
    data=load(body);op=data['operators']['neutral'];args=(op,data['rotor'],data['scalar'],data['parameters'],data['geometry']);h=physical_law(*args);a=np.array([.13,-.07,.19,.02,-.04,.03]);ha=physical_law(*args,fields=a);saved=np.load(ROOT/request['neutral_state']);omega=saved['vectors'][:,0];rng=np.random.default_rng(181804);x=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]);x/=np.linalg.norm(x);y=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]);y/=np.linalg.norm(y)
    protocol={}
    for name,operator in [('H0',h),('finite_A_B',ha),('original_source',physical_law(*args,fields=a,derivative=(2,)))]:
        protocol[name]=dict(transpose_identity_error=float(np.linalg.norm(operator.T@x-np.conj(operator.H@np.conj(x)))),adjoint_bilinear_error=float(abs(np.vdot(y,operator@x)-np.vdot(operator.H@y,x))),actual_adjoint_Hermitian_action_error=float(np.linalg.norm(operator.H@x-operator@x)))
    base=evolve(h,omega,request['duration']);driven=evolve(ha,omega,request['duration']);nf=len(data['scalar']['grid']);basis=data['native']['neutral'];q=pvm(omega,basis,nf);j=physical_law(*args,derivative=(2,));result=dict(kind='NEW_SAME_LAW_FIXED_LINEAR_OPERATOR_PROTOCOL_AND_PHYSICAL_SOURCE_EVENT',request=request,meaning='Private cache rename only avoids shadowing scipy LinearOperator._transpose. No arithmetic/cut/coupling/current change; completed charge2 physics uses the archived pre-rename implementation where .T was unused.',protocol=protocol,reference_residual=float(np.linalg.norm(h@omega-float(saved['energies'][0])*omega)),actual_finite_A_B_source_event=dict(duration=request['duration'],fields=a.tolist(),material_charge_excess=distance(pvm(driven,basis,nf),pvm(base,basis,nf)),initial_material_charge=q,original_current_baseline=float(np.vdot(base,j@base).real),original_current_driven=float(np.vdot(driven,j@driven).real),state_difference_norm=float(np.linalg.norm(driven-base)),base_norm=float(np.linalg.norm(base)),driven_norm=float(np.linalg.norm(driven))),elapsed_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output=Path(sys.argv[2]);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
