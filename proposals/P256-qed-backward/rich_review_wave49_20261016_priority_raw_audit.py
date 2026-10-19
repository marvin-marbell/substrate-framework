"""Read OWN committed source tensors, not a Model clone or producer trajectory."""
from pathlib import Path
from itertools import product
import hashlib,json,os,subprocess,sys,time,zipfile
import numpy as np
from scipy.linalg import block_diag
from tenpy.linalg import np_conserved as npc
from tenpy.networks.site import Site

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_';E='transport_priority2';TAG=P+'priority_raw_audit';PY='/home/administrator/substrate-framework/.venv/bin/python'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def encode(v):
    if isinstance(v,np.ndarray):return encode(v.tolist())
    if isinstance(v,(complex,np.complexfloating)):return [float(v.real),float(v.imag)]
    if isinstance(v,dict):return {str(k):encode(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [encode(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    return v


def state_data(meta):
    record=json.loads(meta.read_text());payload=ROOT/record['raw_file']
    if digest(payload)!=record['raw_sha256']:raise ValueError('own committed tensor checksum mismatch')
    with np.load(payload) as raw:
        B=[raw[f'state_B_{i:02}'] for i in range(41)];norm=float(raw['state_norm']);total=raw['state_charge']
    return B,norm,total,record


def environments(B):
    left=[np.ones((1,1),complex)]
    for tensor in B:left.append(np.einsum('ab,apr,bps->rs',left[-1],tensor.conj(),tensor,optimize=True))
    right=[np.ones((1,1),complex)]
    for tensor in reversed(B):right.append(np.einsum('apr,bps,rs->ab',tensor.conj(),tensor,right[-1],optimize=True))
    return left,list(reversed(right))


def product_mean(B,ops,norm):
    env=np.ones((1,1),complex)
    for i,tensor in enumerate(B):
        if i in ops:env=np.einsum('ab,apr,pq,bqs->rs',env,tensor.conj(),ops[i],tensor,optimize=True)
        else:env=np.einsum('ab,apr,bps->rs',env,tensor.conj(),tensor,optimize=True)
    return env[0,0]/norm


def measure(B,norm_metadata,total,field_basis,L,glabels,gperm,rfperm):
    left,right=environments(B);norm=float(left[-1][0,0].real);densities=[]
    for i,tensor in enumerate(B):densities.append(np.einsum('ab,apr,bqs,rs->qp',left[i],tensor.conj(),tensor,right[i+1],optimize=True)/norm)
    n=np.array([np.trace(densities[10*v+2+s]@np.diag([0.,1.])).real for v in range(4) for s in range(8)]).reshape(4,8)
    local=np.array([[np.trace(densities[10*v+1]@a[np.ix_(gperm,gperm)]).real for a in L] for v in range(4)])
    cas=np.diag([j*(j+1) for j,m,k in glabels]);casimir=[np.trace(densities[10*v+1]@cas[np.ix_(gperm,gperm)]).real for v in range(4)]
    sewing=[];number_distributions=[];parity=np.diag([(-1.)**round(2*j) for j,m,k in glabels])[np.ix_(gperm,gperm)]
    for v in range(4):
        ops={10*v+1:parity};ops.update({10*v+2+s:np.diag([1.,-1.]) for s in range(8)});sewing.append(product_mean(B,ops,norm))
        characters=[product_mean(B,{10*v+2+s:np.diag([1.,np.exp(2j*np.pi*k/9)]) for s in range(8)},norm) for k in range(9)]
        number_distributions.append([sum(np.exp(-2j*np.pi*k*number/9)*characters[k] for k in range(9))/9 for number in range(9)])
    inverse=np.argsort(rfperm);rf=densities[40][np.ix_(inverse,inverse)].reshape(35,9,35,9);f=np.einsum('iaib->ab',rf);r=np.einsum('iaja->ij',rf)
    ext=sorted([x for x in product(range(3),repeat=8) if sum(x)<=2],key=lambda x:(sum(x),x));lookup={x:i for i,x in enumerate(ext)};indices=[lookup[tuple(x)] for x in field_basis];momenta=[]
    for axis in range(8):
        a=np.zeros((len(ext),len(ext)),complex)
        for col,x in enumerate(ext):
            if x[axis]:
                target=list(x);target[axis]-=1;a[lookup[tuple(target)],col]=np.sqrt(x[axis])
        momenta.append((a-a.conj().T)/(1j*np.sqrt(2)))
    p=[a[np.ix_(indices,indices)] for a in momenta];pmean=np.array([np.trace(f@a).real for a in p]);cov=np.empty((8,8));wrong=np.empty((8,8))
    for i in range(8):
        for j in range(8):
            full=((momenta[i]@momenta[j]+momenta[j]@momenta[i])/2)[np.ix_(indices,indices)];cov[i,j]=np.trace(f@full).real-pmean[i]*pmean[j];wrong[i,j]=np.trace(f@(p[i]@p[j]+p[j]@p[i])/2).real-pmean[i]*pmean[j]
    radial=np.array([np.diag(densities[10*v]).real for v in range(4)]);boundary=np.diag([float(sum(x)==1) for x in field_basis]);Rboundary=np.diag([0.]*10+[1.]*25)
    return dict(raw_norm2=norm*norm_metadata**2,tensor_norm2=norm,physical_total_and_sewing_charge=total,n_microsites=n,n_cells=n.sum(axis=1),physical_local_LEFT=local,local_LEFT_Casimir=casimir,local_sewing_expectations=sewing,actual_local_N0_through_N8_probabilities=np.array(number_distributions),quantum_cone_radial_node_probabilities=radial,canonical_field_Pi_initial=pmean,full_BEFOREcut_Pi_covariance=cov,WRONG_cut_Pi_squared_covariance=wrong,field_top_total_Fock_shell_weight=float(np.trace(f@boundary).real),material_top_j2_weight=float(np.trace(r@Rboundary).real)),f


def worker():
    inputs=json.loads((ROOT/(TAG+'_input.json')).read_text())
    for name,value in inputs['input_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError(('own raw audit input changed',name))
    code=ROOT/(P+'priority_transport2_source_capsule.zip');sys.path.insert(0,str(code))
    import rich_gauge_band_frame_20261016_model as frozen
    if not frozen.__file__.startswith(str(code)):raise ValueError('LEFT definitions escaped own frozen source snapshot')
    js=np.arange(0,2.01,.5);labels=[(j,m,k) for j in js for m in np.arange(-j,j+1) for k in np.arange(-j,j+1)];L=[block_diag(*[-np.kron(frozen.jmat(j)[a].T,np.eye(round(2*j+1))) for j in js]) for a in range(3)]
    ci=npc.ChargeInfo([1,2,2,2,2],['N','sew0','sew1','sew2','sew3']);q=np.zeros((55,5),int);q[:,1]=[round(2*j)%2 for j,m,k in labels];gperm=Site(npc.LegCharge.from_qflat(ci,q),sort_charge=True).perm;rfperm=Site(npc.LegCharge.from_qflat(ci,np.zeros((315,5),int)),sort_charge=True).perm
    preparation=json.loads((ROOT/(P+E+'_preparation.json')).read_text());states={};raw={};records={}
    for name in ('neutral','charged'):
        B,norm,total,record=state_data(ROOT/(P+E+'_'+name+'_initial_stage.json'));states[name],f=measure(B,norm,total,preparation['field_basis'],L,labels,gperm,rfperm);raw[name+'_field_density']=f;records[name]=record
    charge=-(np.asarray(states['charged']['n_cells'])-states['neutral']['n_cells']);spin=np.asarray(states['charged']['physical_local_LEFT'])-states['neutral']['physical_local_LEFT'];global_label=spin.sum(axis=0);fractions=np.asarray(states['charged']['n_cells'])/sum(states['charged']['n_cells']);wrong=fractions[:,None]*global_label;error=float(np.linalg.norm(wrong-spin))
    original=ROOT/preparation['neutral_source_receipt']['raw_file'];current=ROOT/records['neutral']['raw_file'];entry_equal=True;bytes_equal=True
    with np.load(original) as old,np.load(current) as new:
        for i in range(41):
            a=old[f'state_B_{i:02}'];b=new[f'state_B_{i:02}'];entry_equal&=np.array_equal(a,b);bytes_equal&=a.tobytes()==b.tobytes()
    assertions=dict(fullCAR32_source_charge=bool(abs(charge.sum()+1)<1e-9),actual_physical_LEFT_source=bool(abs(spin[0,2])>.1),WRONG_n_over_N_global_spin_density_rejected=error>.1,local_central_sewing=all(abs(x-1)<1e-9 for state in states.values() for x in state['local_sewing_expectations']),raw_checkpoint_entry_values_preserved=bool(entry_equal))
    result=dict(kind='OWN_FULLCAR32_COMMITTED_SOURCE_RAW_NETWORK_PHYSICAL_AUDIT_NO_MODEL_CLONE',physical_time=.004,states=states,source_excess_physical_charge=charge,source_excess_actual_local_LEFT=spin,WRONG_global_spin_times_occupied_n_over_N_density=wrong,WRONG_density_error=error,recovery_precision=dict(all_B_values_exactly_equal=bool(entry_equal),all_B_bytes_equal=bool(bytes_equal),scope='Previous recovery certificate used np.array_equal for its bitwise label; THIS is the distinct actual tobytes comparison. Both metadata and physical phases preserved; no evolved-state replay.'),physics_assertions=assertions,source_quality=preparation['source_quality'],basis_order_binding=records['charged']['binding']['basis_order_sha256'],operator_sha256=preparation['compiled_operator_sha256'],method='Independent dense raw-MPS left/right environments, analytic native LEFT matrices with actual Site permutation and direct sewing/number generating-function contractions. No cached Model, native observable MPO or producer state loaded.',limits='This audits OWN source at t.004 ONLY, not its pending finite-time carrier trajectory. r is quantum cone coordinate, not charge-shell radius. Pi is canonical mode momentum, not SI E. Full before-cut covariance differs from squaring cutPi; initial field receiver not judged by means alone. n/N is inclusive particle POVM, not an excitation PVM. Finite source compression chi16 and all g/R/field/exterior debts remain.')
    np.savez_compressed(ROOT/(TAG+'_raw.npz'),**raw);(ROOT/(TAG+'_result.json')).write_text(json.dumps(encode(result),indent=2)+'\n');print(json.dumps(encode(result)),flush=True);assert all(assertions.values()),assertions


def capture():
    files=[ROOT/(P+E+'_preparation.json'),ROOT/(P+'priority_transport2_source_capsule.zip')]
    for name in ('neutral','charged'):
        meta=ROOT/(P+E+'_'+name+'_initial_stage.json');files.extend((meta,ROOT/json.loads(meta.read_text())['raw_file']))
    inputs=dict(source_sha256=digest(__file__),input_sha256={path.name:digest(path) for path in files},meaning='Independent read-only analysis of OWN committed fullCAR32 source state. No Model allocation, producer state or new evolution. Old source and current physical trajectory remain immutable.')
    (ROOT/(TAG+'_input.json')).write_text(json.dumps(inputs,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(TAG+'_source_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as archive:archive.write(__file__,Path(__file__).name);archive.write(ROOT/(TAG+'_input.json'),TAG+'_input.json')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');start=time.time()
    with (ROOT/(TAG+'_stdout.txt')).open('x') as stdout,(ROOT/(TAG+'_stderr.txt')).open('x') as stderr:child=subprocess.run([PY,__file__,'worker'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    report=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-start,source_unchanged=digest(__file__)==inputs['source_sha256'],inputs_unchanged={path.name:digest(path)==inputs['input_sha256'][path.name] for path in files});(ROOT/(TAG+'_exit.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(child.returncode)

if __name__=='__main__':
    if len(sys.argv)>1:worker()
    else:capture()
