"""Read authentic40 non-RF sites and stream derivedRF arrays; NO Model clone."""
from pathlib import Path
from types import SimpleNamespace
import gc,hashlib,inspect,json,math,os,pickle,resource,subprocess,sys,time,zipfile
import numpy as np
from tenpy.networks.site import Site
from tenpy.linalg import np_conserved as npc
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingModel
from tenpy.networks.mpo import MPOEnvironment
from rich_review_wave49_20261016_priority_raw_audit import state_data,encode,digest,environments
from rich_review_wave49_20261016_priority_storage import restore

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_';E='transport_priority2';TAG=P+'priority_stream_current';PY='/home/administrator/substrate-framework/.venv/bin/python';BASE='cc6eb96a1d442b19ee3c4719370a3b6b257e75eb6ea956c7ab836f9ce432b803'
SITES=[]


class ReachedRF(Exception):pass


class PrefixSite(Site):
    def __new__(cls,*args,**kwargs):
        # Frozen cc6 schema: four(r3,g55,eightCAR2) then RF315. Stop BEFORE
        # decoding any of the huge RF site's operator dictionary.
        if len(SITES)==40:raise ReachedRF()
        value=object.__new__(cls);SITES.append(value);return value


class PrefixReader(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='tenpy.networks.site' and name=='Site':return PrefixSite
        return super().find_class(module,name)


def prefix_sites(path):
    if digest(path)!=BASE:raise ValueError('frozen operator input hash mismatch')
    with Path(path).open('rb') as stream:
        reader=PrefixReader(stream)
        try:reader.load()
        except ReachedRF:pass
        else:raise ValueError('expected immutable RF boundary not reached; do not load a full Model here')
        position=stream.tell()
    del reader;gc.collect()
    if len(SITES)!=40 or [s.dim for s in SITES]!=[d for _ in range(4) for d in [3,55]+[2]*8]:raise ValueError('wrong frozen physical-site schema')
    return SITES,dict(authentic_nonRF_site_count=40,parsed_pickle_bytes_before_RF=position,full_Model_returned=False,source='Actual operator-only cc6 pickle; native Site __new__ intercepted only to stop at known RF boundary. All retained Site data are genuine decoded operators, not reconstructed/mock arrays.')


def body_environments(B,patterns,sites,operator_cache):
    stack=[np.ones((1,1),complex)];previous=None;result={}
    for sequence in sorted(patterns):
        common=0 if previous is None else next((i for i in range(40) if sequence[i]!=previous[i]),40)
        stack=stack[:common+1]
        for i in range(common,40):
            name=sequence[i];tensor=B[i]
            if name=='Id':value=np.einsum('ab,apr,bps->rs',stack[-1],tensor.conj(),tensor,optimize=True)
            else:
                key=(i,name)
                if key not in operator_cache:operator_cache[key]=sites[i].get_op(name).to_ndarray()
                value=np.einsum('ab,apr,pq,bqs->rs',stack[-1],tensor.conj(),operator_cache[key],tensor,optimize=True)
            stack.append(value)
        result[sequence]=stack[-1];previous=sequence
    return result


def prepare_cache(mode,source_hash,binding):
    stem=P+'zero_source_repair2_J'+str(mode);meta=ROOT/(stem+'_operator.json');data=json.loads(meta.read_text());raw=ROOT/(stem+'_operator.npz')
    for key in ('base_operator_sha256','law_signature','basis_order_sha256'):
        expected=BASE if key=='base_operator_sha256' else binding[key]
        if data['binding'][key]!=expected:raise ValueError(('current source binding mismatch',mode,key))
    if data['binding']['source_function_sha256']!=source_hash or digest(raw)!=data['raw_sha256']:raise ValueError('derived canonical source/input changed')
    groups={};patterns=set()
    for pair,factors in data['terms']:
        field=[name for name,site in factors if site==40]
        if len(field)!=1 or field[0] not in data['RF_arrays']:raise ValueError('streamed cache is not self-contained for RF')
        sequence=['Id']*40
        for name,site in factors:
            if site!=40:sequence[site]=name
        pattern=tuple(sequence);patterns.add(pattern);groups.setdefault(field[0],[]).append((complex(*pair),pattern,factors))
    return data,raw,groups,patterns


def worker():
    inputs=json.loads((ROOT/(TAG+'_input.json')).read_text())
    for name,value in inputs['input_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError(('own current input changed',name))
    snapshot=ROOT/(P+'priority_transport2_source_capsule.zip');sys.path.insert(0,str(snapshot))
    import rich_local_frame_spatial_carrier_20261016_model as frozen
    if not frozen.__file__.startswith(str(snapshot)):raise ValueError('canonical source definitions escaped own frozen snapshot')
    source_hash=hashlib.sha256(inspect.getsource(frozen.Model.lab_source).encode()).hexdigest();sites,partial=prefix_sites(ROOT/'rich_local_frame_spatial_carrier_20261016_model.pkl');states={};binding=None
    for age in ('initial','full'):
        for kind in ('neutral','charged'):
            key=kind+'_'+age;meta=ROOT/(P+E+'_'+kind+'_'+age+'_stage.json');B,norm,total,record=state_data(meta);left,_=environments(B);normB=float(left[-1][0,0].real);states[key]=dict(B=B,norm=norm,normB=normB,record=record)
            binding=record['binding']
    caches={};patterns=set()
    for mode in (0,4):
        data,raw,groups,selected=prepare_cache(mode,source_hash,binding);caches[mode]=(data,raw,groups);patterns|=selected
    for pattern in patterns:
        for i,name in enumerate(pattern):
            if name not in sites[i].opnames:raise ValueError(('actual frozen nonRF operator missing',i,name))
    operators={}
    for state in states.values():state['body']=body_environments(state['B'],patterns,sites,operators)
    values={};primitive=None
    for mode,(data,raw,groups) in caches.items():
        pieces={key:[] for key in states}
        with zipfile.ZipFile(raw) as archive:
            for fieldname,terms in groups.items():
                with archive.open(data['RF_arrays'][fieldname]+'.npy') as stream:matrix=np.load(stream,allow_pickle=False)
                if matrix.shape!=(315,315):raise ValueError('wrong frozen RF dimension')
                for key,state in states.items():
                    R=state['B'][40][:,:,0];transfer=R.conj()@matrix@R.T
                    for coefficient,pattern,factors in terms:
                        value=coefficient*np.sum(state['body'][pattern]*transfer)/state['normB'];pieces[key].append(value)
                        if key=='charged_initial' and (primitive is None or abs(value)>primitive['magnitude']):primitive=dict(magnitude=float(abs(value)),value=complex(value),coefficient=coefficient,factors=factors,matrix=matrix.copy(),mode=mode)
        values[str(mode)]={key:complex(math.fsum(v.real for v in items),math.fsum(v.imag for v in items)) for key,items in pieces.items()}
        interim=dict(kind='OWN_FULL_CANONICAL_CURRENT_STREAMED_FROM_INPUT_BOUND_OPERATORS',values=values,partial_operator_reader=partial,raw_source_derivative_method_sha256=source_hash,method='Exact operator sum mean on OWN completed MPS states, all2960terms including mechanical source; stream one RF315matrix at a time and cache only small body environments. No full Model, new derivative inventory, source convention or dynamics.',limits='Finite-state exact contraction only; does NOT improve chi16/dt/field/exterior accuracy, imply all lab charge motionOFF from a canonical sourceOFF, or establish photons/current range/mass.')
        (ROOT/(TAG+'_partial_result.json')).write_text(json.dumps(encode(interim),indent=2)+'\n')
    # Actual native-MPO positive COMPLEX action on the strongest single term.
    # Basis context contains REAL decoded sites; no H or Model placeholder.
    ci=sites[0].leg.chinfo;rf=Site(npc.LegCharge.from_qflat(ci,np.zeros((315,5),int)),sort_charge=True);all_sites=sites+[rf]
    if not np.array_equal(rf.perm,np.arange(315)):raise ValueError('RF permutation is not identity in selected frozen law')
    rf.add_op('ReviewCurrentRF',primitive['matrix'],need_JW=False,hc=False,permute_dense=True);factors=[('ReviewCurrentRF' if site==40 else name,site) for name,site in primitive['factors']];lattice=Lattice([1],all_sites,bc='open',bc_MPS='finite');native=CouplingModel(lattice);native.add_local_term(primitive['coefficient'],[(name,[0,site]) for name,site in factors]);mpo=native.calc_H_MPO()
    meta=states['charged_initial']['record']
    with np.load(ROOT/meta['raw_file']) as arrays:psi=restore(SimpleNamespace(sites=all_sites,chinfo=ci),arrays,'state')
    actual=psi.norm**2*MPOEnvironment(psi,mpo,psi).full_contraction(0);expected=primitive['value']*states['charged_initial']['normB']*psi.norm**2;error=float(abs(actual-expected))
    if error>2e-11 or abs(actual)<1e-10:raise ValueError(('native streamed positive primitive failed',actual,expected,error))
    interim['native_positive_primitive']=dict(mode=primitive['mode'],actual=actual,independent_raw_network=expected,absolute_error=error,coefficient=primitive['coefficient'],raw_magnitude=float(abs(actual)))
    interim['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;interim['RF_cache_receipts']={str(mode):dict(terms=len(data['terms']),RF_arrays=len(data['RF_arrays']),raw_sha256=data['raw_sha256'],binding=data['binding']) for mode,(data,raw,groups) in caches.items()}
    (ROOT/(TAG+'_result.json')).write_text(json.dumps(encode(interim),indent=2)+'\n');print(json.dumps(encode(interim)),flush=True)


def capture():
    files=[ROOT/(P+'priority_transport2_source_capsule.zip')]
    for age in ('initial','full'):
        for kind in ('neutral','charged'):
            meta=ROOT/(P+E+'_'+kind+'_'+age+'_stage.json');files.extend((meta,ROOT/json.loads(meta.read_text())['raw_file']))
    for mode in (0,4):files.append(ROOT/(P+'zero_source_repair2_J'+str(mode)+'_operator.json'))
    inputs=dict(source_sha256=digest(__file__),operator_sha256=BASE,input_sha256={path.name:digest(path) for path in files},scope='Actual input-bound canonical source readout on own completed states; no cached Model allocation or field-response prerequisite. RF payload hashes checked in worker; one Model-owned live reciprocal path remains separate.')
    (ROOT/(TAG+'_input.json')).write_text(json.dumps(inputs,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/(TAG+'_source_capsule.zip'),'x',zipfile.ZIP_DEFLATED) as archive:archive.write(__file__,Path(__file__).name);archive.write(ROOT/(TAG+'_input.json'),TAG+'_input.json')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');started=time.time()
    with (ROOT/(TAG+'_stdout.txt')).open('x') as stdout,(ROOT/(TAG+'_stderr.txt')).open('x') as stderr:child=subprocess.run([PY,__file__,'worker'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    receipt=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-started,source_unchanged=digest(__file__)==inputs['source_sha256'],inputs_unchanged={path.name:digest(path)==inputs['input_sha256'][path.name] for path in files});(ROOT/(TAG+'_exit.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True);sys.exit(child.returncode)

if __name__=='__main__':
    if len(sys.argv)>1:worker()
    else:capture()
