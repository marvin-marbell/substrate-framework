"""Whitelisted pure-source inputs only; never load the12GB solved trajectories."""
from pathlib import Path
import ast,hashlib,io,json,os,subprocess,sys,time,zipfile

ROOT=Path(__file__).resolve().parent;P='rich_full108_recoil_review_20261017_';A='rich_recoil_current_wave_20261016_';PY='/home/administrator/substrate-framework/.venv/bin/python'
EXPECTED='e3e1c4df95c45abefb96c8ef6c462ac9daeba36006b73a2646b38c2f05bd260d'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class SemanticBody(ast.NodeTransformer):
    def visit_AnnAssign(self,node):
        return None if node.value is None else ast.Assign(targets=[node.target],value=self.visit(node.value))
    def visit_Call(self,node):
        node=self.generic_visit(node)
        return node.args[-1] if isinstance(node.func,ast.Name) and node.func.id=='cast' else node


def body(source,name):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    statements=node.body[1:] if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str) else node.body
    module=SemanticBody().visit(ast.Module(body=statements,type_ignores=[]));return ast.dump(module,include_attributes=False)


def freeze():
    receipt=ROOT/(P+'frozen_input_receipt.json');target=ROOT/(P+'frozen_code.zip')
    if receipt.exists():
        data=json.loads(receipt.read_text())
        if digest(target)!=data['code_snapshot_sha256']:raise ValueError('frozen code changed')
        return data
    capsule=ROOT/(A+'delivery_capsule.zip')
    if digest(capsule)!=EXPECTED:raise ValueError('authoritative full108 capsule hash mismatch')
    manifest=json.loads((ROOT/(A+'manifest.json')).read_text())['files'];names=[A+'actual6_input.json',A+'actual6_source_capsule.zip',A+'emit3_source_capsule.zip'];payload={}
    with zipfile.ZipFile(capsule) as archive:
        for name in names:
            payload[name]=archive.read(name)
            if hashlib.sha256(payload[name]).hexdigest()!=manifest[name]['sha256']:raise ValueError(('frozen member hash mismatch',name))
    with zipfile.ZipFile(io.BytesIO(payload[names[1]])) as old,zipfile.ZipFile(io.BytesIO(payload[names[2]])) as new:
        oldhash=json.loads(old.read('source_sha256.json'));newhash=json.loads(new.read('source_sha256.json'))
        for name,value in newhash.items():
            if hashlib.sha256(new.read(name)).hexdigest()!=value:raise ValueError(('code member hash mismatch',name))
        dependencies={name:oldhash[name]==newhash.get(name) for name in oldhash if not name.startswith(A)}
        if not all(dependencies.values()):raise ValueError(('old pure dependency changed',dependencies))
        model=A+'model.py';functions={name:body(old.read(model).decode(),name)==body(new.read(model).decode(),name) for name in ('single_particle','field_representation')}
        if not all(functions.values()):raise ValueError(('primitive physical constructor body changed',functions))
    target.write_bytes(payload[names[2]]);physical=json.loads(payload[names[0]])
    data=dict(capsule_sha256=EXPECTED,capsule_bytes=capsule.stat().st_size,whitelisted_members={name:hashlib.sha256(value).hexdigest() for name,value in payload.items()},code_snapshot=target.name,code_snapshot_sha256=digest(target),frozen_model_sha256=newhash[A+'model.py'],actual6_model_sha256=oldhash[A+'model.py'],unchanged_actual6_pure_dependencies=dependencies,actual6_vs_current_primitive_bodies_equal_after_type_doc_normalization=functions,law=physical['law'],meaning='Only tiny pure-source snapshots/physical input read from archive. NO solved/evolved vectors, Model pickle, response constants or author producerCLI loaded. Current source is not falsely relabelled actual6 source; applicable primitive operator functions/dependencies explicitly compared.')
    receipt.write_text(json.dumps(data,indent=2)+'\n');return data


def main():
    attempt=sys.argv[1];tag=P+attempt;frozen=freeze();config=dict(plan_id='issue222-rich-backward-candidates-20261014',captured_plan_revision='fcb655e3c72a7392e792ec68a543ef29482fab9706aaa11eb6bacbe22e745564',template='general@89cd9d45725622e0d65429eca9f2309508c255b54950c3a75498b6f913600efe',work_item='independent-review',frozen=frozen,kick_strength=.23,own_phi_center=1.05,own_phi_width=.31,scope='Own full108 Slater/CAR and75 quantum-boson exact low-order Wick/twisted-C3 response, NOT producer nine t=.2 replay or different-law toy.')
    stage=sys.argv[2] if len(sys.argv)>2 else 'run'
    if stage not in ('run','analysis'):raise ValueError(stage)
    config['stage']=stage
    if stage=='analysis':
        config['OWN_physics_epoch']='wick1'
        config['OWN_physics_files']={suffix:digest(ROOT/(P+'wick1'+suffix)) for suffix in ('_input.json','_exit.json','_result.json','_raw.npz','_source_inputs.zip')}
    path=ROOT/(tag+'_input.json');path.write_text(json.dumps(config,indent=2)+'\n');sources=sorted(ROOT.glob(P+'*.py'));hashes={p.name:digest(p) for p in sources}
    with zipfile.ZipFile(ROOT/(tag+'_source_inputs.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for source in sources:archive.write(source,source.name)
        for source in (path,ROOT/(P+'frozen_input_receipt.json'),ROOT/(P+'frozen_code.zip')):archive.write(source,source.name)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');started=time.time();command=[PY,str(ROOT/(P+stage+'.py')),str(path)]
    with (ROOT/(tag+'_stdout.txt')).open('x') as stdout,(ROOT/(tag+'_stderr.txt')).open('x') as stderr:child=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    report=dict(natural_exit=child.returncode,elapsed_seconds=time.time()-started,command=command,own_sources_unchanged={source.name:digest(source)==hashes[source.name] for source in sources},frozen_code_unchanged=digest(ROOT/(P+'frozen_code.zip'))==frozen['code_snapshot_sha256'])
    (ROOT/(tag+'_exit.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(child.returncode)

if __name__=='__main__':main()
