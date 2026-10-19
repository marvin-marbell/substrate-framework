"""Capture a new physical consumer with immutable inputs and natural exit receipts."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def closure(path, found):
    if path in found:
        return
    found.add(path)
    if path.suffix != '.py':
        return
    for node in ast.walk(ast.parse(path.read_text())):
        names = [node.module] if isinstance(node, ast.ImportFrom) else [item.name for item in node.names] if isinstance(node, ast.Import) else []
        for name in names:
            if name and name.startswith('rich_'):
                other = ROOT/(name+'.py')
                if other.exists():
                    closure(other, found)


def limit():
    maximum = 3800*1024**2
    resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))


def main():
    prefix = sys.argv[1]
    inp = ROOT/(prefix+'_input.json')
    p = json.loads(inp.read_text())
    consumer = ROOT/p['consumer']
    sources = {inp, Path(__file__).resolve()}
    closure(consumer, sources)
    sources.update(ROOT/p[key] for key in ['native_body', 'state_file', 'result_file'])
    capsule = ROOT/(prefix+'_capsule.zip')
    if capsule.exists() or (ROOT/(prefix+'_exit.json')).exists():
        raise RuntimeError('Choose a new immutable attempt')
    before = {x.name:sha(x) for x in sorted(sources)}
    with zipfile.ZipFile(ROOT/(prefix+'_source_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            archive.write(path, path.name)
    command = ['/home/administrator/substrate-framework/.venv/bin/python', str(consumer), str(inp), str(ROOT/(prefix+'_result.json'))]
    stdout = ROOT/(prefix+'_stdout.txt')
    stderr = ROOT/(prefix+'_stderr.txt')
    started = time.time()
    with stdout.open('wb') as out, stderr.open('wb') as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, preexec_fn=limit, env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        code = process.wait()
    after = {x.name:sha(x) for x in sorted(sources)}
    files = sorted(sources | set(ROOT.glob(prefix+'_result*')) | {stdout, stderr})
    receipt = dict(command=command, pid=process.pid, returncode=code, natural_exit=code>=0, successful=code==0, seconds=time.time()-started, peak_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, address_space_limit_MiB=3800, no_timer_deadline=True, source_before=before, source_after=after, source_unchanged=before==after, files={x.name:dict(bytes=x.stat().st_size, sha256=sha(x)) for x in files})
    exitpath = ROOT/(prefix+'_exit.json')
    exitpath.write_text(json.dumps(receipt, indent=2)+'\n')
    with zipfile.ZipFile(capsule, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files+[exitpath]:
            archive.write(path, path.name)
    print(json.dumps(dict(returncode=code, natural_exit=code>=0, successful=code==0, source_unchanged=before==after, seconds=receipt['seconds'], peak_child_RSS_KiB=receipt['peak_child_RSS_KiB'], capsule=capsule.name, capsule_sha256=sha(capsule))), flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
