"""Capture a new physical consumer, natural return, exact bytes and finite inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    prefix = sys.argv[1]
    inp = ROOT / (prefix + '_input.json'); result = ROOT / (prefix + '_result.json')
    sources = [ROOT / ('rich_common_colour_lepton_20261015_' + suffix + '.py') for suffix in ['model', 'calculate', 'capture']]
    sources.append(ROOT / 'rich_colour_composite_20261014_model.py')
    hashes = {p.name:digest(p) for p in sources + [inp]}
    command = [PYTHON, str(sources[1]), str(inp), str(result)]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    stdout = ROOT / (prefix + '_stdout.txt'); stderr = ROOT / (prefix + '_stderr.txt')
    started = time.time()
    with stdout.open('wb') as out, stderr.open('wb') as err:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err, check=False)
    files = sources + [inp, stdout, stderr]
    if result.exists():
        files.append(result)
    if result.with_suffix('.npz').exists():
        files.append(result.with_suffix('.npz'))
    exit_path = ROOT / (prefix + '_exit.json')
    metadata = dict(kind='NEW_COMMON_ROOT_PHYSICAL_CONSUMER_NATURAL_EXIT_NOT_REVIEW', command=command, cwd=str(ROOT),
        returncode=process.returncode, natural_exit=True, started_unix=started, finished_unix=time.time(),
        source_and_input_before=hashes, source_and_input_after={p.name:digest(p) for p in sources + [inp]},
        captured={p.name:dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files},
        environment={key:env[key] for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED']},
        limitations='No build/lint/tests/formatters. Physical calculation only. Exact closure is conditional scalar-density population closure, not vector Dirac pair dynamics. Scientific admission belongs to later explicitly opened independent review.')
    exit_path.write_text(json.dumps(metadata, indent=2) + '\n')
    with zipfile.ZipFile(ROOT / (prefix + '_capsule.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files + [exit_path]:
            archive.write(path, path.name)
    print(json.dumps(dict(prefix=prefix, returncode=process.returncode, elapsed=time.time()-started, result=str(result), exit=str(exit_path))))
    sys.exit(process.returncode)

if __name__ == '__main__':
    main()
