"""Load frozen neutral spectra; mutable plan snapshots are provenance, not physics."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.sparse import block_diag
import covariant_collective_cut2_neutral as producer

def load_spectrum(capsule_path: Path):
    cap=json.loads(capsule_path.read_bytes())
    for name,sha in cap['source_hashes_at_import'].items():
        if producer.digest(producer.ROOT/name)!=sha:
            raise RuntimeError('Physical source changed: '+name)
    for name,sha in cap['input_hashes_at_import'].items():
        historical=cap['input_bytes_at_import_utf8'][name].encode('utf-8')
        if hashlib.sha256(historical).hexdigest()!=sha:
            raise RuntimeError('Imported provenance bytes changed: '+name)
        if name=='plan' or name.endswith(':PLAN_BYTES_AT_IMPORT'):
            continue
        actual=producer.INPUT_AT_IMPORT.get(name)
        if actual is None or hashlib.sha256(actual).hexdigest()!=sha:
            raise RuntimeError('Physical input changed: '+name)
    physical=producer.base.space(0,2)
    blocks=[]
    for row in cap['blocks']:
        sector=producer.base.copy_sector(physical,row['sign'])
        P,_,_,_=producer.charge_basis(physical,sector,row['charge_C_sign'])
        total=sector['embedding']@block_diag((P,P,P),format='csr')
        path=Path(row['U_path'])
        if producer.digest(path)!=row['generated_U_sha256']:
            raise RuntimeError('Generated U changed')
        U=np.memmap(path,mode='r',dtype=np.complex128,shape=tuple(row['shape']),order='F')
        blocks.append(dict(V=U,q=np.array(row['original_sigma_per_column']),P=total,
                           sign=row['sign'],charge_C_sign=row['charge_C_sign']))
    spectrum=producer.base.Spectrum(blocks,3*len(physical['physical']))
    return spectrum,cap['controls']
