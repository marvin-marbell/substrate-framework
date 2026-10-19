"""Durable OWN-state stages and derivative-only operator data; never a Model copy."""
from pathlib import Path
import hashlib
import inspect
import json
import zipfile
import numpy as np
from tenpy.networks.mps import MPS

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'
EPOCH='zero_source_repair2'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fingerprint(psi):
    h=hashlib.sha256();h.update(np.asarray(psi.norm).tobytes())
    for i in range(psi.L):
        for value in (psi.get_B(i,form='B').to_ndarray(),np.asarray(psi.get_SL(i))):
            h.update(repr(value.shape).encode());h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def basis_binding(m):
    order=(m.g['labels'],m.material['labels'],m.basis.tolist(),m.rnodes.tolist(),m.gsites,m.rsites,m.fsites,m.rfsite,[site.perm.tolist() for site in m.sites],[site.leg.to_qflat().tolist() for site in m.sites])
    return hashlib.sha256(repr(order).encode()).hexdigest()


def save_state(arrays,label,psi):
    arrays[label+'_norm']=np.array(psi.norm);arrays[label+'_charge']=np.asarray(psi.get_total_charge(only_physical_legs=True))
    for i in range(psi.L):
        arrays[f'{label}_B_{i:02}']=psi.get_B(i,form='B').to_ndarray();arrays[f'{label}_S_{i:02}']=np.asarray(psi.get_SL(i))


def restore(m,raw,label):
    tensors=[raw[f'{label}_B_{i:02}'].transpose(1,0,2) for i in range(len(m.sites))]
    singular=[raw[f'{label}_S_{i:02}'] for i in range(len(m.sites))]+[np.ones(1)]
    state=MPS.from_Bflat(m.sites,tensors,singular,bc='finite',permute=False,form='B',unit_cell_width=1)
    state.norm=float(raw[label+'_norm'])
    if not np.isfinite(state.norm) or state.norm<0:raise ValueError('invalid restored norm metadata')
    return state


def stage_paths(label):
    if not label or any(not(c.isalnum() or c=='_') for c in label):raise ValueError('stage needs stable explicit label')
    stem=P+EPOCH+'_'+label
    return ROOT/(stem+'_state.npz'),ROOT/(stem+'_stage.json')


def stage_binding(m,operator,psi,dt,steps,chi,initial_time):
    terms=operator.terms if hasattr(operator,'terms') else None
    return dict(base_operator_sha256=m._review_base_sha,law_signature=m.signature,basis_order_sha256=basis_binding(m),input_state_sha256=fingerprint(psi),dt=dt,steps=steps,chi=chi,initial_physical_time=initial_time,final_physical_time=initial_time+dt*steps,law_variant='full_frozen_H' if operator is m.H else 'explicit_filtered_or_control_H',operator_terms_sha256=hashlib.sha256(repr(terms).encode()).hexdigest())


def load_stage(m,label,binding):
    rawpath,meta=stage_paths(label)
    if not meta.exists():return None
    record=json.loads(meta.read_text())
    if record['binding']!=binding:raise ValueError(('stage input mismatch',label))
    rawpath=ROOT/record['raw_file']
    if digest(rawpath)!=record['raw_sha256']:raise ValueError(('stage raw hash mismatch',label))
    with np.load(rawpath) as raw:state=restore(m,raw,'state')
    # Stored tensors retain physical phases; metadata includes cumulative raw
    # amplitude, whereas the interrupted inner loop used normalized stage states.
    state.norm/=np.exp(record['removed'])
    return state,record


def save_stage(m,label,binding,normalized_state,removed,ledgers):
    rawpath,meta=stage_paths(label);physical=normalized_state.copy();physical.norm*=np.exp(removed)
    rawpath=rawpath.with_name(rawpath.stem+'_s'+str(len(ledgers)).zfill(3)+'.npz')
    arrays={};save_state(arrays,'state',physical)
    # A raw file without its matching receipt is not resumable after interruption.
    np.savez_compressed(rawpath,**arrays)
    record=dict(binding=binding,completed_steps=len(ledgers),removed=removed,ledgers=ledgers,raw_file=rawpath.name,raw_sha256=digest(rawpath),generation=dict(storage_source_sha256=digest(__file__),evolve_source_sha256=hashlib.sha256(inspect.getsource(type(m.H).evolve).encode()).hexdigest()),meaning='Own physically executed stage only; not a producer state. B tensors contain all phases. Removed real raw normalization factors are retained separately, not interpreted as Born probabilities or convergence bounds.')
    record['completed_physical_time']=binding['initial_physical_time']+binding['dt']*len(ledgers)
    record['generation']['source_input_receipt']=m._review_capture_receipt
    temporary=meta.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(meta)


def source_operator(m,mode,other=None,terms=None,label=None):
    """Cache ONLY derived RF arrays and term descriptors, never the base Model."""
    label=label or ('J'+str(mode) if other is None else 'K'+str(mode)+'_'+str(other))
    stem=P+EPOCH+'_'+label;meta=ROOT/(stem+'_operator.json');rawpath=ROOT/(stem+'_operator.npz')
    selected=m.terms if terms is None else terms
    binding=dict(base_operator_sha256=m._review_base_sha,law_signature=m.signature,basis_order_sha256=basis_binding(m),source_function_sha256=hashlib.sha256(inspect.getsource(type(m).lab_source).encode()).hexdigest(),mode=mode,other=other,selected_terms_sha256=hashlib.sha256(repr(selected).encode()).hexdigest())
    if meta.exists():
        data=json.loads(meta.read_text())
        if data['binding']!=binding or digest(rawpath)!=data['raw_sha256']:raise ValueError(('derived operator cache binding mismatch',label))
        replacement={}
        with np.load(rawpath) as raw:
            for oldname,key in data['RF_arrays'].items():replacement[oldname]=m.register(m.rfsite,raw[key])
        rebuilt=[(complex(*c),[(replacement.get(name,name) if site==m.rfsite else name,site) for name,site in factors]) for c,factors in data['terms']]
        return m.mpo(rebuilt)
    original=m.terms
    try:
        m.terms=selected;operator=m.lab_source(mode,other)
    finally:m.terms=original
    if not hasattr(operator,'terms'):
        # Empty/native small operators are cheap and use the repaired shared ABI.
        return operator
    used={name for _,factors in operator.terms for name,site in factors if site==m.rfsite}
    extra=sorted(used-m._review_base_RF_names);names={}
    inverse=np.argsort(m.sites[m.rfsite].perm)
    identity_order=np.array_equal(inverse,np.arange(len(inverse)))
    with zipfile.ZipFile(rawpath,'w',zipfile.ZIP_DEFLATED) as archive:
        for index,name in enumerate(extra):
            key='RF_'+str(index);names[name]=key
            value=m.sites[m.rfsite].get_op(name).to_ndarray()
            if not identity_order:value=value[np.ix_(inverse,inverse)]
            with archive.open(key+'.npy','w',force_zip64=True) as stream:
                np.lib.format.write_array(stream,value,allow_pickle=False)
    data=dict(binding=binding,terms=[([float(complex(c).real),float(complex(c).imag)],factors) for c,factors in operator.terms],RF_arrays=names,raw_sha256=digest(rawpath),meaning='Compiled canonical derivative operators only, input/source bound. Base3.3GB operator is referenced by hash, not copied. Own evolved states live in separate stage files.')
    data['source_input_receipt']=m._review_capture_receipt
    meta.write_text(json.dumps(data,indent=2)+'\n')
    return operator
