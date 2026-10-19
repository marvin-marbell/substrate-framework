"""Preserve actual virtual charges; recover own legacy B arrays without cutoff loss."""
from pathlib import Path
import hashlib,inspect,json
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.networks.mps import MPS
from rich_review_wave49_20261016_integrated_storage import basis_binding,stage_binding,fingerprint,digest

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_';EPOCH='transport_priority2'


def save_state(arrays,label,psi):
    arrays[label+'_norm']=np.asarray(psi.norm);arrays[label+'_charge']=np.asarray(psi.get_total_charge(only_physical_legs=True))
    for i in range(psi.L):
        B=psi.get_B(i,form='B').transpose(['vL','p','vR']);key=f'{label}_B_{i:02}'
        arrays[key]=B.to_ndarray();arrays[f'{label}_S_{i:02}']=np.asarray(psi.get_SL(i));arrays[key+'_qtotal']=B.qtotal
        for axis in ('vL','vR'):
            leg=B.get_leg(axis);arrays[key+'_'+axis+'_qflat']=leg.to_qflat();arrays[key+'_'+axis+'_qconj']=np.asarray(leg.qconj)


def restore(m,raw,label):
    Bs=[];singular=[];legL=npc.LegCharge.from_qflat(m.chinfo,np.zeros((1,m.chinfo.qnumber),dtype=int));legacy=label+'_B_00_qtotal' not in raw
    for i,site in enumerate(m.sites):
        key=f'{label}_B_{i:02}';flat=raw[key];singular.append(raw[f'{label}_S_{i:02}'])
        if legacy:
            # Structural forbidden entries are EXACT0 in these saved npc arrays.
            # Native default QCUTOFF wrongly loses tiny retained Schmidt sectors.
            legs=npc.detect_legcharge(flat,m.chinfo,[legL,site.leg,None],qtotal=np.zeros(m.chinfo.qnumber,dtype=int),qconj=-1,cutoff=0.)
            total=np.zeros(m.chinfo.qnumber,dtype=int)
        else:
            vlegs={a:npc.LegCharge.from_qflat(m.chinfo,raw[key+'_'+a+'_qflat'],qconj=int(raw[key+'_'+a+'_qconj'])).bunch()[1] for a in ('vL','vR')}
            legs=[vlegs['vL'],site.leg,vlegs['vR']];total=raw[key+'_qtotal']
        B=npc.Array.from_ndarray(flat,legs,qtotal=total,cutoff=0.,labels=['vL','p','vR'])
        if not np.array_equal(B.to_ndarray(),flat):raise ValueError(('checkpoint tensor changed during charge reconstruction',i))
        Bs.append(B);legL=legs[-1].conj()
    singular.append(np.ones(1));norm=float(raw[label+'_norm'])
    if not np.isfinite(norm) or norm<0:raise ValueError('invalid raw MPS norm')
    # Direct native constructor, NO canonical_form, SVD, normalization or phase reset.
    state=MPS(m.sites,Bs,singular,bc='finite',form='B',norm=norm,unit_cell_width=1)
    if not np.array_equal(state.get_total_charge(only_physical_legs=True),raw[label+'_charge']):raise ValueError('physical total/sewing charge changed')
    return state


def stage_paths(label):
    if not label or any(not(c.isalnum() or c=='_') for c in label):raise ValueError('stable physical stage label required')
    stem=P+EPOCH+'_'+label;return ROOT/(stem+'_state.npz'),ROOT/(stem+'_stage.json')


def load_stage(m,label,binding):
    _,meta=stage_paths(label)
    if not meta.exists():return None
    record=json.loads(meta.read_text());payload=ROOT/record['raw_file']
    if record['binding']!=binding or digest(payload)!=record['raw_sha256']:raise ValueError(('own stage dependency mismatch',label))
    with np.load(payload) as raw:state=restore(m,raw,'state')
    state.norm/=np.exp(record['removed']);return state,record


def save_stage(m,label,binding,normalized_state,removed,ledgers):
    rawpath,meta=stage_paths(label);physical=normalized_state.copy();physical.norm*=np.exp(removed);arrays={};save_state(arrays,'state',physical)
    rawpath=rawpath.with_name(rawpath.stem+'_s'+str(len(ledgers)).zfill(3)+'.npz');np.savez_compressed(rawpath,**arrays)
    record=dict(binding=binding,completed_steps=len(ledgers),removed=removed,ledgers=ledgers,raw_file=rawpath.name,raw_sha256=digest(rawpath),completed_physical_time=binding['initial_physical_time']+binding['dt']*len(ledgers),generation=dict(storage_source_sha256=digest(__file__),evolve_source_sha256=hashlib.sha256(inspect.getsource(type(m.H).evolve).encode()).hexdigest(),source_input_receipt=m._review_capture_receipt),meaning='OWN completed physical stage. Payload including actual virtual qflat/qconj/qtotal, Schmidt values, global real norm and ALL tensor phases is complete BEFORE receipt commits. No normalization during restore; no Model or producer state copy.')
    temporary=meta.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(meta)
