"""Matched fresh neutral and charged response, temporal/full-minus-OFF contrast.
Uses only this owner's raw charged packets; neutral recomputed from declared input.
"""
import gc
import json
from pathlib import Path
from typing import Any
import numpy as np
import rich_spatial_frame_photon_20261016_model as m
from rich_spatial_frame_photon_20261016_run import evolve,kick
ROOT=Path(__file__).resolve().parent
P='rich_spatial_frame_photon_20261016_'


def moments(model,v):
    x=v.reshape(model.shape);prob=np.sum(abs(x)**2,axis=(1,2,3,4))
    return dict(Pi=np.array([np.vdot(v,model.field(v,a)).real for a in range(8)]),a=np.array([np.vdot(v,model.field(v,a,'q')).real for a in range(8)]),density=prob@model.matter.sector.n,LEFT=np.array([np.vdot(x,np.einsum('ij,fjrma->firma',a.toarray(),x,optimize=True)).real for a in model.matter.g['L']]))


def difference(a,b):return {k:a[k]-b[k] for k in a}


def main():
    result={}
    for label in ['paired','R2','field2','q0']:
        p=dict(m.DEFAULTS)
        if label=='R2':p['material_level']=2
        if label=='field2':p['field_total_cut']=2
        if label=='q0':p['wave_number_scale']=0.
        t=.04;ref=m.Model(4,p);v=ref.preparation();end,e=evolve(ref.apply,v,t,12)
        bare,be=evolve(lambda z:ref.apply(z,False,False),v,t,12)
        neutral={};raw: dict[str,Any]=dict(neutral_input=v,neutral_end=end,neutral_bare=bare)
        for axis in (0,2):
            pulse,ki=kick(ref,v,axis,.23);pend,ee=evolve(ref.apply,pulse,t,12)
            poff,eo=evolve(lambda z:ref.apply(z,False,False),pulse,t,12)
            initial=difference(moments(ref,pulse),moments(ref,v))
            full=difference(difference(moments(ref,pend),moments(ref,end)),initial)
            off=difference(difference(moments(ref,poff),moments(ref,bare)),initial)
            neutral[str(axis)]=difference(full,off)
            raw.update({f'neutral_kick{axis}':pulse,f'neutral_kick{axis}_end':pend,f'neutral_kick{axis}_OFF':poff})
        data=dict(neutral={a:{k:v.tolist() for k,v in d.items()} for a,d in neutral.items()},branches={})
        for number,name in [(5,'addition'),(3,'removal')]:
            model=m.Model(number,p);z=np.load(ROOT/(P+label+'_'+name+'.npz'))
            b={}
            for axis in (0,2):
                vin=z['input'];pin=z[f'kick{axis}_input'];baseline=z['baseline'];pend=z[f'kick{axis}_end']
                # OFF baseline reconstructed under its actual H from SAME exact input.
                baseoff,oe=evolve(lambda s:model.apply(s,False,False),vin,t,12)
                poff=z[f'bare{axis}_end']
                initial=difference(moments(model,pin),moments(model,vin))
                full=difference(difference(moments(model,pend),moments(model,baseline)),initial)
                off=difference(difference(moments(model,poff),moments(model,baseoff)),initial)
                contrast=difference(full,off);charged_minus_neutral=difference(contrast,neutral[str(axis)])
                b[str(axis)]=dict(initial_kick={k:v.tolist() for k,v in initial.items()},time_induced_full={k:v.tolist() for k,v in full.items()},time_induced_OFF={k:v.tolist() for k,v in off.items()},full_minus_OFF={k:v.tolist() for k,v in contrast.items()},charged_minus_neutral_full_minus_OFF={k:v.tolist() for k,v in charged_minus_neutral.items()})
            data['branches'][name]=b
            del model;gc.collect()
        result[label]=data
        np.savez_compressed(ROOT/(P+label+'_neutral_contrast_raw.npz'),**raw)
        (ROOT/(P+'contrast_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(label,json.dumps(data),flush=True)
        del ref,raw;gc.collect()
    print('matched neutral and temporal contrasts complete',flush=True)


if __name__=='__main__':main()
