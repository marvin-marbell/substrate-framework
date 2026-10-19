"""Fresh OLD initial physical rays under NEW exact free representation.
This separates representation from replacing the old finite PaP kick by full W.
No old evolved state or output is read; all rays are prepared anew from pure H.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import rich_exact_electric_picture_20261016_model as m
from rich_exact_electric_picture_20261016_run import propagate,moments,temporal_difference
ROOT=Path(__file__).resolve().parent
P='rich_exact_electric_picture_20261016_'


def main():
    p=dict(m.DEFAULTS,field_total_cut=1);ref=m.Model(4,p);neutral=ref.preparation();model=m.Model(5,p)
    source=model.source(ref,neutral,0,True);weight=float(np.linalg.norm(source)**2);v=source/np.sqrt(weight)
    eps=model.impulse(0);zero=model.zeros;T=.04;dt=.02
    finiteU=expm(1j*.23*model.fixed.q[0]);legacy=(v.reshape(model.shape)@finiteU.T).reshape(-1)
    base,eb=propagate(model,v,0.,T,zero,dt)
    oldray,eold=propagate(model,legacy,0.,T,zero,dt)
    trueW,ew=propagate(model,v,0.,T,eps,dt)
    offbase,eob=propagate(model,v,0.,T,zero,dt,False,False)
    offlegacy,eol=propagate(model,legacy,0.,T,zero,dt,False,False)
    initial=moments(model,v,0.,zero,channels=(0,6));baseline=moments(model,base,T,zero,channels=(0,6))
    oldinitial=moments(model,legacy,0.,zero,channels=(0,6));oldend=moments(model,oldray,T,zero,channels=(0,6))
    winitial=moments(model,v,0.,eps,channels=(0,6));wend=moments(model,trueW,T,eps,channels=(0,6))
    off0=moments(model,offbase,T,zero,channels=(),coupled=False,hopping=False)
    offold=moments(model,offlegacy,T,zero,channels=(),coupled=False,hopping=False)
    offw=moments(model,offbase,T,eps,channels=(),coupled=False,hopping=False)
    df=temporal_difference(oldend,baseline,oldinitial,initial);do=temporal_difference(offold,off0,oldinitial,initial)
    wf=temporal_difference(wend,baseline,winitial,initial);wo=temporal_difference(offw,off0,winitial,initial)
    result=dict(kind='SAME_PHYSICAL_INITIAL_OLD_BASELINE_AND_OLD_FINITE_KICK_RAYS_NO_REPLAY',source_weight=weight,T=T,dt=dt,parameters=p,initial_Pi6_finite_kick_shift=oldinitial['Pi'][6]-initial['Pi'][6],initial_Pi6_full_W_shift=winitial['Pi'][6]-initial['Pi'][6],old_ray_temporal_full_minus_decoupled={k:(np.asarray(df[k])-do[k]).tolist() for k in df},full_W_temporal_full_minus_decoupled={k:(np.asarray(wf[k])-wo[k]).tolist() for k in wf},old_ray_free_temporal_Pi6=do['Pi'][6],full_W_free_temporal_Pi6=wo['Pi'][6],old_ray_physical_energy_drift=oldend['physical_energy']-oldinitial['physical_energy'],full_W_physical_energy_drift=wend['physical_energy']-winitial['physical_energy'],integration=[eb,eold,ew,eob,eol])
    np.savez_compressed(ROOT/(P+'matched_ray_raw.npz'),neutral_input=neutral,charged_input=v,matched_old_finite_kick_input=legacy,baseline=base,matched_old_finite_kick_end=oldray,full_W_end=trueW,field_decoupled_baseline=offbase,field_decoupled_old_kick=offlegacy)
    (ROOT/(P+'matched_ray_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
