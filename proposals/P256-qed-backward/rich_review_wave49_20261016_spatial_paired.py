"""Calibrate own fresh-state field feedback against complete-charge-motion OFF echoes."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import expm
import rich_spatial_frame_photon_20261016_model as law
from rich_review_wave49_20261016_spatial_consumer import evolve,observe

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def run():
    source=ROOT/(P+'spatial_consumer_raw.npz');raw=np.load(source);records={};arrays={}
    for branch,number in (('addition',5),('removal',3)):
        m=law.Model(number);inverse=expm(-.17j*m.q[0]-.11j*m.q[4])
        initial=(raw[branch+'_initial'].reshape(-1,m.nfield)@inverse.T).ravel()
        time=.037;un=evolve(m,initial,time);unoff=evolve(m,initial,time,hopping=False,motion=False)
        full0=observe(m,un);off0=observe(m,unoff);rows=[]
        for incoming,outgoing in ((0,6),(2,4)):
            kick=expm(.19j*m.q[incoming]);kicked=(initial.reshape(-1,m.nfield)@kick.T).ravel()
            final=evolve(m,kicked,time);off=evolve(m,kicked,time,hopping=False,motion=False)
            full=observe(m,final);control=observe(m,off)
            def momentum(v):return float(np.vdot(v,m.field(v,outgoing,'p')).real)
            delta=momentum(final)-momentum(un);echo=momentum(off)-momentum(unoff)
            charge=np.array(full['physical_charge_density'])-full0['physical_charge_density']
            offcharge=np.array(control['physical_charge_density'])-off0['physical_charge_density']
            spin=np.array(full['g_LEFT'])-full0['g_LEFT'];offspin=np.array(control['g_LEFT'])-off0['g_LEFT']
            row=dict(incoming=incoming,outgoing=outgoing,time=time,kick=.19,kick_work=float(np.vdot(kicked,m.apply(kicked)).real-np.vdot(initial,m.apply(initial)).real),field_momentum_response=delta,whole_charge_motion_OFF_echo=echo,calibrated_field_response=delta-echo,physical_charge_response=charge.tolist(),whole_OFF_charge_response=offcharge.tolist(),calibrated_charge_response=(charge-offcharge).tolist(),physical_g_LEFT_response=spin.tolist(),whole_OFF_g_LEFT_response=offspin.tolist(),full_norm=float(np.vdot(final,final).real),off_norm=float(np.vdot(off,off).real))
            rows.append(row)
            arrays[f'{branch}_{incoming}_{outgoing}_kicked']=kicked;arrays[f'{branch}_{incoming}_{outgoing}_full']=final;arrays[f'{branch}_{incoming}_{outgoing}_off']=off
        records[branch]=rows;arrays[branch+'_initial']=initial;arrays[branch+'_uncicked_full']=un;arrays[branch+'_uncicked_off']=unoff
        print(branch,json.dumps(rows),flush=True)
    result=dict(kind='OWN_ACTUAL48_PAIRED_KICK_VERSUS_COMPLETE_CHARGE_OFF',raw_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),records=records,limits='Own fresh-source states only, reconstructed by inverse of own earlier retained-field unitary. Both qin transverse polarizations with nonforward copolarized qout. Total-shell cut makes distinct projected coordinates noncommute; large OFF echo is explicitly subtracted, not claimed as matter-induced photon. Even a nonzero calibrated excess is a finite-regulator response, not convergence or propagation.')
    np.savez_compressed(ROOT/(P+'spatial_paired_raw.npz'),**arrays)
    (ROOT/(P+'spatial_paired_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
