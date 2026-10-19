"""Own same-basis q1-vs-q0 charge-excess full-minus-OFF field response, not a gap test."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import expm
import rich_spatial_frame_photon_20261016_model as law
from rich_review_wave49_20261016_spatial_consumer import evolve,observe

ROOT=Path(__file__).resolve().parent;P='rich_review_wave49_20261016_'


def paired(m,initial,label,arrays):
    arrays[label+'_initial']=initial
    un=evolve(m,initial,.037);unoff=evolve(m,initial,.037,hopping=False,motion=False)
    baseline=observe(m,un);offbaseline=observe(m,unoff);rows=[]
    for incoming,outgoing in ((0,6),(2,4)):
        kick=expm(.19j*m.q[incoming]);state=(initial.reshape(-1,m.nfield)@kick.T).ravel()
        full=evolve(m,state,.037);off=evolve(m,state,.037,hopping=False,motion=False)
        a=observe(m,full);b=observe(m,off)
        def momentum(v):return float(np.vdot(v,m.field(v,outgoing,'p')).real)
        response=momentum(full)-momentum(un);echo=momentum(off)-momentum(unoff)
        charge=(np.array(a['physical_charge_density'])-baseline['physical_charge_density'])-(np.array(b['physical_charge_density'])-offbaseline['physical_charge_density'])
        spin=(np.array(a['g_LEFT'])-baseline['g_LEFT'])-(np.array(b['g_LEFT'])-offbaseline['g_LEFT'])
        rows.append(dict(incoming=incoming,outgoing=outgoing,calibrated_field_response=response-echo,field_momentum_response=response,whole_charge_motion_OFF_echo=echo,calibrated_charge_response=charge.tolist(),calibrated_g_LEFT_response=spin.tolist(),kick_work=float(np.vdot(state,m.apply(state)).real-np.vdot(initial,m.apply(initial)).real)))
        arrays[f'{label}_{incoming}_{outgoing}_full']=full;arrays[f'{label}_{incoming}_{outgoing}_off']=off
    arrays[label+'_uncicked_full']=un;arrays[label+'_uncicked_off']=unoff
    return rows


def run():
    current=ROOT/(P+'spatial_consumer_raw.npz');raw=np.load(current)
    previous=json.loads((ROOT/(P+'spatial_paired_result.json')).read_text())
    digest=hashlib.sha256(current.read_bytes()).hexdigest();assert previous['raw_input_sha256']==digest
    p=dict(law.DEFAULTS,wave_number_scale=0.)
    zero=law.Model(4,p);neutral0=evolve(zero,zero.preparation(),.021)
    one=law.Model(4);arrays={};records={'q1':{'neutral':paired(one,raw['neutral_state'],'q1_neutral',arrays)},'q0':{}}
    records['q0']['neutral']=paired(zero,neutral0,'q0_neutral',arrays)
    strengths={}
    for branch,number,create in (('addition',5,True),('removal',3,False)):
        records['q1'][branch]=[dict(r,calibrated_g_LEFT_response=(np.array(r['physical_g_LEFT_response'])-r['whole_OFF_g_LEFT_response']).tolist()) for r in previous['records'][branch]]
        m=law.Model(number,p);values=[];images=[]
        for component in range(8):
            image=m.source(zero,neutral0,component,create);values.append(float(np.vdot(image,image).real));images.append(image)
        component=int(np.argmax(values));state=images[component]/np.sqrt(values[component]);del images
        strengths[branch]=dict(all8=values,component=component)
        records['q0'][branch]=paired(m,state,'q0_'+branch,arrays)
    excess={}
    for branch in ('addition','removal'):
        rows=[]
        for index in range(2):
            qresponses={}
            for name in ('q1','q0'):
                r=records[name][branch][index];n=records[name]['neutral'][index]
                spin=r['calibrated_g_LEFT_response']
                qresponses[name]=dict(field=r['calibrated_field_response']-n['calibrated_field_response'],charge=(np.array(r['calibrated_charge_response'])-n['calibrated_charge_response']).tolist(),g_LEFT=(np.array(spin)-n['calibrated_g_LEFT_response']).tolist())
            rows.append(dict(incoming=records['q1'][branch][index]['incoming'],outgoing=records['q1'][branch][index]['outgoing'],charged_minus_neutral_full_minus_OFF=qresponses,finite_q_minus_uniform_field=qresponses['q1']['field']-qresponses['q0']['field'],finite_q_minus_uniform_charge=(np.array(qresponses['q1']['charge'])-qresponses['q0']['charge']).tolist(),finite_q_minus_uniform_g_LEFT=(np.array(qresponses['q1']['g_LEFT'])-qresponses['q0']['g_LEFT']).tolist()))
        excess[branch]=rows
    result=dict(kind='OWN_ACTUAL48_FIXED8FIELD_Q1_Q0_CHARGED_NEUTRAL_REDUCED_RESPONSE',raw_input_sha256=digest,own_q1_paired_result_sha256=hashlib.sha256((ROOT/(P+'spatial_paired_result.json')).read_bytes()).hexdigest(),q0_allband_source_strengths=strengths,records=records,excess=excess,limits='Fresh q0 same-H neutral/addition/removal sources; q1 reuses review49 OWN exact matched paired states/results, not producer outputs. Same eight real coordinates/basis/inertias for q1/q0; only wavevectors change. q0 has four nonorthogonal constant cos profiles plus sin spectators, not literal isotropic3-mode44 spectra. Charged-neutral and complete-charge-motionOFF differences are reducible finite-regulator response, not IR stiffness, photon pole, mass or convergence proof.')
    np.savez_compressed(ROOT/(P+'spatial_qscale_raw.npz'),**arrays)
    (ROOT/(P+'spatial_qscale_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':run()
