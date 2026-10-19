"""Same saved states, explicit owning H/Hoff energy and identity-term partition.
No evolution, state normalization, source filtering, fitted energy or law change.
"""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from rich_photon_collective_20261015_model import install

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'

def run():
    started=time.monotonic();source,mode=sys.argv[1:3];stem=PREFIX+source;out=PREFIX+mode
    data=json.loads((ROOT/(stem+'_data.json')).read_text());base=PREFIX+data['base_mode'];c=data['physics'];n=data['geometry']['matter'];nf=data['geometry']['field'];nb=5*nf
    with np.load(ROOT/(base+'_primitive.npz')) as a:
        profiles=a['profiles'];offset=float(a['offset']);phi=a['phi'];phi_h=a['phi_h'];kinetic=a['kinetic']
        matrices=[sparse.load_npz(ROOT/(base+f'_Hcomponent{k}.npz')) for k in range(len(profiles))]
        if data['representation']=='oscillator':
            h=install(matrices,[sparse.csr_matrix(t) for t in a['terms']],a['boson'],offset);factors=None
        else:
            factors=a['factors'].reshape(len(matrices),nb);h=None
    matterphi=sparse.kron(matrices[0],sparse.eye(5))+sparse.kron(matrices[1],sparse.diags(c['coupling']*phi))
    for k in range(2,len(matrices),2):matterphi+=sparse.kron(matrices[k],sparse.eye(5))
    matterphi+=sparse.kron(sparse.eye(n),sparse.csr_matrix(phi_h))-offset*sparse.eye(n*5)
    def energies(state):
        x=state.reshape(n,5,nf);flat=x.reshape(n,nb);mp=x.reshape(n*5,nf)
        off=(matterphi@mp)+(kinetic@mp.T).T;offenergy=float(np.vdot(state,off.ravel()).real)
        if h is not None:acted=h@state
        else:
            acted=((kinetic@mp.T).T.reshape(x.shape)+np.einsum('ab,mbf->maf',phi_h,x,optimize=True)-offset*x).reshape(n,nb)
            for k,matrix in enumerate(matrices):acted+=(matrix@flat)*factors[k]
            acted=acted.ravel()
        return float(np.vdot(state,acted).real),offenergy,float(np.vdot(state,state).real)
    cache={};audit=[]
    def correct(point,key,off=False):
        if key not in cache:
            filename,array=key
            with np.load(ROOT/filename) as a:state=a[array]
            cache[key]=energies(state)
        coupled,bare,norm=cache[key];old=point['energy'];point['reported_energy_before_accounting']=old;point['norm']=norm;point['energy']=bare if off else coupled
        if off:point['counterfactual_coupled_H_energy']=coupled
        for layer in point['profile']:
            layer['reference']=-offset*norm/5;layer['total']=sum(v for k,v in layer.items() if k!='total')
        point['profile_sum_defect']=sum(z['total'] for z in point['profile'])-point['energy']
        audit.append(dict(time=point['time'],state_file=key[0],array=key[1],owning_H='Hoff' if off else 'H',energy=point['energy'],counterfactual_coupled_H_energy=coupled,original_report_minus_actual_coupled_H=old-coupled,profile_sum_defect=point['profile_sum_defect']))
    t0=stem+'_t0.npz';correct(data['ground'],(t0,'ground'))
    for arm,points in data['records'].items():
        for sample,point in enumerate(points):
            key=(t0,'ground' if arm=='same_state_off_baseline' else 'pulse') if sample==0 else (stem+f'_{arm}_state{sample}.npz','state')
            correct(point,key,arm.startswith('same_state_off'))
    for sample,comparison in enumerate(data['comparisons'],1):
        comparison['pre_accounting_total_energy_profile_L2']=comparison['total_energy_profile_L2']
        comparison['total_energy_profile_L2']=float(np.linalg.norm(np.array([z['total'] for z in data['records']['coarse'][sample]['profile']])-np.array([z['total'] for z in data['records']['fine'][sample]['profile']])))
    data['source_evolution_kind']=data['kind'];data['kind']='COMPLETED_SAME_STATE_LAW_OWNED_ENERGY_AND_IDENTITY_PARTITION_CONSUMER';data['source_evolution_mode']=source
    data['source_propagation_resources']={k:data[k] for k in ('wall_seconds','peak_RSS_KiB')}
    data['accounting_repair']=dict(no_propagation=True,no_state_normalization=True,no_law_or_source_change=True,root_cause='Coordinate observer hoff=make() aliased coupled H, while off_evolve independently used correct H_MP+K. Original off energy was counterfactual coupled-H expectation. Oscillator hoff was already correct. Identity reference expectation is -offset*norm, not -offset on numerical nonunit vectors.',audit=audit,max_profile_sum_defect=max(abs(x['profile_sum_defect']) for x in audit),max_original_report_minus_actual_coupled_H=max(abs(x['original_report_minus_actual_coupled_H']) for x in audit),meaning='Direct full-state contractions with same primitive H and actual off Kronecker-sum. Local components unchanged except mathematically required identity expectation; no residual redistributed or fitted away.')
    data['wall_seconds']=time.monotonic()-started;data['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (ROOT/(out+'_data.json')).write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(dict(mode=mode,wall_seconds=data['wall_seconds'],peak_RSS_KiB=data['peak_RSS_KiB'],max_profile_sum_defect=data['accounting_repair']['max_profile_sum_defect'])),flush=True)

if __name__=='__main__':run()
