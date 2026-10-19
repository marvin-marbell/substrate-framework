"""Same-own-H moving sources: actual external-A current/charge response.

Reuses the ONE published all-J factor cache shared with the interaction owner.
Only changed external-A H/J factors require new compilation here.
"""
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from rich_common_carrier_motion_20261015_model import build,law
from rich_common_carrier_motion_20261015_factors import compile,load
from rich_frame_field_join_20261015_calculate import krylov,m_isometry


def expect(a,x):return float(np.vdot(x,a@x).real/np.vdot(x,x).real)
def main():
    p=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);start=time.monotonic();cachepath=Path(p['own_factor_result']);cache=json.loads(cachepath.read_text());statepath=Path(p['own_factor_states']);states=np.load(statepath)
    if cache['stage']!='NATURAL_COMPLETION':raise ValueError('Shared cache did not complete naturally')
    if cache['model_sha256']!=hashlib.sha256(Path('rich_common_carrier_motion_20261015_model.py').read_bytes()).hexdigest():raise ValueError('Cached carrier H model bytes differ')
    model=build(p);direction=np.array(p['external_direction'],float);eps=p['external_amplitude'];t=p['response_time'];k=p['response_krylov_steps'];fine=p['response_fine_steps'];result={'kind':'SAME_OWN_MOVING_SOURCE_EXTERNAL_A_CURRENT_CHARGE_RESPONSE','parameters':p,'own_factor_result_sha256':hashlib.sha256(cachepath.read_bytes()).hexdigest(),'own_factor_states_sha256':hashlib.sha256(statepath.read_bytes()).hexdigest(),'factor_input_manifest':cache['snapshots'],'physical_source_audit_reference':p['own_factor_result'],'arms':{},'external_factor_snapshots':{}};arrays={}
    def fetch(label):
        item=cache['snapshots'][label];return load(item['file'],item['sha256'])
    def save(stage):
        result['stage']=stage;out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**arrays);print(stage,flush=True)
    def snapshot(label,operator):result['external_factor_snapshots'][label]=operator.save(out.with_name(out.name+'_'+label+'.pkl'))
    for name,M in [('neutral',0.),('lepton',.5),('antilepton',.5)]:
        v=m_isometry(model,name,M);raw=states[name+'_moving_at_t025'];raw_strength=float(np.linalg.norm(raw)**2);x=raw/np.sqrt(raw_strength);arrays[name+'_initial']=x
        density=[fetch(name+'_rho'+str(i)) for i in range(4)];spin=fetch(name+'_LEFT_z');r2=fetch(name+'_matrix_r2');colour=fetch(name+'_colour_loop_B');axes=[i for i in range(6) if direction[i]!=0];currents={i:fetch(name+'_J'+str(i)+'_zero') for i in axes};h0=fetch(name+'_H_zero')
        derivative=sum((direction[i]*(currents[i]@x) for i in axes),np.zeros_like(x));step=p['derivative_step']
        hd,_=compile(law(model,name,external=step*direction),v);fd=-(hd@x)/(2*step);del hd
        hd,_=compile(law(model,name,external=-step*direction),v);fd+=(hd@x)/(2*step);del hd
        row={'dimension':len(x),'unnormalized_incoming_source_strength':raw_strength,'external_current_derivative_relative_error':float(np.linalg.norm(fd-derivative)/max(np.linalg.norm(derivative),np.finfo(float).tiny)),'wrong_sign_relative_error':float(np.linalg.norm(fd+derivative)/max(np.linalg.norm(derivative),np.finfo(float).tiny)),'current_direction_at_initial':float(np.vdot(x,derivative).real),'paths':{}}
        for label,sign in [('zero',0),('plus',1),('minus',-1)]:
            external=sign*eps*direction
            if sign:h,ha=compile(law(model,name,external=external),v);snapshot(name+'_H_'+label,h);row['factorization_'+label]=ha
            else:h=h0
            coarse,lc=krylov(h,x,t,k);final,lf=krylov(h,x,t,fine);current=0.
            for i in axes:
                if sign:a,_=compile(law(model,name,external=external,derivative=i),v);snapshot(name+'_J'+str(i)+'_'+label,a)
                else:a=currents[i]
                current+=direction[i]*expect(a,final)
            row['paths'][label]={'rho_material_charge':[expect(a,final) for a in density],'LEFT_z':expect(spin,final),'matrix_r2':expect(r2,final),'colour_loop_B':expect(colour,final),'canonical_current_same_A':current,'coarse_fine_state_L2':float(np.linalg.norm(coarse-final)),'coarse_fine_charge_max':max(abs(expect(a,coarse)-expect(a,final)) for a in density),'coarse':lc,'fine':lf};arrays[name+'_'+label]=final;result['arms'][name]=row;save(name+'_'+label+'_ACTUAL_A_RESPONSE')
            if sign:del h
        del h0,currents
    result['signed_excess_response']={}
    for name in ['lepton','antilepton']:
        result['signed_excess_response'][name]={label:[result['arms'][name]['paths'][label]['rho_material_charge'][i]-result['arms']['neutral']['paths'][label]['rho_material_charge'][i] for i in range(4)] for label in ['zero','plus','minus']}
    result['limits']=['Original nonstationary prepared reference and evolved incoming moving-source states are distinct named protocols','External six real Gaussian TE/TM amplitudes shift BEFORE all primitive Peierls/mechanical products','All compatible J in fixed physical M retained for A_z probe','H0/J0/rho/spin/r²/colour reused by exact pinned numeric shared cache; no duplicate unchanged-H compression','Finite Hilbert/EM/radial/charge-family projections remain; no continuum/photon/asymptotic particle/Nature claim'];result['resource']={'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};save('NATURAL_COMPLETION')
if __name__=='__main__':main()
