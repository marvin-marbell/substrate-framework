"""Actual PW colour-loop and common-phi reaction on the SAME native quantum law."""
import json,resource,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from rich_spatial_colour_charge_20261015_snapshot import load
from rich_spatial_colour_charge_20261015_factors import law,route_current
from rich_spatial_colour_charge_20261015_model import BodyLift


def propagate(h,x,t):return x.copy() if t==0 else expm_multiply(-1j*t*h,x,traceA=-1j*t*h.trace())


def main():
    ip=Path(sys.argv[1]);prefix=Path(sys.argv[2]);request=json.loads(ip.read_text());producer=ip.parent/request['body_prefix'];data=load(str(producer)+'_body.pkl');p=data['parameters'];nf=len(data['scalar']['grid']);started=time.time()
    omega=np.load(str(producer)+'_neutral_state.npz')['vectors'][:,0];results={}
    for name in ['neutral','lepton','baryon']:
        op=data['operators'][name];h=law(op,data['rotor'],data['scalar'],p,data['geometry'])
        state=omega if name=='neutral' else data['sources'][name]['0:0.5:0.5']@omega
        strength=float(np.vdot(state,state).real);state=state/np.sqrt(strength)
        currents=[route_current(op,data['rotor'],data['scalar'],p,data['geometry'],e) for e in range(2)]
        def values(x):
            out={key:float(np.vdot(x,BodyLift(op[key],10*nf)@x).real) for key in ['B','C','PW_boundary','electric','rho0','Q']}
            weights=np.sum(abs(x.reshape(-1,nf))**2,axis=0);out.update(phi=float(weights@data['scalar']['grid']),phi2=float(weights@(data['scalar']['grid']**2)),norm_squared=float(np.vdot(x,x).real),route0=float(np.vdot(x,currents[0]@x).real),route1=float(np.vdot(x,currents[1]@x).real));return out
        initial=values(state);kicked=(state.reshape(-1,nf)@expm(-1j*p['field_kick']*data['scalar']['P']).T).ravel()
        pulse=law(op,data['rotor'],data['scalar'],p,data['geometry'],p['A_pulse']+[0.,0.,0.]);driven=propagate(pulse,state,p['pulse_duration']);base_pulse=propagate(h,state,p['pulse_duration'])
        result=dict(source_raw_strength=strength,initial=initial,field_only=[],A_pulse=[])
        for t in request['times']:
            for label,a0,b0 in [('field_only',kicked,state),('A_pulse',driven,base_pulse)]:
                a=values(propagate(h,a0,t));b=values(propagate(h,b0,t));result[label].append(dict(t=t,actual=a,untouched_same_preparation_baseline=b,excess={key:a[key]-b[key] for key in a}))
        results[name]=result
    payload=dict(kind='ACTUAL_PW19_QUANTUM_LOOP_AND_COMMON_SCALAR_BACKREACTION_SAME_CHARGE_ORBIT_LAW',inputs=request,results=results,meaning='B=ReTrU and C=right colour-electric Casimir are LOOP observables, distinct from total colour/tree electric energy. Field-only unitary acts solely on phi, initial matter/loop marginal unchanged; later full coupled reaction compared against untouched same actual CAR-source preparation, never stationary-pole or neutral reset. A pulse drives entire same physical current; no species-only current projector. Conditional finite packet/PW19/SO3/DVR law, no full-H range/continuum/photon/COM admission.',resource=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-started))
    Path(str(prefix)+'_result.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(dict(result=str(prefix)+'_result.json',resource=payload['resource'])))

if __name__=='__main__':main()
