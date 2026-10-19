"""Prepared-source physical discriminator before spectra; no ground-state claim."""
import json,sys,time,resource
from pathlib import Path
import numpy as np
from scipy import sparse
from rich_common_carrier_motion_20261015_model import build,law,fibre,source,source_matrix,observable,compress,prepared_reference
from rich_frame_field_join_20261015_calculate import krylov


def real(z):return float(np.real(z))
def expect(a,x):return real(np.vdot(x,a@x))/real(np.vdot(x,x))
def complex_list(z):return [float(np.real(z)),float(np.imag(z))]


def main():
    p=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);start=time.monotonic();result={'parameters':p,'kind':'PREPARED_OCCUPIED_MOTION_FIRST_DISCRIMINATOR','stages':[]};arrays={}
    def checkpoint(stage):
        result['stages'].append({'stage':stage,'elapsed_seconds':time.monotonic()-start});out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(stage,flush=True)
    model=build(p);result['body_dimensions']={n:s['dimension'] for n,s in model['sectors'].items()};checkpoint('NEW_CAR36_BODY_AVAILABLE')
    v0=fibre(model,'neutral',0.,0.);fullz=prepared_reference(model);z=v0.conj().T@fullz
    result['prepared_reference']={'norm':float(np.linalg.norm(z)),'J0_projection_loss':float(np.linalg.norm(fullz-v0@z)),'meaning':'New bare occupied CAR32 N16 sea times two occupied negative-dimer orbitals in full extra CAR4, own radial ground and auxiliary EM basis vacuum; NOT stationary whole-H vacuum.'}
    arrays['neutral_t0']=z;result['branches']={};t=p['time'];k=p['krylov_steps'];fine=p['fine_steps']
    result['source_covariance_before_evolution']={}
    for name in ['lepton','antilepton']:
        sp0=model['sectors']['neutral']['op']['S'];sp1=model['sectors'][name]['op']['S']
        up=source_matrix(model,name,2,.5);down=source_matrix(model,name,2,-.5)
        defect=float(sparse.linalg.norm((sp1[0]+1j*sp1[1])@down-down@(sp0[0]+1j*sp0[1])-up))
        raw=source(model,name,2,.5)@fullz
        result['source_covariance_before_evolution'][name]={'raising_Frobenius':defect,'new_cell_source_norm_squared':float(np.linalg.norm(raw)**2)}
        if defect>1e-8 or np.linalg.norm(raw)<.1:raise ValueError('Corrected primitive new-cell source fails actual spin covariance/brightness')
    checkpoint('CORRECTED_CAR_SOURCE_BRIGHT_BEFORE_EVOLUTION')
    for bridge in [True,False]:
        tag='full' if bridge else 'bridge_off'
        h0,a0=compress(law(model,'neutral',bridge=bridge),v0);zt,l0=krylov(h0,z,t,k);ztfine,l0fine=krylov(h0,z,t,fine)
        arrays[tag+'_neutral_t']=ztfine
        result[tag+'_reference']={'dimension':h0.shape[0],'factorization':a0,'coarse_fine_state_L2':float(np.linalg.norm(zt-ztfine)),'coarse':l0,'fine':l0fine}
        for name in ['lepton','antilepton']:
            if not bridge and name=='antilepton' and p.get('single_decision_control',False):continue
            v=fibre(model,name,.5,.5);h,a=compress(law(model,name,bridge=bridge),v);sm=[];norms=[]
            for site in range(4):
                sop,sa=compress(source(model,name,site,.5),v,v0);sm.append(sop);norms.append(float(np.linalg.norm(sop@z)**2))
            x=sm[0]@z;xt,ledger=krylov(h,x,t,k);xf,lf=krylov(h,x,t,fine)
            green=[]
            for sop in sm:
                value=np.vdot(sop@ztfine,xf) if name=='lepton' else np.vdot(xf,sop@ztfine)
                green.append(complex_list(value))
            op=model['sectors'][name]['op'];density=[compress(observable(model,name,matter=a),v)[0] for a in op['rho']]
            neutral_density=[compress(observable(model,'neutral',matter=a),v0)[0] for a in model['sectors']['neutral']['op']['rho']]
            rho=[expect(a,xf) for a in density];rho0=[expect(a,ztfine) for a in neutral_density]
            spin=compress(observable(model,name,matter=op['S'][2]),v)[0]
            r2=compress(observable(model,name,power=2),v)[0];colour=compress(observable(model,name,matter=op['B']),v)[0]
            row={'dimension':h.shape[0],'factorization':a,'source_raw_retained_norms_squared':norms,'green_source0_to_sites':green,'rho_material_charge':rho,'neutral_rho_material_charge':rho0,'signed_excess_charge':[a-b for a,b in zip(rho,rho0)],'total_charge':sum(rho),'LEFT_z':expect(spin,xf),'LEFT_z_t0':expect(spin,x),'matrix_r2':expect(r2,xf),'colour_loop_B':expect(colour,xf),'coarse_fine_state_L2':float(np.linalg.norm(xt-xf)),'coarse_fine_green_max':max(abs(np.vdot(sop@ztfine,xt-xf)) for sop in sm),'coarse':ledger,'fine':lf}
            result['branches'][tag+'_'+name]=row;arrays[tag+'_'+name+'_t0']=x;arrays[tag+'_'+name+'_t']=xf
            np.savez_compressed(out.with_suffix('.npz'),**arrays);checkpoint(tag+'_'+name+'_PHYSICAL_MOTION')
        del h0
    result['first_discriminator']={name:{'new_cell_green_absolute': [abs(complex(*g)) for g in result['branches']['full_'+name]['green_source0_to_sites'][2:]],'new_cell_signed_excess_sum':sum(result['branches']['full_'+name]['signed_excess_charge'][2:])} for name in ['lepton','antilepton']}
    for name in ['lepton','antilepton']:
        control=result['branches'].get('bridge_off_'+name)
        if control is not None:result['first_discriminator'][name].update(bridge_off_new_cell_green_absolute=[abs(complex(*g)) for g in control['green_source0_to_sites'][2:]],bridge_off_new_cell_signed_excess_sum=sum(control['signed_excess_charge'][2:]))
    provisional={'kind':'FIRST_ACTUAL_PREPARED_SOURCE_DISCRIMINATOR_PROVISIONAL','observed':result['first_discriminator'],'resolution':{n:r['coarse_fine_state_L2'] for n,r in result['branches'].items()},'limits':['Prepared nonstationary occupied reference, not ground state or quasiparticle pole','Four fixed material positions on one rotating cluster, no free COM or bulk translation','Native root charge-family cut N15..17 and complete extra CAR4 Fock; missing root N14/N18 transitions are projected out','New cell has fixed +1 background at each added site; this is a declared input','Physical frame LEFT spin from shared chosen RIGHT Gauss, not primitive CAR physical spin','Signed excess is charge, Green is propagation amplitude, neither particle position probability','Finite radial3/EM shell1/native/PW/rotation cuts; no Nature/electron/photon/global-closure promotion']}
    out.with_name('rich_common_carrier_motion_20261015_provisional_motion2.json').write_text(json.dumps(provisional,indent=2)+'\n');checkpoint('FIRST_DISCRIMINATOR_DELIVERED')
    result['resource']={'elapsed_seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};checkpoint('NATURAL_COMPLETION')

if __name__=='__main__':main()
