"""Fresh same-law paired-T response, source connections and higher R/field cuts."""
import gc
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import rich_spatial_frame_photon_20261016_model as m
from rich_spatial_frame_photon_20261016_run import evolve,kick,measure
ROOT=Path(__file__).resolve().parent
P='rich_spatial_frame_photon_20261016_'


def part(model,v,label):
    hm=model.matter;p=model.parameters;x=v.reshape(model.shape)
    if label=='E':return (x*np.sum(hm.sector.electric**2,axis=1)[:,None,None,None,None]/(2*p['electric_inertia'])).reshape(-1)
    if label=='g':return (x*hm.g['casimir'].diagonal()[None,:,None,None,None]/(2*p['radial_mass']*hm.rnodes[None,None,:,None,None]**2)).reshape(-1)
    if label=='r':return np.einsum('ij,fgjma->fgima',hm.Hradial[:model.nr,:model.nr].toarray(),x,optimize=True).reshape(-1)
    if label=='root':
        f=hm.sector.bilinear(hm.geo['beta']).toarray()
        return (np.einsum('ij,jgrma->igrma',f,x,optimize=True)*hm.rnodes[None,None,:,None,None]*p['root_coupling']).reshape(-1)
    if label=='R':return np.einsum('fgrj,fij->fgri',x.reshape(model.nf,model.ng,model.nr,-1),model.rotblocks,optimize=True).reshape(-1)
    if label=='hop':return model.hop_action(v)
    if label=='field':return (x@model.p2.T/(2*p['cavity_inertia'])).reshape(-1)
    raise ValueError(label)


def main():
    label=sys.argv[1] if len(sys.argv)>1 else 'paired';p=dict(m.DEFAULTS);t=.04
    if label=='R2':p['material_level']=2
    if label=='field2':p['field_total_cut']=2
    if label=='q0':p['wave_number_scale']=0.
    result: dict[str,Any]=dict(label=label,parameters=p,t=t,physical='same exact path law, independent fresh states; no interpolated or imported background',branches={})
    reference=m.Model(4,p);neutral=reference.preparation();nend,ne=evolve(reference.apply,neutral,t,12)
    result['neutral']=dict(norm=float(np.linalg.norm(nend)**2),evolution=ne,dimension=reference.dimension)
    np.savez_compressed(ROOT/(P+label+'_neutral.npz'),input=neutral,end=nend)
    for number,create,name,mu in [(5,True,'addition',0),(3,False,'removal',1)]:
        model=m.Model(number,p);raw={};source=model.source(reference,neutral,mu,create)
        weight=float(np.linalg.norm(source)**2);v=source/np.sqrt(weight)
        base,be=evolve(model.apply,v,t,12);raw.update(input=v,baseline=base)
        protocols={};errors=[be]
        barebase,bb=evolve(lambda z:model.apply(z,False,False),v,t,12)
        hopbase,hb=evolve(lambda z:model.apply(z,False,True),v,t,12)
        errors.extend([bb,hb])
        for inc,out in [(0,6),(2,4)]:
            pulse,ki=kick(model,v,inc,.23);end,ee=evolve(model.apply,pulse,t,12);errors.append(ee)
            # Co-polarized nonparallel-q channels; all eight fields remain quantum active.
            protocols[f'in{inc}_out{out}']=dict(kick_info=ki,baseline_electric=float(np.vdot(base,model.field(base,out)).real),kicked_electric=float(np.vdot(end,model.field(end,out)).real),qout_feedback=float((np.vdot(end,model.field(end,out))-np.vdot(base,model.field(base,out))).real),qout_current_feedback=float((-np.vdot(end,model.vertex(end,out))+np.vdot(base,model.vertex(base,out))).real))
            raw.update({f'kick{inc}_input':pulse,f'kick{inc}_end':end})
            if label=='paired':
                ai=model.field(v,inc);aj=model.field(base,out)
                leg,le=evolve(model.apply,ai,t,12);errors.append(le)
                C=np.vdot(aj,leg)
                Ji=-model.vertex(v,inc);Jo=-model.vertex(base,out)
                jleg,je=evolve(model.apply,Ji,t,12);errors.append(je);CJ=np.vdot(Jo,jleg)
                spinop=model.matter.g['L'][2].toarray()
                sx=np.einsum('ij,fjrma->firma',spinop,base.reshape(model.shape),optimize=True).reshape(-1)
                spinleg,se=evolve(model.apply,np.einsum('ij,fjrma->firma',spinop,v.reshape(model.shape),optimize=True).reshape(-1),t,12);errors.append(se)
                mixed=np.vdot(sx,jleg)
                protocols[f'in{inc}_out{out}'].update(electric_two_leg=[C.real.item(),C.imag.item()],electric_reverse=[C.real.item(),-C.imag.item()],electric_retarded=float(2*C.imag),current_two_leg=[CJ.real.item(),CJ.imag.item()],spin_current_two_leg=[mixed.real.item(),mixed.imag.item()],current_contact=float(np.vdot(v,model.vertex(v,inc,out)).real))
                raw.update({f'electric{inc}_leg':leg,f'current{inc}_leg':jleg,f'spin{inc}_leg':spinleg})
            bareend,bp=evolve(lambda z:model.apply(z,False,False),pulse,t,12)
            hopend,hp=evolve(lambda z:model.apply(z,False,True),pulse,t,12)
            errors.extend([bp,hp])
            protocols[f'in{inc}_out{out}']['bare_control_feedback']=float((np.vdot(bareend,model.field(bareend,out))-np.vdot(barebase,model.field(barebase,out))).real)
            protocols[f'in{inc}_out{out}']['hopping_off_feedback']=float((np.vdot(hopend,model.field(hopend,out))-np.vdot(hopbase,model.field(hopbase,out))).real)
            protocols[f'in{inc}_out{out}']['hopping_off_own_current']=float(-np.vdot(hopend,model.vertex(hopend,out,hopping=False,motion=True)).real)
            raw.update({f'bare{inc}_end':bareend,f'hoppingoff{inc}_end':hopend})
        connections={}
        for kind in ['E','g','r','root','R','hop','field']:
            target=part(model,source,kind);before=model.source(reference,part(reference,neutral,kind),mu,create)
            raw['source_connection_'+kind]=target-before;connections[kind]=float(np.linalg.norm(target-before))
        canonical_debt={};hv=model.apply(base)
        for axis in (0,6):
            j=-model.vertex(base,axis);pidot=1j*(model.apply(model.field(base,axis))-model.field(hv,axis))
            adot=1j*(model.apply(model.field(base,axis,'q'))-model.field(hv,axis,'q'))
            canonical_debt[str(axis)]=dict(source_current_norm=float(np.linalg.norm(j)),Pi_dot_minus_source_norm=float(np.linalg.norm(pidot-j)),a_dot_minus_Pi_over_I_norm=float(np.linalg.norm(adot-model.field(base,axis)/p['cavity_inertia'])))
            raw[f'canonical_current_debt{axis}']=pidot-j
        result['branches'][name]=dict(dimension=model.dimension,source_weight=weight,source_connections=connections,protocols=protocols,evolution=errors,baseline=measure(model,base))
        result['branches'][name]['canonical_debt']=canonical_debt
        np.savez_compressed(ROOT/(P+label+'_'+name+'.npz'),**raw)
        (ROOT/(P+label+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(name,json.dumps(dict(dimension=model.dimension,protocols=protocols,source_connections=connections,max_residual=max(e['terminal_residual'] for e in errors))),flush=True)
        del model,source,v,base,raw;gc.collect()
    print(json.dumps(dict(label=label,complete=True)),flush=True)


if __name__=='__main__':main()
