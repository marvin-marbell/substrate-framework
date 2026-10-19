"""Actual extended-state nonlinear primitive H/current BODY omissions.
Exact raw projector image quadratics are streamed one at a time. Triangle
bounds hold R/phi/field/PW cuts fixed; they are NOT total physical closure.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from rich_spatial_colour_charge_20261015_snapshot import load as load_native
from rich_spatial_colour_charge_20261015_model import forward,electric_image,frame
from rich_spatial_colour_em_20261015_snapshot import load,source
from rich_spatial_colour_em_20261015_fields import phase_kernel
from rich_spatial_colour_em_20261015_raw_support import native_image_metric


def primitives(body,name):
    basis=body['native'][name];loop=body['loop'];op=body['operators'][name]
    yield 'electric',0,[electric_image(x,loop,body['parameters'],body['geometry']) for x in basis.seeds],op['electric'],dict(kind='electric')
    yield 'B',0,[{m:loop['B']@v for m,v in x.items()} for x in basis.seeds],op['B'],dict(kind='B')
    spins=[np.eye(2)]+frame()['sigma']
    for edge in range(2):
        for channel,(q,matrices) in enumerate(op['hops'][edge]):
            for index,spin in enumerate(spins):
                for adjoint in [False,True]:
                    label=f'hop_e{edge}_c{channel}_s{index}_a{int(adjoint)}'
                    images=[forward(x,loop,edge,channel//2,channel%2,spin,adjoint=adjoint) for x in basis.seeds]
                    retained=matrices[index].conj().T if adjoint else matrices[index]
                    yield label,(1 if adjoint else -1)*round(3*q),images,retained,dict(kind='hop',edge=edge,charge=q,spin_axis=None if index==0 else index-1,adjoint=adjoint)


def environment_image(state,nm,scalar,nrq,kernel,phi):
    nf=len(scalar['grid']);x=state.reshape(nm,nf,nrq)
    y=(kernel@x.transpose(2,0,1).reshape(nrq,-1)).reshape(nrq,nm,nf).transpose(1,2,0)
    if phi:y=y*scalar['grid'][None,:,None]
    return y.reshape(nm,-1)


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent;started=time.time()
    state_json=root/request['state_json'];state_npz=root/request['state_npz'];metadata,data,model,laws=load(state_json,state_npz)
    cache=root/request['native_body_cache'];digest=hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest!=request['native_body_cache_sha256'] or digest!=model['parameters']['native_body_cache_sha256']:raise ValueError('Raw support requires the SAME explicitly trusted pinned native body as this extended H')
    body=load_native(cache);base=model['base'];p=model['parameters'];field=model['field'];rotor=base['rotor'];nrq=rotor['dimension']*field['dimension'];scalar=base['scalar']
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [state_json,state_npz,cache]}
    states={};results={};controls={};records={}
    for case in request['cases']:
        name=case['name'];key=case.get('key');label=name if key is None else name+'_'+key
        raw=data['neutral_reference'] if key is None else source(data,model,name,key)@data['neutral_reference']
        states[label]=(name,raw/np.linalg.norm(raw));results[label]=dict(H_primitive_components=[],current_primitive_components=[],H_body_triangle_norm_bound=0.,current_body_triangle_norm_bounds=[0.]*6,minimum_signed_component_norm_squared=0.)
    def checkpoint(stage):
        payload=dict(kind='RAW_NATIVE_NONLINEAR_IMAGE_SUPPORT_NOT_FULL_PHYSICAL_CLOSURE',stage=stage,request=request,input_sha256=hashes,positive_controls=controls,primitive_metrics=records,results=results,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Native-body omissions of the ACTUAL H/current primitive images, holding projected R/phi/field/PW19 factors fixed. Full charge-orbit states retain coherent all band-pair orders but are NOT assumed H/current closed. Bounds are numerical triangle bounds, not exact coherent total-H residuals. R/Q complements and PW/scalar/angular boundaries are separate; no full pair/photon/COM claim.')
        out.write_text(json.dumps(payload,indent=2)+'\n')
    for name in ['neutral','lepton','baryon']:
        relevant={label:item for label,item in states.items() if item[0] in [name,'anti'+name]}
        if not relevant:continue
        basis=body['native'][name];identity,evidence,grams=native_image_metric(basis,basis.seeds,0)
        gram_error=np.sqrt(sum(np.linalg.norm(grams.get(h,np.zeros((len(basis.seeds),)*2))-basis.grams[h].toarray())**2 for h in basis.grams))
        controls[name]=dict(**evidence,complete_seed_Gram_vs_frozen_native_Frobenius=float(gram_error),native_identity_metric_Frobenius=float(sparse.linalg.norm(identity-sparse.eye(basis.dimension))))
        checkpoint('identity_'+name)
        for primitive,shift,images,retained,definition in primitives(body,name):
            full,evidence,_=native_image_metric(basis,images,shift);outside=(full-retained.conj().T@retained).tocsr()
            metric_file=out.with_name(out.stem+'_'+name+'_'+primitive+'_body_omission.npz');sparse.save_npz(metric_file,outside)
            record=dict(**definition,**evidence,metric=metric_file.name,metric_sha256=hashlib.sha256(metric_file.read_bytes()).hexdigest(),outside_metric_Frobenius=float(sparse.linalg.norm(outside)))
            records[name+'_'+primitive]=record
            for label,(actual_name,state) in relevant.items():
                result=results[label];nm=basis.dimension
                if definition['kind']!='hop':
                    coefficient=1. if definition['kind']=='electric' else -p['plaquette'];y=state.reshape(nm,-1);r2=float(np.sum(y.conj()*(outside@y)).real)*abs(coefficient)**2
                    currents=[0.]*6
                else:
                    adjoint=definition['adjoint'] if actual_name==name else not definition['adjoint']
                    edge=definition['edge'];spin=definition['spin_axis'];q=definition['charge']
                    coefficient=base['geometry']['weights'][edge]*(p['scalar_coupling'] if spin is None else 1j*p['vector_velocity'])
                    kernel=phase_kernel(rotor,base['geometry'],model['coefficients'],field,edge,q,spin_axis=spin)
                    if adjoint:kernel=kernel.conj().T
                    y=environment_image(state,nm,scalar,nrq,kernel,spin is None);r2=float(np.sum(y.conj()*(outside@y)).real)*abs(coefficient)**2
                    currents=[]
                    for mu in range(6):
                        kernel=phase_kernel(rotor,base['geometry'],model['coefficients'],field,edge,q,derivative=(mu,),spin_axis=spin)
                        if adjoint:kernel=kernel.conj().T
                        y=environment_image(state,nm,scalar,nrq,kernel,spin is None);currents.append(float(np.sum(y.conj()*(outside@y)).real)*abs(coefficient)**2)
                result['H_primitive_components'].append(dict(primitive=primitive,signed_omitted_norm_squared=r2))
                result['current_primitive_components'].append(dict(primitive=primitive,signed_omitted_norm_squared=currents))
                result['H_body_triangle_norm_bound']+=float(np.sqrt(abs(r2)))
                result['current_body_triangle_norm_bounds']=[a+float(np.sqrt(abs(b))) for a,b in zip(result['current_body_triangle_norm_bounds'],currents)]
                result['minimum_signed_component_norm_squared']=min(result['minimum_signed_component_norm_squared'],r2,*currents)
                result['negative_component_semantics']='Signed quadratic witnesses are preserved. Absolute values enter ONLY conservative numerical triangle norm sums; significant negative components invalidate the numerical PSD witness, never change H or clamp physical mode energies.'
                result['charge_conjugate_metric_semantics']='Anti uses the exact native C-image body coordinates: forward/adjoint primitive metrics swap, vector signs drop only from individual norms, Q/field/current themselves are NOT scalarized. No positive energy/state/reference is inherited.'
            checkpoint('primitive_'+name+'_'+primitive)
            del images,full,outside
        print(json.dumps(dict(name=name,completed_primitives=sum(k.startswith(name+'_') for k in records),identity=controls[name],peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)
    for result in results.values():result['numerical_PSD_status']='nonnegative_with_roundoff' if result['minimum_signed_component_norm_squared']>=-1e-8 else 'INVALID_SIGNIFICANT_NEGATIVE_IMAGE_METRIC'
    checkpoint('complete')


if __name__=='__main__':main()
