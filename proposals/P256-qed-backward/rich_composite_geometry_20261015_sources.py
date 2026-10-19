"""Actual raw CAR Wilson-source metrics/residues; profiles are NOT position PVMs."""
from __future__ import annotations
import json,pickle,resource,sys,time
from pathlib import Path
from itertools import product
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_runtime import primitive_for,conjugate_packet,coordinate_operations


def embedding(basis):
    keys=sorted({key for packet in basis.vectors for key in packet});lookup={key:i for i,key in enumerate(keys)}
    rows=[];columns=[];values=[]
    for j,packet in enumerate(basis.vectors):
        for key,value in packet.items():rows.append(lookup[key]);columns.append(j);values.append(value)
    return keys,sparse.csr_matrix((values,(rows,columns)),shape=(len(keys),len(basis.vectors)))


def raw_source(primitive,keys,spec,removal=False):
    operations=model.source_terms(*spec);targets=set();entries=[]
    for ops,coefficient in operations:
        if removal:
            transformed=[]
            for index,_ in ops:
                v,f,s,c=model.unpack(index);coefficient*=(-1)**(v+s)
                transformed.append((model.mode(v,f,1-s,c),False))
            ops=transformed
        for column,key in enumerate(keys):
            for target,value in coordinate_operations(primitive,key,ops).items():
                targets.add(target);entries.append((target,column,coefficient*value))
    targets=sorted(targets);lookup={key:i for i,key in enumerate(targets)}
    matrix=sparse.csr_matrix(([e[2] for e in entries],([lookup[e[0]] for e in entries],[e[1] for e in entries])),shape=(len(targets),len(keys)))
    matrix.eliminate_zeros()
    return targets,matrix


def apply_source(primitive,neutral_keys,xraw,target_keys,target_embedding,coupling,spec,removal=False):
    targets,matrix=raw_source(primitive,neutral_keys,spec,removal);lookup={key:i for i,key in enumerate(target_keys)}
    present=np.array([i for i,key in enumerate(targets) if key in lookup],int)
    native_rows=np.array([lookup[targets[i]] for i in present],int)
    projection=target_embedding[native_rows].conj().T
    nf=xraw[0].shape[1];image=np.zeros((coupling['dimension'],nf),complex);full=0.
    for r,x in enumerate(xraw):
        raw=matrix@x;full+=float(np.linalg.norm(raw)**2)
        if len(present):image+=coupling['U'][r].conj().T@(projection@raw[present])
    return image.reshape(-1),full


def conjugate_embedding(primitive,keys,matrix):
    entries=[];targets=set()
    for column,key in enumerate(keys):
        for target,value in conjugate_packet(primitive,{key:1.}).items():
            entries.append((target,column,value));targets.add(target)
    newkeys=sorted(targets);lookup={key:i for i,key in enumerate(newkeys)}
    c=sparse.csr_matrix(([v for _,_,v in entries],([lookup[k] for k,_,_ in entries],[j for _,j,_ in entries])),shape=(len(newkeys),len(keys)))
    return newkeys,c@matrix


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    neutral=load(config['neutral_basis_snapshot']);charged=load(config['charged_basis_snapshot']);p=neutral['p']
    primitive=primitive_for(charged);primitive_for(neutral,primitive)
    reference=np.load(config['reference_snapshot']);omega=reference['state'];E0=float(reference['energy']);Cparity=int(reference['Cparity'])
    nkeys,nmatrix=embedding(neutral['basis']);bkeys,bmatrix=embedding(charged['basis'])
    uN=load(config['neutral_coupling_snapshot']);nf=p['scalar_points']
    xraw=[nmatrix@(u@omega.reshape(uN['dimension'],nf)) for u in uN['U']]
    anti_keys,anti_matrix=conjugate_embedding(primitive,bkeys,bmatrix)
    result={'kind':'ACTUAL_NEW_FULL_RAW_WILSON_SOURCE_METRIC_POLE_PROFILE_AND_C','parameters':p,'reference_energy':E0,'sources':{},'profiles':{},'anti_sources':{}}
    for j2 in (1,3):
        coupling=load(config[f'charged_J{j2}_coupling_snapshot']);h=sparse.load_npz(config[f'charged_J{j2}_H_snapshot'])
        eigen=np.load(config[f'charged_J{j2}_states_snapshot']);energies=eigen['energies'];states=eigen['states'];ground=np.flatnonzero(abs(energies-energies[0])<1e-7)
        profile=[]
        for spec in model.source_specs():
            positions,kind,m2=spec
            if m2!=j2 or (kind=='threehalf')!=(j2==3):continue
            image,full=apply_source(primitive,nkeys,xraw,bkeys,bmatrix,coupling,spec)
            retained=float(np.linalg.norm(image)**2);amplitudes=states[:,ground].conj().T@image;residue=float(np.linalg.norm(amplitudes)**2)
            key=f'{positions}:{kind}:{m2}'
            row={'positions':positions,'spin_source':kind,'twice_m':m2,'full_raw_CAR_strength':full,'retained_strength':retained,
                'omitted_strength_signed':full-retained,'raw_ground_pole_residue':residue,
                'conditional_pole_fraction':residue/retained if retained else None,'ground_gap':float(energies[0]-E0),
                'preparation_cost':float(np.vdot(image,h@image).real/retained-E0) if retained else None,
                'meaning':'Raw canonical epsilon/CG physical-site source, no old band-normalization factors; degenerate fixed-M ground-pole residues summed. Prep cost NOT binding.'}
            result['sources'][key]=row
            degeneracy=1 if positions[0]==positions[1] else 2
            profile.append({'positions':positions,'spin_source':kind,'ordered_position_pole_weight':degeneracy*(j2+1)*residue})
            if len(set(positions))==1:
                anti,afull=apply_source(primitive,nkeys,xraw,anti_keys,anti_matrix,coupling,spec,True)
                result['anti_sources'][key]={'full_raw_CAR_strength':afull,'retained_strength':float(np.linalg.norm(anti)**2),
                    'direct_C_source_image_error':float(np.linalg.norm(anti-Cparity*image)),
                    'raw_ground_pole_residue':float(np.linalg.norm(states[:,ground].conj().T@anti)**2),
                    'same_H_C_partner_gap':float(energies[0]-E0),
                    'energy_scope':'Symmetry-derived same-H C partner, not independently diagonalized antiH and never inherited other-law energies. Actual DIRECT raw annihilator-polynomial source image/metric checked AFTER native and Noether selection.'}
            print(json.dumps({'stage':'raw_Wilson_source_pole','source':key,'strength':full,'residue':residue}),flush=True)
        norm=sum(r['ordered_position_pole_weight'] for r in profile);graph=model.geometry(p)
        for row in profile:
            pos=row['positions'];row['source_probe_fraction']=row['ordered_position_pole_weight']/norm if norm else None
            row['relative_radius_squared']=sum(np.linalg.norm(graph['vertices'][pos[i]]-graph['vertices'][pos[j]])**2 for i in range(3) for j in range(i))/3
        result['profiles'][str(j2)]={'rows':profile,'raw_weight_sum':norm,
            'compact_profile_fraction':sum(r['ordered_position_pole_weight'] for r in profile if len(set(r['positions']))==1)/norm if norm else None,
            'mean_relative_radius_squared':sum(r['ordered_position_pole_weight']*r['relative_radius_squared'] for r in profile)/norm if norm else None,
            'meaning':'Positive normalized COHERENT GROUND-POLE TRANSITION source-coordinate profile; ordered-u multiplicity and complete magnetic multiplet factor included. Not mutually exclusive three-added-particle Born-position PVM, not binding by normalization.'}
        Path(out+'_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    controls=[]
    indices=np.unique(np.linspace(0,len(charged['basis'].vectors)-1,12,dtype=int))
    for index in indices:
        packet=charged['basis'].vectors[index];cp=conjugate_packet(primitive,packet)
        for name,action in [('scalar',lambda v:primitive.collective(v)),('vector',lambda v:primitive.collective(v,2,1)),
                ('electric',primitive.electric),('loopB',lambda v:primitive.field(v,primitive.field_maps['B']))]:
            difference=model.add((1,action(cp)),(-1,conjugate_packet(primitive,action(packet))))
            controls.append({'basis_index':int(index),'operator':name,'raw_C_intertwining_norm':float(np.sqrt(model.inner(difference,difference).real))})
    result['raw_H_C_controls']={'rows':controls,'scope':'Actual unprojected CAR64/PW19 original H ingredients on12 deterministic charged native columns. BILINEAR C law and complete bipartite/PW conjugation are specified algebraically; this sampled witness is NOT a full native operator bound or an independent antiH solve.'}
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
