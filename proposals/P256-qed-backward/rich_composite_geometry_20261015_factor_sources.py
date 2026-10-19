"""Full-raw source metrics on exact occupied/tensor coordinates, no antiC blowup.
Anti source is DIRECT annihilator polynomial evaluated after unitary C^-1 change
of coordinates, precontracted in physical16-mode local Fock space.
"""
from __future__ import annotations
import json,resource,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_sources import embedding
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_runtime import primitive_for,coordinate_operations,conjugate_packet
from rich_composite_geometry_20261015_source_coordinates import operations as dressed_operations
from rich_composite_geometry_20261015_factors import load_h
from rich_composite_geometry_20261015_metrics import isometry


def source_image(primitive,nkeys,xraw,basis,coupling,spec,anti=False):
    entries=[];targets=set()
    for operations,coefficient in model.source_terms(*spec):
        if anti:
            transformed=[]
            for index,_ in operations:
                v,f,s,c=model.unpack(index);coefficient*=(-1)**(v+s)
                transformed.append((model.mode(v,f,1-s,c),False))
            operations=transformed
        for column,key in enumerate(nkeys):
            transformed=dressed_operations(primitive,key,operations) if anti else coordinate_operations(primitive,key,operations)
            for target,value in transformed.items():targets.add(target);entries.append((target,column,coefficient*value))
    targets=sorted(targets);lookup={key:i for i,key in enumerate(targets)}
    matrix=sparse.csr_matrix(([v for _,_,v in entries],([lookup[k] for k,_,_ in entries],[j for _,j,_ in entries])),shape=(len(targets),len(nkeys)))
    matrix.eliminate_zeros();plans={}
    for row,key in enumerate(targets):
        lab=primitive.label(key) if hasattr(primitive,'local') else model.label(key[0])
        block=basis.storage.get(lab)
        if block is not None:
            native_row=block.lookup.get(key)
            if native_row is not None:plans.setdefault(lab,[]).append((row,native_row))
    nf=xraw[0].shape[1];result=np.zeros((coupling['dimension'],nf),complex);full=0.
    native=np.zeros((len(basis.vectors),nf),complex)
    for r,x in enumerate(xraw):
        raw=matrix@x;full+=float(np.linalg.norm(raw)**2);native.fill(0.)
        for lab,indices in plans.items():
            block=basis.storage[lab];augmented,native_rows=map(np.asarray,zip(*indices))
            rhs=np.zeros((len(block.keys),nf),complex);rhs[native_rows]=raw[augmented]
            coefficient=np.conjugate(block.matrix.T@np.conjugate(rhs))
            native[np.asarray(block.columns)]=coefficient
        result+=coupling['U'][r].conj().T@native
    return result.reshape(-1),full


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    neutral=load(config['neutral_basis_snapshot']);charged=load(config['charged_basis_snapshot']);p=neutral['p']
    if charged['p']!=p:raise ValueError('Mismatched actual new source law')
    primitive=primitive_for(charged);primitive_for(neutral,primitive)
    reference=np.load(config['reference_snapshot']);omega=reference['state'];E0=float(reference['energy']);parity=int(reference['Cparity'])
    nkeys,nmatrix=embedding(neutral['basis'])
    uN=load(config['neutral_coupling_snapshot']);nf=p['scalar_points'];xraw=[nmatrix@(u@omega.reshape(uN['dimension'],nf)) for u in uN['U']]
    result={'kind':'ACTUAL_NEW_FULL_RAW_SOURCE_METRIC_GROUNDPOLE_AND_DIRECT_C_ANNIHILATOR_CONSUMER','parameters':p,
        'reference_energy':E0,'sources':{},'anti_sources':{},'profiles':{},
        'anti_coordinate_method':'C_Bdag S_C Omega = Cparity * Bdag (C^-1 S_C C) Omega. S_C is the actual DIRECT particle-hole transformed annihilator polynomial; each C^-1 a C is numerically evaluated on physical16-mode local Fock vectors BEFORE global native/Noether projection. This exact unitary coordinate contraction avoids expanding globalC on every raw tensor coordinate; not a scalar/valence/state fallback. Tree-Wilson source has no loop-winding field operator; other source paths are an explicit cut.'}
    graph=model.geometry(p)
    for j2 in (1,3):
        coupling=load(config[f'charged_J{j2}_coupling_snapshot']);h=load_h(config[f'charged_J{j2}_H_snapshot'])
        eigen=np.load(config[f'charged_J{j2}_states_snapshot']);energies=eigen['energies'];states=eigen['states'];ground=np.flatnonzero(abs(energies-energies[0])<1e-7)
        profile=[]
        for spec in model.source_specs():
            positions,kind,m2=spec
            if m2!=j2 or (kind=='threehalf')!=(j2==3):continue
            image,full=source_image(primitive,nkeys,xraw,charged['basis'],coupling,spec)
            retained=float(np.linalg.norm(image)**2);amplitudes=states[:,ground].conj().T@image;residue=float(np.linalg.norm(amplitudes)**2)
            key=f'{positions}:{kind}:{m2}'
            result['sources'][key]={'positions':positions,'spin_source':kind,'twice_m':m2,'full_raw_CAR_strength':full,
                'retained_strength':retained,'omitted_strength_signed':full-retained,'raw_ground_pole_residue':residue,
                'conditional_pole_fraction':residue/retained if retained else None,'gap':float(energies[0]-E0),
                'preparation_cost':float(np.vdot(image,h@image).real/retained-E0) if retained else None,
                'meaning':'Raw physical-site epsilon/CG canonical source; no forced probabilities or old band factors. Groundpole degeneracy summed; full-source metric before target projection. Prep cost NOT binding.'}
            profile.append({'positions':positions,'spin_source':kind,'weight':(1 if positions[0]==positions[1] else 2)*(j2+1)*residue})
            if len(set(positions))==1:
                anti,afull=source_image(primitive,nkeys,xraw,charged['basis'],coupling,spec,True);anti*=parity
                result['anti_sources'][key]={'N':29,'Ncolour':21,'Nwhite':8,'Q':-1,'baryon_number':-1,
                    'full_raw_CAR_strength':afull,'retained_strength':float(np.linalg.norm(anti)**2),
                    'direct_annihilator_C_coordinate_error':float(np.linalg.norm(anti-parity*image)),
                    'raw_ground_pole_residue':float(np.linalg.norm(states[:,ground].conj().T@anti)**2),
                    'same_H_C_partner_gap':float(energies[0]-E0),
                    'energy_scope':'Same microscopic H C-symmetry partner of actual positive pole, NOT an independently diagonalized antiH or another-law inherited energy. Direct annihilator action/normalization evaluated numerically in unitary C-image coordinates.'}
            print(json.dumps({'stage':'actual_full_raw_spatial_source','source':key,'strength':full,'residue':residue}),flush=True)
        norm=sum(row['weight'] for row in profile)
        for row in profile:
            pos=row['positions'];row['probe_fraction']=row['weight']/norm if norm else None
            row['relative_radius_squared']=sum(np.linalg.norm(graph['vertices'][pos[i]]-graph['vertices'][pos[j]])**2 for i in range(3) for j in range(i))/3
        result['profiles'][str(j2)]={'rows':profile,'raw_weight_sum':norm,
            'compact_source_profile_fraction':sum(r['weight'] for r in profile if len(set(r['positions']))==1)/norm if norm else None,
            'mean_relative_radius_squared':sum(r['weight']*r['relative_radius_squared'] for r in profile)/norm if norm else None,
            'meaning':'Ordered-u and complete-magnetic-multiplet weighted POSITIVE COHERENT GROUND-POLE SOURCE TRANSITION profile; NOT a unique3-added-particle position PVM, capture probability or binding by normalization.'}
        Path(out+'_checkpoint.json').write_text(json.dumps(result,indent=2)+'\n')
    controls=[]
    for index in np.unique(np.linspace(0,len(charged['basis'].vectors)-1,12,dtype=int)):
        packet=charged['basis'].vectors[index];cp=conjugate_packet(primitive,packet)
        for name,action in [('scalar',lambda v:primitive.collective(v)),('vector',lambda v:primitive.collective(v,2,1)),
                ('electric',primitive.electric),('loopB',lambda v:primitive.field(v,primitive.field_maps['B']))]:
            difference=model.add((1,action(cp)),(-1,conjugate_packet(primitive,action(packet))))
            controls.append({'basis_index':int(index),'operator':name,'full_raw_C_H_ingredient_error':float(np.sqrt(model.inner(difference,difference).real))})
    result['actual_C_H_controls']={'rows':controls,'scope':'Full raw original H-ingredient intertwining on12 deterministic native columns; sampled control NOT a full native-C operator bound or independent antiH eigensolve. Algebraic bipartite C law stated separately.'}
    result['physical_isometry_after_source_coordinate_operations']=isometry(charged['basis'],primitive)
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']),flush=True)


if __name__=='__main__':main()
