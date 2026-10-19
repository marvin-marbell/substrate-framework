"""ONE dressed source Gram/full-H entry, with batched quantum field coefficients.
No native catalogue or intermediate native projection. Exact local Fock colour
maps cache FULL generator actions, not compressed squares or a scalar binder.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import product
from pathlib import Path
import gc,json,resource,sys,time
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_sources import embedding
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_compile import save


def add(batch,key,coefficient,field):
    if coefficient==0:return
    if key in batch:batch[key]+=coefficient*field
    else:batch[key]=coefficient*field


def inner(left,right):
    if len(left)>len(right):return np.conjugate(inner(right,left))
    return sum(np.vdot(value,right[key]) for key,value in left.items() if key in right)


def source(primitive,omega,spec):
    result={}
    for operations,coefficient in model.source_terms(*spec):
        for key,field in omega.items():
            for target,value in primitive.coordinate_operations(key,operations).items():add(result,target,coefficient*value,field)
    return {key:value for key,value in result.items() if np.any(value)}


def changed(key,site,column):return key[:site]+(column,)+key[site+1:]


class LocalElectric:
    """Exact local16-Fock generator/Casimir images before global projection."""
    def __init__(self,primitive):
        self.primitive=primitive;self.qcache={};self.ccache={}
        self.lengths=np.asarray(primitive.graph['lengths'],float);self.path=np.r_[0.,np.cumsum(self.lengths[:3])]
        self.terms={a:[(8*f+4*s+c,8*f+4*s+d,primitive.ts[a][c,d])
            for f,s,c,d in product(range(2),range(2),range(3),range(3)) if primitive.ts[a][c,d]!=0] for a in range(8)}
    def charge(self,column,a):
        cachekey=(column,a)
        if cachekey not in self.qcache:
            local=self.primitive.local;result=defaultdict(complex);terms=self.terms[a]
            for mask,value in local.vectors[column].items():
                for first,last,coefficient in terms:
                    step=model.string(mask,[(first,True),(last,False)])
                    if step is not None:result[step[0]]+=value*coefficient*step[1]
            self.qcache[cachekey]=local.resolve(model.trim(result))
        return self.qcache[cachekey]
    def casimir(self,column):
        if column not in self.ccache:
            result=defaultdict(complex)
            for a in range(8):
                for middle,value in self.charge(column,a).items():
                    for target,coefficient in self.charge(middle,a).items():result[target]+=value*coefficient
            self.ccache[column]=model.trim(result)
        return self.ccache[column]
    def action(self,key):
        p=self.primitive;lengths=self.lengths;path=self.path
        result=defaultdict(complex)
        for site in range(1,4):
            for target,value in self.casimir(key[site]).items():result[changed(key,site,target)]+=path[site]*value
        for first,last in ((1,2),(1,3),(2,3)):
            for a in range(8):
                for one,value in self.charge(key[first],a).items():
                    for two,coefficient in self.charge(key[last],a).items():result[changed(changed(key,first,one),last,two)]+=2*path[first]*value*coefficient
        for site,a in product(range(1,4),range(8)):
            for local,value in self.charge(key[site],a).items():
                for pw,coefficient in p.field_maps['R',a][key[-1]]:
                    target=changed(key,site,local);result[target[:-1]+(pw,)]+=2*path[site]*value*coefficient
        for pw,value in p.field_maps['C'][key[-1]]:result[key[:-1]+(pw,)]+=float(sum(lengths))*value
        return model.trim({key:p.p['electric']*value/2 for key,value in result.items()})


def neutral_state(data,coupling,state):
    keys,matrix=embedding(data['basis']);nf=data['p']['scalar_points'];result=np.zeros((len(keys),10,nf),complex)
    for r,u in enumerate(coupling['U']):result[:,r,:]=matrix@(u@state.reshape(coupling['dimension'],nf))
    return {key:value for key,value in zip(keys,result) if np.any(value)}


def count_weights(primitive,state):
    result=defaultdict(float)
    for key,field in state.items():result[primitive.label(key)[:-1]]+=float(np.linalg.norm(field)**2)
    return result


def action(primitive,state,scalar,rotor,electric):
    p=primitive.p;generators=angular.right_angular(rotor);inverse=np.asarray(primitive.graph['inertia_inverse'],float)
    kinetic=np.zeros((10,10),complex)
    for a,b in np.ndindex(3,3):kinetic+=inverse[a,b]*(generators[a]@generators[b]).toarray()/2
    axes=np.asarray(primitive.graph['axes']);weights=np.asarray(primitive.graph['weights'])
    rotation={(edge,i):sum((axes[edge,a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex))
        for edge,i in product(range(4),range(3))}
    nonhop={};quark={};white={};timing={'field':0.,'electric':0.,'hops':0.};started=time.perf_counter()
    for index,(key,field) in enumerate(state.items()):
        tick=time.perf_counter();add(nonhop,key,1.,kinetic@field+field@scalar['H'].T)
        for pw,value in primitive.field_maps['B'][key[-1]]:add(nonhop,key[:-1]+(pw,),-p['plaquette']*value,field)
        timing['field']+=time.perf_counter()-tick;tick=time.perf_counter()
        for target,value in electric.action(key).items():add(nonhop,target,value,field)
        timing['electric']+=time.perf_counter()-tick;tick=time.perf_counter();unit={key:1.+0j}
        for edge in range(4):
            fields=[p['scalar_coupling']*weights[edge]*field*scalar['grid'][None,:]]
            fields.extend(1j*p['vector_velocity']*weights[edge]*(rotation[edge,i]@field) for i in range(3))
            for flavour,iswhite,component,dagger in product(range(2),(False,True),range(4),(False,True)):
                selected=white if iswhite else quark;factor=-1 if component and dagger else 1
                image=primitive.hop(unit,edge,flavour,iswhite,component,dagger)
                for target,value in image.items():add(selected,target,factor*value,fields[component])
        timing['hops']+=time.perf_counter()-tick
        if index%128==0:print(json.dumps({'stage':'one_dressed_source_full_H_batched_action','input_row':index,'input_rows':len(state),'nonhop_rows':len(nonhop),'quark_rows':len(quark),'white_rows':len(white)}),flush=True)
    timing['total_action']=time.perf_counter()-started
    return {'nonhop':nonhop,'quark_hop':quark,'white_hop':white},timing


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    data=load(config['neutral_basis_snapshot']);primitive=primitive_for(data);p=data['p']
    coupling=load(config['neutral_coupling_snapshot']);eigen=np.load(config['neutral_states_snapshot'])
    omega=neutral_state(data,coupling,eigen['states'][:,0]);reference_energy=float(eigen['energies'][0])
    scalar,rotor=model.field_and_rotor(p);initial_local=primitive.local.receipt()
    tick=time.perf_counter();ket=source(primitive,omega,config['source_spec']);source_seconds=time.perf_counter()-tick
    del data,coupling,eigen;gc.collect();gram=inner(ket,ket);electric=LocalElectric(primitive)
    components,timing=action(primitive,ket,scalar,rotor,electric)
    entries={name:inner(ket,value) for name,value in components.items()};hamiltonian_entry=sum(entries.values())
    names=list(components);component_gram=np.array([[inner(components[a],components[b]) for b in names] for a in names])
    full_H_norm2=float(np.sum(component_gram).real)
    controls=[]
    for key in (next(iter(ket)),next(reversed(ket))):
        tick=time.perf_counter();original=primitive.electric({key:1.+0j});cached=electric.action(key)
        difference=model.add((1,original),(-1,cached))
        controls.append({'coordinate':key,'full_raw_original_square_error':float(np.sqrt(model.inner(difference,difference).real)),
            'seconds':time.perf_counter()-tick,'original_output_rows':len(original),'cached_output_rows':len(cached)})
    countrows=count_weights(primitive,ket)
    result={'kind':'FIRST_ACTUAL_DRESSED_QUANTUM_SOURCE_GRAM_AND_FULL_H_ENTRY_NOT_CATALOGUE_OR_BINDING',
        'parameters':p,'source_spec':config['source_spec'],'reference_energy':reference_energy,
        'protocol':'Actual OWN dressed neutralJ0 state on its fullPhi7/R10/PW19 local-tensor coordinates, canonical local epsilon/CG source. No native charged projection, no initial/final boson elastic postselection. This probe is not the earlier bare-product leading protocol.',
        'shapes':{'neutral_raw_rows':len(omega),'source_raw_rows':len(ket),'field_channels_per_row':70,
            'action_rows':{name:len(value) for name,value in components.items()},'source_material_count_pieces':len(countrows),
            'source_coefficient_bytes':sum(value.nbytes for value in ket.values()),
            'action_coefficient_bytes':{name:sum(value.nbytes for value in state.values()) for name,state in components.items()}},
        'Gram_real':float(gram.real),'Gram_imag':float(gram.imag),
        'full_H_entry_real':float(hamiltonian_entry.real),'full_H_entry_imag':float(hamiltonian_entry.imag),
        'prepared_source_Rayleigh_energy':float((hamiltonian_entry/gram).real),
        'full_H_action_norm2':full_H_norm2,'prepared_source_energy_variance':float(full_H_norm2/gram.real-abs(hamiltonian_entry/gram)**2),
        'component_entries':{name:{'real':float(value.real),'imag':float(value.imag)} for name,value in entries.items()},
        'component_norm2':{name:float(inner(value,value).real) for name,value in components.items()},
        'timing':{'source_seconds':source_seconds,**timing},'initial_local_basis':initial_local,'final_local_basis':primitive.local.receipt(),
        'local_colour_generator_maps':len(electric.qcache),'local_colour_Casimir_maps':len(electric.ccache),
        'full_raw_original_electric_controls':controls,
        'electric_law':'FULL sum_tree ell(T_desc+Rloop)^2 plus chordellRloop². Exact local16-Fock Q_a maps and Q_a applied TWICE with all newly generated orthogonal remainders before any global projection; local C+cross Q_sQ_t+Q_sR cached. NOT compressed charge square, colour-off or a binder.',
        'projection':'No native P in this H action. Existing local MGS/2e-14 scalar operator roundoff and finitePhi/PW/SO3 cuts disclosed. Future count-resolved source/one-hop-cloud V_j is a NEW severe field/sea-cloud variational span, not old68440 completion or exact full-law range closure.'}
    save(out+'_probe.pkl',{'p':p,'primitive':primitive,'neutral_raw_state':omega,'source':ket,'H_components':components,'reference_energy':reference_energy,'source_spec':config['source_spec']})
    np.savez(out+'_entries.npz',component_Gram=component_gram,Gram=gram,H_entry=hamiltonian_entry)
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
