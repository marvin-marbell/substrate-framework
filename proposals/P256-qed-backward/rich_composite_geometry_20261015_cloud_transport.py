"""Actual Omega-conditioned tree01 K3: elastic source endpoint vs INCLUSIVE.
Exact quark/white factor extraction is MEASURED, never bare-Phi parity assumed.
All64 ordered quantum-field words retained; commuting forward CAR bilinears
only combine their matter packets into20 exact multisets. No native P or P0.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from itertools import combinations_with_replacement,product
from pathlib import Path
import gc,json,resource,sys,time
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_tensor import local_seed
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_science import load
from rich_composite_geometry_20261015_compile import save
from rich_composite_geometry_20261015_cloud_probe import neutral_state,source,inner as batch_inner

QPOSITIONS=tuple(index for index in range(16) if index%4!=3)
WPOSITIONS=(3,7,11,15)
QINDEX={index:column for column,index in enumerate(QPOSITIONS)}


@lru_cache(maxsize=65536)
def split_local(mask):
    quark=sum(((mask>>index)&1)<<column for column,index in enumerate(QPOSITIONS))
    white=sum(((mask>>index)&1)<<column for column,index in enumerate(WPOSITIONS))
    swaps=sum(((mask>>w)&1)*sum((mask>>q)&1 for q in QPOSITIONS if q>w) for w in WPOSITIONS)
    return quark,white,(-1)**swaps


def scalar_inner(left,right):
    if len(left)>len(right):return np.conjugate(scalar_inner(right,left))
    return sum(np.conjugate(value)*right.get(key,0.) for key,value in left.items())


def quark_root():
    seed=local_seed();qkeys=sorted({split_local(mask)[0] for mask in seed});wkeys=sorted({split_local(mask)[1] for mask in seed})
    qlookup={key:index for index,key in enumerate(qkeys)};wlookup={key:index for index,key in enumerate(wkeys)};matrix=np.zeros((len(qkeys),len(wkeys)),complex)
    for mask,value in seed.items():
        q,w,sign=split_local(mask);matrix[qlookup[q],wlookup[w]]+=sign*value
    u,s,vh=np.linalg.svd(matrix,full_matrices=False)
    return {key:value for key,value in zip(qkeys,u[:,0])},float(np.linalg.norm(matrix-s[0]*u[:,:1]@vh[:1])**2),s


def quark_source(root,kind,m2):
    result=defaultdict(complex)
    for operations,coefficient in model.source_terms((0,0,0),kind,m2):
        mapped=[(QINDEX[index],create) for index,create in operations]
        for mask,value in root.items():
            step=model.string(mask,mapped)
            if step is not None:result[step[0]]+=value*coefficient*step[1]
    result=model.trim(result);strength=float(scalar_inner(result,result).real)
    if strength==0:raise ValueError('Actual local quark source absent; no state replacement')
    return {key:value/np.sqrt(strength) for key,value in result.items()},strength


class WhiteFieldExtractor:
    def __init__(self,primitive):self.primitive=primitive;self.cache={}
    def local(self,column,reference_name,quark):
        cachekey=(column,reference_name)
        if cachekey not in self.cache:
            result=defaultdict(complex)
            for mask,value in self.primitive.local.vectors[column].items():
                q,w,sign=split_local(mask)
                if q in quark:result[w]+=value*sign*np.conjugate(quark[q])
            self.cache[cachekey]=model.trim(result)
        return self.cache[cachekey]
    def extract(self,state,profiles,names,quark_numbers):
        result={};tail=[sum(quark_numbers[site+1:])%2 for site in range(4)]
        for key,field in state.items():
            factors=[self.local(key[site],names[site],profiles[site]) for site in range(4)]
            for values in product(*(factor.items() for factor in factors)):
                white=sum(w<<(4*site) for site,(w,_) in enumerate(values));coefficient=np.prod([value for _,value in values])
                sign=(-1)**sum(w.bit_count()*tail[site] for site,(w,_) in enumerate(values))
                target=(white,key[-1]);value=sign*coefficient*field
                if target in result:result[target]+=value
                else:result[target]=value.copy()
        return result


def compact_source(primitive,omega,site,kind,m2):
    state=source(primitive,omega,([site]*3,kind,m2));full=float(batch_inner(state,state).real);selected={}
    for key,field in state.items():
        counts=np.asarray(primitive.label(key)[:-1]).reshape(4,4);colours=counts[:,0]+counts[:,2];expected=np.full(4,6);expected[site]=9
        if np.array_equal(colours,expected):selected[key]=field
    strength=float(batch_inner(selected,selected).real)
    if strength==0:raise ValueError('Pinned compact source has no support; no receiver replacement')
    return {key:field/np.sqrt(strength) for key,field in selected.items()},full,strength


def active_tensor(left,right):return {a|(b<<12):x*y for a,x in left.items() for b,y in right.items()}


def quark_forward(packet,spin):
    result=defaultdict(complex)
    terms=[([(12+6*f+3*s+c,True),(6*f+3*t+c,False)],spin[s,t])
        for f,c,s,t in product(range(2),range(3),range(2),range(2)) if spin[s,t]!=0]
    for mask,value in packet.items():
        for operations,coefficient in terms:
            step=model.string(mask,operations)
            if step is not None:result[step[0]]+=value*coefficient*step[1]
    return model.trim(result)


def field_step(field,channel,scalar,rotations,p,weight):
    if channel==0:return p['scalar_coupling']*weight*field*scalar['grid'][None,None,None,:]
    return 1j*p['vector_velocity']*weight*np.einsum('rs,npsf->nprf',rotations[channel-1],field)


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    data=load(config['neutral_basis_snapshot']);primitive=primitive_for(data);p=data['p'];coupling=load(config['neutral_coupling_snapshot']);eigen=np.load(config['neutral_states_snapshot'])
    omega=neutral_state(data,coupling,eigen['states'][:,0]);reference_energy=float(eigen['energies'][0]);del data,coupling,eigen;gc.collect()
    root,root_factor_error,root_singular_values=quark_root();kind=config['initial_kind'];m2=config['initial_twice_m'];qinitial,canonical_strength=quark_source(root,kind,m2)
    extractor=WhiteFieldExtractor(primitive);state,raw_strength,compact_strength=compact_source(primitive,omega,0,kind,m2)
    initial=extractor.extract(state,[qinitial,root,root,root],['initial','root','root','root'],[9,6,6,6]);norm=float(batch_inner(initial,initial).real)
    preparations={'initial':{'full_raw_CAR_strength':raw_strength,'compact_conditioning_strength':compact_strength,'raw_normalized_state_norm2':float(batch_inner(state,state).real),
        'quark_factor_projection_norm2':norm,'factorization_signed_missing_norm2':float(batch_inner(state,state).real)-norm,'white_field_rows':len(initial)}}
    targets={};qtargets={};target_strengths={}
    for target_kind,magnetic in (('half_triplet',(-1,1)),('threehalf',(-3,-1,1,3))):
        for target_m in magnetic:
            key=f'{target_kind}:{target_m}';quark,strength=quark_source(root,target_kind,target_m);qtargets[key]=active_tensor(root,quark);target_strengths[key]=strength
            target,raw,conditioned=compact_source(primitive,omega,1,target_kind,target_m)
            field=extractor.extract(target,[root,quark,root,root],['root',key,'root','root'],[6,9,6,6]);projected=float(batch_inner(field,field).real)
            targets[key]=field;preparations[key]={'full_raw_CAR_strength':raw,'compact_conditioning_strength':conditioned,'raw_normalized_state_norm2':float(batch_inner(target,target).real),
                'quark_factor_projection_norm2':projected,'factorization_signed_missing_norm2':float(batch_inner(target,target).real)-projected,'white_field_rows':len(field)}
    result={'kind':'ACTUAL_OMEGA_CONDITIONED_INCLUSIVE_VS_OMEGA_MATCHED_PRIMITIVE_K3','parameters':p,'reference_energy':reference_energy,
        'local_quark_white_seed_factor_error_norm2':root_factor_error,'local_seed_singular_values':root_singular_values.tolist(),
        'preparations':preparations,'initial_quark_canonical_strength':canonical_strength,'target_quark_canonical_strengths':target_strengths,
        'protocol':'S0dag OWN actual dressedOmega then true compact-count0 conditioning; endpoint1 fixed BEFORE data. White sea/Phi/R/PW cloud extracted from ACTUAL state with canonical quark-first/white-last CAR permutation signs. Actual Omega-matched targets independently extracted at1; no initial-field postselection imposed on inclusive receiver.',
        'factorization_scope':'Measured quark-core factorization is a REPRESENTATION identity for this existing one-hop reference slice, not a particle rigidity/admission assumption. If missing norm exceeds numerical metric tolerance, calculation must use a richer exact core decomposition, not silently substitute the bare seed.'}
    Path(out+'_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    if root_factor_error>1e-20 or max(abs(row['factorization_signed_missing_norm2']) for row in preparations.values())>1e-10:
        raise ArithmeticError('Actual dressed quark-core factorization failed; no bare-state/field fallback')
    white_masks=sorted({key[0] for field in [initial]+list(targets.values()) for key in field});white_lookup={mask:i for i,mask in enumerate(white_masks)}
    def array(field):
        value=np.zeros((len(white_masks),19,10,p['scalar_points']),complex)
        for (white,pw),coefficient in field.items():value[white_lookup[white],pw]=coefficient
        return value
    initial_field=array(initial);target_fields={key:array(value) for key,value in targets.items()};scalar,rotor=model.field_and_rotor(p)
    graph=model.geometry(p);axis=np.asarray(graph['axes'])[0];weight=float(np.asarray(graph['weights'])[0]);rotations=[sum((axis[a]*sparse.csr_matrix(rotor['coordinate'][i][a]).toarray() for a in range(3)),np.zeros((10,10),complex)) for i in range(3)]
    spins=(np.eye(2),)+model.PAULI
    packets:dict[tuple[int,...],dict[int,complex]]={():active_tensor(qinitial,root)}
    by_degree:dict[int,dict[tuple[int,...],np.ndarray]]={0:{():initial_field}}
    for degree in range(1,4):
        for label in combinations_with_replacement(range(4),degree):packets[label]=quark_forward(packets[label[:-1]],spins[label[-1]])
        grouped={label:np.zeros_like(initial_field) for label in combinations_with_replacement(range(4),degree)}
        for word in product(range(4),repeat=degree):
            value=initial_field
            for channel in word:value=field_step(value,channel,scalar,rotations,p,weight)
            grouped[tuple(sorted(word))]+=value
        by_degree[degree]=grouped
    labels=list(by_degree[3]);qgram=np.array([[scalar_inner(packets[a],packets[b]) for b in labels] for a in labels]);fgram=np.array([[np.vdot(by_degree[3][a],by_degree[3][b]) for b in labels] for a in labels])
    inclusive=np.sum(qgram*fgram);elastic={};source_matched_inclusive={}
    for target in targets:
        amplitude=sum(scalar_inner(qtargets[target],packets[label])*np.vdot(target_fields[target],by_degree[3][label]) for label in labels)
        elastic[target]={'amplitude_real':float(amplitude.real),'amplitude_imag':float(amplitude.imag),'norm2':float(abs(amplitude)**2)}
        field=sum(scalar_inner(qtargets[target],packets[label])*by_degree[3][label] for label in labels)
        source_matched_inclusive[target]=float(np.linalg.norm(field)**2)
    result.update(inclusive_full_compact_K3_norm2=float(inclusive.real),inclusive_norm2_imag_roundoff=float(inclusive.imag),
        inclusive_short_time_t6_coefficient=float(inclusive.real/36),Omega_matched_targets=elastic,
        Omega_matched_total_norm2=sum(row['norm2'] for row in elastic.values()),
        source_matched_quark_only_ALL_final_fields_norm2=source_matched_inclusive,
        source_matched_quark_only_total_norm2=sum(source_matched_inclusive.values()),
        exact_representation={'quark_forward_channels':4,'ordered_field_words':64,'commuting_forward_CAR_matter_multisets':20,
            'largest_quark_packet_rows':max(map(len,packets.values())),'white_Fock_masks':len(white_masks),'quantum_PW_modes':19,'SO3_functions':10,'Phi_coordinates':p['scalar_points']},
        projection_placement='NO intermediate native/W or initial-field projector. ALL ordered finitePhi7/R10 field products retained; tree01 WilsonI is the actual gauge-fixed link, PW19 incoming quantum cloud retained. Forward CAR bilinears commute because disjoint origin/destination; field matrices are NOT assumed to commute. Full quark/sea states generated at active sites retained, other quark sites unchanged ONLY by exact3-net-hop channel necessity, white sea remains its actual quantum correlated incoming state.',
        meaning='Inclusive P_compact1 Hprimitive^3 psi0 is an actual raw energy^6 moment, coefficient t6/36, NOT finite-time probability, binding, lifetime, freeCOM or a source-optimized receiver. Omega-matched return is a DIFFERENT detector and not a particle gate; emitted/changed fields allowed. Earlier barePhi seed result is NOT inherited. Existing W(PHP)^3/full-H residual and selected-field cloud cuts remain distinct.' )
    save(out+'_transport_CP.pkl',{'p':p,'quark_root':root,'initial_quark_source':qinitial,'quark_packets':packets,'white_masks':white_masks,
        'field_CP_by_degree':by_degree,'target_quark_packets':qtargets,'target_white_field':target_fields,'preparations':preparations})
    np.savez(out+'_K3_Grams.npz',quark_Gram=qgram,field_Gram=fgram)
    result['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
