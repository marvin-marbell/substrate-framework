"""Raw source spectral/propagation readout of the exact new spatial H snapshots.
All source strengths remain unnormalized; full resolvents are not low-pole sums.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from rich_spatial_colour_root_20261015_model import TensorOperator
from rich_spatial_colour_root_20261015_calculate import solve_resolvent


def sparse_matrix(data,name):
    return sparse.csr_matrix((data[name+'_data'],data[name+'_indices'],data[name+'_indptr']),shape=tuple(data[name+'_shape']))


def cjson(value):return dict(real=np.asarray(value).real.tolist(),imag=np.asarray(value).imag.tolist())

def vector_max(data,prefix):
    return max(float(np.linalg.norm([data[prefix+str(i)][j] for i in range(3)])) for j in range(len(data[prefix+'0'])))


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent;prefix=request['producer_prefix'];started=time.time()
    paths=[root/(prefix+suffix) for suffix in ['_input.json','_operators.npz','_states.npz','_result.npz','_result.json']]
    p=json.loads(paths[0].read_text());operators=np.load(paths[1]);states=np.load(paths[2]);events=np.load(paths[3]);result=json.loads(paths[4].read_text())
    scalar=dict(H=operators['scalar_H'],grid=operators['scalar_grid']);e0=result['sectors']['neutral']['ground_energy']
    low_weights={};propagation={};event_facts={};raw_green={}
    for name,sources in result['sources'].items():
        energies=states[name+'_low_energies'];vectors=states[name+'_low_vectors'];low_weights[name]={}
        for key,source in sources.items():
            if key=='spatial_preparation_Gram':continue
            raw=states[name+'_source_state_'+key];weights=abs(vectors.conj().T@raw)**2
            low_weights[name][key]=dict(raw_strength=float(np.vdot(raw,raw).real),computed_low_gaps=(energies-e0).tolist(),
                computed_low_raw_weights=weights.tolist(),uncomputed_high_raw_strength=float(np.vdot(raw,raw).real-weights.sum()),
                meaning='Raw CAR spectral weights on orthonormal computed low eigenspaces; no probability normalization and no claim these low eigenstates exhaust source response.')
        baseline=events[name+'_baseline'];source0=states[name+'_source_state_0:0.5'];normalization=float(np.linalg.norm(source0));captures={}
        for placement in result['sources'][name]['spatial_preparation_Gram']['labels']:
            raw=states[name+'_source_state_'+placement+':0.5'];amplitude=normalization*(raw.conj()@baseline)
            captures[placement]=dict(raw_G_t=cjson(amplitude),conditional_overlap_probabilities=(abs(amplitude)**2/(np.vdot(raw,raw).real*normalization**2)).tolist())
        propagation[name]=dict(times=p['times'],receivers=captures,
            meaning='Raw G_s0(t)=<Omega|S_s exp(-i Hcharged t) S_0dag|Omega> with exact full retained-H propagation. No subtraction of nonzero t0 occupied-sea overlap. Receiver overlap is NOT exclusive composite position; hop-off controls expose field dephasing and pair-cut density contamination.')
    for name,reactions in result['reactions'].items():
        facts={}
        for protocol in ['A','B','field_only']:
            ex=reactions[protocol]['excess'];facts[protocol]=dict(max_abs_rho1_excess=max(map(abs,ex['rho1'])),
                max_abs_route_density_current=max(map(abs,ex['density_transport_current'])),
                max_abs_circulation_current=max(map(abs,ex['circulation_current'])),
                max_Cartesian_current_excess=vector_max(ex,'I'),max_rotational_current_excess=vector_max(ex,'Irot'),
                max_magnetic_moment_excess=vector_max(ex,'mu'),max_rotational_moment_excess=vector_max(ex,'murot'),
                max_abs_tree_flux_excess=max(map(abs,ex['tree_C'])),max_abs_loop_C_excess=max(map(abs,ex['C'])),
                max_abs_phi_excess=max(map(abs,ex['phi'])),max_abs_J2_excess=max(map(abs,ex['J2'])),
                site_POVM_sums={key:value['sum'] for key,value in reactions[protocol]['physical_count_POVM_final'].items()})
            if protocol=='field_only':facts[protocol]['initial_max_matter_geometry_change']=max(map(abs,reactions[protocol]['initial_matter_geometry_changes'].values()))
        if name in result['controls']:
            baseline=reactions['baseline']['rho1'];off=result['controls'][name]['all_primitive_hops_off']['actual_density_history']
            facts['density_motion_controls']=dict(full_H0_max_change=max(abs(x-baseline[0]) for x in baseline),
                ALL_hops_off_max_change=max(abs(x-off[0]) for x in off),
                initial_site1_capture_overlap=reactions['source_capture_baseline']['1'][0],
                full_H0_site1_capture_range=[min(reactions['source_capture_baseline']['1']),max(reactions['source_capture_baseline']['1'])],
                all_hops_off_site1_capture_range=[min(result['controls'][name]['all_primitive_hops_off']['source_capture']['1']),max(result['controls'][name]['all_primitive_hops_off']['source_capture']['1'])],
                interpretation='Capture changes without quark hopping are not spatial B transport. Nonzero density movement with ALL hopping off is unphysical Galerkin local-Ward contamination. No stable moving-composite or gauge-complete EM claim can be made from these histories alone.')
        event_facts[name]=facts
    # Full raw source Green functions, including high/dark retained states, not a low spectral approximation.
    source_p=dict(p,eta=request['source_eta']);frequencies=request['source_frequencies']
    for name,keys in [('baryon',['0:0.5','0:1.5']),('antibaryon',['0:0.5','0:1.5']),('lepton',['0:0.5']),('antilepton',['0:0.5'])]:
        h=TensorOperator(sparse_matrix(operators,name+'_H0_one'),sparse_matrix(operators,name+'_H0_phi'),scalar,field_energy=True)
        raw_green[name]={}
        for key in keys:
            raw=states[name+'_source_state_'+key];values=[];diagnostics=[]
            for w in frequencies:
                solution,status=solve_resolvent(h,e0,raw,w,source_p);values.append(np.vdot(raw,solution));diagnostics.append(status)
            raw_green[name][key]=dict(frequencies=frequencies,eta=request['source_eta'],G_raw=cjson(values),solves=diagnostics,
                raw_strength=float(np.vdot(raw,raw).real),retained_virtual_dimension=h.shape[0],
                meaning='Unnormalized source Lehmann G=<S Omega|(w+Eneutral+i eta-Hcharged)^-1|S Omega>; EVERY retained eigenstate enters, not just the computed low source poles.')
    paired={};index={w:i for i,w in enumerate(frequencies)}
    for name,anti,keys in [('baryon','antibaryon',['0:0.5','0:1.5']),('lepton','antilepton',['0:0.5'])]:
        paired[name]={}
        for key in keys:
            addition=raw_green[name][key]['G_raw'];removal=raw_green[anti][key]['G_raw']
            a=np.array(addition['real'])+1j*np.array(addition['imag']);b=np.array(removal['real'])+1j*np.array(removal['imag'])
            paired[name][key]=dict(C_paired_signed_source_function=cjson([a[i]-b[index[-w]].conjugate() for i,w in enumerate(frequencies)]),
                raw_addition_plus_Cremoval_strength=raw_green[name][key]['raw_strength']+raw_green[anti][key]['raw_strength'],
                meaning='C-paired odd-source two-energy-orientation Lehmann combination, addition G(w)-conjugate(C-removal G(-w)). It is not normalized to a canonical elementary-B anticommutator and is not a full Dirac propagator. Actual neutral J0 rotation relates opposite magnetic spin components here.')
    provenance={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    payload=dict(kind='RAW_SPATIAL_SOURCE_AND_PHYSICAL_RECEIVER_READOUT_NOT_TRANSPORT_ADMISSION',producer_prefix=prefix,inputs=provenance,
        all_computed_low_raw_source_weights=low_weights,full_retained_raw_source_Greens=raw_green,C_paired_signed_source_functions=paired,
        actual_CAR_source_propagation=propagation,physical_protocol_readout=event_facts,
        resource=dict(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-started),
        limitations='Exact readout of frozen NEW graph H/state. Low-line weights disclose uncomputed high spectral weight; full raw source Green functions and time propagation retain all virtuals. Large source t0 overlap and hop-off effects forbid calling capture probability a moving particle. No model/state refit, old producer replay, source rescaling, quantum photon or Nature admission.')
    out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(dict(producer_prefix=prefix,readout=event_facts,resource=payload['resource']),indent=2))


if __name__=='__main__':main()
