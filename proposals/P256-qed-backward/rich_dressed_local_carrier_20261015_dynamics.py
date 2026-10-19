"""Own full-H dynamics; independent reviewer selects different spin/time/controls."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from rich_dressed_local_carrier_20261015_source import Model, lanczos, energy, expectation
from rich_dressed_local_carrier_20261015_prepare import primitive

ROOT = Path(__file__).resolve().parent
P = 'rich_dressed_local_carrier_20261015_'
TIMES = [0.0, 0.4, 0.8]


def sparse_hash(h):
    digest=hashlib.sha256()
    for array in [np.asarray(h.shape,np.int64),h.indptr,h.indices,h.data]:
        digest.update(array.tobytes())
    return digest.hexdigest()


def propagate(h, initial, times=TIMES, dimension=40, coarse=28):
    basis, diagonal, offdiagonal, residual = lanczos(h,initial,dimension)
    values, rotations = eigh_tridiagonal(diagonal,offdiagonal)
    coarse = min(coarse,len(diagonal))
    cv,cr = eigh_tridiagonal(diagonal[:coarse],offdiagonal[:coarse-1])
    states, differences = [], []
    for t in times:
        state = (rotations@(np.exp(-1j*t*values)*rotations[0]))@basis
        alternate = (cr@(np.exp(-1j*t*cv)*cr[0]))@basis[:coarse]
        states.append(state)
        differences.append(float(np.linalg.norm(state-alternate)))
    return states, dict(full_space_H_action=True,dimension=len(diagonal),coarse=coarse,
        times=times,same_state_norm_differences=differences,last_Lanczos_residual=residual,
        norm_errors=[float(abs(np.vdot(s,s)-1)) for s in states])


def measure(model,h,currents,state):
    result = primitive(model,state)
    result.update(energy(h,state))
    result['current']=[expectation(state,j) for j in currents]
    result['electric_power']=[-float(np.vdot(e@state,j@state).real)/model.parameters['electric_inertia'] for e,j in zip(model.E,currents)]
    result['local_count_PVM']=[]
    result['raw_joint_count3_or1_r2_Sz_E']=[]
    for v in range(3):
        countweights=[]
        for k in range(5):
            projected=(model.sector.density[:,v]==k)[:,None,None,None,None]*state.reshape(model.shape)
            countweights.append(float(np.vdot(projected,projected).real))
        result['local_count_PVM'].append(countweights)
        count=3 if model.sector.number==7 else 1
        selected=((model.sector.density[:,v]==count)[:,None,None,None,None]*state.reshape(model.shape)).ravel()
        result['raw_joint_count3_or1_r2_Sz_E'].append(dict(count=count,
            weight=float(np.vdot(selected,selected).real),r2=expectation(selected,model.r2_local[v]),
            Sz=expectation(selected,model.S_local[v][2]),E=[expectation(selected,e) for e in model.E]))
    return result


def source_resolution(model,reference,neutral,state,branch,h,currents):
    """Exact finite bare-source span; complement NOT asserted to be radiation."""
    vectors=[]
    labels=[]
    for site in range(3):
        for orbital in range(2):
            for component in range(2):
                vectors.append(model.source_from(reference,site,branch,component,orbital)@neutral)
                labels.append([site,orbital,component])
    gram=np.array([[np.vdot(a,b) for b in vectors] for a in vectors])
    overlaps=np.array([np.vdot(a,state) for a in vectors])
    values,rotation=np.linalg.eigh(gram)
    keep=values>1e-10
    coefficients=rotation[:,keep]@((rotation[:,keep].conj().T@overlaps)/values[keep])
    coherent=sum(c*v for c,v in zip(coefficients,vectors))
    other=state-coherent
    operators=dict(energy=h,**{'current'+str(e):j for e,j in enumerate(currents)},
        **{'r2_'+str(v):r for v,r in enumerate(model.r2_local)})
    split={}
    for name,operator in operators.items():
        split[name]=dict(total=expectation(state,operator),coherent_raw=expectation(coherent,operator),
            orthogonal_raw=expectation(other,operator),interference=2*float(np.vdot(coherent,operator@other).real))
    return dict(labels=labels,gram_eigenvalues=values.tolist(),rank=int(keep.sum()),
        bare_source_overlap_complex=[[float(z.real),float(z.imag)] for z in overlaps],
        coherent_source_span_weight=float(np.vdot(coherent,coherent).real),
        orthogonal_weight=float(np.vdot(other,other).real),orthogonality=float(abs(np.vdot(coherent,other))),
        decomposition=split,meaning='Unnormalized P_sourcePsi and (1-P_source)Psi with all cross terms. Orthogonal part contains dressing/other excitations; finite experiment does not identify a radiated-only channel.')


def main():
    interface=json.loads((ROOT/(P+'interface.json')).read_text())
    with np.load(ROOT/(P+'states.npz')) as f:
        neutral=f['neutral'].copy()
        inputs={b:f[b+'_component0'].copy() for b in ['addition','removal']}
    reference=Model(6)
    h, currents,_=reference.law()
    neutral_states,neutral_accuracy=propagate(h,neutral)
    neutral_measurements=[measure(reference,h,currents,s) for s in neutral_states]
    result=dict(kind='EXECUTED_FULL_H_DRESSED_CARRIER_NOT_SOURCE_SUBSPACE_EVOLUTION',times=TIMES,
        input_SHA256=interface['state_SHA256'],neutral_accuracy=neutral_accuracy,
        neutral_H_CSR_SHA256=sparse_hash(h),
        neutral=neutral_measurements,branches={},field_intervention={})
    del h,currents
    gc.collect()
    saved={f'neutral_t{i}':s for i,s in enumerate(neutral_states)}
    for branch,number in [('addition',7),('removal',5)]:
        model=Model(number)
        h,currents,_=model.law()
        h_hash=sparse_hash(h)
        initial=inputs[branch]
        states,accuracy=propagate(h,initial)
        measurements=[measure(model,h,currents,s) for s in states]
        resolved=[source_resolution(model,reference,n,s,branch,h,currents) for n,s in zip(neutral_states,states)]
        result['branches'][branch]=dict(accuracy=accuracy,absolute=measurements,source_resolution=resolved,
            H_CSR_SHA256=h_hash,
            excess=[{key:(np.asarray(m[key])-np.asarray(n[key])).tolist() for key in ['n','S','material_L','J','local_r2','E','current','electric_power']} for m,n in zip(measurements,neutral_measurements)],
            survival_amplitude_squared=[float(abs(np.vdot(initial,s))**2) for s in states])
        saved.update({branch+'_t'+str(i):s for i,s in enumerate(states)})
        if branch=='addition':
            phase=.31
            electric=np.repeat(model.sector.electric[:,0],int(np.prod(model.shape[1:])))
            perturbed=np.exp(1j*phase*electric)*initial
            field_states,field_accuracy=propagate(h,perturbed)
            field_measurements=[measure(model,h,currents,s) for s in field_states]
            result['field_intervention']=dict(operator='exp(+i .31 E0)',field_only=True,
                accuracy=field_accuracy,absolute=field_measurements,
                same_state_full_response=[{key:(np.asarray(m[key])-np.asarray(n[key])).tolist() for key in ['n','S','local_r2','E','current','electric_power']} for m,n in zip(field_measurements,measurements)],
                work=energy(h,perturbed)['energy']-energy(h,initial)['energy'])
            saved.update({'addition_field_t'+str(i):s for i,s in enumerate(field_states)})
            del field_states,field_measurements,perturbed,electric
        (ROOT/(P+'dynamics_partial.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(stage=branch,final_excess=result['branches'][branch]['excess'][-1],coherent_weights=[r['coherent_source_span_weight'] for r in resolved],accuracy=accuracy)),flush=True)
        del model,h,currents,states,measurements,resolved
        gc.collect()
    np.savez_compressed(ROOT/(P+'dynamics_states.npz'),**saved)
    (ROOT/(P+'dynamics_result.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
