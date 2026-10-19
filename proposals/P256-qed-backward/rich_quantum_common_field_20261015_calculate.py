"""NEW common-CAR quantum Wilson field consumer; exact evolution in disclosed projection."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply, spsolve
from rich_quantum_common_field_20261015_model import (
    build, rotor, expand_sector, vector_maps, whole_law, compressed, source_between,
    conjugate_sector, operators, law, conjugation, charge_conjugate, signatures,
    gauge_sector, complete_masks, source_matrix, omitted_vector_norm, energy_projection_residual)


def mean(state,op):
    return float(np.vdot(state,op@state).real)


def spectrum(h,k):
    e,v=eigsh(h,k=min(k,h.shape[0]-2),which='SA',tol=2e-10,maxiter=20000)
    order=np.argsort(e); return e[order],v[:,order]


def neutral_C(sector,scalar):
    nf=len(scalar['grid']); count=len(sector['blocks']); blocks=[[None]*count for _ in range(count)]
    for j,source in enumerate(sector['blocks']):
        target_signature=signatures(charge_conjugate(source['sector']['masks'][0])[0])
        for i,target in enumerate(sector['blocks']):
            if target['signature']!=target_signature: continue
            raw=conjugation(source['sector']['masks'],target['sector']['masks'])
            c=target['sector']['P'].conj().T@raw@source['sector']['P']
            blocks[i][j]=sparse.csr_matrix(target['V'].conj().T@(sparse.kron(c,sparse.eye(nf))@source['V']))
    for i,b in enumerate(sector['blocks']):
        if blocks[i][i] is None: blocks[i][i]=sparse.csr_matrix((b['dimension'],b['dimension']),dtype=complex)
    return sparse.bmat(blocks,format='csr')


def full_source_strength(neutral,kind,reference,scalar,field,loop):
    nf=len(scalar['grid']); targets=set()
    for block in neutral['blocks']:
        if kind=='lepton':
            delta=np.zeros(16,dtype=int);delta[9]=1; changes=[delta]
        else:
            changes=[]
            for spins in [(0,0,1),(0,1,0),(1,0,0)]:
                delta=np.zeros(16,dtype=int)
                for f,s in zip((0,0,1),spins):delta[(f*4+s)*2]+=1
                changes.append(delta)
        for delta in changes:
            target=tuple(np.array(block['signature'])+delta)
            if all(value <= (3 if i%2==0 else 1) for i,value in enumerate(target)):targets.add(target)
    total=0.
    for signature in sorted(targets):
        target=gauge_sector(complete_masks(signature),loop)
        if not target['dimension']:continue
        image=np.zeros((target['dimension']*nf,field['size']),complex)
        for block in neutral['blocks']:
            difference=np.array(signature)-np.array(block['signature'])
            if np.any(difference<0) or difference.sum()!=(1 if kind=='lepton' else 3):continue
            coefficient=reference[block['offset']*field['size']:(block['offset']+block['dimension'])*field['size']].reshape(block['dimension'],field['size'])
            source=source_matrix(block['sector'],target,kind)
            image+=sparse.kron(source,sparse.eye(nf))@(block['V']@coefficient)
        total+=np.linalg.norm(image)**2
    return float(total)


def physical_joint_receiver(sector,state,scalar,field):
    # Underlying C_loop and phi PVMs commute with each other and electric number.
    # Energy compression makes their joint effects a POVM, never a reaction-norm proxy.
    nf=len(scalar['grid']); size=len(field['electric']); probability=np.zeros((2,2,3,3))
    bins=[field['electric']<0,field['electric']==0,field['electric']>0]
    for block in sector['blocks']:
        coefficient=state[block['offset']*field['size']:(block['offset']+block['dimension'])*field['size']].reshape(block['dimension'],field['size'])
        full=(block['V']@coefficient).reshape(block['sector']['dimension'],nf,size,size)
        ec,vc=np.linalg.eigh(block['op']['C']); rotated=np.einsum('ab,bfxy->afxy',vc.conj().T,full)
        for a,b,c,d in np.ndindex(probability.shape):
            loopmask=(ec<1e-8) if a==0 else (ec>=1e-8)
            phimask=(scalar['grid']<0) if b==0 else (scalar['grid']>=0)
            selected=rotated[np.ix_(loopmask,phimask,bins[c],bins[d])]
            probability[a,b,c,d]+=np.linalg.norm(selected)**2
    return probability


def save_sparse(stored,prefix,matrix):
    matrix=matrix.tocsr()
    for key,value in dict(data=matrix.data,indices=matrix.indices,indptr=matrix.indptr,shape=matrix.shape).items():stored[prefix+'_'+key]=value


def main():
    inp,out=map(Path,sys.argv[1:3]);p=json.loads(inp.read_text());base=build(p);scalar=base['scalar'];field=rotor(p)
    results={};stored={};sectors={};spectra={};laws={};sources={};nf=len(scalar['grid'])
    for name in ['neutral','lepton','baryon']:
        sector=expand_sector(base['sectors'][name],base['loop'],scalar,p);sectors[name]=sector
        sector['vector']=vector_maps(sector,scalar);h,j,k=whole_law(sector,scalar,field,p);laws[name]=(h,j,k)
        e,v=spectrum(h,p['low_poles']);spectra[name]=(e,v)
        print(json.dumps(dict(stage='whole_H',sector=name,dimension=h.shape[0],blocks=len(sector['blocks']),layers=sector['layers'],ground=float(e[0]))),flush=True)
        save_sparse(stored,name+'_H',h)
        for axis in range(2):save_sparse(stored,name+'_J'+str(axis),j[axis]);save_sparse(stored,name+'_K'+str(axis),k[axis])
        stored[name+'_e']=e;stored[name+'_v']=v
        for index,block in enumerate(sector['blocks']):
            prefix=name+'_block'+str(index);stored[prefix+'_V']=block['V'];stored[prefix+'_signature']=block['signature'];stored[prefix+'_masks']=np.array(block['sector']['masks'],np.uint64)
            save_sparse(stored,prefix+'_P',block['sector']['P'])
    reference=spectra['neutral'][1][:,0];reference_energy=float(spectra['neutral'][0][0]); cn=neutral_C(sectors['neutral'],scalar)
    size=len(field['electric']);reflect=sparse.csr_matrix(np.fliplr(np.eye(field['size'])))
    cneutral=sparse.kron(cn,reflect,format='csr')
    results['neutral_C']=dict(unitarity=float(sparse.linalg.norm(cneutral.conj().T@cneutral-sparse.eye(len(reference)))),H=float(sparse.linalg.norm(laws['neutral'][0]@cneutral-cneutral@laws['neutral'][0])))
    for name in ['lepton','baryon']:
        source=sparse.kron(source_between(sectors['neutral'],sectors[name],name,scalar),field['identity'],format='csr');sources[name]=source
        anti=dict(sectors[name]);anti['blocks']=[];anti['anti']=True
        for block in sectors[name]['blocks']:
            ab=dict(block);ab['sector']=conjugate_sector(block['sector'],base['loop']);ab['op']=operators(ab['sector'],base['loop']);ab['H0']=law(ab['op'],scalar,p)[0];anti['blocks'].append(ab)
        anti['vector']=vector_maps(anti,scalar);ah,aj,ak=whole_law(anti,scalar,field,p);aname='anti'+name;sectors[aname]=anti;laws[aname]=(ah,aj,ak)
        ca=sparse.kron(sparse.eye(anti['dimension']),reflect,format='csr')
        e,v=spectra[name];spectra[aname]=(e,ca@v)
        sources[aname]=ca@source@cneutral.conj().T
        results[aname+'_C']=dict(H=float(sparse.linalg.norm(ah@ca-ca@laws[name][0])),J=[float(sparse.linalg.norm(aj[a]@ca+ca@laws[name][1][a])) for a in range(2)],K=[float(sparse.linalg.norm(ak[a]@ca-ca@laws[name][2][a])) for a in range(2)])
        save_sparse(stored,aname+'_H',ah)
    times=np.array(p['times'])
    # Fundamental unit shift is geometry, not an electron-harmonic approximation.
    u=sparse.diags(np.ones(size-1),-1,shape=(size,size),format='csr');ux=sparse.kron(u,sparse.eye(size),format='csr');uy=sparse.kron(sparse.eye(size),u,format='csr')
    kick_generator=(ux+ux.conj().T)/2+(uy-uy.conj().T)/(4j)
    for name in ['neutral','lepton','baryon','antilepton','antibaryon']:
        sector=sectors[name];h,j,k=laws[name];e,v=spectra[name];source=sources.get(name)
        if source is None:state=reference;source_result={}
        else:
            raw=source@reference;amplitudes=v.conj().T@raw;weights=abs(amplitudes)**2;bright=np.flatnonzero(weights>p['bright_threshold'])
            if not len(bright):raise RuntimeError('No computed low source-bright pole; increase disclosed low_poles, do not invent gap')
            first=bright[0];pole=abs(e-e[first])<1e-7;state=v[:,pole]@amplitudes[pole];state/=np.linalg.norm(state)
            if not name.startswith('anti'):
                strength=full_source_strength(sectors['neutral'],name,reference,scalar,field,base['loop'])
            else:strength=results[name[4:]]['source']['unprojected_CAR_strength']
            frequencies=np.array(p['source_frequencies']);green=[]
            for w in frequencies:
                solved=spsolve((w+reference_energy+1j*p['eta'])*sparse.eye(h.shape[0])-h,raw)
                green.append(np.vdot(raw,solved))
            source_result=dict(retained_raw_strength=float(np.linalg.norm(raw)**2),unprojected_CAR_strength=strength,omitted_source_norm=float(np.sqrt(max(0,strength-np.linalg.norm(raw)**2))),raw_factor=(.5 if 'lepton' in name else .125),all_retained_virtuals=h.shape[0],source_low_bright_gap=float(e[first]-reference_energy),low_pole_raw_weights=weights.tolist(),low_pole_gaps=(e-reference_energy).tolist(),outside_low_pole_window_strength=float(np.linalg.norm(raw)**2-weights.sum()),full_projected_resolvent_real=np.real(green).tolist(),full_projected_resolvent_imag=np.imag(green).tolist(),frequencies=frequencies.tolist())
            stored[name+'_raw_source']=raw;save_sparse(stored,name+'_source',source)
        estate=mean(state,h);initial=state.reshape(sector['dimension'],field['size'])
        kicked=expm_multiply(-1j*p['field_kick']*kick_generator,initial.T).T.ravel()
        obs={key:sparse.kron(compressed(sector,key,scalar),field['identity'],format='csr') for key in ['phi','phi2','C','B','positive','S2','Q','N','boundary','scalar_boundary']}
        obs.update(E_x=sparse.kron(sparse.eye(sector['dimension']),field['E'][0]),E_y=sparse.kron(sparse.eye(sector['dimension']),field['E'][1]),electric_energy=sparse.kron(sparse.eye(sector['dimension']),field['H']),electric_boundary=sparse.kron(sparse.eye(sector['dimension']),field['boundary']),J_x=j[0],J_y=j[1],K_x=k[0],K_y=k[1],wilson_x=sparse.kron(sparse.eye(sector['dimension']),(ux+ux.conj().T)/2),wilson_y=sparse.kron(sparse.eye(sector['dimension']),(uy+uy.conj().T)/2))
        baseline=[];reaction=[];pvm=[];pvm_base=[];evolved=[]
        for t in times:
            baseline_state=expm_multiply(-1j*t*(h-estate*sparse.eye(h.shape[0])),state)
            pulse=expm_multiply(-1j*t*(h-estate*sparse.eye(h.shape[0])),kicked);evolved.append(pulse)
            baseline.append({key:mean(baseline_state,op) for key,op in obs.items()});reaction.append({key:mean(pulse,op) for key,op in obs.items()})
            pvm.append(physical_joint_receiver(sector,pulse,scalar,field));pvm_base.append(physical_joint_receiver(sector,baseline_state,scalar,field))
        excess={key:[r[key]-b[key] for r,b in zip(reaction,baseline)] for key in obs}
        field_bright={key:(abs(v.conj().T@(op@state))**2).tolist() for key,op in obs.items() if key in ['E_x','E_y','wilson_x','wilson_y','J_x','J_y']}
        # Do not inherit linear-response closure zero: evaluate actual whole-H residual at initial and final pulse states.
        residuals=[]
        for label,physical in [('stationary',state),('field_only_t0',kicked),('field_only_final',evolved[-1])]:
            residuals.append(dict(state=label,energy_projection=energy_projection_residual(sector,physical,scalar,field,p),population=omitted_vector_norm(sector,physical,scalar,field,base['loop'],p)))
        results[name]=dict(dimension=h.shape[0],population_layers=sector['layers'],retained_gauge_blocks=len(sector['blocks']),omitted_signature_count=len(sector['omitted']),underlying_retained_population_dimension=sum(b['H0'].shape[0] for b in sector['blocks']),ground_energy=float(e[0]),source=source_result,reference_energy=reference_energy,stationary_energy=estate,low_gaps=(e-estate).tolist(),low_field_current_residues=field_bright,times=times.tolist(),baseline=baseline,reaction=reaction,excess=excess,joint_receiver=np.array(pvm).tolist(),joint_baseline=np.array(pvm_base).tolist(),joint_normalization_error=float(np.max(abs(np.array(pvm).sum(axis=(1,2,3,4))-1))),kick_unitarity_error=float(abs(np.linalg.norm(kicked)-1)),time_norm_error=float(max(abs(np.linalg.norm(x)-1) for x in evolved)),eigen_residual=float(np.linalg.norm(h@state-estate*state)),hermiticity=float(sparse.linalg.norm(h-h.conj().T)),max_colour_Gauss=float(max(b['sector']['gauge_error'] for b in sector['blocks'])),projection_residuals=residuals,physical_source_overlap=[float(abs(np.vdot(state,x))**2) for x in evolved])
        stored[name+'_stationary']=state;stored[name+'_kick']=kicked;stored[name+'_evolved']=np.array(evolved);stored[name+'_joint_receiver']=np.array(pvm)
        print(json.dumps(dict(stage='field_event',sector=name,gap=source_result.get('source_low_bright_gap'),current_max=max(abs(x) for x in excess['J_x']),phi2_max=max(abs(x) for x in excess['phi2']),loop_max=max(abs(x) for x in excess['C']),projection=residuals)),flush=True)
    stored['electric_numbers']=field['electric'];stored['scalar_grid']=scalar['grid'];stored['metric']=field['metric']
    output=dict(kind='NEW_NONLINEAR_QUANTUM_COMMON_CAR_FIELD_CONDITIONAL_PROJECTION_NOT_PHOTON_OR_QED',parameters=p,electric_metric=field['metric'].tolist(),results=results,boundaries='Compact field quantum degrees, not travelling photons. No supplied magnetic restoring term. Population depth2 and low-energy Galerkin strongly cut nonlinear H; source/whole-H residuals explicit, no full pair-event/continuum/COM/physical Noether claim. Grounded radial boundary and chosen tube profiles/Clifford paths/rank3+1/Q/CAR are inputs. All retained virtuals participate in exact projected sparse resolvents and time evolution. Joint receiver is compressed physical commuting PVM, hence inclusive POVM; source overlap is separately labelled, never receiver probability.')
    out.write_text(json.dumps(output,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**stored)


if __name__=='__main__':main()
