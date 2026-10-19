"""Same-law block-raw contraction and disk-backed native operator compiler.
The full microscopic action precedes projection; never a current commutator patch.
"""
from __future__ import annotations
from pathlib import Path
import json,pickle,resource,sys,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_compile import save
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_fierz import electric
from rich_composite_geometry_20261015_operators import OperatorStore
from rich_composite_geometry_20261015_metrics import isometry
from rich_composite_geometry_20261015_factors import reduce_factors,finite_matrix_control




def operator_matrix(basis,action,full_norm=False):
    if not hasattr(basis,'storage'):
        if full_norm:
            norm=sum(float(model.inner(v,v).real) for p in basis.vectors for v in [action(p)])
        else:norm=None
        return basis.matrix(action),norm
    rows=[];columns=[];values=[];norm2=0.
    for lab,source in basis.storage.items():
        rank=source.rank
        if not rank:continue
        ns=len(source.keys);q=source.q[:ns,:rank];entries={};raw_entries=[];raw_keys={}
        for column,key in enumerate(source.keys):
            if np.max(abs(q[column]),initial=0)<2e-14:continue
            image=action({key:1.+0j})
            if full_norm:
                for target,value in image.items():
                    row=raw_keys.setdefault(target,len(raw_keys));raw_entries.append((row,column,value))
            for target_lab,part in basis.split(image).items():
                target_block=basis.storage.get(target_lab)
                if target_block is None:continue
                target_list=entries.setdefault(target_lab,[])
                for target,value in part.items():
                    row=target_block.lookup.get(target)
                    if row is not None:target_list.append((row,column,value))
        if full_norm and raw_entries:
            r,c,v=zip(*raw_entries);t=sparse.csr_matrix((v,(r,c)),shape=(len(raw_keys),ns));norm2+=float(np.linalg.norm(t@q)**2)
        for target_lab,triple in entries.items():
            if not triple:continue
            target=basis.storage[target_lab];r,c,v=zip(*triple)
            t=sparse.csr_matrix((v,(r,c)),shape=(len(target.keys),ns))
            native=target.q[:len(target.keys),:target.rank].conj().T@(t@q)
            rr,cc=np.nonzero(abs(native)>2e-14)
            rows.append(np.asarray(target.columns,dtype=np.int64)[rr]);columns.append(np.asarray(source.columns,dtype=np.int64)[cc]);values.append(native[rr,cc])
    if rows:
        matrix=sparse.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),shape=(len(basis.vectors),)*2)
    else:matrix=sparse.csr_matrix((len(basis.vectors),)*2,dtype=complex)
    return matrix,norm2 if full_norm else None


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    with open(config['basis_snapshot'],'rb') as f:data=pickle.load(f)
    basis=data['basis'];p=data['p'];primitive=primitive_for(data);scalar,rotor=model.field_and_rotor(p)
    multiplets,spin=angular.spin_multiplets(basis,primitive)
    save(out+'_spin.pkl',{'p':p,'multiplets':multiplets,'spin_receipt':spin})
    Path(out+'_spin.json').write_text(json.dumps(spin,indent=2)+'\n')
    controls=[]
    for index in np.unique(np.linspace(0,len(basis.vectors)-1,12,dtype=int)):
        packet=basis.vectors[index];a=primitive.electric(packet);b=electric(primitive,packet)
        difference=model.add((1,a),(-1,b));controls.append({'basis_index':int(index),'full_raw_action_error':float(np.sqrt(model.inner(difference,difference).real))})
    actions={'scalar':lambda v:primitive.collective(v),'electric':lambda v:electric(primitive,v),
        'B':lambda v:primitive.field(v,primitive.field_maps['B']),'C':lambda v:primitive.field(v,primitive.field_maps['C'])}
    for i,a in np.ndindex(3,3):actions[f'vector_{i}_{a}']=lambda v,i=i,a=a:primitive.collective(v,i+1,a)
    hermitian={}
    for name,action in actions.items():
        matrix,_=operator_matrix(basis,action);hermitian[name]=float(sparse.linalg.norm(matrix-matrix.conj().T))
        sparse.save_npz(out+'_op_'+name+'.npz',matrix)
        print(json.dumps({'stage':'full_raw_block_original_operator','operator':name,'dimension':matrix.shape[0],'nnz':matrix.nnz}),flush=True)
        del matrix
    ops=OperatorStore(out,list(actions));save(out+'_operators.pkl',ops)
    gauss=[]
    for a in range(8):
        matrix,norm2=operator_matrix(basis,lambda v,a=a:primitive.gauss(v,a),True)
        gauss.append({'a':a,'full_raw_G_isometry_Frobenius':float(np.sqrt(norm2)),'projected_G_norm':float(sparse.linalg.norm(matrix))})
        del matrix
    receipt={'kind':'ACTUAL_MEMORY_BOUNDED_SAME_LAW_OCCUPIED_NOETHER_COMPILER','parameters':p,'basis':basis.receipt(),'spin':spin,
        'electric_Fierz_full_raw_controls':controls,'Fierz_identity':'sum t_ij t_kl=(delta_il delta_jk-delta_ij delta_kl/3)/2; exact full quartics and Q-R cross BEFORE target projection, no changed H.',
        'colour_Gauss':gauss,'operator_Hermiticity':hermitian,'blocks':{},
        'actual_physical_isometry':isometry(basis,primitive),
        'storage':'Sparse raw action once per coordinate/small physical-count block, Qtargetdag Oraw Qsource; native operator NPZ files disk-backed cache1. CSC spin columns, not quadratic-size CSR-column pointer duplication. Same finite family, no state/cut/coupling fallback.'}
    graph=primitive.graph
    for j2 in config['twice_J']:
        coupling=angular.coupled(basis,multiplets,j2);save(out+f'_J{j2}_coupling.pkl',coupling)
        del coupling
    del basis,data,multiplets,primitive,actions
    for j2 in config['twice_J']:
        with open(out+f'_J{j2}_coupling.pkl','rb') as f:coupling=pickle.load(f)
        h=reduce_factors(ops,coupling,rotor,scalar,p,graph);h.save(out+f'_J{j2}_H0.npz')
        count=min(p['low_eigenstates'],h.shape[0]-2);rng=np.random.default_rng(1988+j2)
        e,v=eigsh(h,k=count,which='SA',tol=p['eigen_tolerance'],v0=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]))
        order=np.argsort(e);e=e[order];v=v[:,order]
        residuals=[float(np.linalg.norm(h@v[:,i]-e[i]*v[:,i])) for i in range(count)]
        np.savez(out+f'_J{j2}_states.npz',energies=e,states=v)
        receipt['blocks'][str(j2)]={'coupled_dimension':coupling['dimension'],'full_phi_dimension':h.shape[0],
            'isometry_error':coupling['isometry_error'],'H_Hermiticity_error':h.hermiticity_error(),
            'energies':e.tolist(),'eigen_residuals':residuals}
        receipt['blocks'][str(j2)]['factor_formula_control']=finite_matrix_control(h)
        Path(out+'_checkpoint.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'stage':'actual_new_pole_block','twice_J':j2,'energy':float(e[0]),'maxres':max(residuals)}),flush=True)
        del h,v
    receipt['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['resource']),flush=True)


if __name__=='__main__':main()
