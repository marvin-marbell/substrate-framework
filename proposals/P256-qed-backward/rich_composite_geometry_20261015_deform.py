"""NEW stretched H on the SAME isometry: exact hop-support reweighting,
actual new full electric action/inertia, own eigensolve. NEVER old energies.
"""
from pathlib import Path
import json,pickle,resource,sys,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import rich_composite_geometry_20261015_model as model
import rich_composite_geometry_20261015_angular as angular
from rich_composite_geometry_20261015_runtime import primitive_for
from rich_composite_geometry_20261015_stream import operator_matrix
from rich_composite_geometry_20261015_operators import OperatorStore
from rich_composite_geometry_20261015_fierz import electric
from rich_composite_geometry_20261015_factors import reduce_factors,finite_matrix_control


def save(path,value):
    with open(path,'xb') as f:pickle.dump(value,f,protocol=5)


def scaled_hop(matrix,counts,factors):
    coo=matrix.tocoo();data=coo.data.copy();edges={tuple(sorted(e)):i for i,e in enumerate(model.EDGES)}
    for start in range(0,len(data),65536):
        stop=min(start+65536,len(data));delta=counts[coo.row[start:stop]]-counts[coo.col[start:stop]]
        positive=np.argmax(delta,axis=1);negative=np.argmin(delta,axis=1)
        valid=(np.count_nonzero(delta,axis=1)==2)&(delta.max(axis=1)==1)&(delta.min(axis=1)==-1)&(positive%4==negative%4)
        if not np.all(valid):raise ArithmeticError('Original hop matrix has forbidden actual physical-count support')
        classified=np.zeros(stop-start,bool)
        for first,second in edges:
            e=edges[first,second];mask=((positive//4==first)&(negative//4==second))|((positive//4==second)&(negative//4==first))
            classified|=mask
            data[start:stop][mask]*=factors[e]
        if not np.all(classified):raise ArithmeticError('Original hopping support includes a nonedge; do not silently reweight a different law')
    out=sparse.csr_matrix((data,(coo.row,coo.col)),shape=matrix.shape);out.eliminate_zeros();return out


def main():
    config=json.loads(Path(sys.argv[1]).read_text());out=sys.argv[2];started=time.time()
    with open(config['basis_snapshot'],'rb') as f:data=pickle.load(f)
    with open(config['base_operators_snapshot'],'rb') as f:base=pickle.load(f)
    with open(config['spin_snapshot'],'rb') as f:spin=pickle.load(f)
    before=json.loads(Path(config['base_capture_before_snapshot']).read_text())
    origin=data['family_origin'];p=data['p'];old=spin['p']
    if origin['sha256']!=before[origin['file']]:raise ValueError('Changed actual isometry ancestor')
    if any(p[k]!=v for k,v in old.items() if k!='geometry'):raise ValueError('Geometry continuation changed another physical/family input')
    basis=data['basis'];primitive=primitive_for(data);scalar,rotor=model.field_and_rotor(p);oldgraph=model.geometry(old)
    counts=np.asarray([lab[:16] for lab in basis.labels],dtype=np.int8);names=list(base)
    hermitian={};controls=[]
    for name in ('B','C'):
        matrix=base[name];sparse.save_npz(out+'_op_'+name+'.npz',matrix);hermitian[name]=float(sparse.linalg.norm(matrix-matrix.conj().T))
    ratio=primitive.graph['weights']/oldgraph['weights'];matrix=scaled_hop(base['scalar'],counts,ratio)
    sparse.save_npz(out+'_op_scalar.npz',matrix);hermitian['scalar']=float(sparse.linalg.norm(matrix-matrix.conj().T));del matrix
    for i,a in np.ndindex(3,3):
        matrix=sparse.csr_matrix((len(basis.vectors),)*2,dtype=complex)
        for b in range(3):
            factors=ratio*primitive.graph['axes'][:,a]*oldgraph['axes'][:,b]
            matrix+=scaled_hop(base[f'vector_{i}_{b}'],counts,factors)
        sparse.save_npz(out+f'_op_vector_{i}_{a}.npz',matrix);hermitian[f'vector_{i}_{a}']=float(sparse.linalg.norm(matrix-matrix.conj().T));del matrix
    matrix,_=operator_matrix(basis,lambda v:electric(primitive,v));sparse.save_npz(out+'_op_electric.npz',matrix)
    hermitian['electric']=float(sparse.linalg.norm(matrix-matrix.conj().T));del matrix
    ops=OperatorStore(out,names);save(out+'_operators.pkl',ops)
    for index in np.unique(np.linspace(0,len(basis.vectors)-1,12,dtype=int)):
        packet=basis.vectors[index]
        for name,action in [('scalar',lambda v:primitive.collective(v)),
                ('vector_0_0',lambda v:primitive.collective(v,1,0)),('vector_1_2',lambda v:primitive.collective(v,2,2)),('electric',primitive.electric)]:
            coordinates=basis.coordinates(action(packet));column=ops[name][:,index].tocoo();predicted=dict(zip(column.row,column.data))
            error=np.sqrt(sum(abs(coordinates.get(k,0.)-predicted.get(k,0.))**2 for k in set(coordinates)|set(predicted)))
            controls.append({'basis_index':int(index),'operator':name,'new_original_raw_then_native_action_error':float(error)})
    receipt={'kind':'ACTUAL_FIXED_ISOMETRY_NEW_STRETCHED_OCCUPIED_HAMILTONIAN','parameters':p,'family_origin':origin,
        'hop_reweighting':'Exact ORIGINAL primitive count-homogeneous support decomposition, not J=i[H,Q]. Each nonzero hop entry changes exactly one same-channel count on one graph edge; scalarw and nine vectorw*n coefficients reweighted from actual new geometry. New full electric action and actual inertia rebuilt. No fitted term or inherited energies.',
        'independent_new_raw_controls':controls,'operator_Hermiticity':hermitian,
        'spin_reuse':'Same physical isometry, SAME spin/PW/SO3 representation and allM. Spin/Gauss/isometry facts independent of metric; cached actual multiplets from matched ancestor, NOT reference/eigenstates or Frame19 spin transfer.',
        'blocks':{}}
    graph=primitive.graph
    for j2 in config['twice_J']:
        coupling=angular.coupled(basis,spin['multiplets'],j2);save(out+f'_J{j2}_coupling.pkl',coupling);del coupling
    del basis,data,primitive,base,spin,counts
    for j2 in config['twice_J']:
        with open(out+f'_J{j2}_coupling.pkl','rb') as f:coupling=pickle.load(f)
        h=reduce_factors(ops,coupling,rotor,scalar,p,graph);h.save(out+f'_J{j2}_H0.npz')
        rng=np.random.default_rng(4889+j2);n=min(p['low_eigenstates'],h.shape[0]-2)
        energies,states=eigsh(h,k=n,which='SA',tol=p['eigen_tolerance'],v0=rng.normal(size=h.shape[0])+1j*rng.normal(size=h.shape[0]))
        order=np.argsort(energies);energies=energies[order];states=states[:,order]
        residual=[float(np.linalg.norm(h@states[:,i]-energies[i]*states[:,i])) for i in range(n)]
        np.savez(out+f'_J{j2}_states.npz',energies=energies,states=states)
        receipt['blocks'][str(j2)]={'coupled_dimension':coupling['dimension'],'full_phi_dimension':h.shape[0],'isometry_error':coupling['isometry_error'],
            'energies':energies.tolist(),'eigen_residuals':residual,'factor_control':finite_matrix_control(h)}
        Path(out+'_checkpoint.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'stage':'new_stretched_pole','twice_J':j2,'energy':float(energies[0])}),flush=True)
        del h,states
    receipt['resource']={'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.time()-started}
    Path(out+'_result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['resource']),flush=True)


if __name__=='__main__':main()
