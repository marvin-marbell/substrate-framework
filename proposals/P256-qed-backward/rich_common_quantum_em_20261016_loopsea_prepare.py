"""Material fork: actual two-hop occupied-sea virtual circuits, no magnetic term.
Reuse ONLY an existing CAR/PW/R isometry and unchanged primitive matrix elements.
NOT its solved states/energies, NOT a flux-cap or parameter sweep.
"""
from pathlib import Path
import json,sys,time,resource
import numpy as np
from scipy import sparse
import rich_composite_geometry_20261015_model as model
from rich_composite_geometry_20261015_factors import load_h
from rich_common_quantum_em_20261016_prepare import load,save,neutral_blocks,body_metric,metric_powers,chord_operators
from rich_common_quantum_em_20261016_geometry import construct


def legacy_scalar_factors(matrix,scalar,counts):
    """Exact body x Phi extraction of the observed legacy CSR format."""
    grid=np.asarray(scalar['grid']);hphi=np.asarray(scalar['H']);nf=len(grid);n=len(counts);eye=sparse.eye(n,format='csr')
    if matrix.shape!=(n*nf,n*nf):raise ValueError('Legacy CSR tensor dimensions do not match actual isometry')
    center=int(np.argmin(abs(grid)));endpoint=int(np.argmax(abs(grid)));middle=matrix[center::nf,center::nf].tocsr();outer=matrix[endpoint::nf,endpoint::nf].tocsr()
    cphi=(outer-middle-(hphi[endpoint,endpoint]-hphi[center,center])*eye)/(grid[endpoint]-grid[center]);coo=cphi.tocoo();same=np.all(counts[coo.row]==counts[coo.col],axis=1);roundoff=float(np.linalg.norm(coo.data[same]))
    if roundoff>1e-10:raise ValueError('Legacy scalar factor has a genuine count-preserving term, inconsistent with declared intersite scalar law')
    cphi=sparse.coo_matrix((coo.data[~same],(coo.row[~same],coo.col[~same])),shape=(n,n)).tocsr();c0=middle-hphi[center,center]*eye-grid[center]*cphi;error2=0.
    for a,b in np.ndindex(nf,nf):
        expected=hphi[a,b]*eye
        if a==b:expected=expected+c0+grid[a]*cphi
        error=matrix[a::nf,b::nf]-expected;error2+=float(np.vdot(error.data,error.data).real)
    receipt={'format':'observed legacy explicit CSR(body tensor Phi), NOT a factored-file physics fallback','all_Phi_points':nf,'body_dimension':n,'full_CSR_dimension':matrix.shape[0],'all_49_scalar_blocks_reconstruction_Frobenius':float(np.sqrt(error2)),'analytically_zero_count_preserving_scalar_roundoff_removed':roundoff,'no_state_energy_or_virtual_cut_used':True}
    if receipt['all_49_scalar_blocks_reconstruction_Frobenius']>1e-9:raise ValueError('Legacy CSR does not have the exact declared scalar tensor form')
    return c0.tocsr(),cphi,receipt


def main():
    out=sys.argv[1];start=time.time();data=load('rich_composite_geometry_20261015_correlated_neutral1_neutral.pkl');coupling=load('rich_composite_geometry_20261015_correlated_reference1_J0_coupling.pkl');old=load_h('rich_composite_geometry_20261015_correlated_reference1_J0_H0.npz');primitive=data['primitive'];labels=coupling['labels'];parameters=primitive.p;previous=json.loads(Path('rich_common_quantum_em_20261016_neutral_prepare2_body.json').read_text())['parameters'];physical_keys=('geometry','scalar_points','scalar_extent','scalar_metric','scalar_quartic','scalar_quadratic','scalar_coupling','electric','plaquette','vector_velocity','orientation_azimuth_points','orientation_polar_points','scalar_volume_ratio','vertex_mass_density')
    if any(parameters[key]!=previous[key]for key in physical_keys):raise ValueError('A primitive physical parameter changed; not the declared representation-only mechanism fork')
    storage=neutral_blocks(data,coupling);save(out+'_neutral_raw_blocks.pkl',{'blocks':storage,'labels':labels});counts=np.asarray([row['counts']for row in labels]);scalar,rotor=model.field_and_rotor(parameters);c0,scalar_hop,factorization=legacy_scalar_factors(old,scalar,counts);del old;Path(out+'_legacy_factorization.json').write_text(json.dumps(factorization,indent=2)+'\n');coo=c0.tocoo();same=np.all(counts[coo.row]==counts[coo.col],axis=1);static=sparse.coo_matrix((coo.data[same],(coo.row[same],coo.col[same])),shape=coo.shape).tocsr();vector=c0-static;gram=body_metric(storage,len(labels));half,inverse,metric=metric_powers(gram,labels)
    print(json.dumps({'stage':'closed_occupied_sea_loop_representation_fork','body_dimension':len(labels),'count_blocks':len(storage),'all_primitive_physical_parameters_unchanged':True,'old_state_files_opened':False}),flush=True)
    chord,receipt=chord_operators(primitive,storage,labels,rotor,out)
    for name,value in{'static':static,'vector':vector,'scalar_hop':scalar_hop,'Gram':gram,'half':half,'inversehalf':inverse}.items():sparse.save_npz(out+'_'+name+'.npz',value)
    np.savez_compressed(out+'_scalar.npz',H=scalar['H'],grid=scalar['grid']);Path(out+'_geometry.json').write_text(json.dumps(construct(),indent=2)+'\n');Path(out+'_body.json').write_text(json.dumps({'mode':'neutral_closed_occupied_sea_loop_fork','labels':[{'counts':list(row['counts']),'twice_J':0}for row in labels],'parameters':parameters,'metric':metric,'chord':receipt,'mechanism_change':'Actual two-hop occupied CAR/sea cloud permits intermediate states of closed four-step currents absent from the first one-hop reference. Same primitive H/charge geometry/field cutoff; no added ring-potential term, no fitted coefficient, and NO assumption that historical source-selected spaces are nested.','physical_parameters_equal_first_model':list(physical_keys),'old_solved_state_files_opened':False,'old_energy_values_used':False,'basis_limitation':'Still a severe finite selected occupied-sea CAR/PW/R/Phi law; NOT full primitive phase or particle proof.'},indent=2)+'\n');print(json.dumps({'stage':'closed_occupied_sea_loop_ingredients_complete','seconds':time.time()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)


if __name__=='__main__':main()
