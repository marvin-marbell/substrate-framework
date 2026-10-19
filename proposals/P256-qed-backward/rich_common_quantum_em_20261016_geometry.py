"""NEW two-coordinate spatial charge connection, symplectic metric and periods.
No magnetic Hamiltonian. Angles live on the declared cover, not independent links.
"""
from itertools import product,combinations
from math import gcd
import json
import numpy as np

MATTER_EDGES=((0,1),(1,2),(2,3),(0,3),(0,2))


def embedding_index(matrix):
    """Index of integer column lattice in its saturation, via maximal minors."""
    rank=matrix.shape[1];value=0
    for rows in combinations(range(len(matrix)),rank):
        value=gcd(value,abs(int(round(np.linalg.det(matrix[list(rows)])))))
        if value==1:break
    return value


def construct(maximum=6):
    positions=np.array([(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)],float)/np.sqrt(8)
    length=np.sqrt(2);q=np.array([2*np.pi/length,0,0]);links=[]
    for start,end in combinations(range(4),2):
        for image in product((-1,0,1),repeat=3):
            displacement=positions[end]+length*np.array(image)-positions[start]
            if abs(np.linalg.norm(displacement)-1)>1e-12:continue
            midpoint=positions[start]+displacement/2
            profile=np.sqrt(2)*displacement[1:]*np.sin(q@midpoint)
            links.append({'start':start,'end':end,'image':list(image),'displacement':displacement.tolist(),'midpoint':midpoint.tolist(),'unrounded_profile':profile.tolist()})
    d=np.zeros((4,len(links)),int);t=np.zeros((len(links),2),int)
    for e,link in enumerate(links):
        d[link['start'],e]=-1;d[link['end'],e]=1;t[e]=np.rint(link['unrounded_profile']).astype(int)
    lap=d@d.T;lp=np.linalg.pinv(lap.astype(float));gt=t.T@t;gti=np.linalg.inv(gt.astype(float))
    bond_indices=[next(e for e,x in enumerate(links)if(x['start'],x['end'])==edge and x['image']==[0,0,0])for edge in MATTER_EDGES]
    bt=t[bond_indices];cycle_words=np.array([[1,1,1,-1,0],[1,1,0,0,-1]],int);cycle_profiles=cycle_words@bt
    # Explicit oriented path2->0->1->2->0: net site +0,-2, plus triangle cloud.
    path=np.array([1,1,0,0,-2],int);path_profile=path@bt
    target=np.array([1,0,-1,0]);matter_d=d[:,bond_indices]
    flux=np.array([(a,b)for a,b in product(range(-maximum,maximum+1),repeat=2)if(a+b)%2==0],int)
    integer_embedding=np.column_stack((d[:3].T,t))
    receipt={'positions':positions.tolist(),'torus_side':float(length),'links':links,'incidence':d.tolist(),'laplacian':lap.tolist(),'laplacian_pseudoinverse':lp.tolist(),'q':q.tolist(),'transverse_axes':[[0,1,0],[0,0,1]],'profile':t.tolist(),'profile_gram':gt.tolist(),'profile_rank':int(np.linalg.matrix_rank(t)),'divergence_residual':float(np.linalg.norm(d@t)),'profile_integer_rounding_max':float(max(np.max(abs(np.asarray(x['unrounded_profile'])-row))for x,row in zip(links,t))),'matter_edges':MATTER_EDGES,'matter_link_indices':bond_indices,'matter_profiles':bt.tolist(),'actual_cycle_words':cycle_words.tolist(),'actual_cycle_profiles':cycle_profiles.tolist(),'actual_cycle_profile_rank':int(np.linalg.matrix_rank(cycle_profiles)),'Wilson_path_word':path.tolist(),'Wilson_endpoint_residual':float(np.linalg.norm(matter_d@path-target)),'Wilson_transverse_shift':(3*path_profile).tolist(),'cover_integer_embedding':integer_embedding.tolist(),'cover_map_finite_kernel_index':embedding_index(integer_embedding),'cover_kernel_NOT_quotiented':True,'electric_parity_sector':0,'electric_fluxes':flux.tolist(),'electric_flux_count':len(flux),'electric_inertia_INPUT':1.,'bare_field_energies':(.5*np.einsum('na,ab,nb->n',flux,gti,flux)).tolist(),'no_magnetic_term':True,'q_is_nonzero_spatial_Fourier_on_torus_NOT_a_claim_of_continuum_dispersion':True}
    return receipt


def main():
    import sys
    output=sys.argv[1];result=construct();open(output,'w').write(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':'actual_spatial_charge_Cartan_geometry','field_flux_count':result['electric_flux_count'],'profile_rank':result['profile_rank'],'actual_cycle_profile_rank':result['actual_cycle_profile_rank'],'cover_kernel_index':result['cover_map_finite_kernel_index'],'divergence_residual':result['divergence_residual']}),flush=True)


if __name__=='__main__':main()
