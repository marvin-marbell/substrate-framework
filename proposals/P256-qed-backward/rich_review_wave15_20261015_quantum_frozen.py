"""Same full-cut source states, scalar/loop unchanged, EM-coordinate coupling frozen."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from rich_quantum_common_field_20261015_model import compressed
from rich_review_wave15_20261015_quantum import load,reconstruct
from rich_common_colour_lepton_20261015_model import scalar_field
from rich_colour_composite_20261014_model import loop_field


def main():
    inp,out=map(Path,sys.argv[1:3]);req=json.loads(inp.read_text());states=np.load(req['owner_input']);data=np.load(req['owner_npz']);observed=json.loads(Path(req['owner_json']).read_text());parameters=req['parameters'];scalar=scalar_field(parameters);loop=loop_field();cut=parameters['electric_cut'];values=np.arange(-cut,cut+1);size=len(values);nb=size*size;center=cut*size+cut;results={}
    metric=9*parameters['permittivity']*parameters['tube_area']/parameters['cycle_circumference']*np.array([[2.,1.],[1.,2.]]);inverse=np.linalg.inv(metric);grid=np.array([(x,y) for x in values for y in values]);he=sparse.diags(np.einsum('ia,ab,ib->i',grid,inverse,grid)/2)
    for name in ['lepton','baryon']:
        h=load(data,name+'_H');matter=h.shape[0]//nb;row=np.arange(matter)*nb;m0=h[row+center][:,row+center];deltas={(sign*k,0) for k in [1,2,3] for sign in [-1,1]}|{(0,sign*k) for k in [1,2,3] for sign in [-1,1]}|{(sign*k,sign*k) for k in [1,2,3] for sign in [-1,1]}
        for dx,dy in deltas:m0+=h[row+(dx+cut)*size+dy+cut][:,row+center]
        hf=sparse.kron(m0,sparse.eye(nb))+sparse.kron(sparse.eye(matter),he);initial=states[name+'_initial'];kicked=states[name+'_kicked'];t=req['time'];a=expm_multiply(-1j*t*hf,initial);b=expm_multiply(-1j*t*hf,kicked)
        sector=reconstruct(data,name,req['retained_gauge_blocks'][name],scalar,loop,parameters);excess={}
        for key in ['phi2','C','positive']:
            op=sparse.kron(compressed(sector,key,scalar),sparse.eye(nb));excess[key]=float(np.vdot(b,op@b).real-np.vdot(a,op@a).real)
        results[name]=dict(dimension=h.shape[0],time=t,norms=[float(np.linalg.norm(a)),float(np.linalg.norm(b))],frozen_physical_material_excess=excess,full_coupled_material_excess=observed['results'][name]['samples'][-1]['physical_material_excess'],physical_EM_coordinate_derivative_current='ZERO: this wrong-case H is explicitly independent of both EM coordinates. Original-law current is NOT relabelled frozen-H current.',scalar_colour_loop='All scalar/colour/matter matrices and exact full population/energy/field cuts unchanged; only common EM path phases fixed to zero.')
    result=dict(kind='SAME_FULL_CUT_SAME_INITIAL_QUANTUM_EM_FROZEN_WRONGCASE',request=req,results=results,meaning='Exact full-H shift coefficients sum at theta=0 supplies frozen matter H. Same original prepared states and unchanged field kinetic. This control removes dynamic matter-field coupling, not low-rank source or physical field-only protocol. It does not claim the absence of magnetism.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
