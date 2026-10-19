"""Own extended-H BANDPAIR CONTROL only: expose physical density-current defects.
No photon/faithful-density-current admission; no new field/reference state is spliced.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import numpy as np
from scipy import sparse
from rich_spatial_colour_em_20261015_model import build,assemble,observable
from rich_spatial_colour_em_20261015_snapshot import factor
from rich_spatial_colour_em_20261015_density import no_hop_law,spatial_current


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent;paths=[root/request['state_json'],root/request['state_npz']]
    metadata=json.loads(paths[0].read_text());data=np.load(paths[1]);p=metadata['parameters'];model=build(p);base=model['base'];field=model['field'];results={}
    for name in ['neutral','lepton','baryon']:
        op=base['operators'][name];nrq=base['rotor']['dimension']*field['dimension'];h=factor(data,name+'_H',op['Q'].shape[0],base['scalar'],nrq)
        state=data['neutral_reference'] if name=='neutral' else data[name+'_source_0:0.5_raw'];state=state/np.linalg.norm(state)
        rho=observable(model,name,matter=op['rho0']);routes=[assemble(model,name,route_derivative=edge) for edge in range(2)]
        hstate=h@state;charge_motion=1j*(h@(rho@state)-rho@hstate);route=sum(current@state for current in routes)
        off=no_hop_law(model,name,h);off_motion=1j*(off@(rho@state)-rho@(off@state))
        complement=(op['density_delta2']-op['density_delta']@op['density_delta'])/4
        missing=float(np.vdot(state,observable(model,name,matter=complement)@state).real)
        dipoles=[]
        for axis in range(3):
            value=base['geometry']['displacement'][0]/2*base['rotor']['R'][:,axis,0]
            dipole=observable(model,name,matter=op['density_delta'],rq=sparse.kron(base['rotor']['integrate'](value),sparse.eye(field['dimension'])))
            full=spatial_current(model,name,[0.,0.,0.],axis);wrong=spatial_current(model,name,[0.,0.,0.],axis,include_mechanical=False)
            change=1j*(h@(dipole@state)-dipole@hstate)
            dipoles.append(dict(axis=axis,full_current_defect=float(np.linalg.norm(change-full@state)),wrong_no_mechanical_defect=float(np.linalg.norm(change-wrong@state)),mechanical_current_norm=float(np.linalg.norm((full-wrong)@state))))
        results[name]=dict(state='own extended neutral ground' if name=='neutral' else 'own normalized unfiltered raw halfspin source',dimension=h.shape[0],exact_snapshot_vs_rebuilt_action=float(np.linalg.norm(hstate-assemble(model,name)@state)),co_moving_Ward_defect=float(np.linalg.norm(charge_motion+route)),physical_route_image_norm=float(np.linalg.norm(route)),all_hops_off_unphysical_vertex_charge_motion=float(np.linalg.norm(off_motion)),local_EM_charge_omitted_norm_squared=missing,density_complement_Frobenius_norm=float(sparse.linalg.norm(complement)),constant_gradient_dipole=dipoles,interpretation='This native bandpair1 P does NOT commute physical site charge; full Rspin/EM products and correct mechanical current do not repair that representation. Current remains the original physical derivative. These are adverse control values, not a physical-current/photon/state admission.')
        out.write_text(json.dumps(dict(kind='CONDITIONAL_BANDPAIR_EXTENDED_H_ADVERSE_CONTROL_NOT_PRIMARY_JOIN',input_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},results=results,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
        print(json.dumps(dict(sector=name,result=results[name])),flush=True)


if __name__=='__main__':main()
