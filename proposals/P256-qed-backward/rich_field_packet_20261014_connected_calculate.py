"""Actual packet consumer for the materially changed 3D supplier geometry."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse.linalg import norm
from rich_field_packet_20261014_connected import ConnectedPacket
from rich_field_packet_20261014_model import PacketParameters, spatial_windows
from rich_field_packet_20261014_localwell import ground, preparation, event
from rich_field_packet_20261014_calculate import attach_windows, evolve, encode
from rich_frame_photon_20261014_model import Parameters


def run(inp,out):
    data=json.loads(inp.read_text());p=PacketParameters(base=Parameters(**data['base']),cutoff=data['cutoff'],qwall=data['qwall'],supplier_spacing=data.get('supplier_spacing',2.))
    grams,first=spatial_windows(p)
    models=[ConnectedPacket(p,n) for n in range(3)];n0,m,n2=models
    for c in models:attach_windows(c,grams,first)
    g,eg,gr=ground(m,data['well']);coeff,q,unitary=preparation(m,grams,data['strength'])
    records,initial,saved=event(m,g,unitary,data['times'])
    saved.update(initial=initial,reference_initial=g,indices=m.indices,field_unitary=unitary)
    controls={};control_grounds={}
    for label,occupation,options in [('current_off',1,{'coupling':False}),('spin_metric_off',1,{'metric':False}),('supplier_bridge_cut',1,{'bridge':False}),('neutral_induction_on',0,{}),('neutral_induction_off',0,{'induction':False})]:
        c=ConnectedPacket(p,occupation,**options);attach_windows(c,grams,first)
        cg,ce,cr=ground(c,data['well'] if occupation else 0.)
        _,_,cu=preparation(c,grams,data['strength'])
        entries,cp,cs=event(c,cg,cu,data['times'])
        controls[label]=entries;control_grounds[label]={'energy':ce,'residual':cr}
        saved[label+'_initial']=cp;saved[label+'_reference']=cg;saved[label+'_final']=cs['packet_'+str(data['times'][-1])]
    creators=[m.source_to(n2,i) for i in range(4)];annih=[n0.source_to(m,i).getH() for i in range(4)]
    plus=np.column_stack([c@initial for c in creators]);minus=np.column_stack([c@initial for c in annih])
    car=max(float(norm(creators[a].getH()@creators[b]+annih[b].getH()@annih[a]-(m.I if a==b else 0))) for a in range(4) for b in range(4))
    k=np.array([.37,-.26,.19]);rho=m.density_fourier(k);j=m.current_fourier(k)
    continuity=float(norm(m.H@rho-rho@m.H+sum(k[a]*j[a] for a in range(3))))
    wards=[]
    for a in range(4):
        form=np.exp(-1j*np.dot(p.centres[a//2],k)-p.base.core_width**2*np.dot(k,k)/2)
        wards.append(float(norm(n2.density_fourier(k)@creators[a]-creators[a]@rho+form*creators[a])))
    source=evolve(n2.H,plus,data['source_time']);removal=evolve(n0.H,minus,data['source_time'])
    jj=np.column_stack([j@g for j in m.J]);neutral_ground,_,_=ground(n0,0.)
    supplier_jj=np.column_stack([-n0.link_derivative((a,))@neutral_ground for a in range(m.nmodes)])
    fieldsources=np.array(records[0]['excess_electric_regions'])
    result=dict(kind='THREE_DIMENSIONAL_CONNECTED_SUPPLIER_FIELD_PACKET_EVENT',parameters=asdict(p),dimensions=[c.size for c in models],field_dimension=m.nfield,
                supplier_positions=m.cube['sites'].tolist(),supplier_edges=m.cube['edges'],supplier_phase=m.cube['phase'].tolist(),supplier_phase_singular_values=np.linalg.svd(m.cube['phase'],compute_uv=False).tolist(),
                ground_energy=eg,ground_residual=gr,control_grounds=control_grounds,physical_source_smearing=coeff.tolist(),gram=m.G.tolist(),dual_metric=m.Minv.tolist(),metric_eigenvalues=np.linalg.eigvalsh(m.G).tolist(),gram_quadrature_error=float(np.linalg.norm(sum(grams)-m.G)),
                field_preparation_far_operator_change=float(np.linalg.norm(unitary.conj().T@m.windows[2]@unitary-m.windows[2].toarray())),initial_excess_electric=fieldsources.tolist(),initial_far_excess_fraction=float(fieldsources[2]/sum(fieldsources)),
                current_Gram_eigenvalues=np.linalg.eigvalsh(jj.conj().T@jj).tolist(),neutral_supplier_current_Gram_eigenvalues=np.linalg.eigvalsh(supplier_jj.conj().T@supplier_jj).tolist(),
                physical_halfspin=float(norm(sum(s@s for s in m.S)-.75*m.I)),source_CAR_max=car,source_charge_Ward=wards,continuum_continuity=continuity,
                source=dict(addition_Gram=encode(plus.conj().T@plus),removal_Gram=encode(minus.conj().T@minus),addition_time_kernel=encode(plus.conj().T@source),removal_time_kernel=encode(minus.conj().T@removal),time=data['source_time']),
                event=records,controls=controls,
                scope='Actual full64 shared supplier, complete mobile N0/N1/N2, six nonorthogonal transverse profiles, explicit joint field projection. Prepared interacting well/bias removed during event; own same-state reference per variant. Electric influx is transverse work. Bothpolarization support measured. No causal-front/free-space/Lorentz/Maxwell-B assertion; a failed geometric transit remains failed.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),plus_sources=plus,minus_sources=minus,**saved)
    print(json.dumps({k:result[k] for k in ['dimensions','ground_residual','initial_far_excess_fraction','source_CAR_max','continuum_continuity','neutral_supplier_current_Gram_eigenvalues']}))
    print(json.dumps([dict(time=e['time'],excess=e['excess_electric_regions'],centroid=e['excess_centroid'],position=e['position_contrast'],spin=e['spin_contrast']) for e in records]))


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
