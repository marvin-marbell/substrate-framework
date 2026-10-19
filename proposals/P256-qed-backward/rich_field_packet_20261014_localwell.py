"""Interacting prepared local well; independent physical local A injection.

This replaces the failed product-state induction-quench protocol, not its law.
Every source, current, energy density and sector is the same Packet Hamiltonian.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh, norm
from rich_field_packet_20261014_model import Packet, PacketParameters, spatial_windows
from rich_field_packet_20261014_calculate import attach_windows, snapshot, evolve, encode
from rich_frame_photon_20261014_model import Parameters


def ground(m,well):
    h=m.H-well*m.N[0]
    if m.occupation:h=h-.035*(.31*m.S[0]-.49*m.S[1]+.815*m.S[2])
    rng=np.random.default_rng(1410222)
    v=rng.normal(size=m.size)+1j*rng.normal(size=m.size)
    energy,state=eigsh(h,k=1,which='SA',v0=v,tol=3e-10,maxiter=18000)
    return state[:,0],float(energy[0]),float(np.linalg.norm(h@state[:,0]-energy[0]*state[:,0]))


def preparation(m,grams,strength):
    # Integral over x<-1 of (.6 u_left,z+.8 u_left,y).A(x).
    # This is a real spatial smearing, NOT a profile-coordinate label.
    coeff=(.6*grams[0,0]+.8*grams[0,1])/m.G[0,0]
    q=sum(coeff[a]*m.Q[a] for a in range(m.nmodes))
    values,vectors=np.linalg.eigh(q.toarray())
    unitary=(vectors*np.exp(1j*strength*values))@vectors.conj().T
    return coeff,q,unitary


def event(m,g,unitary,times):
    packet=(unitary@g.reshape(m.nfield,m.internal)).ravel()
    both=np.column_stack([packet,g]);entries=[];last=0.;saved={}
    for t in times:
        both=evolve(m.H,both,t-last);last=t
        p,r=snapshot(m,both[:,0]),snapshot(m,both[:,1])
        excess=np.array(p['electric_regions'])-r['electric_regions']
        entries.append(dict(time=t,packet=p,reference=r,excess_electric_regions=excess.tolist(),
                            excess_total=float(sum(excess)),excess_centroid=float((p['electric_total']*p['electric_centroid']-r['electric_total']*r['electric_centroid'])/sum(excess)) if abs(sum(excess))>1e-12 and r['electric_centroid'] is not None else None,
                            spin_contrast=(np.array(p['spin'])-r['spin']).tolist(),position_contrast=(np.array(p['position'])-r['position']).tolist()))
        saved['packet_'+str(t)]=both[:,0];saved['reference_'+str(t)]=both[:,1]
    return entries,packet,saved


def run(inp,out):
    data=json.loads(inp.read_text());p=PacketParameters(base=Parameters(**data['base']),cutoff=data['cutoff'],qwall=data['qwall'])
    grams,first=spatial_windows(p);models=[Packet(p,n) for n in range(3)];n0,m,n2=models
    for x in models:attach_windows(x,grams,first)
    g,eg,res=ground(m,data['well']);coeff,q,unitary=preparation(m,grams,data['strength'])
    records,prepared,saved=event(m,g,unitary,data['times'])
    saved.update(initial=prepared,reference_initial=g,field_unitary=unitary,indices=m.indices)
    controls={};control_grounds={}
    for name,occ,options in [('current_off',1,{'coupling':False}),('spin_metric_off',1,{'metric':False}),('supplier_bridge_cut',1,{'bridge':False}),('neutral_induction_on',0,{}),('neutral_induction_off',0,{'induction':False})]:
        c=Packet(p,occ,**options);attach_windows(c,grams,first)
        cg,ce,cr=ground(c,data['well'] if occ else 0.)
        _,_,cu=preparation(c,grams,data['strength'])
        entries,cp,cs=event(c,cg,cu,data['times'])
        controls[name]=entries;control_grounds[name]=dict(energy=ce,residual=cr)
        saved[name+'_initial']=cp;saved[name+'_reference']=cg;saved[name+'_final']=cs['packet_'+str(data['times'][-1])]
    creators=[m.source_to(n2,i) for i in range(4)];annih=[n0.source_to(m,i).getH() for i in range(4)]
    plus=np.column_stack([c@prepared for c in creators]);minus=np.column_stack([c@prepared for c in annih])
    source=evolve(n2.H,plus,data['source_time']);removal=evolve(n0.H,minus,data['source_time'])
    car=max(float(norm(creators[a].getH()@creators[b]+annih[b].getH()@annih[a]-(m.I if a==b else 0))) for a in range(4) for b in range(4))
    k=np.array([.37,-.26,.19]);rho=m.density_fourier(k);j=m.current_fourier(k)
    continuity=float(norm(m.H@rho-rho@m.H+sum(k[a]*j[a] for a in range(3))))
    wards=[]
    for a in range(4):
        form=np.exp(-1j*np.dot(p.centres[a//2],k)-p.base.core_width**2*np.dot(k,k)/2)
        wards.append(float(norm(n2.density_fourier(k)@creators[a]-creators[a]@rho+form*creators[a])))
    source_data=dict(addition_Gram=encode(plus.conj().T@plus),removal_Gram=encode(minus.conj().T@minus),addition_time_kernel=encode(plus.conj().T@source),removal_time_kernel=encode(minus.conj().T@removal),source_time=data['source_time'])
    far=m.windows[2]
    prep_far=float(norm(sp.csr_matrix(unitary).getH()@far@sp.csr_matrix(unitary)-far))
    initial=np.array(records[0]['excess_electric_regions'])
    result=dict(kind='LOCAL_PHYSICAL_FIELD_INJECTION_INTERACTING_MOBILE_EVENT',parameters=asdict(p),dimensions=[x.size for x in models],field_dimension=m.nfield,
                prepared_well=data['well'],well_removed_during_event=True,spin_preparation_bias=.035,ground_energy=eg,ground_residual=res,control_grounds=control_grounds,
                physical_source_smearing=coeff.tolist(),source_region='x<-1, all y,z; Gaussian left-profile test current',initial_excess_electric=initial.tolist(),initial_far_excess_fraction=float(initial[2]/sum(initial)),
                field_preparation_far_operator_change=prep_far,field_preparation_core_commutator=0.,gram=m.G.tolist(),dual=m.Minv.tolist(),metric_eigenvalues=np.linalg.eigvalsh(m.G).tolist(),gram_quadrature_error=float(np.linalg.norm(sum(grams)-m.G)),
                source_CAR_max=car,source_charge_Ward=wards,continuum_continuity=continuity,physical_halfspin=float(norm(sum(s@s for s in m.S)-.75*m.I)),source=source_data,
                event=records,controls=controls,
                limits='Interacting finite prepared well, released for event. Explicit joint Fourier-Q field projection. Local smearing has projected tails/nonlocality measured rather than assumed absent. Electric regional influx is transverse work, not Maxwell Poynting. Travelling-packet claim requires sequential actual excess-energy arrival and geometry controls; nonzero motion/feedback alone is insufficient.')
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),plus_sources=plus,minus_sources=minus,**saved)
    print(json.dumps({k:result[k] for k in ['dimensions','ground_residual','initial_far_excess_fraction','field_preparation_far_operator_change','source_CAR_max','continuum_continuity']}))
    print(json.dumps([dict(t=e['time'],excess=e['excess_electric_regions'],centroid=e['excess_centroid'],position=e['position_contrast'],spin=e['spin_contrast']) for e in records]))


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
