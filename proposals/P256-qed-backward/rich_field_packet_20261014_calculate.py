"""New common-H packet event; full-state propagation and spatial consumers."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply, norm
from rich_field_packet_20261014_model import Packet, PacketParameters, spatial_windows, profiles
from rich_frame_photon_20261014_model import Parameters


def ex(op,state):return float(np.vdot(state,op@state).real)


def encode(x):return {'real':x.real.tolist(),'imag':x.imag.tolist()}


def evolve(h,state,t):return expm_multiply(-1j*t*h,state,traceA=-1j*t*h.diagonal().sum())


def matter_state(m):
    # Explicit prepared local charge well, not a claim of a unique full vacuum.
    allowed=np.flatnonzero(np.diag(m.number[0])>.5) if m.occupation else np.arange(m.ns)
    h=m.HL_values[allowed[0]]+m.p.base.hubbard*np.sum(m.cube['density']==2,axis=1)
    supplier_h=np.diag(h).astype(complex)
    for index,hop in enumerate(m.cube['hops']):
        i,j=m.cube['edges'][index]
        cut=not m.bridge and m.cube['sites'][i,0]<0< m.cube['sites'][j,0]
        supplier_h-=m.p.base.hopping*int(m.induction and not cut)*(hop.toarray()+hop.toarray().conj().T)
    values,vectors=eigh(supplier_h)
    core=np.zeros(m.ns,dtype=complex)
    if len(allowed)>1:core[allowed[:2]]=[np.sqrt(.7),np.sqrt(.3)*np.exp(.7j)]
    else:core[allowed[0]]=1.
    return np.kron(core,vectors[:,0]),values[0]


def fields(m):
    # Both incoming transverse polarizations, independent of mobile cell/spin.
    left0=np.array([1,0,0,0,0,0]);left1=np.array([0,1,0,0,0,0])
    zero=np.flatnonzero(np.all(m.indices==0,axis=1))[0]
    packet=np.zeros(m.nfield,dtype=complex)
    for target,amplitude in [(left0,np.sqrt(.6)),(left1,1j*np.sqrt(.4))]:
        packet[np.flatnonzero(np.all(m.indices==target,axis=1))[0]]=amplitude
    # A Q/P phase adds a directional initial current response without a
    # particle-conditioned field displacement. Entire preparation retained.
    packet=np.sqrt(.85)*packet;packet[zero]=np.sqrt(.15)*np.exp(-.8j)
    reference=np.zeros(m.nfield,dtype=complex);reference[zero]=1.
    return packet,reference


def receiver(m,state):
    axis=np.array([.31,-.49,.815]);axis/=np.linalg.norm(axis)
    sm=sum(axis[a]*sum(s[a] for s in m.spin) for a in range(3));vals,vec=eigh(sm)
    projectors=[vec[:,vals>0]@vec[:,vals>0].conj().T,vec[:,vals<0]@vec[:,vals<0].conj().T]
    tensor=state.reshape(m.nfield,m.ns,64)
    # Disjoint actual electric-region spectral receiver, not mode labels.
    left,right=m.windows[0].diagonal(),m.windows[2].diagonal()
    mask=right>=left
    result=[]
    for c in range(2):
        for ps in projectors:
            for selected in (mask,~mask):
                z=tensor[selected];result.append(float(np.einsum('fmk,mn,fnk->',z.conj(),m.number[c]@ps,z).real))
    return result


def attach_windows(m,grams,first):
    m.windows=[m.electric_window(g) for g in grams]
    m.first=m.electric_window(sum(first))


    m.jphysical=m.current_fourier([0,0,0]);m.jcm=m.cm_current()
def snapshot(m,state):
    tensor=state.reshape(m.nfield,m.internal)
    pop=np.sum(abs(tensor)**2,axis=1)
    electric=[float(np.dot(pop,w.diagonal().real)) for w in m.windows]
    total=sum(electric)
    result=dict(norm=float(np.vdot(state,state).real),electric_regions=electric,electric_total=total,
                electric_centroid=float(np.dot(pop,m.first.diagonal().real)/total) if total>1e-14 else None,
                position=[ex(x,state) for x in m.X],spin=[ex(x,state) for x in m.S],cell=[ex(x,state) for x in m.N],
                mode_current=[ex(x,state) for x in m.J],physical_current=[ex(x,state) for x in m.jphysical],
                cm_charge_current=[ex(x,state) for x in m.jcm],energy=ex(m.H,state),
                # Exact regional transverse electric work, not Maxwell flux.
                electric_region_influx=[float(2*np.vdot(m.fop(w)@state,-1j*(m.H@state)).real) for w in m.windows])
    if m.occupation==1:result['inclusive_probabilities']=receiver(m,state)
    return result


def run(inp,out):
    data=json.loads(inp.read_text());p=PacketParameters(base=Parameters(**data['base']),cutoff=data['cutoff'],qwall=data['qwall'])
    grams,first=spatial_windows(p)
    sectors=[Packet(p,n) for n in range(3)];n0,m,n2=sectors
    for model in sectors:attach_windows(model,grams,first)
    matter,internal_energy=matter_state(m);field,ref_field=fields(m)
    prepared=np.kron(field,matter);reference=np.kron(ref_field,matter)
    times=data['times'];trajectory=evolve(m.H,np.column_stack([prepared,reference]),0.)
    records=[];saved={};last=0.
    for t in times:
        trajectory=evolve(m.H,trajectory,t-last);last=t
        records.append(dict(time=t,packet=snapshot(m,trajectory[:,0]),reference=snapshot(m,trajectory[:,1])))
        saved['packet_'+str(t)]=trajectory[:,0];saved['reference_'+str(t)]=trajectory[:,1]
    controls={}
    variants=[('current_off',1,dict(coupling=False)),('spin_metric_off',1,dict(metric=False)),('supplier_bridge_cut',1,dict(bridge=False)),('neutral_induction_on',0,{}),('neutral_induction_off',0,dict(induction=False))]
    for label,occ,options in variants:
        c=Packet(p,occ,**options);attach_windows(c,grams,first)
        cmatter,_=matter_state(c) if occ==0 else (matter,internal_energy)
        initial=np.kron(field,cmatter);ref=np.kron(ref_field,cmatter)
        both=np.column_stack([initial,ref]);entries=[];last=0.
        for t in times:
            both=evolve(c.H,both,t-last);last=t
            entries.append(dict(time=t,packet=snapshot(c,both[:,0]),reference=snapshot(c,both[:,1])))
        controls[label]=entries;saved[label+'_initial']=initial;saved[label+'_final']=both[:,0]
    k=np.array([.37,-.26,.19]);rho=m.density_fourier(k);j=m.current_fourier(k)
    continuity=float(norm(m.H@rho-rho@m.H+sum(k[a]*j[a] for a in range(3))))
    creators=[m.source_to(n2,i) for i in range(4)];annih=[n0.source_to(m,i).getH() for i in range(4)]
    car=[];wards=[]
    for a in range(4):
        for b in range(4):car.append(float(norm(creators[a].getH()@creators[b]+annih[b].getH()@annih[a]-(m.I if a==b else 0))))
        form=np.exp(-1j*np.dot(p.centres[a//2],k)-p.base.core_width**2*np.dot(k,k)/2)
        wards.append(float(norm(n2.density_fourier(k)@creators[a]-creators[a]@rho+form*creators[a])))
    source_columns=np.column_stack([c@prepared for c in creators]);removal_columns=np.column_stack([c@prepared for c in annih])
    source_return=evolve(n2.H,source_columns,data['source_time'])
    removal_return=evolve(n0.H,removal_columns,data['source_time'])
    step=2e-5;derivatives=[]
    for a in range(m.nmodes):
        shift=np.zeros(m.nmodes);shift[a]=step
        derivatives.append(float(norm((m.shifted(shift)-m.shifted(-shift))/(2*step)+m.J[a])))
    # Initial far support and operator nonlocality are explicitly measurable.
    left=m.fop(m.Q[0]);far=m.fop(m.windows[2]);near=m.fop(m.windows[0])
    commutators=dict(left_Q_far_electric=float(norm(left@far-far@left)),left_Q_near_electric=float(norm(left@near-near@left)),electric_regions_commute=float(norm(near@far-far@near)),bare_E_far_commute=float(norm(m.E@far-far@m.E)),field_prepare_core_commute=0.)
    packet_pop=abs(field)**2
    initial_spatial=[float(np.dot(packet_pop,w.diagonal().real)) for w in m.windows]
    gram_eigen=np.linalg.eigvalsh(m.G)
    result=dict(kind='ACTUAL_TRANSLATED_FIELD_PACKET_MOBILE_ONE_H',parameters=asdict(p),dimensions=[x.size for x in sectors],field_dimension=m.nfield,internal_supplier_dimension=64,
                metric=m.G.tolist(),electric_metric=m.M.tolist(),dual_metric=m.Minv.tolist(),metric_eigenvalues=gram_eigen.tolist(),gram_quadrature_error=float(np.linalg.norm(sum(grams)-m.G)),
                spatial_grams=grams.tolist(),initial_energy_regions=initial_spatial,initial_far_energy_fraction=initial_spatial[2]/sum(initial_spatial),commutators=commutators,
                normalized_profile_Gram=(m.G/np.sqrt(np.diag(m.G)[:,None]*np.diag(m.G)[None,:])).tolist(),source_CAR_max=max(car),source_charge_Ward=wards,
                physical_halfspin=float(norm(sum(s@s for s in m.S)-.75*m.I)),continuum_continuity=continuity,mode_current_derivatives=derivatives,
                cm_current_identity=[float(norm(m.cm_current()[a]+1j*(m.H@m.X[a]-m.X[a]@m.H))) for a in range(3)],
                source_time=data['source_time'],addition_time_kernel=encode(source_columns.conj().T@source_return),removal_time_kernel=encode(removal_columns.conj().T@removal_return),
                addition_source_Gram=encode(source_columns.conj().T@source_columns),removal_source_Gram=encode(removal_columns.conj().T@removal_columns),
                event=records,controls=controls,prepared_internal_energy=float(internal_energy),
                interpretation='Finite explicitly projected electric/quantum-matter medium. Regional electric influx is transverse work/redistribution, not installed Maxwell flux. Nonzero tails/commutators are reported; no universal causal front, free photon, asymptotic scattering or unique vacuum claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),initial=prepared,reference_initial=reference,field_packet=field,field_reference=ref_field,matter=matter,indices=m.indices,source_columns=source_columns,removal_columns=removal_columns,**saved)
    print(json.dumps({key:result[key] for key in ['dimensions','field_dimension','gram_quadrature_error','initial_far_energy_fraction','source_CAR_max','physical_halfspin','continuum_continuity','commutators']}))
    print(json.dumps(dict(trajectory=[dict(time=e['time'],regions=e['packet']['electric_regions'],centroid=e['packet']['electric_centroid'],cell=e['packet']['cell'],spin=e['packet']['spin']) for e in records])))


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
