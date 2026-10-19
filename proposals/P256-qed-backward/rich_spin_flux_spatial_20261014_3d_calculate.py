"""Literal NEW3D common-H two-Wilson-leg field and charge receiver calculation."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, splu
from rich_spin_flux_spatial_20261014_3d_model import LAW, Sector, Parameters
from rich_spin_flux_spatial_20261014_model import mean, centered_sources, gram_frame, charged_source

ROOT = Path(__file__).resolve().parent
PREFIX = 'rich_spin_flux_spatial_20261014_actual4'


def encode(v):
    if isinstance(v, np.ndarray):
        return encode(v.tolist())
    if isinstance(v, complex):
        return [v.real, v.imag]
    if isinstance(v, dict):
        return {k: encode(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)):
        return [encode(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def run(p, spin, times):
    s = Sector(p, int(spin == 1), int(spin == -1))
    e,g,low = s.ground()
    incoming = centered_sources(g, s.F[:2])
    outgoing = centered_sources(g, s.F[2:])
    qi,gi,vi = gram_frame(incoming)
    qo,go,vo = gram_frame(outgoing)
    waves = s.evolve(qi, times, e)
    amp = np.einsum('ar,tai->tri', qo.conj(), waves)
    out = np.sum(np.abs(amp)**2, axis=1)
    pg = np.abs(np.einsum('a,tai->ti',g.conj(),waves))**2
    eta = .35
    impulses = np.column_stack([expm_multiply(-1j*eta*f,g,traceA=0.) for f in s.F[:2]])
    iw = s.evolve(impulses, times, e)
    ipout = np.sum(np.abs(np.einsum('ar,tai->tri',qo.conj(),iw))**2,axis=1)
    ipground = np.abs(np.einsum('a,tai->ti',g.conj(),iw))**2
    probability = np.abs(iw)**2
    detector = (s.electric[:,[10,11]]**2 > 0).astype(float)
    detector_values = np.einsum('tai,ar->tir',probability,detector)
    detector_baseline = np.abs(g)**2@detector
    farenergy = np.einsum('tai,a->ti',probability,np.sum(s.electric[:,[10,11]]**2,axis=1)/(2*p.inertia))
    farenergy_baseline = float(np.abs(g)**2@np.sum(s.electric[:,[10,11]]**2,axis=1)/(2*p.inertia))
    body = np.einsum('tai,a->ti',probability,s.active_density@s.positions[:,0])
    electric = np.einsum('tai,a->ti',probability,s.electric_energy)
    flux = np.array([[[mean(w[:,i],f) for f in s.F] for i in range(2)] for w in iw])
    sources = s.F+s.loop_electric+s.total_currents+s.vector_current
    response = s.retarded(g,e,sources,.43+.11j)
    response_time = 2*np.imag(np.einsum('ar,tai->tri',outgoing.conj(),s.evolve(incoming,times,e)))
    current = {sp: [mean(g,j) for j in s.currents[sp]] for sp in ['active','supplier']}
    data = {'law':LAW,'parameters':asdict(p),'spin_z':spin/2,'dimension':s.dimension,'energy':e,'lowest':low,'norm':float(np.vdot(g,g).real),'eigen_residual':float(np.linalg.norm(s.H@g-e*g)),'constraints':s.constraints(),'intrinsic_flux':[mean(g,f) for f in s.F],'species_current':current,'contact':{sp:[mean(g,k) for k in s.contacts[sp]] for sp in ['active','supplier']},'orientor_L':mean(g,s.Ln),'neutral_torque_norm':float(np.linalg.norm(s.neutral_torque@g)),'boundary_occupancy':float(np.abs(g)**2@(np.max(np.abs(s.electric),axis=1)==p.bound)),'incoming_source_operators':['F_xy_left','F_xz_left'],'outgoing_source_operators':['F_xy_right','F_xz_right'],'incoming_ground_means':[mean(g,f) for f in s.F[:2]],'outgoing_ground_means':[mean(g,f) for f in s.F[2:]],'incoming_Gram':gi,'outgoing_Gram':go,'incoming_Gram_eigenvalues':vi,'outgoing_Gram_eigenvalues':vo,'incoming_rank':qi.shape[1],'outgoing_rank':qo.shape[1],'times':times,'normalized_out_probability':out,'normalized_ground_probability':pg,'normalized_remainder':1-out-pg,'out_probability_max':out.max(axis=0),'out_probability_initial':out[0],'full_source_norm_drift':float(np.max(np.abs(np.sum(np.abs(waves)**2,axis=1)-1))),'retarded_operators':['F0..3','Eloop0..3','Jtotal0..11','Jvectorx,y,z'],'full_retarded':response,'remote_time_retarded':response_time,'remote_time_retarded_max':np.max(np.abs(response_time),axis=0),'impulses':{'eta':eta,'operators':['exp(-i eta F0)','exp(-i eta F1)'],'ground_probability':ipground,'out_probability':ipout,'remainder_probability':1-ipground-ipout,'local_far_E_nonzero_probability':detector_values,'local_far_detector_baseline':detector_baseline,'local_far_detector_max_change':np.max(np.abs(detector_values-detector_baseline),axis=0),'local_far_detector_initial_change':detector_values[0]-detector_baseline,'far_E_energy':farenergy,'far_E_energy_baseline':farenergy_baseline,'far_E_energy_max_change':np.max(np.abs(farenergy-farenergy_baseline),axis=0),'body_x':body,'body_displacement_max':np.max(np.abs(body-mean(g,s.body_x)),axis=0),'electric_energy':electric,'electric_exchange_range':np.ptp(electric,axis=0),'flux':flux,'remote_flux_max_change':np.max(np.abs(flux[:,:,2:]-np.array([mean(g,f) for f in s.F[2:]])),axis=0),'norm_drift':float(np.max(np.abs(probability.sum(axis=1)-1)))},'cut_edges':s.cut_edges,'no_polarization_claim':'Two independent actual orthogonal-plane Wilson-created field legs, NOT proved massless transverse photon polarizations; finite thin ladder boundary still present','measurement_semantics':'Connected sourceGram-normalized full field+chargedstate projectors plus retained ground and remainder; separately strictly local final-cell electric receiver'}
    arrays = {'ground':g,'incoming':qi,'outgoing':qo,'impulses':impulses,'incoming_final':waves[-1],'impulse_final':iw[-1]}
    return data,arrays,s


def g12(p,s,g,e):
    targets = [Sector(p,0,0),Sector(p,2,0),Sector(p,1,1)]
    removal = np.column_stack([charged_source(s,targets[0],x,0,False)@g for x in range(9)])
    additions = [np.column_stack([charged_source(s,target,x,spin,True)@g for x in range(9)]) for target,spin in zip(targets[1:],[0,1])]
    z=.61+.13j
    eye=sparse.eye(targets[0].dimension,format='csc')
    backward=splu((z*eye+targets[0].H-e*eye).tocsc()).solve(removal)
    values=[]
    for target,source in zip(targets[1:],additions):
        eye=sparse.eye(target.dimension,format='csc')
        value=source.conj().T@splu((z*eye-target.H+e*eye).tocsc()).solve(source)
        if target.up==2:
            value+=(removal.conj().T@backward).T
        values.append(value)
    return {'z':z,'dimensions':[t.dimension for t in targets],'G_same_add_remove':values[0],'G_opposite_add':values[1],'source_Gram_same_add':additions[0].conj().T@additions[0],'source_Gram_opposite_add':additions[1].conj().T@additions[1],'source_Gram_remove':removal.conj().T@removal,'source_closure_same':additions[0].conj().T@additions[0]+(removal.conj().T@removal).T,'countercharges0':[-1,-3,-3],'full_virtual':True,'constraints':[t.constraints() for t in targets]}, {'remove_sources':removal,'same_add_sources':additions[0],'opposite_add_sources':additions[1]}


def main():
    request=json.loads((ROOT/'rich_spin_flux_spatial_20261014_3d_input.json').read_text())
    p=Parameters(**request['parameters'])
    times=np.linspace(0.,request['time_stop'],request['time_count'])
    result={'law':LAW,'request':request,'cases':{}}
    stored={}
    reference=None
    for name,params,spin in [('up',p,1),('down',p,-1),('true_cut',replace(p,cut=True),1),('supplier_frozen',replace(p,supplier_hop=0.),1),('torque_off',replace(p,torque=0.),1)]:
        data,arrays,s=run(params,spin,times)
        result['cases'][name]=data
        stored.update({name+'_'+key:value for key,value in arrays.items()})
        if name=='up':
            reference=s
        print(name,'dim',s.dimension,'flux',data['intrinsic_flux'],'ranks',data['incoming_rank'],data['outgoing_rank'],'outmax',data['out_probability_max'],'farEchange',data['impulses']['far_E_energy_max_change'],flush=True)
    if reference is None:
        raise RuntimeError('Missing actual reference')
    gd,ga=g12(p,reference,stored['up_ground'],result['cases']['up']['energy'])
    result['G12']=gd
    stored.update(ga)
    print('full3D G12 dimensions',gd['dimensions'],flush=True)
    # NEW3D physical charged packet from supplier-only ground, not a v2 packet import.
    vacuum=Sector(p,0,0)
    ev,gv,_=vacuum.ground()
    source=charged_source(vacuum,reference,1,0)
    raw=source@gv
    norm=float(np.vdot(raw,raw).real)
    packet=raw/np.sqrt(norm)
    pw=reference.evolve(packet,times)
    prob=np.abs(pw)**2
    density=prob@reference.active_density
    far=density[:,6:9].sum(axis=1)
    electric=prob@reference.electric_energy
    result['charged_packet']={'vacuum_energy':ev,'source_site':1,'tree_string':'edge6 0->1 electric+1; activeup CAR; added static -1 anchor0','source_Gram':norm,'countercharges0':[-1,-2],'times':times,'active_density':density,'far_charge_probability':far,'far_charge_max':float(far.max()),'electric_energy':electric,'electric_exchange_range':float(np.ptp(electric)),'norm_drift':float(np.max(np.abs(prob.sum(axis=1)-1)))}
    stored['charged_packet_seed']=packet
    stored['charged_packet_final']=pw[-1]
    print('3D packetGram',norm,'farcharge',float(far.max()),'electricexchange',float(np.ptp(electric)),flush=True)
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(encode(result),indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**stored)


if __name__=='__main__':
    main()
