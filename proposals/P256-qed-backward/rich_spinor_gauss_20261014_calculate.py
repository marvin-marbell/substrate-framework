"""NEW physical calculation, no archived author CLI replay."""
import json,time,hashlib
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
from scipy import sparse
from rich_spinor_gauss_20261014_model import Parameters,Sector,charged_source,full_resolvent,retarded,scattering,LAW
from rich_spinor_20261014_model import build_sector,Parameters as FrozenParameters
ROOT=Path(__file__).resolve().parent

def encode(x):
    if isinstance(x,np.ndarray):
        if np.iscomplexobj(x):return {'real':x.real.tolist(),'imag':x.imag.tolist()}
        return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def frozen(p):
    """U=1 and E removed: explicitly NON-Gauss comparator, not delivery."""
    old=build_sector(FrozenParameters(p_gap=p.p_gap,d_gap=p.d_gap,rotor_gap=p.rotor_gap,orientation=p.orientation,rotor_spin=p.rotor_spin,charging=p.charging),1)
    s=object.__new__(Sector);s.p=p;s.number=1;s.g=old['geometry'];s.r=s.g['positions'];s.edges=list(__import__('itertools').combinations(range(6),2));s.H=sparse.csr_matrix(old['H']);s.dimension=48;s.spin=[sparse.csr_matrix(a) for a in old['spin']]
    s.transfers=[]
    for a,b in s.edges:
        t=np.zeros((12,12),complex);t[2*a:2*a+2,2*b:2*b+2]=old['orbital'][a,b]*np.eye(2)
        s.transfers.append(sparse.kron(sparse.csr_matrix(t),sparse.eye(4),format='csr'))
    s.currents=[-1j*(t-t.getH()) for t in s.transfers];s.contacts=[-(t+t.getH()) for t in s.transfers]
    return s

def sample(s,name):
    t=time.monotonic();e,v=s.solve(6);print(name,'dimension',s.dimension,'E',e[:4],flush=True)
    ground=np.flatnonzero(abs(e-e[0])<1e-8);lower=int(ground[0]);upper=int(ground[-1]); result={'dimension':s.dimension,'low_energies':e,'ground_degeneracy_selected':len(ground),'branches':[]}
    for i in (lower,upper):
        g=v[:,i];amp,c=scattering(s,g,e[i],[0,0,.7],[.55,0,.25],1.7,.07)
        spin=[float(np.vdot(g,o@g).real) for o in s.spin];moment=[float(np.vdot(g,s.magnetic(a)@g).real) for a in range(3)]
        result['branches'].append({'index':i,'norm':float(np.vdot(g,g).real),'spin':spin,'source_moment':moment,'amplitude':amp,'weight':abs(amp)**2,'contact':c,'current_virtual_norm':float(np.linalg.norm(amp-c)),'source_g_unit_mass_reference':None if abs(spin[2])<1e-9 else -2*moment[2]/spin[2]})
    result['spin_helicity_contrast']=float(np.linalg.norm(abs(result['branches'][0]['amplitude'])**2-abs(result['branches'][1]['amplitude'])**2))
    result['elapsed_seconds']=time.monotonic()-t
    return result,e,v,upper

def run():
    p=Parameters(flux_reference=.3);all_e=None;out={'law':LAW,'parameters':asdict(p),'units':'hbar=|charge|=octahedronradius=1; mass only external reference=1 for labelled g ratio','cases':{}}
    arrays={}
    for name,params in [('electric_only_screened',replace(p,flux_reference=0.)),('coupled',p),('no_orientor_transfer',replace(p,rotor_spin=0.))]:
        s=Sector(params,1,all_e);all_e=s.all_e
        data,e,v,i=sample(s,name);g=v[:,i];data.update(s.state_data(g));data['constraints']=s.constraint()
        data['parameters']=asdict(params)
        flux=[s.wilson([0,2,4]),s.wilson([0,3,4])];ops=flux+[s.magnetic(2),s.spin[2],sparse.diags(np.repeat(s.rho[:,0],4))]
        response=retarded(s,g,e[i],ops,.8+.07j);data['response_operators']=['sinWilson024','sinWilson034','orbitalM_z','electronS_z','rho_anchor'];data['same_state_retarded_response']=response
        data['flux_response_norm']=float(np.linalg.norm(response[:2,:2]));data['spin_magnetic_mixed_norm']=float(abs(response[2,3])+abs(response[3,2]));data['state_id']=hashlib.sha256(s.H.data.tobytes()+g.tobytes()).hexdigest()
        arrays[name+'_ground']=g;arrays[name+'_electric']=s.e;arrays[name+'_bits']=s.bits
        if name=='coupled':
            plus=Sector(params,2,all_e);minus=Sector(params,0,all_e);print('adjacent sectors',minus.dimension,plus.dimension,flush=True)
            adds=[];rems=[];metadata=[]
            for mode in range(12):
                a,ma=charged_source(s,plus,mode,True);b,mb=charged_source(s,minus,mode,False)
                adds.append(a@g);rems.append(b@g);metadata.append({'mode':mode,'addition':ma,'removal':mb})
            adds=np.column_stack(adds);rems=np.column_stack(rems);z=.5+.07j
            gp=adds.conj().T@full_resolvent(plus.H,z+e[i],adds)
            gm=(rems.conj().T@full_resolvent(-minus.H,z-e[i],rems)).T
            data['G12']=gp+gm;data['source_equal_time']=adds.conj().T@adds+(rems.conj().T@rems).T
            data['source_metadata']=metadata;data['adjacent_sector_dimensions']={'N0':minus.dimension,'N1':s.dimension,'N2':plus.dimension}
            data['adjacent_constraints']={'N0':minus.constraint(),'N2':plus.constraint()}
            data['spin_up_down_G_difference']=float(np.linalg.norm((gp+gm)[::2,::2]-(gp+gm)[1::2,1::2]))
            arrays['G12']=gp+gm;arrays['source_add']=adds;arrays['source_remove']=rems
        out['cases'][name]=data
    reference=frozen(p);data,e,v,i=sample(reference,'field_removed_U_identity');out['cases']['field_removed_U_identity']=data
    a=out['cases']['coupled'];b=out['cases']['no_orientor_transfer'];c=out['cases']['field_removed_U_identity']
    out['comparisons']={'no_transfer_amplitude_difference':float(np.linalg.norm(a['branches'][1]['amplitude']-b['branches'][1]['amplitude'])),'removed_field_amplitude_difference':float(np.linalg.norm(a['branches'][1]['amplitude']-c['branches'][1]['amplitude'])),'removed_field_ground_shift':float(a['low_energies'][0]-c['low_energies'][0]),'full_virtual_not_contact_only':a['branches'][1]['current_virtual_norm']}
    out['limitations']=['All K6 links literal; all 12 one-electron/4 orientor and all66 two-electron/4 orientor configurations occur within an electric-energy ball, NOT effective4 projection.','Integer E ball <=4 imposes real link-shift/Wilson/source boundary losses, reported not normalized away.','Static N-dependent countercharge at +x anchor breaks octahedral spatial symmetry and is not isolated electron.','Primitive CAR/spin-half/orientor torque and all energy coefficients supplied; no derived g2, Maxwell, continuum SO3 or propagating photon.','field_removed_U_identity removes Gauss constraint and is only nongauge matter comparison.','Full-H resolvents; selected six eigenstates only state preparation, not virtual truncation.','g_unit_mass_reference uses supplied mass1 and charge-1, not derived electron g.']
    (ROOT/'rich_spinor_gauss_20261014_actual3_result.json').write_text(json.dumps(out,default=encode,indent=2))
    np.savez_compressed(ROOT/'rich_spinor_gauss_20261014_actual3_states.npz',**arrays)
    print(json.dumps({'complete':True,'cases':{k:{'dimension':v['dimension'],'spin_helicity_contrast':v['spin_helicity_contrast'],'source_moment':v['branches'][1]['source_moment']} for k,v in out['cases'].items()},'comparisons':out['comparisons']},default=encode),flush=True)
if __name__=='__main__':run()
