"""NEW calculations on two material contact architectures, all joint states retained."""
from dataclasses import replace
from pathlib import Path
from typing import cast
import json
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve
from rich_dirac_contact_20261014_model import Parameters, build, hamiltonian, spectrum, vertex, contact, spin_current, charged_green, scattering

ROOT=Path(__file__).resolve().parent
PREFIX='rich_dirac_contact_20261014_actual1'


def serial(x):
    if isinstance(x,np.ndarray): return serial(x.tolist())
    if isinstance(x,complex): return {'real':x.real,'imag':x.imag}
    if isinstance(x,np.generic): return serial(x.item())
    if isinstance(x,dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [serial(v) for v in x]
    return x


def doublet_means(e): return np.array([np.mean(e[:2]),np.mean(e[2:4])])


def rest_diagnostics(m,k):
    e,v=spectrum(m,k); w=v[:,:4]; low=e[:4]; center=np.mean(low); mass=(np.mean(low[2:])-np.mean(low[:2]))/2
    vel=np.array([w.conj().T@vertex(m,k,k,d)@w/m['params'].charge for d in np.eye(3)])
    speed=np.sqrt(np.mean([np.trace(a@a).real/4 for a in vel]))
    al=vel/speed; beta=np.diag((low-center)/mass)
    cliff=np.array([[np.linalg.norm(al[i]@al[j]+al[j]@al[i]-2*(i==j)*np.eye(4))/2 for j in range(3)] for i in range(3)])
    j2=sum(a@a for a in m['J']); step=.002
    curves=[]
    for d in np.eye(3):
        ep,_=spectrum(m,k+step*d); em,_=spectrum(m,k-step*d)
        curves.append((doublet_means(ep)+doublet_means(em)-2*doublet_means(e))/step**2)
    curves=np.array(curves)
    scalar=np.mean(curves,axis=1); mass_curve=(curves[:,1]-curves[:,0])/2
    # Direct lattice material curvature before all-state virtual transitions.
    bare=[]; virtual=[]
    for d in np.eye(3):
        second=contact(m,k,k,np.zeros(3),d,np.zeros(3),d)/m['params'].charge**2
        vv=vertex(m,k,k,d)/m['params'].charge
        bare.append([np.trace(v[:,a:a+2].conj().T@second@v[:,a:a+2]).real/2 for a in [0,2]])
        vr=[]
        for a in [0,2]:
            value=0.
            for b in range(a,a+2):
                ids=np.array([j for j in range(len(e)) if abs(e[j]-e[b])>1e-8])
                value+=np.sum(2*np.abs(v[:,ids].conj().T@vv@v[:,b])**2/(e[b]-e[ids])).real/2
            vr.append(value)
        virtual.append(vr)
    samples=[]
    for direction in [np.array([1.,0,0]),np.ones(3)/np.sqrt(3)]:
        for p in [.05,.15,.35,.65]:
            ee,_=spectrum(m,k+p*direction); upper=np.mean(ee[2:4])
            samples.append({'p':p,'direction':direction,'upper_actual':upper,'lower_actual':np.mean(ee[:2]),'parity_centered_Dirac_comparator':center+np.sqrt(mass**2+speed**2*p*p),'absolute_empty_reference_comparator':np.sqrt(np.mean(low[2:])**2+speed**2*p*p),'scalar_center_drift':np.mean(ee[:4])-center})
    return {'k':k,'energies':low,'next_energy':e[4],'parity':cast(NDArray[np.complex128],np.diag(w.conj().T@m['parity']@w)).real,'physical_internal_J2':cast(NDArray[np.complex128],np.diag(w.conj().T@j2@w)).real,'charged_L2':cast(NDArray[np.complex128],np.diag(w.conj().T@m['Lc2']@w)).real,'neutral_L2':cast(NDArray[np.complex128],np.diag(w.conj().T@m['Ln2']@w)).real,'field_number':cast(NDArray[np.complex128],np.diag(w.conj().T@m['field_number']@w)).real,'primitive_Sz':cast(NDArray[np.complex128],np.diag(w.conj().T@m['spin'][2]@w)).real,'center_constant':center,'mass_energy_half_gap':mass,'physical_vector_speed':speed,'current_low4':vel*m['params'].charge,'offparity_current_norm':[np.linalg.norm(a[:2,2:]) for a in vel],'Clifford_anticommutator_errors':cliff,'mass_anticommutator_errors':[np.linalg.norm(a@beta+beta@a) for a in al],'curvature_lower_upper':curves,'scalar_curvature':scalar,'mass_curvature':mass_curve,'direct_material_curvature':bare,'all_state_virtual_curvature':virtual,'inertial_mass_upper':1/curves[:,1],'centered_Dirac_mass':mass/speed**2,'absolute_rest_mass_comparator':np.mean(low[2:])/speed**2,'dispersion_comparators':samples},e,v


def feedback(m,k,e,v):
    p=v[:,:4]; q=v[:,4:]; momentum=k+np.array([.23,.11,-.09]); h=hamiltonian(m,momentum); z=e[3]+.19+.07j
    hp=p.conj().T@h@p; hq=q.conj().T@h@q; coupling=p.conj().T@h@q
    sigma=coupling@solve(z*np.eye(len(hq))-hq,coupling.conj().T,assume_a='gen',check_finite=False)
    exact=p.conj().T@solve(z*np.eye(len(h))-h,p,check_finite=False)
    effective=solve(z*np.eye(4)-hp-sigma,np.eye(4),check_finite=False)
    source=m['source']; weights=np.sum(np.abs(v.conj().T@source)**2,axis=1)
    jin=vertex(m,k,k,np.array([1.,.3,.2])); a=v[:,3]; amplitudes=v.conj().T@jin@a; de=e-e[3]
    chi=np.sum(np.abs(amplitudes)**2*(1/(.37+.08j-de)-1/(.37+.08j+de)))
    fielda=v.conj().T@m['field_number']@a
    mixed=np.sum(fielda.conj()*amplitudes/(.37+.08j-de)-amplitudes.conj()*fielda/(.37+.08j+de))
    green=charged_green(m,momentum,z)
    direct=source.conj().T@solve(z*np.eye(len(h))-h,source,check_finite=False)
    return {'Feshbach_selfenergy':sigma,'selfenergy_norm':np.linalg.norm(sigma),'exact_reconstruction_error':np.linalg.norm(exact-effective),'low4_without_virtual_error':np.linalg.norm(exact-solve(z*np.eye(4)-hp,np.eye(4),check_finite=False)),'source_weight_total':weights.sum(),'source_weight_low4':weights[:4].sum(),'all_source_pole_weights':weights,'G_full_direct_difference':np.linalg.norm(green-direct),'current_response_full':chi,'field_number_current_response':mixed,'virtual_current_weight_total':np.sum(np.abs(amplitudes)**2),'virtual_current_weight_low4':np.sum(np.abs(amplitudes[:4])**2)}


def packet(m,rest):
    count=96; ks=2*np.pi*np.fft.fftfreq(count); mean=rest[0]+.29
    delta=np.angle(np.exp(1j*(ks-mean))); envelope=np.exp(-delta**2/(4*.13**2)); envelope/=np.linalg.norm(envelope)
    _,vv=spectrum(m,np.array([mean,rest[1],rest[2]])); source=vv[:,3]
    eig=[]; vec=[]; velocity=[]; sj=[]
    for kx in ks:
        kk=np.array([kx,rest[1],rest[2]]); ee,ww=spectrum(m,kk); eig.append(ee); vec.append(ww)
        velocity.append(vertex(m,kk,kk,np.eye(3)[0])/m['params'].charge)
        sj.append(spin_current(m,kk,kk,np.eye(3)[0],2))
    eig=np.array(eig); vec=np.array(vec); coefficient=np.einsum('kna,n->ka',vec.conj(),source)*envelope[:,None]
    coords=np.arange(-count//2,count//2); rows=[]; densities=[]
    for t in [0.,8.,16.,24.]:
        momentum_state=np.einsum('kna,ka->kn',vec,coefficient*np.exp(-1j*eig*t))
        position=np.fft.fftshift(np.fft.ifft(momentum_state,axis=0,norm='ortho'),axes=0)
        density=np.sum(np.abs(position)**2,axis=1); densities.append(density)
        drift=np.einsum('kn,knm,km->',momentum_state.conj(),np.array(velocity),momentum_state).real
        spinflow=np.einsum('kn,knm,km->',momentum_state.conj(),np.array(sj),momentum_state).real
        fn=np.einsum('kn,nm,km->',momentum_state.conj(),m['field_number'],momentum_state).real
        sz=np.einsum('kn,nm,km->',momentum_state.conj(),m['spin'][2],momentum_state).real
        rows.append({'time':t,'norm':density.sum(),'mean_x':coords@density,'variance_x':coords**2@density-(coords@density)**2,'velocity':drift,'charge_current':m['params'].charge*drift,'spin_current_zx':spinflow,'primitive_Sz':sz,'field_number':fn})
    return rows,{'packet_ks':ks,'packet_energies':eig,'packet_densities':np.array(densities),'packet_x':coords}


def calculate(geometry,parameters):
    params=Parameters(geometry=geometry,**parameters); m=build(params)
    rest=np.ones(3)*np.pi if geometry=='facing' else np.zeros(3)
    rd,e,v=rest_diagnostics(m,rest)
    # Full three-dimensional grid and a line: no projected-band evolution.
    grid=np.array([[x,y,z] for x in [-np.pi,0,np.pi/2] for y in [-np.pi,0,np.pi/2] for z in [-np.pi,0,np.pi/2]])
    bands=np.array([spectrum(m,k)[0] for k in grid])
    line=np.linspace(-np.pi,np.pi,49); line_bands=np.array([spectrum(m,rest+np.array([p,0,0]))[0] for p in line])
    kin=rest+np.array([.19,-.12,.08]); qin=np.array([.21,.10,-.07]); qout=np.array([-.06,.16,.04])
    inc=np.array([1.,.3,.2]); out=np.array([.2,.8,-.1])
    full=scattering(m,kin,qin,inc,qout,out,.43); low=scattering(m,kin,qin,inc,qout,out,.43,virtual=np.arange(4))
    spinflip=scattering(m,kin,qin,inc,qout,out,.43,initial=3,final=2)
    pkt,arrays=packet(m,rest)
    bare=build(replace(params,cavity_coupling=0.)); eb,vb=spectrum(bare,rest)
    result={'law':m['law'],'parameters':m['parameters'],'dimension':len(e),'rest':rd,'band_grid_minmax':[bands.min(),bands.max()],'Kramers_pair_max_split_grid':np.max(np.abs(bands[:,::2]-bands[:,1::2])),'feedback':feedback(m,rest,e,v),'same_law_nonforward_scattering':full,'low4_virtual_scattering':low,'virtual_truncation_amplitude_error':abs(full['amplitude']-low['amplitude']),'spinflip_scattering':spinflip,'packet':pkt,'supplied_field_dressing':{'uncoupled_low4':eb[:4],'pole_shifts':e[:4]-eb[:4],'field_occupation_low4':rd['field_number']},'physical_rotation_scope':'Onsite sphere+orientor+field J is physical; fixed cubic bond directions break continuous global rotation. Full lattice Noether rotation is not claimed.'}
    arrays.update({'grid_k':grid,'grid_all_energies':bands,'line_offset':line,'line_all_energies':line_bands,'rest_energies':e,'rest_vectors':v,'source':m['source']})
    return result,arrays


def main():
    result={'kind':'NEW_FULL_STATE_CHARGED_PARITY_CONTACT_AND_MATERIAL_FORK','scope':'N0 vacuum addition only, macrocell EM, all angular/neutral/supplied-cavity states; no N2, removal or positron','architectures':{}}; arrays={}
    inputs=json.loads((ROOT/'rich_dirac_contact_20261014_input.json').read_text())
    for geometry in inputs['architectures']:
        r,a=calculate(geometry,inputs['parameters']); result['architectures'][geometry]=r
        arrays.update({geometry+'_'+key:value for key,value in a.items()})
        print(json.dumps(serial({'geometry':geometry,'energies':r['rest']['energies'],'parity':r['rest']['parity'],'Lc2':r['rest']['charged_L2'],'speed':r['rest']['physical_vector_speed'],'curvature':r['rest']['curvature_lower_upper'],'scalar':r['rest']['scalar_curvature'],'clifford':r['rest']['Clifford_anticommutator_errors'],'packet':r['packet'],'scattering':r['same_law_nonforward_scattering']})),flush=True)
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(serial(result),indent=2)+'\n')
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),**arrays)

if __name__=='__main__': main()
