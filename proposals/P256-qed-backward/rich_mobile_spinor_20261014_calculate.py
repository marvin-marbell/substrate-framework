"""New full-state mobile dispersion, packet, and finite-energy recoil calculation."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.linalg import expm
from rich_mobile_spinor_20261014_model import Parameters,build,spectrum,hamiltonian,vertex,spin_current,scattering,charged_green

ROOT=Path(__file__).resolve().parent
PREFIX='rich_mobile_spinor_20261014_actual1'

def expectation(v,a): return float(np.vdot(v,a@v).real)
def complex_value(z): return [float(z.real),float(z.imag)]

def band_row(model,k):
    e,v=spectrum(model,k)
    # Dense Hermitian eigh gives an orthonormal physical frame; no eigsh column weights.
    spin=[[expectation(v[:,j],s) for s in model['spin']] for j in range(6)]
    currents=[[expectation(v[:,j],vertex(model,k,k,np.eye(3)[axis])) for axis in range(3)] for j in range(6)]
    return {'k':list(k),'energies':e[:12].tolist(),'spin':spin,'charge_current':currents,'neutral_L2':[expectation(v[:,j],model['N2']) for j in range(6)],'orthonormal_residual':float(np.linalg.norm(v.conj().T@v-np.eye(len(e))))}

def packet(model):
    ncell=128; ks=2*np.pi*np.fft.fftfreq(ncell); center=ncell//2
    x=np.arange(ncell); envelope=np.exp(-(x-center)**2/(4*5.0**2))*np.exp(1j*.72*(x-center))
    source=np.zeros(len(model['H0']),complex)
    nn=len(model['neutral']['basis']); source[0]=1/np.sqrt(2); source[nn]=1j/np.sqrt(2)
    psi=envelope[:,None]*source[None,:]; psi/=np.linalg.norm(psi)
    fourier=np.fft.fft(psi,axis=0,norm='ortho'); energies=[]; vectors=[]; coefficients=[]
    for i,kx in enumerate(ks):
        e,v=spectrum(model,[kx,.18,0]); energies.append(e); vectors.append(v); coefficients.append(v.conj().T@fourier[i])
    energies=np.array(energies); vectors=np.array(vectors); coefficients=np.array(coefficients)
    records=[]; densities=[]
    for time in [0.,8.,16.,24.]:
        evolved=np.einsum('kij,kj->ki',vectors,coefficients*np.exp(-1j*time*energies))
        realspace=np.fft.ifft(evolved,axis=0,norm='ortho'); density=np.sum(abs(realspace)**2,axis=1)
        spin=np.einsum('ki,aij,kj->a',realspace.conj(),model['spin'],realspace).real
        neutral=np.einsum('ki,aij,kj->a',realspace.conj(),model['Ln'],realspace).real
        current=sum(expectation(evolved[i],vertex(model,[k,.18,0],[k,.18,0],[1,0,0])) for i,k in enumerate(ks))
        records.append({'time':time,'norm':float(density.sum()),'center':float(x@density),'width':float(np.sqrt(((x-x@density)**2)@density)),'charge_current_x':float(current),'spin':spin.tolist(),'neutral_angular_momentum':neutral.tolist(),'boundary_probability':float(density[:8].sum()+density[-8:].sum())})
        densities.append(density)
    return {'geometry':'128-cell periodic x array at conserved external transverse Bloch ky=.18, kz=0; localized Gaussian primitive s orbital, spin +y, neutral l0 source','records':records,'all_internal_bands_initially_and_dynamically_retained':True},np.array(densities)

def response(model):
    kin=np.array([.35,.08,.12]); qin=np.array([.28,.15,.1]); qout=np.array([-.12,.26,-.05])
    epsin=np.cross(qin,[0,0,1.]); epsin/=np.linalg.norm(epsin)
    epsout=np.cross(qout,[1.,0,0]); epsout/=np.linalg.norm(epsout)
    kout=kin+qin-qout; rows=[]
    ei,vi=spectrum(model,kin); ef,vf=spectrum(model,kout)
    for initial in [0,1]:
        for final in [0,1]:
            full=scattering(model,kin,qin,epsin,qout,epsout,.55,initial,final)
            low=scattering(model,kin,qin,epsin,qout,epsout,.55,initial,final,virtual=np.arange(4))
            rows.append({'initial':initial,'final':final,'incoming_spin':[expectation(vi[:,initial],s) for s in model['spin']],'outgoing_spin':[expectation(vf[:,final],s) for s in model['spin']],'amplitude':complex_value(full['amplitude']),'intensity_without_external_field_normalization':float(abs(full['amplitude'])**2),'contact':complex_value(full['contact']),'absorption':complex_value(full['absorption']),'emission':complex_value(full['emission']),'four_virtual_state_amplitude':complex_value(low['amplitude']),'omitted_virtual_amplitude':float(abs(full['amplitude']-low['amplitude'])),'recoil_energy':float(full['recoil_energy']),'outgoing_energy':float(full['outgoing_energy'])})
    return {'kin':kin.tolist(),'qin':qin.tolist(),'qout':qout.tolist(),'kout':kout.tolist(),'epsin':epsin.tolist(),'epsout':epsout.tolist(),'omega_in':.55,'eta':.01,'rows':rows}

def main():
    p=Parameters(); model=build(p)
    variants={'centrosymmetric':build(replace(p,polar_bias=0)),'polar':model,'no_orientation_transfer':build(replace(p,orientation=0)),'no_rotor_spin':build(replace(p,rotor_spin=0)),'no_tunneling':build(replace(p,hopping=0))}
    samples={name:[band_row(m,k) for k in [[0.,0.,0.],[.1,0.,0.],[.35,0.,0.],[.7,0.,0.],[0.,.35,0.],[.35,.35,0.],[0.,0.,.35]]] for name,m in variants.items()}
    k=np.array([.31,.17,.11]); kr=np.array([-k[1],k[0],k[2]])
    u=expm(-1j*np.pi/2*model['J'][2]); h=hamiltonian(model,k)
    symmetry=float(np.linalg.norm(hamiltonian(model,kr)-u@h@u.conj().T)/np.linalg.norm(h))
    q=np.array([.22,-.19,.13]); kin=k-q/2; kout=k+q/2
    continuity=vertex(model,kout,kin,q)-p.charge*(hamiltonian(model,kout)-hamiltonian(model,kin))
    delta=1e-3; e0=spectrum(model,[0,0,0])[0]; hessian=[]
    for axis in range(3):
        d=delta*np.eye(3)[axis]
        # Doublet mean isolates ordinary kinetic curvature from polar linear splitting.
        hessian.append(float((np.mean(spectrum(model,d)[0][:2])+np.mean(spectrum(model,-d)[0][:2])-2*np.mean(e0[:2]))/delta**2))
    packet_data,densities=packet(model); stopped,stopped_density=packet(variants['no_tunneling'])
    spin_current_values={}
    for name,m in variants.items():
        e,v=spectrum(m,[.35,0,0]); w=v[:,:2]
        spin_current_values[name]=[[expectation(w[:,j],spin_current(m,[.35,0,0],[.35,0,0],[1,0,0],axis)) for axis in range(3)] for j in range(2)]
    responses={name:response(m) for name,m in variants.items() if name in ['polar','no_orientation_transfer','no_tunneling']}
    # Exact memory-bearing neutral rotor resolvent and source addition weights, not hole poles.
    e,v=spectrum(model,k); a=v.conj().T@model['source']; weights=np.sum(abs(a)**2,axis=1)
    z=2.3+.02j; full_inverse=np.linalg.inv(z*np.eye(len(h))-h)
    nn=len(model['neutral']['basis']); retained=np.arange(0,len(h),nn); excluded=np.setdiff1d(np.arange(len(h)),retained)
    hpq=h[np.ix_(retained,excluded)]; sigma=hpq@np.linalg.solve(z*np.eye(len(excluded))-h[np.ix_(excluded,excluded)],hpq.conj().T)
    z2=2.8+.02j; sigma2=hpq@np.linalg.solve(z2*np.eye(len(excluded))-h[np.ix_(excluded,excluded)],hpq.conj().T)
    projected=np.linalg.inv(z*np.eye(len(retained))-h[np.ix_(retained,retained)]-sigma)
    result={'law':model['law'],'parameters':model['parameters'],'dimension':len(h),'basis':'charged complete l=0,1,2 shells x primitive spin 1/2 x co-moving neutral complete l=0,1,2 shells','band_samples':samples,'C4_covariance_relative_residual':symmetry,'finite_q_charge_continuity_residual':float(np.linalg.norm(continuity)),'rest_reference':{'empty_charge_energy':0.,'k0_addition_energy':float(e0[0]),'doublet_mean_inverse_inertial_curvature':hessian,'not_relativistic_mass':'particle_gap is an input addition offset; curvature is independently generated by contact hopping; polar lowest band can have displaced minima'},'packet':packet_data,'no_tunneling_packet':stopped,'spin_current':spin_current_values,'responses':responses,'source':{'weight_sum':float(weights.sum()),'expected_18_channels':18,'maximum_single_state_source_weight':float(weights.max()),'weights':weights.tolist()},'neutral_memory':{'P':'18 charged/spin states with neutral l0; Q all 144 neutral excited states','self_energy_norm_at_z':float(np.linalg.norm(sigma)),'frequency_change_norm':float(np.linalg.norm(sigma2-sigma)),'exact_schur_residual':float(np.linalg.norm(projected-full_inverse[np.ix_(retained,retained)]))},'green_norm_at_2_3':float(np.linalg.norm(charged_green(model,k,2.3,.02)))}
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2))
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),k=k,H=h,energies=e,vectors=v,source=model['source'],spin=model['spin'],Ln=model['Ln'],T=np.array(model['T']),packet_densities=densities,no_tunneling_densities=stopped_density)
    print(json.dumps({'law':model['law'],'dimension':len(h),'symmetry_residual':symmetry,'packet':packet_data,'rest_reference':result['rest_reference'],'neutral_memory':result['neutral_memory'],'polar_low_pair_at_kx_0_35':samples['polar'][2]},indent=2))

if __name__=='__main__': main()
