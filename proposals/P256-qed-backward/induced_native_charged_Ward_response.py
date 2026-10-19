"""ActualdressedGaussian/GW electron-field density/puregauge response CONTROL.

Derivativeinsertionsarephysicalcharge/current inactualfullpositiveauxiliarylaw,
not a Ward vertex imposedfrominverseG differences. OriginalE/Ref3/GaussSOURCE
untouched; q0A0legcharge NOT photon-normalizedeR/fullGausscompensatedsourcecharge.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent
OWN=Path(__file__).read_bytes()
import induced_native_charged_green as charged
CHARGED_SHA='7b72b6e05f9990699a8bfd425cd0be062564d608b02ba993884332fde87a34d2'
SOURCE_BYTES={str(Path(__file__).resolve()):OWN,**charged.SOURCES}


def matrix(value:Any)->np.ndarray:
    return np.asarray(value['real'])+1j*np.asarray(value['imag'])


def gauge_vertices(native:dict[str,Any],profile:dict[str,Any],pout:np.ndarray,pin:np.ndarray,edge:np.ndarray,density:np.ndarray,sitephase:np.ndarray)->np.ndarray:
    """Actual J(Theta_grad) actingonactualUg=exp(i alpha f)bareorbit.

    Sourcehopphase exp(i(alpha_x-alpha_y)) andUg_y combine toexp(i alpha_x),
    never constructedbysettingdressedGamma=Ginverse difference.
    """
    V=native['V'];M=edge.shape[1];R=density.shape[1];out=np.zeros((M+R,4,4),dtype=np.complex128);sites=native['sites']
    wave=np.exp(1j*sites@(pin-pout))*sitephase/V
    out[M:]=np.einsum('x,xr->r',wave,density)[:,None,None]*np.eye(4)
    for path in profile['paths']:
        x,y=path['source'],path['target'];field=native['a']*np.sum(edge[list(path['edges'])],axis=0);hop=path['hop']
        forward=np.exp(1j*(sites[y]@pin-sites[x]@pout))*sitephase[x]/V;reverse=np.exp(1j*(sites[x]@pin-sites[y]@pout))*sitephase[y]/V
        out[:M]+=1j*field[:,None,None]*(forward*hop-reverse*hop.conj().T)
    return out


def spectral_sources(supplier:Any,alphas:list[float])->dict[str,Any]:
    native=supplier.native;s=supplier.s;profile=native['Pprofiles'][0];V=native['V'];mom=(2*np.pi*native['sites']+np.pi)/native['N'];zero=np.zeros(3*V);f=np.cos(2*np.pi*native['sites'][:,0]/4)
    targets=[int(np.flatnonzero(np.all(native['sites']==[i,0,0],axis=1))[0]) for i in range(4)];embeds=[charged.embedding(native,mom[i]) for i in targets];Eall=np.column_stack(embeds)
    hfull=s.primitives.matter_hamiltonian(native,profile,zero);allE=[charged.embedding(native,p) for p in mom];bands=[np.linalg.eigh(E.conj().T@hfull@E) for E in allE]
    ev,U=np.linalg.eigh(hfull);P=U[:,:2*V]@U[:,:2*V].conj().T;states=[];covariance=np.zeros((3*V,3*V));qrows=[];vertex_controls=[]
    for alpha in alphas:states.append(dict(alpha=alpha,phase=np.exp(1j*alpha*f),B=[],d=[]))
    countA=countR=0
    for index in native['sites']:
        q=tuple(int(i) for i in index);neg=tuple((-i)%native['N'] for i in q)
        if q>neg:continue
        block=supplier.block(q);edge=block.data['physical_edgeframe'];density=block.data['physical_densityframe'];amp=block.amplitudes;omega=block.omega;countA+=block.M;countR+=block.R
        covariance+=native['a']**2*edge@np.real(amp[:block.M]@amp[:block.M].conj().T)@edge.T
        allowed={tuple((native['sites'][r]+sign*np.asarray(q))%native['N']) for r in targets for sign in (-1,1)}
        for k,ksite in enumerate(native['sites']):
            if tuple(ksite) not in allowed:continue
            eig,Ub=bands[k]
            for st in states:
                O=np.concatenate([gauge_vertices(native,profile,mom[r],mom[k],edge,density,st['phase']) for r in targets],axis=1)
                for column in range(4):
                    amps=amp if eig[column]>0 else amp.conj();B=np.einsum('mij,ml,j->il',O,amps,Ub[:,column],optimize=True)
                    st['B'].append(B);st['d'].append(eig[column]+omega if eig[column]>0 else eig[column]-omega)
                if q in ((0,0,0),(1,0,0)) and k==targets[0]:
                    gradient=native['D'].T@(st['alpha']*f);Ug=np.repeat(st['phase'],4)[:,None]*allE[k]
                    for j in range(min(2,block.M)):
                        literal=s.primitives.path_derivative(native,profile,gradient,native['a']*edge[:,j]);pred=Eall.conj().T@literal@Ug
                        error=float(np.max(abs(O[j]-pred)));vertex_controls.append(dict(alpha=st['alpha'],q=q,frame=j,actual_literal_gauged_hopping_current_eigenstate_vertex_error=error))
        qrows.append(dict(q=q,positive_modes=len(omega),photon_coordinates=block.M,density_coordinates=block.R))
    if (countA,countR,len(qrows))!=(129,63,36):raise ArithmeticError('DressedWard consumer didnotcoverALLphysicalframes')
    hermitian_controls=[]
    for st in states:
        alpha=st['alpha'];phase=np.repeat(st['phase'],4);theta=native['D'].T@(alpha*f);hg=s.primitives.matter_hamiltonian(native,profile,theta);Pg=phase[:,None]*P*phase.conj()[None,:]
        C=native['UE']/native['a']*(np.kron(np.diag(np.diag(native['Lplus'])),np.eye(4))-2*np.repeat(np.repeat(native['Lplus'],4,axis=0),4,axis=1)*Pg);DW=np.zeros_like(hg)
        for path in profile['paths']:
            edges=list(path['edges']);var=float(np.sum(covariance[np.ix_(edges,edges)]));x,y=path['source'],path['target'];hop=path['hop']*np.exp(1j*np.sum(theta[edges]));DW[4*x:4*x+4,4*y:4*y+4]-=.5*var*hop;DW[4*y:4*y+4,4*x:4*x+4]-=.5*var*hop.conj().T
        st['h']=Eall.conj().T@(hg+C+DW)@Eall;st['B']=np.concatenate(st['B'],axis=1);st['d']=np.concatenate(st['d']);st['literal_hg_error']=float(np.max(abs(hg-phase[:,None]*hfull*phase.conj()[None,:])));hermitian_controls.append(float(np.linalg.norm(st['h']-st['h'].conj().T)))
    return dict(native=native,states=states,targets=targets,momenta=[mom[r] for r in targets],density_generator=Eall.conj().T@np.kron(np.diag(f),np.eye(4))@Eall,source_vertex_controls=vertex_controls,static_hermiticity_errors=hermitian_controls,qgroups=qrows)


def G(st:dict[str,Any],z:complex,potential:float=0.)->np.ndarray:
    # ActualexternalA0 couplescharge1 BOTH receiver and everychargedauxiliary;
    # neutralPH/photonmodesunchanged. Commonneutralvacuumreferenceunchanged.
    h=st['h']+potential*np.eye(16);Sigma=charged.self_energy(st['B'],st['d']+potential,z)
    return np.linalg.solve(z*np.eye(16)-h-Sigma,np.eye(16))


def vertex_density(st:dict[str,Any],zout:complex,zin:complex)->np.ndarray:
    # ActualdensityinsertioninEVERYchargedvirtualelectronline; barecharge1.
    B,d=st['B'],st['d'];return np.eye(16)+(B/((zout-d)*(zin-d))[None,:])@B.conj().T


def physical(supplier:Any,cap:dict[str,Any],eps:float)->dict[str,Any]:
    sources=spectral_sources(supplier,[-eps,0.,eps]);minus,base,plus=sources['states'];rho=sources['density_generator'];controls=[];temporal=[];pole_charges=[];matches=[]
    for z in (complex(0,.5),complex(10,.25),complex(30,1)):
        gb=G(base,z);gm=G(minus,z);gp=G(plus,z);response=(gp-gm)/(2*eps);expected=1j*(rho@gb-gb@rho);err=float(np.max(abs(response-expected)))
        actual_vertex=(plus['h']+charged.self_energy(plus['B'],plus['d'],z)-minus['h']-charged.self_energy(minus['B'],minus['d'],z))/(2*eps);ward=1j*(rho@(base['h']+charged.self_energy(base['B'],base['d'],z))-(base['h']+charged.self_energy(base['B'],base['d'],z))@rho)
        vertexerr=float(np.max(abs(actual_vertex-ward)));closure=float(np.max(abs(response-gb@actual_vertex@gb)));bare_only=float(np.max(abs(response-gb@(1j*(rho@base['h']-base['h']@rho))@gb)))
        if max(err,vertexerr,closure)>1e-6 or bare_only<1e-8:raise ArithmeticError('ActualgaugedGWdensity/currentdressedWard consumer failed')
        controls.append(dict(z=z,actual_physical_gauge_response=response,actual_dressed_gauge_vertex=actual_vertex,dressed_response_Ward_error=err,dressed_vertex_Ward_error=vertexerr,actual_response_vertex_closure_error=closure,WRONG_bare_only_dressed_response_defect=bare_only))
        temporal_response=(G(base,z,eps)-G(base,z,-eps))/(2*eps);Gamma=vertex_density(base,z,z);errA0=float(np.max(abs(temporal_response-gb@Gamma@gb)));wrongA0=float(np.max(abs(temporal_response-gb@gb)))
        if errA0>1e-6 or wrongA0<1e-8:raise ArithmeticError('ActualuniformA0virtualcharge insertion response failed')
        for nu in (.17,.61):
            zout=z+nu;v=vertex_density(base,zout,z);invout=zout*np.eye(16)-base['h']-charged.self_energy(base['B'],base['d'],zout);invin=z*np.eye(16)-base['h']-charged.self_energy(base['B'],base['d'],z);error=float(np.max(abs(nu*v-(invout-invin))))
            if error>1e-9:raise ArithmeticError('ActualfinitefrequencydressedchargedfielddensityWard failed')
            temporal.append(dict(z=z,actual_external_frequency=nu,actual_virtual_line_charge_inserted_density_vertex=v,actual_finite_frequency_dressed_density_Ward_error=error,actual_uniformA0_physical_response_error=errA0,WRONG_bare_density_response_defect=wrongA0))
    for recindex,index in enumerate((0,1)):
        frozen=cap['charged_receivers'][recindex];sl=slice(4*index,4*index+4)
        for r in frozen['retarded_probes']:
            z=complex(r['z']['real'],r['z']['imag']);delta=float(np.max(abs(G(base,z)[sl,sl]-matrix(r['G']))));matches.append(dict(receiver=recindex,z=z,actual_frozen7b_Green_consumer_error=delta))
            if delta>1e-10:raise ArithmeticError('NewphysicalWardsource no longerjoinsfrozen7bGreen')
        for pole in frozen['actually_isolated_discrete_poles']:
            if pole['energy']<=0:continue
            energy=pole['energy'];R=matrix(pole['residue']);Gamma=vertex_density(base,complex(energy),complex(energy))[sl,sl];charge=float(np.trace(R@Gamma).real/pole['multiplicity']);error=float(np.max(abs(R@Gamma@R-R)))
            if abs(charge-1)>1e-9 or error>1e-9:raise ArithmeticError('Actualchargedpole residue-times-densityWard charge failed')
            pole_charges.append(dict(receiver=recindex,actual_isolated_auxiliarylaw_pole=energy,actual_frozen_residue_matrix=R,actual_virtualdensity_vertex_at_pole=Gamma,observed_charge_per_spin_from_frozen_residue_times_density_vertex=charge,actual_R_Gamma_R_equals_R_error=error,WRONG_bare_vertex_charge_per_spin=float(np.trace(R).real/pole['multiplicity'])))
    return dict(N=4,ALL_qgroups=36,ALL_photon_coordinates=129,ALL_density_coordinates=63,physical_gauge_probe_amplitude=eps,actual_AP_momenta=sources['momenta'],actual_auxiliary_states=len(base['d']),actual_source_literal_vertex_controls=sources['source_vertex_controls'],actual_static_h_hermiticity_errors=sources['static_hermiticity_errors'],actual_nonuniform_gauge_h_literal_errors=[s['literal_hg_error'] for s in sources['states']],actual_DRESSED_gauge_current_density_response_controls=controls,actual_DRESSED_temporal_density_charge_insertions=temporal,actual_isolated_pole_residue_density_charge_consumers=pole_charges,actual_frozen7b_Green_reuse_consumers=matches,physics='SameactualGaussian/GW h+bareCself/exchange+FULLorderedPeierlsDW and ALL36q/3865mode/positiveandnegativeelectron virtualstates. Nonuniformgauge probe alpha cosqx: literalh(gradalpha), literalallupdatedJ/secondcontact andactualgauge-transformednegativeprojectorC; actualvirtual-state eigenfunctionsUg and allcoherentA+phi vertices, thenG re-evaluated. VertexNOTimposedasWarddifference. UniformexternalA0 legcharge couplesbothreceiverandvirtualchargedstates, neutralbosonfrequenciesunchanged; directfinitepotential G response comparedtoexplicitdensityinsertionsinsideallvirtualelectronlines. Positiveisolatedfrozen7b pole residues normalizedbyphysicaldensityvertex givecharge1 withoutfitting/whitening.',scope='NEWfiniteGaussian/GW chargedfieldleg response, not completeconservingmicroscopicQEDvertex/g-2/Thomson eR/commoncone. q0uniformA0 isexternalbookkeepingchargeprobe; zeroqphysicalscalarCoulombmodeabsent. OriginalcompensatedE/Ref3 globalGaussSOURCE charge isneutral, NOTpromotedascharge1; no originalSOURCE/probability/receivernormcutover. Staticnonuniformpuregauge/densitytemporalWardcomponents achieved; generalnonzeroq finitefrequency/transversevertex and fullnonlinear/source/Gausslift remainopen.')


def run(path:Path,eps:float)->dict[str,Any]:
    start=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CHARGED_SHA:raise ArithmeticError('Frozenactual3a9/7b chargedGreen supplier required')
    if eps<=0:raise ValueError('Physicalpuregauge/potentialprobeamplitude required')
    cap=json.loads(raw);supplier=charged.load_supplier();desc=dict(plan='issue-222-induced-photon-delta13-continuation-20261008',scientific_execution_revision='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7',input_receipts=dict(charged=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),photon=dict(path=str(charged.RECEIPT),sha256=charged.RECEIPT_SHA)),actual_charged_input_descriptor=cap['input_descriptor'],probe_amplitude=eps,original_Lambda_UNCHANGED='2^8192',original_chi_UNCHANGED='2^-4096',original_clock_UNCHANGED='4096',branch='DistinctphysicaldressedGaussian/GW charged-field Ward-response consumer NOToriginalE/Ref3/source/eR')
    result=physical(supplier,cap,eps);captured={**charged.SOURCES,**SOURCE_BYTES};supplier.s.assert_frozen()
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(charged.packed(desc),sort_keys=True).encode()).hexdigest(),actual_DRESSED_native_charged_Ward_response=result,source_at_import={Path(name).name:dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in captured.items()},source_unchanged_since_import={name:Path(name).read_bytes()==data for name,data in captured.items()},input_unchanged=path.read_bytes()==raw,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--charged',type=Path,required=True);p.add_argument('--probe',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.charged,a.probe)
    if not all(r['source_unchanged_since_import'].values()) or not r['input_unchanged']:raise RuntimeError('Source/inputchanged duringactualdressedWard response')
    a.output.write_text(json.dumps(charged.packed(r),indent=2)+'\n');print(json.dumps(charged.packed(dict(input_sha256=r['input_sha256'],physical=r['actual_DRESSED_native_charged_Ward_response'],elapsed_seconds=r['elapsed_seconds'])),indent=2))
