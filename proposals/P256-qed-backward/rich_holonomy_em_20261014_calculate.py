"""NEW full eight-mode Fock occupied-G/pair/spatial-field/spin consumer.
All finite virtual states enter resolvents; sparse solves are not spectral truncation.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import eigsh, norm, splu
from rich_holonomy_em_20261014_model import build, sector, evolve, expectation, matter_at, spatial_current


def ground_sector(model,number):
    ix=sector(model,number);block=model['H'][ix][:,ix].tocsc()
    val,vec=eigsh(block,k=2,which='SA',tol=1e-11,v0=np.ones(len(ix)))
    order=np.argsort(val);val=val[order];vec=vec[:,order]
    psi=np.zeros(model['H'].shape[0],complex);psi[ix]=vec[:,0]
    return ix,block,val,psi


def resolvent(block,z,sources):
    matrix=z*eye(block.shape[0],format='csc')-block
    values=splu(matrix.tocsc()).solve(sources)
    residual=float(np.linalg.norm(matrix@values-sources)/max(np.linalg.norm(sources),1e-30))
    return values,residual


def receiver(model,psi,p):
    sa=sum(p['spin_axis'][a]*model['matter_spin'][a] for a in range(3))
    spinval,spinvec=eigh(sa)
    sign=p['pulse_direction']@np.array(model['field']['q'])
    transformed=spinvec.conj().T@psi.reshape(256,model['nf'])
    return [float(np.sum(abs(transformed[np.ix_(spinval>1e-8,sign>0)])**2)),float(np.sum(abs(transformed[np.ix_(spinval>1e-8,sign<=0)])**2)),float(np.sum(abs(transformed[np.ix_(spinval<=1e-8,sign>0)])**2)),float(np.sum(abs(transformed[np.ix_(spinval<=1e-8,sign<=0)])**2))]


def measure(model,psi,p):
    density=abs(psi.reshape(256,model['nf']))**2
    s2=sum(a@a for a in model['matter_spin']);val,vec=eigh(s2)
    proj=vec[:,abs(val-.75)<1e-8]@vec[:,abs(val-.75)<1e-8].conj().T
    half=float(np.linalg.norm(proj@psi.reshape(256,model['nf']))**2)
    nl=float(sum(model['nums'][0][i,i].real*density[i].sum() for i in range(256)))
    return {'current':[expectation(a,psi) for a in model['J']],'edge_current':[expectation(a,psi) for a in model['edgeJ']],'frame_spin':[expectation(a,psi) for a in model['S']],'frame_spin_squared':expectation(model['S2'],psi),'jhalf_weight':half,'Q':[expectation(a,psi) for a in model['Q']],'P':[expectation(a,psi) for a in model['P']],'Q2':[expectation(a@a,psi) for a in model['Q']],'left_number':nl,'left_charge':p['charge']*(nl-2),'receiver':receiver(model,psi,p),'norm':float(np.vdot(psi,psi).real)}


def run(p):
    model=build(p);h=model['H'];dim=h.shape[0];nf=model['nf'];frame=model['frame']
    i4,b4,e4,ground=ground_sector(model,4);e0=e4[0]
    i3,b3,e3,_=ground_sector(model,3);i5,b5,e5,_=ground_sector(model,5)
    d=model['source'];addition=np.column_stack([a@ground for a in d]);removal=np.column_stack([a.conj().T@ground for a in d])
    car=addition.conj().T@addition+(removal.conj().T@removal).T
    greens=[]
    for re,im in p['frequencies']:
        z=re+1j*im
        xp,rp=resolvent(b5-e0*eye(len(i5)),z,addition[i5])
        xm,rm=resolvent(-(b3-e0*eye(len(i3))),z,removal[i3])
        g=addition[i5].conj().T@xp+(removal[i3].conj().T@xm).T
        greens.append({'z':[re,im],'real':g.real.tolist(),'imag':g.imag.tolist(),'residual_add':rp,'residual_remove':rm})
    cf=model['CF'];external=(.21,-.14)
    plus=build(p,frame,external=external);minus=build(p,frame,external=(-external[0],-external[1]))
    covariance=float(norm(cf@plus['H']@cf.conj().T-minus['H']))
    gauss=max(float(np.linalg.norm(a@frame['P'])) for a in frame['G'])
    sourceleak=max(float(np.linalg.norm((np.eye(224)-frame['P']@frame['P'].conj().T)@a@frame['P'])) for a in frame['raw_source'])
    delta=2e-4;derivatives=[]
    for a in range(2):
        ext=np.eye(2)[a]*delta;hp=build(p,frame,external=ext)['H'];hm=build(p,frame,external=-ext)['H']
        derivatives.append({'axis':a,'J_minus_dH_dA':float(norm((hp-hm)/(2*delta)+model['J'][a])),'K_d2H_dA2':float(norm((hp+hm-2*h)/delta**2-model['K'][a][a]))})
    nl=kron(csr_matrix(model['nums'][0]),eye(nf),format='csr')
    edge_continuity=float(norm(1j*(h@nl-nl@h)+sum(model['edgeJ'])))
    spatial=[]
    for wave in [(0.,0.,0.),(.37,-.29,.41),(1.1,.6,-.8)]:
        k=np.asarray(wave);form=np.exp(-p['charge_width']**2*(k@k)/2)
        rho=p['charge']*sum(form*np.exp(-1j*k@centre)*kron(csr_matrix(number-2*np.eye(256)),eye(nf),format='csr') for centre,number in zip(model['geometry']['centres'],model['nums']))
        j=spatial_current(model,k)
        spatial.append({'wavevector':list(wave),'continuity':float(norm(1j*(h@rho-rho@h)+1j*sum(k[a]*j[a] for a in range(3)))),'J_norms':[float(norm(a)) for a in j]})
    frozen=build(p,frame,induction=0.)
    frozen_difference=[float(norm(model['J'][a]-frozen['J'][a])) for a in range(2)]
    # Complete 70-state occupied matter calculation, diagnostic only, no Hessian replacement.
    im=np.array([left*16+right for left in range(16) for right in range(16) if left.bit_count()+right.bit_count()==4])
    def material_energy(q):
        return eigh(matter_at(model,np.array(q))[np.ix_(im,im)],eigvals_only=True)[0]
    eps=.002;zero=material_energy([0,0]);curvature=np.empty((2,2))
    for a in range(2):
        shift=np.eye(2)[a]*eps
        curvature[a,a]=(material_energy(shift)+material_energy(-shift)-2*zero)/eps**2
    curvature[0,1]=curvature[1,0]=sum(a*b*material_energy(np.array([a,b])*eps) for a in (-1,1) for b in (-1,1))/(4*eps**2)
    amplitudes={label:np.column_stack([(a@ground)[i4] for a in ops]) for label,ops in [('JJ',model['J']),('QQ',model['Q'])]}
    responses=[]
    for re,imfreq in p['frequencies']:
        z=re+1j*imfreq;tensors={}
        for label,amps in amplitudes.items():
            xp,rp=resolvent(b4-e0*eye(len(i4)),z,amps)
            xm,rm=resolvent(-(b4-e0*eye(len(i4))),z,amps)
            tensor=amps.conj().T@xp-(amps.conj().T@xm).T
            tensors[label]={'real':tensor.real.tolist(),'imag':tensor.imag.tolist(),'residuals':[rp,rm]}
        responses.append({'z':[re,imfreq],**tensors})
    source=d[0]+(.4+.3j)*d[7];added=source@ground;source_norm=float(np.linalg.norm(added));added/=source_norm
    # Honest prepared family: physical total jhalf projection; higher virtual spins remain H.
    s2=sum(a@a for a in model['matter_spin']);spinval,spinvec=eigh(s2);w=spinvec[:,abs(spinval-.75)<1e-8]
    projected=w@(w.conj().T@added.reshape(256,nf));projection_weight=float(np.linalg.norm(projected)**2)
    added=projected.ravel()/np.linalg.norm(projected)
    direction=np.array(p['pulse_direction']);qdrive=sum(direction[a]*model['Q'][a] for a in range(2));jdrive=sum(direction[a]*model['J'][a] for a in range(2))
    field_prepared=evolve(qdrive,added,p['field_pulse']);current_prepared=evolve(jdrive,added,p['current_pulse'])
    events={};states={'reference':ground,'added':added,'field_prepared':field_prepared,'current_prepared':current_prepared}
    for label,prepared in [('field',field_prepared),('current',current_prepared),('undriven',added)]:
        records=[]
        for time in p['times']:
            psi=evolve(h,prepared,time);records.append({'time':time,**measure(model,psi,p)});states[f'{label}_t{time}']=psi
        events[label]=records
    # Actual occupied particle-hole promotion, not a charge-sector relabelled positron.
    one=np.array([left*16+right for left in range(16) for right in range(16) if left.bit_count()+right.bit_count()==1])
    ev,uv=eigh(matter_at(model,np.zeros(2))[np.ix_(one,one)])
    upper=uv[:,4:]@uv[:,4:].conj().T
    # Source mode basis is left modes first, whereas tensor bit order has right first.
    order=np.array([one.tolist().index(16*(1<<a)) for a in range(4)]+[one.tolist().index(1<<a) for a in range(4)])
    upper=upper[np.ix_(order,order)]
    promoted=sum(upper[a,b]*model['matter_sources'][a]@model['matter_sources'][b].conj().T for a in range(8) for b in range(8))
    promoted=kron(csr_matrix(promoted),eye(nf),format='csr');baseline=expectation(promoted,ground)
    pair=evolve(jdrive,ground,p['current_pulse']);pairrecords=[]
    for time in p['times']:
        psi=evolve(h,pair,time)
        pairrecords.append({'time':time,'N':expectation(model['N'],psi),'upper_occupation':expectation(promoted,psi),'increase':expectation(promoted,psi)-baseline,'current':[expectation(a,psi) for a in model['J']]})
    controls={}
    for label,changes in [('induction_off',{'hop':0.}),('angle_zero',{'theta':0.})]:
        pp={**p,**changes};off=build(pp,frame);_,_,oe,og=ground_sector(off,4)
        oa=off['source'][0]@og;oa/=np.linalg.norm(oa)
        drive=sum(direction[a]*off['Q'][a] for a in range(2));fp=evolve(drive,oa,p['field_pulse'])
        last=evolve(off['H'],fp,p['times'][-1]);base=evolve(off['H'],oa,p['times'][-1])
        controls[label]={'J_norms':[float(norm(a)) for a in off['J']],'contact_norms':[float(norm(off['K'][a][a])) for a in range(2)],'ground_Q2':[expectation(a@a,og) for a in off['Q']],'ground_energy':float(oe[0]),'lowest_gap':float(oe[1]-oe[0]),'spin_field_minus_undriven':[expectation(a,last)-expectation(a,base) for a in off['S']],'last':measure(off,last,pp)}
    controls['current_source_off']={'pair_upper_occupation':baseline,'pair_increase':0.,'current':[expectation(a,ground) for a in model['J']]}
    density=abs(ground.reshape(256,p['field_grid'],p['field_grid']))**2
    boundary=float(density[:,[0,-1],:].sum()+density[:,:,[0,-1]].sum())
    result={'candidate':p['candidate'],'scope':p['scope'],'dimensions':{'raw_frame_colour_per_site':224,'physical_fullFock':256,'field':nf,'whole':dim,'sectors':{str(n):len(sector(model,n)) for n in range(9)}},'geometry':{'centres':model['geometry']['centres'].tolist(),'line_integrals':model['geometry']['line'].tolist(),'rank':int(np.linalg.matrix_rank(model['geometry']['line'])),'momentum_rms':model['geometry']['momentum_rms'],'coulomb_kernel':model['geometry']['coulomb'].tolist(),'Gaussian_charge_integrals':(model['geometry']['density'].sum(axis=(1,2,3))*model['geometry']['dv']).tolist()},'source_symmetry':{'raw_Gauss':gauss,'raw_source_leakage':sourceleak,'C_covariance_nonzero_A':covariance,'C_same_A_not_required':float(norm(cf@plus['H']@cf.conj().T-plus['H'])),'CAR':float(np.linalg.norm(car-np.eye(8)))},'whole_current':{'convention':'J=-dH/dA; K=d²H/dA²=-dJ/dA; edgeJ is particle flux left→right','derivatives':derivatives,'edge_continuity':edge_continuity,'spatial_fourier':spatial,'frozen_A0_difference':frozen_difference,'current_gram':(amplitudes['JJ'].conj().T@amplitudes['JJ']).real.tolist(),'field_gram':(amplitudes['QQ'].conj().T@amplitudes['QQ']).real.tolist(),'contact_expectation':[[expectation(a,ground) for a in row] for row in model['K']]},'induction':{'primitive_E_only':True,'material_zero_energy':float(zero),'material_curvature':curvature.tolist(),'material_curvature_eigenvalues':eigh(curvature,eigvals_only=True).tolist(),'curvature_not_replacement_H':True,'ground_Q2':[expectation(a@a,ground) for a in model['Q']],'outer_rows_weight':boundary},'occupied_G':{'reference_N':4,'add_N':5,'remove_N':3,'charges':[0,-1,1],'addition_norms':np.diag(addition.conj().T@addition).real.tolist(),'removal_norms':np.diag(removal.conj().T@removal).real.tolist(),'addition_gap':float(e5[0]-e0),'removal_gap':float(e3[0]-e0),'complete_finite_resolvent':greens},'full_response':responses,'preparation':{'source_norm':source_norm,'jhalf_source_projection_weight':projection_weight,'finite_prepared_family':'Project added actual source onto total physical LEFT-frame j=1/2, then FIELD-only pulse. All higher-j virtual states retained in H.'},'events':events,'pair_receiver':{'definition':'Upper four actual one-particle matter orbitals occupation in N4; particle-hole promotion, not QED positron.','ground_upper':baseline,'records':pairrecords},'controls':controls,'unpaid':['Primitive electric metric, colour/CAR/centre and ordered paths/geometry are inputs.','Two fixed Gaussian sites and two collective profiles, not free COM or autonomous spatial continuum photon.','Higher total spin can mix under fixed anisotropic geometry; initial jhalf family not proof every dressed eigenstate is jhalf.','Off-case finite coordinate gap is Dirichlet regulator; unbounded vacuum/bulk density/Lorentz gap-inertia not closed.']}
    return result,states,model


def main():
    root=Path(__file__).resolve().parent
    path=Path(sys.argv[1]) if len(sys.argv)>1 else root/'rich_holonomy_em_20261014_input.json'
    result,states,model=run(json.loads(path.read_text()))
    stem=Path(sys.argv[2]) if len(sys.argv)>2 else root/'rich_holonomy_em_20261014_actual3'
    Path(str(stem)+'_result.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(str(stem)+'_states.npz',**states,profile=model['geometry']['fields'],spatial_axis=model['geometry']['axis'],density=model['geometry']['density'],longitudinal_electric=model['geometry']['electric'],field_coordinates=np.array(model['field']['q']))
    print(json.dumps({key:result[key] for key in ['dimensions','source_symmetry','induction','preparation']}))
    print(json.dumps({'field_last':result['events']['field'][-1],'undriven_last':result['events']['undriven'][-1],'pair_last':result['pair_receiver']['records'][-1]}))

if __name__=='__main__':
    main()
