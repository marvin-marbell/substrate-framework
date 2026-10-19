"""New literal occupied lattice physics consumer; every finite pole retained."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye
from scipy.sparse.linalg import eigsh, expm_multiply, spsolve
from rich_spatial_holonomy_20261014_model import build, momentum_basis, fourier_source


def norm(a):
    if hasattr(a, 'data') and not isinstance(a,np.ndarray):
        return float(np.linalg.norm(a.data))
    return float(np.linalg.norm(a))


def expect(a,v):
    return float(np.vdot(v,a@v).real)


def evolve(h,v,t):
    return expm_multiply(-1j*t*h,v,traceA=-1j*t*h.diagonal().sum())


def serialize_complex(a):
    return [[[float(x.real),float(x.imag)] for x in row] for row in np.atleast_2d(a)]


def diagonalize(model,number):
    blocks=[]
    for m in range(model['parameters']['sites']):
        u=momentum_basis(model,number,m)
        block=(u.conj().T@model['H']@u).toarray()
        e,v=eigh(block,overwrite_a=True,check_finite=False)
        blocks.append({'m':m,'U':u,'e':e,'v':v})
    return blocks


def run(p):
    model=build(p); h=model['H']; sites=p['sites']; sea=2*sites
    sectors={n:diagonalize(model,n) for n in (sea-1,sea,sea+1)}
    eg=min(b['e'][0] for b in sectors[sea])
    references=[]; reference_m=[]
    for b in sectors[sea]:
        for j in np.flatnonzero(abs(b['e']-eg)<1e-8):
            references.append(b['U']@b['v'][:,j]); reference_m.append(b['m'])
    refs=np.column_stack(references); count=len(references); ground=references[0]
    f=model['frame']; d=f['source']; cf=model['CF']
    raw={'Gauss_P':max(norm(a@f['P']) for a in f['G']), 'positive_frame_eigenvalues':np.linalg.eigvalsh(sum(a@a for a in f['S'])).tolist(), 'source_CAR':max(norm(a.conj().T@b+b@a.conj().T-(np.eye(16) if i==j else 0)) for i,a in enumerate(d) for j,b in enumerate(d)), 'bare_source_physical_gram':max(norm(f['P'].conj().T@a@f['P']) for a in f['bare'])}
    raw['source_spin_adjoint_matrices']=[serialize_complex(np.array([[np.trace(d[b].conj().T@(s@d[a]-d[a]@s))/np.trace(d[b].conj().T@d[b]) for b in range(4)] for a in range(4)])) for s in f['S']]
    raw['source_C_intertwining']=max(norm(cf@model['source'][4*j+a]@cf.conj().T-sum(model['source_C'][a,b]*model['source'][4*j+b].conj().T for b in range(4))) for j in range(sites) for a in range(4))
    raw['physical_spin_C_plus_errors']=[norm(cf@s@cf.conj().T-s) for s in model['S']]
    raw['physical_spin_C_minus_errors']=[norm(cf@s@cf.conj().T+s) for s in model['S']]
    raw['C_H_zero']=norm(cf@h@cf.conj().T-h)
    external=.13; hp=build(p,external=external,frame=f)['H']; hm=build(p,external=-external,frame=f)['H']
    raw['C_H_external_source_flip']=norm(cf@hp@cf.conj().T-hm)
    raw['neutral_reference_C_projector_error']=norm(cf@refs-refs@(refs.conj().T@(cf@refs)))
    raw['neutral_reference_degeneracy']=count
    raw['translation_commutator']=norm(model['T']@h-h@model['T'])
    raw['translation_cube']=norm(model['T']@model['T']@model['T']-eye(h.shape[0]))
    raw['continuity']=[norm(1j*(h@n-n@h)-(model['edgeJ'][(j-1)%sites]-model['edgeJ'][j])) for j,n in enumerate(model['nums'])]
    delta=1e-5; hdplus=build(p,external=delta,frame=f)['H']; hdminus=build(p,external=-delta,frame=f)['H']
    raw['current_A_derivative']=norm((hdplus-hdminus)/(2*delta)+model['J'])
    raw['contact_A_derivative']=norm((hdplus-2*h+hdminus)/delta**2-model['K'])
    z=complex(*p['frequency']); spectral={}; matrices={}; car_errors=[]
    for momentum in range(sites):
        sources=fourier_source(model,momentum); green=np.zeros((4,4),complex); car=np.zeros((4,4),complex); entries={}
        for label,number,operators in [('addition',sea+1,sources),('removal',sea-1,[a.conj().T for a in sources])]:
            poles=[]; total=0.; first=np.inf; weighted_energy=0.
            for b in sectors[number]:
                amplitudes=[]
                for reference in references:
                    columns=np.column_stack([a@reference for a in operators])
                    amplitudes.append(b['v'].conj().T@(b['U'].conj().T@columns))
                weight=sum(np.sum(abs(a)**2,axis=1) for a in amplitudes)/count
                gaps=b['e']-eg
                for i in range(len(gaps)):
                    matrix=sum(np.outer(a[i].conj(),a[i]) if label=='addition' else np.outer(a[i],a[i].conj()) for a in amplitudes)/count
                    car+=matrix; green+=matrix/(z-gaps[i] if label=='addition' else z+gaps[i])
                    # No weight cutoff: dark states and high-field states stay in the capsule.
                    poles.append({'target_m':b['m'],'energy_relative_same_sea':float(gaps[i]),'weight':float(weight[i])})
                bright=np.flatnonzero(weight>1e-10)
                if len(bright): first=min(first,float(gaps[bright[0]]))
                total+=float(sum(weight)); weighted_energy+=float(weight@gaps)
            entries[label]={'total_weight':total,'first_bright_gap':first,'energy_weighted_sum':weighted_energy,'all_poles':poles}
        spectral[str(momentum)]=entries; matrices[str(momentum)]=serialize_complex(green)
        car_errors.append(norm(car-np.eye(4)))
    pair_poles=[]; pair_response=0j; total_pair=0.
    for b in sectors[sea]:
        weights=np.zeros(len(b['e']))
        for reference in references:
            amp=b['v'].conj().T@(b['U'].conj().T@(model['J']@reference)); weights+=abs(amp)**2/count
        for gap,weight in zip(b['e']-eg,weights):
            pair_poles.append({'target_m':b['m'],'gap':float(gap),'weight':float(weight)})
            if gap>1e-8:
                total_pair+=float(weight); pair_response+=weight*(1/(z-gap)-1/(z+gap))
    dispersion={str(n):[{'m':b['m'],'k':2*np.pi*b['m']/model['geometry']['circumference'],'E_ground':float(b['e'][0]),'gap_same_reference':float(b['e'][0]-eg)} for b in sectors[n]] for n in (sea-1,sea+1)}
    phi=p['twist_step']; twist_data={}
    for sign in (-1,1):
        mt=build(p,twist=sign*phi,frame=f); twist_data[sign]={}
        for n in (sea-1,sea,sea+1):
            values=[]
            for b in sectors[n]:
                block=b['U'].conj().T@mt['H']@b['U']
                e=eigsh(block,k=1,which='SA',v0=b['v'][:,0],tol=2e-11,return_eigenvectors=False)
                values.append(float(e[0]))
            twist_data[sign][n]=values
    curvature=[]; length=model['geometry']['circumference']
    for n in (sea-1,sea+1):
        for b in sectors[n]:
            m=b['m']; minus=twist_data[-1][n][m]-min(twist_data[-1][sea]); plus=twist_data[1][n][m]-min(twist_data[1][sea]); zero=float(b['e'][0]-eg)
            d2=(plus+minus-2*zero)/phi**2*length**2/p['charge']**2
            curvature.append({'N':n,'m':m,'same_reference_gap_minus':minus,'gap_zero':zero,'same_reference_gap_plus':plus,'inverse_inertia_flux_curvature':d2,'inertia_if_positive':1/d2 if d2>0 else None,'one_sided_mechanical_slopes':[(zero-minus)/phi*length,(plus-zero)/phi*length]})
    indices=np.flatnonzero(np.repeat([state.bit_count()==sea+1 for state in range(model['matter'])],model['nf']))
    hc=h[indices][:,indices]; jc=model['J'][indices][:,indices]
    seed=(model['source'][0]+(.23+.31j)*model['source'][5])@ground; seed/=np.linalg.norm(seed)
    seed_charge=seed[indices]; q=model['Q'][indices][:,indices]
    kicked=evolve(q,seed_charge,p['pulse'])
    packet={'undriven':[],'field_only_pulse':[]}; operators={'density':model['nums'],'spin':model['S']}
    for label,initial in [('undriven',seed_charge),('field_only_pulse',kicked)]:
        for t in p['times']:
            local=evolve(hc,initial,t); vector=np.zeros(h.shape[0],complex); vector[indices]=local
            densities=[expect(a,vector)-sum(expect(a,r) for r in references)/count for a in operators['density']]
            field_probability=np.sum(abs(vector.reshape(model['matter'],model['nf']))**2,axis=0)
            packet[label].append({'time':t,'excess_number_by_site':densities,'spin':[expect(a,vector) for a in operators['spin']],'S2':expect(model['S2'],vector),'Q':expect(model['Q'],vector),'P':expect(model['P'],vector),'field_probability':field_probability.tolist(),'current':expect(model['J'],vector),'edge_flux':[expect(a,vector) for a in model['edgeJ']],'electric_energy':expect(model['electric'],vector),'energy':expect(h,vector),'norm':float(np.linalg.norm(vector))})
    columns=np.column_stack([(a@ground)[indices] for a in fourier_source(model,0)])
    resolvent=z*eye(len(indices),format='csc')-(hc-eg*eye(len(indices),format='csr'))
    solved=spsolve(resolvent,columns)
    inserted=spsolve(resolvent,jc@solved)
    threepoint=columns.conj().T@inserted
    source_current={'addition_m0_RJR':serialize_complex(threepoint),'residual':norm(resolvent@solved-columns),'insertion_residual':norm(resolvent@inserted-jc@solved),'reference':'first actual N6 ground vector, no jhalf filter, all N7 states retained'}
    spin_half_initial=expect(model['S2'],seed)
    result={'candidate':p['candidate'],'parameters':p,'reference':{'Nsea':sea,'E':float(eg),'momenta':reference_m,'degeneracy':count,'ensemble':'equal ground-manifold mixture for spectral G/pair; first pure member for mobile event and threepoint','S2_ensemble':sum(expect(model['S2'],r) for r in references)/count},'raw_algebra_and_same_H':raw,'source_G':matrices,'source_CAR_errors':car_errors,'source_spectra_all_finite_poles':spectral,'occupied_pair_current':{'total_inelastic':total_pair,'retarded':serialize_complex(pair_response),'all_poles':pair_poles,'contact_ensemble':sum(expect(model['K'],r) for r in references)/count},'actual_allowed_momentum_dispersion':dispersion,'actual_flux_inertial_curvature':curvature,'twist_raw_energies':{str(s):{str(n):v for n,v in data.items()} for s,data in twist_data.items()},'source_current_threepoint':source_current,'unfiltered_source_initial_S2':spin_half_initial,'mobile_events':packet,'field_boundary_probability_reference':sum(np.sum(abs(r.reshape(model['matter'],model['nf']))[:,[0,-1]]**2) for r in references)/count,'scope':p['scope']}
    states={'reference':refs,'source_packet':seed,'field_kicked_N7':kicked,'N7_indices':indices}
    return result,states,model


def main():
    path=Path(sys.argv[1]); p=json.loads(path.read_text()); result,states,_=run(p)
    stem=Path(sys.argv[2]); stem.with_name(stem.name+'_result.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(stem.with_name(stem.name+'_states.npz'),**states)
    print(json.dumps({'candidate':result['candidate'],'reference':result['reference'],'dispersion':result['actual_allowed_momentum_dispersion'],'curvature':result['actual_flux_inertial_curvature'],'CAR':result['source_CAR_errors'],'source_C':result['raw_algebra_and_same_H']['source_C_intertwining'],'C_H':result['raw_algebra_and_same_H']['C_H_zero'],'pair_weight':result['occupied_pair_current']['total_inelastic']}))

if __name__=='__main__':
    main()
