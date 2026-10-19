"""NEW occupied interacting positive-frame consumer, not old-producer replay."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, expm
from rich_frame_holonomy_20261014_model import raw_frame, paths, canonical_c, local, two_site


def norm(a):
    return float(np.linalg.norm(a))


def sectors(model):
    h=model['H']; n=np.rint(np.diag(model['N']).real).astype(int)
    out={}
    for number in sorted(set(n)):
        ix=np.flatnonzero(n==number)
        e,v=eigh(h[np.ix_(ix,ix)])
        full=np.zeros((len(n),len(ix)),complex); full[ix]=v
        out[int(number)]=(e,full)
    return out


def occupied(model,number):
    sec=sectors(model); e,v=sec[number]; ground=v[:,0]; eg=e[0]
    j2=sum(a@a for a in model.get('J',model['S']))
    add={}; rem={}
    ng=len(model['source']); green=np.zeros((ng,ng),complex); car=np.zeros((ng,ng),complex)
    for label,target,sources in [('addition',number+1,model['source']),('removal',number-1,[a.conj().T for a in model['source']])]:
        et,vt=sec[target]
        amp=np.stack([vt.conj().T@a@ground for a in sources],axis=1)
        weights=np.sum(abs(amp)**2,axis=1)
        gram=amp.conj().T@amp if label=='addition' else amp.T@amp.conj()
        car+=gram
        for i in range(len(et)):
            pole=np.outer(amp[i].conj(),amp[i]) if label=='addition' else np.outer(amp[i],amp[i].conj())
            green+=pole/(3+.07j-(et[i]-eg) if label=='addition' else 3+.07j+(et[i]-eg))
        record={'lowest_gap':float(et[0]-eg),'lowest_J2':float(np.vdot(vt[:,0],j2@vt[:,0]).real),'total_weight':float(sum(weights)), 'all_poles':[{'energy':float(et[i]-eg),'weight':float(weights[i])} for i in range(len(et)) if weights[i]>1e-10]}
        if label=='addition': add=record
        else: rem=record
    currents=model['current'] if isinstance(model['current'],list) else [model['current']]
    spectral=[]
    for j in currents:
        amps=v.conj().T@j@ground
        spectral.append({'total_inelastic':float(sum(abs(amps[1:])**2)), 'first_bright_gap':float(e[next((i for i in range(1,len(e)) if abs(amps[i])**2>1e-10),0)]-eg), 'retarded_z_3_07':str(sum(abs(amps[i])**2*(1/(3+.07j-(e[i]-eg))-1/(3+.07j+(e[i]-eg))) for i in range(1,len(e))))})
    return {'E_filled':float(eg),'filled_number':number,'filled_J2':float(np.vdot(ground,j2@ground).real),'addition':add,'removal':rem,'current_spectral':spectral,'source_G_z_3_07':[[str(x) for x in row] for row in green], 'source_CAR_spectral_error':norm(car-np.eye(ng))},sec,ground


def main():
    inp=Path(sys.argv[1]); p=json.loads(inp.read_text()); f=raw_frame(); d=f['source']; eye=np.eye(16)
    cf,u=canonical_c(d)
    source_errors={'CAR':max(norm(a.conj().T@b+b@a.conj().T-(eye if i==j else 0)) for i,a in enumerate(d) for j,b in enumerate(d)), 'Gauss':max(norm(g@f['P']) for g in f['G']), 'bare_Gram':sum(norm(f['P'].conj().T@a@f['P']) for a in f['bare']), 'source_leakage':max(norm(a@f['P']-f['P']@b) for a,b in zip(f['raw_source'],d)),'canonical_unitarity':norm(cf.conj().T@cf-eye),'C_charge':norm(cf@f['N']@cf.conj().T-(4*eye-f['N'])),'C_spin':max(norm(cf@s@cf.conj().T-s) for s in f['S']), 'C_source':max(norm(cf@d[a]@cf.conj().T-sum(u[a,b]*d[b].conj().T for b in range(4))) for a in range(4))}
    s2=sum(s@s for s in f['S'])
    raw_s=[np.kron(a,np.eye(16)) for a in f['left']]
    raw_s2=sum(a@a for a in raw_s)
    raw_path_error=0.
    for axis in range(3):
        r=expm(-1j*p['theta']*raw_s[axis])
        for a,b in zip(f['raw_source'],paths(f,p['theta'])[axis]):
            raw_path_error=max(raw_path_error,norm(f['P'].conj().T@(r@a@r.conj().T-r.conj().T@a@r)@f['P']-b))
    source_errors['raw_path_projection']=raw_path_error
    source_errors['raw_rotation_Casimir']=max(norm(raw_s2@expm(-1j*p['theta']*a)-expm(-1j*p['theta']*a)@raw_s2) for a in raw_s)
    source_errors['bare_Noether']=max(norm(s@b-b@s) for s in raw_s for b in f['bare'])
    source_errors['centre_sewing']=norm(expm(-2j*np.pi*f['S'][2])-np.diag([(-1)**i.bit_count() for i in range(16)]))
    path_records=[]
    for theta,imb in [(p['theta'],0.),(0.,0.),(p['theta'],.23)]:
        br=paths(f,theta,imb); raw_scalar=[]; vector=[]
        # Resolve path source on full CAR source basis. It is not defined as gamma.
        for axis in range(3):
            mat=np.array([[np.trace(d[b].conj().T@br[axis][a])/8 for b in range(4)] for a in range(4)])
            raw_scalar.append([str(np.trace(mat[:2,:2])/2),str(np.trace(mat[2:,2:])/2)])
            vector.append(norm(mat))
        mm=local(f,{**p,'theta':theta},imbalance=imb)
        path_records.append({'theta':theta,'imbalance':imb,'raw_scalar_by_axis_channel':raw_scalar,'path_matrix_norm':vector,'current_norm':[norm(j) for j in mm['current']],'same_halfspin_casimir':list(np.linalg.eigvalsh(s2[np.ix_([1,2,4,8],[1,2,4,8])]))})
    unbalanced=local(f,p,k=(.42,0,0),imbalance=.23)
    balanced=local(f,p,k=(.42,0,0))
    path_records[-1]['unsubtracted_H_contamination']=norm(unbalanced['H']-balanced['H'])
    path_records[-1]['unsubtracted_current_contamination']=norm(unbalanced['current'][0]-balanced['current'][0])
    vac=np.zeros(16,complex); vac[0]=1
    r=expm(-1j*p['theta']*f['S'][0])
    source_errors['rotate_after_annihilation_dark']=max(norm((r-r.conj().T)@a.conj().T@d[b]@vac) for a in d for b in range(4))
    rest=local(f,p)
    field_parity=np.diag([(-1)**sum(n) for n in rest['field']['basis']])
    cfull=np.kron(cf,field_parity)
    rr,sec,g=occupied(rest,2)
    rr.update({'C_H':norm(cfull@rest['H']@cfull.conj().T-rest['H']), 'C_current':[norm(cfull@a@cfull.conj().T+a) for a in rest['current']], 'rest_rotation':max(norm(rest['H']@j-j@rest['H']) for j in rest['J']), 'positive_frame_spectrum':list(np.linalg.eigvalsh(p['frame_cost']*s2)), 'sector_sizes':{str(k):len(v[0]) for k,v in sec.items()}})
    free=local(f,{**p,'feedback':0.}); free_result,_,_=occupied(free,2)
    occupation=np.kron(np.eye(16),rest['field']['H']-1.5*np.eye(rest['nf']))
    rr['field_number']=float(np.vdot(g,occupation@g).real)
    rr['added_field_number']=float(np.vdot(sec[3][1][:,0],occupation@sec[3][1][:,0]).real)
    rr['added_frame_S2']=float(np.vdot(sec[3][1][:,0],np.kron(s2,np.eye(rest['nf']))@sec[3][1][:,0]).real)
    # Inertia is from the actual many-body addition pole, with its own filled state.
    step=.003; plus,_,_=occupied(local(f,p,k=(step,0,0)),2)
    rr['addition_curvature'] = 2*(plus['addition']['lowest_gap']-rr['addition']['lowest_gap'])/step**2
    oddix=[1,2,4,8]
    jbare=1j*(rest['forward'][0]-rest['forward'][0].conj().T)
    v2=float(np.trace((jbare@jbare)[np.ix_(oddix,oddix)]).real/4)
    rr['tree_path_velocity_squared']=v2
    rr['inertial_mass_from_curvature']=1/rr['addition_curvature']
    rr['rest_gap_over_inertial_mass_times_tree_v2']=rr['addition']['lowest_gap']*rr['addition_curvature']/v2
    rr['C_positive_framekin']=norm(cf@s2@cf.conj().T-s2)*p['frame_cost']
    rr['C_Gauss_domain']=max(norm(a@f['P']@cf) for a in f['G'])
    rr['centre_C_preservation']=norm(cf@np.diag([(-1)**i.bit_count() for i in range(16)])-np.diag([(-1)**i.bit_count() for i in range(16)])@cf)
    rr['gap_squared_over_velocity_squared_not_fitted']='No Lorentz identification; actual addition curvature and independently supplied parameters only.'
    chain=two_site(f,p); cr,cs,cg=occupied(chain,4)
    # Exact Fock C on two physical cells, including Jordan-Wigner signs.
    dim=256; cf2=np.zeros((dim,dim),complex)
    ds=[a[:dim*chain['nf']:chain['nf'],:dim*chain['nf']:chain['nf']] for a in chain['source']]
    uu=np.kron(np.eye(2),u); targets=[sum(uu[a,b]*ds[b].conj().T for b in range(8)) for a in range(8)]
    full=np.zeros(dim,complex); full[-1]=1
    for state in range(dim):
        v=full.copy()
        original=np.zeros(dim,complex); original[0]=1
        for a in reversed(range(8)):
            if state&(1<<a):
                v=targets[a]@v
                original=ds[a]@original
        cf2+=np.outer(v,original.conj())
    c2=np.kron(cf2,field_parity)
    cr['C_H']=norm(c2@chain['H']@c2.conj().T-chain['H']); cr['C_current']=norm(c2@chain['current']@c2.conj().T+chain['current'])
    cr['C_unitarity']=norm(c2.conj().T@c2-np.eye(len(c2)))
    cr['C_spin']=max(norm(c2@s@c2.conj().T-s) for s in chain['S'])
    cr['C_charge']=norm(c2@chain['N']@c2.conj().T-(8*np.eye(len(c2))-chain['N']))
    cr['filled_C_overlap']=float(abs(np.vdot(cg,c2@cg)))
    cr['filled_field_number']=float(np.vdot(cg,np.kron(np.eye(256),chain['field']['H']-1.5*np.eye(chain['nf']))@cg).real)
    # A real external source pulse acts on the literal filled state. Own H baseline.
    pulse=expm(-1j*p['pulse']*chain['current'])@cg
    e,v=cs[4]; times=[]
    for t in (0.,.8,2.1):
        evolved=v@(np.exp(-1j*t*e)*(v.conj().T@pulse))
        base=v@(np.exp(-1j*t*e)*(v.conj().T@cg))
        sn2=sum(a@a for a in chain['S']); fn=np.kron(np.eye(256),chain['field']['H']-1.5*np.eye(chain['nf']))
        times.append({'t':t,'current_delta':float((np.vdot(evolved,chain['current']@evolved)-np.vdot(base,chain['current']@base)).real),'frame_S2_delta':float((np.vdot(evolved,sn2@evolved)-np.vdot(base,sn2@base)).real),'field_number_delta':float((np.vdot(evolved,fn@evolved)-np.vdot(base,fn@base)).real),'norm_error':abs(float(np.vdot(evolved,evolved).real)-1)})
    cr['external_source_event']=times
    cr['pulse_absorbed_energy']=float(np.vdot(pulse,chain['H']@pulse).real-cr['E_filled'])
    off=two_site(f,{**p,'theta':0.}); cr['angle_zero_current']=norm(off['current'])
    no_pulse=expm(-0j*chain['current'])@cg
    off_evolved=v@(np.exp(-2.1j*e)*(v.conj().T@no_pulse))
    off_base=v@(np.exp(-2.1j*e)*(v.conj().T@cg))
    cr['source_off_event']=norm(off_evolved-off_base)
    cr['density_current_continuity']=norm(1j*(chain['H']@np.kron(f['N'],np.eye(16*chain['nf']))-np.kron(f['N'],np.eye(16*chain['nf']))@chain['H'])+chain['current'])
    result={'candidate':'physical-frame-conjugated-scalar-source-two-channel-positive-C-fock-v1','input':p,'raw_dimension':224,'physical_dimension':16,'number_dimensions':{str(n):int(sum(np.isclose(np.diag(f['N']).real,n))) for n in range(5)},'source':source_errors,'paths':path_records,'rest':rr,'feedback_off':free_result,'two_site':cr,'limitations':['Finite 2-cell x-edge geometry, no continuum Lorentz or photon; rest is one-cell interacting Fock, not tree Bloch mass.', 'Fundamental colour/CAR/centre sewing and externally driven neutral frame paths supplied.', 'Neutral vector oscillator driving law supplied, no Maxwell B2 credited derived; all regulated states retained.', 'Two site external current pulse is a same-law occupied event, not quantum-photon inclusive scattering.']}
    result['candidate']='physical-frame-conjugated-scalar-source-two-channel-positive-C-current-sourced-fock-v2'
    result['material_successor']='Actual3/4 replace Noether torque S.Q by actual link current J.Q; C includes neutral-vector parity. Actual1/2 were torque comparators, not this law. Actual3 retained old output label; exact source digest and this explicit corrected identity disambiguate.'
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'candidate':result['candidate'],'source':source_errors,'rest_gap':rr['addition']['lowest_gap'],'rest_C':rr['C_H'],'chain_C':cr['C_H'],'pair_weight':cr['current_spectral'][0]['total_inelastic'],'event':times}))

if __name__=='__main__':main()
