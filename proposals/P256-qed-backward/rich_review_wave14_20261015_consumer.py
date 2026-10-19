"""Wave14 independent finite consumers; only NEW gauge/common-quantum laws."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply


def norm(a):
    return float(sparse.linalg.norm(a) if sparse.issparse(a) else np.linalg.norm(a))


def mean(v,a):
    return float(np.vdot(v,a@v).real)


def ground(h):
    e,v=eigsh(h,k=1,which='SA',tol=1e-11,maxiter=30000)
    return float(e[0]),v[:,0]


def gauge(request):
    from rich_rotcov_gauge_20261015_model import build,london_profiles
    from rich_rotcov_induced_20261015_model import oscillator,quadrature,comm
    p=json.loads(Path(request['owner_input']).read_text());p.update(request.get('parameter_overrides',{}))
    print(json.dumps(dict(stage='gauge_build',parameters=p)),flush=True)
    m=build(p);h=m['H'];dims=m['dims'];nf=dims[-1]
    print(json.dumps(dict(stage='gauge_built',dimension=h.shape[0])),flush=True)
    n=m['N'].diagonal().real;jz=m['J'][2].diagonal().real
    idx=np.flatnonzero((abs(n-2)<1e-8)&(abs(jz)<1e-8));e,g=ground(h[idx][:,idx]);v=np.zeros(h.shape[0],complex);v[idx]=g
    print(json.dumps(dict(stage='gauge_ground',energy=e,residual=norm(h@v-e*v))),flush=True)
    raw=m['sources'][0]@v;strength=norm(raw)**2;s=raw/np.sqrt(strength)
    initial=expm_multiply(-.19j*m['PX'][0],s);initial=expm_multiply(.13j*m['X'][1],initial)
    kick=expm_multiply(-.29j*(m['Q'][2]+.4*m['Q'][4]),initial)
    bh=m['born_huang'];body=int(np.prod(dims[:3]));bhop=p['charge']**2/2*sparse.kron(sparse.eye(body),sparse.kron(sparse.csr_matrix(bh),sparse.eye(nf)))
    records=[];states=[]
    mechanical=m['supplier_energy']+m['gauge_matter_coupling']
    qvals,u=np.linalg.eigh(m['field']['Q'][2]);bodyj=jz.reshape(-1,nf)[:,0]-sum(f[2][0,0].real for f in m['field']['F'])
    for t in request['times']:
        a=expm_multiply(-1j*t*h,kick);b=expm_multiply(-1j*t*h,initial);states.append(a)
        # Qz and Jz simultaneous diagonalization, not separate subsystem angular PVMs.
        fz=sum(f[2] for f in m['field']['F']);_,u=np.linalg.eigh(m['field']['Q'][2]+11*fz)
        qvalues=np.diag(u.conj().T@m['field']['Q'][2]@u).real;fvalues=np.diag(u.conj().T@fz@u).real
        probs=abs(a.reshape(-1,nf)@u.conj())**2;probb=abs(b.reshape(-1,nf)@u.conj())**2
        bins={};base={}
        for i,k in np.ndindex(probs.shape):
            key=f'{bodyj[i]+fvalues[k]:.6f},{qvalues[k]:.6f}'
            bins[key]=bins.get(key,0.)+float(probs[i,k]);base[key]=base.get(key,0.)+float(probb[i,k])
        records.append(dict(time=t,norm=norm(a),energy_drift=mean(a,h)-mean(kick,h),radius_excess=mean(a,m['radius'])-mean(b,m['radius']),supplier_excess=mean(a,m['supplier_energy'])-mean(b,m['supplier_energy']),direct_current_excess=[mean(a,x)-mean(b,x) for x in m['current']],J_drift=[mean(a,x)-mean(kick,x) for x in m['J']],joint_probability=sum(bins.values()),joint_total_variation=sum(abs(bins[k]-base[k]) for k in bins)/2))
        records[-1]['supplier_coordinate_bare_energy_excess']=records[-1].pop('supplier_excess')
        records[-1]['supplier_invariant_r2_excess']=[mean(a,x)-mean(b,x) for x in m['supplier_r2']]
        records[-1]['supplier_physical_minimal_energy_excess']=mean(a,mechanical)-mean(b,mechanical)
    wrong=expm_multiply(-1j*request['times'][-1]*(h-bhop),kick)
    print(json.dumps(dict(stage='gauge_event_completed')),flush=True)
    osc=oscillator(p['geometry_cut']);ext=oscillator(p['geometry_cut']+2);slots=[ext['basis'].index(x) for x in osc['basis']]
    exact=sum(x@x for x in ext['P'])[np.ix_(slots,slots)]/(2*p['reduced_mass']*p['length']**2)
    quad=quadrature(osc,p['length'],p['radial_quadrature']);kin=quad['u']@quad['kinetic'](p['reduced_mass'])@quad['u'].conj().T
    # Independently build fixed-original-A derivatives from continuum integrals.
    currents=[sparse.csr_matrix(h.shape,dtype=complex) for _ in range(6)];contacts=[x.copy() for x in currents]
    for f,qq in enumerate(m['supplier_quadratures']):
        original,radial,chi,grad=london_profiles(qq['points'],p);li=qq['linear'](original,p['supplier_masses'][f],-p['charge']);cross=p['charge']**2/p['supplier_masses'][f]*qq['integrate'](np.einsum('xai,xbi->xab',original,radial));ct=p['charge']**2/p['supplier_masses'][f]*qq['integrate'](np.einsum('xai,xbi->xab',original,original))
        def embed(x,field):
            pair=np.kron(x,np.eye(4)) if f==0 else np.kron(np.eye(4),x)
            return sparse.kron(sparse.eye(body),sparse.kron(sparse.csr_matrix(pair),sparse.csr_matrix(field)))
        for a in range(6):
            currents[a]-=embed(li[a],np.eye(nf));contacts[a]+=embed(ct[:,:,a,a],np.eye(nf))
            for b in range(6):currents[a]-=embed(cross[:,:,a,b],m['field']['Q'][b])
    # Core contribution is obtained from the pure constructor with supplier off.
    from rich_rotcov_gauge_20261015_primitive import build as primitive
    from rich_rotcov_gauge_20261015_sparse import SparseRoundoff
    core=primitive(p,SparseRoundoff())
    derivative=[norm(currents[a]+core['current'][a]-m['current'][a]) for a in range(6)]
    contact=[norm(contacts[a]+core['contact'][a][a]-m['contact'][a][a]) for a in range(6)]
    chiops=[sparse.kron(sparse.eye(body),sparse.kron(sparse.csr_matrix(c),sparse.eye(nf))) for c in m['chi_total']]
    result=dict(kind='INDEPENDENT_NEW_GAUGE_PHYSICAL_CONSUMER',dimension=h.shape[0],neutral_energy=e,neutral_eigen_residual=norm(h@v-e*v),source_strength=strength,source_J2=mean(s,sum(x@x for x in m['J'])),moving_J2=mean(initial,sum(x@x for x in m['J'])),source_same_fibre_commutators=[norm(comm(m['sources'][0],x)) for x in chiops],source_CAR=norm(sum(x@x.conj().T+x.conj().T@x for x in m['sources'])-4*sparse.eye(h.shape[0])),gradient_p2_defect=norm(kin-exact),born_huang_min=float(np.linalg.eigvalsh(bh).min()),born_huang_norm=norm(bh),omit_BH_final_distance=norm(wrong-states[-1]),independent_current_assembly_errors=derivative,independent_contact_assembly_errors=contact,current_covariant_ground_defects=[norm((a-b)@v) for a,b in zip(m['current'],m['covariant_current'])],H_J=[norm(comm(h,x))/norm(h) for x in m['J']],longitudinal_cross_norm=norm(m['longitudinal_cross']),joint_commutator=norm(comm(m['J'][2],m['Q'][2])),field_kick_t0_radius_change=mean(kick,m['radius'])-mean(initial,m['radius']),samples=records,limits='One retained continuum-gradient/London law, not Maxwell or continuum Ward closure. Supplier-only same-fibre source closure does not prove arbitrary local-gauge covariance.')
    result['sparsity_roundoff_ledger']=m['sparse_roundoff']
    result['parameters']=p
    return result,dict(neutral=v,source=s,initial=initial,kicked=kick,event=np.array(states))


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());result,states=gauge(request)
    out.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),**states);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
