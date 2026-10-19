from __future__ import annotations
import argparse,itertools,json,math
from typing import Any,Callable,cast
import numpy as np
from scipy.special import eval_genlaguerre,gammaln
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator,eigsh,cg,expm_multiply
from original_charged_principal import original_frame
from rotation_band import RotationBand
from sea_selection import SELECTED


def accept(error,label,tol=2e-8):
    value=float(error)
    if not math.isfinite(value) or value>tol:raise AssertionError(f'{label}: {value} > {tol}')
    return value


def adverse(error,label,tol=2e-8):
    try:accept(error,label,tol)
    except AssertionError as exc:return str(exc)
    raise AssertionError('Wrong construction passed SAME assertion: '+label)


def exterior(h,rows):
    lookup={row:i for i,row in enumerate(rows)}
    result=np.zeros((len(rows),len(rows)),complex)
    for col,row in enumerate(rows):
        for at,old in enumerate(row):
            rest=list(row);rest.pop(at)
            for new in range(8):
                if new in rest:continue
                target=tuple(sorted(rest+[new]))
                result[lookup[target],col]+=(-1)**(at+sum(k<new for k in rest))*h[new,old]
    return result


def displacement(n,s):
    # Exact matrix elements of exp(i*s*(a+a†)) BEFORE Galerkin compression.
    result=np.empty((n,n),complex)
    for m in range(n):
        for k in range(n):
            lo=min(m,k);hi=max(m,k)
            result[m,k]=math.exp(-s*s/2+(gammaln(lo+1)-gammaln(hi+1))/2)*(1j*s)**(hi-lo)*eval_genlaguerre(lo,hi-lo,s*s)
    return result


class HermitianAction(LinearOperator):
    def __init__(self,size,action):
        self.apply=action
        super().__init__(np.dtype(np.complex128),(size,size))

    def _matvec(self,x):return self.apply(x)

    def _rmatvec(self,x):return self.apply(x)


class Coupled:
    def __init__(self,n,M,zeta,occupation,UE,K,frame,external=None):
        self.n=n;self.M=M;self.zeta=zeta;self.UE=UE;self.K=K
        self.occupation=occupation;self.frame=frame
        self.external=np.zeros(3) if external is None else np.asarray(external,float)
        self.rows:tuple[tuple[int,...],...]=tuple(itertools.combinations(range(8),occupation))
        self.d=len(self.rows);self.ell=np.arange(-M,M+1);self.l=len(self.ell)
        self.phdim=n*self.l;self.size=self.phdim*self.d
        self.nx=np.array([sum(k<4 for k in row) for row in self.rows],float)
        self.omega=2*math.sqrt(UE*K)
        self.phenergy=(self.omega*(np.arange(n)[:,None]+.5)+UE*(self.ell[None,:]+zeta)**2/6).ravel()
        self.coulomb=UE*(self.nx-2)**2/3
        self.onsite=exterior(np.kron(np.eye(2),(.2+3*.12)*frame[0]),self.rows)
        self.hops=[];self.phases=[]
        f=np.array([1.,-1.,0.])/math.sqrt(2)
        for edge,nu in enumerate([1,1,-2]):
            hop=np.zeros((8,8),complex)
            hop[:4,4:]=-(.12*frame[0]+1j*.3*frame[edge+1])/2
            self.hops.append(exterior(hop,self.rows))
            shift=np.zeros((self.l,self.l),complex)
            for i,e in enumerate(self.ell):
                target=int(e+nu+M)
                if 0<=target<self.l:shift[target,i]=1
            self.phases.append(csr_matrix(np.exp(1j*self.external[edge])*np.kron(displacement(n,f[edge]*math.sqrt(UE/self.omega)),shift)))
        self.operator=HermitianAction(self.size,self.action)

    def reshape(self,v):return np.asarray(v).reshape(self.phdim,self.d)

    def action(self,v):
        wave=self.reshape(v)
        out=(self.phenergy[:,None]+self.coulomb[None,:])*wave+wave@self.onsite.T
        for Z,T in zip(self.phases,self.hops):out+=Z@wave@T.T+Z.conj().T@wave@T.conj()
        return out.ravel()

    def current(self,v,weights):
        wave=self.reshape(v);out=np.zeros_like(wave,dtype=complex)
        for w,Z,T in zip(weights,self.phases,self.hops):out+=1j*w*(Z@wave@T.T-Z.conj().T@wave@T.conj())
        return out.ravel()

    def contact(self,v,first,second):
        wave=self.reshape(v);out=np.zeros_like(wave,dtype=complex)
        for w,u,Z,T in zip(first,second,self.phases,self.hops):out-=w*u*(Z@wave@T.T+Z.conj().T@wave@T.conj())
        return out.ravel()

    def solve(self,k):
        solver=cast(Callable[...,Any],eigsh)
        seed=np.random.default_rng(2221007+self.occupation).normal(size=self.size)
        values,vectors=solver(self.operator,k=k,which='SA',tol=2e-11,v0=seed)
        order=np.argsort(values);values=values[order];vectors=vectors[:,order]
        residuals=[float(np.linalg.norm(self.action(vectors[:,j])-values[j]*vectors[:,j])) for j in range(k)]
        accept(max(residuals),'actual coupled Galerkin eigenpair residual',2e-8)
        return values,vectors,residuals

    def triality(self):
        # Every exact harmonic hop shifts ell and n_x by equal amounts modulo3.
        return (np.tile(self.ell,self.n)[:,None]-self.nx[None,:]).astype(int)%3

    def source_triality(self,source):
        states=np.asarray(source).reshape(-1,self.phdim,self.d)
        norm=float(np.vdot(states,states).real)
        if norm==0:raise AssertionError('No actual prepared source for spectral sector selection')
        weights=[float(np.linalg.norm(states*(self.triality()==sector)[None,:,:])**2/norm) for sector in range(3)]
        sector=int(np.argmax(weights))
        accept(1-weights[sector],'actual prepared point source has one conserved spectral sector',2e-10)
        return sector,weights

    def source_spectrum(self,k,sector):
        mask=(self.triality()==sector).ravel();indices=np.flatnonzero(mask)
        def lift(v):
            out=np.zeros(self.size,dtype=np.complex128);out[indices]=np.asarray(v).ravel()
            return out
        def restricted(v):return self.action(lift(v))[indices]
        rng=np.random.default_rng(2221012+self.occupation)
        witness=rng.normal(size=self.size)+1j*rng.normal(size=self.size);witness/=np.linalg.norm(witness)
        accept(np.linalg.norm(self.action(witness*mask)-self.action(witness)*mask),'actual source-sector projector commutes with full H',2e-10)
        solver=cast(Callable[...,Any],eigsh)
        values,small=solver(HermitianAction(len(indices),restricted),k=k,which='SA',tol=2e-11,v0=rng.normal(size=len(indices)))
        order=np.argsort(values);values=values[order];small=small[:,order]
        vectors=np.column_stack([lift(small[:,j]) for j in range(k)])
        residuals=[float(np.linalg.norm(self.action(vectors[:,j])-values[j]*vectors[:,j])) for j in range(k)]
        accept(max(residuals),'actual source-sector eigenpairs in WHOLE H',2e-8)
        return values,vectors,residuals

    def embed(self,wave,target):
        out=np.zeros((target.n,target.l,self.d),complex)
        out[:self.n,target.M-self.M:target.M+self.M+1]=wave.reshape(self.n,self.l,self.d)
        return out.reshape(target.phdim,self.d)

    def full_residual_bound(self,psi,E):
        target=Coupled(self.n+6,self.M+2,self.zeta,self.occupation,self.UE,self.K,self.frame,self.external)
        full=self.embed(self.reshape(psi),target)
        visible=float(np.linalg.norm(target.action(full.ravel())-E*full.ravel()))
        tail=0.
        for Z,T in zip(target.phases,target.hops):
            for phase,fermion in [(Z,T),(Z.conj().T,T.conj().T)]:
                f=full@fermion.T
                lost=float(np.vdot(f,f).real-np.vdot(phase@f,phase@f).real)
                # Entire phase is unitary; only exact oscillator-tail projection loses norm.
                tail+=math.sqrt(max(0.,lost)+8e-15*float(np.vdot(f,f).real))
        return dict(enlarged_residual=visible,remaining_displacement_tail_upper=tail,full_infinite_comparison_residual_upper=visible+tail)


def source_operators(neutral,charged):
    lookup={row:i for i,row in enumerate(charged.rows)}
    creators=np.zeros((4,charged.d,neutral.d),complex)
    for col,row in enumerate(neutral.rows):
        for c in range(4):
            if c not in row:creators[c,lookup[tuple(sorted((*row,c)))],col]=(-1)**sum(j<c for j in row)
    labels=[np.eye(4)]+[SELECTED[0]@SELECTED[j] for j in range(1,4)]
    return [np.einsum('ic,ijk->cjk',V,creators) for V in labels]


def case(n,M,zeta,UE,K,frame,ward):
    neutral=Coupled(n,M,zeta,4,UE,K,frame)
    charged=Coupled(n,M,zeta,5,UE,K,frame)
    ev,vec,res=neutral.solve(3);psi=vec[:,0];E0=float(ev[0])
    if ev[1]-ev[0]<1e-7:raise AssertionError('Neutral degeneracy requires complete ground projector, not chosen vector')
    wave=neutral.reshape(psi)
    pn=np.sum(abs(wave)**2,axis=0)
    mean=float(pn@neutral.nx);variance=float(pn@(neutral.nx-2)**2)
    accept(abs(mean-2),'actual translation/C neutral mean')
    B=source_operators(neutral,charged)
    expected_norm=np.diag(4-neutral.nx)
    norm_operator=accept(np.linalg.norm(sum(t.conj().T@t for t in B[0])-expected_norm),'complete reference norm operator')
    missing_ref=adverse(np.linalg.norm(B[0][0].conj().T@B[0][0]-expected_norm),'complete reference norm operator')
    gram=np.empty((4,4),complex)
    sources=[]
    for label in B:sources.append(np.array([wave@t.T for t in label]))
    for a in range(4):
        for b in range(4):gram[a,b]=np.vdot(sources[a],sources[b])
    gram_error=accept(np.linalg.norm(gram-2*np.eye(4)),'actual complex quartet Gram',2e-7)
    jray=[];eray=[];total=[];identity_errors=[]
    for a,label in enumerate(B):
        src=sources[a];norm=float(np.vdot(src,src).real)
        exact=0j;electric=0j;light=0j
        for t,sw in zip(label,src):
            Hsw=charged.reshape(charged.action(sw.ravel()))
            original=neutral.reshape(neutral.action(psi))@t.T
            exact+=np.vdot(sw,Hsw-E0*sw)
            electric+=np.vdot(sw,charged.coulomb[None,:]*sw-(neutral.coulomb[None,:]*wave)@t.T)
            light+=np.vdot(sw,Hsw-original-(charged.coulomb[None,:]*sw-(neutral.coulomb[None,:]*wave)@t.T))
        actual=float(electric.real/norm);prediction=UE*(1-variance)/3
        eray.append(actual);jray.append(float(light.real/norm));total.append(float(exact.real/norm))
        identity_errors.append(accept(abs(actual-prediction),'retained Gauss/covariance electric commutator'))
        accept(abs(exact.real/norm-(actual+light.real/norm)),'same fullH source commutator energy')
    cross=UE*variance/3
    missing_cross=adverse(abs(UE/3-eray[0]),'retained Gauss/covariance electric commutator')
    alph=np.array([1.,.3j,-.2+.4j,.6]);alph/=np.linalg.norm(alph)
    mixed=cast(np.ndarray,sum(alph[a]*sources[a] for a in range(4)))
    accept(abs(np.vdot(mixed,mixed).real-2),'complex-quartet normalization')
    source_energy=np.empty((4,4),dtype=np.complex128)
    for a in range(4):
        for b in range(4):
            source_energy[a,b]=sum(np.vdot(sa,charged.reshape(charged.action(sb.ravel()))-E0*sb) for sa,sb in zip(sources[a],sources[b]))/2
    accept(np.linalg.norm(source_energy-source_energy.conj().T),'full complex source Hermitian Rayleigh matrix',2e-7)
    accept(np.linalg.norm(source_energy.diagonal().real-total),'actual full source Rayleigh diagonal',2e-7)
    mixed_energy=float(np.vdot(alph,source_energy@alph).real)
    out:dict[str,Any]=dict(oscillator_levels=n,harmonic_Fourier_labels=list(range(-M,M+1)),harmonic_Bloch_label=zeta,
        neutral_dimension=neutral.size,charged_dimension=charged.size,complete_reference=4,E0=E0,neutral_gap=float(ev[1]-ev[0]),
        neutral_eigen_residuals=res,neutral_ground_mean_nx=mean,neutral_variance_nx=variance,
        complete_reference_norm_error=norm_operator,quartet_Gram_error=gram_error,electric_contributions=eray,
        electric_prediction=UE*(1-variance)/3,Coulomb_self_upper=UE/3,negative_retained_cross=-cross,
        light_commutator_contributions=jray,created_Rayleigh_excess=total,electric_identity_errors=identity_errors,
        adverse=dict(missing_reference=missing_ref,missing_cross=missing_cross),full_residual=neutral.full_residual_bound(psi,E0),
        complex_source_coefficients=[dict(real=float(x.real),imag=float(x.imag)) for x in alph],
        complex_source_Rayleigh_excess=mixed_energy,source_Rayleigh_eigenvalues=np.linalg.eigvalsh(source_energy).tolist(),
        source_Rayleigh_matrix=[[dict(real=float(x.real),imag=float(x.imag)) for x in row] for row in source_energy],
        ground_scope='actual finite exact-Galerkin matrix ground in SELECTED Bloch fibre; not a normalizable fullR harmonic ground or native compact phase')
    if ward:
        sector,sector_weights=charged.source_triality(mixed)
        ec,vc,rc=charged.source_spectrum(6,sector);Ec=float(ec[0]);ground=np.flatnonzero(ec-Ec<1e-7)
        if len(ground)==len(ec):raise AssertionError('Charged cluster reaches selected spectrum edge')
        G=np.linalg.qr(vc[:,ground])[0];gap=float(ec[len(ground)]-Ec)
        accept(np.linalg.norm(G.conj().T@G-np.eye(len(ground))),'orthonormal complete charged ground cluster')
        contact=[];pair=[];ward_errors=[];cross_responses=[];wrong_ordered=[];f_curvatures=[]
        one=np.ones(3);wa=np.array([1.,-1.,0.])/math.sqrt(2);wb=np.array([1.,1.,-2.])/math.sqrt(6)
        solver=cast(Callable[...,Any],cg)
        def projected(v):return charged.action(v)-Ec*v+G@(G.conj().T@v)
        A=HermitianAction(charged.size,projected)
        def reduced(rhs):
            rhs=rhs-G@(G.conj().T@rhs)
            x,status=solver(A,rhs,rtol=2e-11,atol=0,maxiter=3000)
            if status!=0:raise AssertionError(f'actual charged current resolvent did not converge: {status}')
            accept(np.linalg.norm(projected(x)-rhs),'actual charged reduced resolvent',2e-8)
            return x
        phmomentum=np.tile(charged.ell/math.sqrt(6),charged.n)
        def harmonic_generator(v):return (charged.reshape(v)*phmomentum[:,None]).ravel()
        for u in G.T:
            density=(charged.reshape(u)*charged.nx[None,:]).ravel()
            Ju=charged.current(u,one)
            independent=1j*(charged.reshape(charged.action(u))*charged.nx[None,:]).ravel()-1j*charged.action(density)
            accept(np.linalg.norm(Ju-independent),'sameH microscopic current continuity')
            Cu=charged.contact(u,one,one)
            NxJ=(charged.reshape(Ju)*charged.nx[None,:]).ravel()
            JNx=charged.current(density,one)
            accept(np.linalg.norm(Cu-1j*(NxJ-JNx)),'sameH contact double Ward')
            x=reduced(Ju)
            c=float(np.vdot(u,Cu).real);p=2*float(np.vdot(Ju,x).real)
            contact.append(c);pair.append(p);ward_errors.append(accept(abs(c-p),'actual ground contact plus both channels static Ward',2e-7))
            Ja=charged.current(u,wa);Jb=charged.current(u,wb)
            independent_b=1j*(harmonic_generator(charged.action(u))-charged.action(harmonic_generator(u)))
            accept(np.linalg.norm(Jb-independent_b),'retained free harmonic microscopic current')
            xa=reduced(Ja);xb=reduced(Jb)
            cab=np.vdot(u,charged.contact(u,wa,wb))
            ab=np.vdot(Ja,xb);ba=np.vdot(Jb,xa)
            response=cab-ab-ba
            accept(abs(response.imag),'Hermitian static mixed source retains both ordered channels',2e-7)
            cross_responses.append(dict(real=float(response.real),imag=float(response.imag),first_ordered_real=float(ab.real),first_ordered_imag=float(ab.imag),second_ordered_real=float(ba.real),second_ordered_imag=float(ba.imag)))
            wrong_ordered.append(abs((cab-2*ab).imag))
            f_curvatures.append(float((np.vdot(u,charged.contact(u,wa,wa))-2*np.vdot(Ja,xa)).real))
        avgc=float(np.mean(contact));avgp=float(np.mean(pair));mixed_mean=float(np.mean([x['real'] for x in cross_responses]))
        accept(abs(mixed_mean),'sameH mixed-source harmonic translation Ward on complete charged cluster',2e-7)
        wrong_channels=adverse(max(wrong_ordered),'Hermitian static mixed source retains both ordered channels',2e-7)
        delta=.003
        def cluster_energy(alpha,beta):
            comparison=Coupled(n,M,zeta,5,UE,K,frame,alpha*wa+beta*wb)
            energies,_,_=comparison.source_spectrum(len(ground)+1,sector)
            if energies[len(ground)]-energies[len(ground)-1]<gap/2:raise AssertionError('External-source continuation lost the actual charged cluster gap')
            return float(np.mean(energies[:len(ground)]))
        mixed_fd=(cluster_energy(delta,delta)-cluster_energy(delta,-delta)-cluster_energy(-delta,delta)+cluster_energy(-delta,-delta))/(4*delta**2)
        f_fd=(cluster_energy(delta,0)+cluster_energy(-delta,0)-2*Ec)/delta**2
        accept(abs(mixed_fd-mixed_mean),'actual external-source charged cluster mixed curvature',2e-5)
        accept(abs(f_fd-np.mean(f_curvatures)),'actual external-source charged cluster restored-mode curvature',2e-5)
        # The actual complex source propagates through the SAME full coupled H, all four references.
        initial=mixed.reshape(4,charged.size).T/math.sqrt(2)
        source_amplitudes=G.conj().T@initial
        source_ground_weight=float(np.vdot(source_amplitudes,source_amplitudes).real)
        if source_ground_weight<1e-10:raise AssertionError('Actual source has no source-sector ground spectral weight')
        exact_trace=charged.d*sum(charged.phenergy)+charged.phdim*(sum(charged.coulomb)+np.trace(charged.onsite).real)
        exp_solver=cast(Callable[...,Any],expm_multiply)
        propagation=[]
        for time in [0.,.25,.5,1.]:
            evolved=initial if time==0 else exp_solver(-1j*time*charged.operator,initial,traceA=-1j*time*exact_trace)
            accept(abs(np.vdot(evolved,evolved).real-1),'sameH full-reference propagated complex quartet norm',2e-7)
            density=np.sum(abs(evolved.reshape(charged.phdim,charged.d,4))**2,axis=(0,2))
            rho=float(density@charged.nx-2)
            current=sum(np.vdot(evolved[:,c],charged.current(evolved[:,c],one)) for c in range(4))
            energy=sum(np.vdot(evolved[:,c],charged.action(evolved[:,c])) for c in range(4)).real-E0
            accept(abs(energy-mixed_energy),'sameH full coupled complex quartet conserved source Rayleigh excess',2e-7)
            propagation.append(dict(time=time,charge_x=rho,charge_R=1-rho,source_current=float(current.real),Rayleigh_excess=float(energy)))
        motion=abs(propagation[-1]['charge_x']-propagation[0]['charge_x'])
        if motion<1e-5:raise AssertionError('Actual fullH charged packet has not demonstrated motion on selected interval')
        out['actual_source_sector_current_join']=dict(Ec=Ec,source_triality=sector,source_triality_weights=sector_weights,source_sector_ground_weight=source_ground_weight,ground_multiplicity=len(ground),next_gap=gap,charged_eigen_residuals=rc,
            full_residual=charged.full_residual_bound(G[:,0],Ec),contact=contact,two_channels=pair,Ward_errors=ward_errors,
            actual_two_ordered_static_mixed_source_response=cross_responses,harmonic_translation_cluster_mean=mixed_mean,
            physical_external_source_curvatures=dict(delta=delta,mixed_fd=mixed_fd,mixed_resolvent=mixed_mean,restored_mode_fd=f_fd,restored_mode_resolvent=float(np.mean(f_curvatures))),
            missing_ordered_channel=wrong_channels,full_reference_complex_packet_propagation=propagation,
            frozen_transport=adverse(motion,'sameH actual complex charged packet motion',1e-5),
            missing_contact=adverse(abs(avgp),'actual ground contact plus both channels static Ward',2e-7),
            frozen_current_response=adverse(abs(avgc),'actual ground contact plus both channels static Ward',2e-7),
            scope='actual source-addressed invariant-sector charged ground/current/contact, both independent ordered static sources and propagating full-reference complex point source in selected finite Bloch-Galerkin comparison; NOT the unrelated global chargedground/reference-free mass; cluster curvature mean, not unique spin-branch susceptibility; no native photon S-matrix/3D Lorentz identification')
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--sizes',nargs='+',type=int,default=[5,7,9]);p.add_argument('--UE',type=float,default=.37);p.add_argument('--K',type=float,default=.9);p.add_argument('--zeta',type=float,default=0.);p.add_argument('--ward',action='store_true');args=p.parse_args()
    band=RotationBand(a=1.,m=.2,r=.12,v=.3,improved=False)
    selected,frame=original_frame(band,5,np.zeros(3),True)
    results=[case(n,(n+1)//2,args.zeta,args.UE,args.K,frame,args.ward and n==args.sizes[-1]) for n in args.sizes]
    print(json.dumps(dict(law='EXPLICIT DIFFERENT noncompact quadratic-curl photon root; not native cosine replacement',
        comparison_UE=args.UE,comparison_K=args.K,frame_Gram=selected['G'].tolist(),frame_scope='nativeN5 identity-toron source-selected FRAME only; not an admitted ground',
        scalar_field_quantum=False,transverse_and_harmonic_quantum=True,harmonic_directions_deleted=False,
        harmonic_ground_normalizable=False,normalizable_harmonic_state='Bloch wavepacket over zeta; selected fibre alone is not claimed fullR ground',
        quantum_root_debt=['fundamental quantum postulates','microscopic band/field couplers','frozenBODY comparison versus full-field BO transfer','reference-separated spatial/current/phase and Lorentz limits'],
        results=results,native_compact_HC_paid=False,native_phase_proved=False,campaign_complete=False),indent=2,allow_nan=False))

if __name__=='__main__':main()
