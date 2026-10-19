"""Own quantum packets and literal rich hopping circuits with every field P.
No author state, eigen-solve, precomputed loop operator or net-shift substitution.
"""
from pathlib import Path
from bisect import bisect_left
import json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parent;P='rich_common_loop_review_20261017_'
PATHS={'triangle01_12_20':[(0,1),(1,1),(4,-1)],'square01_12_23_30':[(0,1),(1,1),(2,1),(3,-1)]}


def norm2(value):return float(np.vdot(value,value).real)


def unit(value):
    value=np.asarray(value,complex);n=norm2(value)
    if n<=0:raise ValueError('zero packet is not normalizable')
    return value/np.sqrt(n)


def body_step(model,value,part,direction):
    a=part['c0'] if direction==1 else part['c0'].conj().T
    b=part['cphi'] if direction==1 else part['cphi'].conj().T
    flat=value.reshape(model.n,-1);out=(a@flat).reshape(value.shape)
    if b.nnz:out+=(b@flat).reshape(value.shape)*model.grid.reshape((1,len(model.grid))+((1,) if value.ndim==3 else ()))
    return out


def native_word(model,operator_class,state,category,path):
    current=state.copy();ledger=[]
    for edge,direction in path:
        part=next(p for p in model.parts if p['edge']==edge and p['category']==category)
        shift=tuple(direction*np.asarray(part['shift']));a=part['c0'] if direction==1 else part['c0'].conj().T;b=part['cphi'] if direction==1 else part['cphi'].conj().T
        # Oriented one-hop operator: no added adjoint, static H or free energy.
        step=operator_class(model.n,model.grid,model.hphi,model.flux,{shift:(a,b)},free=False)
        before=norm2(current);current=step@current
        ledger.append(dict(edge=edge,direction=direction,integer_shift=list(map(int,shift)),input_norm2=before,retained_output_norm2=norm2(current)))
    return current,ledger


def sparse_word(model,initial,category,path,each_projector=True,final_projector=True):
    """Independent exact field-support representation, NOT a reduced field law.
    A fixed charge word translates each supported m uniquely. Dictionary keys
    outside the85flux basis are retained in the uncut reference, never clamped.
    """
    current={key:value.copy() for key,value in initial.items()};allowed={tuple(v) for v in model.flux};ledger=[]
    for edge,direction in path:
        part=next(p for p in model.parts if p['edge']==edge and p['category']==category);shift=direction*np.asarray(part['shift']);target={};lost=0.
        for m,value in current.items():
            image=body_step(model,value,part,direction);where=tuple(np.asarray(m)+shift)
            if each_projector and where not in allowed:lost+=norm2(image)
            else:target[where]=image
        current=target;ledger.append(dict(edge=edge,direction=direction,lost_image_norm2=lost,field_support=[list(map(int,m)) for m in current]))
    if final_projector:current={key:value for key,value in current.items() if key in allowed}
    return current,ledger


def dense(model,packet):
    out=np.zeros((model.n,len(model.grid),len(model.flux)),complex);where={tuple(v):i for i,v in enumerate(model.flux)}
    for m,value in packet.items():out[:,:,where[m]]=value
    return out.reshape(-1)


def packet_stats(model,packet):
    keys=list(packet);vectors=np.column_stack([packet[k].reshape(-1) for k in keys]);rho=vectors.conj().T@vectors;normal=float(np.trace(rho).real)
    body_probability=sum(abs(value)**2 for value in packet.values()).sum(axis=1)/normal
    return dict(norm2=normal,field_coordinates=[list(k) for k in keys],field_weights=[norm2(packet[k])/normal for k in keys],field_boundary_weight=sum(norm2(packet[k]) for k in keys if max(map(abs,k))==6)/normal,reduced_transverse_purity=float(np.trace(rho@rho).real/normal**2),physical_Q_site=(body_probability@model.q).tolist(),meaning='Own input, not a ground/reference reproduction. Native total-Q0 basis already includes charge-dependent longitudinal Gauss dressing; reduced m0 does not mean full electricfield0.')


def project_input_weight(model,packet,category,path):
    allowed={tuple(v) for v in model.flux};answer=0.;net_answer=0.
    for initial,value in packet.items():
        m=np.asarray(initial);valid=True
        for edge,direction in path:
            part=next(p for p in model.parts if p['edge']==edge and p['category']==category);m=m+direction*np.asarray(part['shift']);valid=valid and tuple(m) in allowed
        if valid:answer+=norm2(value)
        if tuple(m) in allowed:net_answer+=norm2(value)
    return answer,net_answer


def literal_car_fixture(primitive):
    """Separate CAR64 two-fermion exchange on square edges; no hopping units.
    This bare Slater sign fixture is Gauss/count legal, not asserted to be one
    of the native proper-J0 isometry vectors. Native tests below carry R/colour.
    """
    mode=lambda v,f,s,c:16*v+8*f+4*s+c
    occupied=[]
    for v in range(4):
        for s,c in ((0,0),(1,1),(1,2)) if v%2==0 else ((1,0),(0,1),(0,2)):occupied.append(mode(v,0,s,c))
        for s,c in ((0,0),(0,1),(1,2)) if v in (0,2) else ((0,1),(1,0),(1,2)):occupied.append(mode(v,1,s,c))
        occupied.append(mode(v,0,0 if v%2==0 else 1,3));occupied.append(mode(v,1,0 if v in (0,2) else 1,3))
    occupied=sorted(occupied);initial=sum(1<<i for i in occupied);records=[]
    assert len(occupied)==32
    for member,label in ((0,'down_quark'),(3,'charged_white')):
        links=[(mode(1,1,0,member),mode(0,1,0,member)),(mode(3,1,0,member),mode(2,1,0,member)),(mode(2,1,0,member),mode(1,1,0,member)),(mode(0,1,0,member),mode(3,1,0,member))]
        state=occupied.copy();amplitude=1;mask=initial;reference_amplitude=1;step_signs=[]
        for create,annihilate in links:
            j=bisect_left(state,annihilate)
            if j==len(state) or state[j]!=annihilate:raise ValueError('CAR source mode unexpectedly empty')
            sign=(-1)**j;state.pop(j);j=bisect_left(state,create)
            if j<len(state) and state[j]==create:raise ValueError('Pauli-blocked intended exchange circuit')
            sign*=(-1)**j;state.insert(j,create);amplitude*=sign;step_signs.append(sign)
            legacy=primitive.string(mask,[(create,True),(annihilate,False)])
            if legacy is None:raise ValueError('frozen primitive blocked a live CAR word')
            mask,sgn=legacy;reference_amplitude*=sgn
        target=sum(1<<i for i in state)
        assert target==initial and amplitude==-1 and reference_amplitude==amplitude and mask==target
        assert abs(amplitude-1)==2 # omitting Jordan-Wigner signs is physically wrong
        records.append(dict(root=label,initial_mask=str(initial),final_mask=str(target),chronological_CAR_bilinears=links,chronology='01,23,12,30; separate exchange fixture, NOT the primary ordered one-particle path',per_step_fermion_signs=step_signs,exact_amplitude=amplitude,frozen_primitive_amplitude=reference_amplitude,WRONG_signless_amplitude=1,meaning='Two identical occupied fermions exchange while returning to the SAME Slater determinant. Minus sign is spectator-independent here, not an energy coefficient or loop probability. Primary quantum words retain literal01,12,23,30 chronology.'))
    return records


def main():
    config=json.loads(Path(sys.argv[1]).read_text());stem=Path(sys.argv[1]).name.removesuffix('_input.json');started=time.time();sys.path.insert(0,config['frozen_code'])
    import rich_common_quantum_em_20261016_operator as frozen
    import rich_composite_geometry_20261015_model as primitive
    if config['frozen_code'] not in frozen.__file__:raise ValueError('not using frozen native operator source')
    model=frozen.Model(config['matter_prefix']);assert model.n==4407 and len(model.grid)==7 and len(model.flux)==85
    rng=np.random.default_rng(config['seed']);shape=(model.n,len(model.grid));scale=max(abs(model.grid));spacing=min(abs(np.diff(model.grid)))
    A=(rng.normal(size=shape)+1j*rng.normal(size=shape))*np.exp(-model.grid[None,:]**2/(.8*spacing)**2)
    B=(rng.normal(size=shape)+1j*rng.normal(size=shape))*np.exp(-(abs(model.grid[None,:])-scale)**2/(.8*spacing)**2)
    A=unit(A);B-=A*np.vdot(A,B);B=unit(B)
    interior=tuple(config['field_coordinates'][0]);boundary=tuple(config['field_coordinates'][1]);weight=config['Schmidt_weights'];phase=np.exp(1j*config['relative_phase'])
    packets={'interior':{interior:A},'entangled_boundary':{interior:np.sqrt(weight[0])*A,boundary:np.sqrt(weight[1])*phase*B}}
    stats={name:packet_stats(model,packet) for name,packet in packets.items()};arrays={'packet_A_bodyPhi':A,'packet_B_bodyPhi':B,'native_flux_basis':model.flux,'Phi_grid':model.grid};results={};wrong=[]
    for name,packet in packets.items():
        psi=dense(model,packet);results[name]=[]
        for category,root in ((2,'down_quark_nminus1'),(3,'charged_white_nminus3')):
            for path_name,path in PATHS.items():
                image,ledger=native_word(model,frozen.ChargeOperator,psi,category,path)
                independent,independent_ledger=sparse_word(model,packet,category,path)
                raw,_=sparse_word(model,packet,category,path,each_projector=False,final_projector=False)
                shortcut,_=sparse_word(model,packet,category,path,each_projector=False,final_projector=True)
                expected=dense(model,independent);net_image=dense(model,shortcut);actual=norm2(image);uncut=sum(norm2(v) for v in raw.values());net=norm2(net_image)
                error=float(np.linalg.norm(image-expected)/max(np.sqrt(actual),1e-15));wrong_error=float(np.linalg.norm(image-net_image)/max(np.sqrt(net),1e-15))
                assert actual>1e-18 and error<2e-12 and actual<=net+1e-10*max(net,1) and net<=uncut+1e-10*max(uncut,1)
                if name=='entangled_boundary' and path_name.startswith('square'):
                    assert wrong_error>.01;wrong.append(wrong_error)
                if path_name.startswith('triangle'):assert wrong_error<2e-12
                reverse=[(edge,-direction) for edge,direction in reversed(path)];adjoint,_=native_word(model,frozen.ChargeOperator,psi,category,reverse)
                mean=np.vdot(psi,image);mean_adj=np.vdot(psi,adjoint);assert abs(mean_adj-mean.conjugate())<1e-10*max(1,abs(mean))
                re=(image+adjoint)/2;im=(image-adjoint)/(2j);var_re=norm2(re)-float(mean.real)**2;var_im=norm2(im)-float(mean.imag)**2;bound=abs(norm2(adjoint)-actual)/4
                assert var_re>=-1e-12 and var_im>=-1e-12 and np.sqrt(max(var_re,0)*max(var_im,0))+1e-10*max(1,bound)>=bound
                state_weight,net_state_weight=project_input_weight(model,packet,category,path)
                result=dict(root=root,path=path_name,chronological_edges=path,path_length=len(path),norm_squared_units='energy^'+str(2*len(path)),raw_selected_body_uncut_flux_word_norm2=uncut,net_shift_only_norm2_NOT_actual_word=net,actual_every_intermediate_P_norm2=actual,operator_image_retention_fraction=actual/uncut,bare_input_state_survival_weight=state_weight,bare_input_net_only_weight=net_state_weight,independent_literal_vector_relative_error=error,WRONG_net_only_relative_vector_error=wrong_error,complex_loop_mean=[float(mean.real),float(mean.imag)],Hermitian_Re_loop_variance=var_re,Hermitian_Im_loop_variance=var_im,Robertson_standard_deviation_product_lower_bound=bound,native_step_ledger=ledger,independent_step_projection_ledger=independent_ledger,meaning='Raw nonunitary hopping-word action strengths, not probabilities/induction energies. Both c0 vector and cPhi scalar terms retained each step; native body/Phi/R/PW selection remains in BOTH cut and uncut-flux references.')
                results[name].append(result)
                label=name+'_'+root+'_'+path_name;cube=image.reshape(model.n,len(model.grid),len(model.flux))
                support_indices=np.flatnonzero(np.sum(abs(cube)**2,axis=(0,1)))
                arrays[label+'_actual_image']=cube[:,:,support_indices]
                arrays[label+'_actual_image_fluxes']=model.flux[support_indices]
                arrays[label+'_raw_fluxes']=np.asarray(list(raw));arrays[label+'_raw_images']=np.stack(list(raw.values()),axis=-1)
                np.savez_compressed(ROOT/(stem+'_raw.npz'),**arrays)
                (ROOT/(stem+'_partial_result.json')).write_text(json.dumps(dict(packet_stats=stats,completed=results),indent=2)+'\n')
        del psi
    car=literal_car_fixture(primitive);gauss=float(np.max(abs(model.longitudinal@model.d.T-model.integer_q)));assert gauss<1e-11 and len(wrong)==2
    output=dict(kind='OWN_DISTINCT_LOOP_RICH_LITERAL_QUANTUM_CIRCUIT_REVIEW',config=config,model=model.receipt,Gauss_integer_basis_residual=gauss,packet_stats=stats,circuits=results,independent_CAR64_exchange_fixtures=car,meaning='Fresh selected native physical packets only, no author solved vector or ground/evolution replay. Interior positive loops and deliberately boundary-entangled shortcut countercase. Norm² has energy^(2pathlength) units, not state probability; weighted image retention differs from bare state survival. Quantum loop means/quadrature variances are not photons or persistent circulation.',limits=['Fundamental body4407/Phi7/R10/PW19 and85flux cut retained','Uncut-flux reference does NOT remove body/Phi/R/colour selection','Own input entanglement was constructed, not dynamically generated/ground inferred','No virtual energy denominators/effective magnetic energy derived from word strength','No approval of author ground/24 Ritz modes/poles/continuum/fullclosure','Connected256CAR/localfield construction proceeds independently'],elapsed_seconds=time.time()-started)
    (ROOT/(stem+'_result.json')).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(dict(kind=output['kind'],packet_stats=stats,wrong_shortcut_errors=wrong,Gauss_residual=gauss,elapsed_seconds=output['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
