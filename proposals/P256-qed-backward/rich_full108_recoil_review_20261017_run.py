"""Exact low-order response of the SAME full108CAR/75-field law on own Slaters."""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.linalg import expm,null_space
import rich_full108_recoil_review_20261017_wick as wick

ROOT=Path(__file__).resolve().parent;P='rich_full108_recoil_review_20261017_'


def encode(value):
    if isinstance(value,np.ndarray):return encode(value.tolist())
    if isinstance(value,(complex,np.complexfloating)):return [float(value.real),float(value.imag)]
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def publish(path,value):path.write_text(json.dumps(encode(value),indent=2)+'\n')


def distribution(character):
    # U eigenvalue exp(-2pi*i*r/3); Fourier inverse has PLUS sign.
    weights=np.exp(2j*np.pi*np.outer(np.arange(3),np.arange(3))/3)
    return np.einsum('ra,sb,abk->rsk',weights,weights,character,optimize=True).real/9


def prepare_source(one):
    cells=one['geometry']['cells'];weights=[np.array([1.,.3+.2j,.12-.1j]),np.array([.8,.2-.4j,.25]),np.array([1.,.08+.03j,.02-.01j])]
    wave=np.ones(len(cells),complex)
    for axis in range(3):wave*=np.exp(2j*np.pi*np.outer(cells[:,axis],np.arange(3))/3)@weights[axis]
    spinor=np.array([1.,.37+.19j,-.28+.41j,.61-.23j]);raw=np.kron(wave,spinor);raw/=np.linalg.norm(raw);u=one['orbitals'].conj().T@raw
    sea=one['orbitals'].conj().T@one['orbitals'][:,one['occupied']];rho=sea@sea.conj().T
    plus=(np.eye(108)-rho)@u;minus=rho@u;wp=float(np.vdot(plus,plus).real);wm=float(np.vdot(minus,minus).real)
    if min(wp,wm)<=1e-8:raise ValueError('fixed own canonical packet did not activate BOTH signed CAR sources')
    particle=np.column_stack([sea,plus/np.sqrt(wp)]);hole=sea@null_space((sea.conj().T@u/np.sqrt(wm)).conj()[None,:])
    return {54:sea,55:particle,53:hole},{54:1.,55:wp,53:wm},dict(raw_physical_packet=raw,canonical_packet=u,addition_orbital=plus/np.sqrt(wp),removed_orbital=minus/np.sqrt(wm),source_norm_sum=wp+wm,source_coefficient_norm2=float(np.vdot(u,u).real),meaning='Fresh own coherent3D packet, generic given primitive spinor. Slaters are exact normalized canonical source RAYS; hole occupied-column global phase is conventional, not used for cross-N amplitudes. rho_ij=<c_j^dag c_i>. No author prepared/evolved vector.')


def field_momenta(extended,indices,width):
    states=extended['shell_states'];lookup={s:i for i,s in enumerate(states)};operators=[]
    for axis in range(4):
        a=np.zeros((len(states),len(states)),complex)
        for col,state in enumerate(states):
            if state[axis]:
                target=list(state);target[axis]-=1;a[lookup[tuple(target)],col]=np.sqrt(state[axis])
        operators.append(np.kron(np.eye(extended['nphi']),(a-a.conj().T)/(1j*np.sqrt(2)*width)))
    small=[op[np.ix_(indices,indices)] for op in operators]
    products=[[((a@b+b@a)/2)[np.ix_(indices,indices)] for b in operators] for a in operators]
    return small,products


def source_current_families(A,field,extended,indices,T,U):
    families={name:[] for name in ('J','QJ','JQ','WRONG_QJ','WRONG_JQ')};rows=[]
    for pol in range(2):
        qplus=(extended['coordinates'][2*pol]+1j*extended['coordinates'][2*pol+1]).toarray()/np.sqrt(2)
        qsmall=qplus[np.ix_(indices,indices)];qgrades=[np.linalg.norm(U@qsmall@U.conj().T-np.exp(-2j*np.pi*g/3)*qsmall) for g in range(3)];qgrade=int(np.argmin(qgrades))
        for sign in (1,-1):
            large=np.array([-(extended['derivatives'][2*pol][c]+1j*sign*extended['derivatives'][2*pol+1][c]).toarray()/np.sqrt(2) for c in range(16)]);small=large[:,indices][:,:,indices]
            norm=wick.operator_norm(A,small);residual=[wick.covariance_residual(A,small,T,U,np.exp(-2j*np.pi*g/3))/norm for g in range(3)];grade=int(np.argmin(residual))
            counter=qplus if qgrade==(-grade)%3 else qplus.conj().T;counter_small=counter[np.ix_(indices,indices)]
            qj=(counter@large)[:,indices][:,:,indices];jq=(large@counter)[:,indices][:,:,indices]
            families['J'].append(small);families['QJ'].append(qj);families['JQ'].append(jq);families['WRONG_QJ'].append(counter_small@small);families['WRONG_JQ'].append(small@counter_small)
            rows.append(dict(polarization=pol,profile_sign=sign,measured_joint_grade=grade,current_normalized_fullFock_HS_residuals_by_grade=residual,current_HS_norm=norm,matter_only_NOT_demanded_grade_residual=wick.covariance_residual(A,small,T,np.eye(len(U)),np.exp(-2j*np.pi*grade/3))/norm,QJ_grade0_relative_residual=wick.covariance_residual(A,qj,T,U)/wick.operator_norm(A,qj),JQ_grade0_relative_residual=wick.covariance_residual(A,jq,T,U)/wick.operator_norm(A,jq),beforecut_QJ_minus_JQ_coefficient_norm=float(np.linalg.norm(qj-jq))))
    return {name:np.array(values) for name,values in families.items()},rows


def main():
    started=time.time();inputpath=Path(sys.argv[1]);config=json.loads(inputpath.read_text());stem=inputpath.name.removesuffix('_input.json');law=config['frozen']['law'];code=ROOT/config['frozen']['code_snapshot'];sys.path.insert(0,str(code))
    import rich_recoil_current_wave_20261016_model as native
    import rich_spatial_translation_20261015_model as translation
    if not native.__file__.startswith(str(code)) or not translation.__file__.startswith(str(code)):raise ValueError('frozen pure-source import escaped own input snapshot')
    algebra=wick.exact_CAR_formula_fixture();one=native.single_particle(law);field=translation.shell_fields(law);extended=translation.shell_fields(dict(law,shell_cutoff=law['shell_cutoff']+1))
    if (len(one['labels']),field['dimension'],extended['dimension'])!=(108,75,175):raise ValueError('not selected full108/75 law')
    A=np.zeros((16,108,108),complex)
    for block in range(9):A[:,12*block:12*block+12,12*block:12*block+12]=one['components'][block]
    if np.linalg.norm(A-A.conj().transpose(0,2,1))>1e-12:raise ValueError('own primitive body basis is not Hermitian')
    B=np.array([field['boson'].toarray()-one['offset']*np.eye(75)]+[x.toarray() for x in field['coupling']]);BF=np.array([B[0]]+[x.toarray() for x in field['frozen']]);T=np.diag(np.exp(-2j*np.pi*one['momentum']/3));U=field['field_translation'].toarray()
    cells=one['geometry']['cells'];lookup={tuple(x):i for i,x in enumerate(cells)};spatial=np.zeros((108,108),complex)
    for source,x in enumerate(cells):
        target=x.copy();target[2]=(target[2]+1)%3;destination=lookup[tuple(target)];spatial[4*destination:4*destination+4,4*source:4*source+4]=np.eye(4)
    geometric_error=float(np.linalg.norm(one['orbitals'].conj().T@spatial@one['orbitals']-T));Hnorm=wick.operator_norm(A,B[1:],B[0]);cov=wick.covariance_residual(A,B[1:],T,U,bare=B[0])/Hnorm;wrongcov=wick.covariance_residual(A,B[1:],T,U.conj().T,bare=B[0])/Hnorm
    indices=np.array([p*len(extended['shell_states'])+extended['shell_states'].index(s) for p in range(field['nphi']) for s in field['shell_states']]);families,vertices=source_current_families(A,field,extended,indices,T,U)
    momenta,products=field_momenta(extended,indices,law['shell_width']);orbitals,strengths,source=prepare_source(one)
    phi=np.exp(-(field['phi']-config['own_phi_center'])**2/(4*config['own_phi_width']**2));phi/=np.linalg.norm(phi);initial=np.kron(phi,np.eye(field['shell_dimension'])[:,0]);protocols=dict(baseline=initial)
    for pol,name in enumerate(('xcos','ycos')):protocols[name]=expm(1j*config['kick_strength']*field['coordinates'][2*pol].toarray())@initial
    Tp=[np.linalg.matrix_power(T,a) for a in range(3)];Up=[np.linalg.matrix_power(U,a) for a in range(3)];cases={};raw=dict(primitive_body_coefficients=A,full_coupled_boson_coefficients=B,frozen_coefficients=BF,source_packet=source['canonical_packet'],physical_source_packet=source['raw_physical_packet'],scalar_Phi_nodes=field['phi'],onebody_orbitals=one['orbitals'],onebody_momentum=one['momentum'],spatial_cells=cells,field_basis=field['shell_states'],field_translation=U,matter_onebody_translation=T)
    receipt=dict(kind='OWN_EXACT_FULL108_CAR_75_QUANTUM_FIELD_LOW_ORDER_RESPONSE',config=config,formula_fixture=algebra,operator_covariance=dict(independent_geometric_matter_translation_error=geometric_error,H_joint_relative_fullFock_HS_residual=cov,WRONG_inverse_field_character_H_residual=wrongcov,vertices=vertices),own_source={k:v for k,v in source.items() if not isinstance(v,np.ndarray)},source_strengths=strengths,cases=cases,meaning='Exact full-Fock Wick/twisted determinants through stated time order, NOT finite-time TDVP reproduction. No local occupancy/pair cap, old source span, field/MPO/current-state direction removal or author solved/evolved vector.',limits=['Primitive supplied fixed spin/Clifford/CAR; not Noether-derived spin','One spatialq/finiteC3 and75 boson regulator; no nonuniform longitudinal/Gauss sector','Finite projected-Q input pulse, NOT full-W or pure/asymptotic photons; source field signed populations already present','Frozen retains real body hopping/convection and independent free scalar/EM propagation; original-law current probes are NOT its own zero field-source derivatives','N53 starts with one valence hole, not two newly created pairs','No author t=.2 magnitudes, TDVP error bars, freeCOM, photon poles/radiation, Lorentz/mass/proton/Nature proof'])
    for number,V in orbitals.items():
        rho=V@V.conj().T;prepared=[wick.prepare(A,V,t) for t in Tp];M=prepared[0]['aa'];raw['Slater_'+str(number)]=V
        for name,f in protocols.items():
            key=f'N{number}_{name}';chars=np.empty((3,3,3),complex);frozen_chars=np.empty_like(chars)
            for a in range(3):
                for b in range(3):chars[a,b],_=wick.prepared_expansion(prepared[a],B,f,Up[b]);frozen_chars[a,b],_=wick.prepared_expansion(prepared[a],BF,f,Up[b])
            joint=distribution(chars);frozen_joint=distribution(frozen_chars);gm={family:wick.gram(A,V,f,D,M) for family,D in families.items()};responses={family:np.array([wick.current_response(A,B,V,f,D,M) for D in families[family]]) for family in ('J','QJ','JQ')};frozen_responses={family:np.array([wick.current_response(A,BF,V,f,D,M) for D in families[family]]) for family in ('J','QJ','JQ')}
            wrongq=wick.gram(A,V,f,families['QJ']-families['WRONG_QJ'],M);wrongjq=wick.gram(A,V,f,families['JQ']-families['WRONG_JQ'],M);tc=M[1:,0];meanonly=np.empty_like(M);meanonly[0,0]=1;meanonly[0,1:]=tc;meanonly[1:,0]=tc;meanonly[1:,1:]=np.outer(tc,tc)
            missing_sea=gm['J']-wick.gram(A,V,f,families['J'],meanonly)
            inserted={}
            for family in ('J','QJ','JQ'):
                characters=np.array([wick.inserted_character(A,V,f,families[family],Tp[a],Up[a],prepared[a])[0] for a in range(3)])
                sectors=np.array([sum(np.exp(2j*np.pi*a*r/3)*characters[a] for a in range(3))/3 for r in range(3)])
                inserted[family]=dict(raw_character_matrices=characters,raw_current_conditioned_total_C3_sector_Gram_matrices=sectors,raw_Gram_eigenvalues=np.linalg.eigvalsh((gm[family]+gm[family].conj().T)/2),meaning='Raw unnormalized current-insertion sector metrics, not emission probabilities. Nonunitary J can reweight initial coherent total-C3 components; no demanded unitary matter shift.')
            field_response={}
            for label,O in [(f'Pi{a}',momenta[a]) for a in range(4)]+[(f'Pi{a}Pi{b}',products[a][b]) for a in range(4) for b in range(a,4)]:
                active,_=wick.prepared_expansion(prepared[0],B,f,O);frozen,_=wick.prepared_expansion(prepared[0],BF,f,O);field_response[label]=dict(coupled=active,frozen_own_generator=frozen)
            hmean=np.einsum('a,aij->ij',np.conj(f)@(B[1:]@f).T,A);drho=-1j*(hmean@rho-rho@hmean);frozen_h=np.diag(one['energies']);frozen_drho=-1j*(frozen_h@rho-rho@frozen_h)
            realrho=one['orbitals']@rho@one['orbitals'].conj().T;realdrho=one['orbitals']@drho@one['orbitals'].conj().T;real_frozen=one['orbitals']@frozen_drho@one['orbitals'].conj().T
            density=np.diag(realrho).real.reshape(27,4).sum(axis=1);density_rate=np.diag(realdrho).real.reshape(27,4).sum(axis=1);frozen_density_rate=np.diag(real_frozen).real.reshape(27,4).sum(axis=1)
            energy=sum(M[a,0]*np.vdot(f,b@f) for a,b in enumerate(B));bf=B@f;bd=B.conj().transpose(0,2,1)@f;energy2=np.sum(M*(bd.conj()@bf.T))
            row=dict(number=number,protocol=name,raw_canonical_source_norm2=strengths[number],Slater_orthonormality_error=float(np.linalg.norm(V.conj().T@V-np.eye(number))),minimum_twisted_overlap_singular_value=min(item['minimum'] for item in prepared),source_body_density=density,body_density_time_coefficient1=density_rate,frozen_real_body_convection_coefficient1=frozen_density_rate,field_initial_signed_C3_populations=joint[:,:,0].sum(axis=0),joint_matter_field_C3_probability_time_coefficients=joint,frozen_joint_coefficients=frozen_joint,joint_characters_time_coefficients=chars,frozen_characters_time_coefficients=frozen_chars,whole_joint_total_character_derivative_max=float(max(abs(chars[a,a,1:]).max() for a in range(3))),currents_and_beforecut_products_time_coefficients=responses,original_law_probes_on_frozen_generator_time_coefficients=frozen_responses,raw_insertion_Grams=gm,physical_source_weighted_Grams={family:strengths[number]*value for family,value in gm.items()},WRONG_intermediate_cut_QJ_difference_image_norm2=np.diag(wrongq).real,WRONG_intermediate_cut_JQ_difference_image_norm2=np.diag(wrongjq).real,connected_fullFock_virtual_current_Gram_omitted_by_meanonly_closure=missing_sea,current_conditioned_recoil=inserted,field_means_covariance_receivers=field_response,full_H_energy=energy,full_H_square_all_cross_terms=energy2,full_H_variance=energy2-energy**2,time_coefficient_semantics='Observable(t)=c0+c1*t+c2*t²+O(t³). Current/JQ/QJ arrays contain ONLY c0,c1. Exact selected-order fullFock intermediates, NOT finite-time magnitudes or a remainder/error bound.')
            cases[key]=row;raw[key+'_field_state']=f;raw[key+'_rho_ij_cdagj_ci']=rho;raw[key+'_joint_character_coefficients']=chars
            np.savez_compressed(ROOT/(stem+'_raw.npz'),**raw);publish(ROOT/(stem+'_partial_result.json'),receipt);print(json.dumps(dict(completed=key,raw_source_weight=strengths[number],matter_recoil_coefficient2_norm=float(np.linalg.norm(joint[:,:,2].sum(axis=1))),field_recoil_coefficient2_norm=float(np.linalg.norm(joint[:,:,2].sum(axis=0))),joint_total_character_derivative_max=row['whole_joint_total_character_derivative_max'],beforecut_missing_QJ_min=float(np.diag(wrongq).real.min()))),flush=True)
    paired={}
    for number in (55,53):
        for pol,name in enumerate(('xcos','ycos')):
            charge=cases[f'N{number}_{name}'];neutral=cases['N54_'+name];base=cases[f'N{number}_baseline'];nb=cases['N54_baseline'];key=f'N{number}_{name}'
            paired[key]=dict(dynamic_neutral_subtracted_recoil_coefficients=(charge['joint_matter_field_C3_probability_time_coefficients']-base['joint_matter_field_C3_probability_time_coefficients'])-(neutral['joint_matter_field_C3_probability_time_coefficients']-nb['joint_matter_field_C3_probability_time_coefficients']),dynamic_neutral_subtracted_signed_current_coefficients={family:(charge['currents_and_beforecut_products_time_coefficients'][family]-base['currents_and_beforecut_products_time_coefficients'][family])-(neutral['currents_and_beforecut_products_time_coefficients'][family]-nb['currents_and_beforecut_products_time_coefficients'][family]) for family in ('J','QJ','JQ')},body_initial_pulse_density_invariance_error=float(np.max(abs(charge['source_body_density']-base['source_body_density']))),raw_initial_field_populations_ALREADY_PRESENT=charge['field_initial_signed_C3_populations'])
    receipt['paired_propagating_neutral_time_coefficient_contrasts']=paired;receipt['elapsed_seconds']=time.time()-started
    assertions=dict(independent_actual_geometric_translation=geometric_error<2e-12,actual_joint_H_covariance=cov<3e-12,WRONG_field_character_rejected=wrongcov>1e-8,all_four_signed_T_current_joint_grades=all(v['measured_joint_grade']!=0 and min(v['current_normalized_fullFock_HS_residuals_by_grade'])<3e-12 for v in vertices),QJ_JQ_beforecut_grade0=all(max(v['QJ_grade0_relative_residual'],v['JQ_grade0_relative_residual'])<3e-12 for v in vertices),whole_joint_C3_conservation_through_exact_order=max(r['whole_joint_total_character_derivative_max'] for r in cases.values())<2e-9,physical_beforecut_missing_intermediate_countercase=all(np.min(cases[f'N{number}_baseline']['WRONG_intermediate_cut_QJ_difference_image_norm2'])>1e-12 for number in (55,53)),same108_quantum_matter_and_field_recoil_both_T_both_source_signs=all(np.linalg.norm(cases[f'N{number}_{name}']['joint_matter_field_C3_probability_time_coefficients'][:,:,2].sum(axis=0))>1e-10 and np.linalg.norm(cases[f'N{number}_{name}']['joint_matter_field_C3_probability_time_coefficients'][:,:,2].sum(axis=1))>1e-10 for number in (55,53) for name in ('xcos','ycos')))
    receipt['physics_assertions']=assertions;publish(ROOT/(stem+'_result.json'),receipt);print(json.dumps(dict(physics_assertions=assertions,elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
    assert all(assertions.values()),assertions

if __name__=='__main__':main()
