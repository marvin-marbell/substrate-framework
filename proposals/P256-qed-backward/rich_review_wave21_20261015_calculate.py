"""Incremental NEW completed-supplier consumer; never replays old producer CLI.
Full native tensor spaces, explicit signed resolvents and quantum Phi/PW feedback.
"""
from pathlib import Path
from typing import Any
import hashlib,json,resource,sys,time
import numpy as np
from scipy.linalg import expm,qr
from scipy.sparse.linalg import LinearOperator,gmres,expm_multiply
from rich_spatial_colour_charge_20261015_snapshot import load
from rich_spatial_colour_charge_20261015_factors import law,route_current
from rich_spatial_colour_charge_20261015_model import BodyLift
from rich_review_wave18_20261015_law import physical_law as independent_law
ROOT=Path(__file__).resolve().parent


def norm(x: Any)->float:return float(np.linalg.norm(x))


def strength(x: Any)->float:return float(np.vdot(x,x).real)


def complex_json(x: Any)->Any:return dict(real=np.asarray(x).real.tolist(),imag=np.asarray(x).imag.tolist())


def solve(h: Any,energy: float,rhs: Any,w: float,request: Any,p: Any)->Any:
    z=energy+w+1j*request['eta'];matrix=LinearOperator(h.shape,matvec=lambda v:z*v-h@v,rmatvec=lambda v:z.conjugate()*v-h.H@v,dtype=complex);diagonal=z-h.diagonal();preconditioner=LinearOperator(h.shape,matvec=lambda v:v/diagonal,dtype=complex);iterations=[0]
    def counted(value: Any)->None:iterations[0]+=1
    started=time.monotonic();x,info=gmres(matrix,rhs,M=preconditioner,rtol=request['relative_tolerance'],atol=request['relative_tolerance']*1e-2,restart=p['gmres_restart'],maxiter=p['gmres_cycles'],callback=counted,callback_type='pr_norm');residual=norm(matrix@x-rhs)
    return x,dict(info=int(info),iterations=iterations[0],absolute_full_residual=residual,relative_full_residual=residual/norm(rhs),virtual_dimension=h.shape[0],frequency=w,eta=request['eta'],elapsed_seconds=time.monotonic()-started,meaning='Entire native tensor solve; no low24 virtual cutoff, state fallback or residual concealment.')


def evolve(h: Any,state: Any,t: float)->Any:
    return state.copy() if t==0 else expm_multiply(-1j*t*h,state,traceA=-1j*t*h.trace())


def pvm(state: Any,basis: Any)->Any:
    weights=np.sum(abs(state.reshape(basis.dimension,-1))**2,axis=1);labels=sorted(set(basis.h));values=np.array([sum(weights[basis.h==label]) for label in labels]);return dict(h3=list(map(int,labels)),physical_charge=[float(x/3) for x in labels],raw_weights=values.tolist(),raw_sum=float(sum(values)))


def pvm_distance(a: Any,b: Any,raw: float)->Any:
    delta=(np.array(a['raw_weights'])-b['raw_weights'])/raw;l1=float(sum(abs(delta)));return dict(conditional_L1=l1,conditional_TV=l1/2,conditional_L2=norm(delta))


def reduced_difference(a: Any,b: Any,nf: int)->float:
    # EXACT low-rank observable contraction, not a physical-state/evolution cut.
    # Full matter+R marginal rank<=nf; no (21080)^2 density allocation/cancellation
    # of large norm squares, and no singular-value/rank truncation.
    left=a.reshape(-1,nf);right=b.reshape(-1,nf);_,r=qr(np.column_stack([left,right]),mode='economic');la=r[:,:nf];rb=r[:,nf:];return norm(la@la.conj().T-rb@rb.conj().T)


def save_checkpoint(output: Path,result: Any,arrays: Any)->None:
    path=output.with_name(output.name.replace('_result.json','_checkpoint.json'));path.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(path.with_suffix('.npz'),**arrays)


def main()->None:
    started=time.monotonic();request:Any=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);limit=request['memory_limit_MiB']*1024**2;resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    for name,expected in request['required_input_sha256'].items():
        with (ROOT/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError('Pinned supplier/source/state input changed: '+name)
    # Trusted local pickle is unpickled ONLY after its explicit expected digest.
    data=load(ROOT/request['body']);p=data['parameters'];nf=len(data['scalar']['grid']);args=(data['rotor'],data['scalar'],p,data['geometry']);captured=np.load(ROOT/request['neutral_state']);omega=captured['vectors'][:,0];e0=float(captured['energies'][0]);saved=np.load(ROOT/request['science_states']);anti=np.load(ROOT/request['C_states']);producer=json.loads((ROOT/request['science_result']).read_text());h={name:law(op,*args) for name,op in data['operators'].items()};states={'neutral':omega,'lepton':saved['lepton_0:0.5:0.5_pole'],'baryon':saved['baryon_0:0.5:0.5_pole'],'antilepton':anti['antilepton_0:0.5:0.5_pole'],'antibaryon':anti['antibaryon_0:0.5:0.5_pole']};raw={name:data['sources'][name]['0:0.5:0.5']@omega for name in states if name!='neutral'};arrays={name+'_pole':state for name,state in states.items()};arrays.update({name+'_raw_source':state for name,state in raw.items()});result:Any=dict(kind='NEW_INCREMENTAL_NATIVE_FULL_VIRTUAL_RESPONSE_AND_QUANTUM_PHI_TO_PW_LOOP_CONSUMER',request=request,bindings={},dynamics={},responses={},raw_source_resolvents={},C={});rng=np.random.default_rng(21020261015)
    for name,state in states.items():
        op=data['operators'][name];energy=float(np.vdot(state,h[name]@state).real);probe=rng.normal(size=h[name].shape[0])+1j*rng.normal(size=h[name].shape[0]);probe/=norm(probe);second=rng.normal(size=len(probe))+1j*rng.normal(size=len(probe));second/=norm(second);old=independent_law(op,*args);jx=law(op,*args,derivative=(0,));oldjx=independent_law(op,*args,derivative=(0,));contact=law(op,*args,derivative=(0,0));oldcontact=independent_law(op,*args,derivative=(0,0));diagonal=h[name].diagonal();ids=[0,len(probe)//2,len(probe)-1];diag_errors=[]
        for index in ids:
            unit=np.zeros_like(probe);unit[index]=1.;diag_errors.append(float(abs((h[name]@unit)[index]-diagonal[index])))
        result['bindings'][name]=dict(virtual_dimension=len(probe),normalized_stationary_reference_norm2=strength(state),own_reference_energy=energy,stationary_full_H_residual=norm(h[name]@state-energy*state),compact_vs_prior_independent_H_action_relative_error=norm(h[name]@probe-old@probe)/norm(old@probe),compact_vs_independent_current_action_error=norm(jx@probe-oldjx@probe),compact_vs_independent_contact_action_error=norm(contact@probe-oldcontact@probe),actual_adjoint_bilinear_error=float(abs(np.vdot(second,h[name]@probe)-np.vdot(h[name].H@second,probe))),selected_true_diagonal_entry_errors=diag_errors)
    for name in ['lepton','baryon']:
        result['bindings'][name]['raw_snapshot_error']=norm(raw[name]-saved[name+'_0:0.5:0.5_raw']);partner='anti'+name;result['bindings'][partner]['raw_snapshot_error']=norm(raw[partner]-anti[partner+'_0:0.5:0.5_raw']);c=data['C'][name];mapped=c@raw[name];phase=np.vdot(mapped,raw[partner])/strength(mapped);current=law(data['operators'][name],*args,derivative=(0,));acurrent=law(data['operators'][partner],*args,derivative=(0,));result['C'][name]=dict(raw_positive_strength=strength(raw[name]),raw_anti_strength=strength(raw[partner]),actual_C_source_phase=complex_json(phase),phase_aligned_raw_C_source_error=norm(raw[partner]-phase*mapped),compact_H_C_action_error=norm(h[partner]@mapped-c@(h[name]@raw[name])),compact_current_C_odd_action_error=norm(acurrent@mapped+c@(current@raw[name])))
    save_checkpoint(output,result,arrays);print('Checkpointed exact same-H references/raw sources and compact integration',flush=True)
    unitary=expm(-1j*p['field_kick']*data['scalar']['P']);result['Phi_unitary_defect']=norm(unitary.conj().T@unitary-np.eye(nf))
    for name in ['lepton','baryon']:
        state=raw[name];weight=strength(state);op=data['operators'][name];kicked=(state.reshape(-1,nf)@unitary.T).ravel();routes=[route_current(op,*args,e) for e in range(2)];observables={key:BodyLift(op[key],10*nf) for key in ['B','C','electric','rho0','Q']};initial=pvm(state,data['native'][name]);off=law(op,*args,path_scale=(0.,0.));mechanical=law(op,*args,path_scale=(0.,0.),derivative=(0,));entry=dict(source_raw_strength=weight,initial_full_matter_R_marginal_difference_raw_Frobenius=reduced_difference(state,kicked,nf),all_hops_off_original_mechanical_current_action_norm=norm(mechanical@state),histories={})
        def values(x: Any,graph_on: bool)->Any:
            measured={key:float(np.vdot(x,operator@x).real/weight) for key,operator in observables.items()};prob=np.sum(abs(x.reshape(-1,nf))**2,axis=0)/weight;measured.update(phi=float(prob@data['scalar']['grid']),raw_norm2=strength(x))
            if graph_on:measured.update(route0=float(np.vdot(x,routes[0]@x).real/weight),route1=float(np.vdot(x,routes[1]@x).real/weight))
            return measured
        for label,operator in [('coupled',h[name]),('ALL_hops_off',off)]:
            history=[]
            for t in request['times']:
                vectors=evolve(operator,np.column_stack([state,kicked]),t);base,driven=vectors.T;a=values(driven,label=='coupled');b=values(base,label=='coupled');pa=pvm(driven,data['native'][name]);pb=pvm(base,data['native'][name]);history.append(dict(time=t,conditional_actual=a,conditional_matched_baseline=b,conditional_excess={key:a[key]-b[key] for key in a},raw_full_matter_R_marginal_difference_Frobenius=reduced_difference(base,driven,nf),material_charge_excess=pvm_distance(pa,pb,weight),baseline_material_charge_change_from_initial=pvm_distance(pb,initial,weight)))
                arrays[name+'_'+label+'_t'+str(t)+'_baseline']=base;arrays[name+'_'+label+'_t'+str(t)+'_driven']=driven
            entry['histories'][label]=history
        result['dynamics'][name]=entry;save_checkpoint(output,result,arrays);print('Completed actual Phi/loop feedback',name,flush=True)
    for name,state in states.items():
        op=data['operators'][name];currents=[law(op,*args,derivative=(a,)) for a in range(3)];images=[j@state for j in currents];energy=float(np.vdot(state,h[name]@state).real);greens={};details={}
        for w in [request['frequency'],-request['frequency']]:
            x,status=solve(h[name],energy,images[0],w,request,p);greens[w]=np.array([np.vdot(left,x) for left in images]);details[str(w)]=status;arrays[name+'_current_solved_'+str(w)]=x
        w=request['frequency'];chi=greens[w]+greens[-w].conj();contact=np.array([np.vdot(state,law(op,*args,derivative=(a,0))@state).real for a in range(3)]);fi=producer['responses'][name]['frequencies'].index(w);reported=producer['responses'][name]['retarded'];expected=np.array(reported['real'])[fi,:,0]+1j*np.array(reported['imag'])[fi,:,0];entry=dict(virtual_dimension=h[name].shape[0],stationary_energy=energy,current_column=0,plus_order=complex_json(greens[w]),minus_frequency_order=complex_json(greens[-w]),retarded_column=complex_json(chi),physical_contact_column=contact.tolist(),kernel_with_contact=complex_json(chi+contact),own_vs_producer_retarded_column_error=norm(chi-expected),own_vs_producer_contact_column_error=norm(contact-np.array(producer['responses'][name]['physical_contact'])[:,0]),wrong_omit_negative_signed_order_column_error=norm(greens[w]-chi),solves=details,scope='Representative ONE current RHS column, all three physical left currents, both signed full-retained orders. Not all producer tensor columns/frequencies or66 solves rerun. Physical contact is independent source derivative; no low24 virtual replacement.')
        result['responses'][name]=entry;save_checkpoint(output,result,arrays);print('Completed full-virtual signed current response',name,flush=True)
    for name,partner in [('lepton','antilepton'),('baryon','antibaryon')]:
        values={}
        for sector,w in [(name,request['frequency']),(partner,-request['frequency'])]:
            rhs=raw[sector];x,status=solve(h[sector],e0,rhs,w,request,p);value=np.vdot(rhs,x);key=sector+':0:0.5:0.5';reported=producer['raw_source_Greens'][key];index=reported['frequencies'].index(w);expected=complex(reported['G']['real'][index],reported['G']['imag'][index]);values[sector]=value;result['raw_source_resolvents'][sector]=dict(frequency=w,eta=request['eta'],virtual_dimension=h[sector].shape[0],raw_strength=strength(rhs),G=complex_json(value),own_vs_producer_G_error=float(abs(value-expected)),conditional_G=complex_json(value/strength(rhs)),wrong_forced_point25_strength_G_error=float(abs(value/.25-value/strength(rhs))),solve=status);arrays[sector+'_raw_source_solved']=x
        result['C'][name]['raw_odd_source_combination_at_positive_frequency']=complex_json(values[name]-values[partner].conjugate());save_checkpoint(output,result,arrays);print('Completed raw source/C signed response',name,flush=True)
    result.update(meaning='Completed-supplier increment only. Frozen18 actual H0/native/algebra/all56sources/C/tail review is reused with attribution, not replayed. Own selected5-sector current column BOTH signed orders plus4 raw-source solves use full23240/40600/147560 virtual spaces; no claim all66 rerun. Phi-only local unitary preserves initial whole matter/R/PW marginal, later actual ReTrU B/PW Casimir C/route response compared to same raw preparation. ALL-hops-off remains a genuine quantum Phi decoupling/material-charge control; residual mechanical/lab current is distinct.',limits=['Normalized stationary pole refs, raw source strengths/residues and conditional probabilities are different. No source normalization inside Green/dynamic action.','h=3rho is label, physicalQ=h/3; PW/SU3 colour not EM. Sourcecapture not location; EMcurrent no lepton-to-uud conversion.','Non-nested chosen family/positive .01504645 spin split not convergence/global ordering/proton/binding or Nature.','Prior57.38percent omission is ONE scalar colourless TREE current; electric loss ONE H component. Total Cartesian/mechanical/H tails unbounded; small Ward is not gauge/range/precision bound.','PW19/SO3j0+j1/scalar7/selectedseed geometry/Clifford inputs remain. No new coupling/rank/fit/gate or photon/COM/continuum claim.'],solve_count=sum(len(entry['solves']) for entry in result['responses'].values())+len(result['raw_source_resolvents']),elapsed_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),**arrays);print(json.dumps(dict(result=output.name,solve_count=result['solve_count'],elapsed_seconds=result['elapsed_seconds'],peak_RSS_KiB=result['peak_RSS_KiB'])),flush=True)


if __name__=='__main__':main()
