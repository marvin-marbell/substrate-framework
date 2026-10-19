"""Full signed SIX-profile current/contact and source resolvents of exact saved H.
Low eigenvectors define/label stationary poles ONLY; no low-pole truncation of virtuals.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres
from rich_spatial_colour_em_20261015_snapshot import load,source,matrix
from rich_spatial_colour_em_20261015_model import assemble,observable,angular


def complex_json(value):return dict(real=np.asarray(value).real.tolist(),imag=np.asarray(value).imag.tolist())


def mean(state,operator):return float(np.vdot(state,operator@state).real)


def solve(h,energy,rhs,frequency,p):
    z=frequency+energy+1j*p['eta'];diagonal=z-h.diagonal();count=[0]
    operator=LinearOperator(h.shape,matvec=lambda x:z*x-h@x,rmatvec=lambda x:z.conjugate()*x-h@x,dtype=complex)
    preconditioner=LinearOperator(h.shape,matvec=lambda x:x/diagonal,dtype=complex)
    def callback(value):count[0]+=1
    solution,info=gmres(operator,rhs,M=preconditioner,rtol=p['resolvent_tolerance'],atol=p['resolvent_tolerance']*1e-2,restart=p['gmres_restart'],maxiter=p['gmres_cycles'],callback=callback,callback_type='pr_norm')
    residual=float(np.linalg.norm(operator@solution-rhs))
    return solution,dict(info=int(info),iterations=count[0],absolute_residual=residual,relative_residual=residual/max(float(np.linalg.norm(rhs)),1e-30))


def main():
    inp,out=map(Path,sys.argv[1:3]);request=json.loads(inp.read_text());root=inp.parent;started=time.time();state_json=root/request['state_json'];state_npz=root/request['state_npz']
    metadata,data,model,laws=load(state_json,state_npz);p=model['parameters'];base=model['base'];field=model['field'];reference=data['neutral_reference'];e0=metadata['reference_energy']
    p=dict(p);p.update({key:request[key] for key in ['eta','resolvent_tolerance','gmres_restart','gmres_cycles']})
    frequencies=request['frequencies'];solved_frequencies=sorted(set(frequencies+[-x for x in frequencies]));results={};source_results={}
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [state_json,state_npz]}
    def checkpoint(stage):
        payload=dict(kind='FULL_RETAINED_SPATIAL_COLOUR_EM_SIGNED_RESPONSE_NOT_PHOTON_DERIVATION',stage=stage,input_sha256=hashes,request=request,eta=p['eta'],results=results,sources=source_results,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-started)
        out.write_text(json.dumps(payload,indent=2)+'\n')
    # Actual mixed poles from the SAME new neutral whole H.
    energies=data['neutral_modes_e'];vectors=data['neutral_modes_v'];nr=base['rotor']['dimension']
    mode_ops={}
    for mu in range(6):
        mode_ops['J'+str(mu)]=assemble(model,'neutral',derivative=(mu,))
        mode_ops['Q'+str(mu)]=observable(model,'neutral',rq=np.kron(np.eye(nr),field['Q'][mu]))
        mode_ops['Pi'+str(mu)]=observable(model,'neutral',rq=np.kron(np.eye(nr),field['Pi'][mu]))
    results['neutral_modes']=dict(gaps=(energies-energies[0]).tolist(),actual_total_J_M=data['neutral_modes_JM'].tolist(),raw_residues={key:(abs(vectors.conj().T@(op@reference))**2).tolist() for key,op in mode_ops.items()},bare_Pi_squared_basis_spectrum=np.linalg.eigvalsh(field['H_E']).tolist(),meaning='Actual full NoEther ladders complete every M of computed neutral low-window multiplets. Positive field+physical-current-bright poles of this finite SAME H are NOT bare Pi²/BASIS box photons or restoration; two TE/TM families are not two plane-wave polarizations.')
    checkpoint('neutral_modes')
    for case in request['current_cases']:
        name=case['name'];key=case.get('key');label=name if key is None else name+'_'+key;h=laws[name]
        if key is None:state=reference
        else:state=data[name+'_source_'+key+'_pole']
        energy=mean(state,h);eigen_residual=float(np.linalg.norm(h@state-energy*state));currents=[assemble(model,name,derivative=(mu,)) for mu in range(6)];images=[current@state for current in currents]
        contact=np.empty((6,6))
        for mu in range(6):
            for nu in range(mu,6):
                value=mean(state,assemble(model,name,derivative=(mu,nu)));contact[mu,nu]=value;contact[nu,mu]=value
        greens={};diagnostics={}
        for frequency in solved_frequencies:
            values=np.empty((6,6),complex);details=[]
            for nu,rhs in enumerate(images):
                solution,status=solve(h,energy,rhs,frequency,p);details.append(status)
                for mu,left in enumerate(images):values[mu,nu]=np.vdot(left,solution)
            greens[frequency]=values;diagnostics[str(frequency)]=details
            results[label]=dict(reference_energy=energy,stationary_eigen_residual=eigen_residual,physical_J2=float(sum(np.linalg.norm(angular(model,name,axis)@state)**2 for axis in range(3))),retained_virtual_dimension=h.shape[0],solved_frequencies=list(greens),solves=diagnostics)
            checkpoint('current_partial_'+label+'_'+str(frequency))
        chi=np.array([greens[w]+greens[-w].conj() for w in frequencies]);kernel=chi+contact
        results[label].update(frequencies=frequencies,contact=contact.tolist(),retarded=complex_json(chi),kernel_with_contact=complex_json(kernel),current_Gram=complex_json([[np.vdot(a,b) for b in images] for a in images]),meaning='G_ij(w)=<J_i(w+E-H+ieta)^-1J_j>. chi=G(w)+conj G(-w), entrywise conjugation supplies the exchanged spectral order. K=+Hsecond from the SAME full path+mechanical source. All retained virtuals participate, including every field/spin/angular/sea state; nonconvergence remains explicit.')
        results[label]['sign_convention']='retarded is the -i commutator tensor. H(a)=H-J.a+K.a²/2 and j(a)=J-K.a, so the physical total current response is delta<j>=-(K+retarded).a. kernel_with_contact is this effective-action/curvature kernel, NOT the opposite-sign delta-j coefficient.'
        checkpoint('current_complete_'+label)
    for case in request['source_cases']:
        name=case['name'];key=case['key'];label=name+'_'+key;h=laws[name];insertion=source(data,model,name,key);raw=insertion@reference;metric=matrix(data,name+'_source_'+key+'_full_metric')
        full=mean(reference,observable(model,'neutral',matter=metric));values=[];details=[]
        for frequency in frequencies:
            solved,status=solve(h,e0,raw,frequency,p);values.append(np.vdot(raw,solved));details.append(status)
        source_results[label]=dict(full_raw_strength=full,retained_raw_strength=float(np.linalg.norm(raw)**2),frequencies=frequencies,source_sector_resolvent=complex_json(values),solves=details,retained_virtual_dimension=h.shape[0],meaning='Raw original-site source acting on NEW neutral reference, resolved in its ACTUAL charged sector. This positive-excitation source-sector resolvent is NOT the full electron addition-plus-removal Green function. Full metric precedes target projection; all RETAINED virtuals participate. Direct anti removal is not a pasted negative pole.')
        checkpoint('source_complete_'+label)
    checkpoint('complete');print(json.dumps(dict(stage='complete',current_cases=list(results),source_cases=list(source_results),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)


if __name__=='__main__':main()
