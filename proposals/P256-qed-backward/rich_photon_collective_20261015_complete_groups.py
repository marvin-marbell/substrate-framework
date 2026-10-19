"""SAME-H completion of four omitted bright degenerate modes, not pole polishing.
Block solver constrained against saved13 vectors; full source weights stay raw.
"""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, lobpcg

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'

def run():
    started=time.monotonic();mode=sys.argv[1];request=json.loads((ROOT/(PREFIX+mode+'_physics.json')).read_text());base=PREFIX+request['base_mode'];archive=np.load(ROOT/(base+'_primitive.npz'));coordinates=archive['coordinates'];factors=archive['factors'];phi_h=archive['phi_h'];kinetic=archive['kinetic'];profiles=archive['profiles'];offset=float(archive['offset']);ground_file=np.load(ROOT/(base+'_ground.npz'));ground=ground_file['state'];e0=float(ground_file['energy']);nf=len(kinetic);nb=5*nf;n=len(ground)//nb;matrices=[sparse.load_npz(ROOT/(base+f'_Hcomponent{k}.npz')) for k in range(len(profiles))];flatf=factors.reshape(len(matrices),nb);kfield=sparse.csr_matrix(kinetic);source_data=json.loads((ROOT/(base+'_spectra.json')).read_text())
    def action(vector):
        x=np.asarray(vector).reshape(n,5,nf);out=(kfield@x.reshape(n*5,nf).T).T.reshape(x.shape)+np.einsum('ab,mbf->maf',phi_h,x,optimize=True)-offset*x;flat=x.reshape(n,nb);result=out.reshape(n,nb)
        for matrix,weight in zip(matrices,flatf):result+=(matrix@flat)*weight
        return result.ravel()
    def actions(vectors):return np.column_stack([action(vector) for vector in np.asarray(vectors).T])
    h=LinearOperator((n*nb,n*nb),matvec=action,rmatvec=action,matmat=actions,rmatmat=actions,dtype=complex)
    with np.load(ROOT/(base+'_lowgroups.npz')) as old:known=old['vectors'];old_energies=old['energies']
    qsource=np.empty((len(ground),8),complex)
    for axis in range(8):
        vector=(ground.reshape(n,5,nf)*coordinates[axis]).ravel();vector-=np.vdot(ground,vector)*ground;qsource[:,axis]=vector
    full_qnorm=(abs(qsource)**2).sum(axis=0);old_capture=(abs(known.conj().T@qsource)**2).sum(axis=0);qsource-=known@(known.conj().T@qsource);gram=qsource.conj().T@qsource;remaining,rotation=eigh(gram);count=request['missing_vectors'];seed=qsource@rotation[:,-count:];seed/=np.sqrt(remaining[-count:])[None,:];del qsource
    diagonal=np.broadcast_to(np.diag(phi_h).real[:,None]+np.diag(kinetic).real[None,:]-offset,(n,5,nf)).copy().reshape(n,nb)
    for matrix,weight in zip(matrices,flatf):diagonal+=matrix.diagonal().real[:,None]*weight
    inverse=1/(diagonal.ravel()-e0+1.)
    def precondition(vector):return inverse*np.asarray(vector).reshape(-1)
    pre=LinearOperator(h.shape,matvec=precondition,rmatvec=precondition,matmat=lambda x:inverse[:,None]*x,rmatmat=lambda x:inverse[:,None]*x,dtype=complex)
    values,missing,lambda_history,residual_history=lobpcg(h,seed,Y=known,M=pre,tol=request['eigen_tolerance'],maxiter=request['maximum_iterations'],largest=False,retLambdaHistory=True,retResidualNormsHistory=True)
    missing_residual=[float(np.linalg.norm(action(missing[:,k])-values[k]*missing[:,k])) for k in range(count)];del seed,diagonal,inverse,pre
    combined=np.column_stack([known,missing]);del known,missing
    metric=combined.conj().T@combined;metric_values,metric_vectors=eigh(metric);combined=combined@(metric_vectors@np.diag(metric_values**-.5)@metric_vectors.conj().T);acted=actions(combined);energies,rotation=eigh(combined.conj().T@acted);vectors=combined@rotation;del combined,acted
    residuals=np.array([np.linalg.norm(action(vectors[:,k])-energies[k]*vectors[:,k]) for k in range(len(energies))]);qsource=np.empty((len(ground),8),complex);jsource=np.empty_like(qsource);gx=ground.reshape(n,nb)
    for axis in range(8):
        q=(ground.reshape(n,5,nf)*coordinates[axis]).ravel();q-=np.vdot(ground,q)*ground;current=np.zeros_like(gx)
        for k in range(2,len(matrices),2):
            coefficient=profiles[k,axis]
            if coefficient:current+=coefficient*((matrices[k]@gx)*flatf[k+1]-(matrices[k+1]@gx)*flatf[k])
        j=current.ravel();j-=np.vdot(ground,j)*ground;qsource[:,axis]=q;jsource[:,axis]=j
    amplitudes=dict(coordinate=vectors.conj().T@qsource,current=vectors.conj().T@jsource);norms=dict(coordinate=(abs(qsource)**2).sum(axis=0),current=(abs(jsource)**2).sum(axis=0));groups=[];gaps=energies-e0
    for k,energy in enumerate(gaps):
        tolerance=max(1e-10,5*float(residuals[k]),5*groups[-1]['max_residual'] if groups else 0.)
        if not groups or abs(energy-groups[-1]['energy'])>tolerance:groups.append(dict(energy=float(energy),multiplicity=0,coordinate_axis_weights=[0.]*8,current_axis_weights=[0.]*8,max_residual=0.,complete_below_upper_edge=False,vector_indices=[]))
        group=groups[-1];group['multiplicity']+=1;group['vector_indices'].append(k);group['max_residual']=max(group['max_residual'],float(residuals[k]))
        for label,amplitude in amplitudes.items():group[label+'_axis_weights']=(np.array(group[label+'_axis_weights'])+abs(amplitude[k])**2).tolist()
    evidence=[];covered=set()
    for band in range(2):
        axes=list(range(4*band,4*band+4));pole=[];expected=[]
        for axis in axes:
            source=source_data['spectra'][axis]['coordinate'];index=int(np.argmax(source['weights']));pole.append(source['energies'][index]);expected.append(source['weights'][index])
        center=float(np.mean(pole));selected=np.flatnonzero(abs(gaps-center)<request['source_branch_window']);capture=np.sum(abs(amplitudes['coordinate'][selected][:,axes])**2,axis=0);relative=np.abs(capture-np.array(expected))/np.array(expected);passed=bool(len(selected)==4 and float(max(relative))<request['source_weight_relative_tolerance'] and float(max(residuals[selected]))<request['accepted_eigen_residual'])
        evidence.append(dict(band=band+1,source_branch_center=center,actual_vector_indices=selected.tolist(),actual_multiplicity=len(selected),full_source_Lanczos_atom_weight=expected,completed_group_weight=capture.tolist(),relative_weight_disagreement=relative.tolist(),passed=passed,meaning='Four independent real quadrature/polarization channels recovered. Full-source atom residues crosscheck actual eigenprojector; higher-frequency tail remains raw. Source basis used ONLY as eigensolver seed,never a pulse filter/postselection.'))
        if passed:covered.update(map(int,selected))
    for group in groups:
        group['complete_below_upper_edge']=bool(group['vector_indices'] and set(group['vector_indices']).issubset(covered))
        group['completeness_meaning']='Complete for measured bright coordinate/current group after block recovery and independent full-source weight closure' if group['complete_below_upper_edge'] else 'Observed group only; dark/upperwindow completeness not asserted'
    result=dict(kind='SAME_H_MISSING_DEGENERATE_BRIGHT_GROUP_COMPLETION_NOT_THIRD_CONSTRUCTION',base_mode=request['base_mode'],energy=e0,groups=groups,source_closure=evidence,full_coordinate_axis_norm=norms['coordinate'].tolist(),full_current_axis_norm=norms['current'].tolist(),old13_coordinate_capture=old_capture.tolist(),completed_coordinate_capture=(abs(amplitudes['coordinate'])**2).sum(axis=0).tolist(),completed_current_capture=(abs(amplitudes['current'])**2).sum(axis=0).tolist(),raw_uncaptured_coordinate=(norms['coordinate']-(abs(amplitudes['coordinate'])**2).sum(axis=0)).tolist(),raw_uncaptured_current=(norms['current']-(abs(amplitudes['current'])**2).sum(axis=0)).tolist(),missing_block_eigen_residuals=missing_residual,original13_energies=old_energies.tolist(),recovered_block_energies=dict(real=np.asarray(values).real.tolist(),imag=np.asarray(values).imag.tolist()),remaining_seed_Gram_eigenvalues=remaining.tolist(),orthogonality_defect=float(np.linalg.norm(vectors.conj().T@vectors-np.eye(len(energies)))),lambda_history=[dict(real=np.asarray(item).real.tolist(),imag=np.asarray(item).imag.tolist()) for item in lambda_history],residual_history=[dict(real=np.asarray(item).real.tolist(),imag=np.asarray(item).imag.tolist()) for item in residual_history],wall_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,interpretation='Original below-window flags didnotprove degenerate-group completeness. This repairs missing physical source-channel states on SAME H,not a mass/coupling/cut change or a request to polish every pole. Only bright groups with explicit source closure are complete; full current/coordinate tails remain measured,not normalized away.')
    np.savez_compressed(ROOT/(PREFIX+mode+'_vectors.npz'),energies=energies,vectors=vectors);(ROOT/(PREFIX+mode+'_data.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(mode=mode,source_closure=evidence,missing_residual=missing_residual,wall_seconds=result['wall_seconds'],peak_RSS_KiB=result['peak_RSS_KiB'])),flush=True)
    if not all(item['passed'] for item in evidence):raise RuntimeError('Bright-group completion didnotmeet explicit source/eigen residual closure; raw results preserved')

if __name__=='__main__':run()
