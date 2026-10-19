"""Recover completed saved eigenspaces after metadata serialization failure.
NO eigensolver replay: actual same-H residual/current/source contractions only.
"""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

ROOT=Path(__file__).resolve().parent
PREFIX='rich_photon_collective_20261015_'

def run():
    started=time.monotonic();mode=sys.argv[1];request=json.loads((ROOT/(PREFIX+mode+'_physics.json')).read_text());base=PREFIX+request['base_mode'];archive=np.load(ROOT/(base+'_primitive.npz'));coordinates=archive['coordinates'];factors=archive['factors'];phi_h=archive['phi_h'];kinetic=archive['kinetic'];profiles=archive['profiles'];offset=float(archive['offset']);ground_file=np.load(ROOT/(base+'_ground.npz'));ground=ground_file['state'];e0=float(ground_file['energy']);nf=len(kinetic);nb=5*nf;n=len(ground)//nb;matrices=[sparse.load_npz(ROOT/(base+f'_Hcomponent{k}.npz')) for k in range(len(profiles))];flatf=factors.reshape(len(matrices),nb);kfield=sparse.csr_matrix(kinetic);source_data=json.loads((ROOT/(base+'_spectra.json')).read_text())
    def action(vector):
        x=np.asarray(vector).reshape(n,5,nf);out=(kfield@x.reshape(n*5,nf).T).T.reshape(x.shape)+np.einsum('ab,mbf->maf',phi_h,x,optimize=True)-offset*x;flat=x.reshape(n,nb);result=out.reshape(n,nb)
        for matrix,weight in zip(matrices,flatf):result+=(matrix@flat)*weight
        return result.ravel()
    h=LinearOperator((n*nb,n*nb),matvec=action,rmatvec=action,dtype=complex)
    with np.load(ROOT/request['vectors_file']) as saved:energies=saved['energies'];vectors=saved['vectors']
    residuals=np.array([np.linalg.norm(h@vectors[:,k]-energies[k]*vectors[:,k]) for k in range(len(energies))]);orthogonality=float(np.linalg.norm(vectors.conj().T@vectors-np.eye(len(energies))));qsource=np.empty((len(ground),8),complex);jsource=np.empty_like(qsource);gx=ground.reshape(n,nb)
    for axis in range(8):
        q=(ground.reshape(n,5,nf)*coordinates[axis]).ravel();q-=np.vdot(ground,q)*ground;current=np.zeros_like(gx)
        for k in range(2,len(matrices),2):
            coefficient=profiles[k,axis]
            if coefficient:current+=coefficient*((matrices[k]@gx)*flatf[k+1]-(matrices[k+1]@gx)*flatf[k])
        j=current.ravel();j-=np.vdot(ground,j)*ground;qsource[:,axis]=q;jsource[:,axis]=j
    amplitudes=dict(coordinate=vectors.conj().T@qsource,current=vectors.conj().T@jsource);norms=dict(coordinate=(abs(qsource)**2).sum(axis=0),current=(abs(jsource)**2).sum(axis=0));gaps=energies-e0;groups=[]
    for k,energy in enumerate(gaps):
        tolerance=max(1e-10,5*float(residuals[k]),5*groups[-1]['max_residual'] if groups else 0.)
        if not groups or abs(energy-groups[-1]['energy'])>tolerance:groups.append(dict(energy=float(energy),multiplicity=0,coordinate_axis_weights=[0.]*8,current_axis_weights=[0.]*8,max_residual=0.,complete_below_upper_edge=False,vector_indices=[]))
        group=groups[-1];group['multiplicity']+=1;group['vector_indices'].append(k);group['max_residual']=max(group['max_residual'],float(residuals[k]))
        for label,amplitude in amplitudes.items():group[label+'_axis_weights']=(np.array(group[label+'_axis_weights'])+abs(amplitude[k])**2).tolist()
    closure=[];covered=set()
    for band in range(2):
        axes=list(range(4*band,4*band+4));pole=[];expected=[]
        for axis in axes:
            source=source_data['spectra'][axis]['coordinate'];index=int(np.argmax(source['weights']));pole.append(source['energies'][index]);expected.append(source['weights'][index])
        center=float(np.mean(pole));indices=np.flatnonzero(abs(gaps-center)<request['source_branch_window']);capture=np.sum(abs(amplitudes['coordinate'][indices][:,axes])**2,axis=0);relative=np.abs(capture-np.array(expected))/np.array(expected);passed=bool(len(indices)==4 and float(max(relative))<request['source_weight_relative_tolerance'] and float(max(residuals[indices]))<request['accepted_eigen_residual'] and orthogonality<1e-8)
        closure.append(dict(band=band+1,source_branch_center=center,actual_vector_indices=indices.tolist(),actual_multiplicity=len(indices),full_source_Lanczos_atom_weight=expected,completed_group_weight=capture.tolist(),relative_weight_disagreement=relative.tolist(),passed=passed,meaning='Four independent realquadrature/polarization channels from SAVED recoveredvectors close actual fullsource atomweights. Analysisprojection only,not pulsepreparationfilter.'))
        if passed:covered.update(map(int,indices))
    for group in groups:
        group['complete_below_upper_edge']=bool(group['vector_indices'] and set(group['vector_indices']).issubset(covered));group['completeness_meaning']='Complete measured brightgroup after fourmissingchannels recovered and fullsource/eigenresidual closure' if group['complete_below_upper_edge'] else 'Observed dark/uppergroup only; completenessnotasserted'
    original=json.loads((ROOT/(base+'_spectra.json')).read_text());old_capture=np.sum(np.array([g['coordinate_axis_weights'] for g in original['groups']]),axis=0);captureQ=np.sum(abs(amplitudes['coordinate'])**2,axis=0);captureJ=np.sum(abs(amplitudes['current'])**2,axis=0)
    result=dict(kind='NATURALLY_COMPLETED_SAVED_VECTOR_GROUP_METADATA_AND_PHYSICAL_RECOVERY',base_mode=request['base_mode'],vectors_file=request['vectors_file'],energy=e0,groups=groups,source_closure=closure,full_coordinate_axis_norm=norms['coordinate'].tolist(),full_current_axis_norm=norms['current'].tolist(),old13_coordinate_capture=old_capture.tolist(),completed_coordinate_capture=captureQ.tolist(),completed_current_capture=captureJ.tolist(),raw_uncaptured_coordinate=(norms['coordinate']-captureQ).tolist(),raw_uncaptured_current=(norms['current']-captureJ).tolist(),actual_H_eigen_residuals=residuals.tolist(),orthogonality_defect=orthogonality,original_solver_history='UNKNOWN: solver completedvectors but failedJSONserialization of complexhistory; no historyinvented,no eigenproblemreplay.',wall_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,meaning='Original13 partialprojector flags are notauthoritative. Completed17saved physicalvectors are independently contracted with SAME microscopicH,current,Q and fullsource norms. Only brightgroups passing explicitclosure arecomplete; current/coordinate highfrequencytails stayraw.')
    (ROOT/(PREFIX+mode+'_data.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(mode=mode,source_closure=closure,max_eigen_residual=float(max(residuals)),wall_seconds=result['wall_seconds'],peak_RSS_KiB=result['peak_RSS_KiB'])),flush=True)
    if not all(item['passed'] for item in closure):raise RuntimeError('Saved modevectors do not close actual fullsource brightgroups; partialdata preserved')

if __name__=='__main__':run()
