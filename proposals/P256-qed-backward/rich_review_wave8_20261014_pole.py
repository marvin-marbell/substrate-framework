"""Actual occupied spatial reference and complete allowed-momentum source poles."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import eigsh, norm
from rich_spatial_holonomy_20261014_model import build, momentum_basis, fourier_source

ROOT=Path(__file__).resolve().parent
PREFIX='rich_review_wave8_20261014_pole'


def main():
    p=json.loads((ROOT/(PREFIX+'_input.json')).read_text())
    model=build(p)
    sector_min=[];neutral=[];orth=[];translation=[]
    for n in range(13):
        mins=[]
        for k in range(3):
            u=momentum_basis(model,n,k)
            if not u.shape[1]:
                continue
            h=(u.getH()@model['H']@u).toarray()
            orth.append(float(norm(u.getH()@u-sp.eye(u.shape[1]))))
            translation.append(float(norm(model['T']@u-np.exp(2j*np.pi*k/3)*u)))
            if n==6:
                e,v=eigh(h)
                neutral.append((float(e[0]),k,u,e,v))
                mins.append(float(e[0]))
            else:
                e=eigh(h,subset_by_index=[0,0],eigvals_only=True)
                mins.append(float(e[0]))
        sector_min.append(min(mins))
    energy,k0,u0,en,vn=min(neutral,key=lambda item:item[0])
    # Every exact degenerate neutral partner retained as a projector, not ARPACK duplicate columns.
    ix=np.flatnonzero(abs(en-energy)<1e-8)
    ground=np.asarray(u0@vn[:,ix]);rank=ground.shape[1]
    source_gram=np.zeros((4,4),complex)
    records=[];z=.43+.23j
    for momentum in range(3):
        operators=fourier_source(model,momentum)
        plus=np.stack([operator@ground for operator in operators],axis=1)
        minus=np.stack([operator.getH()@ground for operator in operators],axis=1)
        gp=np.einsum('iar,ibr->ab',plus.conj(),plus)/rank
        gm=np.einsum('iar,ibr->ab',minus.conj(),minus)/rank
        source_gram+=gp+gm.T
        row={'momentum_index':momentum,'physical_momentum':float(2*np.pi*(momentum if momentum<=1 else momentum-3)/(3*p['spacing'])),
             'CAR_Gram':float(np.linalg.norm(gp+gm.T-np.eye(4))),
             'wrong_removal_no_transpose':float(np.linalg.norm(gp+gm-np.eye(4)))}
        for label,number,target,source,sign in [('addition',7,(k0+momentum)%3,plus,-1),('removal',5,(k0-momentum)%3,minus,1)]:
            u=momentum_basis(model,number,target)
            h=(u.getH()@model['H']@u).toarray()
            e,v=eigh(h)
            flat=source.reshape(model['H'].shape[0],4*rank)
            projected=np.asarray(u.getH()@flat)
            residues=v.conj().T@projected
            weights=np.sum(abs(residues)**2,axis=1)/rank
            bright=np.flatnonzero(weights>1e-10)
            amplitudes=np.einsum('nar,nbr->nab',residues.reshape(len(e),4,rank).conj(),residues.reshape(len(e),4,rank))/rank
            denominators=z+sign*(e-energy)
            kernel=np.einsum('nab,n->ab',amplitudes,1/denominators)
            row[label]=dict(target_momentum=target,dimension=len(e),gap=float(e[0]-energy),
                            first_bright_gap=float(e[bright[0]]-energy) if len(bright) else None,
                            first_bright_weight=float(weights[bright[0]]) if len(bright) else None,
                            all_gaps=(e-energy).tolist(),all_weights=weights.tolist(),
                            projection_loss=float(np.linalg.norm(flat-u@projected)),
                            source_completeness=float(np.linalg.norm(residues.conj().T@residues-projected.conj().T@projected)),
                            resolvent_real=kernel.real.tolist(),resolvent_imag=kernel.imag.tolist())
        records.append(row)
    spin=model['S'];reference_spin2=float(np.trace(ground.conj().T@(model['S2']@ground)).real/rank)
    rotation_fidelity=[]
    for s in spin:
        sg=np.asarray(s@ground)
        rotation_fidelity.append(float(np.linalg.norm(ground.conj().T@sg)**2/rank))
    source_casimir=[]
    for c in model['source']:
        double=sum(s@(s@c-c@s)-(s@c-c@s)@s for s in spin)
        source_casimir.append(float(norm(double-.75*c)))
    step=2e-5
    hp=build(p,external=step,frame=model['frame'])['H']
    hm=build(p,external=-step,frame=model['frame'])['H']
    derivative=float(norm((hp-hm)/(2*step)+model['J']))
    contact=float(norm((hp-2*model['H']+hm)/step**2-model['K']))
    bare=build(p,hopping_scale=0.,frame=model['frame'])
    bare_min=[]
    for n in (5,6,7):
        values=[]
        for k in range(3):
            u=momentum_basis(bare,n,k)
            h=(u.getH()@bare['H']@u).toarray()
            values.append(float(eigh(h,subset_by_index=[0,0],eigvals_only=True)[0]))
        bare_min.append(min(values))
    result=dict(kind='NEW_INDEPENDENT_LITERAL_OCCUPIED_SPATIAL_SOURCE_POLE_AUDIT',parameters=p,
                all_number_sector_ground_energies=sector_min,neutral_number=6,neutral_momentum=k0,neutral_rank=rank,
                neutral_energy=energy,neutral_residual=float(np.linalg.norm(model['H']@ground-energy*ground)),
                global_lowest_number_sectors=[n for n,e in enumerate(sector_min) if abs(e-min(sector_min))<1e-8],
                source_allowed_momentum_records=records,total_local_CAR_Gram_error=float(np.linalg.norm(source_gram-3*np.eye(4))),
                maximum_basis_orthogonality=max(orth),maximum_translation_eigen_error=max(translation),
                whole_H_translation=float(norm(model['H']@model['T']-model['T']@model['H'])),
                reference_frame_S2=reference_spin2,source_adjoint_halfspin_max=max(source_casimir),
                H_frame_rotation_commutators=[float(norm(model['H']@s-s@model['H'])) for s in spin],
                reference_frame_rotation_leakage=[float(np.linalg.norm(s@ground-ground@(ground.conj().T@(s@ground)))) for s in spin],
                physical_current_derivative=derivative,contact_derivative=contact,
                wrong_Noether_current_difference=float(norm(model['J']-spin[0])),
                no_hopping_number_5_6_7_energies=bare_min,
                scope='Literal 3-site interacting occupied reference, complete 12-mode Fock and all allowed spatial source momenta/all virtual states. Three momenta on ONE ring are not continuum free3D inertial mass. Field-frame adjoint halfspin, frame-reference invariance, whole spatial/field NoEther and closed dressed halfsector are distinct. No fit or rest Clifford Lorentz claim.')
    (ROOT/(PREFIX+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ROOT/(PREFIX+'_states.npz'),neutral_reference=ground)
    print(json.dumps({key:result[key] for key in ['all_number_sector_ground_energies','neutral_momentum','neutral_rank','neutral_residual','global_lowest_number_sectors','whole_H_translation','reference_frame_S2','source_adjoint_halfspin_max','physical_current_derivative','contact_derivative']},indent=2))
    print(json.dumps([dict(momentum=r['physical_momentum'],addgap=r['addition']['gap'],addbright=r['addition']['first_bright_gap'],removegap=r['removal']['gap']) for r in records],indent=2))


if __name__=='__main__':
    main()
