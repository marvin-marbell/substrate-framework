"""Only affected phase-export repair plus a new before/post-cut order discriminator."""
from pathlib import Path
import json,sys
import numpy as np
import rich_full108_recoil_review_20261017_wick as wick
from rich_full108_recoil_review_20261017_run import encode,publish
from rich_full108_recoil_review_20261017_capture import digest

ROOT=Path(__file__).resolve().parent;P='rich_full108_recoil_review_20261017_'


def cx(value):
    a=np.asarray(value,float);return a[...,0]+1j*a[...,1]


def main():
    inputpath=Path(sys.argv[1]);config=json.loads(inputpath.read_text());stem=inputpath.name.removesuffix('_input.json');physics=P+config['OWN_physics_epoch'];result=json.loads((ROOT/(physics+'_result.json')).read_text());summary={};phase_loss=0.;sum_error=0.;hermitian_error=0.
    for suffix,expected in config['OWN_physics_files'].items():
        if digest(ROOT/(physics+suffix))!=expected:raise ValueError(('own physical payload changed',suffix))
    # Original COMPLEX characters/Grams are the primary saved measurements.
    # Reconstruct their exact Fourier sector metrics; do not rerun dynamics/Wick.
    for key,row in result['cases'].items():
        for family,metric in row['current_conditioned_recoil'].items():
            characters=cx(metric['raw_character_matrices']);sectors=np.array([sum(np.exp(2j*np.pi*a*r/3)*characters[a] for a in range(3))/3 for r in range(3)])
            phase_loss=max(phase_loss,float(np.max(abs(sectors.imag))));sum_error=max(sum_error,float(np.max(abs(sectors.sum(axis=0)-cx(row['raw_insertion_Grams'][family])))));hermitian_error=max(hermitian_error,float(np.max(abs(sectors-sectors.conj().transpose(0,2,1)))))
            metric['raw_current_conditioned_total_C3_sector_Gram_matrices']=encode(sectors)
        p=np.asarray(row['joint_matter_field_C3_probability_time_coefficients']);f=np.asarray(row['frozen_joint_coefficients']);current=cx(row['currents_and_beforecut_products_time_coefficients']['J']);fcurrent=cx(row['original_law_probes_on_frozen_generator_time_coefficients']['J'])
        summary[key]=dict(raw_source_weight=row['raw_canonical_source_norm2'],matter_modular_recoil_time_coefficient2=p[:,:,2].sum(axis=1),field_modular_recoil_time_coefficient2=p[:,:,2].sum(axis=0),frozen_joint_time_derivative_max=float(np.max(abs(f[:,:,1:]))),initial_field_signed_populations=row['field_initial_signed_C3_populations'],signed_current_initial_and_coefficient1=current,signed_current_coefficient1_coupled_minus_frozen_original_probe=current[:,1]-fcurrent[:,1],raw_J_Gram_diagonal=np.diag(cx(row['raw_insertion_Grams']['J'])).real,raw_QJ_Gram_diagonal=np.diag(cx(row['raw_insertion_Grams']['QJ'])).real,WRONG_missing_shell3_QJ_image_norm2=row['WRONG_intermediate_cut_QJ_difference_image_norm2'],WRONG_missing_shell3_JQ_image_norm2=row['WRONG_intermediate_cut_JQ_difference_image_norm2'],all108_fullFock_virtual_current_Gram_missing_in_meanonly_closure=np.diag(cx(row['connected_fullFock_virtual_current_Gram_omitted_by_meanonly_closure'])).real,full_H_mean=cx(row['full_H_energy']),full_H_variance_including_cross_terms=cx(row['full_H_variance']),frozen_real_body_convection_coefficient1_norm=float(np.linalg.norm(row['frozen_real_body_convection_coefficient1'])))
    sys.path.insert(0,str(ROOT/config['frozen']['code_snapshot']))
    from rich_spatial_translation_20261015_model import shell_fields
    law=config['frozen']['law'];field=shell_fields(law);ext=shell_fields(dict(law,shell_cutoff=3));indices=np.array([p*len(ext['shell_states'])+ext['shell_states'].index(s) for p in range(field['nphi']) for s in field['shell_states']]);wrong_order=[];correct_order=[]
    # NEW physical discriminator only: exact Q and Peierls J commute BEFORE
    # projection in THIS coordinate-only law. Two formulas do not demand a
    # nonzero difference; cut-in-the-middle can manufacture noncommutation.
    U=field['field_translation'].toarray()
    for row in result['operator_covariance']['vertices']:
        pol,sign,grade=row['polarization'],row['profile_sign'],row['measured_joint_grade'];q=(ext['coordinates'][2*pol]+1j*ext['coordinates'][2*pol+1]).toarray()/np.sqrt(2);small=q[np.ix_(indices,indices)]
        qgrade=int(np.argmin([np.linalg.norm(U@small@U.conj().T-np.exp(-2j*np.pi*g/3)*small) for g in range(3)]));counter=q if qgrade==(-grade)%3 else q.conj().T;small=counter[np.ix_(indices,indices)]
        current=np.array([-(ext['derivatives'][2*pol][c]+1j*sign*ext['derivatives'][2*pol+1][c]).toarray()/np.sqrt(2) for c in range(16)]);js=current[:,indices][:,:,indices]
        correct_order.append((counter@current-current@counter)[:,indices][:,:,indices]);wrong_order.append(small@js-js@small)
    correct_order=np.array(correct_order);wrong_order=np.array(wrong_order);order_tests={}
    with np.load(ROOT/(physics+'_raw.npz')) as raw:
        A=raw['primitive_body_coefficients']
        for number in (55,53):
            V=raw['Slater_'+str(number)];f=raw[f'N{number}_baseline_field_state'];M=wick.moments(A,A,V@V.conj().T)
            right=np.diag(wick.gram(A,V,f,correct_order,M)).real;wrong=np.diag(wick.gram(A,V,f,wrong_order,M)).real
            order_tests[str(number)]=dict(correct_beforecut_QJ_minus_JQ_image_norm2=right,WRONG_postcut_QJ_minus_JQ_image_norm2=wrong)
            if np.max(abs(right))>1e-24 or np.min(wrong)<1e-12:raise ValueError(('before/post-cut physical order discriminator failed',number,right,wrong))
    analysis=dict(kind='OWN_PHASE_COMPLETE_FULL108_RECOIL_REVIEW_ANALYSIS',physical_epoch=physics,physics_natural_exit=json.loads((ROOT/(physics+'_exit.json')).read_text())['natural_exit'],phase_export_repair=dict(original_issue='wick1 used.real on current-sector GRAM MATRICES, which preserves probabilities/diagonals but can drop off-diagonal imaginary phases. Primary full COMPLEX character matrices, full Grams, response coefficients and raw own states were already saved.',max_previously_unexported_sector_Gram_imaginary_part=phase_loss,corrected_complex_sector_sum_to_full_Gram_residual=sum_error,corrected_sector_Gram_Hermiticity_residual=hermitian_error,scope='Only Fourier export rebuilt from OWN saved complex characters. No nine-case physical approval rerun or author data used. run.py one.real export cast fixed in later source epoch.'),new_physical_order_discriminator=order_tests,scope_correction='QJ and JQ are two explicitly evaluated BEFORE-CUT products, but in THIS Peierls coordinate-only law they COMMUTE to roundoff. A demanded nonzero difference is incorrect. Intermediate P2 can spuriously manufacture an order difference; shell3 restores the unprojected-coordinate product.',cases_summary=summary,physical_result_with_complex_sector_metrics=result)
    if max(sum_error,hermitian_error)>2e-10:raise ValueError('complex current-sector metric reconstruction failed')
    publish(ROOT/(stem+'_result.json'),analysis);print(json.dumps(encode(dict(physics_natural_exit=analysis['physics_natural_exit'],phase_export_repair=analysis['phase_export_repair'],new_physical_order_discriminator=order_tests,cases_summary=summary))),flush=True)

if __name__=='__main__':main()
