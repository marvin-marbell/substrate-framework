#!/usr/bin/env python3
"""Keep integer-Gauss transverse characters in Gaussian reference endpoint jets."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in (
    'covariant_gauss_affine_endpoint_jets.py','covariant_literal_wilson_loop_control.py',
    'covariant_reference_gauss_log_jets.py','covariant_reference_log_jet_words.py',
    'covariant_photon_energy_control.py','covariant_induced_photon.py','sea_selection.py','rotation_band.py')}
from covariant_induced_photon import Parameters,phi_frame
from covariant_photon_energy_control import gaussian_energy_form_control
from covariant_reference_log_jet_words import sf_majorants,compositions


def native_integer_affine_kernels() -> list[dict[str,Any]]:
    """Actual integer four-edge electric Gaussian; all canonical occupations."""
    rows=[]
    for charge in (0,1):
        for occupied in itertools.combinations(range(8),4+charge):
            rho=[sum(mode//2==x for mode in occupied)-1-int(x==0)*charge for x in range(4)]
            if sum(rho):raise ArithmeticError('Integer body/Ref total charge is not zero')
            # SAME literal shortest source tree: source paths avoid edge2.
            paths=np.array([[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]],int)
            integer=(np.array(rho)@paths).tolist()
            if any(integer[x]-integer[(x-1)%4]!=rho[x] for x in range(4)):
                raise ArithmeticError('Literal body-to-Ref source tree violates integer Gauss')
            shift=Fraction(sum(integer),4)
            longitudinal=sum((Fraction(e)-shift)**2 for e in integer)
            for beta in (4.,16.,64.):
                cutoff=math.ceil(12*math.sqrt(beta));m=np.arange(-cutoff,cutoff+1)
                for increment in (.1,.4):
                    exact=np.sum(np.exp(-2*(m+float(shift))**2/beta+1j*m*increment))
                    zero=np.sum(np.exp(-2*m*m/beta+1j*m*increment))
                    phase=np.exp(-1j*float(shift)*increment)
                    # Poisson images retain the literal fractional cycle coset.
                    images=np.arange(-8,9)
                    x=increment+2*math.pi*images
                    poisson=math.sqrt(math.pi*beta/2)*np.sum(np.exp(-beta*x*x/8-1j*float(shift)*x))
                    relative=abs(exact-poisson)/max(1.,abs(exact))
                    if relative>1e-12:raise ArithmeticError('Integer/Floquet Poisson heat identity failed')
                    rows.append(dict(body_charge=charge,Ref_hole_charge=-charge,canonical_occupation=list(occupied),
                        actual_integer_density=rho,actual_integer_particular_flux=integer,
                        actual_transverse_cycle_shift=str(shift),actual_longitudinal_norm_squared=str(longitudinal),
                        beta=beta,angle_increment=increment,
                        full_integer_heat_real=float(np.real(exact)),full_integer_heat_imag=float(np.imag(exact)),
                        exact_Poisson_image_identity_relative_error=relative,
                        omission_of_transverse_character_relative_error=float(abs(exact-zero)/max(1.,abs(exact))),
                        unwrapped_phase_only_approximation_relative_error=float(abs(exact-phase*zero)/max(1.,abs(exact))),
                        exact_electric_energy_identity='sum_e E_e²=4(m+cycle_shift)²+longitudinal_norm²'))
    return rows


def native_dressing_control() -> list[dict[str,Any]]:
    gamma4,_=phi_frame(Parameters(kappa_power=400))
    values,U=eigh(1j*gamma4[0]@gamma4[1]@gamma4[2]);embedding=U[:,values>0]
    _,basis=eigh(embedding.conj().T@gamma4[0]@embedding);embedding=embedding@basis[:,::-1]
    gamma=np.array([embedding.conj().T@g@embedding for g in gamma4[:3]])
    def half(phases: np.ndarray) -> np.ndarray:
        B=np.eye(4,dtype=complex)/.18;F=np.zeros((4,4),complex)
        for phase,(x,y,mu,orientation) in zip(phases,((0,1,1,1),(1,2,2,1),(2,3,1,-1),(3,0,2,-1))):
            B[x,y]-=.5*phase;B[y,x]-=.5*phase.conjugate()
            z=orientation*gamma[mu,0,1]/2
            F[x,y]+=z*phase;F[y,x]-=z*phase.conjugate()
        inverse=np.linalg.inv(B)
        T=np.block([[inverse,-inverse@F],[-F.conj().T@inverse,B+F.conj().T@inverse@F]])
        perm=[0,4,1,5,2,6,3,7];T=T[np.ix_(perm,perm)]
        eigenvectors=eigh(T)
        return (eigenvectors[1]*np.sqrt(eigenvectors[0]))@eigenvectors[1].conj().T
    mean=half(np.ones(4,complex));ev,eu=eigh(mean@mean);positive=eu[:,ev<1]
    packet=np.array([.5j,-.25,.2+.3j,.1]);packet/=np.linalg.norm(packet)
    psi=positive@positive.conj().T@np.kron(packet[:,None],np.eye(2));psi/=np.linalg.norm(psi,'fro')
    coefficients=psi@embedding.conj().T
    neutral=list(itertools.combinations(range(8),4));charged=list(itertools.combinations(range(8),5))
    charged_index:dict[tuple[int,...],int]={occupied:i for i,occupied in enumerate(charged)}
    def source(phases: np.ndarray) -> np.ndarray:
        result=np.zeros((4*len(charged),len(neutral)),complex)
        for column,occupied in enumerate(neutral):
            mask=sum(1<<mode for mode in occupied)
            for mode in range(8):
                if mode in occupied:continue
                row=charged_index[tuple(sorted((*occupied,mode)))]
                below=(mask&((1<<mode)-1)).bit_count()
                for ref in range(4):result[ref*len(charged)+row,column]=(-1)**(below+ref)*coefficients[mode,ref]*phases[mode//2]
        return result
    bare=source(np.ones(4,complex));rows=[]
    slope=np.array([0.,-.25,-.5,.25])
    for theta in (-.4,-.1,.1,.4):
        chi=theta*slope;Dsite=np.diag(np.repeat(np.exp(-1j*chi),2))
        local=half(np.full(4,np.exp(.25j*theta)))
        tree_phases=np.ones(4,complex);tree_phases[2]=np.exp(1j*theta)
        tree=half(tree_phases)
        dn=np.array([np.exp(-1j*(sum(chi[mode//2] for mode in occupied)-sum(chi))) for occupied in neutral])
        dc=np.array([np.exp(-1j*(sum(chi[mode//2] for mode in occupied)-sum(chi))) for occupied in charged])
        literal=source(np.exp(1j*chi))
        transformed=np.tile(dc,4)[:,None]*literal*dn.conjugate()[None,:]
        source_error=float(np.linalg.norm(transformed-bare,2))
        matter_error=float(np.linalg.norm(tree-Dsite@local@Dsite.conj().T,2))
        if max(source_error,matter_error)>1e-12:raise ArithmeticError('Actual literal-tree/source unitary pullback failed')
        actual_shifts=[];connection=[]
        for occupied in neutral:
            rho=[sum(mode//2==x for mode in occupied)-1 for x in range(4)]
            electric=np.array(rho)@np.array([[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]])
            actual_shifts.append(float(electric.mean()))
            connection.append(float(np.array(rho)@slope))
        momentum_error=float(np.max(abs(np.array(actual_shifts)-np.array(connection))))
        rows.append(dict(coexact_loop_angle=theta,actual_literal_source_tree_chord=2,
            exact_positive_matter_half_unitary_defect=matter_error,
            actual_Ref4_CAR_Wilson_source_pullback_defect=source_error,
            physical_transverse_momentum_affine_connection_defect=momentum_error,
            source_full_neutral_CAR_dimension=len(neutral),source_full_charged_CAR_dimension=len(charged),
            physical_Ref_hole_channels=4,largest_retained_transverse_density_connection=float(np.max(np.abs(np.array(connection))))))
    return rows



def run() -> dict[str,Any]:
    p=Parameters(kappa_power=400);volume=p.spatial_size**3
    loop_path=ROOT/'covariant_literal_wilson_loop_control_capsule_20261008.json'
    loop_bytes=loop_path.read_bytes();loop=json.loads(loop_bytes)
    Bloop=loop['macro_literal_loop_A_word_factor'];Bpath=loop['macro_literal_source_path_A_word_factor']
    sf=sf_majorants(volume,5)['SF_relative_coefficients']
    density=(4*volume+1)*Bpath
    # Use the conservative established full-CAR/free-band whitening factor.
    energy=gaussian_energy_form_control(p,9,.25)
    d=math.log(8)+energy['photon_maximum_energy']/2
    endpoint=[1.];endpoint_rows=[]
    for n in range(1,6):
        terms=[]
        for source_density_grade in range(n+1):
            sf_grade=n-source_density_grade
            price=float(sf[sf_grade])*Bloop**sf_grade*density**source_density_grade/math.factorial(source_density_grade)
            terms.append(dict(active_SF_grade=sf_grade,Gauss_density_phase_grade=source_density_grade,norm_coefficient_upper=price))
        endpoint.append(sum(row['norm_coefficient_upper'] for row in terms)*math.exp(n*d)*(1+2**-40))
        endpoint_rows.append(dict(grade=n,actual_Gauss_phase_terms=terms,
            whitened_endpoint_coefficient_log10_upper=math.log10(endpoint[n])))
    C=Fraction(13,32)*(4*volume+1)**2
    transfer=[1.]
    for n in range(1,6):
        transfer.append(sum(sum(endpoint[i]*endpoint[n-2*r-i] for i in range(n-2*r+1))*float(C**r)/math.factorial(r)
            for r in range(n//2+1))*(1+2**-40))
    logjets=[0.]+[sum(math.prod(transfer[j] for j in word) for word in compositions(n)) for n in range(1,6)]
    for n,row in enumerate(endpoint_rows,1):row['Gauss_affine_SF_Coulomb_log_jet_log10_upper']=math.log10(logjets[n])
    terminal=[]
    for weight in range(5):
        for prefix in compositions(weight):
            for last in range(1,6):
                if weight+last<5:continue
                word=prefix+(last,);length=len(word)
                price=-p.beta_power/2*sum(word)+length*math.log10(p.time_slices)
                price+=sum(math.log10(logjets[j]) for j in word)-math.lgamma(length+1)/math.log(10)
                terminal.append(dict(jet_grades=list(word),conditional_log10_upper=price))
    peak=max(row['conditional_log10_upper'] for row in terminal)
    total=peak+math.log10(sum(10**(row['conditional_log10_upper']-peak) for row in terminal))
    native=native_integer_affine_kernels()
    dressing=native_dressing_control()
    inputs=dict(parameters=asdict(p),integer_active_charge=1,integer_Ref_hole_charge=-1,
        original_Ref_background=-4,full_active_orbitals_per_site=4,filled_background_per_site=2,
        initial_reference_energy=.25,maximum_reference_jet_grade=5,maximum_terminal_grade=9,
        literal_loop_capsule=loop_path.name,literal_loop_capsule_sha256=hashlib.sha256(loop_bytes).hexdigest(),
        literal_loop_input_descriptor_sha256=loop['input_descriptor_sha256'],native_active_numbers=[4,5],
        native_box_graph='four-edge cycle, invariant2spin canonical control',native_betas=[4,16,64],
        native_angle_increments=[.1,.4],native_unitary_loop_angles=[-.4,-.1,.1,.4],
        actual_native_source_paths=[[0,0,0,0],[-1,0,0,0],[-1,-1,0,0],[0,0,0,1]])
    return dict(input_descriptor=inputs,input_descriptor_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
        source_at_import={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        source_bytes_at_import={name:data.decode() for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        actual_integer_Gauss_affine_kernel_rows=native,
        actual_literal_matter_source_receiver_dressing_rows=dressing,
        maximum_actual_transverse_character_omission_error=max(row['omission_of_transverse_character_relative_error'] for row in native),
        macro_density_Wilson_generator_coefficient_upper=density,active_SF_Gauss_affine_endpoint_rows=endpoint_rows,
        conditional_terminal_words=terminal,conditional_terminal_sum_log10_upper=total,
        equations=[
            'Every literal integer Gauss solution is E=Eparticular(rho)+integercycle. Orthogonal Hodge projection gives a rho-dependent transverse character; Coulomb energy alone does NOT remove it.',
            'On the noncompact coexact Gaussian reference, retain D_rho(g,A)=exp(-i g G_rho(A)), G_rho=sum_x(n_x-2) A_path(x)-A_path(Ref). Eparticular=sum_x rho_x path(x-to-Ref) has divE=rho; its heat phase is exp(-i g Eparticular,T dot (Aprime-A)). The fixed background and Ref hole are INCLUDED. ||G_rho P_energy||<=(4V+1)B_path; no real-PH density sector is dropped.',
            'The retained reference transfer is SF_tree,g D_rho(g) Kph exp(-g² C_rho) D_rho(g)dag SF_tree,g. Its endpoint E_g=SF_tree,g D_rho(g) has ordinary coefficients sum_(i+j=n) SF_tree,i (-iG_rho)^j/j!, including every higher density/current contact.',
            'At g0, SF0 is positive and field-independent. SF0^-1 SF_i is the adjoint of SF_i SF0^-1, so the established full-CAR SF coefficient bounds also bound the left relative endpoint. The displayed conservative free-band/field whitening factor is retained.',
            'Whitened transfer coefficients contain sum_(i+j+2r=n) Y_i (-C_rho)^r Y_jdag/r!. Thus the exact ordered-composition log-resolvent norm bounds include SF, affine transverse phase and longitudinal Coulomb insertions together throughgrade5.',
            'With the ACTUAL body-to-Ref Wilson tree, SF_tree,g=D_rho SF_local,g D_rho dag. Hence E_g=D_rho SF_local,g and K_tree=D_rho K_local D_rho dag on the Gaussian coexact reference. Source/receiver transform with BOTH charged/neutral D factors; the literal Wilson source becomes bare only in this transformed representation. The physical transverse electric field is Pi+g dG_rho, which maps back to Pi. No source retrofit, zero-time subtraction, or omitted external density-current connection is allowed.',
            'The compact integer-cycle kernels retain ALL Poisson images and their fractional cosets. These are affine lattice shifts produced by integer charges, NOT fractional-charge representations. Harmonic/statistical transfer is retained exactly.',
            'These are zero-g reference coefficient bounds. The 65 terminal words are conditional on a matching along-real-g remainder and actual fullGauss/commonrange source/receiver identification; that theorem is NOT asserted.'],
        microscopic_commonrange_or_source_admission_proved=False,
        scope='Actual integer-Gauss transverse-character discriminator and mixed Gaussian reference endpoint/log-jet prices. No microscopic supplier replacement or actual source functional change.')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();result=run();result['runtime_seconds']=time.monotonic()-started
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('source_at_import','input_descriptor_sha256',
        'maximum_actual_transverse_character_omission_error','active_SF_Gauss_affine_endpoint_rows',
        'conditional_terminal_sum_log10_upper','runtime_seconds')}))


if __name__=='__main__':
    main()
