#!/usr/bin/env python3
"""New C-symmetrized prepared macro density, with actual CAR/field consumers.

This is a new preparation, not a C claim about the previous uniform64 density.
It stays in the fixed onsite neutral Ref4 sector and does not describe an
external electron plus reference hole. No ground/photon-vacuum admission.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import expm
from rotation_band import CHARGE_CONJUGATION
from induced_phi_backreaction import PhiConfig,models,scalar_paths,SELECTED,matrix_unit_basis,IMPORT_BYTES as BASE_BYTES

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES={**BASE_BYTES,**{name:(ROOT/name).read_bytes() for name in (
    'induced_phi_ir_mean_certificate.py','induced_phi_ir_well_transfer.py',
    'induced_phi_ir_current_contact.py','induced_phi_centre_holonomy.py',Path(__file__).name)}}
SOURCE_HASHES={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_BYTES.items()}
B=CHARGE_CONJUGATION
D=np.diag([-1.,1.,1.,1.])
CENTRES=(1,1j,-1,-1j)


def annihilators() -> np.ndarray:
    result=np.zeros((4,16,16),complex)
    for i in range(4):
        for state in range(16):
            if state&(1<<i):
                result[i,state^(1<<i),state]=(-1)**((state&((1<<i)-1)).bit_count())
    return result


def determinant(matrix: np.ndarray) -> complex:
    n=len(matrix);total=0j
    for permutation in itertools.permutations(range(n)):
        inversions=sum(permutation[i]>permutation[j] for i in range(n) for j in range(i+1,n))
        term=complex((-1)**inversions)
        for i,j in enumerate(permutation):term*=matrix[i,j]
        total+=term
    return total


def local_charge_unitary() -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
    a=annihilators();dagger=a.conj().transpose(0,2,1)
    majorana=np.eye(16,dtype=complex)
    for i in range(4):majorana=majorana@(a[i]+dagger[i])
    # R(W) a_i^dag R(W)^dag=sum_j W_ji a_j^dag. Four Majoranas
    # send a_i to -a_i^dag; W=-B^T therefore yields B_ij a_j^dag.
    W=-B.T;rotation=np.zeros((16,16),complex)
    occupations=[tuple(i for i in range(4) if state&(1<<i)) for state in range(16)]
    for i,I in enumerate(occupations):
        for j,J in enumerate(occupations):
            if len(I)==len(J):rotation[i,j]=determinant(W[np.ix_(I,J)])
    C=np.asarray(rotation@majorana,dtype=np.complex128)
    transformation=max(float(np.linalg.norm(C@a[i]@C.conj().T-np.einsum('j,jab->ab',B[i],dagger))) for i in range(4))
    square=C@C;phase=complex(square[0,0]);square_error=float(np.linalg.norm(square-phase*np.eye(16)))
    unitarity=float(np.linalg.norm(C@C.conj().T-np.eye(16)))
    parity=np.diag([(-1)**state.bit_count() for state in range(16)])
    parity_error=float(np.linalg.norm(C@parity-parity@C))
    clifford=max(float(np.linalg.norm(B@g.conj()@B.conj().T-(-g if mu==0 else g))) for mu,g in enumerate(SELECTED))
    if max(transformation,square_error,unitarity,parity_error,clifford)>1e-13:
        raise ArithmeticError('Actual CAR charge implementer or Clifford intertwiner failed')
    # Exact entries are integers +/-1 or +/-i, no fitted Bogoliubov matrix.
    if not np.all(np.isin(C.real,[-1.,0.,1.])) or not np.all(np.isin(C.imag,[-1.,0.,1.])):
        raise ArithmeticError('Charge implementer lost exact monomial representation')
    return C,a,dict(C_real=C.real.tolist(),C_imag=C.imag.tolist(),
        unitary_error=unitarity,CAR_transform_error=transformation,Clifford_intertwiner_error=clifford,
        square_scalar=[phase.real,phase.imag],square_scalar_error=square_error,even_parity_error=parity_error,
        global_implementation='Tensor product of this EVEN 16x16 local C over all65 mobile species and V sites; disjoint even CAR factors commute. Field pullback is the explicit real-orthogonal Phi/colour/EM transformation. Ref4 identity factor.')


def quadratic(a: np.ndarray,matrix: np.ndarray) -> np.ndarray:
    result=np.zeros((16,16),complex)
    for i in range(4):
        for j in range(4):result+=matrix[i,j]*a[i].conj().T@a[j]
    return result


def ref_gauss_control(C: np.ndarray,a: np.ndarray) -> dict[str,Any]:
    state=np.eye(16,dtype=complex)[:,-1]
    number=sum((ai.conj().T@ai for ai in a),np.zeros((16,16),complex))
    basis=matrix_unit_basis()/np.sqrt(8)
    ref_error=max(float(np.linalg.norm(quadratic(a,T)@state)) for T in basis)
    gauss_error=max(float(np.linalg.norm(C@quadratic(a,T)@C.conj().T-quadratic(a,-B@T.conj()@B.conj().T))) for T in basis)
    relative_ref=float(np.linalg.norm((number-4*np.eye(16))@state))
    relative_mobile=float(np.linalg.norm(C@(number-2*np.eye(16))@C.conj().T+(number-2*np.eye(16))))
    if max(ref_error,gauss_error,relative_ref,relative_mobile)>1e-12:raise ArithmeticError('Actual Ref4/mobile Gauss charge intertwining failed')
    return dict(Ref4_SU4_charge_norm=ref_error,Ref4_relative_EM_charge_norm=relative_ref,
        mobile_half_background_charge_transform_error=relative_mobile,colour_charge_transform_error=gauss_error,
        scope='Ref4 factor is left unchanged in the SAME fixed onsite neutral singlet sector. This is not C symmetry of an electron+Refhole preparation or an unspecified dynamical reference Hamiltonian.')


def mean_frames(holonomy: dict[str,Any]) -> dict[tuple[int,...],np.ndarray]:
    classes={tuple(row['twist']):np.asarray(row['G_decimal'],float) for row in holonomy['classes']}
    frames={}
    for twist in itertools.product(range(4),repeat=3):
        absolute=[min(t,4-t) for t in twist]
        permutation=sorted(range(3),key=lambda i:absolute[i]);representative=tuple(absolute[i] for i in permutation)
        transform=np.zeros((4,4));transform[0,0]=1
        for j,i in enumerate(permutation):transform[i+1,j+1]=-1 if twist[i]==3 else 1
        gram=transform@classes[representative]@transform.T
        eigenvalues,eigenvectors=np.linalg.eigh(gram)
        if eigenvalues[0]<=0:raise ArithmeticError('Native fullGram control lost positivity')
        square_root=(eigenvectors*np.sqrt(eigenvalues))@eigenvectors.T
        frames[twist]=np.einsum('mn,nij->mij',square_root,SELECTED)
    return frames


def literal_field_consumer(cfg: PhiConfig,holonomy: dict[str,Any]) -> dict[str,Any]:
    # A real full-field finiteN6 control, NOT the macro physical centre image.
    size=6;sites=np.asarray(list(itertools.product(range(size),repeat=3)),int)
    basis=matrix_unit_basis()/np.sqrt(8);frames=mean_frames(holonomy)
    components=.003*np.sin((1+np.arange(len(sites)))[:,None,None]*.17+(1+np.arange(4))[None,:,None]*.23+(1+np.arange(15))[None,None,:]*.31)
    perturbation=np.einsum('xma,aij->xmij',components,basis)
    link_angles=.11*np.cos(sites@np.asarray([.31,.23,.19]))
    generators=np.asarray([basis[2]+.3*basis[7],basis[4]-.2*basis[11],basis[1]+.4*basis[13]])
    links=np.asarray([[expm(1j*(.08+.017*np.sin(x@np.asarray([.13,.29,.37])+axis))*generators[axis]) for axis in range(3)] for x in sites])
    qtheta=2*np.pi*np.asarray([1,2,1])/size
    def index(x):return int(x[0]*size**2+x[1]*size+x[2])
    errors={'Hamiltonian':0.,'EM_current':0.,'EM_contact':0.,'paired_fullGram_mean':0.}
    wrong_fixed_frame=0.;block_count=0;scalar_control=[]
    for _,model,role in models(cfg):
        onsite,paths=scalar_paths(model)
        charge=0 if role=='neutral heavy -Lambda' else 1
        if any(scalars[0].imag!=0 or np.any(scalars[1:].real!=0) for _,scalars in paths):
            raise ArithmeticError('Actual frozen scalar paths violate real-mass/imaginary-spatial C algebra')
        scalar_control.append(dict(role=role,EM_charge=charge,path_count=len(paths),onsite=float(onsite),lengths=[int(np.sum(abs(d))) for d,_ in paths]))
        for twist,frame in frames.items():
            negative=tuple((-t)%4 for t in twist)
            converted=np.asarray([(-1 if mu==0 else 1)*B@F.conj()@B.conj().T for mu,F in enumerate(frame)])
            errors['paired_fullGram_mean']=max(errors['paired_fullGram_mean'],float(np.linalg.norm(converted-frames[negative])))
            phi=frame[None]+perturbation
            transformed=np.asarray([np.asarray([(-1 if mu==0 else 1)*B@F.conj()@B.conj().T for mu,F in enumerate(row)]) for row in phi])
            # A quarter fullGram MUST NOT inherit fixed-frame particle-hole C.
            theta=2*np.pi*np.asarray(twist)/(4*size)+np.asarray([.17,.13,-.11])
            coefficients=np.asarray([float(onsite),0.,0.,0.])
            opposite=coefficients.copy()
            for d,s in paths:
                phase=np.exp(1j*d@theta)
                coefficients+=2*np.real(s*phase);opposite+=2*np.real(s*phase.conjugate())
            h=np.einsum('m,mij->ij',coefficients,frame)
            wrong=np.einsum('m,mij->ij',opposite,frame)
            wrong_fixed_frame=max(wrong_fixed_frame,float(np.linalg.norm(B@h.conj()@B.conj().T+wrong)/(1+np.linalg.norm(h))))
            for x in sites:
                ix=index(x)
                for d,s in paths:
                    axis=int(np.flatnonzero(d)[0]);length=int(d[axis]);direction=np.eye(3,dtype=int)[axis]
                    target=(x+d)%size;iy=index(target)
                    transporter=np.eye(4,dtype=complex);converted_transport=np.eye(4,dtype=complex);derivative=0.
                    for j in range(length):
                        position=(x+j*direction)%size;k=index(position)
                        seam=CENTRES[twist[axis]] if position[axis]==size-1 else 1
                        link=seam*links[k,axis]*np.exp(1j*charge*link_angles[k])
                        transporter=transporter@link
                        converted_transport=converted_transport@(B@link.conj()@B.conj().T)
                        derivative+=charge*cfg.a*np.cos(position@qtheta)/np.sqrt(size**3)
                    Fx=np.einsum('m,mij->ij',s,phi[ix]);Fy=np.einsum('m,mij->ij',s,phi[iy])
                    Cx=np.einsum('m,mij->ij',s,transformed[ix]);Cy=np.einsum('m,mij->ij',s,transformed[iy])
                    forward=(Fx@transporter+transporter@Fy)/2
                    actual=(Cx@converted_transport+converted_transport@Cy)/2
                    errors['Hamiltonian']=max(errors['Hamiltonian'],float(np.linalg.norm(actual+B@forward.conj()@B.conj().T)))
                    old_j=1j*derivative*forward;new_j=1j*derivative*actual
                    old_jj=-derivative**2*forward;new_jj=-derivative**2*actual
                    errors['EM_current']=max(errors['EM_current'],float(np.linalg.norm(new_j-B@old_j.conj()@B.conj().T)))
                    errors['EM_contact']=max(errors['EM_contact'],float(np.linalg.norm(new_jj+B@old_jj.conj()@B.conj().T)))
                    block_count+=1
    if max(errors.values())>1e-10:raise ArithmeticError('Literal full-field C Hamiltonian/current/contact consumer failed')
    if wrong_fixed_frame<1e-6:raise ArithmeticError('Quarter-root fixed-frame C adversary did not discriminate')
    return dict(actual_forward_blocks=block_count,N=size,all64_fullGram_frames=True,scalar_paths=scalar_control,
        max_absolute_errors=errors,wrong_fixed_fullGram_C_relative_error=wrong_fixed_frame,
        control='Actual all64 finiteN6 classical fullten centres from frozen native receipt; all216 sites, generic four-Phi Cart15 perturbations, noncommuting SU4 links, real noncompactEM phases, literal one/two-link ordered products, full4 orbital columns. This is a native exact-algebra control, NOT substitution for the implicit macro roots.',
        one_particle_signs='B h* Bdag=-h_C; B J* Bdag=+J_C; B Gamma2* Bdag=-Gamma2_C. CAR particle-hole supplies the additional minus sign, hence manybody J is odd and Gamma2 is even.')


def run(mean_path: str,well_path: str,current_path: str,holonomy_path: str) -> dict[str,Any]:
    started=time.monotonic();inputs={}
    for role,path in [('mean',mean_path),('well',well_path),('current',current_path),('native_holonomy',holonomy_path)]:
        raw=Path(path).read_bytes();inputs[role]=dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),capsule=json.loads(raw))
    mean=inputs['mean']['capsule'];well=inputs['well']['capsule'];current=inputs['current']['capsule']
    if well['input_descriptor']['mean_input_sha256']!=mean['input_sha256'] or current['input_descriptor']['mean_input_sha256']!=mean['input_sha256']:
        raise ArithmeticError('Joined consumers do not use the same actual finite mean')
    if current['input_descriptor']['mean_receipt_sha256']!=inputs['mean']['sha256']:
        raise ArithmeticError('Current consumer has different actual mean receipt bytes')
    if well['input_descriptor']['mean_capsule_sha256']!=hashlib.sha256(json.dumps(mean,sort_keys=True).encode()).hexdigest():
        raise ArithmeticError('Well consumer has different actual mean capsule')
    cfg=PhiConfig(Lambda=1.,**mean['microscopic_config']);C,a,local=local_charge_unitary()
    gauss=ref_gauss_control(C,a);native=literal_field_consumer(cfg,inputs['native_holonomy']['capsule'])
    volume=mean['finite_size']**3;total=65*2*volume+4
    if total%4:raise ArithmeticError('C-symmetrized preparation does not descend through diagonalZ4')
    transferred={name:well['prices'][name] for name in ('density_trace_norm_distance_upper','physical_state_norm_distance_upper','weighted_Phi_action_error','all_flavour_current_action_error','all_flavour_contact_action_error','all_flavour_current_expectation_error','all_flavour_contact_expectation_error')}
    for value in transferred.values():
        if Fraction(int(value['numerator']),int(value['denominator']))<=0:raise ArithmeticError('Actual well price missing/nonpositive')
    descriptor=dict(input_receipts={role:{key:value[key] for key in ('path','sha256')} for role,value in inputs.items()},
        finite_size=mean['finite_size'],Lambda_exponent=mean['Lambda_exponent'],microscopic_config=mean['microscopic_config'],
        preparation='NEW128-component C-symmetrized prepared macro density, not old uniform64 retag',
        governing_plan='issue-222-induced-photon-delta13-continuation-20261008',plan_revision='d13d7d28ebfd7cb8386e7442897c596481aeab95cd55def1a8b3a60c1e89f4dd')
    return dict(branch=descriptor['preparation'],input_descriptor=descriptor,
        input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        local_CAR_implementation=local,actual_Ref4_Gauss_control=gauss,literal_full_field_consumer=native,
        exact_state_definition='rho_C=(rho64+C rho64 Cdag)/2=(1/128) sum_t [|Psi_t><Psi_t|+|C Psi_t><C Psi_t|]. Exact-reference rho_ref,C uses the same operation. This definition is C-invariant because C^2 is a scalar. It deliberately permits different sections/phases and avoids asserting C covariance of the old64 preparation. All mobile species remain fixed half-filled; original Ref4 factor and background unchanged.',
        field_transformation={'Phi0':'-B Phi0* Bdag','Phij':'+B Phij* Bdag','SU4_links':'B U* Bdag','noncompact_EM':'theta -> -theta; Pi -> -Pi','central_sectors':'t -> -t mod4','actual_mean':'G_-t=D G_t D, D=diag(-1,+1,+1,+1); C_-t=D sqrtG_t D by unique positive square root. Implicit finite functional and exact spectral fast pressure are invariant, and strictconvex uniqueness transfers roots. No offdiagonal component is set to zero.'},
        exact_Hamiltonian_intertwining='Actual real onsite/mass scalar and imaginary spatial scalars give Hmobile[A] -> Hmobile[-A] under explicit local CAR particle-hole plus Phi/colour pullback. This holds for arbitrary CartesianPhi and noncommuting links, not just Clifford means. Traceless onsite term has no particle-hole constant. QuarticGram, alignment, ReTrcurl, CartesianPhi kinetic and original .3chi colour Casimir are invariant real orthogonal/Haar transformations. Ref4 restricted charges are zero; half-filled mobile charged background transforms rho->-rho. Primitive UE/a Pi^2 and real-gradient Coulomb rho^T L+ rho are invariant. JointGauss descent intertwines SU4 outerautomorphism and degree4 residual. This is the fixedRef4 neutral sector, not an external-charge C theorem.',
        prepared_source_generating_functional='For this NEWexplicit rho_C and the displayed restricted Hamiltonian, Z[A]=Tr(U[A]rho_C U[0]dag)=Z[-A] by unitary trace conjugation. Every odd externalEM functional derivative vanishes; physical current is odd and full contact is even. This is a source-symmetry construction subject to parent full-H/Ref-sector mathematical review, NOT an already admitted photon-vacuum or scattering response.',
        transferred_actual_well_prices=transferred,
        transfer_reason='C is unitary and commutes with the physical restricted Hamiltonian at A=0. Each transformed component/reference pair has the same norm/trace and invariant Phi-weighted graph budgets. Convexity gives the SAME density bound for the128 preparation; current/contact errors have equal norms with odd/even sign. No numerical midpoint, energydegeneracy or rounded phase enters.',
        exact_counts=dict(volume=volume,mobile_species=65,charged_species=64,prepared_density_components=128,mobile_occupied=65*2*volume,Ref4_occupied=4,total_number=total,diagonalZ4_class=total%4),
        actual_IR_modes=current['modes'],actual_current_contact_errors={key:current[key] for key in ('max_current_error','max_contact_error','wrong_ordered_current_max','wrong_two_link_cross_contact_max')},
        separately_priced_diagnostic_operator_images=current['operator_image_rounding_prices'],
        status='Actual quantum CAR implementer, literal field/current/contact controls and joined finite mean/well consumers executed. Mathematical invariance/transfer statements remain explicit for parent review.',
        limitations='Prepared nonstationary local-well source. No globalground, thermodynamic selectedphase, EM/photonvacuum, commoncone, radiativeCompton or electron+Refhole C admission. The C-symmetrized state is NEW and does not alter oldsource/receiver/finiteSCREEN requirements. FullCAR numerical mean-image errors may be large and are not used as physical reference substitution.',
        source_at_import=SOURCE_HASHES,own_source_at_import=SOURCE_BYTES[Path(__file__).name].decode(),
        all_sources_unchanged_at_completion=all((ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()),
        runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,platform=platform.platform()),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mean',required=True);parser.add_argument('--well',required=True);parser.add_argument('--current',required=True);parser.add_argument('--holonomy',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=run(args.mean,args.well,args.current,args.holonomy)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(input_sha256=report['input_sha256'],source_at_import=report['source_at_import'],native=report['literal_full_field_consumer'],local=report['local_CAR_implementation'],elapsed_seconds=report['elapsed_seconds']),indent=2))
