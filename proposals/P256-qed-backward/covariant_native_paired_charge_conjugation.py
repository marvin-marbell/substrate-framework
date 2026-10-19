#!/usr/bin/env python3
"""Actual native all-column C/Gauss/RefCAR check; frozen protocols unchanged."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
ROOT=Path(__file__).resolve().parent
NAMES=('covariant_native_paired_charge_conjugation.py','covariant_native_charged_removal_gf.py',
 'covariant_native_charged_site_gf.py','covariant_native_electric_heat_comparison.py',
 'covariant_native_closed_auxiliary_loop.py','covariant_native_source_gram_moments.py',
 'covariant_native_haar_range_jets.py','covariant_source_band_halfstep_control.py',
 'covariant_induced_photon.py','covariant_photon_energy_control.py','sea_selection.py','rotation_band.py')
SOURCE_BYTES_AT_IMPORT={name:(ROOT/name).read_bytes() for name in NAMES}
PLAN=Path('/home/administrator/.omp/agent/memory/shared/plans/issue-222-induced-photon-delta13-continuation-20261008.md')
PLAN_BYTES_AT_IMPORT=PLAN.read_bytes()
from covariant_native_charged_removal_gf import annihilator,decode_complex,complex_array,ADDITION_BYTES_AT_IMPORT,ADDITION_SHA
import covariant_native_electric_heat_comparison as electric
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import coo_matrix


def conjugate_occupation(occupied: tuple[int,...], modes: int) -> tuple[tuple[int,...],int]:
    complement=[mode for mode in range(modes) if mode not in occupied]
    sign=(-1)**sum(mode-index for index,mode in enumerate(occupied))
    permuted=[mode^1 for mode in complement]
    inversions=sum(permuted[i]>permuted[j] for i in range(len(permuted)) for j in range(i+1,len(permuted)))
    return tuple(sorted(permuted)),sign*(-1)**inversions


def fock_conjugation(modes: int,number: int) -> np.ndarray:
    source=list(itertools.combinations(range(modes),number))
    target={value:index for index,value in enumerate(itertools.combinations(range(modes),modes-number))}
    result=np.zeros((len(target),len(source)))
    for index,occupied in enumerate(source):
        conjugate,sign=conjugate_occupation(occupied,modes)
        result[target[conjugate],index]=sign
    return result


def physical_conjugation(source: dict[str,Any],target: dict[str,Any],reference: bool) -> Any:
    ns,nt=len(source['physical']),len(target['physical']);rows=[];columns=[];entries=[]
    for column,(state,flux,electric_flux) in enumerate(source['physical']):
        light,aux=source['states'][state]
        lc,ls=conjugate_occupation(light,8);ac,asign=conjugate_occupation(aux,4)
        target_state=target['state_index'][(lc,ac)]
        row=target['index'][(target_state,-flux)]
        if target['physical'][row][2] != tuple(-value for value in electric_flux):
            raise ArithmeticError('Actual charge conjugation did not reverse every physical electric edge')
        if electric.native.density(lc,ac,target['charge']) != tuple(-value for value in electric.native.density(light,aux,source['charge'])):
            raise ArithmeticError('Actual C/Gauss density mismatch')
        for node in range(3):
            if reference:
                for ref in range(4):
                    rows.append((ref*3+node)*nt+row);columns.append((ref*3+node)*ns+column)
                    # FULL16-mode Hodge: Ref1 crosses seven body-complement
                    # modes; H_ref|r>=(-1)^r|filled except r>.
                    entries.append(-ls*asign*(-1)**ref)
            else:
                rows.append(node*nt+row);columns.append(node*ns+column);entries.append(ls*asign)
    return coo_matrix((entries,(rows,columns)),shape=((12 if reference else 3)*nt,(12 if reference else 3)*ns)).tocsr()


def canonical(half: np.ndarray,pref: float,number: int) -> np.ndarray:
    occupations=np.array(list(itertools.combinations(range(len(half)),number)))
    return pref*np.linalg.det(half[occupations[:,None,:,None],occupations[None,:,None,:]])


def run() -> dict[str,Any]:
    addition=json.loads(ADDITION_BYTES_AT_IMPORT)
    gamma=decode_complex(addition['selected_gamma']);embedding=decode_complex(addition['selected_embedding'])
    Omega=decode_complex(addition['actual_Omega'])
    neutral=electric.native.space(0,2);removal=electric.native.space(-1,2);particle=electric.native.space(1,2)
    Cn=physical_conjugation(neutral,neutral,False);Cp=physical_conjugation(removal,particle,True)
    C3=fock_conjugation(8,3);Ca=fock_conjugation(4,2);C=np.kron(C3,Ca)
    point_rows=[]
    for normal in (-.09,0.,.09):
        actual=gamma.copy();actual[1]*=1+normal
        for theta in (.37,1.23,2.71):
            plus,pp=electric.matched_tree_half(actual,theta,0.)
            minus,pm=electric.matched_tree_half(actual,-theta,0.)
            auxplus,ap=electric.auxiliary_path_half(actual,.04,complex((1+np.exp(1j*theta))/2))
            auxminus,am=electric.auxiliary_path_half(actual,.04,complex((1+np.exp(-1j*theta))/2))
            three=np.kron(canonical(plus,pp,3),canonical(auxplus,ap,2))
            five=np.kron(canonical(minus,pm,5),canonical(auxminus,am,2))
            point_rows.append(dict(Phi=normal,angle=theta,
                actual_complete_SF_C_relative_defect=float(np.linalg.norm(C@three@C.T-five)/np.linalg.norm(five)),
                actual_active_det_half=complex_array(np.array(np.linalg.det(plus))),
                actual_auxiliary_det_half=complex_array(np.array(np.linalg.det(auxplus)))))
    mean,_=electric.native.square_jets(gamma,.09,0.)
    values,U=eigh(mean[0]@mean[0]);negative=U[:,values>1]
    # This is the CHARGE-CONJUGATE selected-spin embedding, not unchanged E.
    # Ebar=E* sigma_x makes tau_body psi_negative* sigma_x = psi_positive.
    sigma_x=np.array([[0.,1.],[1.,0.]])
    antiparticle_embedding=embedding.conj()@sigma_x
    packets=list(np.eye(4))
    for value in ([.1,.8,.4,.1],[.5,.4,.1,.6]):
        packet=np.array(value);packet/=np.linalg.norm(packet);packets.append(packet)
    source_rows=[]
    for number,packet in enumerate(packets):
        template=np.kron(packet[:,None],np.eye(2))
        negative_coefficient=negative@negative.conj().T@template
        negative_coefficient/=np.linalg.norm(negative_coefficient,'fro')
        Sh=annihilator(neutral,removal,negative_coefficient,antiparticle_embedding)
        Sa=electric.native.source(neutral,particle,electric.native.source_coefficients(gamma,packet),embedding,3)
        transformed=Cp@Sh@Cn.T
        plus=transformed-Sa;minus=transformed+Sa
        source_rows.append(dict(packet_index=number,packet=packet.tolist(),
            actual_C_source_equal_positive_defect=float(np.linalg.norm(plus.data)),
            actual_C_source_equal_negative_defect=float(np.linalg.norm(minus.data)),
            actual_source_Frobenius_norm=float(np.linalg.norm(Sa.data)),
            all_neutral_columns=Sa.shape[1],all_particle_Ref_channels_rows=Sa.shape[0]))
    transformed_Omega=Cn@Omega;overlap=np.vdot(Omega,transformed_Omega)
    descriptor=dict(plan_id=PLAN.stem,plan_sha256_at_IMPORT=hashlib.sha256(PLAN_BYTES_AT_IMPORT).hexdigest(),
        actual_neutral_capsule_sha256=ADDITION_SHA,UE=1.,electric_cutoff=2,Haar_grid=32,
        old_reference='SAME4RefCAR modes; initialRef4/background4 ->Ref3, frozen source ABI unchanged',
        new_reference='SAME4RefCAR modes; NEWneutralRef0/background0 ->Ref1, static counterparticle +1',
        neutral_operator_identity='Reference occupation minus background iszero for BOTH neutral protocols; SAMEbody/aux/Phi/electric FN and KN, with their one-dimensional Refneutral states',
        active_N3_toN5_C='Full Hodge complement followed by onsite positive/negative-beta spin exchange, electric E->-E, auxN2 complement/spin exchange',
        reference_C='Ref1->Ref3 Hodge with(-1)^r and FULL16-mode crosssign; neutralRef0->Ref4',
        source='r_dag W_inverse a_negative with charge-conjugate embedding Ebar=E* sigma_x, reversed integerWilsonpaths',
        selected_dimension=2,Phi3='threelevels ONEcoordinate',source_packets='four site GF plus ORIGINALscattering/receiverpacket, no fit',
        scope='Native SAME primitiveE SF/Phi3/finiteCAR/allintegercycles law only; no originalrealEM/fullSU4globalZ4/fullDiracCARanticommutator/commoncone inferred')
    return dict(input_descriptor=descriptor,input_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        sources_at_IMPORT={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes=data.decode()) for name,data in SOURCE_BYTES_AT_IMPORT.items()},
        runtime_packages=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__),
        actual_complete_point_SF_C_rows=point_rows,actual_physical_C_permuted_base_dimensions=[Cn.shape[0],Cp.shape[0]//4],
        actual_neutral_C_squared_defect=float(np.linalg.norm((Cn@Cn-scipy.sparse.eye(Cn.shape[0])).data)),
        actual_neutral_Omega_C_overlap=complex_array(np.array(overlap)),
        actual_neutral_Omega_C_orthogonal_residual=float(np.linalg.norm(transformed_Omega-overlap*Omega)),
        actual_C_source_ALLcolumn_rows=source_rows,actual_antiparticle_embedding=complex_array(antiparticle_embedding),
        proof='For every real field, B(-A)=B(A)^T and F(-A)=-F(A)^T. With tau exchanging beta blocks, T(-A)=tau T(A)^(-T) tau; det(T)=1. The positive half inherits this identity. Exterior Hodge lifts it to N3/N5 and selfdualauxN2, including equal detBpref. E->-E maps every globalGauss state, preserves ALLedge electric heat and the regulator; C leaves Phi and itsheat untouched. Thus reflected F/K conjugate for ALL columns, with SAMEneutral lambdaN. Ref backgrounds cancel identically in BOTHneutral protocols. Source signs and negative-band conjugate embedding are explicitly tested on EVERYdeclaredneutral column; no source firstmoment/pole/mass promotion.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic();result=run();result['runtime_seconds']=time.monotonic()-start
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(point_C_max_relative_defect=max(row['actual_complete_SF_C_relative_defect'] for row in result['actual_complete_point_SF_C_rows']),
        actual_source_rows=result['actual_C_source_ALLcolumn_rows'],Omega_C_orthogonal_residual=result['actual_neutral_Omega_C_orthogonal_residual'],runtime_seconds=result['runtime_seconds'])),flush=True)


if __name__=='__main__':
    main()
