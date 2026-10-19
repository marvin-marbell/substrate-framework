"""Exact original uniform colour-stiffness cubic and fast Hermite resolvent.

All45 normal Phi coordinates and all45 original SU4 colour coordinates enter.
This executed primitive calculation is not a completed full90 CAR source: that
consumer must retain its actual mixed quadratic Gaussian and quantum mean.
The public algebra applies at any declared positive Clifford Gram mean.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT=Path(__file__).resolve().parent
SOURCE_BYTES:dict[str,bytes]={Path(__file__).name:Path(__file__).read_bytes()}


class CapturedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname:str)->Any:
        raw=Path(self.path).read_bytes();SOURCE_BYTES[Path(self.path).name]=raw
        return self.source_to_code(raw,self.path)


class CapturedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname:str,path:Any=None,target:Any=None)->Any:
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.origin is not None:
            source=Path(spec.origin)
            if source.parent==ROOT and source.suffix=='.py':
                spec.loader=CapturedLoader(fullname,str(source));return spec
        return None


finder=CapturedFinder();sys.meta_path.insert(0,finder)
import induced_phi_full_uniform_fermionic_source_unitary as joined
sys.meta_path.remove(finder)
SOURCE_BYTES.update(joined.SOURCE_BYTES)
import numpy as np
from scipy.linalg import expm

iv:Any=joined.iv
exact,pack,cap,endpoint=joined.exact,joined.pack,joined.cap,joined.endpoint
midpoint,interval=joined.midpoint,joined.interval
JOINED_SOURCE_SHA='fb9dc99e3582fbd559ad96a176e427a0a8f3173f70ad13421e15c1879261dd8c'


def clifford_primitive_resolvent(geom:dict[str,Any],g:list[Any])->dict[str,Any]:
    """Actual coefficient law, no rounded matrices define the wave/corrector."""
    M=geom['M'];gamma=joined.full.supplier.primitive.SELECTED
    # M entries are exact Gaussian integers; all traces below are exact integers.
    if np.max(abs(np.einsum('aij,bji->ab',M,M)-4*np.eye(15)))!=0:
        raise ArithmeticError('Actual Clifford word convention differs')
    mean_comm=1j*(M[:,None]@gamma[None]-gamma[None]@M[:,None])
    delta_comm=1j*(M[:,None]@M[None]-M[None]@M[:,None])
    raw=np.einsum('abij,cmji->mabc',delta_comm,mean_comm)
    tensor=raw.transpose(0,1,3,2)+raw.transpose(0,3,1,2)
    integers=np.rint(tensor.real).astype(np.int64)
    if np.max(abs(tensor.imag))!=0 or np.max(abs(tensor.real-integers))!=0:
        raise ArithmeticError('Exact Hermitian Clifford trace tensor failed')
    H=[]
    for a in range(15):
        H.append(sum(g[m]*exact(F(int(np.trace(mean_comm[a,m]@mean_comm[a,m]).real),32)) for m in range(4)))
    if min(endpoint(h,0) for h in H)<=0:
        raise ArithmeticError('Primitive colour oscillator is not positive')
    D=[[[exact(0) for _ in range(15)] for _ in range(15)] for _ in range(45)]
    roots=[iv.sqrt(x) for x in g];den=exact(2560)*iv.sqrt(exact(8))
    for i,column in enumerate(geom['columns']):
        for slot,coefficient in column.items():
            mu,word=divmod(slot,15)
            for a,b in np.argwhere(integers[mu,:,:,word]!=0):
                D[i][int(a)][int(b)]+=coefficient*roots[mu]*exact(int(integers[mu,a,b,word]))/den
    omega=[iv.sqrt(exact(F(3,50))*h) for h in H]
    covariance=[exact(F(3,20))/w for w in omega]
    X=[[[D[i][a][b]/(omega[a]+omega[b]) for b in range(15)] for a in range(15)] for i in range(45)]
    Y=[[[X[i][a][b]*iv.sqrt(covariance[a]*covariance[b]) for b in range(15)] for a in range(15)] for i in range(45)]
    force=[3*sum(D[i][a][a]*covariance[a] for a in range(15)) for i in range(45)]
    coefficient_residual=max(joined.absolute_cap((omega[a]+omega[b])*X[i][a][b]-D[i][a][b]) for i in range(45) for a in range(15) for b in range(15))
    return dict(H=H,D=D,omega=omega,covariance=covariance,X=X,Y=Y,force=force,coefficient_residual=coefficient_residual)


def primitive_polynomial_prices(algebra:dict[str,Any],phi_covmax:Any,phi_covmin:Any)->dict[str,Any]:
    """Actual joint45-Phi Gaussian times primitive fast-colour Gaussian benchmark.

    Gaussian Phi need not be diagonal; CovPhi<=cmax I controls ALL45 entries.
    This benchmark does not assert zero full90 matter mixing.
    """
    total=cap(sum(x*x for row in algebra['Y'] for matrix_row in row for x in matrix_row))
    p2=cap(6*phi_covmax*total)  # 3 independent axes, Var(z^T Y z)=2 Tr(Y²).
    pnorm=cap(iv.sqrt(p2))
    amax=cap(1/(2*phi_covmin));Gphi=exact(8*F(2)**-4096)
    slow_residual=cap(Gphi*amax*pnorm)
    return dict(fast_polynomial_variance_sum=pack(total),corrector_norm_squared_upper=pack(p2),
        corrector_norm_upper=pack(pnorm),normalized_corrected_state_distance_upper=pack(pnorm),
        exact_corrected_state_norm_squared='1+||P f||²: P is odd under (Phi,C)->(-Phi,-C); exact orthogonality, not a fitted normalization.',
        remaining_slow_harmonic_action_norm_upper=pack(slow_residual),
        underlying_wave='Exact existing full45 functional Gaussian fPhi and original primitive fast Gaussian fC; this executed resolvent benchmark is not the final mixed90 wave.',
        proof='P=sum_i Phi_i sum_axis(C_axis^T X_i C_axis-Tr(Cov0 X_i)); independent centred colour axes give6 sum_i Tr(Y_i²). CovPhi<=cmax I gives||P f||²<=6 cmax sum_iAB Y_iAB², even when all45Phi correlate. Lslow P f=Gphi(Aphi Phi)·R(C) f, bounded byGphi amax||P f||. Final mixed90 consumer must separately retain actual ACC,m_C and their action differences.')


def literal_primitive(algebra:dict[str,Any],geom:dict[str,Any],native:dict[str,Any])->dict[str,Any]:
    """Run original alignment and fast polynomial differential operator."""
    D=np.asarray([[[float(midpoint(x)) for x in row] for row in matrix] for matrix in algebra['D']])
    Y=np.asarray([[[float(midpoint(x)) for x in row] for row in matrix] for matrix in algebra['Y']])
    omega=np.asarray([float(midpoint(x)) for x in algebra['omega']])
    covariance=np.asarray([float(midpoint(x)) for x in algebra['covariance']])
    Dbar=D*np.sqrt(covariance[None,:,None]*covariance[None,None,:])
    identity=np.max(abs((omega[None,:,None]+omega[None,None,:])*Y-Dbar))
    rng=np.random.default_rng(22290);z=rng.normal(size=(3,15));direction=rng.normal(size=45);direction/=np.linalg.norm(direction)
    matrices=np.einsum('i,iab->ab',direction,Y);qmat=np.einsum('i,iab->ab',direction,Dbar)
    action=sum(2*np.dot(omega*row,matrices@row)-2*np.dot(omega,np.diag(matrices)) for row in z)
    force=sum(row@qmat@row-np.trace(qmat) for row in z)
    if identity>2e-14 or abs(action-force)>2e-13:
        raise ArithmeticError('Actual fast-Hermite differential cancellation failed')
    M=geom['M'];T=M/np.sqrt(8);colour=rng.normal(size=(3,15));colour/=np.linalg.norm(colour);colour*=2**-12
    links=np.asarray([expm(1j*np.einsum('a,aij->ij',row,T)/8) for row in colour])
    def alignment(frame:np.ndarray)->float:
        return float(.2*64*sum(np.sum(abs(frame-U@frame@U.conj().T)**2) for U in links)/8)
    epsilon=2**-12;delta=np.einsum('i,imab->mab',direction,geom['deltas'])/8
    observed=(alignment(native['frame']+epsilon*delta)-alignment(native['frame']-epsilon*delta))/(2*epsilon)
    predicted=sum(row@np.einsum('i,iab->ab',direction,D)@row for row in colour)
    relative=abs(observed-predicted)/max(abs(predicted),np.finfo(float).tiny)
    if relative>1e-6:
        raise ArithmeticError('Original physical primitive Phi-colour cubic mismatch')
    return dict(ALL45_fast_Hermite_coefficient_identity_defect=float(identity),
        actual_fast_differential_action=float(action),actual_mean_subtracted_cubic=float(force),
        differential_identity_defect=float(abs(action-force)),
        original_alignment_Phi_derivative=float(observed),derived_ALL45_Phi_C2_coefficient=float(predicted),
        original_alignment_relative_defect=float(relative),colour_coordinate_norm=float(np.linalg.norm(colour)),
        primitive_Phi35_nonzero_cubic_entries=int(np.count_nonzero(D[geom['index35']])),
        primitive_Phi35_fast_force=float(midpoint(algebra['force'][geom['index35']])) )


def run(root_path:Path,full_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=root_path.read_bytes();fraw=full_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=joined.ROOT_RECEIPT_SHA or hashlib.sha256(fraw).hexdigest()!=joined.FULL_RECEIPT_SHA or hashlib.sha256(SOURCE_BYTES[Path(joined.__file__).name]).hexdigest()!=JOINED_SOURCE_SHA:
        raise ArithmeticError('Frozen actual finiteN4 mean/full45/joined prerequisites required')
    rootcap=json.loads(raw);fullcap=json.loads(fraw);root=rootcap['actual_stationary_root']
    iv.dps=3000;g0=interval(root['parameters']['g0']);gs=interval(root['parameters']['gs']);g=[g0,gs,gs,gs]
    native,geom=joined.literal_geometry(root,fullcap);algebra=clifford_primitive_resolvent(geom,g)
    q=fullcap['actual_full45_coupled_quantum_consumer'];Lambda=exact(F(2)**8192);Gphi=exact(8*F(2)**-4096)
    lower=exact(F(q['certified_eigenvalue_lower_divided_Lambda']['lower']))
    upper=cap(exact(F.from_float(float(max(q['actual_full45_effective_eigenvalues_divided_Lambda']))))+interval(q['eigen_reconstruction_operator_error']))
    cmax=cap(iv.sqrt(Gphi/(Lambda*lower))/2);cmin=exact(endpoint(iv.sqrt(Gphi/(Lambda*upper))/2,0))
    prices=primitive_polynomial_prices(algebra,cmax,cmin);literal=literal_primitive(algebra,geom,native)
    sparse=[]
    for i in range(45):
        for a in range(15):
            for b in range(a,15):
                if endpoint(algebra['D'][i][a][b],0)!=0 or endpoint(algebra['D'][i][a][b],1)!=0:
                    sparse.append(dict(Phi_coordinate=i,colour_grade_A=a,colour_grade_B=b,cubic_coefficient_divided_Lambda54=pack(algebra['D'][i][a][b]),resolvent_coefficient_divided_Lambda78=pack(algebra['X'][i][a][b]),normalized_fast_Hermite_coefficient=pack(algebra['Y'][i][a][b])))
    descriptor=dict(plan='issue-222-induced-photon-delta13-continuation-20261008',assigned_scientific_revision=joined.scalar.uniform.ASSIGNED_REVISION,
        law='EXACT_ORIGINAL_UNIFORM_COLOUR_STIFFNESS_CUBIC_AND_FAST_HERMITE_RESOLVENT',
        root_input=dict(path=str(root_path),sha256=hashlib.sha256(raw).hexdigest()),full45_input=dict(path=str(full_path),sha256=hashlib.sha256(fraw).hexdigest()),
        exact_root_parameters=root['parameters'],Phi_labels=geom['labels'],colour_grade_words=geom['words'],
        parameters=dict(V=64,a='1/64',Lambda='2^8192',chi='2^-4096',Gphi='8chi',Gc='3chi/10',alignment='.2 Lambda^5/4 rawW/8',colour_magnetic='.7 Lambda^5/4 sum(4-ReTr plaquette)',
        colour_chart='U_axis=exp(i sum_A T_A C_axis,A/8), T=Mgrade/sqrt8; coordinate15*axis+grade, nativeedges3*site+axis',
        scope='Primitive resolvent and its physical benchmark only; actual mixed90 quantumCAR source is a distinct consumer, not admitted here.'))
    proof=dict(normal_coordinates=45,colour_coordinates=45,all45_primitive_colour_metric_diagonal=[pack(x) for x in algebra['H']],
        actual_fast_frequencies_divided_Lambda38=[pack(x) for x in algebra['omega']],
        actual_fast_covariances_divided_Lambda_minus78=[pack(x) for x in algebra['covariance']],
        actual_all45_fast_zero_point_Phi_force_divided_Lambda38=[pack(x) for x in algebra['force']],
        actual_sparse_all45_Phi_C2_and_resolvent_coefficients=sparse,
        algebraic_resolvent_coefficient_numerical_enclosure_residual_upper=pack(algebra['coefficient_residual']),
        actual_polynomial_preparation_prices=prices,actual_original_primitive_and_Hermite_consumers=literal,
        exact_operator_identity='Kc=Lambda^5/4 Horbit/5; D_i=Lambda^5/4 d_i Horbit/10. Omega_A=sqrt(Gc Kc_AA). Xi_iAB=DiAB/(OmegaA+OmegaB). ForP=sum_i Phi_i sum_axis(C^T Xi C-Tr(Cov0 Xi)), (Hfast-Efast)P ffast=[sum_i Phi_i sum_axis(C^T Di C-Tr(Cov0 Di))] ffast EXACTLY. ThusminusP corrects the non-small primitivePhiC² excitedfast component, rather than only proposing a formal resolvent.',
        next_consumer_contract='Useactualmixed90RiccatiGaussian, Ctilde=C+ACC^-1ACPhi Phi, andNEWquantumfeedbackmean. PrimitiveXi maythenbeused withtrueACC−alpha0 andslow/mixedaction leftovers priced. ActualmixedPH isNONZERO; this benchmark doesNOTsupply/zero thatmatrix. No full90-source/nonuniform/localGauss/originalSOURCE claim.')
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),actual_primitive_fast_Hermite_resolvent=proof,
        source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},input_unchanged=dict(root=root_path.read_bytes()==raw,full45=full_path.read_bytes()==fraw),
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--full45',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=run(args.root,args.full45)
    if not all(result['source_unchanged_since_import'].values()) or not all(result['input_unchanged'].values()):
        raise ArithmeticError('Scientific source/input epoch changed during actual primitive resolvent execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n');proof=result['actual_primitive_fast_Hermite_resolvent']
    print(json.dumps(dict(input_sha256=result['input_sha256'],prices=proof['actual_polynomial_preparation_prices'],literal=proof['actual_original_primitive_and_Hermite_consumers'],sparse_entries=len(proof['actual_sparse_all45_Phi_C2_and_resolvent_coefficients']),elapsed_seconds=result['elapsed_seconds']),indent=2))
