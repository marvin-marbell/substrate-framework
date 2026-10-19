"""ActualWHOLEboson linearGaussian compensatedE source survival/stability.

Extends frozen neutral electron/Refhole star's zero/oneboson compression to
ALLbosonnumbers, same linearJA+scalarpotential coupling/static/DW/source.
No nonlinearPeierls/fullPhi/SU4/Noether/originalcanonicalY vacuum promotion.
Exactrationalbinarypoint operator-envelope prices untruncatedunitary; observed
spectral/matrix floating estimates separately labelled numerical controls.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent
OWN=Path(__file__).read_bytes()
import induced_native_reference_neutral_green as neutral
NEUTRAL_SHA='0b0057da791093091cbad93f119689ad460f475f2f38eeefc418bf2b90ace98e'
SOURCE_AT_START={str(Path(__file__).resolve()):OWN,**neutral.SOURCES,**neutral.charged.SOURCES}


def unpack(v:Any)->np.ndarray:
    return np.asarray(v['real'])+1j*np.asarray(v['imag'])


def rational(v:float)->Fraction:
    return Fraction.from_float(float(v))


def frobenius_squared_exact(x:np.ndarray)->Fraction:
    """Exactsum of squaredbinary64points, powersoftwo groupedwithoutrounding."""
    buckets:dict[int,int]={}
    for value in np.concatenate((np.real(x).reshape(-1),np.imag(x).reshape(-1))):
        numerator,denominator=float(value).as_integer_ratio()
        if numerator:buckets[denominator]=buckets.get(denominator,0)+numerator*numerator
    if not buckets:return Fraction(0)
    largest=max(buckets)
    return Fraction(sum(total*(largest//denominator)**2 for denominator,total in buckets.items()),largest**2)


def hermitian_source_moments(h:np.ndarray,v:np.ndarray)->dict[str,Any]:
    """ExactHermitianmidpoint binarypoint h, exactRAWsourcev; no whitening.

    Hermitianmidpoint is specifiedoperator interpretation, with numerical
    input anti-Hermitian defect reported. Frobeniusprojection iscontractive.
    """
    n=h.shape[0];channels=v.shape[1]
    H=[[( (rational(h[i,j].real)+rational(h[j,i].real))/2,(rational(h[i,j].imag)-rational(h[j,i].imag))/2 ) for j in range(n)] for i in range(n)]
    V=[[(rational(v[i,a].real),rational(v[i,a].imag)) for a in range(channels)] for i in range(n)];HV=[]
    for i in range(n):
        row=[]
        for a in range(channels):
            real=sum((H[i][j][0]*V[j][a][0]-H[i][j][1]*V[j][a][1] for j in range(n)),Fraction(0));imag=sum((H[i][j][0]*V[j][a][1]+H[i][j][1]*V[j][a][0] for j in range(n)),Fraction(0));row.append((real,imag))
        HV.append(row)
    p=sum((vr*vr+vi*vi for row in V for vr,vi in row),Fraction(0))/4
    m1=sum((V[i][a][0]*HV[i][a][0]+V[i][a][1]*HV[i][a][1] for i in range(n) for a in range(channels)),Fraction(0))/4
    m1imag=sum((V[i][a][0]*HV[i][a][1]-V[i][a][1]*HV[i][a][0] for i in range(n) for a in range(channels)),Fraction(0))/4
    if m1imag:raise ArithmeticError('SpecifiedHermitianoperator exactsourceenergyhasimaginarypart')
    m2h=sum((vr*vr+vi*vi for row in HV for vr,vi in row),Fraction(0))/4
    return dict(p=p,m1=m1,m2_static=m2h,input_static_antiHermitian_norm=float(np.linalg.norm(h-h.conj().T,2)),h_frobenius_squared_point=frobenius_squared_exact(h))


def stream_actual_all_boson_prices(supplier:Any,Z:np.ndarray,source:np.ndarray)->dict[str,Any]:
    native=supplier.native;JA,rho=neutral.projected_operators(native,native['Pprofiles'][0],Z);n=Z.shape[1];Gram=np.zeros((n,n),dtype=np.complex128);weighted=np.zeros_like(Gram);identity=np.eye(n);ref=0;rows=[];bound=Fraction(0);stable_bound=Fraction(0);modes=A=R=0;largest_anti=0.
    for site in native['sites']:
        q=tuple(int(i) for i in site)
        if q>tuple((-i)%native['N'] for i in q):continue
        block=supplier.block(q);edge=block.data['physical_edgeframe'];density=block.data['physical_densityframe'];Oa=(edge.T@JA.reshape(3*native['V'],-1)).reshape(block.M,n,n);Op=(density.T@rho.reshape(native['V'],-1)).reshape(block.R,n,n)-density[ref,:,None,None]*identity[None,:,:];O=np.concatenate((Oa,Op))
        largest_anti=max(largest_anti,float(np.max(abs(O-O.conj().transpose(0,2,1)))))
        # EXACTmodel usesHermitianmidpoints Obar=(O+Odag)/2. Physicalinput
        # anti-Hermitiannoise reported, NOT suppressed asmicroscopicerror.
        Oherm=(O+O.conj().transpose(0,2,1))/2
        Vann=np.einsum('mij,ml->ijl',Oherm,block.amplitudes,optimize=True);flat=Vann.reshape(n,-1);Gram+=flat@flat.conj().T
        weighted+=(Vann/np.sqrt(block.omega)[None,None,:]).reshape(n,-1)@(Vann/np.sqrt(block.omega)[None,None,:]).reshape(n,-1).conj().T
        price=Fraction(0)
        for m in range(block.M+block.R):
            # Cauchy: ||sum_m amp_m Obar_m||F² <= count sum_m
            # |amp_m|² ||Obar_m||F² <= count sum_m |amp_m|²||Opoint_m||F².
            opnorm=frobenius_squared_exact(O[m]);amplitudenorm=frobenius_squared_exact(block.amplitudes[m]);price+=(block.M+block.R)*opnorm*amplitudenorm
        omega_min=float(np.min(block.omega))
        if omega_min<=0:raise ArithmeticError('FullbosonstabilityrequiresactualpositiveΩ; nozero-modefallback')
        bound+=price;stable_bound+=price/rational(omega_min);modes+=len(block.omega);A+=block.M;R+=block.R
        rows.append(dict(q=q,positive_modes=len(block.omega),actual_positive_frequency_min=omega_min,exact_coherent_operator_envelope=ratio(price),exact_frequency_weighted_operator_envelope=ratio(price/rational(omega_min))))
        print('ActualALL-bosonunitarycoupling qgroup',q,'modes',len(block.omega),flush=True)
    if (len(rows),A,R,modes)!=(36,129,63,3865):raise ArithmeticError('Missingactualnativequantumphoton/densitycoupler')
    return dict(Gram=Gram,weighted=weighted,all_qgroups=rows,numerical_M2_bath=float(np.trace(source.conj().T@Gram@source).real/4),exact_operator_Gram_envelope=bound,exact_operator_weighted_Gram_envelope=stable_bound,source_input_current_density_antiHermitian_max=largest_anti,mode_counts=dict(qgroups=36,A_frames=129,phi_frames=63,positive_modes=3865))


def ratio(v:Fraction)->dict[str,Any]:
    return dict(numerator=str(v.numerator),denominator=str(v.denominator),decimal_approx=float(v))


def run(path:Path)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=NEUTRAL_SHA:raise ArithmeticError('Frozenactuale961/0b0057 neutral source required')
    cap=json.loads(raw);supplier=neutral.charged.load_supplier();native=supplier.native
    if (native['UE'],native['a'])!=(1.,.015625):raise ArithmeticError('ActualUEphys1/a1/64 nativecoupler required; no UE64shorthand')
    descriptor=dict(plan_id='issue-222-induced-photon-delta13-continuation-20261008',execution_revision='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7',neutral_receipt=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),photon_receipt=dict(path=str(neutral.charged.RECEIPT),sha256=neutral.charged.RECEIPT_SHA),actual_native_units=dict(UEphys=native['UE'],a=native['a'],u=native['UE']/native['a']),exact_model='ExactHermitianmidpoints of frozenbinary64 h andcomputedphysicalcouplerO_m; exactbinary64positiveΩ/amplitudes/sourcev. No Γwhitening/sourceprobabilityfit. ALLboson-numberlinearGaussian Hamiltonian, zero/oneboson sourcee961 isitscompression.',neutral_input_descriptor=cap['input_descriptor'])
    Z=unpack(cap['bare_empty_Z']);v=unpack(cap['source_all_four_v_alpha']);h=unpack(cap['static_neutral_H_projected']);mom=hermitian_source_moments(h,v);stream=stream_actual_all_boson_prices(supplier,Z,v);Gram=unpack(cap['neutral_star']['coherent_bath_Gram']);join=float(np.max(abs(stream['Gram']-Gram)))
    if join>1e-8:raise ArithmeticError('Actualuntruncatedcouplingsdonotjoinfrozenzero/onebosonstar')
    m2_upper=mom['m2_static']+mom['p']*stream['exact_operator_Gram_envelope'];num_m2=float(mom['m2_static'])+stream['numerical_M2_bath'];source_times=[]
    for t in (Fraction(0),Fraction(1,4096),Fraction(4096)):
        radius=t*t*m2_upper/2;centreimag=-t*mom['m1'];source_times.append(dict(time=ratio(t),raw_source_real_interval=dict(lower=ratio(mom['p']-radius),upper=ratio(mom['p']+radius)),raw_source_imag_interval=dict(lower=ratio(centreimag-radius),upper=ratio(centreimag+radius)),exact_ALL_boson_unitary_Taylor_disk_radius=ratio(radius),centre=dict(real=ratio(mom['p']),imag=ratio(centreimag)),real_interval_excludes_zero=mom['p']>radius))
    # sqrt(x)<=max(1,x) givesanexact rationalgloballowerenergyboundwithout
    # float eigenvalues/rounding. Numericalcompleted-squarematrix isseparate.
    static_bound=max(Fraction(1),mom['h_frobenius_squared_point']);energy_lower=-static_bound-stream['exact_operator_weighted_Gram_envelope'];hmid=(h+h.conj().T)/2;numerical_lower=float(np.linalg.eigvalsh(hmid-stream['weighted'])[0])
    if not source_times[1]['real_interval_excludes_zero']:raise ArithmeticError('Actualfullbosoncontrol failednonzero1/4096survivalwithconservativeprice; retainphysicscapsuleandderiveasharperphysicalprice')
    result=dict(source_norm_exact_binarypoint=ratio(mom['p']),actual_common_neutral_reference='Samefrozenquantumbathvacuum, all64fixedbaresea, Ref4; Hph=ΣΩbdagb aftersubtractingCOMMONneutralphotonvacuumenergy. Nochargedprivatephase; Refhole3spectatorindicesfour, source1/2all4 unchanged.',source_M1_exact_binarypoint=ratio(mom['m1']),source_M2_static_exact_binarypoint=ratio(mom['m2_static']),source_M2_actual_numerical_control=num_m2,source_M2_ALL_boson_exact_operator_envelope=ratio(m2_upper),physical_raw_all_boson_unitary_source_time_enclosures=source_times,ALL_boson_stability_energy_lower_exact=ratio(energy_lower),completed_square_matrix_eigenvalue_lower_NUMERICAL_ONLY=numerical_lower,actual_frozen_e961_coherent_coupling_Gram_join=join,actual_mode_counts=stream['mode_counts'],source_input_antiHermitian_noise=dict(static=mom['input_static_antiHermitian_norm'],current_density=stream['source_input_current_density_antiHermitian_max']),actual_physical_source_coupling_groups=stream['all_qgroups'],Hamiltonian='H=h+ΣΩ bdagb+Σ(Vann b+Vann†bdag), Vann=ΣJA*bA+(rho_electron−phi_ref I)*bphi. Supplier b=<0|field|1>, upperright0←1blockVann Ustatic. SourceH onebosoncomponentVann†v, soM2=.25Trv†(h²+ΣVannVann†)v. Complete square Ω(b+Vann†/Ω)dag(b+Vann†/Ω)−VannVann†/Ω. Fieldsnotboson-numbertruncated.',unitary_proof='For selfadjointbounded-belowfinite-mode linearboson H andexactsource s inD(H), spectral |exp(−itE)−1+itE|<=t²E²/2 gives |<s|U(t)|s>−p+itM1|<=t²M2/2. Exact rationalCauchyEnvelope pricesALLcoherentcouplings/ALLboson-numbersectors, not first-momentansatz. SpecifiedexactbinarypointH/Om Hermitianmidpoints; inputantiHermnoiseexposed. Proof is ofthisdeclaredpointGaussianlaw, not an intervalpricedmicroscopicoriginalSOURCE.',scope='NEWwhole-boson LINEARGaussian projectedpositiveelectron/ref-hole source, no bosonnumbercutoff. Frozenstar retardedG isaZERO/ONEcompression only. Actualunitary1/4096 nonzero survival nowpriced,4096intervalreportedwithoutnonzeroclaim. DynamicalPhi/SU4/Noether, nonlinearPeierls, exactCAR Pauli overlap withseaPH, selfconsistentvacuum andoriginalcanonicalY SOURCE remainopen. Exactpoint sourcep close.5 notforced. NoeR/anomaly/commoncone/admission.')
    captured={**neutral.charged.SOURCES,**neutral.SOURCES,**SOURCE_AT_START};supplier.s.assert_frozen()
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(neutral.charged.packed(descriptor),sort_keys=True).encode()).hexdigest(),actual_ALL_boson_compensated_source_unitary=result,source_at_import={Path(name).name:dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in captured.items()},source_unchanged_since_import={name:Path(name).read_bytes()==data for name,data in captured.items()},input_unchanged=path.read_bytes()==raw,runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--neutral',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.neutral)
    if not all(r['source_unchanged_since_import'].values()) or not r['input_unchanged']:raise RuntimeError('Actualsource/inputchanged duringfullquantumunitaryprice')
    a.output.write_text(json.dumps(neutral.charged.packed(r),indent=2)+'\n');print(json.dumps(neutral.charged.packed(dict(input_sha256=r['input_sha256'],physical=r['actual_ALL_boson_compensated_source_unitary'],elapsed_seconds=r['elapsed_seconds'])),indent=2))
