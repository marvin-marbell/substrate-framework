#!/usr/bin/env python3
"""NEW C real-cover admission: whole-chart LEFT-SF price and smooth boundary.

This is an operator/domain and free-harmonic boundary calculation, NOT a macro
interacting source/Pe4 probability. The declared RPA heat remains REFERENCE.
The original Gaussian control is not replaced or retagged. No photon Gibbs
trace, isolated harmonic ground, global fullF=MfullF, or bounded a† is assumed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
if __name__=='__main__':
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
import mpmath as mp
import numpy as np
from scipy.linalg import eigh
import physical_c_real_cover_kernel as kernel
SOURCE_BYTES=Path(__file__).read_bytes()
KERNEL_BYTES=Path(kernel.__file__).read_bytes()


def whole_chart_LEFT_SF_bound() -> dict[str,Any]:
    """Actual declared normal tube, ALL occupations, SAME quantum colour links."""
    with mp.workdps(160):
        volume=mp.mpf(2)**72;naux=mp.mpf(10)**1150/16
        radius=mp.mpf('1e-1000');frob=2*radius
        rows=[];logarithm=mp.mpf(0)
        for name,count,k in [('active',mp.mpf(1),mp.mpf('.1256305951044483')),('aux',naux,mp.mpf('1e-250'))]:
            b0=1/(2*k)-3;bmax=1/(2*k)+3
            db=k*(frob+frob**2);df=4*k*frob
            eta_b=-mp.log1p(-db/b0)
            eta=eta_b+2*mp.log1p(df/b0)
            condition_M=max(1/b0,bmax)*16
            spread=2*mp.log(condition_M)+2*eta
            flow=eta*mp.expm1(spread/2)/spread
            species_log=volume*(eta_b+eta+4*flow)
            weighted=count*species_log;logarithm+=weighted
            rows.append(dict(species=name,count=mp.nstr(count,60),kappa=mp.nstr(k,60),
                canonical_ALL_occupation_range='0..4V',eta_h=mp.nstr(eta,80),
                LEFT_onebody_halfSF_flow=mp.nstr(flow,80),
                scalarpref_AND_allFock_log=mp.nstr(weighted,80)))
        epsM=mp.expm1(logarithm);epsK=2*epsM+epsM**2
        steps=mp.mpf(2)**19;delta=1/mp.mpf(2)**20
        neutral_T_error=2*epsK/(1-epsK)
        absolute_prefix=steps*neutral_T_error
        tau_floor=mp.exp(-steps*delta)
        weighted_relative=absolute_prefix/tau_floor
        return dict(actual_declared_Cart_normal_radius=mp.nstr(radius),matrix_Frobenius_radius=mp.nstr(frob),
            exact_same_quantum_colour=True,normal_only_reference=True,rows=rows,
            WHOLE_chart_LEFT_M_minus_I_norm_upper=mp.nstr(epsM,80),
            global_operator_DeltaK_over_lambda0_upper=mp.nstr(epsK,80),
            neutral_normalizer_ratio_interval=[f'1-{mp.nstr(epsK,30)}',f'1+{mp.nstr(epsK,30)}'],
            normalized_neutral_T_operator_error_upper=mp.nstr(neutral_T_error,80),
            prefix_steps=int(steps),smooth_energy_width=mp.nstr(delta,50),
            SAME_boundary_absolute_HS_prefix_error_upper=mp.nstr(absolute_prefix,80),
            conditioned_REFERENCE_tau_floor=mp.nstr(tau_floor,80),
            conditioned_SAME_boundary_weighted_relative_prefix_upper=mp.nstr(weighted_relative,80),
            conditioned_normalized_HS_vector_error_upper=mp.nstr(2*weighted_relative,80),
            conditioned_density_trace_error_upper=mp.nstr(4*weighted_relative,80),
            derivation='B/L relative conditioning BEFORElog: eta_b=-log(1-db/b0), eta=eta_b+2log(1+df/b0). Hermitian log-monotonicity and intrinsicC averaging give ||dhC||<=eta. LEFT halfSF generator integral gives flow=eta expm1(spread/2)/spread; ALL n<=4V and HALFscalarpref give V(eta_b+eta+4flow) per flavour. Uniform on the WHOLE compact normal tube, with SAME exact colour and uncompactified unitary EM links. Set M=I off chart. Outside column B is unchanged: DeltaK=D Kchart0+Kchart0 Ddag+D Kchart0 Ddag, ||DeltaK||<=epsK lambda0. Norm normalization and contraction telescope give displayed SAME-boundary bounds.',
            hypothesis='Actual nonzero conditioned preparationGram required. No claim for unconditioned Gaussian tau; no global photon Schatten/canonical trace. Charged/source/photon graph errors do NOT follow from this neutral vector estimate.',
            macro_source_or_Pe4_admitted=False)


def smooth_real_harmonic_boundary() -> dict[str,Any]:
    """Actual continuum integrals on R^3, never a Galerkin harmonic ground."""
    with mp.workdps(70):
        nu=mp.mpf('.02');delta=1/mp.mpf(2)**20
        cap=mp.sqrt(2*delta/nu);steps=mp.mpf(2)**19
        def density(u):
            if u==1:return mp.mpf(0)
            return 4/mp.sqrt(mp.pi)*cap**3*u**2*mp.exp(-(cap*u)**2-2*u**4/(1-u**4))
        grid=[0,mp.mpf('.25'),mp.mpf('.5'),mp.mpf('.75'),mp.mpf('.9'),mp.mpf('.98'),1]
        gram=mp.mpf(mp.quad(density,grid))
        p2=mp.mpf(mp.quad(lambda u:density(u)*(cap*u)**2,grid))/gram
        def gradient_integrand(u):
            if u==1:return mp.mpf(0)
            derivative=-cap*u-4*u**3/(cap*(1-u**4)**2)
            return density(u)*derivative**2
        q2=mp.mpf(mp.quad(gradient_integrand,grid))/gram
        propagated=mp.mpf(mp.quad(lambda u:density(u)*mp.exp(-2*steps*nu*(cap*u)**2/2),grid))/gram
        sharp=[]
        for width in [mp.mpf('.01'),mp.mpf('.001'),mp.mpf('.0001')]:
            # Explicit continuous linear boundary ramp; its radial derivative
            # exposes the divergent q² price of the sharp momentum-ball limit.
            inner=1-width
            base=lambda u:4/mp.sqrt(mp.pi)*cap**3*u**2*mp.exp(-(cap*u)**2)
            sg=mp.mpf(mp.quad(base,[0,inner]))+mp.mpf(mp.quad(lambda u:base(u)*((1-u)/width)**2,[inner,1]))
            sq=mp.mpf(mp.quad(lambda u:base(u)*(cap*u)**2,[0,inner]))
            sq+=mp.mpf(mp.quad(lambda u:base(u)*(-cap*u*(1-u)/width-1/(cap*width))**2,[inner,1]))
            sharp.append(dict(ramp_relative_width=str(width),actual_q2=mp.nstr(sq/sg,60),actual_Gram=mp.nstr(sg,60)))
        if not gram>0 or propagated<mp.exp(-2*steps*delta):
            raise ArithmeticError('Actual real smooth boundary integral or tau floor failed')
        return dict(real_domain='R^3 NORMALIZED harmonic a=sqrt(beta)qtheta, pa=ptheta/sqrt(beta); Gaussian Fourier density pi^(-3/2)exp(-|pa|²)',
            harmonic_TRANSFER_H='nu|p|²/2',lambda_norm=1,isolated_ground_exists=False,
            exact_bump='f(E)=exp[-(E/Delta)^2/(1-(E/Delta)^2)] belowDelta; zero above. Flat at upper edge; even smooth extension nearE=0.',
            actual_preparation_Gram=mp.nstr(gram,60),actual_cost_minus_log=mp.nstr(-mp.log(gram),60),
            actual_p2=mp.nstr(p2,60),actual_q2=mp.nstr(q2,60),
            actual_log_tau=mp.nstr(mp.log(propagated)/2,60),proved_tau_floor=mp.nstr(mp.exp(-steps*delta),60),
            actual_harmonic_reference_ladder_creation_Gram=mp.nstr((p2+q2+3)/2,60),
            ladder_scope='This harmonic reference ladder is NOT a propagating physical photon. Its large actual Gram records the weighted graph cost; a† is not treated as bounded.',
            adverse_sharp_ball_ramps=sharp,
            domain='Filtered momentum wavefunction is Cinf compactly supported, flat at the sphere boundary, hence its inverse Fourier transform is Schwartz. Sharp ball has a boundary delta in the momentum derivative, so qpsi is NOT L2.',
            interacting_macro_preparation_Gram_inferred=False)


def actual_source_projection_price() -> dict[str,Any]:
    """Actual uncompressed node source vs projected complete carrier source."""
    records=[]
    p0=kernel.NativeParameters(normal_order=1,colour_order=1)
    law,_,_=kernel.native_law(p0)
    mean=law.potential(np.zeros((2,3)),np.zeros((2,4,4,4)),np.broadcast_to(np.eye(4),(2,3,4,4)).copy())
    g,m4=kernel.alpha.second_quantized(mean['h'],4);_,m5=kernel.alpha.second_quantized(np.eye(8),5)
    _,frame=eigh(g);sea=frame[:,-1]
    creation=kernel.creation_matrices(m4,m5)
    for order in (2,3,4):
        p=kernel.NativeParameters(quadrature_order=order,normal_order=1,colour_order=1)
        field=kernel.field_regulator(p);basis=field['basis']
        packets=[]
        for cover,_,_ in field['points']:
            phases=(kernel.alpha.charge_contract(kernel.alpha.Parameters())['bare_e']*kernel.real_graph()['Q']@cover).reshape(2,3)
            packets.append(law.source_packet_jet(phases,np.broadcast_to(np.eye(4),(2,3,4,4)).copy(),(np.zeros((2,3)),np.zeros((2,3))))['psi'])
        packets=np.array(packets)
        trial=np.zeros((field['dimension'],70),complex);trial[1]=sea
        nodal=basis@trial
        uncompressed=np.einsum('nba,nf,bgf->ang',packets,nodal,creation,optimize=True)
        projected=kernel.apply_source(kernel.source_coefficients(field,packets),creation,trial.reshape(-1))
        full=float(np.linalg.norm(uncompressed)**2);compressed=float(np.linalg.norm(projected)**2)
        price=full-compressed
        if price<0 or abs(full-1)>1e-10:raise ArithmeticError('Actual source CAR/projection price failed')
        records.append(dict(quadrature_order=order,real_coordinates=5,complete_neutral_carriers=70,complete_charged_carriers=56,
            input_photon_Gram=float(np.linalg.norm(trial)**2),actual_uncompressed_source_Gram=full,
            actual_grade1_compressed_source_Gram=compressed,actual_omitted_source_norm_squared=price))
    e=kernel.alpha.charge_contract(kernel.alpha.Parameters())['bare_e']
    exact=(1+math.exp(-e**2/4))/2
    return dict(rows=records,exact_continuum_grade1_compressed_source_Gram=exact,
        exact_continuum_omitted_source_norm_squared=1-exact,
        proof='Actual incoming coexact one-photon packet; source Wilson phase is e q_hx/sqrt2. Pgrade1 discards simultaneous coexact1 and harmonic excitations generated by its Weyl character. Exact Gaussian matrix element gives (1+exp(-e²/4))/2. Full nodal CAR source Gram1, all70/56 carrier columns, no source normalization or Pe gate.',
        consumer_limit='Source compression is a positive real price, not merely approximate wiring. No full interacting charged-half/Pe4 bound is inferred from these omitted norms.')


def interacting_bump_graph_contract() -> dict[str,Any]:
    return dict(operator='T0=KN0/lambdaN0 on the actual full real-Gauss physical space; g(T0)=f(-logT0), g smooth and zero below exp(-Delta). Extendg smoothly above1; physical spectrum<=1.',
        sufficient_domain_lemma='For selfadjoint q or p, bounded [q,T0]/[p,T0] on its core implies g(T0) preserves Domq/Domp. Fourier functional calculus gives ||[B,g(T0)]|| <= ||[B,T0]|| integral |t ghat(t)|dt. This is finite for a smooth compact extensiong. It is NOT an assertion that arbitrary energy filtering preserves Schwartz.',
        chart_q='q commutes with chi and SF0 multiplication. [q,Rreference] is bounded for exact real free harmonic and coexact Mehler heat. SAME compact Cart/colour chart bounds SF0 and all real-q Wilson derivatives for finite V,Naux, including physical Q factors. Thus [q,Fchart]/[p,Fchart] are bounded.',
        outside_q='The macro retained2Eaux-minusFreshYoung margin per site exceeds3, absorbing coexact Mehler zero-point: max latticeOmega<3 and2V-2coexact modes. Thus actual HoutTRANSFER>=g p²/2+Lambda sum|Phi|⁴/2+positivecoexact quadratic potential. [q,Hout]=i g p; Duhamel and ||p exp(-sHout)||<=1/sqrt(g e s) give ||[q,exp(-tHout)]||<=2sqrt(g t/e). Here g=nu/beta for physicalqtheta, ornu for normalizeda=sqrt(beta)qtheta.',
        outside_p='[p,Hout] contains the coexact linearq force and actual linearPhi Wilson derivative; Coulomb and quartic are q-independent. Point force<=C sum|Phi|, hence coercivity bounds force exp(-sHout) by finite C V^(3/4)(2/Lambda)^(1/4)(1/(4e s))^(1/4). Coexactq term scales s^(-1/2). Both are Duhamel-integrable, yielding bounded [p,exp(-tHout)]. Finite derivative constants depend on ALL flavours and literal charges; no tiny numerical bound is claimed.',
        physical_frame='q/p commute with physical colourGamma(V); normal/colour kinetics use physicalDminus=QdQdag. Source physical Ref3 connection cancellation is measured in separate actual Dminus capsule.',
        conclusion='The declared smooth reference bump preserves the real electromagnetic first q/p graph domain of the actual interacting K0, under its displayed coercive Friedrichs law. Full incoming a_fdag and receiver a_outdag S Xi are defined, with their ACTUAL finite graph norms. This is a domain proof, NOT quantitative B-weighted comparison, macro prepGram, all highTheta moments or Pe4 admission.',
        charged_half_log_domain='For bounded positive KC, complete charged half source is in H_C domain since sup_{0<=x<=||KC||/lambdaN} x|logx|² is finite, including true zero. One neutral lambda only.',
        remaining_quantitative_consumers=['Actual interacting macro preparationGram/cost','Photon-weighted neutral comparison, not just normalized vector error','Actual charged/source first-output law and literalPe4 Gram','Full-coordinate and Hermite/heat/source regulator errors','Physical mass/dispersion/current scattering output and induced electric law'])


def charged_output_operator_control() -> dict[str,Any]:
    """Dimension-free complete-half/time law, including zero spectrum."""
    neutral=whole_chart_LEFT_SF_bound()
    with mp.workdps(130):
        epsK=mp.mpf(neutral['global_operator_DeltaK_over_lambda0_upper'])
        relative_TC=2*epsK/(1-epsK)
        rows=[]
        for spacing in (mp.mpf('.02'),mp.mpf('.01'),mp.mpf('.005')):
            for timevalue in (mp.mpf(0),mp.mpf('.25'),mp.mpf('.5'),mp.mpf(1)):
                s=abs(timevalue/spacing);angle_gap=1/(1+s)
                # cosh(pi*s)exp[-s(pi-angle_gap)], without overflow.
                trigonometric=(1+mp.exp(-2*mp.pi*s))*mp.exp(s*angle_gap)/2
                constant=4*mp.sqrt(2)/mp.pi*trigonometric/mp.sin(angle_gap/2)**mp.mpf('1.5')
                rows.append(dict(spacing=mp.nstr(spacing),time_PHYSICAL=mp.nstr(timevalue),
                    actual_contour_constant=mp.nstr(constant,70),
                    complete_charged_half_time_operator_error_over_sqrt_aC0=mp.nstr(constant*mp.sqrt(relative_TC),70)))
        outside=mp.mpf(kernel.full_family_definition()['outside_global_H_lower'])/(mp.mpf(2)**72)-3
        if outside<=0:raise ArithmeticError('Actual outside margin does not absorb reference zero-point')
        return dict(actual_relative_TC_error_over_aC0_upper=mp.nstr(relative_TC,80),rows=rows,
            outside_coercive_zero_point_margin_per_site=mp.nstr(outside,80),
            definition='Ti=KCi/lambdaNi with ONE neutral normalization each law; aC0=||TC0|| is an ACTUAL charged/neutral ratio, NOT assumed1. Complete half/time consumer is Ti^(1/2+i t/a); truezero maps0.',
            contour_proof='For z=.5+i s, A^z-B^z=(sin(pi z)/pi)integral_ray t^z(A+t)^-1(A-B)(B+t)^-1dt. Rotate theta=sign(s)(pi-1/(1+|s|)); c=cos(theta/2). Resolvent integrand uses min(delta/(c²r²),2/(cr)); splitr=delta/(2c). Integral<=4sqrt2 c^(-3/2)sqrt(delta). sin(pi z) contourphase=cosh(pi s)exp(-|s||theta|), evaluated stably. This removes the false exp(pi t/a) loss; constantO((1+|t/a|)^(3/2)). No spectralgap/photon trace required.',
            source_photon_contract='Let b=||(Bin psi1-Bin psi0)||/||Bin psi0|| be ACTUAL weighted input graph error. Normalized photonstate error<=2b, sourceS norm<=sqrt2 gives du<=2sqrt2 b. Kernel-time source error<=sqrt(aC1)du+Cs sqrt(deltaTC) sqrt(graw0). Divide by ACTUALsqrt(ghalf0) for relativewhole-half error; normalized state error<=2timesrelative. No b orghalf0 lower bound borrowed from neutral vector error.',
            receiver_probability_contract='For actual positive mu_i, frameeffects Ei=RiRi†/mui satisfy ||E1-E0||<=2||R1R1†-R0R0†||/mu0, without Gramminimum/fullrank. Probabilitydifference<=effecterror+4timesrelativewhole-half/time sourceerror. End-to-end CONDITIONAL bound requires actual weighted input/receiver/source receipts.',
            first_output_contract='For positive A,B with norms<=a,b and delta=||A-B||<=1, ||AlogA-BlogB||<=delta[2+2log2-logdelta+a+b]. Resolvent integral split[0,delta],[delta,1],[1,infinity] proves this attruezero, dimensionfree. RawsourceS norms/du price first-output numerator; divide by ACTUALhalfGram and account its measured variation. No old h56 mean promoted.',
            literal_Pe4_contract='Pe4 is TC spectral window[exp(-4),1]. Sharp stability needs measured boundary tails, not neutral vectorerror. Choose smoothscalarf supported inside window and equal1 on smallerwindow. Pe4Gram1>=reference innerwindowGram minus actual source-state error and operatorLipschitzprice Cf deltaTC ghalf0. Reference innerwindowGram/Cf must be computed; no invented chargedgap/sourcegate.',
            missing_macro_receipts=['aC0','actual weighted Bin error','actual ghalf0/innerwindowPe4 Gram','actual receiver frameerror/mu0','fullcoordinate/Hermite/source regulator price'],
            macro_source_output_admitted=False)



def actual_null_factor_power_smoke() -> dict[str,Any]:
    """Actual frozen SF charge-middle adversary; DIAGNOSTIC, no source gate."""
    from scipy.linalg import expm
    p=kernel.NativeParameters();law,_,direction=kernel.native_law(p)
    phases=np.array([[.13,-.07,.09],[-.04,.11,-.08]])
    links=np.broadcast_to(np.eye(4),(2,3,4,4)).copy()
    factors=[]
    for remainder in (np.zeros((2,4,4,4)),np.broadcast_to(1e-8*direction,(2,4,4,4))):
        point=law.potential(phases,remainder,links)
        h,masks=kernel.alpha.second_quantized(point['h'],4)
        middle=[i for i,mask in enumerate(masks) if (mask&15).bit_count()==2]
        factors.append(expm((h+point['pref']*np.eye(70))/2)[:,middle])
    svd=[np.linalg.svd(f,full_matrices=False) for f in factors]
    scale=svd[0][1][0]**2
    positive=[f@f.conj().T/scale for f in factors]
    delta=float(np.max(abs(eigh(positive[1]-positive[0],eigvals_only=True))))
    def function(which,kind,s):
        u,values,_=svd[which];values=values**2/scale
        coefficients=values**(.5+1j*s) if kind=='power' else values*np.log(values)
        return (u*coefficients)@u.conj().T
    rows=[]
    for s in (0.,25.,100.,200.):
        gap=1/(1+s)
        factor=(1+math.exp(-2*math.pi*s))*math.exp(s*gap)/2
        constant=4*math.sqrt(2)/math.pi*factor/math.sin(gap/2)**1.5
        actual=float(np.linalg.norm(function(1,'power',s)-function(0,'power',s),2))
        upper=constant*math.sqrt(delta)
        if actual>upper*(1+1e-9):raise ArithmeticError('Actual SF/null factor violates complete-half contour bound')
        rows.append(dict(time_over_spacing=s,actual_complete_half_time_operator_error=actual,
            actual_contour_upper=upper))
    a,b=(float(np.max(eigh(k,eigvals_only=True))) for k in positive)
    entropy_actual=float(np.linalg.norm(function(1,'entropy',0)-function(0,'entropy',0),2))
    entropy_upper=delta*(2+2*math.log(2)-math.log(delta)+a+b)
    if entropy_actual>entropy_upper*(1+1e-9):raise ArithmeticError('Actual SF/null factor violates output entropy bound')
    return dict(actual_operator_delta=delta,actual_rank=36,actual_physical_carriers=70,
        actual_true_null_dimension=34,rows=rows,
        actual_first_output_AlogA_error=entropy_actual,actual_first_output_entropy_upper=entropy_upper,
        scope='Actual frozen C-SF zero-electric-middle range-rotation adversary used ONLY to exercise the operator theorem including true nulls. This diagnostic is NOT the NEW real-cover field transfer and installs NO gate in its source. Powers from exact thin-SVD positive range, zero on actual nulls; no fake eigenvalue floor.')

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.time()
    result=dict(whole_chart_LEFT_SF=whole_chart_LEFT_SF_bound(),actual_REAL_smooth_boundary=smooth_real_harmonic_boundary(),
        actual_source_projection_price=actual_source_projection_price(),interacting_bump_graph_contract=interacting_bump_graph_contract(),
        complete_charged_half_time_output_control=charged_output_operator_control(),
        actual_true_null_power_entropy_smoke=actual_null_factor_power_smoke())
    inputs=json.dumps(dict(kernel_source=hashlib.sha256(KERNEL_BYTES).hexdigest(),radius='1e-1000',sigma='1e-1500',beta='1e150',kaux='1e-250',Nt=2**20),sort_keys=True,separators=(',',':'))
    result['execution']=dict(source_at_IMPORT=hashlib.sha256(SOURCE_BYTES).hexdigest(),source_bytes_at_IMPORT=SOURCE_BYTES.decode(),
        kernel_source_at_IMPORT=hashlib.sha256(KERNEL_BYTES).hexdigest(),kernel_bytes_at_IMPORT=KERNEL_BYTES.decode(),
        captured_ledger=kernel.LEDGER_HASH,frozen_Alpha56_sources=kernel.alpha.SOURCE_HASHES,
        input_json=inputs,input_sha256=hashlib.sha256(inputs.encode()).hexdigest(),runtime_seconds=time.time()-started,
        runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,mpmath=mp.__version__,platform=platform.platform()),
        own_startup_unchanged=Path(__file__).read_bytes()==SOURCE_BYTES,kernel_startup_unchanged=Path(kernel.__file__).read_bytes()==KERNEL_BYTES)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=args.output,source=result['execution']['source_at_IMPORT'],kernel=result['execution']['kernel_source_at_IMPORT'],
        input_sha256=result['execution']['input_sha256'],macro_neutral_vector_upper=result['whole_chart_LEFT_SF']['conditioned_normalized_HS_vector_error_upper'],
        actual_free_harmonic_prepGram=result['actual_REAL_smooth_boundary']['actual_preparation_Gram'],runtime_seconds=result['execution']['runtime_seconds'])))
if __name__=='__main__':main()
