"""Primitive normalPhi/SU4 static quadratic backreaction price.

Explicit bareCAR-quadratic CONTROL, not original SOURCE/vacuum/unitary transfer.
Execute actual 45normalPhi+45relativecolour primitive alignment/Noether
embedding and all-heavy/all64 conservative path prices, without floatingLambda.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import block_diag
ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_pauli_band_volume as volume
import induced_phi_pauli_four_link_loop as primitive_loop
import induced_phi_backreaction as primitive
control=volume.control
pauli=control.pauli
quantum=control.quantum
force=pauli.contact.transfer.packets.force
loops=primitive_loop.loops
SOURCE_BYTES={**volume.SOURCE_BYTES,**primitive_loop.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
VOLUME_SHA='7ccf67dddd91d89e085ae9b3efd9a98b052a1a4388af280fb18979edba9237e5'


def interval(mean:dict[str,Any],name:str)->Any:
    enc=mean['centre_enclosure'];return loops.iv.mpf([loops.exact(loops.rational(enc[name+'_lower'])).a,loops.exact(loops.rational(enc[name+'_upper'])).b])


def embedding(native:dict[str,Any],q:float)->dict[str,Any]:
    basis=primitive.matrix_unit_basis();geom=force.physical_normal_geometry(native)
    # Actual CartT->primitiveB map, not an assumed coordinate/kinetic identity.
    conversion=np.einsum('bij,aji->ba',basis,force.T).real/4;S=np.kron(np.eye(4),conversion)
    E=S@geom['normal'];O=S@geom['orbit'];H=O.T@O
    delta=np.zeros((4,4,4),complex);delta[0]=pauli.magnetic_shape(native)['delta'];cart=force.cart_coordinates(delta).reshape(60);z=S@cart
    gram_linear=np.einsum('mij,nji->mn',delta,native['frame']).real/4;gram_linear+=gram_linear.T
    J=1/5;curl=np.zeros((45,45),complex)
    for face,(i,j) in enumerate(((0,1),(0,2),(1,2))):
        qi=q if i==0 else 0.;qj=q if j==0 else 0.
        curl[face*15:(face+1)*15,i*15:(i+1)*15]=(1-np.exp(1j*qj))*np.eye(15)
        curl[face*15:(face+1)*15,j*15:(j+1)*15]=(np.exp(1j*qi)-1)*np.eye(15)
    Phi=(4*np.sin(q/2)**2)*J*(E.T@E)
    colour=J*np.kron(np.eye(3),H)+7/20*(curl.conj().T@curl)
    B=np.asarray(block_diag(Phi,colour));minimum=float(np.linalg.eigvalsh(B)[0])
    elastic=primitive.elastic_hessian(native['frame'],basis,2/native['a'],0.)
    restricted=float(z@elastic@z);expected=native['g0']/(2*native['a'])
    orth=float(np.linalg.norm(E.T@O));Gram=float(np.linalg.norm(gram_linear));scale=float(np.max(np.abs(E.T@E-np.eye(45)/8)))
    if max(orth,Gram,scale,abs(restricted-expected))>1e-10 or minimum<=0:raise ArithmeticError('Actual primitiveCart/noether/alignment/quartic embedding failed')
    return dict(normalPhi=45,relativecolour=45,actual_normal_angular_alignment_cross_error=orth,actual_Cart_to_primitiveB_metric_error=scale,actual_Pauli_Gramlinear_error=Gram,actual_Pauli_primitiveB_norm_squared=float(z@z),actual_Pauli_quartic_Hessian_divided_by_Lambda=restricted,actual_expected_Pauli_quartic_Hessian_divided_by_Lambda=expected,actual_colour_orbit_Gram_min=float(np.linalg.eigvalsh(H)[0]),actual_full90_alignment_curl_Hessian_divided_by_Lambda_to_5over4_min=minimum,actual_qtheta=q,physical_formula='CartnormalPhi embedding E=S Qnormal has E^T E=I45/8 and E^T O=0. Primitive J=Lambda^(5/4)/5 gives normalPhi mass Lambda^(5/4) qhat²/40; colourmass Lambda^(5/4)H_orbit/5 plusPOSITIVE7curl²/20. Pauli Gramdark restrictedquartic=Lambda*g0/(2a). No droppedcolourmixing masquerades as fullfield.',kinetic_scope='Original primitiveB principalcometric chi; CartnormalPhi coordinates transform it to8chi. Relativecolour exactGauss principalcometric has positive .6chi plusGauss-square. Hence fullmixed-coordinate principalminimum >=.6chi; no flattening/fittedmetric. This is BASEPOINTquadratic, not nonlinear fullPhi transfer.')


def uniform_mass(model:Any,onsite:float,paths:list[Any])->F:
    mass=F.from_float(float(onsite));axes:list[dict[int,F]]=[{}, {}, {}]
    for displacement,c in paths:
        axis=int(np.flatnonzero(displacement)[0]);length=int(np.sum(abs(displacement)))
        if length in axes[axis]:raise ArithmeticError('DuplicateprimitiveWilsonlength')
        axes[axis][length]=F.from_float(float(complex(c[0]).real))
    for coefficients in axes:
        if model.improved:
            if set(coefficients)!={1,2} or coefficients[2]<=0:raise ArithmeticError('Actual improvedWilsonquadraticpath contract changed')
            u,v=coefficients[1],coefficients[2];candidates=[2*u+2*v,-2*u+2*v];vertex=-u/(4*v)
            if -1<=vertex<=1:candidates.append(-u*u/(4*v)-2*v)
            mass+=min(candidates)
        else:
            if set(coefficients)!={1}:raise ArithmeticError('Actual one-linkWilsonpath contract changed')
            mass-=2*abs(coefficients[1])
    if mass<=0:raise ArithmeticError('Actual primitivepolynomialuniformPhi0 masslower lost')
    return mass


def path_prices(mean:dict[str,Any],a:Any,g0:Any,gs:Any)->tuple[Any,Any,Any,Any,Any,list[dict[str,Any]]]:
    iv=loops.iv;cfg=primitive.PhiConfig(Lambda=1.,**mean['microscopic_config']);heavy=iv.mpf(0);charged=iv.mpf(0);cross=iv.mpf(0);XrhoSum=iv.mpf(0);rhoASum=iv.mpf(0);records=[]
    for load,model,role in primitive.models(cfg):
        on,paths=primitive.scalar_paths(model);mass=loops.exact(uniform_mass(model,float(on),paths));on=loops.exact(F.from_float(abs(float(on))));C1=on;C2=on;JA=iv.mpf(0)
        for displacement,c in paths:
            length=int(np.sum(abs(displacement)));c0=loops.exact(F.from_float(abs(float(complex(c[0]).real))));ci=loops.exact(F.from_float(abs(float(complex(c[1]).imag))))
            C1+=2*length*(c0+ci);C2+=2*length*length*(c0+ci);JA+=2*length*(c0*iv.sqrt(g0)+ci*iv.sqrt(gs))
        if float(mass.a)<=0:raise ArithmeticError('Actual primitiveuniformPhi0 masslower lost')
        gap=2*mass*iv.sqrt(g0);C1=8*C1*(1+iv.sqrt(g0)+iv.sqrt(gs));C2=128*C2*(1+iv.sqrt(g0)+iv.sqrt(gs));JA=8*a*JA
        # 90 normalizedfieldvertices; C2 includes allsecond/mixedcontacts.
        norm=2*90*C1*C1/gap+90*C2
        # BOTH transverseA andBOTH rho sine/cosine; charged only.
        XA=2*iv.sqrt(180)*C1*JA/gap+iv.sqrt(180)*a*C2
        Xrho=2*iv.sqrt(180)*C1*2/gap;rhoA=2*iv.sqrt(4)*JA*2/gap
        records.append(dict(load=int(load),role=role,actual_primitive_uniform_gap_lower=loops.bounds(gap),actual_all90_static_Hessian_norm_upper=loops.bounds(norm),actual_field_photon_direct_cross_norm_upper=loops.bounds(XA),actual_field_density_cross_norm_upper=loops.bounds(Xrho),actual_density_photon_cross_norm_upper=loops.bounds(rhoA)))
        if role.startswith('neutral'):heavy+=norm
        else:charged+=loops.exact(int(load))*norm;cross+=loops.exact(int(load))*XA;XrhoSum+=loops.exact(int(load))*Xrho;rhoASum+=loops.exact(int(load))*rhoA
    return heavy,charged,cross,XrhoSum,rhoASum,records


def run(path:Path)->dict[str,Any]:
    started=time.monotonic();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=VOLUME_SHA:raise ArithmeticError('Frozen00e/7cc Fouriervolume epoch required')
    cap=json.loads(raw);mean=cap['input_descriptor']['actual_root_mean_input'];native=quantum.build_native(mean,4);iv=loops.iv
    a=loops.exact(F(1,64));g=loops.exact(F(64));g0=interval(mean,'g0');gs=interval(mean,'gs');Lambda=loops.exact(F(2)**8192);Lambda54=loops.exact(F(2)**10240);chi=loops.exact(F(2)**-4096)
    N=262144;q=2*iv.pi/N;qhat2=4*iv.sin(q/2)**2;rmin=gs/2
    if float(g0.a)<=float(gs.b):raise ArithmeticError('Actualg0>gs interval needed for exactcolourmass lower')
    alpha=qhat2/40
    if float(alpha.b)>=float((rmin/5).a):raise ArithmeticError('MacroPhi alignmentmass is not limitingfull90sector')
    heavy,charged,cross,Xrho,rhoA,records=path_prices(mean,a,g0,gs)
    # Quartic Phi Hessian positive; keep it for Pauli native control, but
    # omit its POSITIVE contribution in this global90 lower, never negate it.
    hmin=Lambda54*alpha-Lambda*heavy-charged
    if float((hmin/(Lambda54*alpha)).a)<=0:raise ArithmeticError('Actual90fastfield staticmass bound not positive')
    # Same rho/A0Schur: positive-density kernel >= qhat²/(2g), not omitted.
    screened_cross=cross+Xrho*(2*g/qhat2)*rhoA
    correction=screened_cross*screened_cross/hmin
    frequency2=loops.exact(F(3,5))*chi*hmin
    native_controls=[embedding(native,float(2*np.pi/4)),embedding(native,float(2*np.pi/N))]
    desc=dict(plan=cap['input_descriptor']['plan'],plan_revision=pauli.REVISION,scientific_execution_revision=pauli.REVISION,input_receipts=dict(volume=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())),original_source_abi_UNCHANGED=cap['input_descriptor']['original_source_abi_UNCHANGED'],actual_macro_N=N,actual_Lambda='2^8192',actual_chi='2^-4096',actual_a='1/64',actual_g='64',original_clock_UNCHANGED='4096',control='DISTINCTbareCAR quadraticmaterialresponse plusactualprimitivePhi/colour bosonicnormalquadratic. NO original-source/vacuum/unitary/fullnonlinearPhi admission. SAME declaredparameters, no cone fit or omittedrho.')
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),actual_primitive90normal_static_embedding_controls=native_controls,actual_full90_fast_static_mass_lower=loops.bounds(hmin),actual_mass_relative_remainder_upper=loops.bounds((Lambda*heavy+charged)/(Lambda54*alpha)),actual_all90_densitySchur_field_photon_cross_norm_upper=loops.bounds(screened_cross),actual_all90_static_Phi_colour_Schur_photon_kernel_correction_norm_upper=loops.bounds(correction),actual_basepoint_normal_fast_frequency_squared_lower=loops.bounds(frequency2),actual_primitive_species_path_prices=records,physical_price_derivation='ForunitCart/relativecolour Fourierfields, ||h1_a||HS<=C1=8(|onsite|+2sumL|c|)(1+sqrtg0+sqrtgs); normalizedmodeL2 plus spinmatrixHS/operator bounds, twoorientedpaths/endpoints and BOTHqsectors. |TrP h2_ab|<=C2=128(|onsite|+2sumL²|c|)(1+sqrtg0+sqrtgs):4spintrace*2orientations*2Phiendpoints*2derivativeorder*2qsectors*2fieldCauchy. Bothinterbandbubble norm<=2 sum_a||h1_a||HS²/gap<=180C1²/gap; 90x90contact norm<=90C2. Charged all1+63 loads; heavy -Lambda h has Lambda timesbound. Quarticpositive ignoredONLYforlower. PrimitivePhi45 normalCartmass>=Lambda^(5/4) qhat²/40; colour>=Lambda^(5/4)rmin/5. DensityHS=2 perunitmode; positiveSchur inverse<=2g/qhat² and allcrosses retained. Then full90normal Schur correction <=||screenedcross||²/Hmin; not singlePauli truncation.',scientific_interpretation='Within this EXPLICITquadraticcontrol, actualprimitivePhi/colour backreaction on photon statickernel is too small to repair macroscopic unequalcone byretaggingPhi. Exact giantLambda/chi used, not underflowzero. NativephysicalPauli materialcoupling/nonzeroquantumsource transfer remain independentfacts.',review_status='Executed actualprimitiveembedding and intervalbound arithmetic, with explicit conservativepath derivation FORparentmathematicalreview; NOT automatic generaltheorem/vacuum/particle/commoncone admission.',limitations='Staticquadraticcontrol ONLY. FrequencydependentRPA poles, primitivePi+Coul interactingground, nonlinearPhi/source packets and totalNoetherforce not supplied by thisprice. BoundisNOT4096-clock unitary/heat transfer. Literal macro quantumpropagator NOTexecuted.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--volume',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.volume)
    if not all(r['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actualprimitive staticbackreaction pricing')
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(input_sha256=r['input_sha256'],relative_mass_error=r['actual_mass_relative_remainder_upper'],photon_correction=r['actual_all90_static_Phi_colour_Schur_photon_kernel_correction_norm_upper'],frequency_squared=r['actual_basepoint_normal_fast_frequency_squared_lower'],embedding=r['actual_primitive90normal_static_embedding_controls'],elapsed_seconds=r['elapsed_seconds']),indent=2))
