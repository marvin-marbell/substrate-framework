"""Targeted Csamescope proof: ANYreal EM probe, COMPLEXnormalized y/z current."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from fractions import Fraction as F
import sys
import time
from typing import Any
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parent
OWN_BYTES=Path(__file__).read_bytes()
import induced_phi_spin_shape_contact as contact
from induced_phi_em_ygraph_clock_transfer import exact,positive_exact
from induced_phi_em_macro_preparation import scalars
from induced_phi_ir_well_transfer import rational,sqrt_bounds
quantum=contact.quantum
SOURCE_BYTES={**contact.SOURCE_BYTES,Path(__file__).name:OWN_BYTES}
REVISION='d86132736ac1f3d1592b01da60e6e981c5005500265f0d012e979610896318c7'
RESPONSE_SHA='fee24268839d0955c698787b79f9832841e65cc19f9a34fb1363fab087eeaacd'


def full_forward_blocks(native:dict[str,Any],profile:dict[str,Any],phi:np.ndarray,links:np.ndarray,theta:np.ndarray)->list[dict[str,Any]]:
    blocks=[]
    for path in profile['paths']:
        U=np.eye(4,dtype=complex)
        for edge in path['edges']:U=U@links[edge]
        x,y=path['source'],path['target'];fx=np.einsum('a,aij->ij',path['scalars'],phi[x]);fy=np.einsum('a,aij->ij',path['scalars'],phi[y])
        phase=path['seam']*np.exp(1j*np.sum(theta[list(path['edges'])]))
        blocks.append(dict(path=path,forward=(fx@U+U@fy)*phase/2))
    return blocks


def contact_matrix(native:dict[str,Any],blocks:list[dict[str,Any]],probe:np.ndarray,current:np.ndarray)->np.ndarray:
    H=np.zeros((4*native['V'],4*native['V']),dtype=complex)
    for block in blocks:
        path=block['path'];x,y=path['source'],path['target'];indices=list(path['edges'])
        weight=-np.sum(probe[indices])*np.sum(current[indices]);hop=block['forward']
        H[4*x:4*x+4,4*y:4*y+4]+=weight*hop
        # ComplexSOURCEcoefficient multipliesBOTHorderedrealcontact blocks;
        # doNOTconjugateweight onreverse (thatwouldmakewrongHermitianmap).
        H[4*y:4*y+4,4*x:4*x+4]+=weight*hop.conj().T
    return H


def CAR_norm_upper(matrix:np.ndarray)->float:
    # RigorousfiniteCARupper: splitHerm/antiHerm, sumtheirwholeFocknorms.
    # No manybodyoccupationfit; eachHermitianh diagonalizesdGamma(h).
    value=0.
    for H in ((matrix+matrix.conj().T)/2,(matrix-matrix.conj().T)/(2j)):
        eigen=np.linalg.eigvalsh(H)
        value+=max(float(eigen[eigen>0].sum()),float(-eigen[eigen<0].sum()))
    return value


def actual_mixed_contact_control(mean:dict[str,Any])->dict[str,Any]:
    native=quantum.build_native(mean,4);V,M=native['V'],native['M'];rng=np.random.default_rng(20261014)
    f=np.sqrt(2/V)*np.cos(2*np.pi*native['sites'][:,0]/4);dy=np.zeros(3*V);dz=np.zeros(3*V);dy[1::3]=f;dz[2::3]=f
    # NativeALL60Cart/nonuniformPhi andorderednoncommutingSU4 links.
    perturb=.03*rng.normal(size=(V,4,15));T=contact.transfer.packets.force.T
    phi=np.broadcast_to(native['frame'],(V,4,4,4)).copy()+np.einsum('xaA,Aij->xaij',perturb,T)
    angles=.03*rng.normal(size=(3*V,15));links=np.array([expm(1j*np.einsum('A,Aij->ij',a,T)) for a in angles])
    theta=native['Q']@(np.sqrt(2)*rng.normal(size=M))
    e_random=native['Q']@rng.normal(size=M);e_random/=np.linalg.norm(e_random)
    index={tuple(site):i for i,site in enumerate(native['sites'])};e_loop=np.zeros(3*V)
    e_loop[3*index[(0,0,0)]+1]=1;e_loop[3*index[(0,1,0)]+2]=1;e_loop[3*index[(0,0,1)]+1]=-1;e_loop[3*index[(0,0,0)]+2]=-1;e_loop/=2
    probes=[(dy+dz)/np.sqrt(2),e_random,e_loop]
    source=np.array([1,1j],dtype=complex)/np.sqrt(2);current=source[0]*dy+source[1]*dz
    fields=[full_forward_blocks(native,p,phi,links,theta) for p in native['Pprofiles']]
    perprofile_caps=[];mixed_zero=[];cross_parent=[]
    for p,blocks in zip(native['Pprofiles'],fields):
        cap=0.
        for length in (1,2):
            hopmax=max(float(np.linalg.norm(b['forward'],2)) for b in blocks if b['path']['length']==length)
            cap+=16*length**2*hopmax
        perprofile_caps.append(cap)
        mixed_zero.append(float(np.linalg.norm(contact_matrix(native,blocks,dy,dz),2)))
        # FrozenPhi/control derivative implementation is independent offull
        # nonuniformblocks: verifyitsaxialmixedcontacts aswell, notreuseoutput.
        frozen=quantum.path_derivative(native,p,theta,dy,2,other=dz)
        cross_parent.append(float(np.linalg.norm(frozen,2)))
    cap_total=sum(p['load']*cap for p,cap in zip(native['Pprofiles'],perprofile_caps));rows=[]
    for e in probes:
        axisnorm=np.linalg.norm(e.reshape(V,3),axis=0);factor=float(abs(source[0])*axisnorm[1]+abs(source[1])*axisnorm[2]);CAR=0.;pathmass={1:0.,2:0.};wrong=0.
        for p,blocks in zip(native['Pprofiles'],fields):
            Ce=contact_matrix(native,blocks,e,current);CAR+=p['load']*CAR_norm_upper(Ce)
            Cy=contact_matrix(native,blocks,e,dy);Cz=contact_matrix(native,blocks,e,dz)
            wrong+=p['load']*float(np.linalg.norm(Ce-(source[0]*Cy+source[1]*Cz),2))
            for length in (1,2):
                # Checkliteralshift-CS rowmass, sourceaxisweightsnotdropped.
                for axis,d in ((1,dy),(2,dz)):
                    weights=[abs(float(np.sum(e[list(b['path']['edges'])]))*float(np.sum(d[list(b['path']['edges'])]))) for b in blocks if b['path']['length']==length and b['path']['axis']==axis]
                    observed=float(sum(weights));bound=length*length*axisnorm[axis]
                    if observed>bound+1e-10:raise ArithmeticError('Physicalarbitraryprobe shift-CS bound failed')
                    pathmass[length]=max(pathmass[length],observed)
        rows.append(dict(real_probe_unit_norm=float(np.linalg.norm(e)),real_probe_Gauss_norm=float(np.linalg.norm(native['D']@e)),outside_selected_yz_span_norm=float(np.sqrt(max(0.,float(np.linalg.norm(e)**2-np.dot(e,dy)**2-np.dot(e,dz)**2)))),actual_complex_source_CAR_contact_upper=CAR,axial_CS_factor=factor,Csame_generalized_uniform_row_cap=cap_total,Csame_actual_probe_scaled_row_cap=cap_total*factor,literal_complex_ordered_contact_map_error=wrong,actual_shift_CS_path_masses=pathmass))
        if CAR>cap_total*factor+1e-8 or factor>1+1e-10 or wrong>1e-9:raise ArithmeticError('Actualgenericrealprobe/complexcurrentcontact bound failed')
    # ForbiddenCURVEDchargedpathsupportfalsifier; NOTanaddedactualH term.
    curved_edges=[3*index[(0,0,0)]+1,3*index[(0,1,0)]+2]
    curved_mixed=float(np.sum(dy[curved_edges])*np.sum(dz[curved_edges]))
    if curved_mixed==0 or max(mixed_zero+cross_parent)>1e-12:raise ArithmeticError('Axialmixedcontact/nonaxialfalsifier failed')
    for row in rows:row['forbidden_nonaxial_path_mixed_yz_weight_falsifier']=curved_mixed
    return dict(N=4,V=V,M=M,macro_nonNyquist_unitnorm_cosine_normalization=True,source_complex_current_coefficients=dict(real=np.real(source).tolist(),imag=np.imag(source).tolist()),dy_norm=float(np.linalg.norm(dy)),dz_norm=float(np.linalg.norm(dz)),dy_dot_dz=float(dy@dz),actual_full_nonuniform_Phi_SU4_mixed_yz_contact_norms=mixed_zero,independent_frozen_Phi_path_derivative_mixed_yz_norms=cross_parent,actual_generic_noncommuting_link_commutator_norm=float(np.linalg.norm(links[0]@links[1]-links[1]@links[0],2)),rows=rows,scope='Operatorcontractcontrols atdeclaredgenericfullCartPhi/SU4 fields, notsourcePhiGaussian/interactingpropagator/initialeta-chartsample. Csame_fielduniformrowformulaverified; macrooriginalCsamevalueboundusespinnedb367Fref/dHop sourcechart.')


def original_source_uniform_hop_caps(bridge:dict[str,Any],mean:dict[str,Any])->dict[str,Any]:
    # RetainACTUALsource chart uniformbounds, NOTmeanreplacement ofPhi.
    phi_error=exact(bridge['actual_joint_support_Phi_operator_difference_to_C000'])
    frame=exact(bridge['actual_222_000_Cart_frame_distance_upper'])
    eta=(phi_error-frame)/2
    if eta!=F(2)**-2048:raise ArithmeticError('OriginalcompactPhi/SU4jointsource radius changed')
    enclosure=mean['centre_enclosure'];root0=sqrt_bounds(rational(enclosure['g0_upper']))[1];roots=sqrt_bounds(rational(enclosure['gs_upper']))[1]
    Cref=Cprice=F(0);rows=[]
    for profile in quantum.build_native(mean,2)['Pprofiles']:
        scalar=scalars(profile)
        for length in (1,2):
            c0,cj=scalar['path'][length];sabs=abs(c0)+abs(cj)
            Fref=abs(c0)*root0+abs(cj)*roots
            Fmax=Fref+sabs*frame
            dHop=phi_error*sabs+sqrt_bounds(F(length))[1]*eta*Fmax
            Cref+=16*scalar['load']*length**2*Fref;Cprice+=16*scalar['load']*length**2*dHop
            rows.append(dict(charged_load=scalar['load'],literal_axial_length=length,Fref_uniform_ALLTheta_reference_hop_operator_cap=positive_exact(Fref),dHop_uniform_ACTUALsource_Phi_SU4_to_reference_difference=positive_exact(dHop),four_spin_forward_reverse_CAR_row_multiplier=16,length_squared_directional_shift_CS_multiplier=length**2))
    if Cref!=exact(bridge['reference_whole_Theta_contact_operator_cap']) or Cprice!=exact(bridge['corrected_initial_whole_Theta_contact_operator_price']):raise ArithmeticError('Actualprimitiveuniformreference/sourcehopcaps doNOTreconstructfrozenCsame')
    return dict(original_joint_source_radius=positive_exact(eta),actual_uniform_Phi_operator_difference_to_C000=positive_exact(phi_error),actual_222_000_frame_bridge=positive_exact(frame),rows=rows,proof='PointwiseonACTUALsourcejointnormal/linkb2support, ||Phi_x−C000||op<=2eta+actualdeltaFrame. Everydistinctone/twolink colourpath ||U_path−z||<=sqrtL eta; ThetaPeierlsphaseunitmodulus forALLTheta. LiteralendpointSYMMETRIZEDnonuniformhop difference<=dHop_L=(2eta+deltaFrame)sum|c|+sqrtL eta*(Fref+deltaFrame sum|c|), uniformEVERYlink/site/colour/CARray, includingoriginalprojectedheavyisometry. FullPhi sourceNOTreplacedbymean; FrefdenotesCOMPARISONoperatorONLY, dHopremainsinCsame. ThisACTUALper-link bound iswhatshift-CS integrates.',forbidden_extension='ProofrequiresSTRAIGHTAXIALchargedone/twolink paths andfixedsourcepolarizationsdy/dz unitcos(qx), notarbitrarycurved/multiaxischargedpaths. SU4neutralplaquette/heavy/Pi²/Coulomb terms haveNOTheta coordinatecontact. AddednonaxialchargedpathneedsNEWcontactanalysis; suppliedcurvedsupportfalsifier hasnonzeroCyz andmustNOTbeadmittedbythisproof.')


def run(response_path:Path)->dict[str,Any]:
    started=time.monotonic();raw=response_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=RESPONSE_SHA:raise ArithmeticError('Parentfrozen e363/fee242 response epoch required')
    response=json.loads(raw);_,record=quantum.checked(response['input_descriptor']['input_receipts']['source']);force={};bridge={}
    for key in ('transition','receiver','source_energy','green','response','inserted','force','current','retarded','clock','bridge'):
        _,record=quantum.checked(record['input_descriptor']['input_receipts'][key])
        if key=='force':force=record
        if key=='bridge':bridge=record
    _,mean=quantum.checked(bridge['input_descriptor']['input_receipts']['mean'])
    Cref=exact(bridge['reference_whole_Theta_contact_operator_cap']);Cprice=exact(bridge['corrected_initial_whole_Theta_contact_operator_price']);Csame=exact(force['prices']['FULL_current_initial_directional_contact_cap']);Cany=exact(force['prices']['FULL_current_arbitrary_EM_direction_contact_cap'])
    if Csame!=Cref+Cprice:raise ArithmeticError('Actualdirectionalcontact scalarprice changed')
    desc=dict(plan=response['input_descriptor']['plan'],plan_revision=REVISION,scientific_execution_revision=REVISION,input_receipts=dict(response=dict(path=str(response_path),sha256=hashlib.sha256(raw).hexdigest())),source_abi=response['input_descriptor']['source_abi'],source='UNCHANGEDoriginalPhi/source/e363/fee242; PROOFscopecompletionONLY, allfrozenbytes/probabilitiesclocks preserved',contract='Csame forANYREALunitcoexactPi_probe andANYCOMPLEXunitselectedy/zcurrent; sourcecurrentandprobe needNOTsameray/q',price_change=False)
    desc['actual_original_source_uniform_per_link_contact_contract']=original_source_uniform_hop_caps(bridge,mean)
    desc['source_E_and_probability_contract']='OriginalunitarysourceE,p=.5 unchanged. SupplierNEWsourceE occursBEFOREchargedEuclideanfilter; p=.5 mustNOTbeassignedtopostfilterZC/ZN.'
    return dict(input_descriptor=desc,input_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),existing_reference_contact_cap=positive_exact(Cref),existing_initial_Phi_contact_price=positive_exact(Cprice),existing_Csame_universal_realprobe_complexsource_cap=positive_exact(Csame),existing_Cany_valid_coarser_cap=positive_exact(Cany),actual_full_operator_mixed_contact_control=actual_mixed_contact_control(mean),exact_universal_contact_proof='Literalallchargedone/twolink paths STRAIGHTAXIAL. Selecteddy,dz realunitcos(qx) areconstantalongtheirrespectivelinks; eachpathdotdy=L f_x onypaths and0else, similarlydz. ANYrealunitcoexacte decomposesorthogonallybyedgeaxis (ey,ez,ex). AtlengthL, Σx |(Σ_{k=0}^{L-1} e_y(x+k y))(L f_x)| <=L||Σ_k shift_y^k ey||2||f||2 <=L²||ey||2. SAMEz. Fourspinforward/reverse CARnorm<=16Fref (or16dHop) timesabsoluteweight. Summingliteralb36734–42 rowsgives||Ce,y||<=Cref||ey||2 andFULL−REF<=Cprice||ey||2, likewisez. HenceforCOMPLEXu, ||Ce,Ju||<=Csame[|uy|||ey||2+|uz|||ez||2]<=Csame||u||2 sqrt(||ey||2²+||ez||2²)<=Csame||u||2||e||2. ThisisUNIVERSALANYrealcoexactprobe, notonlyselected2probe span andnotjustsameaxis. Cyz=Czy=0 bypathsupportexact, butoutside-spanmixedcontactse,y aregenericallynonzeroandcoveredbyshift-CS. Complexu coefficientsmultiplyBOTHforward/reverse contactblocks unchanged; contactitselfneednotHermitian.',price_and_consumer_contract='Csame line73 isb367Cref+Cprice. Jref/Jprice line70–71 require sqrt2 forcomplexsourceCURRENToperator; CONTACTbilinearhasadditionalorthogonalprobeaxisnorms, soCsameneedsNOsqrt2. InitialPi productΠe JuPsi=JuΠePsi−iCe,JuPsi validwith existingCsame forEVERYrealunitdirection, includingallcurrent-sourcePi moments. chargedresponse212 pi=n(Jref*pi0+Csame) remainsvalid; anypi=n(Jref*pi0+Cany) conservativevalid. chargedresponse200 delta_PiJ Cprice andCref*insertion validbySAMEshift-CS bound, notcomplexGramfixassumption. No reprice/newfallback/no oldsourceediting required; nativeoperatorchecks NOTinteractingresponsecurve.',exact_source_locations=['induced_phi_em_ygraph_centre_bridge.py34–42 literaluniformFref/dHop/16L² rows','induced_phi_em_ygraph_centre_bridge.py68 actualunitnormcosqx y/z axialconstantproof','induced_phi_em_gaussian_variational.py123–138 actualaxialone/twolink primitivepaths','induced_phi_ir_c_full_action.py128–144 fullnonuniformPhi orderedSU4/realEMsympaths','induced_phi_em_full_source_force_clock.py70–73 and88,98','induced_phi_em_charged_photon_response.py188,200,212'],scope='TargetedmixedcontactproofcompletionboundParent e363/fee242. ScientificactualN4 fullnonuniformCartPhi/SU4/coexactcomplexsourceoperatorcontrol; no fullPhi/sourcechange/retardedHrefcurve/pole/commoncone/Compton/newadmission. Same Csamescalarprice valid, broaderCanyretained. No sourceE before/afterEuclideanfilter alias: untouchedoriginalunitary-sourcep=.5 only.',source_at_import={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes_utf8=data.decode()) for name,data in SOURCE_BYTES.items()},source_unchanged_since_import={name:(ROOT/name).read_bytes()==data for name,data in SOURCE_BYTES.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--response',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.response)
    if not all(result['source_unchanged_since_import'].values()):raise ArithmeticError('Source changed during actual calculation')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_sha256','actual_full_operator_mixed_contact_control','elapsed_seconds')},indent=2))
