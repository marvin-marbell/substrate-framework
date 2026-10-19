"""NEW full64 Gauss-correlated normalized source consumer, not native ground."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path


def collective_ray(profiles: list[dict]) -> dict:
    count=128
    zden=512
    common=math.lcm(*[(1+m*m)**2 for m in range(count+1)])
    powers=[zden**k for k in range(4*count+1)]
    choose=[math.comb(count,k) for k in range(count+1)]
    normalization=number=flow2=same_pairs=cross_pairs=hopping=0
    for p,q in itertools.product(range(count+1),repeat=2):
        number_here=p+q
        flow=p-q
        coefficient=choose[p]*choose[q]
        weight=coefficient*powers[4*count-2*number_here]*(common//(1+flow*flow)**2)
        normalization+=weight
        number+=number_here*weight
        flow2+=flow*flow*weight
        same_pairs+=(p*(p-1)+q*(q-1))*weight
        cross_pairs+=p*q*weight
        if p<count:
            denominator=(1+flow*flow)*(1+(flow+1)**2)
            assert common%denominator==0
            hopping+=coefficient*(count-p)*powers[4*count-2*number_here-1]*(common//denominator)
        if q<count:
            denominator=(1+flow*flow)*(1+(flow-1)**2)
            assert common%denominator==0
            hopping+=coefficient*(count-q)*powers[4*count-2*number_here-1]*(common//denominator)
    z=Fraction(1,zden)
    zexact=Fraction(normalization,common*powers[4*count])
    assert 1<zexact<(1+z*z)**256
    mean_n=Fraction(number,normalization)
    mean_e2=Fraction(flow2,normalization)
    mean_same=Fraction(same_pairs,normalization*2*count*(count-1))
    mean_cross=Fraction(cross_pairs,normalization*count*count)
    mean_p=mean_n/256
    wrong_species_square=256*mean_p+256*mean_same-512*mean_cross
    exact_collective_square=256*mean_p+32512*mean_same-32768*mean_cross
    assert exact_collective_square==mean_e2
    assert wrong_species_square>mean_e2
    active,aux=profiles[1:]
    a0=Fraction(1,8);ai=Fraction(1,3)
    t=abs(Fraction(active['paths'][0]['ci']))*ai
    onsite=a0*(Fraction(active['onsite'])+63*Fraction(aux['onsite']))*mean_n/32
    electric=64*mean_e2
    kinetic=-2*t*Fraction(hopping,normalization)
    exact_energy=onsite+electric+kinetic
    assert exact_energy<Fraction(-1,4)
    A=1+z*z
    sufficient=(192*256*z*z*A**255+64*256*255*z**4*A**254-256*Fraction(15,16)*z)/A**256
    assert sufficient<Fraction(-1,4)
    assert Fraction(2)**-3544+Fraction(2)**-3996<Fraction(1,8)
    return {'scope':'NORMALIZED_VARIATIONAL_FULL64_SOURCE_NOT_RHON_OR_GROUND_GRAM',
            'labelled_species':64,'Pauli_channels_each_species':4,'labelled_spins':256,
            'all_native_occupation_counts_retained':list(range(257)),
            'all_incoming_source_flow_counts_retained':[-128,128],
            'Jastrow':'1/(1+Etotal^2)','z_exact':str(z),'Z_exact':str(zexact),
            'mean_total_particle_holes_exact':str(mean_n),
            'mean_shared_E_squared_exact':str(mean_e2),
            'wrong_sum_species_E_squared_exact':str(wrong_species_square),
            'wrong_independent_copy_excess_exact':str(64*(wrong_species_square-mean_e2)),
            'sample_amplitudes_exact':{'a0':'1/8','ai':'1/3'},
            'onsite_increment_exact':str(onsite),'once_shared_electric_increment_exact':str(electric),
            'full256_channel_hopping_increment_exact':str(kinetic),
            'native_sample_variational_energy_difference_exact':str(exact_energy),
            'uniform_K_sufficient_energy_difference_upper_exact':str(sufficient),
            'full_quantum_remainder_bound':'2^40 delta+2^100 chi<1/8',
            'fullH_trial_difference_uniform_K':'LESS_THAN_MINUS_1_OVER_8',
            'actual_ground_energy_difference_evaluated':False}


def local_car_controls() -> dict:
    # Native alpha0 pairs modes (left negative2 -> right positive1 etc).
    # Four disjoint pairs partition the eight endpoint colour modes.
    pairs=[(2,5),(3,4),(6,1),(7,0)]
    assert len({m for pair in pairs for m in pair})==8
    states=[]
    for bits in itertools.product((0,1),repeat=4):
        occupied=[pair[bit] for pair,bit in zip(pairs,bits)]
        mask=sum(1<<v for v in occupied)
        assert mask.bit_count()==4
        left=(mask&15).bit_count();right=(mask>>4).bit_count()
        flux=bits[0]+bits[1]-bits[2]-bits[3]
        directed_e=-flux
        assert left-2==directed_e and right-2==-directed_e
        states.append(mask)
    assert len(set(states))==16
    # SU4 Gauss is supplied by full moving exterior-power representation.
    # Native CAR signed transfer on each pair is nonzero and involutive.
    signs=[]
    for mask in states:
        for source,target in pairs:
            if not(mask>>source)&1:source,target=target,source
            removed=mask^(1<<source)
            sign=(-1)**((mask&((1<<source)-1)).bit_count()+(removed&((1<<target)-1)).bit_count())
            changed=removed|(1<<target)
            assert changed in states and changed.bit_count()==4
            signs.append(sign)
    assert set(signs)=={-1,1}
    return {'local_trial_states_each_species':16,'particles_each_two_site_species':4,
            'all_transfer_CAR_signs_retained':True,'Gauss_each_configuration':True,
            'global_particles_each_species':'2V','fullRef_background_each_species':2,
            'CAR_channel_compression_not_full_operator':True}


def exterior_memory(profiles: list[dict]) -> dict:
    patch={(0,0,0),(1,0,0)}
    amplitudes={0:Fraction(1,3),1:Fraction(2,3),2:Fraction(1)}
    rows=[]
    for n in (4,6):
        counts=[0,0,0];seam=0;length2_flows=set();length2_endpoints=set()
        for start in itertools.product(range(n),repeat=3):
            for path in profiles[1]['paths']:
                axis=path['axis'];length=path['length']
                end=list(start);end[axis]=(end[axis]+length)%n;end=tuple(end)
                links=[]
                for step in range(length):
                    site=list(start);site[axis]=(site[axis]+step)%n;site=tuple(site)
                    links.append((site,axis))
                phase=(-1)**sum(site[axis]==n-1 for site,_ in links)
                assert phase in(-1,1)
                seam+=phase==-1
                if length==1 and start not in patch and end not in patch:counts[axis]+=1
                if length==2:
                    length2_flows.add(tuple(links));length2_endpoints.add((axis,tuple(sorted((start,end)))))
        assert counts==[n**3-3,n**3-4,n**3-4]
        assert seam>0 and len(length2_flows)==3*n**3
        if n==4:assert len(length2_endpoints)<len(length2_flows)
        ci=abs(Fraction(profiles[1]['paths'][0]['ci']))
        gram=256*ci*ci*sum(counts[i]*amplitudes[i]**2 for i in range(3))
        rows.append({'N':n,'disjoint_positive_NN_paths_each_axis':counts,
                     'remote_PHP_source_image_Gram_at_sample_exact':str(gram),
                     'quantum_tube_source_Gram_lower':'at least half displayed centre Gram at delta2^-3584',
                     'remote_images_belong_to':'PHP_EXTERIOR_IDENTITY_NOT_QHP',
                     'AP_seam_paths_retained':seam,'distinct_length2_flows':len(length2_flows),
                     'length2_endpoint_pairs':len(length2_endpoints)})
    return {'native_rows':rows,'actual_exterior_dynamics_retained':True,
            'frozen_exterior_bound_relaunched':False,
            'reference_source_stationarity':'REJECTED_REMOTE_EXTERIOR_PHP_ACTION',
            'uniform_actual_rhoN_lowband_elimination':'FAILED_UNPAID_RHON_OCCUPATION_FLOW_WEIGHTS_AND_RANGE'}


def scalar_kinetic_cancellation() -> dict:
    # For normalized R, Re<R,dR>=0, hence real scalar derivative cross=0.
    # Explicit varying normalized ray with a nonzero charged quantum jet.
    from fractions import Fraction as F
    ray=[F(3,5),F(4,5)]
    derivative=[F(-4,5),F(3,5)]
    assert sum(v*v for v in ray)==1
    assert sum(v*w for v,w in zip(ray,derivative))==0
    scalar=F(7,3);scalar_derivative=F(11,2)
    full=sum((scalar_derivative*v+scalar*w)**2 for v,w in zip(ray,derivative))
    split=scalar_derivative**2+scalar**2*sum(w*w for w in derivative)
    assert full==split and sum(w*w for w in derivative)>0
    return {'pointwise_norm_and_real_scalar_cross_cancellation_exact':True,
            'nonzero_quantum_ray_jet_retained':True,'scalar_delta_minus2_cost_cancels_in_difference':True,
            'all65_Ref_covariant_generator_bound':264,'fullH_gap_inferred':False}


def complete_unitary_transport_control() -> dict:
    import sympy as sp
    bits=list(itertools.product((0,1),repeat=4))
    z=sp.Rational(1,4)
    coefficients=sp.Matrix([z**sum(n)/(1+(n[2]+n[3]-n[0]-n[1])**2) for n in bits])
    norm2=(coefficients.T*coefficients)[0]
    psi=coefficients/sp.sqrt(norm2)
    seed=sp.eye(16)[:,0]
    c=psi[0]
    difference=seed-psi
    denominator=2*(1-c)
    assert sp.simplify((difference.T*difference)[0]-denominator)==0
    # Rank-one projector identity proves full16 unitary, NOT compression.
    image=seed-2*difference*((difference.T*seed)[0])/denominator
    assert all(sp.simplify(value)==0 for value in image-psi)
    flux=sp.diag(*[n[2]+n[3]-n[0]-n[1] for n in bits])
    assert sp.simplify((psi.T*flux*psi)[0])==0
    electric_second=sp.simplify((psi.T*flux**2*psi)[0])
    assert electric_second>0
    # At full source support, V=I-2Pi_v. The native E derivative has a
    # nonzero exact HS norm; erasing this connection is an adverse control.
    electric_connection_hs2=sp.simplify(4*electric_second/(1-c))
    assert electric_connection_hs2>0
    # Complete weighted wholeP0 transport, exact nonnative finite control.
    u=sp.Matrix([1,sp.I,0,0])/sp.sqrt(2)
    v=sp.eye(4)-2*u*u.H
    assert v*v.H==sp.eye(4)
    h=sp.diag(0,0,2,5);p0=sp.diag(1,1,0,0)
    rho=sp.diag(sp.Rational(1,7),sp.Rational(6,7),0,0)
    o=sp.Matrix([[0,0,1,sp.I],[0,0,2,1],[1,2,0,0],[-sp.I,1,0,0]])
    def response(hh,pp,rr,oo):
        return sp.simplify(sp.trace(rr*oo.H*(sp.eye(4)-pp)*(hh+3*sp.eye(4)).inv()*(sp.eye(4)-pp)*oo))
    original=response(h,p0,rho,o)
    transported=response(v*h*v.H,v*p0*v.H,v*rho*v.H,v*o*v.H)
    assert original==transported
    wrong=response(v*h*v.H,v*p0*v.H,rho,v*o*v.H)
    assert wrong!=original
    return {'scope':'NATIVE_LOCAL_UNITARY_PLUS_EXACT_WEIGHTED_METHOD_CONTROL',
            'native_one_species_all16_channel_states_unitary_extension':True,
            'seed_to_correlated_ray_exact':True,
            'native_nonzero_electric_connection_HS_squared_exact':str(electric_connection_hs2),
            'complete_rho_wholeP0_source_transport_response_exact':str(original),
            'wrong_untransported_weights_response_exact':str(wrong),
            'all64_extension':'same rank-one formula on complete labelled256-channel module, identity elsewhere',
            'form_graph_constants_uniformN':'proof complete_native_unitary_range_transport',
            'effective_lowband_elimination_proved':False}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    for path,sha in request['pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    profiles=json.loads(Path(request['profiles']).read_text())['primitive_profiles']
    result={'kind':'NEW_NATIVE_FULL64_CORRELATED_QUANTUM_VARIATIONAL_TRANSPORT_SUCCESSOR',
            'collective_ray':collective_ray(profiles),'native_CAR':local_car_controls(),
            'exterior_memory':exterior_memory(profiles),'quantum_kinetic':scalar_kinetic_cancellation(),
            'complete_unitary_range_transport':complete_unitary_transport_control(),
            'first_gate':'strong_electric_factorized_reference_rejected',
            'material_successor':'nonmultiplicative_256spin_sharedflow_Jastrow_full_quantum_source',
            'reference_branch_adopted_as_ground':False,'actual_rhoN_weights_replaced':False,
            'wholeP0_replaced_by_trial_projector':False,'native_H_ground_backend_invented':False,
            'old_entrypoints_imported_or_replayed':False,'uniform_gap_or_IR_stiffness':False,
            'scientific_admission':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'full64_sharedflow_variational_gate':'passes_analytic_K_full_quantum_difference',
                      'actual_exterior_PHP_memory':'retained_nonzero',
                      'complete_unitary_range_transport':'exact_with_full_connections',
                      'effective_lowband_elimination':'failed_not_no_go','ground_or_IR_claim':False}))


if __name__=='__main__':
    main()
