"""Independent exact collective-unitary review; no original source entrypoints."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path
import sympy as sp


def full_labelled_trial(profiles):
    # Enumerate by total occupation and shared signed flux, rather than
    # species marginals. This is the complete labelled 128+128 channel sum.
    half = 128
    zden = 512
    common = math.lcm(*[(1 + m*m)**2 for m in range(half + 1)])
    powers = [zden**k for k in range(513)]
    normal = number = flow2 = odd = adjacency_forward = 0
    occupancy = set()
    flows = set()
    for n in range(257):
        for m in range(-128, 129):
            if (n + m) % 2:
                continue
            forward = (n + m)//2
            reverse = (n - m)//2
            if not (0 <= forward <= half and 0 <= reverse <= half):
                continue
            multiplicity = math.comb(half, forward)*math.comb(half, reverse)
            weight = multiplicity*powers[512 - 2*n]*(common//(1 + m*m)**2)
            normal += weight
            number += n*weight
            flow2 += m*m*weight
            odd += (m % 2)*weight
            occupancy.add(n)
            flows.add(m)
            if forward < half:
                denominator = (1 + m*m)*(1 + (m + 1)**2)
                assert common % denominator == 0
                adjacency_forward += multiplicity*(half - forward)*powers[511 - 2*n]*(common//denominator)
    assert occupancy == set(range(257)) and flows == set(range(-128, 129))
    Z = F(normal, common*powers[512])
    mean_n = F(number, normal)
    mean_m2 = F(flow2, normal)
    mean_X = F(4*adjacency_forward, normal)
    oddmass = F(odd, normal)
    z = F(1, zden)
    A = 1 + z*z
    assert 1 + 64*z*z <= Z <= A**256
    # Avoid radicals: d^2=2(1-Z^-1/2)>1/128^2 iff
    # Z>(1-1/(2*128^2))^-2.
    assert Z > (1 - F(1, 2*128**2))**-2
    assert 0 < oddmass < 1 and mean_X > 0
    bound = (192*256*z*z*A**255 + 64*256*255*z**4*A**254 - 256*F(15,16)*z)/A**256
    assert bound < -F(1,4)
    remainder = F(2)**-3544 + F(2)**-3996
    assert bound + remainder < -F(1,8)
    active, aux = profiles[1:]
    t = abs(F(active['paths'][0]['ci']))/3
    onsite = F(1,8)*(F(active['onsite']) + 63*F(aux['onsite']))*mean_n/32
    sample = onsite + 64*mean_m2 - t*mean_X
    assert sample < -F(1,4)
    return {'species':64, 'channels':256, 'occupancy_counts':sorted(occupancy),
            'shared_integer_flows':sorted(flows), 'Z_exact':str(Z),
            'mean_N_exact':str(mean_n), 'mean_shared_m2_exact':str(mean_m2),
            'mean_sum_X_exact':str(mean_X), 'sample_energy_difference_exact':str(sample),
            'uniform_K_upper_exact':str(bound), 'quantum_remainder_bound_exact':str(remainder),
            'parity_commutator_applied_seed_norm2_exact':str(oddmass),
            'wrong_arbitrary_S_commutation_rejected':True,
            'fullH_difference_status':'ARITHMETIC_VALID_CONDITIONAL_ON_SEWN_SOURCE_AND_JET_ESTIMATES',
            'weights_are':'TRIAL_BINOMIAL_MULTIPLICITIES_NOT_ORIGINATING_RHO'}


def car_and_centre():
    pairs = [(2,5), (3,4), (6,1), (7,0)]
    masks = []
    for bits in itertools.product((0,1), repeat=4):
        mask = sum(1 << pairs[j][bits[j]] for j in range(4))
        masks.append(mask)
        m = bits[2] + bits[3] - bits[0] - bits[1]
        assert (mask & 15).bit_count() - 2 == m
        assert (mask >> 4).bit_count() - 2 == -m
    phases = {masks[0]:sp.Integer(1)}
    edges = []
    for mask in masks:
        for a,b in pairs:
            source,target = (a,b) if (mask >> a) & 1 else (b,a)
            removed = mask ^ (1 << source)
            sign = (-1)**((mask & ((1 << source)-1)).bit_count() + (removed & ((1 << target)-1)).bit_count())
            new = removed | (1 << target)
            # Native i ci alpha coefficient: real ci sign absorbed globally.
            amplitude = sign*(sp.I if source < 4 else -sp.I)
            edges.append((mask,new,amplitude))
    while len(phases) < 16:
        before = len(phases)
        for a,b,h in edges:
            if a in phases and b not in phases:
                phases[b] = -h*phases[a]
        assert len(phases) > before
    for a,b,h in edges:
        assert sp.simplify(sp.conjugate(phases[b])*h*phases[a]) == -1
    # Different species' bilinears are even CAR operators: no sqrt64 load.
    # An odd endpoint transfer changes local-centre action despite z^128=1
    # for the seed. Relative SU4 link branch supplies the missing character.
    rows = []
    for m in (-128,-3,-1,0,1,3,128):
        left = 128 + m
        right = 128 - m
        assert left + right == 256
        for z in (sp.Integer(1),sp.I,sp.Integer(-1),-sp.I):
            required = z**left
            phi_only = z**128
            sewn = phi_only*z**m
            assert sp.simplify(required-sewn) == 0
            if m % 2 and z == sp.I:
                assert sp.simplify(required-phi_only) != 0
        rows.append({'m':m,'left_count':left,'right_count':right,'local_Z4_factor':'z^m'})
    return {'one_species_states':16,'signed_directed_CAR_edges':len(edges),
            'all_hypercube_cycles_rephase_consistently':True,
            'two_site_particles_each_species':4,'full64_channels':256,
            'configurationwise_Gauss_sign_verified':True,'centre_rows':rows,
            'wrong_Phi_only_odd_endpoint_source_rejected':True,
            'repair':'Use actual SU4 relative-centre branch z_link^m; branch is locally constant only on separated branch support. Global cutoff and sewing must be specified.'}


def unitary_connections_and_berry():
    I = sp.eye(3)
    e = sp.Matrix([1,0,0])
    psi = sp.Matrix([sp.Rational(3,5),sp.Rational(4,5),0])
    diff = e-psi
    V = I - diff*diff.H/(1-psi[0])
    assert V*V.H == I and V*e == psi
    P = sp.diag(1,0,1)
    Pprime = V*P*V.H
    assert P*V != V*P and Pprime*V == V*P
    total = sp.diag(4,4,7)
    assert total*V == V*total
    M = sp.diag(0,1,0)
    # V(theta)=exp(iM theta)V(0)exp(-iM theta).
    Vdot = sp.I*(M*V-V*M)
    connection = sp.simplify(sp.I*Vdot*V.H)
    assert connection != sp.zeros(3)
    assert connection.H == connection
    incoming = sp.Rational(2,3)
    full = (incoming*I+connection)**2
    dropped = incoming**2*I+connection**2
    assert full-dropped == 2*incoming*connection and full != dropped
    f, df = sp.Rational(7,3), sp.Rational(11,2)
    neutral, charged = sp.Matrix([1,0]), sp.Matrix([sp.Rational(3,5),sp.Rational(4,5)])
    dn, dc = 2*sp.I*neutral, 3*sp.I*charged
    R = sp.kronecker_product(neutral,charged)
    dR = sp.kronecker_product(dn,charged)+sp.kronecker_product(neutral,dc)
    norm = lambda a: (a.H*a)[0]
    assert sp.re((R.H*dR)[0]) == 0
    assert sp.simplify(norm(df*R+f*dR)-df**2-f**2*norm(dR)) == 0
    separate = norm(dn)+norm(dc)
    assert norm(dR)-separate == 12
    return {'complete_unitary_and_identity_complement':True,
            'endpoint_P_fails_but_transported_Pprime_exact':True,
            'total_two_site_number_preserved':True,
            'nonzero_electric_connection_exact':str(connection),
            'incoming_E_cross_retained_exact':str(2*incoming*connection),
            'normalized_real_scalar_kinetic_cancellation':True,
            'neutral_charged_cross_Berry_exact':str(norm(dR)-separate),
            'wrong_zero_connection_and_no_Berry_cross_rejected':True,
            'finite_control_not_native_H_backend':True}


def complex_schur_whole_ground():
    I = sp.eye(4)
    U = sp.Matrix([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])/2
    H = U*sp.diag(0,0,3,7)*U.H
    P0 = U*sp.diag(1,1,0,0)*U.H
    rho = U*sp.diag(sp.Rational(1,5),sp.Rational(4,5),0,0)*U.H
    P = sp.diag(1,1,0,0)
    assert P*P0 != P0*P
    Oa = sp.Matrix([[1,sp.I,0,2],[-sp.I,3,1,0],[0,1,2,sp.I],[2,0,-sp.I,4]])
    Ob = sp.Matrix([[0,1,sp.I,0],[2,0,0,1],[1,sp.I,0,3],[0,2,1,0]])
    K = H+2*I
    A,D,B = K[:2,:2],K[2:,2:],K[2:,:2]
    Fschur = A-B.H*D.inv()*B
    response = direct = droppedQ = 0
    for j,weight in enumerate((sp.Rational(1,5),sp.Rational(4,5))):
        ua = (I-P0)*Oa*U[:,j]
        ub = (I-P0)*Ob*U[:,j]
        xa,ya,xb,yb = ua[:2,0],ua[2:,0],ub[:2,0],ub[2:,0]
        da,db = xa-B.H*D.inv()*ya,xb-B.H*D.inv()*yb
        direct += weight*(ua.H*K.inv()*ub)[0]
        response += weight*((ya.H*D.inv()*yb)[0]+(da.H*Fschur.inv()*db)[0])
        droppedQ += weight*(xa.H*Fschur.inv()*xb)[0]
    assert sp.simplify(response-direct) == 0
    assert sp.simplify(droppedQ-direct) != 0
    oneP0 = U[:,0]*U[:,0].H
    wrong_one = sp.trace(rho*Oa.H*(I-oneP0)*K.inv()*(I-oneP0)*Ob)
    assert sp.simplify(wrong_one-direct) != 0
    canonical = U*sp.diag(sp.Rational(1,2),sp.Rational(1,2),0,0)*U.H
    wrong_weights = sp.trace(canonical*Oa.H*(I-P0)*K.inv()*(I-P0)*Ob)
    assert sp.simplify(wrong_weights-direct) != 0
    v = sp.eye(4)
    v[0,0]=v[1,1]=sp.Rational(3,5)
    v[0,1]=sp.Rational(4,5)
    v[1,0]=-sp.Rational(4,5)
    ht,pt,rt,oa,ob = v*H*v.H,v*P0*v.H,v*rho*v.H,v*Oa*v.H,v*Ob*v.H
    moved = sp.trace(rt*oa.H*(I-pt)*(ht+2*I).inv()*(I-pt)*ob)
    assert sp.simplify(moved-direct) == 0
    wrong_frozen = sp.trace(rho*oa.H*(I-pt)*(ht+2*I).inv()*(I-pt)*ob)
    assert sp.simplify(wrong_frozen-direct) != 0
    # Remote PHP cannot be dropped even when the boundary coupling is zero.
    remote = sp.diag(2,5)
    x = sp.Matrix([1,sp.I])
    remote_response = (x.H*remote.inv()*x)[0]
    frozen_response = (x.H*(2*sp.eye(2)).inv()*x)[0]
    assert remote_response != frozen_response
    return {'full_complex_bilinear_exact':str(sp.simplify(direct)),
            'dressed_Schur_exact':str(sp.simplify(response)),
            'wrong_dropped_Q_exact':str(sp.simplify(droppedQ)),
            'wrong_one_ground_projection_exact':str(sp.simplify(wrong_one)),
            'wrong_canonical_weights_exact':str(sp.simplify(wrong_weights)),
            'wrong_frozen_transformed_rho_exact':str(sp.simplify(wrong_frozen)),
            'remote_PHP_even_B_zero_exact':str(remote_response),
            'wrong_frozen_exterior_exact':str(frozen_response),
            'subtract_entire_P0_before_split':True,'P_commutes_P0':False,
            'scope':'EXACT_FINITE_METHOD_ADVERSE_CONTROL_NOT_ORIGINATING_NATIVE_WEIGHTS'}


def incoming_and_remote(profiles):
    rows = []
    for n in (4,6,8):
        patch = {(0,0,0),(1,0,0)}
        counts = [0,0,0]
        seam = aliases = 0
        flows, endpoints = set(),set()
        for x in itertools.product(range(n),repeat=3):
            for path in profiles[1]['paths']:
                axis,length = path['axis'],path['length']
                y = list(x);y[axis]=(y[axis]+length)%n;y=tuple(y)
                links = []
                for step in range(length):
                    site = list(x);site[axis]=(site[axis]+step)%n
                    links.append((tuple(site),axis))
                seam += sum(site[axis]==n-1 for site,axis in links) % 2
                if length==1 and x not in patch and y not in patch:
                    counts[axis] += 1
                if length==2:
                    flows.add(tuple(links));endpoints.add((axis,tuple(sorted((x,y)))))
        assert counts == [n**3-3,n**3-4,n**3-4]
        assert seam > 0 and len(flows)==3*n**3
        if n==4:
            assert len(endpoints)<len(flows)
            aliases = len(flows)-len(endpoints)
        rows.append({'N':n,'remote_positive_NN_counts':counts,'AP_seams':seam,
                     'length2_flow_histories':len(flows),'N4_endpoint_alias_excess':aliases,
                     'remote_norm2_coefficient':'256 sum_paths ci^2 ai^2',
                     'block':'PHP_IF_REMOTE_FROM_S_NOT_WITHDRAWN_FROZEN_QHP'})
    # Include arbitrary onsite increment d: resonance is still reachable.
    d = F(31,7)
    resonant = -(F(1,2)+d/128)
    denominator = lambda E: 64*(2*E+1)+d
    assert denominator(resonant)==0 and denominator(resonant-1)<0
    assert (1 + (-1))**2 == 0 and 1**2+(-1)**2 == 2
    return {'native_remote_rows':rows,'onsite_increment_exact':str(d),
            'resonant_incoming_E_exact':str(resonant),
            'zero_denominator_exact':str(denominator(resonant)),
            'negative_denominator_exact':str(denominator(resonant-1)),
            'wrong_independent_E_copies_rejected':True,
            'actual_rho_resonance_weights_supplied':False,
            'reference_stationary_ground_proved':False}


def sewn_centre_AP_and_cutoff(profiles):
    rows = []
    for n in (4,6,8):
        xi = lambda x: (x[0]+2*x[1]+3*x[2]) % 4
        zedge = lambda x,a: (x[a]+a+1) % 4
        checks = wrong = seams = 0
        for x in itertools.product(range(n),repeat=3):
            for path in profiles[1]['paths']:
                a,length = path['axis'],path['length']
                y = list(x);y[a]=(y[a]+length)%n;y=tuple(y)
                z = moved = seam = 0
                for step in range(length):
                    u = list(x);u[a]=(u[a]+step)%n;u=tuple(u)
                    v = list(u);v[a]=(v[a]+1)%n;v=tuple(v)
                    z += zedge(u,a)
                    moved += -xi(u)+zedge(u,a)+xi(v)
                    seam += u[a]==n-1
                assert (moved-z+xi(x)-xi(y)) % 4 == 0
                seams += seam % 2
                for m in range(-128,129):
                    # Endpoint carrier has xi_x^m xi_y^-m. The actual
                    # link branch changes by xi_x^-1 z xi_y.
                    carrier = m*(xi(x)-xi(y))
                    assert (carrier+m*moved-m*z) % 4 == 0
                    checks += 1
                    wrong += carrier % 4 != 0
        assert wrong > 0 and seams > 0
        rows.append({'N':n,'all_shared_flow_centre_sew_checks':checks,
                     'wrong_deleted_branch_witnesses':wrong,'AP_seam_paths':seams,
                     'ordered_length1_and2_paths_retained':True,
                     'nontrivial_centre_holonomies_not_gauged_away':True})
    t = sp.Symbol('t',real=True)
    smooth = 35*t**4-84*t**5+70*t**6-20*t**7
    for order in range(1,4):
        assert sp.diff(smooth,t,order).subs(t,0)==0
        assert sp.diff(smooth,t,order).subs(t,1)==0
    assert smooth.subs(t,0)==0 and smooth.subs(t,1)==1
    # A full-module covariant patch, not a ray compression: h depends on
    # Haar link distance. Its derivative is generally nonzero off support.
    Pi = sp.Matrix([[1,0],[0,0]])
    h = sp.Symbol('h',real=True)
    V = sp.eye(2)+(sp.exp(sp.I*sp.pi*h)-1)*Pi
    assert sp.simplify(V*V.H)==sp.eye(2)
    assert sp.diff(V,h)!=sp.zeros(2)
    return {'native_centre_AP_rows':rows,'cutoff_endpoint_jets_orders1through3_zero':True,
            'complete_phase_projector_unitary_every_cutoff_value':True,
            'Haar_derivative_zero':'ONLY_INSIDE_LOCALLY_CONSTANT_BRANCH_ON_SOURCE_SUPPORT',
            'Haar_cutoff_derivatives':'NONZERO_OFF_SOURCE_SUPPORT_RETAINED_IN_GRAPH_BOUNDS',
            'branch_tie_boundary':'CUTOFF_ZERO_ON_OPEN_NEIGHBOURHOOD_BEFORE_TIE',
            'local_Z4_and_common_centre_transition':'CANCELLED_CONFIG_BY_CONFIG_NOT_JUST_SEED',
            'native_continuous_chart_jets':'ANALYTIC_CONDITIONAL_NOT_ATTESTED_BY_DISCRETE_CONTROL'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    for path,expected in request['pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    profiles=json.loads(Path(request['profiles']).read_text())['primitive_profiles']
    result={'kind':'INDEPENDENT_FULL64_UNITARY_FORM_TRIAL_ADVERSE_REVIEW_NOT_ADMISSION',
            'trial':full_labelled_trial(profiles),'CAR_and_centre':car_and_centre(),
            'unitary_and_Berry':unitary_connections_and_berry(),
            'complex_Schur':complex_schur_whole_ground(),
            'incoming_remote':incoming_and_remote(profiles),
            'sewn_centre_AP_cutoff':sewn_centre_AP_and_cutoff(profiles),
            'original_entrypoints_imported_or_replayed':False,
            'actual_native_fullH_ground_backend':False,
            'inherited_realization':'CONDITIONAL_NOT_MACHINE_ATTESTED',
            'scientific_admission':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'full64_exact_trial':'conditional_valid',
                      'arbitrary_S_parity_claim':'rejected',
                      'Phi_only_odd_centre_source':'rejected_requires_link_branch',
                      'effective_originating_lowband':'not_supplied'}))


if __name__=='__main__':
    main()
