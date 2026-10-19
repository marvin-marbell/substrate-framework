"""NEW physical1 algebra/form consumer; no native ground or Hamiltonian backend."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
import sympy as sp


def native_gate(profiles: list[dict]) -> dict:
    active, auxiliary = profiles[1:]
    assert active['load'] == 1 and auxiliary['load'] == 63
    assert active['paths'] == auxiliary['paths']
    ci = abs(Fraction(active['paths'][0]['ci']))
    ratio2_lower = 4 * ci**2 / 128
    assert ratio2_lower > 1
    shared = lambda flow, incoming: 64 * ((incoming + sum(flow))**2 - incoming**2)
    assert shared([1], -1) == -64
    assert shared([1], Fraction(-1, 2)) == 0
    assert shared([1, -1], 0) == 0
    assert 64 * (1**2 + (-1)**2) == 128
    # Exact local canonical Pauli channel, all64 labelled species. Spectator
    # occupations embed each one-particle two-mode channel into global2V.
    coherent_eigenvalue = 64 * ci
    orthogonal_first_image_squared = 64 * ci**2
    assert coherent_eigenvalue**2 == 64 * orthogonal_first_image_squared
    return {'physical_hopping': 1, 'full64_coherent_2norm_over_64_squared_lower_on_K': str(ratio2_lower),
            'strong_electric_sufficient_gate': 'FAILED_NOT_NO_GO',
            'incoming_fullE_negative_increment': -64, 'incoming_character_zero_increment': 0,
            'shared_opposite_active_aux_cost': 0, 'wrong_independent_copy_cost': 128,
            'coherent_norm_squared_coefficient': str(coherent_eigenvalue**2),
            'orthogonal_one_step_Gram_coefficient': str(orthogonal_first_image_squared),
            'coherent_norm_not_ground_Gram': True, 'quantum_normal_gap_available': False}


def canonical_embedding() -> dict:
    rows = []
    for nside in (4, 6):
        volume = nside**3
        sites = 4
        counts = [math.comb(4*sites, n)*math.comb(4*(volume-sites), 2*volume-n)
                  for n in range(4*sites+1)]
        whole = math.comb(4*volume, 2*volume)
        assert sum(counts) == whole
        assert all(counts)
        rows.append({'N': nside, 'global_particles_each': 2*volume,
                     'all_local_n': list(range(4*sites+1)),
                     'local_exterior_multiplicities': [str(v) for v in counts],
                     'one_species_dimension': str(whole), 'all65_dimension_exact': f'{whole}^65',
                     'exterior_identity_not_exterior_vector': True})
    return {'species': {'neutral': 1, 'active': 1, 'auxiliary': 63},
            'Ref': 'FULL_UNCOMPRESSED_NATIVE_MODULE', 'canonical_direct_sum': rows,
            'physical_Gauss_sewing_not_tensor_factorization': True,
            'aux_irrep_selection': False, 'rhoN_replaced_by_trace': False}


def smooth_graph() -> dict:
    t = sp.symbols('t', real=True)
    h = 35*t**4 - 84*t**5 + 70*t**6 - 20*t**7
    for order in range(1, 4):
        assert sp.diff(h, t, order).subs(t, 0) == 0
        assert sp.diff(h, t, order).subs(t, 1) == 0
    assert h.subs(t, 0) == 0 and h.subs(t, 1) == 1
    assert sp.factor(sp.diff(h, t)) == -140*t**3*(t-1)**3
    # Coefficient absolute sums are conservative uniform derivative constants.
    c1 = sum(abs(v) for v in sp.Poly(sp.diff(h,t),t).all_coeffs())
    c2 = sum(abs(v) for v in sp.Poly(sp.diff(h,t,2),t).all_coeffs())
    z, chi = sp.symbols('z chi', positive=True)
    psi = sp.Function('psi')(z)
    f = sp.Function('f')(z)
    comm = sp.expand(-4*chi*sp.diff(f*psi,z,2)+4*chi*f*sp.diff(psi,z,2))
    expected = -4*chi*(sp.diff(f,z,2)*psi+2*sp.diff(f,z)*sp.diff(psi,z))
    assert sp.simplify(comm-expected) == 0
    assert comm != 0
    return {'transition_polynomial': str(h), 'C3_endpoint_jets': True,
            'h_derivative_uniform_coefficient_bounds': [str(c1),str(c2)],
            'alpha_derivative_bounds': [f'pi*{c1}/2', f'pi*{c2}/2'],
            'native_kinetic_commutator_exact': str(comm), 'sharp_projector_used': False,
            'whole_graph_form_bounds': 'proof independent_smooth_graph_construction',
            'uncapped_fourth_tail_UI_assumed': False}


def response_controls() -> dict:
    # A noncommuting exact block control of the FULL dressed-source identity,
    # not a fabricated physical native Hamiltonian/ground.
    a = sp.Matrix([[5,sp.I],[-sp.I,9]])
    d = sp.Matrix([[7,1],[1,8]])
    b = sp.Matrix([[1,sp.I],[0,1]])/3
    k = a.row_join(b.conjugate().T).col_join(b.row_join(d))
    assert k == k.conjugate().T
    assert all(k[:j,:j].det() > 0 for j in range(1,5))
    assert a*(b.conjugate().T*b) != (b.conjugate().T*b)*a
    f = a-b.conjugate().T*d.inv()*b
    sources = [sp.Matrix([1,sp.I,2,1-sp.I]),sp.Matrix([sp.I,2,1,3])]
    exacts=[]; wrongs=[]
    for u in sources:
        x,y=u[:2,:],u[2:,:]
        w=x-b.conjugate().T*d.inv()*y
        full=sp.simplify((u.conjugate().T*k.inv()*u)[0])
        transported=sp.simplify((y.conjugate().T*d.inv()*y+w.conjugate().T*f.inv()*w)[0])
        assert full == transported
        wrong=sp.simplify((x.conjugate().T*f.inv()*x)[0])
        assert wrong != full
        exacts.append(full);wrongs.append(wrong)
    weights=[sp.Rational(1,7),sp.Rational(6,7)]
    weighted=sum(w*v for w,v in zip(weights,exacts))
    trace=(exacts[0]+exacts[1])/2
    assert weighted != trace
    # Degenerate wholeP0 removal differs from single-vector and mean subtraction.
    p0=sp.diag(1,1,0,0);single=sp.diag(1,0,0,0)
    u=sp.Matrix([1,2,3,4])
    wholegram=((sp.eye(4)-p0)*u).norm()**2
    singlegram=((sp.eye(4)-single)*u).norm()**2
    assert wholegram==25 and singlegram==29
    # Remote exterior term belongs to PHP, even with no boundary B.
    exterior=sp.diag(2,3)
    assert (a+exterior).inv()!=a.inv()
    return {'scope':'NONNATIVE_EXACT_METHOD_CONTROL_NOT_GROUND_OR_PHYSICAL_RESPONSE',
            'noncommuting_full_dressed_source_identity':True,
            'full_responses_exact':[str(v) for v in exacts],
            'wrong_discard_Q_source_responses_exact':[str(v) for v in wrongs],
            'actual_style_unequal_weight_response_exact':str(weighted),
            'wrong_trace_response_exact':str(trace),
            'wholeP0_Gram_control':str(wholegram),'wrong_onevector_Gram_control':str(singlegram),
            'omitted_exterior_PHP_with_Bzero_changes_response':True}


def magnetic_supplier(profiles: list[dict]) -> dict:
    e=sp.symbols('b_dot_E',real=True)
    norm2=sp.Integer(4)
    plus=128*e+64*norm2;minus=-128*e+64*norm2
    assert sp.expand(plus+minus)==512
    second=sp.expand(plus**2+minus**2)
    assert second==32768*e**2+131072
    r,D=sp.symbols('r D',nonnegative=True)
    lower=(r*r-sp.Rational(1,2))**2-(D*D+1)/4
    assert sp.factor(r**4-D*r-lower)==(-D+2*r)**2/4
    neutral=profiles[0]
    dn=2*abs(Fraction(neutral['onsite']))+12*(abs(Fraction(neutral['paths'][0]['c0']))+abs(Fraction(neutral['paths'][0]['ci'])))
    dch=Fraction(855444290449274477,35184372088832)
    lam=2**8192
    fullD=dn+dch/lam
    smax=1024*lam*(fullD**2+1)+131072
    gmin=Fraction(512**2,smax)
    assert 0<gmin<2
    threshold=Fraction(1024,gmin)
    bound_s64=gmin**2/(512+64*gmin)
    # The normalized variational trial is optimized without a solver.
    G,M,s,q=sp.symbols('G M s q',positive=True)
    objective=2*q*G-q*q*(M+s*G)
    optimum=G/(M+s*G)
    assert sp.simplify(objective.subs(q,optimum)-G*G/(M+s*G))==0
    # Exact uniform-in-loop-length supplier has G bound independent ell but
    # energy cutoff growing linearly, rejecting a false IR interpretation.
    ell,K=sp.symbols('ell K',positive=True)
    loop_G=sp.simplify((128*ell)**2/(32768*ell**2*K+8192*ell**2))
    assert sp.diff(loop_G,ell)==0
    loop_cut=sp.simplify(256*ell/loop_G)
    assert sp.diff(loop_cut,ell)>0
    return {'scope':'UNIVERSAL_ANALYTIC_ACTUAL_GROUND_SUM_RULE_UNDER_INHERITED_REALIZATION',
            'loop_flow_norm_squared':4,'wholeP0_removed_signed_first_moment':512,
            'signed_second_moment_polynomial_exact':str(second),
            'quantum_Phi_Haar_neutral_charged_remainder':'COMMUTES_EXACTLY_WITH_INTEGER_CLOSED_LOOP; state dependence retained',
            'Dn_exact':str(dn),'Dch_exact':str(dch),'Lambda_exponent':8192,
            'Smax_exact':str(smax),'Gmin_exact':str(gmin),'R_s64_lower_exact':str(bound_s64),
            'finite_energy_spectral_cutoff_exact':str(threshold),
            'spectral_mass_below_cutoff_lower_exact':str(gmin/2),
            'normalized_variational_trial_exact':str(optimum),
            'large_loop_G_lower_formula':str(loop_G),'large_loop_energy_cutoff_formula':str(loop_cut),
            'IR_extension':'FAILED_THRESHOLD_GROWS_NOT_SHRINKS',
            'actual_rhoN_or_ground_matrix_evaluated':False,'finite_frequency_not_stiffness_or_atom':True}


def main() -> None:
    sys.set_int_max_str_digits(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text())
    for path,expected in request['pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    profiles=json.loads(Path(request['profiles']).read_text())['primitive_profiles']
    result={'kind':'NEW_NATIVE_PHYSICAL1_FULL_CANONICAL_TRANSPORT_AND_MAGNETIC_ANALYTICAL_CONSUMER',
            'native_gate':native_gate(profiles),'canonical_embedding':canonical_embedding(),
            'smooth_graph':smooth_graph(),'source_response_controls':response_controls(),
            'actual_ground_analytic_magnetic_supplier':magnetic_supplier(profiles),
            'proof_sha256':hashlib.sha256(Path(request['proof']).read_bytes()).hexdigest(),
            'old_entrypoints_imported_or_rerun':False,'native_H_or_ground_backend_invented':False,
            'actual_Nuniform_lowband_transport_approximation':'EXPLICITLY_REJECTED_UNPAID_LOCAL_RANGE_REMAINDER',
            'uniform_gapped_branch':False,'stiffness_order_IR_atom':False,
            'scientific_admission':False,'campaign_complete':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'native_strong_electric_gate':'failed','successor':'wholeP0_signed_loop_normalized_actual_ground_analytic_supplier',
                      'finite_positive_frequency_lower':'derived','sharp_IR_or_gap':False}))


if __name__=='__main__':
    main()
