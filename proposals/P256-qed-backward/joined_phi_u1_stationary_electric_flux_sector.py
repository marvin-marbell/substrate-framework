"""Actual HE64+HC64 electric sectors, exact native controls and sector-aware bounds.

This is new finite scientific arithmetic, not a root or a ground eigensolver.
Operator identities apply to full Phi/ordered SU4; the quantitative variational
bracket is explicitly a frozen uniform-Phi, U=I BODY statement.  No photon pole,
prepared expectation, global beta minimizer or whole-field ground is inferred.
The receipt is the documentation of the derivation and of its failed gap price.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
from typing import Any, cast
import zipfile

import numpy as np
import sympy as sy
from threadpoolctl import threadpool_limits

import induced_phi_nonuniform_field_car as field
import joined_phi_u1_stationary_full_phi_fiber as established

ROOT = Path(__file__).resolve().parent
PREFIX = 'joined_phi_u1_stationary_electric_flux_sector'
sys.set_int_max_str_digits(0)
PINS = {
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_root_actual_receipt.json': '4691ffaa9cd81902942cbc228fc47fafff14518be6b5df2e84d08c29c7da4756',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json': '5b97862fc7fca21b8797d549d7d4dfdde58b101e8c46c7a7cfb6531a669c4354',
    'induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json': 'da1998065d6622c4ac780409c29ae10711e01b5c5143edf620557fe15234e9e6',
}
FULLPHI_PIN = '945ad7d3b2143ed077af074511ec641b9ce878109db6c7e2997d9790e86029c6'

PROOF = {
    'allrho_and_orientation': 'D has +1 at native source x and -1 at target y. The forward term is c_x^dag T_xy c_y exp(i k.Theta), k_e=1 on the ordered native path; its CAR charge increment is delta_x-delta_y=Dk. rho_x=sum over active and EACH of63 auxiliary copies (n_s,x-2)+(NRef-4)delta_x,anchor. Neutral is U1 neutral. Active128/Ref4 and active129/Ref3 both have totalrho0; baseline is128 at EVERY site, not2 for all species together. Local active^dag Ref transfer at anchor changes the two charges oppositely and has inc0. Omitting Ref in the source gives totalrho1 and is NOT a valid longitudinal Gauss solution.',
    'full_electric_identity': 'Let Q be the native orthonormal192x129 basis of kerD, L=DD^T, P_L=D^T Lplus D, P_T=QQ^T=I-P_L. On the common smooth CAR/rotor/fullPhi/fullSU4 core, E=Q Pi+D^T Lplus rho. Its192 components commute strongly: Pi commute, site occupation/Ref charges commute, and these tensor factors commute. DE=rho, P_T E=QPi and P_L E=D^T Lplus rho. Orthogonality gives sum E_e^2=Pi^2+rho Lplus rho since Lplus L Lplus=Lplus on the meanzero charges. Thus literal HE64+HC64=64sum E_e^2, an OPERATOR identity, not an expectation. Equality extends by closure of the positive forms.',
    'native_commutator_and_conservation': '[Pi,exp(i k.Theta)]=Q^T k exp(i k.Theta), [rho,c_x^dag T c_y]=Dk c_x^dag T c_y. Hence [E_e,hop_k]=(P_T k+P_L k)_e hop_k=k_e hop_k, including ordered TWO links and AP sign. FullPhi and SU4 multiplication coefficients commute with E; their kinetic operators do also. Neutral hops and onsite/Ref terms have inc0. Every native k is INTEGER, so exp(2pi i E_e) commutes with each hop and with the complete Hamiltonian. These are192 commuting conserved unitaries, central in the native integer-character hop algebra (subject to divergence/period constraints), not automatically central for every gauge-invariant dressed observable. A Theta-independent reduced CAR transfer changes E by D^T Lplus delta and need not commute with these phases. This is not a claim that unbounded E_e commute with hopping or that exp(2pi i bare QPi) is conserved.',
    'saturated_cycles_and_fibers': 'A spanning tree produces C192x129 with DC=0 and identity chord minor; integer elimination proves kerD intersect Z192=C Z129. Write G=C^TC and A=C^TQ. In the actual beta fiber Pi=A^-1(m+beta), m in Z129; QPi=C G^-1(m+beta). For EACH meanzero integer rho choose the deterministic integer treeflow kT with DkT=rho. FullE=kT+C G^-1(m+beta-C^TkT). The conserved alpha is the last transverse vector modulo C Z129. A native hop k changes (m,rho) to(m+C^Tk,rho+Dk), and alpha changes by the integer cycle k-kT(rho+Dk)+kT(rho). Incoming alpha is therefore retained, not set0.',
    'finite_cosets_not_one_seed': 'For fixed beta,rho the distinct alpha classes are the affine quotient Z129/G Z129, translated by beta-C^TkT. There are detG=number of spanning trees of the ACTUAL64-site graph. Native-flow seed alpha0 at beta0 has E=kT+Cz, all integer edge fluxes; its orbit under ALL native paths cannot reach a different exp(2pi i E) character. The full beta0 fiber also contains m=unit chord,rho0, E=P_T(unit chord edge), which is noninteger. Thus a native integer-flow cyclic seed DOES NOT span the full beta fiber. This is a kinematic separation, not an assertion that any chosen seed spans even its own alpha sector or that alpha0 minimizes.',
    'full_next_representation': 'Use basis |m,F_active,F_aux1,...,F_aux63,F_Ref> in every beta, including all affine finite cosets, and compute rho from ALL occupations. Store E=kT(rho)+alpha+Cz, not a reset electric vacuum after each hop. A native ordered path shifts E by its integer inc and applies the literal CAR matrix/sign and fullPhi/SU4 coefficient. This exact electric-flow/Fock graph retains all same-site spin, one/two/AP, incoming offsets, Ref and cross-copy covariance. Tensor the fullPhi/Haar form rather than treating a fixed-body test as its ground. The tested matching collective representation below is a genuinely different finite variational compression of this graph, not a claim of complete-sector or tail coverage.',
    'sector_operator_lower': 'On an exp(2pi i E)=exp(2pi i alpha) sector, the joint spectral values are E=alpha+k with k integer192 and the required CAR divergence. Therefore sum E^2>=sum dist(alpha_e,Z)^2 without choosing alpha0. At frozen uniform Clifford Phi and U=I, the onsite lower is Eref=-128sqrtg0(onsiteActive+63onsiteAux) in the background (source floor adds onsiteActive sqrtg0). A single native path T=c0 sqrtg0 Gamma0+i ci sqrtgs Gamma_axis has singular values |c0sqrtg0+ci sqrtgs| and |c0sqrtg0-ci sqrtgs|, each twice: nuclear norm=4max(|c0|sqrtg0,|ci|sqrtgs). The two-site CAR operator hop+hopdag is bounded below by minus this nuclear norm on ANY occupations. Summing all64 translates of EVERY one/two path and all63 copies yields Hbody>=Eref+64sum dist(alpha_e,Z)^2-Khop. This is an operator lower, no perturbative electric denominator or fake incoming-independent gap. FullPhi analogue is H>=Hbos+Mneutral+64dist^2-sum_s N_s|onsite_s|max_x||Phi_x,0||-sum_s,path load_s||T_xy(Phi,U)||_1; it retains the actual field-dependent price, not the frozen numeric Khop.',
    'matching_variational_representation': 'Choose the32 disjoint axis0 edges with source coordinate0 or2 (no AP seam). In every species on every selected bond, retain two disjoint CAR channels negativeGamma0 at y -> positiveGamma0 at x via Gamma_axis. Each channel is an even two-mode bilinear and distinct channels commute. The rest of the negative modes remain occupied; each species still has128 particles. Its normalized local spinor is (|0>-t_s phased|1>)/sqrt(1+t_s^2). Multiply by the incoming Fourier character. Occupation transfer n_b gives FullE=E_in+sum_b n_b inc_b. Active contributes2 channels; ALL63 auxiliary copies contribute126, not2. The exact symmetric occupation graph per bond has(2+1)(126+1)=381 states. Its offdiagonal is -|ci_s|sqrtgs sqrt((N_s-n_s)(n_s+1)), diagonal is2sum_s M_s n_s+64(|E_in|^2+2E_in.inc n+|inc|^2 n^2). In the32-bond compression allocate the incoming constant globally ONCE. Nonselected hops have zero compressed matrix elements: they change particle count in a retained CAR pair; c0 within the selected bond maps a negative/positive channel into its filled/empty spectator, not its paired channel; two-link endpoints are never a selected nearest-neighbour pair. They remain in the global lower and in the full graph, not discarded from the law.',
    'coherent_trial_exact_energy': 'For p_s=t_s^2/(1+t_s^2), mu=sum_s N_s p_s and var=sum_s N_s p_s(1-p_s), a selected bond contributes 64(2 E_in.inc mu+|inc|^2(mu^2+var))+2sum_s N_s M_s p_s-2sum_s N_s |ci_s|sqrtgs t_s/(1+t_s^2). Its normalization is exactly1. Disjoint matching edge incidences have zero mutual inner products, so add32 such contributions to Eref+64|E_in|^2. HC and HE separately can have interbond correlations; THEIR SUM cancels them by the exact fullE identity, not by independent-copy assumption. Within one bond the all63 covariance mu^2 is indispensable. Every incoming transverse offset in any actual beta/rho0 sector can be used; no minimizer is selected. This upper concerns the complete fixed-body ground IN THAT conserved alpha sector because its trial is normalized in that sector; not a global fullPhi trial or gap.',
    'price_boundary': 'The sector-aware lower and normalized matching upper form a same-law fixed-body bracket. The wide nuclear-norm price cannot certify a narrow vacuum window or any interacting gap. The receipt quantifies that failure, and tests the128-channel/64-charged-species381-state next representation and adverse incoming-offset controls. No alpha0 minimum, all-seed spanning, prepared-pole or whole-ground inference is made.',
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dot(a: list[F], b: list[F]) -> F:
    return sum((x*y for x,y in zip(a,b)),F(0))


def basis(n: int, index: int) -> list[F]:
    out=[F(0)]*n;out[index]=F(1);return out


class ElectricGraph:
    def __init__(self, d: np.ndarray, sites: np.ndarray):
        self.d=d
        self.c,self.chords=established.cycles(d)
        self.green=established.rational_green(sites)
        self.ends=[tuple(map(int,np.flatnonzero(d[:,e]))) for e in range(192)]
        self.adj: dict[int,list[tuple[int,int]]]=defaultdict(list)
        for e,(x,y) in enumerate(self.ends):
            self.adj[x].append((y,e));self.adj[y].append((x,e))
        self.parent={0:(-1,-1)};self.order=[0]
        for x in self.order:
            for y,e in self.adj[x]:
                if y not in self.parent:self.parent[y]=(x,e);self.order.append(y)
        self.tree=[self.parent[x][1] for x in self.order[1:]]

    def div(self, k: list[F]) -> list[F]:
        out=[F(0)]*64
        for e,(x,y) in enumerate(self.ends):
            out[x]+=int(self.d[x,e])*k[e];out[y]+=int(self.d[y,e])*k[e]
        return out

    def long(self, rho: list[F], neutral: bool=True) -> list[F]:
        if neutral and sum(rho,F(0))!=0:raise ValueError('Full Gauss charge is not meanzero; retain Ref')
        support=[x for x,v in enumerate(rho) if v]
        potential=[sum((self.green[x][y]*rho[y] for y in support),F(0)) for x in range(64)]
        return [sum((int(self.d[x,e])*potential[x] for x in ends),F(0)) for e,ends in enumerate(self.ends)]

    def transverse(self, k: list[F]) -> list[F]:
        ell=self.long(self.div(k));return [a-b for a,b in zip(k,ell)]

    def treeflow(self, rho: list[F]) -> list[F]:
        if sum(rho,F(0)) or any(v.denominator!=1 for v in rho):raise ValueError('Integer neutral allrho required')
        b=rho.copy();out=[F(0)]*192
        for x in reversed(self.order[1:]):
            y,e=self.parent[x];out[e]=b[x]*int(self.d[x,e]);b[y]-=int(self.d[y,e])*out[e];b[x]=F(0)
        assert b[0]==0 and self.div(out)==rho
        return out

    def full(self, m: list[F], rho: list[F]) -> list[F]:
        # Chord injection J obeys C^T J=I, hence P_T J=C G^-1.
        k=[F(0)]*192
        for e,value in zip(self.chords,m):k[e]=value
        pt=self.transverse(k);pl=self.long(rho)
        return [a+b for a,b in zip(pt,pl)]

    def character(self, k: list[F]) -> list[F]:
        return [sum((int(self.c[e,j])*k[e] for e in np.flatnonzero(self.c[:,j])),F(0)) for j in range(129)]

    def energy(self, m: list[F], rho: list[F]) -> dict[str,str]:
        E=self.full(m,rho);ell=self.long(rho);pt=[a-b for a,b in zip(E,ell)]
        he=64*dot(pt,pt);hc=64*dot(ell,ell);total=64*dot(E,E)
        assert he+hc==total
        return dict(HE=str(he),HC=str(hc),total=str(total))


def exact_graph_controls(g: ElectricGraph, data: dict[str,Any]) -> dict[str,Any]:
    L=g.d@g.d.T
    for x in range(64):
        assert sum(g.green[x],F(0))==0
        for y in range(64):
            assert sum((int(L[x,z])*g.green[z][y] for z in np.flatnonzero(L[x])),F(0))==F(int(x==y))-F(1,64)
    rows=[];incidences={}
    for profile in data['profiles'][1:]:
        for path in field.native_paths(profile,data['native']):
            k=[F(0)]*192
            for e in path['edges']:k[e]+=1
            rho=g.div(k);expected=basis(64,int(path['source']));expected[int(path['target'])]-=1
            assert rho==expected
            ell=g.long(rho);pt=g.transverse(k)
            assert [a+b for a,b in zip(pt,ell)]==k and dot(pt,ell)==0
            m=g.character(k);before=g.full([F(0)]*129,[F(0)]*64);after=g.full(m,rho)
            assert after==k and before==[F(0)]*192
            # Tree representative changes only by an integer saturated cycle.
            shift=[a-b for a,b in zip(k,g.treeflow(rho))]
            assert g.div(shift)==[F(0)]*64
            z=[shift[e] for e in g.chords]
            assert [sum((int(g.c[e,j])*z[j] for j in np.flatnonzero(g.c[e])),F(0)) for e in range(192)]==shift
            key=(int(path['source']),int(path['axis']),int(path['length']))
            incidences[key]=k
            rows.append(dict(role=profile['role'],source=int(path['source']),target=int(path['target']),axis=int(path['axis']),length=int(path['length']),seam=int(path['seam']),ordered_edges=list(map(int,path['edges'])),commutator_increment_exact=True,constantTheta=g.energy([F(0)]*129,rho),Wilson_dressed=g.energy(m,rho)))
    k1=incidences[(0,0,1)];k2=incidences[(0,0,2)];rho1=g.div(k1);rho2=g.div(k2)
    m1=g.character(k1);m2=g.character(k2)
    constant1=g.energy([F(0)]*129,rho1);wilson1=g.energy(m1,rho1)
    assert constant1==dict(HE='0',HC='21',total='21') and wilson1['total']=='64'
    constant2=g.energy([F(0)]*129,rho2);wilson2=g.energy(m2,rho2)
    assert wilson2['total']=='128'
    # Background has2 charges per site in each of64 species. Source at anchor
    # has active129 and Ref3, so local charge cancels EXACTLY.
    background=[F(64*2-128)]*64
    source=background.copy();source[0]+=1;source[0]-=1
    assert source==background==[F(0)]*64
    wrong=background.copy();wrong[0]+=1
    try:g.long(wrong)
    except ValueError:wrong_rejected=True
    else:raise AssertionError('Missing Ref accepted as neutral')
    # Move the source from anchor0 to target16. Ref hole remains at0.
    target=int(data['native']['Pprofiles'][0]['paths'][0]['target'])
    ref_rho=basis(64,target);ref_rho[0]-=1
    wrong_ref=basis(64,target)
    ref_correct=64*dot(g.long(ref_rho),g.long(ref_rho))
    wrong_projected=64*dot(g.long(wrong_ref,neutral=False),g.long(wrong_ref,neutral=False))
    assert ref_correct==21 and sum(wrong_ref,F(0))==1
    incoming=g.transverse(k1)
    incoming_after=[a+b for a,b in zip(incoming,k1)]
    incoming_delta=64*(dot(incoming_after,incoming_after)-dot(incoming,incoming))
    assert incoming_delta==64*(2*dot(incoming,k1)+dot(k1,k1)) and incoming_delta!=64
    cov=[]
    for copies in (1,2,63,64):
        flux=[copies*x for x in k1];rho=g.div(flux)
        row=g.energy(g.character(flux),rho)
        assert F(row['total'])==64*copies**2
        cov.append(dict(copies=copies,collective=row,wrong_independent_total=str(64*copies),missing_cross_total=str(64*copies*(copies-1)),scope='Normalized simultaneous onePH BODY Fock/Fourier basis vector, not a Slater expectation'))
    return dict(scope='EXACT rational normalized BODY Fourier/Fock basis energies; full-field statement is the separately derived operator identity',all_native_charged_ordered_path_rows=rows,path_count=len(rows),onePH_constantTheta=constant1,onePH_Wilson=wilson1,two_link_constantTheta=constant2,two_link_Wilson=wilson2,background_rho=background,source_active129_Ref3_rho=source,Ref_wrong=dict(nonzero_total_charge_rejected=wrong_rejected,source_at_target_correct_HC=str(ref_correct),wrong_Ref_omitted_Lplus_projection=str(wrong_projected),wrong_total_charge='1',explanation='Lplus silently projects an inadmissible nonneutral source; its finite number is NOT HC of the source law'),incoming_control=dict(E_in_squared=str(dot(incoming,incoming)),E_in_dot_inc=str(dot(incoming,k1)),correct_hop_delta=str(incoming_delta),wrong_reset_delta='64',nonzero_cross_retained=True),all63_covariance=cov)


def coset_controls(g: ElectricGraph) -> dict[str,Any]:
    zeros=[F(0)]*64;unit=basis(129,0);E=g.full(unit,zeros)
    assert any(v.denominator!=1 for v in E)
    assert g.character(E)==unit and g.div(E)==zeros
    residue=[v%1 for v in E]
    orbit=[]
    for multiple in (0,1,2):
        vector=[multiple*v for v in E];orbit.append(dict(multiple=multiple,energy=str(64*dot(vector,vector)),residues=[str(v%1) for v in vector]))
    assert len({tuple(r['residues']) for r in orbit})==3
    # det(I+B^TB)=det(I+BB^T), avoiding an unnecessary129-dimensional inverse.
    B=sy.Matrix(g.c[g.tree,:].tolist())
    gram_det=int((sy.eye(63)+B*B.T).det(method='domain-ge'))
    cofactor=int(cast(sy.Integer,sy.Matrix((g.d@g.d.T)[1:,1:].tolist()).det(method='domain-ge')))
    cos=(1,0,-1,0);spectrum=Counter(sum(2-2*cos[j] for j in q) for q in product(range(4),repeat=3))
    matrix_tree=F(1,64)
    for lam,multiplicity in spectrum.items():
        if lam:matrix_tree*=lam**multiplicity
    assert matrix_tree.denominator==1 and gram_det==cofactor==matrix_tree.numerator
    # A constantTheta onePH occupies a non-native flux sector even at beta0.
    inc=basis(192,0);rho=g.div(inc);tree=g.treeflow(rho)
    constantE=g.full([F(0)]*129,rho);alpha=[a-b for a,b in zip(constantE,tree)]
    assert any(v.denominator!=1 for v in alpha)
    return dict(saturated_C=g.c.tolist(),chords=g.chords,tree_edges=g.tree,rank=129,chord_identity=True,full_beta0_native_seed_not_spanning=True,witness_m=list(map(str,unit)),witness_E=list(map(str,E)),witness_residue_mod1=list(map(str,residue)),distinct_beta0_cosets=orbit,constantTheta_onePH_alpha=list(map(str,alpha)),finite_coset_index=str(gram_det),matrix_tree_cofactor=str(cofactor),laplacian_eigenvalue_multiplicities={str(k):v for k,v in sorted(spectrum.items())},index_factorization={str(k):int(v) for k,v in sy.factorint(gram_det).items()},index_three_exact_independent_routes=True,scope='All fixed-beta affine classes counted exactly; no numerical beta or alpha ground ordering')


def collective_car_controls(profiles: list[dict[str,Any]], t: list[F]) -> dict[str,Any]:
    """Construct the exact local CAR and ALL-copy symmetric occupation graph."""
    # A simultaneous Clifford conjugation diagonalizes Gamma0 and makes its
    # Gamma_axis intertwiner the identity; this is not a changed primitive.
    gamma0=sy.diag(-1,-1,1,1)
    gamma1=sy.Matrix([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
    assert gamma0*gamma1+gamma1*gamma0==sy.zeros(4)
    labels=list(product(range(2),repeat=2))
    masks=[3+(1<<(2 if q0 else 4))+(1<<(3 if q1 else 5)) for q0,q1 in labels]
    lookup={mask:i for i,mask in enumerate(masks)}
    def car(mask: int, a: int, b: int) -> tuple[int,int]|None:
        if not mask>>b&1:return None
        sign=(-1)**((mask&((1<<b)-1)).bit_count());new=mask^(1<<b)
        if new>>a&1:return None
        sign*=(-1)**((new&((1<<a)-1)).bit_count())
        return new|(1<<a),sign
    c0=sy.zeros(4);forward=sy.zeros(4)
    for j,mask in enumerate(masks):
        for x,y in product(range(4),repeat=2):
            action=car(mask,x,4+y)
            if action and action[0] in lookup:
                i=lookup[action[0]]
                c0[i,j]+=action[1]*gamma0[x,y]
                forward[i,j]+=action[1]*sy.I*gamma1[x,y]
            action=car(mask,4+y,x)
            if action and action[0] in lookup:
                c0[lookup[action[0]],j]+=action[1]*gamma0[x,y]
    number=sy.diag(*[sum(q) for q in labels]);ci=forward+forward.conjugate().T
    assert c0==sy.zeros(4)
    assert number*forward-forward*number==forward
    assert sorted(ci.eigenvals().items())==[(-2,1),(0,2),(2,1)]
    Ns=[2*p['load'] for p in profiles]
    moments=[];edges=[];states=list(product(range(Ns[0]+1),range(Ns[1]+1)))
    for species,(N,v,profile) in enumerate(zip(Ns,t,profiles)):
        weights=[F(int(sy.binomial(N,n)))*v**(2*n)/(1+v*v)**N for n in range(N+1)]
        norm=sum(weights,F(0));mu=sum((n*w for n,w in enumerate(weights)),F(0))
        second=sum((n*n*w for n,w in enumerate(weights)),F(0))
        hop=sum((F(int(sy.binomial(N,n)))*(N-n)*v**(2*n+1)/(1+v*v)**N for n in range(N)),F(0))
        p=v*v/(1+v*v)
        assert norm==1 and mu==N*p and second-mu*mu==N*p*(1-p)
        assert hop==N*v/(1+v*v)
        ci_exact=F(next(path for path in profile['paths'] if path['axis']==0 and path['length']==1)['ci'])
        moments.append(dict(role=profile['role'],channels=N,binomial_norm_exact=str(norm),mean_transfer_exact=str(mu),second_transfer_exact=str(second),nearest_neighbour_spin_expectation_coefficient_exact=str(hop)))
        for a,b in states:
            n=(a,b)[species]
            if n<N:
                target=(a+1,b) if species==0 else (a,b+1)
                edges.append(dict(source=[a,b],target=list(target),species=species,offdiagonal_squared_coefficient_times_gs=str(ci_exact**2*(N-n)*(n+1)),offdiagonal_sign_after_channel_phase='negative'))
    assert len(states)==381 and len(edges)==632
    incoming=sy.Symbol('incoming_dot_inc',real=True)
    diagonal=[dict(state=[a,b],total_transfer=a+b,electric_change_exact=str(64*((a+b)**2+2*incoming*(a+b))),onsite_change_coefficients=[str(2*a*F(profiles[0]['onsite'])),str(2*b*F(profiles[1]['onsite']))]) for a,b in states]
    return dict(scope='Exact Clifford-conjugated ACTUAL one-bond CAR compression and ALL63-copy381-state graph; not full spatial/field ground diagonalization',local_CAR_masks=masks,local_c0_compression_exact_zero=True,local_ci_matrix=[[str(v) for v in row] for row in ci.tolist()],local_number_commutator_increment_exact=True,local_ci_spectrum_exact=['-2','0','0','2'],collective_state_count=len(states),collective_undirected_edges=len(edges),collective_edges=edges,collective_diagonal=diagonal,normalized_binomial_moment_controls=moments,incoming_symbol_retained=True,two_link_retained_in_operator_lower_and_full_graph=True)


def bracket(g: ElectricGraph, data: dict[str,Any], mean: dict[str,Any]) -> dict[str,Any]:
    iv=established.gauss.iv;iv.dps=110
    ex=established.gauss.exact;pack=established.gauss.bounds
    root0=iv.sqrt(field.full.enclosure(F(mean['parameters']['g0']['lower']),F(mean['parameters']['g0']['upper'])))
    roots=iv.sqrt(field.full.enclosure(F(mean['parameters']['gs']['lower']),F(mean['parameters']['gs']['upper'])))
    profiles=data['profiles'][1:]
    eref=-128*root0*ex(sum((F(p['onsite'])*p['load'] for p in profiles),F(0)))
    lower_price=ex(0);prices=[]
    for p in profiles:
        total=ex(0);terms=[]
        for path in p['paths']:
            a=ex(abs(F(path['c0'])))*root0;b=ex(abs(F(path['ci'])))*roots
            # Safe max interval: exact max of outward rational endpoints.
            lo=max(established.gauss.endpoint(a,0),established.gauss.endpoint(b,0))
            hi=max(established.gauss.endpoint(a,1),established.gauss.endpoint(b,1))
            nuclear=field.full.enclosure(4*lo,4*hi)
            total+=64*nuclear
            terms.append(dict(axis=path['axis'],length=path['length'],c0_exact=str(path['c0']),ci_exact=str(path['ci']),nuclear_per_native_path=pack(nuclear),native_translates=64))
        lower_price+=p['load']*total
        prices.append(dict(role=p['role'],load=p['load'],all_one_two_terms=terms,all_copies_hopping_lower_price=pack(p['load']*total)))
    matching=[]
    for path in field.native_paths(profiles[0],data['native']):
        if path['axis']==0 and path['length']==1 and int(data['native']['sites'][path['source'],0]) in (0,2):
            assert path['seam']==1
            matching.append(int(path['edges'][0]))
    assert len(matching)==32 and len({x for e in matching for x in g.ends[e]})==64
    N=[2*p['load'] for p in profiles];t=[F(1,24),F(1,24)]
    p=[v*v/(1+v*v) for v in t]
    mu=sum((n*x for n,x in zip(N,p)),F(0));var=sum((n*x*(1-x) for n,x in zip(N,p)),F(0))
    ci=[abs(F(next(v for v in prof['paths'] if v['axis']==0 and v['length']==1)['ci'])) for prof in profiles]
    onsite=2*root0*ex(sum((n*F(prof['onsite'])*x for n,prof,x in zip(N,profiles,p)),F(0)))
    hop=-2*roots*ex(sum((n*b*v/(1+v*v) for n,b,v in zip(N,ci,t)),F(0)))
    electric=64*ex(mu*mu+var)
    shift=32*(electric+onsite+hop)
    assert established.gauss.endpoint(shift,1)<0
    independent=64*sum((n*x for n,x in zip(N,p)),F(0))
    missing_cross=64*(mu*mu+var)-independent
    assert missing_cross>0
    def incoming_case(label: str, ein: list[F]) -> dict[str,Any]:
        assert g.div(ein)==[F(0)]*64
        squared=dot(ein,ein);cross=sum((ein[e] for e in matching),F(0))
        delta=shift+ex(128*mu*cross)
        residues=[v%1 for v in ein]
        distance=sum((min(v,1-v)**2 for v in residues),F(0))
        lower=eref+ex(64*distance)-lower_price
        upper=eref+ex(64*squared)+delta
        return dict(label=label,E_in_squared=str(squared),E_in_matching_sum=str(cross),sector_distance_lower_squared=str(distance),incoming_seed_energy=pack(eref+ex(64*squared)),matching_trial_shift_from_incoming_seed=pack(delta),fixed_body_sector_ground_lower=pack(lower),fixed_body_sector_ground_upper=pack(upper),bracket_width_upper=pack(upper-lower),incoming_E=list(map(str,ein)),scope='Fixed uniformPhi/U=I background128+8064/Ref4, actual conserved alpha sector; not fullPhi ground or gap')
    zero=[F(0)]*192
    witness=g.full(basis(129,0),[F(0)]*64)
    # Closed saturated cycle yields a large adverse offset in the SAME alpha0
    # sector. An offset-sensitive trial must not silently erase its cross term.
    adverse=[F(8 if e%3==0 else 0) for e in range(192)]
    cases=[incoming_case('zero_alpha0_seed_NO_minimum_assumption',zero),incoming_case('noninteger_fullbeta0_coset',witness),incoming_case('integer_cycle_offset_same_alpha0',adverse)]
    assert F(cases[2]['E_in_matching_sum'])==256
    # Exact positive local covariance formula with ALL126 auxiliary channels.
    cov=dict(active_aux_cross= str(128*(N[0]*p[0])*(N[1]*p[1])),aux_aux_cross=str(64*N[1]*(N[1]-1)*p[1]**2),active_active_cross=str(64*N[0]*(N[0]-1)*p[0]**2))
    assert sum((F(v) for v in cov.values()),F(0))==missing_cross
    collective=collective_car_controls(profiles,t)
    cases[0]['actual_collective_CAR_representation']=collective
    return dict(scope='Rigorous fixed-Phi BODY lower/variational bracket at frozen exact mean enclosures; implicit mean supplier remains conditional, no mean recomputation',reference_energy=pack(eref),global_operator_lower_relative_to_Eref=pack(-lower_price),lower_price_all_actual_one_two_paths=prices,matching_edges=matching,matching_bonds=32,active_channels=N[0],ALL63_aux_channels=N[1],symmetric_collective_dimension_per_bond=(N[0]+1)*(N[1]+1),exact_rational_t=list(map(str,t)),exact_trial_normalization='1',mu_per_bond=str(mu),variance_per_bond=str(var),onsite_per_bond=pack(onsite),hopping_per_bond=pack(hop),fullHEHC_per_bond=pack(electric),matching_upper_shift=pack(shift),wrong_independent_copy_electric_per_bond=str(independent),missing_cross_electric_per_bond=str(missing_cross),all63_covariance_breakdown=cov,incoming_cases=cases,price_failed_to_certify_gap=True,failure_quantification=dict(lower_price_over_variational_gain_upper=pack(lower_price/(-shift)),explanation='The lower is far below the normalized upper. Neither bracket width nor finite381-state compression supplies the omitted-sector/tail control needed for a vacuum or gap. Offset cross terms, including same-alpha integer-cycle offsets, materially change the trial energy.'),next_representation='381-state all128-channel collective bond blocks with exact64|E_in+n inc|^2, tensor matching blocks and couple every native one/two/AP CAR transition in the full affine-coset graph. All129 reciprocal directions/all finite cosets and fullPhi tails remain required, not asserted solved.')


def run(output: Path) -> dict[str,Any]:
    started=time.monotonic();captured={};records={}
    for name,pin in PINS.items():
        raw=(ROOT/name).read_bytes()
        if sha(raw)!=pin:raise ArithmeticError('Pinned complete input changed: '+name)
        captured[name]=raw;records[name]=json.loads(raw)
    fullphi_raw=Path(established.__file__).read_bytes()
    if sha(fullphi_raw)!=FULLPHI_PIN:raise ArithmeticError('Corrected fullPhi helper source changed')
    mean=records['induced_phi_nonuniform_fermionic_source_unitary_resumed_actual_receipt.json']['input_descriptor']['exact_NEW_allq_BO_mean']
    centres=[F(v) for v in mean['centre_exact_rational']]
    with threadpool_limits(limits=1):
        data=field.load_mean(g_override=[float(centres[0]),float(centres[1]),float(centres[1]),float(centres[1])],mean_descriptor=mean)
        d=np.rint(data['native']['D']).astype(np.int64)
        assert np.array_equal(d,data['native']['D'])
        g=ElectricGraph(d,data['native']['sites'])
        controls=exact_graph_controls(g,data)
        cosets=coset_controls(g)
        price=bracket(g,data,mean)
    sources={}
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).parent==ROOT and Path(file).suffix=='.py':sources[Path(file).name]=Path(file).read_bytes()
    for table in (field.CAPTURED,established.preparation.SOURCE_BYTES,established.gauss.SOURCE_BYTES):
        for name,raw in table.items():
            if name in sources and sources[name]!=raw:raise ArithmeticError('Source epoch mismatch: '+name)
            sources[name]=raw
    # Capture embedded supplier epochs separately; never call current code their
    # producer. Complete original receipts above are also frozen verbatim.
    for inputname,record in records.items():
        for name,value in record.get('source_at_import',{}).items():
            if isinstance(value,dict) and 'bytes_utf8' in value:
                raw=value['bytes_utf8'].encode()
                assert sha(raw)==value['sha256']
                sources['archived_supplier/'+Path(inputname).stem+'/'+name]=raw
    result=dict(kind='ACTUAL_HE64_HC64_FULL_ELECTRIC_SECTORS_AND_OFFSET_RETAINING_BODY_BRACKET',law=dict(N=4,V=64,a='1/64',HE=64,HC=64,all_charged_flavours=64,auxiliary_copies=63,background_active=128,source_active=129,Ref_background=4,Ref_source=3,baseline_per_site=128,no_bare_EM_B2=True),proof=PROOF,exact_graph_controls=controls,actual_fiber_cosets=cosets,fixedPhi_operator_variational_bracket=price,claim_boundaries=dict(operator='FullPhi/fullSU4 full electric identity, integer Wilson/CAR commutator and conserved phases.',exact='Rational native graph/Green/treeflow/basis-energy/coset/index controls, not normalized full-field expectation values.',conditional='Frozen exact uniform mean enclosures only for fixedPhi numeric bracket; no solver/root rerun.',finite='BODY matching all64-flavour coherent variational state and global operator lower, not prepared-current or pole.',open='All-sector stationary ground characterization, beta/alpha minimizers, fullPhi/colour tails, same-ground photon pole and QED closure.'),input_sha256={k:sha(v) for k,v in captured.items()},source_sha256={k:sha(v) for k,v in sources.items()},runtime=dict(executable=sys.executable,python=sys.version,numpy=np.__version__,sympy=sy.__version__,interval_dps=110),elapsed_seconds=time.monotonic()-started)
    assert all((ROOT/name).read_bytes()==raw for name,raw in captured.items())
    assert all((ROOT/name).read_bytes()==raw for name,raw in sources.items() if not name.startswith('archived_supplier/'))
    result['source_inputs_unchanged']=True
    archive=output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in {**sources,**captured}.items():z.writestr(name,raw)
    result['immutable_source_input_capture']=dict(path=archive.name,sha256=sha(archive.read_bytes()),source_members=len(sources),complete_input_members=len(captured))
    output.write_text(json.dumps(result,indent=2,default=str)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();receipt=run(args.output)
    print(json.dumps(dict(receipt=args.output.name,receipt_sha256=sha(args.output.read_bytes()),archive=receipt['immutable_source_input_capture'],coset_index=receipt['actual_fiber_cosets']['finite_coset_index'],native_seed_spans_full_fiber=False,onePH_constantTheta=receipt['exact_graph_controls']['onePH_constantTheta'],onePH_Wilson=receipt['exact_graph_controls']['onePH_Wilson'],matching_upper_shift=receipt['fixedPhi_operator_variational_bracket']['matching_upper_shift'],price_certifies_ground_gap=False,elapsed_seconds=receipt['elapsed_seconds']),indent=2))
