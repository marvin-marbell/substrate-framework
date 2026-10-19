"""Independent rational review controls; never imports or executes old producers."""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import ast

ROOT = Path(__file__).resolve().parent
NAMES = [
    'physical_singlet_circuit_source_20261011.py',
    'physical_singlet_circuit_epoch4_proof_20261011.json',
    'physical_singlet_circuit_epoch4_receipt_20261011.json',
    'physical_singlet_circuit_delivery_20261011.json',
    'joined_phi_u1_uniform_sharp_gauss_proof_20261010.json',
    'rotation_band.py', 'local_orbital_stability.py',
    'joined_phi_u1_bulk_flux_ring_consumer.py',
    'induced_phi_uniform_selfconsistent_full_normal_quantum_final_actual_receipt.json',
]
hashes = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in NAMES}
assert hashes[NAMES[0]] == 'dea3025ae8509cd6669ac26df9fcb1a5d7b416aee0642778d86f2be7cb8aca03'
assert hashes[NAMES[1]] == '56d7a2fbd1bac380ff714db20e66f68a5fb2e07788dcf2701a13741e1898c038'
# Exact SU(4) covariance: sum_A (T_A)ij(T_A)kl.
def k(i, j, l, m):
    return Q(int(i == m and j == l), 2)-Q(int(i == j and l == m), 8)

same4 = Q(0)
aabb = Q(0)
abab = Q(0)
for i in range(4):
    for j in range(4):
        for l in range(4):
            for m in range(4):
                same4 += k(i,j,j,l)*k(l,m,m,i)+k(i,j,l,m)*k(j,l,m,i)+k(i,j,m,i)*k(j,l,l,m)
                aabb += k(i,j,j,l)*k(l,m,m,i)
                abab += k(i,j,l,m)*k(j,l,m,i)
diag = 4*same4
cross = 2*(aabb+abab)
ordered_jordan = 4*diag+12*cross
assert (same4, aabb, abab) == (Q(435,16),Q(225,16),Q(-15,16))
assert ordered_jordan == 750 and 4*ordered_jordan == 3000
# Contrary: commuting/ABAB=AABB shortcut gives a different exact answer.
wrong_cross = 4*aabb
wrong_jordan = 4*(4*diag+12*wrong_cross)
assert wrong_jordan != 3000
proof = json.loads((ROOT/NAMES[1]).read_text())
c = proof['primitive_binding']['neutral_paths'][0]['coefficients']
csum = sum(Q(re)**2+Q(im)**2 for re,im in c)
m2 = Q(15,4)*csum
assert m2 == Q(proof['uniform_prices']['neutral_M2_exact'])
m4 = 3*m2*m2
assert m4 < 2**24
eta,chi,lam,bigf = Q(1,2**4200),Q(1,2**4096),Q(2**8192),Q(2**10240)
alpha,delta,t = Q(1,2**20),Q(1,2**100),Q(2**7168)
neutral = lam/Q(2**2100)*(-2*alpha*m2+Q(4,3)*alpha**3*Q(2**24))
prices = {
 'Phi_base':60*chi/eta, 'Jordan':3000*lam*eta**2,
 'alignment':Q(9,2)*bigf*eta, 'Haar_base_cross':Q(27,4)*chi*t,
 'colour_potential':Q(7,5)*bigf/t*(450+48*7168+24),
 'Phi_circuit':Q(2**30)*chi/eta, 'Haar_circuit':Q(2**30)*chi,
 'full_rawE_once':Q(2**30), 'charged_matter':Q(2**24)/Q(2**2100),
}
assert neutral < -Q(2**6072)
assert neutral+sum(prices.values()) < -Q(2**6070)
# Adverse alpha and missing negative scale reject the target (no narrowed assertion).
wrong_sign_total = -neutral+sum(prices.values())
zero_alpha_total = sum(prices.values())
assert wrong_sign_total >= -Q(2**6070) and zero_alpha_total >= -Q(2**6070)
# Full operator caps, including every sector, not a tangent-state truncation.
a = Q(2**10)*alpha+Q(2**35)*delta
assert a < Q(1,2**9)
# 60 Cart coordinates/site; 15 colour generators/matching edge; all-E and packet cross.
jet_prices = {'Phi_over_chi_eta':240*a*a,
              'Haar_over_chi':Q(9,4)*120*a*a,
              'rawE_once_with_packet':7680*a*a+2}
assert all(x < 2**30 for x in jet_prices.values())
# 64 species, onsite and all six directed paths with adjoints; sqrt120<11.
charged_absolute_coefficient = 64*(4*64*11+6*8*64*11)
assert charged_absolute_coefficient < 2**24
roles=proof['primitive_binding']['charged_roles']
assert sum(r['count'] for r in roles)==64
assert all(Q(r['fullPhi_onsite_scalar_exact']) < 64 for r in roles)
assert Q(roles[0]['fullPhi_onsite_scalar_exact']) != Q(roles[1]['fullPhi_onsite_scalar_exact'])
# Structural capsule witnesses: source and supplier parsed, neither executed.
source_ast=ast.parse((ROOT/NAMES[0]).read_text())
assert any(isinstance(n,ast.FunctionDef) and n.name=='uniform_prices' for n in source_ast.body)
print(json.dumps({'kind':'INDEPENDENT_EXACT_ENERGY_CONTROLS_NOT_SCIENTIFIC_ADMISSION',
 'input_sha256':hashes,'Jordan_exact':{'Tr_A4':str(same4),'Tr_A2B2':str(aabb),'Tr_ABAB':str(abab),'sum_ordered_anticommutator_squared':str(ordered_jordan),'Hamiltonian_coefficient':3000,'wrong_commuting_coefficient_rejected':str(wrong_jordan)},
 'neutral_M2_exact':str(m2),'Wick_M4_bound_strictly_below_2_24':m4<2**24,
 'strict_per_site_below_minus_2_6070':True,'adverse_wrong_sign_rejected':True,'adverse_zero_alpha_rejected':True,
 'allsector_jet_prices':{n:str(v) for n,v in jet_prices.items()},
 'charged_absolute_coefficient':charged_absolute_coefficient,'distinct_onsite_slopes':True,
 'physical_floating_evaluation':False},sort_keys=True))
