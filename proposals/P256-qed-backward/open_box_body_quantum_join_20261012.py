"""New exact same-fstar lower/quantum-upper originating excess consumer, not science admission."""
from __future__ import annotations

from fractions import Fraction as Q
from math import isqrt
import hashlib
import json
from pathlib import Path


def sqrt_upper(value: Q) -> Q:
    scale = 1 << 256
    return Q(isqrt((value.numerator*scale*scale)//value.denominator)+1,scale)


def main() -> None:
    root = Path(__file__).parent
    pins = {
        'open_box_body_actual2_20261012_receipt.json': '205e2b66fa2a043cd826761c027ac3dcc65796c82ecf7469f290bcf65f5f86f8',
        'open_box_body_proof_20261012.json':'4c1f6b439bdd448695e6db4e5d1c689a3882eae1fc7bd7115e32bd46c08434b2',
        'open_box_body_interface_20261012.json':'e942f34fa4a94f11fdc9f9a4203d29709695353f24519f879c60131b4b628d8f',
        'quantum_minimum_sea_actual3_20261012_interface.json':'7b0c7889c323cb917ab6ee2530fb8502931db9065b8ab10428c57afb3362edcf',
        'quantum_minimum_sea_actual3_20261012_proof.json':'5b0c5f3fafc05370c20d58314f3e00f9dfcd61e0dbf9825acadf5cebfe57365f',
        'quantum_minimum_sea_actual3_20261012_natural_exit.json':'baff03b0b3f8587554c38dce67e3e6a274e2f9e2dda40c606daff7734a6df020',
        'quantum_minimum_sea_actual3_20261012_executed.zip':'d7899688303648934f6730ef3c8ecde6ab92370e2e647b7a6b96134a286a0fba',
    }
    for name,expected in pins.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    lower = json.loads((root/'open_box_body_actual2_20261012_receipt.json').read_text())
    upper = json.loads((root/'quantum_minimum_sea_actual3_20261012_interface.json').read_text())
    upper_proof = json.loads((root/'quantum_minimum_sea_actual3_20261012_proof.json').read_text())
    supplier_exit = json.loads((root/'quantum_minimum_sea_actual3_20261012_natural_exit.json').read_text())
    assert supplier_exit['returncode'] == 0 and supplier_exit['natural_exit']
    assert upper['quarter_normalization'] == 'x_mu=Tr(B_mu^2)/4'
    assert upper['f'] == '64*sum(x_mu^2)-2*BZmean(sqrt(M^2*x0+sum(D_i^2*x_i)))'
    assert upper['xstar'] == 'unique argmin on R_+^4 of f_infinity(x)'
    assert upper_proof['controls']['minimum']['BZ_weight_means'] == lower['fstar']['averaged_native_weights']
    assert upper_proof['controls']['minimum']['each_x_strict_lower'] == lower['fstar']['component_strict_lower'] == '1/128'
    N, R, L, moment = 1 << 257, 1 << 256, 1 << 64, Q(1 << 14)
    gate = lower['paid_large_parameter_gate']
    assert int(gate['minimum_N_even']) == N and int(gate['R']) == R and int(gate['L']) == L
    Etotal, epsQ = Q(gate['total_upper']), Q(upper['epsilon_quantum'])
    assert Etotal < Q(1,1 << 160) and epsQ < Q(1,1 << 3000)
    eps_upper = Q(1 << 16,N)+epsQ
    dch = Q(lower['Dch'])
    charged = 12*dch/Q(1 << 8192)
    eta = eps_upper+Etotal+charged
    assert eps_upper < Q(1,1 << 240) and eta < Q(1,1 << 159)
    body_norm2 = eta/64
    body_delta_probability = eta/(64*Q(1,1 << 128))
    Jordan_probability = eta/Q(1,1 << 80)
    body_component_probability = eta*1024
    assert body_delta_probability < Q(1,1 << 37)
    assert Jordan_probability < Q(1,1 << 79)
    assert body_component_probability < Q(1,1 << 149)
    # The cap removal is integrated second-moment control, not a simultaneous
    # patch event. Compare actual physical invariant norm squares at a site.
    uncapped_deviation = 2*moment/L**2+96*R**4*Q(40,1 << 2035)
    assert uncapped_deviation < Q(1,1 << 112)
    site_to_body_L1 = 6*sqrt_upper(uncapped_deviation)  # M^(1/4)<12
    site_to_minimum_L1 = site_to_body_L1+sqrt_upper(body_norm2)
    assert site_to_body_L1 < Q(1,1 << 53)
    assert site_to_minimum_L1 < Q(1,1 << 52)
    site_component_failure = 256*site_to_minimum_L1
    assert site_component_failure < Q(1,1 << 44)
    receipt = {
        'kind':'EXECUTED_PINNED_SAME_FSTAR_FULL_QUANTUM_UPPER_TO_ORIGINATING_BODY_EXCESS_AND_PER_SITE_NORM_NOT_PHASE',
        'pins':pins,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'domain':'Every even N>=2^257; R2^256 L2^64; formulas evaluated at minimum N and monotone upper for larger N',
        'quantum_upper_excess_exact':str(eps_upper),'neutral_lower_error_exact':str(Etotal),
        'charged_price_exact_per_Lambda_site':str(charged),'eta_upper_exact':str(eta),'eta_lt_2power_minus159':True,
        'retained_before_drop':'Q/(Lambda V)+(F/Lambda)<W> and complete charged -Dch sqrt(<s>)/Lambda; only NONNEGATIVE Q/F discarded for this numerical upper of eta',
        'body_norm_distance_squared_upper_exact':str(body_norm2),
        'prob_body_norm_distance_gt_2power_minus64_upper_exact':str(body_delta_probability),
        'prob_body_Jordan_excess_gt_2power_minus80_upper_exact':str(Jordan_probability),
        'prob_body_xmu_lt_1over256_upper_exact':str(body_component_probability),
        'actual_site_to_body_squared_deviation_upper_exact':str(uncapped_deviation),
        'actual_site_norm_square_L1_to_xstar_upper_exact':str(site_to_minimum_L1),
        'prob_actual_site_TrPhi_mu_squared_over4_lt_1over256_upper_exact':str(site_component_failure),
        'prob_actual_site_component_failure_lt_2power_minus44':True,
        'measure':'Inherited actual originating minimizing Bloch translation-averaged family physical Gauss/sewn-cover Born marginal; boxes volume weighted and partitions averaged. No longitudinal gauge-volume probability.',
        'proof':'Strong convexity128 gives64 E distance2+E DeltaJ<=eta. Markov uses same-fstar supplier variational upper. Capped tree average Jensen and cap second-moment tail give E||Phi^tree-B||2<=2M/L2+96R4a. Cauchy and E||B||2<=E s give E|TrPhi_mu2/4-x_Bmu|<=.5M^(1/4)sqrt(deviation). Add sqrt(E body distance2), then Markov gap xstar_mu-1/256>1/256. No simultaneous all-sites-good event or uncapped site Jordan-polynomial implication.',
        'parent_admission_required':True,'scientific_admission':False,'law_changed':False,
        'not_inferred':['Numerical ground or trial=ground','unique matrix frame','field stationarity','phase or magnetic stiffness','charged IR atom/residue','commoncone/Compton']}
    (root/'open_box_body_quantum_join_20261012_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Pinned SAME-fstar full quantum upper/body lower join executed: eta<2^-159; actual per-site component failure<2^-44; no phase/IR admission')


if __name__ == '__main__':
    main()
