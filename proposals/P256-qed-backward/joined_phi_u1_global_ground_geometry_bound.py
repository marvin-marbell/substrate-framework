"""NEW Jordan: root-independent ground form moments, not ground identification."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import zipfile
from fractions import Fraction as F
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INPUT = "joined_phi_u1_stationary_flat_holonomy_typed_certified_20261010_receipt.json"
EXPECTED = "1c79b1a74c4236079645ebf9765a48c1c2263ec7b7209e9f103704c98752da94"
PROOF = {
    "law": "NEW Jordan quartic only; N4/V64, Lambda2^8192, chi2^-4096, Lambda54=2^10240, HE=HC64, all64 charged canonical copies/Ref4 and neutral128. No adoption or root premise.",
    "matter_global": "Let X=||Phi||HS=||Cart||/sqrt2. Every literal one/two-hop matrix entry has magnitude at most its absolute profile coefficient times X: endpoint averages, full SU4 paths and real Peierls phases have unit norm. Reverse paths are included in w=|onsite|+2sum(|c0|+|ci|). All256 row/column sums give ||h_s||<=256w_s X. Fixed-N exterior dGamma lower >=-N||h_s||; neutral physical -Lambda is bounded using absolute value, never positive-gap subtraction. Ref4 has no hopping Hamiltonian. Thus all matter >=-Lambda C X, C=256[128w_n+(128w_a+8064w_u)/Lambda]. Executed rational profile budgets bound C<2^40 globally, not only in a local tube.",
    "coercivity": "For arbitrary Hermitian Phi, PSD Gram/SOS gives VGram>=Lambda X^4/64 at aV=1; NEW VJ>=VGram. If C<=2^40, X^4/128-CX>=-2^64: for X<=2^24 use CX<=2^64; for X>=2^24 use X^3/128>=2^65>=C. Hence H+Lambda2^64 >= K+Lambda X^4/128+Lambda54 W+HC, W=A/40+(7/10)P. X is HS, not Cart.",
    "trial_orbit": "Use O0={Phi=0, U_xy=g_x g_y^dag}: compact embedded SU4^64/diagonalSU4 orbit, dimension945, in ambient R3840 x SU4^192 of dimension6720; codimension5775. Choose f=(1-dist(F,O0)/delta)_+, delta2^-2048, actual normalization Z=integral f^2>0. All65 matter copies occupy all4 colours on the32 even sites (N128), Ref4; these local exterior4 factors are SU4 singlets. f is gauge-invariant, so the full section is physical. Every matter expectation is EXACT0 for every F: traceless onsite fields and zero offsite onebody density. This does not make it a ground or source.",
    "tube_and_domain": "Complete auxiliary product metric has Ric>=0, T=B/sqrt8 orthonormal and TrT^2=.5. On each regular minimizing normal ray before first cut/focal time, full transverse Riccati has6719 directions. j(r)~r^5774 and u=j^(1/6719) has u(0)=0,u''<=0; hence j(2r)<=2^6719j(r). Zero extension after first cut preserves scaling, not concavity. Unique footpoint a.e./normal integration gives VolTube(2r)<=2^6720VolTube(r). Thus ||grad f||2^2/||f||2^2<=4*2^6720/delta^2. K<=4chi grad2 gives16chi2^6720/delta^2. Zero-trace Lipschitz tent is in the Friedrichs FORM domain, not D(H2); no gap/projector/local root is used.",
    "geometry_trial": "In the delta-tube, X^2<=delta^2/2. For any unitary links, A<=12X^2<=6delta^2 (six endpoint incidences/site). Relative to closest pure-gauge links, ||DeltaU_e||HS<=distance_e/sqrt2. A face obeys P_face=.5||I-Uface||HS^2<=2sum_face||DeltaU_e||HS^2; each link belongs to4 faces, so total P<=8sum_e||DeltaU_e||HS^2<=4delta^2. Therefore W<=(59/20)delta^2. VJ<=16VGram<=64Lambda X^4<=64Lambda delta^4.",
    "electric_cover": "rho_x=128 parity_x from all64 charged copies; Ref4 background charge0. L parity=12 parity, so HC=64*64*128^2/12=2^24/3. Theta-constant beta0 fibre has HE0. For normalized original-Hilbert beta packets concentrated at0, HE<=64 is retained as a finite conservative upper; no real-cover L2 vacuum, arbitrary Borel evaluation or alpha0 integer-flow assumption is made. The finite-energy continuous-C* beta-packet limit supplies the same bound.",
    "ground_consequence": "VGram<=VJ<=16VGram preserves the closed, confining, compact-resolvent physical Bloch fibres. The global fibre minimum betaStar exists by the same norm-resolvent Bloch form continuity; its E0<=trial beta0 energy. Every normalized vector in its ground space obeys the executed moment bound from the form inequality. This identifies neither betaStar nor the ground vector/degeneracy. Markov is a probability bound on W, not exact flat support, distance-to-allflat certification or Clifford selection.",
    "stationary_current": "For a real unit direction v in orthonormal129 Theta coordinates, edge flow Qv has norm1. Every literal path has length<=2, so its Peierls derivative factor |inc.Qv|<=sqrt2<2 and second factor<=4. Neutral charge0 contributes no U1 current. Thus J_v=dH(theta+av)/da is matter-only with ||J_v(F)||<=Cj X, Cj=512(128w_a+8064w_u), and contact ||K_v(F)||<=Ck X, Ck=1024(128w_a+8064w_u). Both are <2^40 by the executed profile bound. The FULL H still includes Pi/HC/Phi/Haar. Ground quartic moment implies ||J_v P0||^2<2^116 and ||P0 K_v P0||<2^58. For omega>0, the positive Q-accessible resolvent P0 J_v(1-P0)(H-E0+omega)^-1 J_v P0 has norm<=2^116/omega. These are domain/total-weight bounds, not evaluated spectrum, IR control or a pole. A momentum-source probe additionally differentiates HE and is not this matter-phase derivative.",
}


def run(output: Path) -> dict[str, Any]:
    sys.set_int_max_str_digits(0)
    start = time.monotonic()
    source = Path(__file__).resolve()
    source_bytes = source.read_bytes()
    raw = (ROOT / INPUT).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    archive_path = output.with_name(output.stem + "_input_bytes.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("executed_source/" + source.name, source_bytes)
        archive.writestr("input/" + INPUT, raw)
    saved = json.loads(raw)
    profiles = saved["primitive_profiles"]
    assert len(profiles) == 3
    weights = [abs(F(p["onsite"])) + 2*sum((abs(F(row["c0"])) + abs(F(row["ci"])) for row in p["paths"]), F(0)) for p in profiles]
    lam, chi, fast, delta = F(2)**8192, F(2)**-4096, F(2)**10240, F(2)**-2048
    C = 256*(128*weights[0] + (128*weights[1] + 8064*weights[2])/lam)
    assert 0 < C < F(2)**40
    assert all(int(row["length"]) <= 2 for p in profiles for row in p["paths"])
    Cj = 512*(128*weights[1] + 8064*weights[2])
    Ck = 1024*(128*weights[1] + 8064*weights[2])
    assert 0 < Cj < F(2)**40 and 0 < Ck < F(2)**40
    D, orbit = 64*4*15+192*15, 64*15-15
    assert D == 6720 and orbit == 945 and D-orbit == 5775
    kinetic = 16*chi*F(2)**D/delta**2
    quartic = 64*lam*delta**4
    geometry = F(59,20)*fast*delta**2
    HC, HE = F(2)**24/3, F(64)
    trial = kinetic+quartic+geometry+HC+HE
    assert trial/lam < F(2)**-1467 < 1
    W_upper = (F(2)**64+trial/lam)/(fast/lam)
    X4_upper = 128*(F(2)**64+trial/lam)
    assert W_upper < F(2)**-1983 and X4_upper < F(2)**72
    assert W_upper/F(2)**-1024 < F(2)**-959
    assert Cj**4 * X4_upper < F(2)**232
    assert Ck**4 * X4_upper < F(2)**232
    # Exact wrong-unit and missing-colour tests reject plausible false claims.
    # Uniform four-field Gram gI, g1/4 is merely a saturation control, not a root.
    g = F(1,4)
    X_HS2, X_Cart2 = 1024*g, 2048*g
    exact_gram_potential = 16384*g*g
    assert exact_gram_potential == X_HS2**2/64
    wrong_Cart_lower = X_Cart2**2/64
    assert wrong_Cart_lower == 4*exact_gram_potential > exact_gram_potential
    Haar_P = F(192*4)  # Phi0 alone does not control colour plaquettes.
    Haar_geometry = F(7,10)*fast*Haar_P
    assert Haar_geometry/lam > F(2)**2048
    wrong_one_flavour_HC = F(64*64*2**2,12)
    assert HC == 4096*wrong_one_flavour_HC
    result = {
        "stage":"NEW_JORDAN_ROOT_INDEPENDENT_GLOBAL_GROUND_FORM_MOMENTS_NOT_GROUND_IDENTIFICATION",
        "plan_id":"issue-222-induced-photon-delta13-continuation-20261008",
        "captured_revision":"55352a0aabbcc1da539641239f48f2807ec1e2a609b5d533ca0f186d8df27b6e",
        "proof":PROOF,
        "derived_budgets":{"profile_weights":[str(w) for w in weights],"matter_C_over_Lambda":str(C),"matter_C_coarse":"<2^40","trial_over_Lambda":"<2^-1467","W_ground_expectation":"<2^-1983","X_HS_fourth_ground_moment":"<2^72","Markov_probability_W_gt_2^-1024":"<2^-959","geometry_trial_coefficient":"59/20","ambient_dimension":D,"orbit_dimension":orbit,"codimension":D-orbit,"HC":str(HC),"HE_original_packet_upper":str(HE)},
        "stationary_current_domain":{"real_unit_Theta_direction":True,"Cj":str(Cj),"Ck":str(Ck),"ground_J_squared":"<2^116","ground_contact_mean_norm":"<2^58","positive_Q_resolvent_norm":"<=2^116/omega, omega>0","not_a_pole_or_IR_bound":True,"not_a_HE_momentum_probe":True},
        "controls":{"HS_correct_quartic":str(exact_gram_potential),"wrong_Cart_quartic_lower":str(wrong_Cart_lower),"Phi0_uniform_Haar_P":str(Haar_P),"full64_HC":str(HC),"wrong_single_HC":str(wrong_one_flavour_HC)},
        "source_sha256":hashlib.sha256(source_bytes).hexdigest(),"input_sha256":{INPUT:EXPECTED},
        "immutable_input_archive":{"path":archive_path.name,"sha256":hashlib.sha256(archive_path.read_bytes()).hexdigest()},
        "source_inputs_unchanged":source.read_bytes()==source_bytes and (ROOT/INPUT).read_bytes()==raw,
        "elapsed_seconds":time.monotonic()-start,
        "OPEN":["actual ground betaStar/vector/degeneracy and allflat proximity","proper Jordan31/22 and off-flat variational exclusion","physical stationary photon spectral support/pole/dispersion and useful IR remainder","commoncone/originalSOURCE/adoption/campaign"],
    }
    assert result["source_inputs_unchanged"]
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    result=run(parser.parse_args().output)
    print(json.dumps({"stage":result["stage"],"budgets":result["derived_budgets"],"elapsed_seconds":result["elapsed_seconds"]},indent=2))
