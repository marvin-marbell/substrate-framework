"""Independent moment proof controls; no producer import or native solver/action."""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = "selected_memory_moments_review_"
Vector = list[F]
Matrix = list[Vector]


def dot(a: Vector, b: Vector) -> F:
    return sum((x*y for x, y in zip(a, b)), F(0))


def mv(a: Matrix, b: Vector) -> Vector:
    return [dot(row, b) for row in a]


def subtract(a: Vector, b: Vector) -> Vector:
    return [x-y for x, y in zip(a, b)]


def moments(h: Matrix, rho: Vector, state: Vector, energy: F) -> tuple[F, F, F]:
    norm = dot(state, state)
    rstate = [r*x for r, x in zip(rho, state)]
    mean = dot(state, rstate)/norm
    source = [x-mean*y for x, y in zip(rstate, state)]
    action = subtract(mv(h, source), [energy*x for x in source])
    return dot(source, source)/norm, dot(source, action)/norm, dot(action, action)/norm


def contrary_controls() -> dict[str, Any]:
    # Weighted path graph: exact ground is the uniform vector; all charge moves are1.
    h = [[F(1), F(-1), F(0)], [F(-1), F(3), F(-2)], [F(0), F(-2), F(2)]]
    rho = [F(-1), F(0), F(1)]
    ground = [F(1)]*3
    assert mv(h, ground) == [F(0)]*3
    m0, m1, m2 = moments(h, rho, ground, F(0))
    assert (m0, m1, m2) == (F(2, 3), F(1), F(2))
    variance = m2/m0-(m1/m0)**2
    assert variance == F(3, 4) > 0
    # The two nonzero eigenvalues solve lambda²-6lambda+6=0:3±sqrt3.
    # Positive variance therefore coexists with exactly two spectral atoms.
    hop = [[F(0), F(-1), F(0)], [F(-1), F(0), F(-2)], [F(0), F(-2), F(0)]]
    halfcomm = [[-(rho[i]-rho[j])**2*h[i][j]/2 for j in range(3)] for i in range(3)]
    assert all(halfcomm[i][j] == -hop[i][j]/2 for i in range(3) for j in range(3))
    assert dot(ground, mv(halfcomm, ground))/3 == m1
    bound = max(sum(abs(z) for z in row) for row in hop)
    comm = [[h[i][j]*(rho[j]-rho[i]) for j in range(3)] for i in range(3)]
    assert max(sum(abs(z) for z in row) for row in comm) == bound
    assert dot(mv(comm, ground), mv(comm, ground))/3 == m2
    mminus1 = F(1, 2)  # Exact inverse on the zero-mean subspace (solve Hx=rho).
    inverse_witness = [F(-5, 6), F(1, 6), F(2, 3)]
    assert mv(h, inverse_witness) == rho
    assert dot(rho, inverse_witness)/3 == mminus1
    assert m0*m0/m1 < mminus1

    symmetric_h = [[F(1), F(-1), F(0)], [F(-1), F(2), F(-1)], [F(0), F(-1), F(1)]]
    single = moments(symmetric_h, rho, ground, F(0))
    assert single[2]/single[0]-(single[1]/single[0])**2 == 0
    highest = [F(1), F(-2), F(1)]
    assert mv(symmetric_h, highest) == [3*x for x in highest]
    excited = moments(symmetric_h, rho, highest, F(3))
    assert excited[1] == F(-2, 3) < 0

    # A charge jump2 multiplies the double commutator contact by4.
    jump_h = [[F(1), F(-1)], [F(-1), F(1)]]
    jump = moments(jump_h, [F(0), F(2)], [F(1), F(1)], F(0))
    assert jump[1] == 2 and jump[1] != F(1, 2)

    # Exact nonnormalized state prices expectation and current-vector changes.
    exact_ground = [F(1), F(0)]
    s = F(1001, 1000)
    v = [s*F(12, 13), s*F(5, 13)]
    e = dot(v, v)-1
    distance = sum(abs(x-y) for x, y in zip(v, exact_ground))  # l1 upper for l2.
    k = [[F(0), F(1)], [F(1), F(0)]]
    first_difference = abs(dot(v, mv(k, v))-dot(exact_ground, mv(k, exact_ground)))/2
    assert first_difference <= (2+e)*distance/2
    j = [[F(0), F(1)], [F(-1), F(0)]]
    jdifference = subtract(mv(j, v), mv(j, exact_ground))
    assert dot(jdifference, jdifference) <= distance**2
    squared_weight_difference = abs(dot(mv(j, v), mv(j, v))-1)
    assert squared_weight_difference <= 2*(1+e)*distance+distance**2

    # Actual binary64 Hv-Dv cancellation with a bounded selected-size diagonal.
    diagonal = 2**20
    coupling = F(1, 3)
    fv = [0.6, 0.8]
    bv = [F(x) for x in fv]
    full_float = [diagonal*fv[0]+float(coupling)*fv[1], float(coupling)*fv[0]+diagonal*fv[1]]
    off_float = [F(full_float[i]-diagonal*fv[i]) for i in range(2)]
    off_exact = [coupling*bv[1], coupling*bv[0]]
    off_error = subtract(off_float, off_exact)
    generic_action_cap = F(1, 2**24)
    off_cap = 2*generic_action_cap+F(32*2**22, 2**53)
    assert 0 < dot(off_error, off_error) <= off_cap**2*dot(bv, bv)
    # H(rho v)-rho Hv must also price uncancelled floating diagonal terms.
    r = [-1.0, 1.0]
    rv = [r[i]*fv[i] for i in range(2)]
    hrv = [diagonal*rv[0]+float(coupling)*rv[1], float(coupling)*rv[0]+diagonal*rv[1]]
    current_float = [F(hrv[i]-r[i]*full_float[i]) for i in range(2)]
    current_exact = [2*coupling*bv[1], -2*coupling*bv[0]]
    current_error = subtract(current_float, current_exact)
    current_cap = 512*generic_action_cap+F(1024*2**22, 2**53)
    assert 0 < dot(current_error, current_error) <= current_cap**2
    return {"weighted_path_moments": [str(x) for x in (m0, m1, m2)], "positive_variance_with_two_atoms": str(variance), "atom_energies": "3-sqrt3,3+sqrt3", "single_energy_variance_zero": True, "stationary_excited_first_moment": str(excited[1]), "jump_two_rejects_unit_jump_contact": [str(jump[1]), "1/2"], "approximate_normalized_lift_prices": {"norm_defect": str(e), "distance_upper": str(distance), "first_moment_difference": str(first_difference)}, "actual_Hv_Dv_rounding_error_squared": str(dot(off_error, off_error)), "actual_uncancelled_current_rounding_error_squared": str(dot(current_error, current_error)), "inverse_moment_bound": [str(m0*m0/m1), str(mminus1)]}


def audit_scalars(r: dict[str, Any], frame: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    bcap = sum((2*abs(F(p["d_exact"]))*(int(p["spin_replicas"])+1)+4*abs(F(p["c0_exact"]))*(int(p["spin_replicas"])+3) for p in frame["profiles"]), F(0))
    assert bcap == F(supplied["hop_and_current_norm_upper_exact"]) and bcap < 4096
    dimension = r["complete_dimension"]
    u = F(1, 2**53)
    gamma = (2*dimension+8)*u/(1-(2*dimension+8)*u)
    underflow = F(100*dimension, 2**1074)+F(2, 2**490)
    guards = {k: F(z) for k, z in r["derived_response_error_bounds"].items()}
    e = guards["norm_defect"]
    distance = 2*F(r["source_residual_upper_exact"])/F(r["gap_lower_exact"])+e
    assert distance == F(supplied["source_distance_upper_exact"]) and e < F(1, 1000)
    action_error = 64*u/(1-64*u)*2**22+F(768, 2**48)+40*u/(1-40*u)*2**22+4*u*(2**22+abs(F(r["mu_float"])))+underflow
    m0, m1, m2 = (F(r[k]) for k in ["rho_connected_weight_float", "double_commutator_first_moment_float", "current_commutator_weight_float"])
    db = guards["source_insertion_error"]
    m0_error = 2*db+db**2+gamma*abs(m0)/(1-gamma)+underflow
    off_error = 2*action_error+32*u*2**22
    m1_error = bcap*(2+e)*distance/2+(gamma*(bcap+off_error)+off_error)*(1+e)**2/2+underflow
    jround = 512*action_error+1024*u*2**22+underflow
    # A sharper direct decomposition fits this price even with ||v||<=1+e.
    direct_jround = (256*action_error+384*u*2**22)*(1+e)+underflow
    assert direct_jround < jround
    jerror = bcap*distance+jround
    assert m2/(1-4*gamma)+underflow < 151**2
    m2_error = 302*jerror+jerror**2+gamma*abs(m2)/(1-gamma)+underflow
    intervals = [(m0-m0_error, m0+m0_error), (m1-m1_error, m1+m1_error), (m2-m2_error, m2+m2_error)]
    assert intervals == [tuple(F(x) for x in row) for row in supplied["moment_intervals_exact"]]
    assert all(lo > 0 for lo, hi in intervals)
    variance = intervals[2][0]/intervals[0][1]-(intervals[1][1]/intervals[0][0])**2
    inverse_bound = intervals[0][0]**2/intervals[1][1]
    assert variance == F(supplied["positive_energy_variance_lower_exact"]) and variance > 0
    assert inverse_bound == F(supplied["inverse_density_response_moment_lower_exact"])
    assert inverse_bound < F(r["source_resolvent_interval_exact"][1])
    return {"B_float": float(bcap), "source_distance": float(distance), "off_action_error": float(off_error), "jround": float(jround), "direct_jround": float(direct_jround), "moment_error_floats": [float(x) for x in (m0_error, m1_error, m2_error)], "intervals_float": [[float(x) for x in row] for row in intervals], "variance_lower": float(variance), "inverse_moment_lower": float(inverse_bound)}


def main() -> None:
    input_path = ROOT/(PREFIX+"inputs_20261011.json")
    inputs = json.loads(input_path.read_bytes())
    own = Path(__file__).read_bytes()
    for name, digest in inputs["file_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    with zipfile.ZipFile(ROOT/inputs["frozen_executed_archive"]) as archive:
        source = archive.read("local_so5_ground_current_moment_bounds_20261011.py")
        assert hashlib.sha256(source).hexdigest() == inputs["producer_sha256"]
        supplied = json.loads(archive.read("local_so5_ground_current_moment_bounds_20261011_receipt.json"))
        r = json.loads(archive.read("local_so5_ground_electric_memory_consumer_20261011_receipt.json"))
        frame = json.loads(archive.read("local_so5_rep_review_recovered_corrected_freeframe_20261011_proof.json"))
        parent_exit = json.loads(archive.read("local_so5_ground_current_moment_bounds_attempt1_20261011_exit.json"))
    assert parent_exit["natural_exit"] and parent_exit["returncode"] == 0
    result = {"verdict": "VALID", "scope": inputs["scope"], "analytic_review": inputs["analytic_review"], "controls": contrary_controls(), "scalar_audit": audit_scalars(r, frame, supplied), "source_sha256": hashlib.sha256(own).hexdigest(), "inputs_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(), "producer_sha256": inputs["producer_sha256"], "original_curvature_verdict_unchanged": True, "moments_previously_unpriced": "The earlier curvature artifact retains floating diagnostics. These NEW supplemental prices, not retrospective relabelling, certify the moments.", "not_absence_of_spectral_atoms": True, "bulk_stiffness": False, "fullPhi_ground": False, "phase": False, "pole": False, "current_norm_selected_finite_sector_only": True, "parent_execution": parent_exit, "all_inputs_unchanged": all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest for name, digest in inputs["file_sha256"].items()), "source_unchanged": Path(__file__).read_bytes() == own}
    assert result["all_inputs_unchanged"] and result["source_unchanged"]
    (ROOT/(PREFIX+"proof_20261011.json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
