"""Independent asymmetric q/-q sector and literal finite-CAR execution receipt."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import mpmath as mp
import numpy as np
from scipy.linalg import expm
import induced_phi_car_superadiabatic_transfer as transfer

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).read_bytes()

def run():
    started = time.monotonic()
    mp.mp.dps = 100
    G = mp.diag([2,5])
    A = mp.matrix([[3,1j],[-1j,2]])
    q, mq = (1,0,0),(3,0,0)
    plus = [[[1+2j,3-1j],[-2+1j,1-1j]],[[2-1j,1+3j],[4+2j,3+2j]]]
    reverse_plus = [[[2-3j,-1+2j],[5+1j,2+4j]],[[3+1j,4-2j],[1-4j,-2+1j]]]
    gaps = [[7,11],[13,17]]
    reverse_gaps = [[19,23],[29,31]]
    def adjoint(values):
        return [[[complex(axis[j][i]).conjugate() for j in range(2)] for i in range(2)] for axis in values]
    def transpose(values):
        return [[values[j][i] for j in range(2)] for i in range(2)]
    common = dict(role='active filled charged',load=1,physical_scale='charged',negative_rank=2,positive_rank=2)
    records = [dict(common,q_integer=q,k_in=(0,0,0),k_out=(1,0,0),PH_plus=plus,PH_minus=adjoint(reverse_plus),gap_plus=gaps,gap_minus=transpose(reverse_gaps)),
               dict(common,q_integer=mq,k_in=(1,0,0),k_out=(0,0,0),PH_plus=reverse_plus,PH_minus=adjoint(plus),gap_plus=reverse_gaps,gap_minus=transpose(gaps))]
    blocks = {q:dict(G=G,A=A),mq:dict(G=G,A=A.conjugate())}
    solved = transfer.all_q_counterrotations(blocks,records)
    eta = mp.matrix([mp.mpc(1,2)/32,mp.mpc(-4,1)/64])
    eta_reverse = eta.conjugate()
    def sector(item,coordinate):
        result = mp.zeros(4)
        for i in range(2):
            for j in range(2):
                result[2+i,j] = (coordinate.T*item['dual_PH']['plus'][2*i+j]['t'])[0]
                result[i,2+j] = (coordinate.T*item['dual_PH']['minus'][2*i+j]['t'])[0]
        return result
    forward,reverse = sector(solved[0],eta),sector(solved[1],eta_reverse)
    T = mp.zeros(8)
    for i in range(4):
        for j in range(4):
            T[4+i,j],T[i,4+j] = forward[i,j],reverse[i,j]
    anti = transfer.norm(T+T.H)
    # This wrong check identifies minus(q,k) with plus(q,k), instead of the
    # reversed -q momentum/band sector. Asymmetry makes that identification fail.
    isolated = max(transfer.norm(solved[0]['dual_PH']['minus'][2*i+j]['t']+solved[0]['dual_PH']['plus'][2*j+i]['t'].conjugate()) for i in range(2) for j in range(2))
    # The old identical Kato sign on the two PH sectors turns the true paired
    # anti-H relation into a Hermitian generator. No imaginary coefficient is cut.
    wrong = T.copy()
    for item,coordinate,left,right in ((solved[0],eta,4,0),(solved[1],eta_reverse,0,4)):
        for i in range(2):
            for j in range(2):
                wrong[left+i,right+2+j] = -(coordinate.T*item['dual_PH']['minus'][2*i+j]['t'])[0]
    wrong_anti = transfer.norm(wrong+wrong.H)
    equation = max(transfer.norm(leg['equation_defect']) for item in solved for legs in item['dual_PH'].values() for leg in legs)
    tn = np.asarray(T.tolist(),dtype=complex)
    one_particle = expm(tn)
    annih = [transfer.joined.scalar.car_annihilator(i,8).toarray() for i in range(8)]
    dgamma = sum((tn[i,j]*annih[i].conj().T@annih[j] for i in range(8) for j in range(8)),np.zeros((256,256),complex))
    literal = expm(dgamma)
    psi = np.zeros(256,complex)
    psi[(1<<0)+(1<<1)+(1<<4)+(1<<5)] = 1
    rotated = literal@psi
    number = np.asarray([state.bit_count() for state in range(256)])
    particle_number = float(np.vdot(rotated,number*rotated).real)
    car_unitarity = float(np.linalg.norm(literal.conj().T@literal-np.eye(256)))
    one_unitarity = float(np.linalg.norm(one_particle.conj().T@one_particle-np.eye(8)))
    if not (anti < mp.mpf('1e-90') and equation < mp.mpf('1e-90') and isolated > mp.mpf('.01') and wrong_anti > mp.mpf('.01')):
        raise ArithmeticError('Asymmetric q/-q control did not discriminate paired-sector law')
    if car_unitarity > 2e-12 or one_unitarity > 2e-12 or abs(particle_number-4) > 2e-12:
        raise ArithmeticError('Literal paired finite-CAR exponential failed')
    descriptor = dict(law='Independent N4 asymmetric Fourier/CAR adjoint control, not a physical all-q source execution',q=q,minus_q=mq,
        G=transfer.encode(G),A_q=transfer.encode(A),A_minus_q=transfer.encode(A.conjugate()),records=records,
        eta_q=transfer.encode(eta),Fourier='exp(iqx)/8; eta_-q=conj(eta_q)',precision=100)
    # Complex declared inputs are recorded as exact Python literal bytes, rather
    # than pretending that JSON can encode the complex field/CAR law directly.
    descriptor['records'] = repr(records)
    captures = dict(transfer.SOURCE_BYTES)
    captures[Path(__file__).name] = SELF
    return dict(input_descriptor=descriptor,input_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
        actual_independent_asymmetric_control=dict(full_sector_anti_H_defect=transfer.display_scalar(anti),literal_resolvent_equation_defect=transfer.display_scalar(equation),
            adverse_isolated_q_adjoints_defect=transfer.display_scalar(isolated),adverse_old_identical_Kato_sign_anti_H_defect=transfer.display_scalar(wrong_anti),
            retained_imaginary_generator_norm=float(np.linalg.norm(tn.imag)),G_A_commutator_norm=transfer.display_scalar(transfer.norm(G*A-A*G)),
            literal_eight_mode_CAR_exponential_unitarity_defect=car_unitarity,one_particle_unitarity_defect=one_unitarity,
            actual_particle_number=particle_number,actual_state_norm_squared=float(np.vdot(rotated,rotated).real),
            pairing='minus(q,k;i,j)=-conj(plus(-q,k+q;j,i)); both different PH sectors independently resolved'),
        source_at_import={name:dict(path=str(ROOT/name),sha256=hashlib.sha256(raw).hexdigest(),bytes_utf8=raw.decode()) for name,raw in captures.items()},
        source_unchanged_since_import={name:(ROOT/name).read_bytes()==raw for name,raw in captures.items()},input_unchanged={},
        runtime=dict(executable=sys.executable,python=sys.version,mpmath=mp.__version__,numpy=np.__version__),elapsed_seconds=time.monotonic()-started)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = run()
    if not all(result['source_unchanged_since_import'].values()):
        raise ArithmeticError('Asymmetric control source changed during execution')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['actual_independent_asymmetric_control'],indent=2))
