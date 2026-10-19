"""NEW Jordan full-ground form moments via native tracezero/Frobenius bounds."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
from typing import Any
import numpy as np
from threadpoolctl import threadpool_limits
import induced_phi_nonuniform_field_car as field
import joined_phi_u1_global_ground_geometry_bound as previous

ROOT = Path(__file__).resolve().parent
PROOF = dict(previous.PROOF)
PROOF.update({
    "matter_global": "On a literal nearest bond hxy=[c0 average(Phi0)+i ci average(Phii)]U, both averages are Hermitian in the source frame. Their HS cross is imaginary times a real trace and vanishes. Endpoint convexity bounds each average HS square by the mean endpoint HS squares. Summing distinct onsite/nearest matrix blocks gives ||h_neutral||HS^2<=C0 X0^2+sum Cj Xj^2. Neutral has nearest paths only. On N4, charged length2 forward/reverse paths share the same pair of sites; retain ||a+b||^2<=2(||a||^2+||b||^2), giving factor4 rather than2 on length2 channels. All SU4 and charged Peierls paths remain unitary. Every h has Trh=0 because onsite primitive fields are traceless and no path returns onsite. Thus GLOBAL128 KyFan and both Fock extremal energies obey |dGamma(h)|<=Tr|h|/2<=8||h||HS. This is not blockhalf filling. Profile channels below give neutral norm<26X and each charged norm<64X. Hence all matter >=-Lambda C X with C=8[26+4096/Lambda]<256. Neutral physical -Lambda, charged scale1, Ref4, all63 auxiliaries retained.",
    "coercivity": "For X=||Phi||HS, VJ>=VGram>=Lambda X^4/64. X^4/128-256X>=-2^13: if X<=32 use256X<=8192, else X^3/128>=256. Thus H+Lambda2^13>=K+Lambda X^4/128+Lambda54 W+HC globally. The prior root-independent physical Phi0/puregauge trial and compact-fibre/cover/form-domain derivation are unchanged, not rerun or identified with a ground vector.",
    "stationary_current": "For a real unit129Theta direction, each literal one/two path first phase factor<=2 and second<=4. The onsite terms have no phase derivative; neutral charge0 current vanishes. Both derivative matrices are Hermitian and tracezero. Same collision/endpoint HS estimate bounds ||h'_s||HS<128X and ||h''_s||HS<256X. GLOBAL128 Fock norm<=8HS; all64 charged copies give ||J_v||<2^16 X and ||K_v||<2^17 X. The FULL H retains Pi/HC/Phi/Haar. Quartic ground moment<2^21 implies ||J_v P0||^2<2^43, ||P0 K_v P0||<2^23, and positive Q-accessible resolvent norm<=2^43/omega for omega>0. This is the Theta-Peierls matter-current probe, not the independent Pi-translation kinetic probe; no IR/pole/dispersion inference.",
})


def channels(profile: dict[str, Any], onsite: bool = True) -> list[F]:
    result = [F(profile['onsite'])**2 if onsite else F(0), F(0), F(0), F(0)]
    for row in profile['paths']:
        length = int(row['length'])
        assert length in (1, 2)
        factor = 2 if length == 1 else 4
        result[0] += factor*F(row['c0'])**2
        result[1+int(row['axis'])] += factor*F(row['ci'])**2
    return result


def run(output: Path) -> dict[str, Any]:
    sys.set_int_max_str_digits(0)
    started = time.monotonic()
    raw = (ROOT/previous.INPUT).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == previous.EXPECTED
    profiles = json.loads(raw)['primitive_profiles']
    assert len(profiles) == 3 and all(int(row['length']) == 1 for row in profiles[0]['paths'])
    squared = [channels(p) for p in profiles]
    hopping = [channels(p, False) for p in profiles]
    assert max(squared[0]) < 26**2
    assert all(max(c) < 64**2 for c in squared[1:])
    assert all(max(c) < 64**2 for c in hopping[1:])
    lam, fast = F(2)**8192, F(2)**10240
    C = 8*(26+F(4096)/lam)
    assert C < 256
    trial = 2**6724 + 64*lam*F(2)**-8192 + F(59,20)*fast*F(2)**-4096 + F(2)**24/3 + 64
    assert trial/lam < F(2)**-1467
    x4 = 128*(F(2)**13+trial/lam)
    W = (F(2)**13+trial/lam)/(fast/lam)
    assert x4 < F(2)**21 and W < F(2)**-2034
    assert W/F(2)**-1024 < F(2)**-1010
    assert (F(2)**16)**4*x4 < F(2)**86
    assert (F(2)**17)**4*x4 < F(2)**92
    snapshots = {Path(__file__).name:Path(__file__).read_bytes(),Path(previous.__file__).name:Path(previous.__file__).read_bytes(),previous.INPUT:raw}
    data = field.load_mean(g_override=[1,1,1,1],mean_descriptor={'scope':'Global native HS control only, no root/covariance/preparation adoption'})
    native = data['native']
    phi = np.zeros((64,4,4,4),complex)
    phi[:,0] = np.diag([3,1,-2,-2])
    X2 = float(np.vdot(phi,phi).real)
    native_controls = []
    for i,p in enumerate(data['profiles']):
        links = np.repeat(np.eye(4,dtype=complex)[None],192,axis=0)
        if i:
            links *= np.exp(1j*np.pi/4)
        h = field.assemble(p,native,phi,links)
        hs2 = float(np.vdot(h,h).real)
        upper = float(squared[i][0])*X2
        assert abs(np.trace(h)) < 1e-10 and hs2 <= upper + 1e-7
        spectral = np.linalg.eigvalsh(h)
        fock_norm = max(abs(float(np.sum(spectral[:128]))),abs(float(np.sum(spectral[-128:]))))
        assert fock_norm <= 8*np.sqrt(hs2)+1e-8
        wrong = F(p['onsite'])**2+2*sum(F(row['c0'])**2 for row in p['paths'])
        if i:
            assert hs2 > float(wrong)*X2 + 1
        native_controls.append({'role':profiles[i]['role'],'HS_squared':hs2,'certified_channel_upper':upper,'Fock_GLOBAL128_norm':fock_norm,'wrong_unpaid_length2_collision_upper':float(wrong)*X2})
    snapshots.update(field.CAPTURED)
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.suffix=='.py' and p.parent==ROOT:
                snapshots.setdefault(p.name,p.read_bytes())
    assert all((ROOT/name).read_bytes()==content for name,content in snapshots.items())
    archive = output.with_name(output.stem+'_input_bytes.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(snapshots.items()):z.writestr(name,content)
    result = {'kind':'NEW_JORDAN_GLOBAL_NATIVE_FROBENIUS_FORM_MOMENTS_NOT_GROUND_IDENTIFICATION','plan_id':'issue-222-induced-photon-delta13-continuation-20261008','captured_revision':'0bec1a8c2e8ddf0d21d0a9f2c4b4460abc88f7b481f50f0e2d9d0218d29ab42c','proof':PROOF,'exact_profile_squared_channels':[list(map(str,c)) for c in squared],'matter_C_upper':str(C),'global_ground_budgets':{'X_HS_fourth':'<2^21','geometry_W':'<2^-2034','Markov_W_gt_2^-1024':'<2^-1010','J_P0_squared':'<2^43','P0_contact_mean_norm':'<2^23','positive_Q_resolvent_norm':'<=2^43/omega, omega>0'},'independent_native_controls':native_controls,'source_input_sha256':{name:hashlib.sha256(content).hexdigest() for name,content in snapshots.items()},'immutable_input_archive':{'path':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},'source_inputs_unchanged':True,'OPEN':['ground_betaStar/vector/degeneracy/phase/allflat_proximity','proper31/offflat/charged_full_ground','stationary_sameground_spatial_photon/IR/pole/dispersion','commoncone/originalSOURCE/adoption/campaign'],'elapsed_seconds':time.monotonic()-started}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    with threadpool_limits(limits=1):
        receipt=run(parser.parse_args().output)
    print(json.dumps({'kind':receipt['kind'],'budgets':receipt['global_ground_budgets'],'elapsed_seconds':receipt['elapsed_seconds']},indent=2))
