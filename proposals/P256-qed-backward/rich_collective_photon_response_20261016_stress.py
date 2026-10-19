"""High-precision actual small-lattice cold transfer stabilization discriminator."""
import json
from pathlib import Path
import numpy as np
import mpmath as mp
import rich_unitary_rotor_photon_20261016_thermal as rotor
import rich_collective_photon_response_20261016_model as model

ROOT=Path(__file__).resolve().parent
P='rich_collective_photon_response_20261016_'


def mp_det(B,twist):
    prod=mp.eye(len(B[0]))
    for b in B:prod=mp.matrix(b.tolist())*prod
    mat=mp.eye(len(B[0]))+mp.diag([complex(z) for z in twist])*prod
    return float(mp.log(abs(mp.det(mat)))),mat


def main():
    mp.mp.dps=80;rng=np.random.default_rng(46301);pos,arcs,D=rotor.geometry(2);V=len(pos);rows=[];raw={}
    for beta,M in ((1.,8),(8.,32),(16.,64)):
        theta=rng.normal(0,.5,(M,len(arcs)));alpha=rng.normal(0,.3,V);s=rng.choice([-1,1],(M,V))
        data=model.path_data(theta,alpha,s,arcs,D,beta,stabilized=True)
        exact,mat=mp_det(data['B'],data['twist']);stable=model.stable_logdet(data['B'],data['twist'])[1]
        direct=np.linalg.slogdet(np.eye(V)+data['twist'][:,None]*rotor.product(data['B']))[1]
        matrices=[*data['B'],np.diag(data['twist'])];G=model.stable_green(matrices)
        mpG=np.array(mp.inverse(mat).tolist(),dtype=complex)
        f,_=model.spatial_modes(pos,arcs,(np.pi,));vs=[model.interval_vertices(th,arcs,V,beta/M,f[0],True) for th in theta]
        scalar=np.exp(data['lam']*s)[:,None,:];A=np.array([v[0] for v in vs])*scalar
        AA=np.array([v[3] for v in vs])*scalar;a=np.cos(2*np.pi*np.arange(M)/M)/np.sqrt(M)
        dB=a[:,None,None]*A;ddB=a[:,None,None]**2*AA;step=.008
        calc=model.stable_direction(data['B'],dB,ddB,data['twist'],2*stable,step)
        plus,_=mp_det(data['B']+step*dB+.5*step**2*ddB,data['twist'])
        minus,_=mp_det(data['B']-step*dB+.5*step**2*ddB,data['twist'])
        high=(2*plus+2*minus-4*exact)/step**2+((plus-minus)/step)**2
        rows.append(dict(beta=beta,M=M,stable_logabsdet_error=abs(stable-exact),direct_logabsdet_error=abs(direct-exact),
                         stabilized_Green_error=float(abs(G-mpG).max()),stabilized_source_derivative_error=abs(calc[0]-high)))
        raw[f'B_b{beta:g}']=data['B'];raw[f'twist_b{beta:g}']=data['twist']
    result=dict(kind='80_DIGIT_SAME_DISCRETE_COLD_TRANSFER_NOT_CONTINUOUS_H_TRACE',rows=rows,mpmath_version=mp.__version__)
    np.savez_compressed(ROOT/(P+'stress_raw.npz'),**raw)
    (ROOT/(P+'stress_result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert max(r['stable_logabsdet_error'] for r in rows)<1e-8,result
    assert max(r['stabilized_Green_error'] for r in rows)<1e-8,result


if __name__=='__main__':main()
