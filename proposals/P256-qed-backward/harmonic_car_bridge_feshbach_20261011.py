"""New complete-source endpoint trial Feshbach representation; lower evidence only."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time, zipfile
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parent
INPUT='harmonic_car_bridge_geometry_attempt4_20261011_receipt.json'
FRAME='joined_phi_u1_ground_photon_collective_delivered_actual_20261010.npz'
STATE='local_so5_resum_actual04_20261011_state.npz'
PATHS='charged_first_shell_actual04_20261011_receipt.json'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def decode(M):return np.array([[complex(*z) for z in row] for row in M])
def child(epoch):
    start=time.monotonic();source=json.loads((ROOT/INPUT).read_bytes());old=json.loads((ROOT/PATHS).read_bytes())['all64_boundary']
    with np.load(ROOT/FRAME,allow_pickle=False) as data:phi=data['Phi_binary_section']
    had=np.array([[1,0,1,0],[0,1,0,1],[-1,0,1,0],[0,-1,0,1]],float)/np.sqrt(2);frame=np.einsum('ia,mij,jb->mab',had,phi,had);gamma=frame[1]/np.sqrt(np.trace(frame[1]@frame[1]).real/4);spin=np.column_stack([np.eye(4)[:,0],gamma@np.eye(4)[:,0],np.eye(4)[:,1],gamma@np.eye(4)[:,1]])
    def endpoint(role,creation,u,site):
        r=next(z for z in source['source_roles'] if z['physical_copy_count']==(1 if role==0 else 63) and z['orientation']==('creation' if creation else 'annihilation'));mom=r['endpoint_reduced_matrices'][str(u)];ix=[2*site,2*site+1];return {k:decode(v)[np.ix_(ix,ix)] for k,v in mom.items()}
    rows=[];weighted=0.;wrongdiag=0.;energy=source['source_energy_float'];minimum_gram=1.
    for alloc in old['all64_once_shared_pointwise_norms']:
        rr=[]
        for role,p in enumerate(old['roles']):
            path=next(z for z in p['paths'] if z['axis']==alloc['axis'] and z['length']==alloc['length']);physical=np.array([[complex(float(F(z[0])),float(F(z[1]))) for z in row] for row in path['matrix_exact']]);M=spin.conj().T@physical@spin;shift=64*path['external_edges_min']-2*energy;sites=[]
            role_geometry=next(z for z in source['paths'] if (z['axis'],z['length'])==(alloc['axis'],alloc['length']))['roles'][role]
            geometry_choices={tuple(z['sites']):z for z in role_geometry['all_sites_both_charge_choices']}
            for sa in range(2):
                for sb in range(2):
                    choice=geometry_choices.get((sa,sb))
                    if choice is None:continue
                    phases=np.diag([1,-1,1,-1]);N=phases@M@phases;terms=[]
                    for qa in [-1,1]:
                        orientation=next(z for z in choice['orientations'] if z['qA']==qa)
                        ua,ub=[float(F(z)) for z in orientation['affine_u_exact']]
                        A=endpoint(role,qa==1,ua,sa);B=endpoint(role,qa==-1,ub,sb)
                        G=np.kron(A['G'],B['G']);T=np.kron(A['T'],B['G'])+np.kron(A['G'],B['T'])+shift*G
                        ge=np.linalg.eigvalsh(G);minimum_gram=min(minimum_gram,float(ge[0]));assert ge[0]>0,'No numerical nullspace crop permitted'
                        vals,V=eigh((T+T.conj().T)/2,(G+G.conj().T)/2);assert vals[0]>0
                        for a in range(2):
                            for b in range(2):
                                coef=(N if qa==1 else N.conj())[2*a:2*a+2,2*b:2*b+2].reshape(-1);w=float((coef.conj()@G@coef).real)
                                if w==0:continue
                                amplitudes=V.conj().T@G@coef;weights=np.abs(amplitudes)**2;assert abs(float(weights.sum())-w)<1e-9
                                inverse=float(np.sum(weights/vals));lower=inverse*inverse/w
                                # Trial spectral measure retained in full four-endpoint span. Never an upper bound.
                                trial_inverse_square=float(np.sum(weights/vals**2));heat=[float(np.sum(weights*np.exp(-t*vals))) for t in [1/256,1/64,1/16]]
                                wrong=float((coef.conj()@G@np.linalg.solve(np.diag(np.diag(T)),G@coef)).real);mu=float((coef.conj()@T@coef).real/w)
                                assert inverse+1e-9>=w/mu
                                terms.append({'qA':qa,'spin_pair':[a,b],'source_weight':w,'trial_spectral_values':vals.tolist(),'trial_source_weights':weights.tolist(),'Feshbach_inverse_expectation_lower_float':inverse,'Cauchy_inverse_square_lower_float':lower,'trial_inverse_square_NO_UPPER_CLAIM':trial_inverse_square,'trial_heat_NO_UPPER_CLAIM':heat,'wrong_diagonal_only_inverse':wrong,'mean_Jensen_inverse_expectation_lower':w/mu})
                    sites.append({'sites':[sa,sb],'terms':terms,'inverse_square_lower_sum':sum(t['Cauchy_inverse_square_lower_float'] for t in terms),'wrong_diagonal_inverse_square_proxy':sum(t['wrong_diagonal_only_inverse']**2/t['source_weight'] for t in terms)})
            best=max(sites,key=lambda z:z['inverse_square_lower_sum']);weighted+=float(F(alloc['root_allocation_exact']))*p['count']*best['inverse_square_lower_sum'];wrongdiag+=float(F(alloc['root_allocation_exact']))*p['count']*best['wrong_diagonal_inverse_square_proxy'];rr.append({'role':p['role'],'count':p['count'],'all_sites':sites})
        rows.append({'axis':alloc['axis'],'length':alloc['length'],'roles':rr})
    # Exact Schur adverse control rejects promoting a PHP inverse to an upper bound.
    # A=[[2,1],[1,2]]: <e0,A^-1e0>=2/3 > 1/<e0,Ae0>=1/2.
    assert F(2,3)>F(1,2)
    result={'kind':'EXECUTED_NEW_ENDPOINT_FESHBACH_SOURCE_SPECTRAL_REPRESENTATION_LOWER_DIAGNOSTIC_NOT_PHP_CERTIFICATE','input_receipt_sha256':sha((ROOT/INPUT).read_bytes()),'paths':rows,'weighted_Cauchy_inverse_square_lower_float':weighted,'wrong_diagonal_only_proxy_float':wrongdiag,'minimum_full_endpoint_Gram_eigenvalue':minimum_gram,'no_nullspace_or_PH_crop':True,'meaning':'L=complete endpoint <S,H_ch S>,G=<S,S>. For positive full charged A, endpoint variational <s,A^-1s> >= (Gv)^* L^-1 (Gv). Cauchy then gives ||A^-1s||^2 >= lower_expectation^2/||s||^2. Full4 generalized endpoint spectral/heat measures are PROXIES, not full spectral measures or upper resolvent/heat control. Orthogonal changed spin-pair labels are summed; each within-label endpoint basis retains all4 vectors. The weighted max-site envelope remains conditional, not an allorder criterion or universal obstruction.','contrary_controls':{'exact_Schur_PHP_upper_rejected':True,'correct_inverse_expectation':'2/3','PHP_inverse_expectation':'1/2','mean_Jensen_below_endpoint_variational_inverse_exercised':True,'diagonal_omission_is_not_an_admitted_representation':True},'observed_new_next_criterion':'The source gap and moment-heat upper guards remain>=1, and the endpoint variational inverse-square lower diagnostic is also evaluated. A normalized linked criterion must reorganize the physical multi-root/source representation with complete exterior and overlap control; merely sharpening these same first-shell denominators cannot be asserted to pass. New candidate should be discriminated by exact full-source charged Stieltjes/Feshbach bounds and two-root shared-flow terms before any allorder admission. No normalization-away of the failed guard.','fullPhi_ground':False,'phase':False,'allorder':False,'campaign_complete':False,'elapsed_seconds':time.monotonic()-start}
    (ROOT/('harmonic_car_bridge_'+epoch+'_receipt.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='paths'},indent=2))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',required=True);parser.add_argument('--child',action='store_true');args=parser.parse_args()
    if args.child:return child(args.epoch)
    base='harmonic_car_bridge_'+args.epoch;own=Path(__file__);members={p.name:p.read_bytes() for p in [own,ROOT/INPUT,ROOT/FRAME,ROOT/STATE,ROOT/PATHS,ROOT/'harmonic_car_bridge_derivation_20261011.json']}
    assert sha(members[STATE])=='dd06400e1b7f70df935cf7ca02cf5ebf598481a8d5e209d1241aba5ed3fea2a2'
    def capture(suffix):
        with zipfile.ZipFile(ROOT/(base+suffix),'x',zipfile.ZIP_DEFLATED) as z:
            for n,raw in sorted(members.items()):z.writestr(n,raw)
    capture('_preexecution.zip');command=[sys.executable,own.name,'--epoch',args.epoch,'--child'];start=time.monotonic()
    with (ROOT/(base+'_stdout.txt')).open('xb') as out,(ROOT/(base+'_stderr.txt')).open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=out,stderr=err);code=process.wait()
    record={'command':command,'returncode':code,'natural_exit':True,'elapsed_seconds':time.monotonic()-start,'source_input_sha256':{n:sha(raw) for n,raw in members.items()},'source_inputs_unchanged':all((ROOT/n).read_bytes()==raw for n,raw in members.items())};(ROOT/(base+'_exit.json')).write_text(json.dumps(record,indent=2)+'\n')
    for suffix in ['_exit.json','_stdout.txt','_stderr.txt','_receipt.json']:
        p=ROOT/(base+suffix)
        if p.exists():members[p.name]=p.read_bytes()
    capture('_executed.zip');print(json.dumps(record,indent=2));sys.exit(code)
if __name__=='__main__':main()
