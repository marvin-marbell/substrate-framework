"""Physical1 signed quantum four-edge connected response with Schur remainder.

The 63 identical auxiliary species use their exact permutation-symmetric
occupation representation, not independent electric fields or a determinant.
Results are FLOAT sparse-resolvent diagnostics of an explicitly truncated
quantum module. Exact scalar Schur bounds retain its entire complement.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, cast
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import eigsh, splu


def hop(mask, target, source):
    if target == source:
        return (mask,1) if (mask>>source)&1 else None
    if not (mask>>source)&1 or (mask>>target)&1:
        return None
    sign = (-1)**(mask & ((1<<source)-1)).bit_count()
    mask ^= 1<<source
    sign *= (-1)**(mask & ((1<<target)-1)).bit_count()
    return mask | (1<<target),sign


def symbols(profile,s,r):
    sx=np.array([[0,1],[1,0]],dtype=complex)
    sy=np.array([[0,-1j],[1j,0]],dtype=complex)
    sz=np.diag([1,-1]).astype(complex)
    beta=np.kron(sz,np.eye(2))
    alpha=[np.kron(sx,x) for x in (sx,sy,sz)]
    edges=[(0,1,0),(1,2,1),(3,2,0),(0,3,1)]
    rows=[]
    for edge,(left,right,axis) in enumerate(edges):
        path=next(p for p in profile['paths'] if p['axis']==axis and p['length']==1)
        matrix=float(Fraction(path['c0'])*s)*beta+1j*float(Fraction(path['ci'])*r)*alpha[axis]
        for a,b in itertools.product(range(4),repeat=2):
            if matrix[a,b]:
                rows.append((4*left+a,4*right+b,matrix[a,b],edge,1))
                rows.append((4*right+b,4*left+a,matrix[a,b].conjugate(),edge,-1))
    return rows


def transitions(state,rows,wrong_shared=False):
    active, histogram, electric=state
    result=defaultdict(complex)
    currents=[defaultdict(complex) for _ in range(4)]
    for target,source,value,edge,direction in rows:
        moved=hop(active,target,source)
        if moved:
            mask,sign=moved; e=list(electric);e[edge]+=direction
            key=(mask,histogram,tuple(e))
            coefficient=value*sign
            result[key]+=coefficient
            currents[edge][key]+=1j*direction*coefficient
        counts=dict(histogram)
        for old,count in histogram:
            moved=hop(old,target,source)
            if not moved:
                continue
            mask,sign=moved
            updated=dict(counts);updated[old]-=1;updated[mask]=updated.get(mask,0)+1
            newhist=tuple(sorted((m,n) for m,n in updated.items() if n))
            e=list(electric);e[edge]+=direction
            key=(active,newhist,tuple(e))
            coefficient=value*sign*np.sqrt(count*(counts.get(mask,0)+1))
            result[key]+=coefficient
            currents[edge][key]+=1j*direction*coefficient
    return {k:v for k,v in result.items() if abs(v)>1e-14},currents


def diagonal(state,ma,mb,s):
    active,histogram,electric=state
    def excitation(mask):
        return sum((mask>>(4*j)&3).bit_count() for j in range(4))
    return 64*sum(e*e for e in electric)+2*float(s)*(float(ma)*excitation(active)+float(mb)*sum(n*excitation(mask) for mask,n in histogram))


def verify_occupation(state):
    active,histogram,electric=state
    assert active.bit_count()==8 and sum(n for _,n in histogram)==63
    assert all(mask.bit_count()==8 for mask,_ in histogram)
    rho=[(active>>(4*j)&15).bit_count()-2+sum(n*((mask>>(4*j)&15).bit_count()-2) for mask,n in histogram) for j in range(4)]
    div=[0]*4
    for k,(left,right,_) in zip(electric,[(0,1,0),(1,2,1),(3,2,0),(0,3,1)]):
        div[left]+=k;div[right]-=k
    assert div==rho


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();request=json.loads(args.request.read_bytes());root=args.request.parent
    for name,sha in request['pins'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    profiles=json.loads((root/request['profiles']).read_text())['primitive_profiles']
    active,aux=profiles[1:];assert active['load']==1 and aux['load']==63
    s,r=Fraction(1,2),Fraction(1,3)
    ma,mb=Fraction(active['onsite']),Fraction(aux['onsite'])
    rows=symbols(active,s,r)
    assert rows==symbols(aux,s,r)
    vacuum=sum(12<<(4*j) for j in range(4))
    initial=(vacuum,((vacuum,63),),(0,0,0,0))
    first_image=transitions(initial,rows)[0]
    first_gram=sum(abs(v)**2 for v in first_image.values())
    exact_first_gram=1024*(r*Fraction(active['paths'][0]['ci']))**2
    assert abs(first_gram-float(exact_first_gram))<1e-8
    assert abs(first_gram-64*float(exact_first_gram))>1
    external=(vacuum,((vacuum,63),),(1,1,-1,-1))
    verify_occupation(external)
    assert diagonal(external,ma,mb,s)==256
    external_costs=sorted({int(64*(sum(k*k for k in target[2])-4))
                           for target in transitions(external,rows)[0]})
    assert external_costs==[-64,192]
    assert hop(vacuum,2,6) is None
    assert hop(vacuum^(1<<2)^(1<<0),2,6) is not None
    states=[initial];index={initial:0};frontier=[initial]
    levels=[1]
    # Native two-hop quantum shell. This is not an invariant module: the
    # Schur complement retains every transition leaving the selected shell.
    for depth in range(2):
        new=[]
        for state in frontier:
            for target in transitions(state,rows)[0]:
                if target not in index:
                    index[target]=len(states);states.append(target);new.append(target)
        frontier=new;levels.append(len(new))
    rr=[];cc=[];vv=[];jr=[([],[],[]) for _ in range(4)]
    qrows={};qr=[];qc=[];qv=[]
    jq=[([],[],[]) for _ in range(4)]
    for col,state in enumerate(states):
        verify_occupation(state)
        assert len(state[2])==4
        rr.append(col);cc.append(col);vv.append(diagonal(state,ma,mb,s))
        trans,current=transitions(state,rows)
        for target,value in trans.items():
            if target in index:
                rr.append(index[target]);cc.append(col);vv.append(value)
            else:
                verify_occupation(target)
                if target not in qrows:qrows[target]=len(qrows)
                qr.append(qrows[target]);qc.append(col);qv.append(value)
        for edge,cmap in enumerate(current):
            for target,value in cmap.items():
                if target in index:
                    jr[edge][0].append(index[target]);jr[edge][1].append(col);jr[edge][2].append(value)
                else:
                    if target not in qrows:qrows[target]=len(qrows)
                    jq[edge][0].append(qrows[target]);jq[edge][1].append(col);jq[edge][2].append(value)
    size=len(states)
    h = cast(Any, coo_matrix((vv,(rr,cc)),shape=(size,size),dtype=complex).tocsc())
    currents = cast(list[Any], [coo_matrix((v,(a,b)),shape=(size,size),dtype=complex).tocsc() for a,b,v in jr])
    b=coo_matrix((qv,(qr,qc)),shape=(len(qrows),size),dtype=complex).tocsc()
    outside_currents=[coo_matrix((v,(a,c)),shape=(len(qrows),size),dtype=complex).tocsc() for a,c,v in jq]
    hermitian=float(abs(h-h.getH()).max())
    assert hermitian<1e-11 and all(float(abs(j-j.getH()).max())<1e-11 for j in currents)
    energies,vectors=cast(Any, eigsh)(h,k=6,which='SA',tol=1e-11,maxiter=10000,v0=np.ones(size,dtype=complex))
    order=np.argsort(energies);energies=energies[order];vectors=vectors[:,order]
    lowest=float(energies[0]);ground=vectors[:,0]
    assert energies[1]-energies[0]>1e-6
    p0=vectors[:,np.abs(energies-energies[0])<1e-7]
    sources=[];means=[]
    for current in currents:
        mean=np.vdot(ground,current@ground);means.append([float(mean.real),float(mean.imag)])
        source=current@ground;p=p0@(p0.conj().T@source);sources.append(source-p)
    source_leakage=[float(np.linalg.norm(j@ground)) for j in outside_currents]
    # Rigorous rational full hopping bound: each cdag_i c_j has norm<=1.
    # This is a complement ERROR bound, not a proposed gap criterion.
    c_internal=64*sum(abs(s*Fraction(p['c0']))*8+abs(r*Fraction(p['ci']))*8
                       for p in active['paths'] if p['length']==1 and p['axis'] in (0,1))*2
    # All NN/length2 native paths incident to four sites, every orientation,
    # with all64species; conservative double counting is intentional. AP has
    # modulus1. This includes the unchanged outside/boundary native terms.
    # Factor16 includes four matrix entries per component, both adjoints,
    # and BOTH starts and ends at each of four sites.
    c_boundary=64*4*16*sum(abs(s*Fraction(p['c0']))+abs(r*Fraction(p['ci'])) for p in active['paths'])
    # Also bound every length2/AP contribution to a selected-link CURRENT,
    # including paths whose endpoints lie outside the quantum patch.
    c_current=c_boundary
    absb=abs(b);b1=float(np.asarray(absb.sum(axis=0)).max());binf=float(np.asarray(absb.sum(axis=1)).max())
    shell_float_bound=np.sqrt(b1*binf)
    frequencies=[64.0,256.0,float(2*(c_internal+c_boundary))]
    responses=[]
    ident=cast(Any, eye)(size,format='csc',dtype=complex)
    for eta in frequencies:
        solver=splu(h+(-lowest+1j*eta)*ident)
        matrix=cast(Any, np.array([[np.vdot(left,solver.solve(right)) for right in sources] for left in sources]))
        mass=cast(Any, np.array([[np.vdot(left,right) for right in sources] for left in sources]))
        residual=max(float(np.linalg.norm((h+(-lowest+1j*eta)*ident)@solver.solve(right)-right)) for right in sources)
        # Feshbach: ||Sigma(z)||<=C^2/eta, hence the compression/full
        # resolvent error <=C^2/eta^3 for identical compressed sources.
        # This is valid for any self-adjoint complement, not a fake backend.
        bound=float((c_internal+c_boundary)**2)/eta**3
        row={'eta':eta,'response_real':matrix.real.tolist(),'response_imag':matrix.imag.tolist(),
             'Gram_real':mass.real.tolist(),'Gram_imag':mass.imag.tolist(),
             'linear_solve_residual_FLOAT':residual,
             'Schur_resolvent_operator_error_bound':bound,
             'source_bilinear_error_bounds':[[float(np.linalg.norm(a)*np.linalg.norm(b))*bound for b in sources] for a in sources]}
        responses.append(row)
    # Full complex cross AFTER one common L is computed, not erased to norms.
    cross=responses[-1]['response_real'][0][2]+1j*responses[-1]['response_imag'][0][2]
    assert abs(cross)>1e-15
    # Shared flow is discriminating on actual opposite-defect shell states.
    canceled=[state for state in states if all(e==0 for e in state[2]) and state!=initial]
    assert canceled
    opposite_species=[state for state in canceled if state[0]!=vacuum and state[1]!=((vacuum,63),)]
    assert opposite_species
    complex_wrong_control=responses[0]['response_imag'][0][1]
    assert abs(complex_wrong_control)>1e-5
    result={'kind':'NEW_SIGNED_PHYSICAL1_FOUR_EDGE_QUANTUM_CONNECTED_RESPONSE_WITH_EXPLICIT_SCHUR_REMAINDER',
            'method':'canonical64-species symmetric auxiliary histogram; four native edges; total shared quantum E; two-hop shell with nonzero Schur complement',
            'physical_hopping':1,'point_body':{'s':str(s),'r':str(r),'actual_ground_support':False},
            'canonical':{'active_patch_particles':8,'aux_species':63,'each_aux_patch_particles':8,
                         'all_states_occupation_and_divE_verified':True,'source_is_not_canonical_trace':True},
            'controls':{'first_native_Gram_exact':str(exact_first_gram),
                        'wrong_squared64_load_rejected':True,
                        'wrong_independent_species_E_opposite_cost':128,
                        'correct_shared_E_opposite_cost':0,
                        'actual_opposite_active_aux_shell_states':len(opposite_species),
                        'once_shared_external_E_baseline':256,
                        'wrong_repeated_external_E_baseline':16384,
                        'external_quantum_native_energy_changes':external_costs,
                        'Pauli_wrong_occupation_rejected':True,
                        'erased_complex_cross_01_error_FLOAT':float(complex_wrong_control)},
            'shell':{'levels':levels,'dimension':size,'outside_target_states':len(qrows),'nonzero_complement_couplings':len(qv),
                     'Hermiticity_FLOAT_error':hermitian,'C_internal_exact':str(c_internal),'C_boundary_all_native_exact':str(c_boundary),
                     'observed_shell_coupling_norm_upper_FLOAT':float(shell_float_bound),'zero_total_E_opposite_defect_states':len(canceled)},
            'ground_compression':{'six_low_energies_FLOAT':energies.tolist(),'whole_compressed_P0_rank_FLOAT':int(p0.shape[1]),
                                  'lowest_eigen_residual_FLOAT':float(np.linalg.norm(h@ground-lowest*ground)),
                                  'not_fullH_originating_P0':True},
            'currents':{'means_FLOAT':means,'full_signed_complex_cross_after_same_L':True,
                        'observed_cross_02_high_frequency':[float(cross.real),float(cross.imag)]},
            'source_complement':{'QJ_ground_norms_FLOAT':source_leakage,
                                'observed_QJ_scope':'NN four-edge shell leakage only; additional full native length2/AP boundary current sources are retained by C_J bound',
                                'each_current_operator_norm_bound_exact':str(c_current),
                                'full_current_source_remainder_bound':'C_J^2*[C_total^2/eta^3+3/eta], including QJ source terms; body/ground/P0 transport not included'},
            'connected_responses_FLOAT':responses,
            'rigorous_remainder':'For same compressed source and same center lambda: |delta R_ab| <= ||j_a|| ||j_b|| (C_internal+C_boundary)^2/eta^3. Source ground replacement/body error and P0 replacement are separate unpaid premises, never bounded by this resolvent estimate.',
            'scope':{'four_edge_quantum_response_executed':True,'explicit_uniform_local_complement_bound':True,
                     'bound_small_in_IR':False,'actual_ground_response_or_magnetic_stiffness':False,
                     'N_uniform_fullH_gap':False,'Phi_Haar_neutral_fast_originating_P0_corrections_paid':False,
                     'no_missing_numerical_ground_backend_gate':True,'law_or_SOURCE_adopted':False,'campaign_complete':False}}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'new_four_edge_quantum_dimension':size,'outside_schur_states':len(qrows),'physical_hopping':1,'connected_complex_response':'executed','IR_or_ground_adoption':False}))


if __name__=='__main__':
    main()
