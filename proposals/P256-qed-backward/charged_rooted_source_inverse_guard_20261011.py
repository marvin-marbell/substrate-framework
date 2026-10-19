from __future__ import annotations
import hashlib,json,sys,time
from fractions import Fraction as F
from pathlib import Path
sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent
INPUTS=['charged_output_lower_attempt3_20261011_receipt.json','charged_image_proof_control_20261011_receipt.json','charged_first_shell_actual04_20261011_receipt.json','local_so5_ground_upper_complete_state_consumer_20261011_receipt.json']
def digest(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def run()->None:
    start=time.monotonic();pins={n:digest(ROOT/n) for n in INPUTS};own=digest(Path(__file__))
    lower,image,old,upper=[json.loads((ROOT/n).read_bytes()) for n in INPUTS]
    assert lower['both_branches_certified'] and lower['both_u_signs_exact_unitary_equivalent']
    floors={b['branch']:F(b['floor_exact']) for b in lower['branches']}
    E0=F(upper['ground_energy_interval_exact'][1]);assert E0==F(lower['E0_upper_exact'])
    roles=old['all64_boundary']['roles'];assert sorted(r['count'] for r in roles)==[1,63]
    rows=[];q=F(0)
    for allocated in old['all64_boundary']['all64_once_shared_pointwise_norms']:
        axis,length=allocated['axis'],allocated['length'];weight=F(allocated['root_allocation_exact'])
        source_square=F(0);role_rows=[];minimum=None
        for role in roles:
            p=next(p for p in role['paths'] if p['axis']==axis and p['length']==length)
            matrix=[[(F(z[0]),F(z[1])) for z in line] for line in p['matrix_exact']]
            assert len(matrix)==4 and all(len(line)==4 for line in matrix)
            hs_square=sum(a*a+b*b for line in matrix for a,b in line)
            # Hermitian CAR hop dGamma([[0,M],[M*,0]]) has norm sum(singular M).
            # Its square <=rank(M)*Tr(M*M)<=4||M||HS^2, in finite CAR only.
            K2=4*hs_square
            branch='active' if role['count']==1 else 'aux'
            gap=64*p['external_edges_min']+2*floors[branch]-2*E0
            assert gap>0;minimum=gap if minimum is None else min(minimum,gap)
            source_square+=role['count']*K2
            role_rows.append({'role':role['role'],'count':role['count'],'full_Dirac_hop_norm_squared_upper_exact':str(K2),'charged_first_shell_gap_lower_exact':str(gap)})
        assert minimum is not None
        inverse_square=source_square/minimum**2;q+=weight*inverse_square
        rows.append({'axis':axis,'length':length,'allocation_exact':str(weight),'roles':role_rows,'source_norm_square_upper_exact':str(source_square),'inverse_source_norm_square_upper_exact':str(inverse_square),'inverse_source_norm_square_upper_float':float(inverse_square)})
    # Explicit full16-CAR example M=diag(1,2): onebody eigenvalues +/-1,+/-2,
    # fullFock norm3. Frobenius square5 alone is not a valid squared norm cap.
    masks=list(range(16));H=[[F(0) for _ in masks] for _ in masks]
    for j,mask in enumerate(masks):
        for p,r,w in [(0,2,1),(2,0,1),(1,3,2),(3,1,2)]:
            if mask>>r&1:
                sign=(-1)**((mask&((1<<r)-1)).bit_count());tail=mask^(1<<r)
                if not tail>>p&1:
                    sign*=(-1)**((tail&((1<<p)-1)).bit_count());H[tail|(1<<p)][j]+=sign*w
    v=[F(0)]*16
    for mask,sign in [(3,1),(6,-1),(9,1),(12,1)]:v[mask]=F(sign,2)
    assert sum(x*x for x in v)==1 and [sum(H[i][j]*v[j] for j in masks) for i in masks]==[3*x for x in v]
    assert 5<9<=2*5
    result={'kind':'CHARGED_FIRST_SHELL_SOURCE_WEIGHTED_INVERSE_BOUNDARY','scope':'Conditional frozen Clifford-Phi chosen matching-product of selected local grounds, actual1+aux63 and fullDirac endpoints. First-shell H0 charged branches only; no fullPhi ground, actual interacting ground or allorder linked theorem.',
      'charged_cell_lower_exact':{k:str(v) for k,v in floors.items()},'one_external_edge_two_cell_gap_lower_exact':str(64+floors['active']+floors['aux']-2*E0),'one_external_edge_two_cell_gap_lower_float':float(64+floors['active']+floors['aux']-2*E0),'paths':rows,
      'source_specific_copy_orthogonality':'Changed local canonical-copy number labels give orthogonal first-shell sources/resolvent blocks. SUM count*K_role^2 is a source bound, NOT sqrt(count) operator norm or field independence. Within each copy all Dirac/spin endpoint amplitudes remain coherent; the shared field is counted once.',
      'candidate_root_inverse_square_sum_exact':str(q),'candidate_root_inverse_square_sum_float':float(q),'candidate_guard_less_than_one':q<1,'candidate_guard_is_not_an_allorder_theorem':True,'old_nonpositive_free_envelope_replaced_by_charged_lower':True,
      'remaining_material_debt':'Normalized interacting root dressing with complete crosscopy higher-PH/electric exterior and overlapping shared-flow control. A positive charged first-shell gap alone does not control collective neutral or lattice modes.',
      'controls':{'fullCAR_rank_factor_wrongcase_rejected':True,'fullCAR_16dim_eigenstate_energy':3,'wrong_Frobenius_norm_square_cap':5,'actual_norm_square_lower':9,'valid_rank2_cap':10},
      'phase':False,'bulk_stiffness':False,'IRatom':False,'fullPhi_ground':False,'stationary_lattice_ground':False,'sharp_photon':False,'originalSOURCE':False,'campaign_complete':False,
      'input_sha256':pins,'own_source_sha256':own,'inputs_unchanged':all(digest(ROOT/n)==h for n,h in pins.items()),'source_unchanged':digest(Path(__file__))==own,'elapsed_seconds':time.monotonic()-start}
    (ROOT/'charged_rooted_source_inverse_guard_20261011_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_exact') and k!='paths'},indent=2))
if __name__=='__main__':run()
