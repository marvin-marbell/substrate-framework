"""Canonical log-source generating functional, not normalized sector subtraction.
Z=Tr exp(i sum eta_a J_a) Utheta [rho0+s rhoB+t rhoL+st rhoBL] Utheta†.
Physical mean=(1/i)d_eta logZ; Weyl covariance=-d_eta_a d_eta_b logZ.
"""
import numpy as np
from scipy import sparse
from rich_composite_geometry_20261015_science import load
from rich_common_native_species_20261016_analysis import native,prepare,source_pair

SOURCE_POWERS={'0':(0,0),'B':(1,0),'L':(0,1),'BL':(1,1)}


def multiply(left,right):
    result={}
    for(a,b),value in left.items():
        for(c,d),other in right.items():
            if a+c<=1 and b+d<=1:result[a+c,b+d]=result.get((a+c,b+d),0.)+value*other
    return result


def polynomial(values):return {SOURCE_POWERS[key]:value for key,value in values.items()}


def inverse_trace(norms):
    n0,nB,nL,nBL=(norms[key]for key in ('0','B','L','BL'))
    return {(0,0):1/n0,(1,0):-nB/n0**2,(0,1):-nL/n0**2,(1,1):2*nB*nL/n0**3-nBL/n0**2}


def extract(raw):
    names=raw['channels'];gram=raw['Gram'];vertex=[i for i,name in enumerate(names)if name.startswith('vertex_')];source=[i for i,name in enumerate(names)if name.startswith('source_delta_')];receiver=[i for i,name in enumerate(names)if name.startswith('receiver_')];current=[i for i,name in enumerate(names)if name.startswith('source_current_')];contact=[i for i,name in enumerate(names)if name.startswith('source_contact_')]
    vgroups=[[names.index('vertex_00')]]+[[names.index(f'vertex_{i}{j}')for i in range(1,4)]for j in range(1,4)];mean=np.array([gram[0,i]for i in receiver]);vmean=np.array([sum(gram[0,i]for i in group)for group in vgroups]);dressed=np.array([[sum(gram[receiver[a],i]for i in vgroups[b])for b in range(4)]for a in range(4)])
    vnorm=float(np.sum(gram[np.ix_(vertex,vertex)]).real);vexpect=sum(vmean);trace=raw['raw_trace'];elastic=abs(vexpect)**2/trace
    return {'raw_trace':trace,'current_mean':mean,'vertex_mean':vmean,'current_dressed_vertex':dressed,'vertex_norm2':vnorm,'vertex_elastic_norm2':float(elastic),'vertex_outside_single_source_norm2_signed':float(vnorm-elastic),'vertex_R_above_j1_norm2_signed':float(np.sum(raw['R_complement_Gram'][np.ix_(vertex,vertex)]).real),'vertex_new_raw_material_support_norm2':float(np.sum(raw['new_raw_support_Gram'][np.ix_(vertex,vertex)]).real),'active_source_DeltaH_norm2':float(np.sum(gram[np.ix_(source,source)]).real),'owning_source_current_norm2':float(np.sum(gram[np.ix_(current,current)]).real),'owning_source_contact_norm2':float(np.sum(gram[np.ix_(contact,contact)]).real),'vertex_Hermiticity_expectation_imag':float(abs(vexpect.imag)),'retarded_total_current_u2':float(-.5*vexpect.real)}


def connected(raw_states):
    extracted={name:extract(value)for name,value in raw_states.items()};norms={name:value['raw_trace']for name,value in extracted.items()};inverse=inverse_trace(norms);inverse2=multiply(inverse,inverse);mean_coefficient=np.zeros(4);covariance_coefficient=np.zeros((4,4))
    def source_coefficient(values,denominator):return multiply(polynomial(values),denominator).get((1,1),0.)
    for a in range(4):
        mean_coefficient[a]=-.5*np.real(source_coefficient({name:value['vertex_mean'][a]for name,value in extracted.items()},inverse))
        for b in range(4):
            sym={name:np.real(value['current_dressed_vertex'][a,b]+value['current_dressed_vertex'][b,a])for name,value in extracted.items()};first=-.5*source_coefficient(sym,inverse)
            ma=polynomial({name:value['current_mean'][a]for name,value in extracted.items()});mb=polynomial({name:value['current_mean'][b]for name,value in extracted.items()});va=polynomial({name:value['vertex_mean'][a]for name,value in extracted.items()});vb=polynomial({name:value['vertex_mean'][b]for name,value in extracted.items()});cross=multiply(multiply(ma,vb),inverse2).get((1,1),0.)+multiply(multiply(mb,va),inverse2).get((1,1),0.);covariance_coefficient[a,b]=np.real(first+.5*cross)
    weights={name:multiply({SOURCE_POWERS[name]:1.},inverse).get((1,1),0.)for name in SOURCE_POWERS}
    return {'raw':extracted,'raw_source_traces':norms,'canonical_linear_trace_weights':weights,'connected_mean_u2_components':mean_coefficient,'connected_mean_u2_total':float(np.sum(mean_coefficient)),'connected_Weyl_current_covariance_u2_tensor':covariance_coefficient,'connected_Weyl_current_variance_u2_total':float(np.sum(covariance_coefficient)),'definition':'Mixed s,t coefficient of logZ at eta0: Cmean=[st]M/n; pulse difference coefficient=-1/2[st]V/n. Ccov_u2_ab=-1/2[st](Re<J_a V_b>+Re<J_b V_a>)/n +1/2[st](M_a V_b+M_b V_a)/n². All original traces/partitions kept; NOT naive normalizedR36-R35-R33+R32 or amputatedS-matrix.','scope':'Actual LEADING retarded moments,not an integrated finite-u trajectory. Eta-vector resolves physical current scalar/vector components; tensor sum gives total-current variance coefficient. Offdiagonal complex outputGram retained separately; scalar/vector interference NEVER replaced by sum of norms.'}


def native_comparison(config,scalar):
    metadata=load(config['native_metadata_snapshot']);labels=metadata['labels'];operators={name:sparse.load_npz(config['native_'+name+'_snapshot'])for name in ('Gram','rotation','electric','plaquette','quark_vector','white_vector','quark_scalar','white_scalar','loopB','loopC')};space=native(operators,labels,36,scalar);seed=next(probe for probe in metadata['probes']if probe['descriptor']['family']=='bare'and probe['descriptor']['B_site']==0 and probe['descriptor']['L_site']==2);state,strength=prepare(seed,space);result={};tensor=state.reshape(-1,7)
    def action(pair,value):return (space['inverse']@(pair[0]@value+pair[1]@(value*scalar['grid'][None,:])))
    for direction in ('white','quark'):
        other='quark'if direction=='white'else'white';edge=2 if direction=='white'else 0;other_edge=0 if other=='quark'else 2;a=source_pair(space,direction,edge,0,config['theta']);b=source_pair(space,other,other_edge,1);zero=sparse.csr_matrix(a[0].shape,dtype=complex);parts=[]
        for ai,bi in ((0,0),(0,1),(1,0),(1,1)):
            aa=(a[0]if ai==0 else zero,a[1]if ai==1 else zero);bb=(b[0]if bi==0 else zero,b[1]if bi==1 else zero);parts.append(action(aa,action(bb,tensor))-action(bb,action(aa,tensor)))
        gram=np.array([[np.vdot(left,space['gram']@right)for right in parts]for left in parts]);comm=sum(parts,np.zeros_like(tensor));slope=1j*np.vdot(tensor,space['gram']@comm);result[direction]={'PAP_first_slope':float(slope.real),'PAP_first_commutator_norm2':float(np.sum(gram).real),'parts_Gram':gram,'parts_order':['VV','VS','SV','SS'],'canonical_joint_retained_strength':strength,'meaning':'SEPARATE original nativeR10+sourcebodyP finite matrix model,actual weak A Ginv B products. Same unmaskedB0L2 source independently reconstructed; no othersector native answers enter primitiveconnectedfunctional. NonzeroSS identifies body/sourceP because fullPhi multiplication commutes.'}
    return result
