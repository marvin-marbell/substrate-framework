"""Required physical LEFT/RIGHT/source/CAR evidence on the actual new basis."""
from itertools import product
import numpy as np
from scipy import sparse
from rich_common_carrier_motion_20261015_model import angular,source_matrix,prepared_reference,fibre,source,NROOT,mode,string,extra_operator
from rich_common_frame_orbit_20261015_frame import gauss_frame,coordinate_noether,radial_form_readout
from rich_spatial_colour_charge_20261015_model import insert


def audit(model):
    result={'frame_irreps':{},'actual_source_tensor_covariance':{},'actual_charge_source_selection':{},'raw_full_prepared_CAR_strengths':{},'signed_retained_sources':{}}
    js=sorted(set(j for s in model['sectors'].values() for j in s['spin']['j']))
    result['bounded_new_cell_siblings']={}
    for name,s in model['sectors'].items():
        op=s['op'];sx,sy,sz=op['S'];a0,ax,ay,az=op['hops'][2][0][1];qcell=op['rho'][2]+op['rho'][3]
        result['bounded_new_cell_siblings'][name]={'bridge_scalar_Sz_commutator':float(sparse.linalg.norm(sz@a0-a0@sz)),'bridge_vector_z_x_equals_iy':float(sparse.linalg.norm(sz@ax-ax@sz-1j*ay)),'bridge_vector_z_y_equals_minus_ix':float(sparse.linalg.norm(sz@ay-ay@sz+1j*ax)),'bridge_new_cell_charge_shift':float(sparse.linalg.norm(qcell@a0-a0@qcell+a0)),'total_Q_scalar_identity_error':float(sparse.linalg.norm(op['Q']-(18-s['total_N'])*sparse.eye(s['dimension']))),'new_cell_charge_Sz_commutator':float(sparse.linalg.norm(qcell@sz-sz@qcell))}
    field=model['field'];errors=[]
    for family,i,j in product(range(2),range(3),range(3)):
        expected=sum((1j*np.cross(np.eye(3)[i],np.eye(3)[j])[k]*field['Q'][3*family+k] for k in range(3)),np.zeros_like(field['Q'][0],complex))
        errors.append(float(np.linalg.norm(field['F'][i]@field['Q'][3*family+j]-field['Q'][3*family+j]@field['F'][i]-expected)))
    result['actual_field_Q_vector_covariance_max_Frobenius']=max(errors)
    for j0 in js:
        j=float(j0);d=int(round(2*j+1));sp={'labels':[(0,j,float(m)) for m in np.arange(-j,j+1)],'j':np.full(d,j),'m':np.arange(-j,j+1),'multiplets':[{'id':0,'j':j,'start':0,'stop':d}],'S':[sparse.csr_matrix(a) for a in angular(j)]}
        fr=gauss_frame(sp,[j]);sector={'spin':sp,'frame':fr};result['frame_irreps'][str(j)]={'RIGHT_Gauss':fr['Gauss'],'isometry':fr['isometry'],'actual_LEFT_equals_reduced':fr['left_equals_reduced'],'coordinate':coordinate_noether(sector,int(round(2*j))),'multiplicity_meaning':'One coordinate irrep is checked once and tensor-repeated over ALL actual new native multiplicities, not primitive CAR assigned physical spin'}
    neutral=model['sectors']['neutral'];zero=prepared_reference(model);v0=fibre(model,'neutral',0.,0.)
    for name in ['lepton','antilepton']:
        one=model['sectors'][name];sp0=neutral['op']['S'];sp1=one['op']['S'];plus0=sp0[0]+1j*sp0[1];plus1=sp1[0]+1j*sp1[1]
        result['actual_source_tensor_covariance'][name]={};result['actual_charge_source_selection'][name]={};result['signed_retained_sources'][name]={}
        for site in range(4):
            up=source_matrix(model,name,site,.5);dn=source_matrix(model,name,site,-.5)
            result['actual_source_tensor_covariance'][name][str(site)]={'Sz_up':float(sparse.linalg.norm(sp1[2]@up-up@sp0[2]-.5*up)),'Sz_down':float(sparse.linalg.norm(sp1[2]@dn-dn@sp0[2]+.5*dn)),'raising':float(sparse.linalg.norm(plus1@dn-dn@plus0-up)),'raising_top':float(sparse.linalg.norm(plus1@up-up@plus0))}
            charge=-1 if name=='lepton' else 1
            result['actual_charge_source_selection'][name][str(site)]=float(sparse.linalg.norm(one['op']['Q']@up-up@neutral['op']['Q']-charge*up))
            result['signed_retained_sources'][name][str(site)]={str(m):{'norm_squared':float(np.linalg.norm(source(model,name,site,m)@zero)**2),'Jhalf_projection_leakage':float(np.linalg.norm((source(model,name,site,m)@zero)-fibre(model,name,.5,m)@(fibre(model,name,.5,m).conj().T@(source(model,name,site,m)@zero))))} for m in [-.5,.5]}
    sea=sum(1<<mode(f,1,s,c) for f,s,c in product(range(2),range(2),range(4)));vac=np.zeros(19,complex);vac[0]=1
    for site,spin in product(range(2),repeat=2):
        values={}
        for creation in [True,False]:
            terms=[([(mode(1,m,spin,3),creation)],(-1)**(site*m)/np.sqrt(2)) for m in range(2)]
            packet=insert({sea:vac},terms);values['creation' if creation else 'removal']=sum(float(np.vdot(v,v).real) for v in packet.values())
        values['sum']=sum(values.values());result['raw_full_prepared_CAR_strengths'][f'root_site{site}_spin{spin}']=values
    ext=np.zeros(6,complex);lookup={m:i for i,m in enumerate(model['masks'][2])}
    for i,j in product(range(2),repeat=2):
        step=string(0,[(2*j+1,True),(2*i,True)]);ext[lookup[step[0]]]+=(-1)**(i+j)*step[1]/2
    for site,spin in product(range(2),repeat=2):
        plus=extra_operator(model['masks'][3],model['masks'][2],[([(2*site+spin,True)],1.)])@ext
        minus=extra_operator(model['masks'][1],model['masks'][2],[([(2*site+spin,False)],1.)])@ext
        result['raw_full_prepared_CAR_strengths'][f'added_site{site+2}_spin{spin}']={'creation':float(np.linalg.norm(plus)**2),'removal':float(np.linalg.norm(minus)**2),'sum':float(np.linalg.norm(plus)**2+np.linalg.norm(minus)**2)}
    maxerror=0.;wrong=0.;samples=0
    for root,n in NROOT.items():
        packet=model['data']['native'][root].seeds[0]
        for mask,extra,spin in product(packet,range(16),range(2)):
            for band in range(2):
                index=mode(1,band,spin,3);r=string(mask,[(index,False)]);e=string(extra,[(spin,True)])
                direct=string(mask|(extra<<32),[(32+spin,True),(index,False)])
                if r is None or e is None:
                    if direct is not None:raise ValueError('Raw CAR36 bridge support mismatch')
                    continue
                expected=r[1]*e[1]*(-1)**(n-1);maxerror=max(maxerror,abs(direct[1]-expected));wrong=max(wrong,abs(direct[1]-r[1]*e[1]));samples+=1
    result['direct_CAR36_bridge_grade']={'samples':samples,'correct_sign_error':maxerror,'wrong_ungraded_error':wrong};result['radial']=[radial_form_readout(model['radial'],float(j)) for j in js]
    return result
