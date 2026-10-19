"""New gauge-invariant spin/loop-source consumer, not unconstrained Peierls source."""
from pathlib import Path
from dataclasses import replace,asdict
import json,hashlib,time
import numpy as np
from rich_spinor_gauss_20261014_model import Parameters,Sector,retarded,charged_source,full_resolvent
ROOT=Path(__file__).resolve().parent

def encode(x):
    if isinstance(x,np.ndarray):return {'real':x.real.tolist(),'imag':x.imag.tolist()} if np.iscomplexobj(x) else x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def run():
    out={'discriminator':'literal gauge-invariant magnetic Wilson flux and physical loop source','cases':{},'source':'H(lambda)=H-lambda Fz; Fz=sinWilson(0,2,1)-sinWilson(0,3,1), equatorial triangles opposite signed z area; NOT Peierls translation of every link','units':'hbar=chargeabs=radius=1; lambda energy, Fz dimensionlesscompactflux','limitations':['Projected Wilson sin is finite-cutoff flux observable, not continuum B normalization.','Spin1/2,CAR,orientor inputs primitive; no electron g/field propagation inference.','Peierls geometric-current source and actual Wilson flux source are distinct consumers.']}
    all_e=None
    for name,p in [('electric_only',Parameters()),('reference',Parameters(flux_reference=.3)),('electric_only_no_transfer',Parameters(rotor_spin=0.))]:
        s=Sector(p,1,all_e);all_e=s.all_e;e,v=s.solve(6);g=v[:,1];sz=s.spin[2]
        f=s.wilson([0,2,1])-s.wilson([0,3,1]);m=s.magnetic(2)
        means={}
        for index in (0,1):
            u=v[:,index];sf=(sz@f+f@sz)/2
            means[str(index)]={'norm':float(np.vdot(u,u).real),'spin_z':float(np.vdot(u,sz@u).real),'flux_z':float(np.vdot(u,f@u).real),'geometric_current_Mz':float(np.vdot(u,m@u).real),'sym_spin_flux':float(np.vdot(u,sf@u).real),'connected_sym_spin_flux':float(np.vdot(u,sf@u).real-np.vdot(u,sz@u).real*np.vdot(u,f@u).real),'flux_variance':float(np.vdot(f@u,f@u).real-np.vdot(u,f@u).real**2)}
        responses={}
        for z in [.05+.02j,.8+.07j]:responses[str(z)]=retarded(s,g,e[1],[f,sz,m],z)
        physical=[];baseH=s.H.copy()
        for lam in [-.002,0.,.002]:
            s.H=baseH-lam*f;pe,pv=s.solve(6)
            # Never rotate distinct energies. Select actual lowest-pair eigenstate by Sz.
            spins=np.array([np.vdot(pv[:,i],sz@pv[:,i]).real for i in (0,1)]);i=int(np.argmax(spins));u=pv[:,i]
            physical.append({'lambda':lam,'selected_energy':float(pe[i]),'lowest_pair_energies':pe[:2],'norm':float(np.vdot(u,u).real),'spin_z':float(spins[i]),'flux_z':float(np.vdot(u,f@u).real),'source_slope_expected_minus_flux':float(-np.vdot(u,f@u).real),'geometric_current_Mz':float(np.vdot(u,m@u).real)})
        s.H=baseH
        data={'parameters':asdict(p),'dimension':s.dimension,'ground_means':means,'operator_order':['Fz','Sz','geometricCurrentMz'],'same_state_retarded':responses,'physical_loop_source':physical,'loop_source_flux_susceptibility_centered':(physical[2]['flux_z']-physical[0]['flux_z'])/.004,'loop_source_energy_slope_centered':(physical[2]['selected_energy']-physical[0]['selected_energy'])/.004,'original_ground_state_id':hashlib.sha256(baseH.data.tobytes()+g.tobytes()).hexdigest(),'constraints':s.constraint()}
        if name=='electric_only':
            plus=Sector(p,2,all_e);minus=Sector(p,0,all_e)
            adds=[];rems=[];metadata=[]
            for mode in range(12):
                a,ma=charged_source(s,plus,mode,True);b,mb=charged_source(s,minus,mode,False)
                adds.append(a@g);rems.append(b@g);metadata.append({'mode':mode,'addition':ma,'removal':mb})
            adds=np.column_stack(adds);rems=np.column_stack(rems);z=.5+.07j
            green=adds.conj().T@full_resolvent(plus.H,z+e[1],adds)+(rems.conj().T@full_resolvent(-minus.H,z-e[1],rems)).T
            data['same_ground_G12']=green;data['source_equal_time']=adds.conj().T@adds+(rems.conj().T@rems).T
            data['source_metadata']=metadata;data['same_ground_density_spin']=s.state_data(g)
            data['G_spin_up_down_difference']=float(np.linalg.norm(green[::2,::2]-green[1::2,1::2]))
            data['adjacent_sector_dimensions']={'N0':minus.dimension,'N1':s.dimension,'N2':plus.dimension}
            data['adjacent_constraints']={'N0':minus.constraint(),'N2':plus.constraint()}
        out['cases'][name]=data;print(name,json.dumps({'means':means,'source':physical,'susceptibility':data['loop_source_flux_susceptibility_centered']},default=encode),flush=True)
    (ROOT/'rich_spinor_gauss_20261014_flux_actual2_result.json').write_text(json.dumps(out,default=encode,indent=2))
if __name__=='__main__':run()
