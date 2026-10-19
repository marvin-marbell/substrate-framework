"""Actual V64 canonical-CAR/direct-det and nodal consumers; no producer outputs."""
import json
import numpy as np
from rich_review_wave35_20261016_consume import ROOT,P,POS,ARCS,B,occupations,amp,contract,logguide,declared,sources,car


def direct(up,down,E,F,params,coeff):
    occ=occupations(up,down);bits=sum(1<<(2*int(v)+s) for s in range(2) for v in np.nonzero(occ[s])[0])
    psi=amp(up,down,E,F,params);rho=occ.sum(0)-1;h=(2*(rho@rho)+(E@E)/32)*psi
    hjunk=h;j=np.zeros((len(coeff),2),complex);contact=np.zeros((len(coeff),2,2),complex)
    neighbors=[]
    for e,(aa,bb,_) in enumerate(ARCS):
        for direction in (-1,1):
            if abs(E[e]+direction)>1:continue
            a,b=(aa,bb) if direction==1 else (bb,aa)
            for spin in (0,1):
                _,carh=car(bits,2*a+spin,2*b+spin)
                if not carh:continue
                u=up.copy();d=down.copy();o=u if spin==0 else d;o[o==a]=b;o.sort();field=E.copy();field[e]+=direction
                other=amp(u,d,field,F,params);hv=carh*other;h+=hv;hjunk-=other
                for k,c in enumerate(coeff):
                    j[k]-=1j*direction*c[:,e]*hv
                    contact[k]-=np.outer(c[:,e].conj(),c[:,e])*hv*psi
                neighbors.append((u,d,field))
    return h,j,contact,hjunk,neighbors


def main():
    config=json.loads((ROOT/(P+'input.json')).read_text());params=config['params'];F,_,_=declared.kernel(**config['kernel'])
    coeff=np.array([sources(4,q)[0] for q in config['q']]);up=np.nonzero(POS.sum(1)%2==0)[0];down=np.setdiff1d(np.arange(64),up);E=np.zeros(192,int)
    rng=np.random.default_rng(3500901);rows=[]
    for k in range(8):
        x=contract(up,down,E,F,params,coeff);h,j,c,bad,neighbors=direct(up,down,E,F,params,coeff)
        scale=max(abs(h),abs(x['psi']),1e-100);jscale=max(np.max(abs(j)),1e-100)
        row=dict(H_relative=abs(h-x['Hpsi'])/scale,J_relative=float(np.max(abs(j-x['J']))/jscale),contact_relative=float(np.max(abs(c-x['contact']))/max(np.max(abs(c)),1e-100)),wrong_current_relative=float(np.max(abs(j+x['J']))/jscale),noJW_H_relative=abs(bad-h)/scale,gauss=x['gauss'],HF_relative=float(np.max(abs(x['HF']-x['HF_direct']))/scale))
        assert row['gauss']==0 and row['H_relative']<1e-9 and row['J_relative']<1e-9 and row['contact_relative']<1e-9 and row['HF_relative']<1e-9
        rows.append(row)
        for _ in range(6):
            _,_,_,_,neighbors=direct(up,down,E,F,params,coeff)
            up,down,E=neighbors[rng.integers(len(neighbors))]
    assert any(r['wrong_current_relative']>1 for r in rows)
    # Exact symmetric dimer pair amplitude: Neel coefficient nonzero, its one-hop state
    # is an exact determinant node. This is an operator-control input, NOT selected vacuum.
    u=np.nonzero(POS.sum(1)%2==0)[0];d=np.setdiff1d(np.arange(64),u);field=np.zeros(192,int);D=np.zeros((64,64))
    for a in u:
        b=ARCS[3*a,1];D[a,b]=1;D[b,a]=1
    a=int(u[0]);b=int(ARCS[3*a,1]);nu=u.copy();nu[nu==a]=b;nu.sort();field[3*a]=1
    node=contract(nu,d,field,D,params,coeff);node_direct=direct(nu,d,field,D,params,coeff)[0]
    guide=np.exp(logguide(nu,d,field,D,D,params,.2,1e-12))
    retained_H2=guide*(abs(node['Hpsi'])**2/guide)
    lost_if_null_guide=abs(node['Hpsi'])**2
    assert node['psi']==0 and node['gauss']==0 and not node['stable']
    assert abs(node_direct)>0 and abs(node['Hpsi']-node_direct)<1e-10*abs(node_direct)
    assert guide>0 and retained_H2>0 and lost_if_null_guide>0
    result=dict(kind='ACTUAL_V64_OPERATOR_AND_NODE_POSITIVE_CONTROLS_NOT_SELECTED_STATE_SAMPLING',selected_kernel=config['kernel'],direct_determinant_controls=rows,node_control=dict(psi=node['psi'],Hpsi=node['Hpsi'],direct_Hpsi=node_direct,positive_guide=guide,retained_H2=retained_H2,omitting_node_loses_H2=lost_if_null_guide,gauss=node['gauss'],stable_inverse_used=node['stable']),noJW_detected=any(r['noJW_H_relative']>.01 for r in rows))
    assert result['noJW_detected']
    np.savez_compressed(ROOT/(P+'controls_raw.npz'),F=F,nodal_F=D,node_up=nu,node_down=d,node_E=field)
    encoded=json.dumps(result,indent=2,default=lambda x:x.item())
    (ROOT/(P+'controls_findings.json')).write_text(encoded+'\n');print(encoded,flush=True)

if __name__=='__main__':main()
