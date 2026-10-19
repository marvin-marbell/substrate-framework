"""Exact same-own-state supplemental gauge/units audit; no dynamics/producer replay."""
from itertools import product
import hashlib,json,sys,time,resource
from pathlib import Path
import numpy as np
from scipy import sparse
from rich_spatial_colour_charge_20261015_snapshot import load as load_body
from rich_spatial_colour_em_20261015_join import from_body,assemble,native_C
from rich_spatial_colour_em_20261015_model import observable,angular
from rich_spatial_colour_em_20261015_fields import profiles,gaussian_green
from rich_spatial_colour_em_20261015_density import spatial_current


def norm(a):return float(sparse.linalg.norm(a)) if sparse.issparse(a) else float(np.linalg.norm(a))
def mean(v,a):return float(np.vdot(v,a@v).real)


def main():
    inp,out=map(Path,sys.argv[1:3]);case=json.loads(inp.read_text());p=json.loads((inp.parent/case['owner_input']).read_text());start=time.monotonic();bodypath=inp.parent/p['native_body_cache']
    if hashlib.sha256(bodypath.read_bytes()).hexdigest()!=p['native_body_cache_sha256']:raise ValueError('Unpinned BODY')
    body=load_body(bodypath);model=from_body(body,p);data=np.load(inp.parent/case['own_states']);ref=data['neutral_reference'];field=model['field'];rotor=body['rotor'];nr=rotor['dimension'];ne=field['dimension'];h=assemble(model,'lepton');initial=data['lepton_raw_normalized_source'];gauge=[]
    for axis in range(3):
        position=np.einsum('qia,a->qi',rotor['R'],body['geometry']['displacement']/2)[:,axis];dipole=observable(model,'lepton',matter=body['operators']['lepton']['density_delta'],rq=np.kron(rotor['integrate'](position).toarray(),np.eye(ne)));current=spatial_current(model,'lepton',[0.,0.,0.],axis);hop=spatial_current(model,'lepton',[0.,0.,0.],axis,include_mechanical=False);comm=1j*(h@(dipole@initial)-dipole@(h@initial));gauge.append(dict(axis=axis,R3_constant_gradient_dipole_defect=norm(comm-current@initial),wrong_no_mechanical=norm(comm-hop@initial),mechanical_current_image_norm=norm((current-hop)@initial)))
    # Direct Gaussian-weight integration of the REAL electric metric.
    nodes,weights=np.polynomial.hermite.hermgauss(8);indices=list(product(range(8),repeat=3));x=np.array([[nodes[i],nodes[j],nodes[k]] for i,j,k in indices]);w=np.array([weights[i]*weights[j]*weights[k] for i,j,k in indices]);a=profiles(p['em_width']*x,p['em_width'],p['epsilon']);metric=p['epsilon']*p['em_width']**3*np.einsum('xai,xbi,x->ab',a,a,w*np.exp(np.sum(x*x,axis=1)))
    # Gaussian convolution is matched to the SAME site-density UV width.
    n,w1=np.polynomial.hermite.hermgauss(12);ix=list(product(range(12),repeat=3));u=np.array([[n[i],n[j],n[k]] for i,j,k in ix])*np.sqrt(2)*p['charge_sigma'];wu=np.array([w1[i]*w1[j]*w1[k] for i,j,k in ix])/np.pi**1.5;points=np.array([[.19,-.27,.33],[-.41,.12,.23]]);errors=[]
    for point in points:
        direct=np.einsum('x,xai->ai',wu,profiles(point[None,:]+u,p['em_width'],p['epsilon']));actual=profiles(point,p['em_width'],p['epsilon'],p['charge_sigma']);errors.append(norm(direct-actual))
    k=np.array([.31,.17,-.22]);rho=np.exp(-p['charge_sigma']**2*(k@k)/2);path_errors=[]
    for route in body['geometry']['quadrature']:
        line=rho*np.sum(np.exp(-1j*(route['points']@k))*(route['dl']@k));ends=body['geometry']['vertices'];target=rho*(np.exp(-1j*k@ends[0])-np.exp(-1j*k@ends[1]));path_errors.append(float(abs(1j*line-target)))
    sources={}
    for name,values in body['source_matter'].items():
        sourceQ=body['operators'][name]['Q'];neutralQ=body['operators']['neutral']['Q'];delta=float(sourceQ.diagonal().real.mean()-neutralQ.diagonal().real.mean());sources[name]=dict(N=float(body['operators'][name]['N'].diagonal().real.mean()),Q=delta,source_charge_intertwining_max=max(norm(sourceQ@s-s@neutralQ-delta*s) for s in values.values()),all_source_components=len(values))
    C={};c=native_C(model,'lepton');anti_h=assemble(model,'antilepton');mapped=c@initial;C['H_on_own_source']=norm(anti_h@mapped-c@(h@initial));C['current_odd_on_own_source']=[norm(assemble(model,'antilepton',derivative=(mu,))@mapped+c@(assemble(model,'lepton',derivative=(mu,))@initial)) for mu in range(6)]
    primary=json.loads((inp.parent/case['primary_json']).read_text());old=np.load(inp.parent/case['primary_npz']);published={}
    for name,key in [('lepton','0:0.5:0.5'),('baryon','0:0.5:0.5'),('baryon','0:1.5:0.5')]:
        pole=old[name+'_source_'+key+'_pole'];hp=assemble(model,name);energy=mean(pole,hp);published[name+':'+key]=dict(pole_energy=energy,gap=energy-primary['reference_energy'],normalized_pole_norm=norm(pole)**2,pole_eigen_residual=norm(hp@pole-energy*pole),physical_J2=sum(norm(angular(model,name,axis)@pole)**2 for axis in range(3)),reported_raw_weight=primary['sources'][name][key]['pole_raw_weight'],reported_retained_raw_strength=primary['sources'][name][key]['raw_retained_strength'])
    result=dict(kind='WAVE19_SAME_OWN_EXTENDED_STATE_PHYSICAL_GAUGE_AUDIT',state_sha256=hashlib.sha256((inp.parent/case['own_states']).read_bytes()).hexdigest(),primary_source_npz_sha256=hashlib.sha256((inp.parent/case['primary_npz']).read_bytes()).hexdigest(),Gaussian_electric_Gram_error=norm(metric-np.eye(6)),Gaussian_same_density_convolution_errors=errors,raw_same_Gaussian_path_continuity_errors=path_errors,V_self=model['V_self'],V_cross=model['V_cross'],R3_constant_gradient=gauge,sources=sources,actual_C=C,published_coupled_poles_verified_against_rebuilt_H=published,meaning='Supplemental projections on OWN new coupled reference/source and independent kernel quadratures. Published pole-state inspection is separately labelled, not a borrowed initial state for own physical event. Constant-gradient R3 identity is not T3 holonomy. Nonzero retained defects remain unpaid; no patched current.',resource=dict(seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['resource']))


if __name__=='__main__':main()
