"""Selected one-edge connected consumer after circulation reflection-null result.

This is a new probe, not repeated operational evolution. The circulation with
source/sink and first bridge all on y=z=0 is reflection-odd. Keep that null as
observed; choose an actual one-edge physical field on the intervened face.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable
import zipfile
import numpy as np
from scipy import sparse
from rich_open_gauss_carrier_20261015_model import NeutralCube
from rich_review_wave28_20261015_consume import Polynomial, evolve, variant, z

ROOT = Path(__file__).resolve().parent
P = 'rich_review_wave28_20261015_'
PYTHON = '/home/administrator/substrate-framework/.venv/bin/python'


def selected(n: Any, c: Any, omega: Any, source: Any, sink: Any, times: list[float],
             cut: bool, prop: Callable[..., Any]) -> dict[str, Any]:
    _,t1,t2,t3 = times
    hn,_,jn,_ = variant(n,cut)
    hc,_,jc,_ = variant(c,cut)
    field_edge,bridge = int(c.right_edges[0]),int(c.bridges[0])
    fn = sparse.diags(n.electric[:,field_edge].astype(float)/16,format='csr')
    fc = sparse.diags(c.electric[:,field_edge].astype(float)/16,format='csr')
    qn,qc = -jn[bridge],-jc[bridge]
    charged1 = prop(hc,source @ omega,t1)
    charged2 = prop(hc,np.column_stack((charged1,fc @ charged1)),t2-t1)
    charged3 = prop(hc,np.column_stack((charged2[:,0],qc @ charged2[:,0],charged2[:,1],qc @ charged2[:,1])),t3-t2)
    bra = prop(hn,omega,t3)
    g,gj,gf,raw = [np.vdot(bra,sink.T @ charged3[:,k]) for k in range(4)]
    neutral1 = prop(hn,omega,t1)
    neutral2 = prop(hn,neutral1,t2-t1)
    f = np.vdot(neutral1,fn @ neutral1)
    j = np.vdot(neutral2,qn @ neutral2)
    jf = np.vdot(neutral2,qn @ prop(hn,fn @ neutral1,t2-t1))
    subtraction = gj*f+gf*j+g*jf-2*g*j*f
    return dict(complete=z(raw),connected=z(raw-subtraction),subtraction=z(subtraction),
                BBdag=z(g),BJBdag=z(gj),BFBdag=z(gf),background_F=z(f),background_J=z(j),background_JF=z(jf),
                times=times,field_edge=field_edge,field_edge_endpoints=c.edges[field_edge],
                physical_current_edge=bridge,current_edge_endpoints=c.edges[bridge],
                operator='B_4(t3) [-J_bridge(t2)] [E_right_edge(t1)/16] B_0†(0)',
                cumulant='raw-GJ*f-GF*j-G*JF+2*G*j*f; all charge-allowed disconnected partitions retained',
                source='raw positive-frequency addition B† from declared Ω; noncanonical Wilson strengths retained',
                meaning='Complete ordered unamputated field-times-charge-current amplitude, no -i or timeordering; not probability, irreducible vertex or on-shell scattering',
                cut_semantics='Actual cut active-edge current: bridge J=0. Neutral Ω evolved with cut H on both sides; cut BBdag need not vanish.')


def worker() -> None:
    start = time.monotonic()
    settings = json.loads((ROOT/(P+'input.json')).read_text())
    for key in ('model','reference'):
        if hashlib.sha256((ROOT/settings[key]).read_bytes()).hexdigest()!=settings[key+'_SHA256']:
            raise ValueError('Frozen input mismatch')
    n,c = NeutralCube(),NeutralCube(5,4)
    data = np.load(ROOT/settings['reference'])
    if str(data['basis_SHA256'])!=n.basis_hash or not np.array_equal(data['electric'],n.electric):
        raise ValueError('Neutral basis mismatch')
    omega = data['psi'].astype(complex)
    source = n.charged_source(c,0,0,True)
    sink = n.charged_source(c,4,0,True)
    poly = Polynomial(settings['chebyshev_degree'])
    result: dict[str, Any] = {}
    for name,cut in (('full',False),('cut',True)):
        primary = selected(n,c,omega,source,sink,settings['correlation_times'],cut,evolve)
        alternate = selected(n,c,omega,source,sink,settings['correlation_times'],cut,poly)
        result[name] = dict(**primary,numerical_resolution=dict(alternate=alternate,
            complete_abs_difference=float(abs(complex(*primary['complete'])-complex(*alternate['complete']))),
            connected_abs_difference=float(abs(complex(*primary['connected'])-complex(*alternate['connected'])))))
    result['selection'] = dict(reason='First measured circulation fourpoint was reflection-null (~2e-17): y↔z flips right circulation while source0,sink4,first x bridge remain fixed. That null is retained, not a no-interaction result. This selected actual one-edge E/16 probe breaks that measurement symmetry without changing H, Ω, source, times or operational pulse.',
                               initial_circulation_result=P+'result.json',selection_before_new_evolution=True,
                               no_borrowed_outputs=True,scope='One selected one-edge complete correlator; no profile catalogue or rerun of operational experiment')
    result['numerical_method'] = dict(degree=poly.degree,enclosures=list(poly.enclosures.values()),max_Bessel_tail_500_terms=poly.max_tail)
    result['elapsed_seconds'] = time.monotonic()-start
    (ROOT/(P+'selected_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main() -> None:
    inputs = [P+'selected.py',P+'consume.py',P+'input.json',
              'rich_open_gauss_carrier_20261015_model.py','rich_open_gauss_carrier_20261015_neutral_reference.npz',
              'rich_photon_fullq_20261014_model.py','rich_photon_20261014_loop.py']
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs}
    command = [PYTHON,P+'selected.py','--worker']
    with zipfile.ZipFile(ROOT/(P+'selected_source_capsule.zip'),'w',zipfile.ZIP_DEFLATED) as cap:
        for name in inputs:
            cap.write(ROOT/name,name)
    start = time.monotonic()
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (ROOT/(P+'selected_stdout.txt')).open('w') as out,(ROOT/(P+'selected_stderr.txt')).open('w') as err:
        done = subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
    receipt = dict(command=command,returncode=done.returncode,natural_exit=done.returncode>=0,elapsed_seconds=time.monotonic()-start,
                   input_SHA256=hashes,sources_and_inputs_unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in hashes.items()))
    (ROOT/(P+'selected_exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    raise SystemExit(done.returncode)


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else main()
