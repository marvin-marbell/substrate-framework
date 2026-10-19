"""Material 3D matter connectivity, retaining all64 supplier configurations.

The planar predecessor lacks supplier induction for one polarization. This
helical four-rung geometry has transverse offsets in both y,z and all four
actual inter-rung nearest-cell paths. Every physical profile phase, source
cloud, ion position, density/current and longitudinal energy is rebuilt.
"""
import numpy as np
from scipy import sparse as sp
from rich_field_packet_20261014_model import Packet, supplier, line_profile


def helical_supplier(p):
    c=supplier(p)
    sites=[]
    for rung in range(4):
        angle=rung*np.pi/2+.23
        for sign in (-1.,1.):
            sites.append([(rung-1.5)*p.supplier_spacing,sign*.65*np.cos(angle),sign*.65*np.sin(angle)])
    sites=np.array(sites)
    edges=[(i,i+1) for i in range(0,8,2)]
    edges += [(2*rung+a,2*(rung+1)+b) for rung in range(3) for a in range(2) for b in range(2)]
    bits=c['bits'];lookup={b:i for i,b in enumerate(bits)};hops=[]
    for i,j in edges:
        rows=[];cols=[];values=[]
        for column,state in enumerate(bits):
            for flavour in range(2):
                a,b=i+8*flavour,j+8*flavour
                if state&(1<<a) and not state&(1<<b):
                    intermediate=state^(1<<a)
                    sign=(-1)**((state&((1<<a)-1)).bit_count()+(intermediate&((1<<b)-1)).bit_count())
                    rows.append(lookup[intermediate|(1<<b)]);cols.append(column);values.append(sign)
        hops.append(sp.csr_matrix((values,(rows,cols)),shape=(64,64),dtype=complex))
    c.update(sites=sites,edges=edges,hops=hops,phase=np.array([line_profile(sites[i],sites[j],p,p.base.supplier_width) for i,j in edges]))
    return c


class ConnectedPacket(Packet):
    def __init__(self,p,occupation=1,**options):
        super().__init__(p,occupation,**options)
        self.cube=helical_supplier(p)
        self.HL_values,self.charge_data=self.longitudinal()
        self.HL=sp.kron(self.If,sp.diags(self.HL_values.ravel()),format='csr')
        self.H=(self.E+self.HL+self.Hdiag+self.Hbody+self.link_derivative(())).tocsr()
        self.J=[(-self.link_derivative((a,))+self.body_current(a)).tocsr() for a in range(self.nmodes)]
