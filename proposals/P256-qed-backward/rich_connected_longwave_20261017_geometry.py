"""NEW anchored16-site FCC tube. Fourier profiles OBSERVE, never truncate links."""
from collections import deque
from itertools import combinations,product
import json,sys
import numpy as np


def construct():
    a=np.sqrt(2.);period=np.array([4*a,a,a]);local=np.array([(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)])/np.sqrt(8.)
    positions=np.concatenate([local+[j*a,0,0]for j in range(4)]);links=[]
    for s,t in combinations(range(16),2):
        for image in product((-1,0,1),repeat=3):
            d=positions[t]+period*np.array(image)-positions[s]
            if abs(np.linalg.norm(d)-1)<1e-12:links.append({'start':s,'end':t,'image':list(image),'displacement':d.tolist(),'intercell':s//4!=t//4})
    incidence=np.zeros((16,len(links)),int);adj=[[]for _ in range(16)]
    for e,l in enumerate(links):
        s,t=l['start'],l['end'];incidence[s,e]=-1;incidence[t,e]=1;adj[s].append((t,e,1));adj[t].append((s,e,-1))
    lap=incidence@incidence.T;lp=np.linalg.pinv(lap.astype(float));transverse=np.eye(len(links))-incidence.T@lp@incidence
    nodes,weights=np.polynomial.legendre.leggauss(8);profiles=[];names=[]
    for harmonic,axis,trig in product((1,2),(1,2),('sin','cos')):
        q=2*np.pi*harmonic/period[0];row=[]
        for l in links:
            d=np.asarray(l['displacement']);points=positions[l['start']]+(nodes[:,None]+1)*d/2
            row.append(d[axis]*float(weights@(np.sin(q*points[:,0])if trig=='sin'else np.cos(q*points[:,0])))/2)
        profiles.append(row);names.append({'harmonic':harmonic,'q':[q,0,0],'axis':axis,'quadrature':trig})
    raw=np.asarray(profiles).T;tt=transverse@raw
    triangles={}
    for s in range(16):
        for t,e,sgn in adj[s]:
            for u,f,signf in adj[t]:
                if f==e:continue
                for v,g,signg in adj[u]:
                    if v!=s or g in(e,f):continue
                    word=[(e,sgn),(f,signf),(g,signg)];flow=np.zeros(len(links),np.int8)
                    for edge,sign in word:flow[edge]+=sign
                    delta=sum((sign*np.asarray(links[edge]['displacement'])for edge,sign in word),np.zeros(3))
                    if np.linalg.norm(delta)>1e-10:continue
                    key=min(flow.tobytes(),(-flow).tobytes());triangles.setdefault(key,word)
    windings=[]
    for start,axis in product(range(16),range(3)):
        initial=(start,0,0,0);goal=[start,0,0,0];goal[axis+1]=1;goal=tuple(goal);seen={initial:None};queue=deque([initial])
        while queue and goal not in seen:
            node=queue.popleft();v=node[0];im=np.array(node[1:])
            for t,e,sign in adj[v]:
                nxtim=im+sign*np.array(links[e]['image'])
                if np.max(abs(nxtim))>1:continue
                nxt=(t,*map(int,nxtim))
                if nxt not in seen:seen[nxt]=(node,e,sign);queue.append(nxt)
        if goal not in seen:raise ArithmeticError('Winding path not connected')
        word=[];n=goal
        while seen[n]is not None:
            previous,e,sign=seen[n];word.append((e,sign));n=previous
        windings.append({'start':start,'axis':axis,'word':list(reversed(word))})
    chords=[next(e for e,l in enumerate(links)if(l['start'],l['end'],l['image'])==(4*c,4*c+3,[0,0,0]))for c in range(4)];parent={0:None};tree=[];queue=deque([0])
    while queue:
        v=queue.popleft()
        for t,e,sign in adj[v]:
            if e in chords or t in parent:continue
            parent[t]=v;tree.append((v,t,e,sign));queue.append(t)
    subtrees=[]
    for _,child,_,_ in tree:
        members=[]
        for v in range(16):
            n=v
            while n is not None and n!=child:n=parent[n]
            if n==child:members.append(v)
        subtrees.append(members)
    particular=np.zeros((len(links),16),int)
    for (_,_,edge,sign),members in zip(tree,subtrees):particular[edge,members]=sign
    cotree=[e for e in range(len(links))if e not in{row[2]for row in tree}];cycles=np.eye(len(links),dtype=int)[:,cotree]-particular@incidence[:,cotree]
    uniform=np.array([l['displacement']for l in links])/period[None,:]
    return {'positions':positions.tolist(),'periods':period.tolist(),'links':links,'incidence':incidence.tolist(),'laplacian_pseudoinverse':lp.tolist(),'transverse_projector':transverse.tolist(),'probe_profiles_raw':raw.tolist(),'probe_profiles_cycle_projected':tt.tolist(),'probe_names':names,'probe_rank':int(np.linalg.matrix_rank(tt,tol=1e-10)),'cycle_dimension':len(links)-15,'triangle_words':list(triangles.values()),'winding_paths':windings,'colour_tree':tree,'colour_subtrees':subtrees,'colour_chords':chords,'colour_chord_starts':[4*c for c in range(4)],'colour_chord_ends':[4*c+3 for c in range(4)],'integer_particular_flow':particular.tolist(),'integer_cycle_matrix':cycles.tolist(),'cotree_links':cotree,'cycle_kinetic_metric':(cycles.T@cycles).tolist(),'charge_cycle_offset_metric':(cycles.T@particular).tolist(),'tree_charge_metric':(particular.T@particular).tolist(),'canonical_angle_period_lattice':'2pi Z81; Eint=integerParticular*Nint+integerCycle*m, m integer81. Charge-dependent orthogonal-transverse offsets NOT omitted.','uniform_profiles':uniform.tolist(),'canonical_Gauss_cycle_residual':int(np.max(abs(incidence@cycles))),'canonical_Gauss_particular_residual_on_Q0':int(np.max(abs((incidence@particular-np.eye(16,dtype=int))[:,1:]- (incidence@particular-np.eye(16,dtype=int))[:,:1]))),'anchored_sites_quantum_INTERNAL_R':True,'profiles_are_OBSERVERS_NOT_FIELD_CUT':True}


if __name__=='__main__':
    data=construct();open(sys.argv[1],'w').write(json.dumps(data,indent=2)+'\n');print(json.dumps({'sites':16,'links':len(data['links']),'intercell_links':sum(l['intercell']for l in data['links']),'cycles':data['cycle_dimension'],'triangle_words':len(data['triangle_words']),'probe_rank':data['probe_rank'],'winding_lengths':sorted(set(len(w['word'])for w in data['winding_paths']))}),flush=True)
