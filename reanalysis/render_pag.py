"""Render a PAG with correct endpoint marks.

Earlier figures were drawn with networkx's undirected renderer, which discarded
the edge marks entirely, so a graph in which FCIT had committed to a direction
looked identical to one in which it had not. This draws the three PAG endpoint
marks explicitly: a filled arrowhead, an open circle, or a bare tail.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow

def circle_layout(nodes, r=1.0):
    n=len(nodes); ang=np.linspace(np.pi/2, np.pi/2+2*np.pi, n, endpoint=False)
    return {v:(r*np.cos(a), r*np.sin(a)) for v,a in zip(nodes,ang)}

def _crossings(order, edges, pos):
    segs=[(pos[a],pos[b]) for a,_,b in edges if a in pos and b in pos]
    def ccw(P,Q,R): return (R[1]-P[1])*(Q[0]-P[0]) > (Q[1]-P[1])*(R[0]-P[0])
    n=0
    for i in range(len(segs)):
        for j in range(i+1,len(segs)):
            (A,B),(C,D)=segs[i],segs[j]
            if len({A,B,C,D})<4: continue
            if ccw(A,C,D)!=ccw(B,C,D) and ccw(A,B,C)!=ccw(A,B,D): n+=1
    return n

def best_circle_layout(nodes, edges, tries=4000, seed=3):
    """Circle layout with the node ordering chosen to minimise edge crossings."""
    import random
    rnd=random.Random(seed); nodes=list(nodes)
    best=None; bestn=None
    for t in range(tries):
        order=nodes[:] if t==0 else rnd.sample(nodes,len(nodes))
        pos=circle_layout(order)
        c=_crossings(order,edges,pos)
        if bestn is None or c<bestn: bestn, best = c, pos
        if bestn==0: break
    return best, bestn

def spring_layout(nodes, edges, seed=7):
    import networkx as nx
    G=nx.Graph(); G.add_nodes_from(nodes)
    G.add_edges_from([(a,b) for a,_,b in edges])
    P=nx.spring_layout(G, seed=seed, k=1.5, iterations=600)
    xs=[p[0] for p in P.values()]; ys=[p[1] for p in P.values()]
    cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2
    sc=1.05/max(max(abs(x-cx) for x in xs), max(abs(y-cy) for y in ys))
    return {v:((p[0]-cx)*sc,(p[1]-cy)*sc) for v,p in P.items()}

def draw(ax, edges, pos, title=None, highlight=(), nodefs=9):
    NR=0.115
    for a,mark,b in edges:
        if a not in pos or b not in pos: continue
        (x1,y1),(x2,y2)=pos[a],pos[b]
        dx,dy=x2-x1,y2-y1; L=np.hypot(dx,dy)
        if L==0: continue
        ux,uy=dx/L,dy/L
        sx,sy=x1+ux*NR, y1+uy*NR
        ex,ey=x2-ux*NR, y2-uy*NR
        key=tuple(sorted((a,b)))
        hot = key in highlight
        col='#B8860B' if hot else '#333333'
        lw = 2.2 if hot else 1.1
        ax.plot([sx,ex],[sy,ey],color=col,lw=lw,zorder=1,solid_capstyle='round')
        m1,m2 = mark[0], mark[-1]     # endpoint at a, endpoint at b
        for (px,py,vx,vy,m) in [(sx,sy,-ux,-uy,m1),(ex,ey,ux,uy,m2)]:
            if m=='o':
                ax.add_patch(Circle((px-vx*0.042,py-vy*0.042),0.040,fill=True,
                             fc='white',ec=col,lw=1.6,zorder=3))
            elif m in '<>':
                ax.add_patch(FancyArrow(px-vx*0.055,py-vy*0.055,vx*0.055,vy*0.055,
                             width=0.0,head_width=0.052,head_length=0.055,
                             fc=col,ec=col,length_includes_head=True,zorder=3))
    for v,(x,y) in pos.items():
        ax.add_patch(Circle((x,y),NR,fc='#EEF3F7',ec='#33475B',lw=1.2,zorder=4))
        ax.text(x,y,v,ha='center',va='center',fontsize=nodefs,zorder=5,
                color='#12222F',linespacing=0.9)
    if title: ax.set_title(title,fontsize=12,pad=8)
    ax.set_xlim(-1.42,1.42); ax.set_ylim(-1.42,1.42); ax.set_aspect('equal'); ax.axis('off')

def parse(lines):
    E=[]
    for ln in lines:
        p=ln.strip().split()
        if len(p)>=4 and p[0].rstrip('.').isdigit() and set(p[2])<=set('<->o-'):
            E.append((p[1],p[2],p[3]))
        elif len(p)==3 and set(p[1])<=set('<->o-'):
            E.append(tuple(p))
    return E
