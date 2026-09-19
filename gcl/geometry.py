"""Finite simplicial geometry and reversible coordinate operations."""
import copy
import itertools
import math
import numpy as np
import torch
from scipy.optimize import linprog

class Complex:
    def __init__(self, vertices, maximal, reference=None):
        self.vertices = copy.deepcopy(vertices)
        self.maximal = [tuple(sorted(s)) for s in maximal]
        self.reference = list(reference or [0., 1.])
        self.clean()

    @classmethod
    def base(cls):
        a = math.sqrt(3)/2; h = math.sqrt(2/3)
        return cls([[0,0,0],[1,0,0],[.5,a,0],[.5,-a,0],
                    [.5,a/3,h],[.5,a/3,-h]], [(0,1,2,4),(0,1,2,5),(0,1,3)])

    def clean(self):
        unique = sorted(set(self.maximal))
        self.maximal = [s for s in unique if not any(set(s)<set(t) for t in unique)]

    def tensor(self):
        return torch.tensor(self.vertices, dtype=torch.float64)

    def faces(self):
        return sorted({f for s in self.maximal for n in range(1,len(s)+1)
                       for f in itertools.combinations(s,n)}, key=lambda s:(len(s),s))

    def edges(self):
        return [s for s in self.faces() if len(s)==2]

    def state(self):
        return {'vertices':copy.deepcopy(self.vertices),'maximal':self.maximal,
                'reference':self.reference.copy()}

    def expand_reference(self, q):
        if not all(math.isfinite(v) for v in q):
            raise ValueError('Finite reference values required')
        self.reference = [min(self.reference[0],*q),max(self.reference[1],*q)]

    def validate(self):
        x = np.array(self.vertices, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Invalid coordinates')
        if any(np.linalg.norm(x[i]-x[j])<1e-9 for i,j in itertools.combinations(range(len(x)),2)):
            raise ValueError('Coincident vertices')
        for s in self.maximal:
            if not s or min(s)<0 or max(s)>=len(x):
                raise ValueError('Invalid simplex indices')
            if np.linalg.matrix_rank(x[list(s)[1:]]-x[s[0]],tol=1e-9) != len(s)-1:
                raise ValueError('Degenerate simplex')
        # Convex hull intersections may contain only the declared common face.
        # On each nondegenerate simplex barycentric coordinates are unique.
        for a,b in itertools.combinations(self.maximal,2):
            n,m = len(a),len(b); shared = set(a)&set(b)
            mat = np.zeros((x.shape[1]+2,n+m))
            mat[:x.shape[1],:n]=x[list(a)].T
            mat[:x.shape[1],n:]=-x[list(b)].T
            mat[-2,:n]=1;mat[-1,n:]=1
            rhs=np.zeros(x.shape[1]+2);rhs[-2:]=1
            outside=np.array([v not in shared for v in a+b],dtype=float)
            r=linprog(-outside,A_eq=mat,b_eq=rhs,bounds=(0,None),method='highs')
            if r.success and (not shared or -r.fun>1e-7):
                raise ValueError('Non-face intersection')
            if not r.success and r.status!=2:
                raise ValueError('Intersection validation inconclusive')
        return {'vertices':len(x),'ambient_dimension':x.shape[1],
                'maximal_dimension':max(map(len,self.maximal))-1,
                'faces':len(self.faces()),'edges':len(self.edges())}

    def subdivide(self, support):
        s=tuple(sorted(support))
        if s not in self.maximal: raise ValueError('Need maximal simplex')
        i=len(self.vertices);self.vertices.append(self.tensor()[list(s)].mean(0).tolist())
        self.maximal.remove(s)
        self.maximal += [tuple(sorted((*f,i))) for f in itertools.combinations(s,len(s)-1)]
        self.clean();self.validate();return i

    def lift(self, support, sign=1):
        s=tuple(sorted(support));x=self.tensor();p=x[list(s)]
        if s not in self.faces(): raise ValueError('Unknown support simplex')
        b=p[1:]-p[0]
        center=p[0]+torch.linalg.lstsq(b, b.square().sum(1)/2).solution
        r2=(center-p[0]).square().sum()
        length=torch.pdist(p).max()
        if length.square()<=r2: raise ValueError('No positive prescribed apex height')
        height=(length.square()-r2).sqrt()
        self.vertices=torch.cat([x,torch.zeros((len(x),1),dtype=x.dtype)],1).tolist()
        i=len(x);self.vertices.append(torch.cat([center,sign*height[None]]).tolist())
        self.maximal.append(tuple((*s,i)));self.clean();self.validate();return i

    def suspension(self, support):
        s=tuple(sorted(support));first=self.lift(s)
        p=self.tensor()[first].clone();p[-1]*=-1
        second=len(self.vertices);self.vertices.append(p.tolist())
        self.maximal.append(tuple((*s,second)));self.clean();self.validate()
        return first,second

    def round_vertices(self, support, decimals=1):
        old=self.state();x=self.tensor();x[list(support)]=x[list(support)].round(decimals=decimals)
        self.vertices=x.tolist()
        try: return self.validate()
        except Exception:
            self.__dict__.update(Complex(**old).__dict__)
            raise
