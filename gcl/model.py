"""New 151M formation candidate; explicit vertex and face computations."""
from dataclasses import dataclass,asdict
import contextlib,copy,math
import torch
from torch import nn
from torch.nn import functional as F
from .attention import Attention
from .geometry import Complex

@dataclass
class Config:
    width:int=768
    depth:int=16
    heads:int=12
    cell_rank:int=736
    triangle_rank:int=32
    tetra_rank:int=64
    vocab:int=50304
    valid_vocab:int=50257
    max_length:int=1024

def unit(x):return x.float()/x.float().square().mean(-1,keepdim=True).add(1e-6).sqrt()

class Cell(nn.Module):
    def __init__(self,d,r):
        super().__init__();self.up=nn.Linear(d,r,bias=False);self.down=nn.Linear(r,d,bias=False)
    def forward(self,x):return self.down(F.gelu(self.up(x)))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.attn_norm=nn.LayerNorm(c.width,bias=False);self.cell_norm=nn.LayerNorm(c.width,bias=False)
        self.attn=Attention(c);self.cells=nn.ModuleList([Cell(c.width,c.cell_rank) for _ in range(4)])
        self.face_nodes=[(0,1,2),(1,2,3),(0,1,2,3)]
        self.faces=nn.ModuleList([Cell(c.width,r) for r in [c.triangle_rank,c.triangle_rank,c.tetra_rank]])
        old=Complex.base().vertices;self.geometry=Complex([old[i] for i in [0,1,2,4]],[(0,1,2,3)])
        self.newborns=nn.ModuleList();self.birth_specs=[];self.disabled=set();self.disabled_faces=set();self.refresh()

    def refresh(self):
        p=self.geometry.tensor();self.parents={i:[] for i in range(len(p))};self.metric={}
        for a,b in self.geometry.edges():
            self.parents[b].append(a);self.metric[a,b]=float(torch.exp(-(p[a]-p[b]).square().sum()/2))

    def forward(self,x,past=None,cache=False,observe=False):
        a,kv=self.attn(self.attn_norm(x),past,cache);x=x+a;z=self.cell_norm(x);h=[];face_out=[]
        for i,cell in enumerate(self.cells):
            ctx=z
            for j in self.parents[i]:ctx=ctx+.1*self.metric[j,i]*h[j]
            v=cell(ctx);h.append(torch.zeros_like(v) if i in self.disabled else v)
        # Genuine group product; depends on the declared filled face, not only edges.
        for fi,(nodes,face) in enumerate(zip(self.face_nodes,self.faces)):
            product=torch.ones_like(h[0],dtype=torch.float32)
            for i in nodes:product=product*torch.tanh(unit(h[i]))
            v=face(product.to(z.dtype));face_out.append(torch.zeros_like(v) if fi in self.disabled_faces else v)
        delta=sum(h)+sum(face_out)
        for k,cell in enumerate(self.newborns):
            i=k+4;ctx=z
            for j in self.parents[i]:ctx=ctx+.1*self.metric[j,i]*h[j]
            for nodes,v in zip(self.face_nodes,face_out):
                if set(nodes).issubset(self.parents[i]):ctx=ctx+.1*v
            v=cell(ctx);v=torch.zeros_like(v) if i in self.disabled else v;h.append(v);delta=delta+v
        info=None
        if observe:
            with torch.no_grad():
                info={'vertex_rms':[float(v.float().square().mean().sqrt()) for v in h],
                      'face_rms':[float(v.float().square().mean().sqrt()) for v in face_out],
                      'delta_rms':float(delta.float().square().mean().sqrt())}
        return x+delta,kv,info

    def birth(self,kind='LIFT',support=(0,1,2,3),rank=64,seed=1):
        if rank<2 or kind=='PAIR_JOIN' and rank%2:raise ValueError('Even total rank required for paired birth')
        geo=copy.deepcopy(self.geometry)
        if kind=='SUBDIVIDE':geo.subdivide(support);ranks=[rank]
        elif kind=='LIFT':geo.lift(support);ranks=[rank]
        elif kind=='PAIR_JOIN':
            first=geo.lift(support);geo.lift(tuple((*support,first)));ranks=[rank//2]*2
        else:raise ValueError(kind)
        assert set(self.geometry.edges()).issubset(geo.edges())
        p=next(self.parameters());start=4+len(self.newborns)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for r in ranks:
                cell=Cell(self.c.width,r);nn.init.normal_(cell.up.weight,std=.02);nn.init.zeros_(cell.down.weight)
                self.newborns.append(cell.to(p))
        self.geometry=geo;self.refresh();self.birth_specs.append({'kind':kind,'support':list(support),'rank':rank,'seed':seed})
        return list(range(start,4+len(self.newborns)))

class Model(nn.Module):
    def __init__(self,c=None,seed=79101):
        super().__init__();self.cfg=c or Config();self.seed=seed;self.version=0;c=self.cfg
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding=nn.Embedding(c.vocab,c.width);self.blocks=nn.ModuleList([Block(c) for _ in range(c.depth)])
            self.final_norm=nn.LayerNorm(c.width,bias=False)
            for m in self.modules():
                if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight,std=.02)
            for n,p in self.named_parameters():
                if n.endswith(('down.weight','attn.out.weight')):nn.init.normal_(p,std=.02/math.sqrt(2*c.depth))

    def stepped(self):self.version+=1
    def forward(self,ids,labels=None,cache=None,use_cache=False,observe=False):
        if self.training and (cache is not None or use_cache):raise ValueError('Cache is inference-only')
        if cache is not None and cache['version']!=self.version:raise ValueError('Stale model cache')
        off=0 if cache is None else cache['kv'][0][0].shape[-2]
        if off+ids.shape[1]>self.cfg.max_length:raise ValueError('Context exceeded; no silent truncation')
        x=self.embedding(ids);kv=[];traces=[]
        for i,b in enumerate(self.blocks):
            x,k,info=b(x,None if cache is None else cache['kv'][i],use_cache,observe);kv.append(k)
            if info is not None:traces.append(info)
        logits=F.linear(self.final_norm(x),self.embedding.weight)
        if self.cfg.vocab>self.cfg.valid_vocab:
            logits=logits.clone();logits[...,self.cfg.valid_vocab:]=float('-inf')
        result={'logits':logits,'cache':{'version':self.version,'kv':kv} if use_cache else None,'traces':traces}
        if labels is not None:
            count=int((labels!=-100).sum())
            if not count:raise ValueError('No supervised targets')
            loss=F.cross_entropy(logits.float().flatten(0,1),labels.long().flatten(),ignore_index=-100,reduction='sum')
            result.update(loss=loss/count,loss_sum=loss,targets=count)
        return result

    def topology(self):
        return [{'births':copy.deepcopy(b.birth_specs),'geometry':b.geometry.state(),'face_nodes':b.face_nodes} for b in self.blocks]

    @classmethod
    def restore(cls,p):
        m=cls(Config(**p['config']),p['seed'])
        if len(p['topology'])!=len(m.blocks):raise ValueError('Topology depth mismatch')
        for b,state in zip(m.blocks,p['topology']):
            for spec in state['births']:b.birth(**spec)
            b.geometry=Complex(**state['geometry']);b.geometry.validate();b.face_nodes=[tuple(f) for f in state['face_nodes']]
            if len(b.face_nodes)!=len(b.faces) or len(b.geometry.vertices)!=4+len(b.newborns):raise ValueError('Topology/module mismatch')
            b.refresh()
        m.load_state_dict(p['model'],strict=True);m.version=p['version'];return m

    def birth(self,block,**kwargs):
        ids=self.blocks[block].birth(**kwargs);self.stepped();return ids

    def project(self,names,bits=7):
        named=dict(self.named_parameters());report=[]
        if not 2<=bits<=23:raise ValueError('Bits outside declared range')
        with torch.no_grad():
            for name in names:
                if not ('.cells.' in name or '.faces.' in name or '.newborns.' in name):raise ValueError('Projection is cell/face local only')
                p=named[name];old=p.clone()
                if bits<23:
                    s=p.abs().amax(-1,keepdim=True).clamp_min(1e-30)/(2**bits-1);p.copy_(s*torch.round(p/s))
                report.append({'name':name,'delta_rms':float((p-old).square().mean().sqrt())})
        self.stepped();return report

    def round_parameters(self,names,decimals=2):
        """Literal decimal projection, distinct from row-relative bit projection.

        Applied only on an explicitly selected clone. It is not guaranteed to
        preserve function, free semantic capacity, or reduce FP32 storage.
        """
        if not isinstance(decimals,int) or not -6<=decimals<=8:raise ValueError('Decimal range -6..8')
        named=dict(self.named_parameters());names=list(names)
        if any(n not in named or not any(s in n for s in ['.cells.','.faces.','.newborns.']) for n in names):
            raise ValueError('Decimal projection is cell/face local only')
        report=[]
        with torch.no_grad():
            for n in names:
                p=named[n];old=p.clone();p.copy_(torch.round(p,decimals=decimals))
                report.append({'name':n,'delta_rms':float((p-old).square().mean().sqrt())})
        self.stepped();return report

    @contextlib.contextmanager
    def intervention(self,block,vertices=(),faces=(),lift=None,round_coordinates=None):
        b=self.blocks[block];old=copy.deepcopy(b.geometry);dv=b.disabled.copy();df=b.disabled_faces.copy()
        try:
            b.disabled.update(vertices);b.disabled_faces.update(faces)
            if lift is not None:
                node,height=lift;x=b.geometry.tensor();x=torch.cat((x,torch.zeros(len(x),1,dtype=x.dtype)),1);x[node,-1]=height
                b.geometry.vertices=x.tolist();b.geometry.validate()
            if round_coordinates is not None:b.geometry.round_vertices(*round_coordinates)
            b.refresh();self.stepped();yield
        finally:b.geometry=old;b.disabled=dv;b.disabled_faces=df;b.refresh();self.stepped()
