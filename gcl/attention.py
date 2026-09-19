"""Rotary causal attention, including rectangular cached masks."""
import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dim = c.heads, c.width // c.heads
        assert c.width % c.heads == 0 and self.dim % 2 == 0
        self.qkv = nn.Linear(c.width, 3*c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        self.register_buffer('inv_freq', 10000.**(-torch.arange(0,self.dim,2).float()/self.dim), persistent=False)

    def rotate(self, x, offset):
        p = torch.arange(offset, offset+x.shape[-2], device=x.device).float()
        a = p[:,None]*self.inv_freq[None,:]
        co, si = a.cos().to(x.dtype), a.sin().to(x.dtype)
        even, odd = x[...,::2], x[...,1::2]
        return torch.stack((even*co-odd*si, even*si+odd*co),-1).flatten(-2)

    def forward(self, x, past=None, cache=False):
        b,t,d=x.shape
        off=0 if past is None else past[0].shape[-2]
        q,k,v=self.qkv(x).view(b,t,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        q,k=self.rotate(q,off),self.rotate(k,off)
        if past is not None:
            k=torch.cat((past[0],k),-2);v=torch.cat((past[1],v),-2)
        # Rectangular cached attention is aligned by absolute coordinates, not a
        # square upper-left is_causal mask. True means the key is allowed.
        mask=None if off==0 else (torch.arange(off,off+t,device=x.device)[:,None] >= torch.arange(off+t,device=x.device)[None,:])
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=off==0,dropout_p=0.)
        return self.out(y.transpose(1,2).contiguous().view(b,t,d)), ((k,v) if cache else None)
