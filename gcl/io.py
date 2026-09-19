"""Portable tensor-only checkpoints. Integrity checks are not authentication."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from .model import Model

ASSETS = Path(__file__).parent / 'assets'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def object_hash(value): return hashlib.sha256(canonical(value)).hexdigest()

def write(path, value):
    with Path(path).open('xb') as f: f.write(canonical(value)+b'\n')

def tensor_hash(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        t=t.detach().cpu().contiguous()
        h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def setup(seed=79101):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True)

def tokenizer():
    import tiktoken
    d=read(ASSETS/'gpt2.json')
    e=tiktoken.Encoding(name='gcl_gpt2',pat_str=d['pat_str'],mergeable_ranks={base64.b64decode(b):int(i) for b,i in d['mergeable_ranks']},special_tokens=d['special_tokens'])
    if e.n_vocab!=50257 or e.eot_token!=50256: raise ValueError('TOKENIZER_MISMATCH')
    return e

def load(directory, device='cpu'):
    directory=Path(directory).resolve(); m=read(directory/'MANIFEST.json')
    p=(directory/m['weights_filename']).resolve()
    if not p.is_relative_to(directory) or sha(p)!=m['weights_sha256']: raise ValueError('WEIGHTS_MISMATCH')
    setup(m['seed'])
    state=torch.load(p,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()): raise ValueError('TENSOR_ONLY_REQUIRED')
    if tensor_hash(state)!=m['tensor_sha256']: raise ValueError('TENSOR_MISMATCH')
    model=Model.restore({**m,'model':state}).eval().requires_grad_(False)
    if tensor_hash(model.state_dict())!=m['tensor_sha256'] or object_hash(model.topology())!=m['topology_sha256']: raise ValueError('RECONSTRUCTION_MISMATCH')
    return model.to(device),m

def save(model, directory, variant='experimental-candidate'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    state={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
    p=directory/'weights.pt';torch.save(state,p)
    meta={'config':asdict(model.cfg),'seed':model.seed,'version':model.version,'topology':model.topology(),'parameters':sum(p.numel() for p in model.parameters()),'variant':variant,'weights_filename':'weights.pt','weights_sha256':sha(p),'weights_bytes':p.stat().st_size,'tensor_sha256':tensor_hash(state),'topology_sha256':object_hash(model.topology())}
    write(directory/'MANIFEST.json',meta)
    return meta

@torch.inference_mode()
def generate(model, ids, eot, cap=8):
    if not ids or cap<1 or len(ids)+cap>model.cfg.max_length: raise ValueError('EMPTY_OR_OVERSIZE_CONTEXT')
    model.eval();device=next(model.parameters()).device
    x=torch.tensor([ids],device=device);cache=None;tokens=[]
    for _ in range(cap):
        r=model(x,cache=cache,use_cache=True);z=r['logits'][0,-1,:model.cfg.valid_vocab]
        if not torch.isfinite(z).all(): raise ValueError('NONFINITE_LOGITS')
        t=int(z.argmax());tokens.append(t)
        if t==eot: break
        cache=r['cache'];x=torch.tensor([[t]],device=device)
    return {'tokens':tokens,'terminated':tokens[-1]==eot,'capped':tokens[-1]!=eot}
