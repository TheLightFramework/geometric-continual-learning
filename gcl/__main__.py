import argparse
import json
from .io import load,tokenizer,generate

def main():
    p=argparse.ArgumentParser(description='Read-only geometric language model. Not a chat assistant.')
    p.add_argument('command',choices=['inspect','generate'])
    p.add_argument('--weights-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--prompt');p.add_argument('--max-new-tokens',type=int,default=32)
    a=p.parse_args();m,meta=load(a.weights_dir,a.device)
    out={k:meta[k] for k in ['variant','parameters','tensor_sha256','topology_sha256']}
    if a.command=='generate':
        e=tokenizer();r=generate(m,e.encode_ordinary(a.prompt or ''),e.eot_token,a.max_new_tokens)
        out.update(r);out['text']=e.decode(r['tokens'][:-1] if r['terminated'] else r['tokens'])
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
