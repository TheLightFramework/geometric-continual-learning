"""Fixed-dose acquisition/consolidation demonstration; no automatic promotion.

This small authored fixture is engineering evidence, not a replication of the
historical corpus-based study. It freezes its contract before any optimizer step.
"""
import argparse
import copy
import json
from pathlib import Path
import platform
import time
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from .io import load,save,setup,write,read,tensor_hash,object_hash,tokenizer,generate,sha
from .transactions import Store,artifact

def encode(row, enc):
    return {'prompt':row['prompt'],'answer':row['answer'],'input_ids':enc.encode_ordinary(row['prompt']), 'target_ids':enc.encode_ordinary(row['answer'])+[enc.eot_token]}

def tensors(row, device):
    p,t=row['input_ids'],row['target_ids']
    if not p or not t: raise ValueError('EMPTY_EXAMPLE')
    return torch.tensor([p+t[:-1]],device=device),torch.tensor([[-100]*(len(p)-1)+t],device=device)

def ce(model, row):
    x,y=tensors(row,next(model.parameters()).device)
    if x.shape[1]>model.cfg.max_length: raise ValueError('CONTEXT_EXCEEDED')
    return model(x,labels=y)['loss']

def kl(model,teacher,ids):
    x=torch.tensor([ids],device=next(model.parameters()).device)
    with torch.no_grad(): target=teacher(x)['logits'][...,:teacher.cfg.valid_vocab].float().softmax(-1)
    pred=model(x)['logits'][...,:model.cfg.valid_vocab].float().log_softmax(-1)
    return F.kl_div(pred,target,reduction='sum')/x.numel()

@torch.inference_mode()
def evaluate(model, rows,eot):
    model.eval();items=[]
    for row in rows:
        r=generate(model,row['input_ids'],eot,8)
        items.append({**row,**r,'exact':r['tokens']==row['target_ids'],'answer_ce':float(ce(model,row))})
    return {'exact':sum(x['exact'] for x in items),'n':len(items),'mean_ce':sum(x['answer_ce'] for x in items)/len(items),'items':items}

def run(parent_dir,lesson_path,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    lesson=read(lesson_path)
    if not all(lesson[k] for k in ['train','acquisition_eval','retention','kl_inputs']): raise ValueError('EMPTY_SURFACE')
    cfg=lesson['recipe'];setup(cfg['seed'])
    parent,meta=load(parent_dir,device);enc=tokenizer();setup(cfg['seed'])
    rows={k:[encode(r,enc) for r in lesson[k]] for k in ['train','acquisition_eval','retention']}
    if set(x['prompt'] for x in rows['train']) & set(x['prompt'] for x in rows['acquisition_eval']): raise ValueError('EVALUATION_PROMPTS_OVERLAP_TRAINING')
    kl_rows=[enc.encode_ordinary(s) for s in lesson['kl_inputs']]
    if any(not x or len(x)>parent.cfg.max_length for x in kl_rows): raise ValueError('INVALID_KL_INPUT')
    selected=[n for n,p in parent.named_parameters() if n.startswith(tuple(f'blocks.{i}.' for i in cfg['train_blocks']))]
    if not selected: raise ValueError('EMPTY_TRAINABLE_REGION')
    contract={'required_tests':['acquisition','retention','finite','protected_exact','parent_unchanged','reload_exact'],'evidence_standing':'authored engineering fixture; not scientific confirmation','thresholds':lesson['thresholds'],'recipe':cfg,'lesson_sha256':sha(lesson_path),'reference':{'tensor_sha256':meta['tensor_sha256'],'query_distribution':'kl_inputs in sealed lesson; synthetic, not a prose retention benchmark'},'replay_distribution':'retention examples in sealed lesson; explicit supervised replay','trainable_names':selected}
    write(out/'LESSON.json',lesson);write(out/'CONTRACT.json',contract)
    print('Contract sealed before fitting:',sha(out/'CONTRACT.json'),flush=True)
    store=Store(out/'store');parent_id=store.initialize([artifact(Path(parent_dir)/'weights.pt'),artifact(Path(parent_dir)/'MANIFEST.json')],'immutable model reference')
    before={k:evaluate(parent,rows[k],enc.eot_token) for k in ['acquisition_eval','retention']}
    write(out/'BEFORE.json',before)
    model=copy.deepcopy(parent)
    for n,p in model.named_parameters(): p.requires_grad_(n in selected)
    protected_before=tensor_hash({n:p for n,p in parent.state_dict().items() if n not in selected})
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],betas=(.9,.95),eps=1e-8,weight_decay=0,foreach=False)
    start=time.perf_counter();step=0
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    with (out/'trajectory.jsonl').open('x',encoding='utf-8') as log:
        for phase,count in [('acquisition',cfg['acquisition_steps']),('consolidation',cfg['consolidation_steps'])]:
            for _ in range(count):
                model.train();opt.zero_grad(set_to_none=True)
                task=ce(model,rows['train'][step%len(rows['train'])]);loss=task
                replay=regularizer=None
                if phase=='consolidation':
                    replay=ce(model,rows['retention'][step%len(rows['retention'])]);regularizer=kl(model,parent,kl_rows[step%len(kl_rows)])
                    loss=loss+cfg['replay_weight']*replay+cfg['kl_weight']*regularizer
                if not torch.isfinite(loss): raise ValueError('NONFINITE_LOSS')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                opt.step();model.stepped();step+=1
                record={'step':step,'phase':phase,'task_ce':float(task.detach()),'replay_ce':None if replay is None else float(replay.detach()),'kl':None if regularizer is None else float(regularizer.detach()),'preclip_gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-start}
                log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                print(record,flush=True)
            write(out/(phase.upper()+'.json'),{k:evaluate(model,rows[k],enc.eot_token) for k in ['acquisition_eval','retention']})
    after=read(out/'CONSOLIDATION.json');candidate_meta=save(model,out/'candidate')
    # Fresh model construction plus strict tensor load; separate fresh-process test
    # is performed by the release audit, not falsely implied here.
    restored,restored_meta=load(out/'candidate',device)
    repeated={k:evaluate(restored,rows[k],enc.eot_token) for k in ['acquisition_eval','retention']}
    thresholds=lesson['thresholds']
    tests={'acquisition':after['acquisition_eval']['exact']/after['acquisition_eval']['n']>=thresholds['acquisition_exact'],
           'retention':after['retention']['exact']>=before['retention']['exact'] and after['retention']['mean_ce']<=before['retention']['mean_ce']+thresholds['retention_ce_delta'],
           'finite':all(bool(torch.isfinite(p).all()) for p in model.parameters()),
           'protected_exact':tensor_hash({n:p for n,p in model.state_dict().items() if n not in selected})==protected_before,
           'parent_unchanged':tensor_hash(parent.state_dict())==meta['tensor_sha256'] and sha(Path(parent_dir)/'weights.pt')==meta['weights_sha256'],
           'reload_exact':candidate_meta['tensor_sha256']==restored_meta['tensor_sha256'] and repeated==after}
    refs=[artifact(out/'candidate/weights.pt'),artifact(out/'candidate/MANIFEST.json')]
    receipt={'parent':parent_id,'artifacts':refs,'contract_sha256':sha(out/'CONTRACT.json'),'tests':tests,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'python':platform.python_version(),'torch':torch.__version__,'device':device,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}
    write(out/'EVALUATION.json',receipt)
    decision=store.review({'parent':parent_id,'artifacts':refs,'contract':artifact(out/'CONTRACT.json'),'evaluation':artifact(out/'EVALUATION.json')})
    write(out/'DECISION.json',decision)
    assert store.current()==parent_id
    print('REVIEW ONLY. Parent unchanged. Eligible:',decision['eligible'],'Failed:',decision['failed'],flush=True)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights-dir',required=True);p.add_argument('--lesson',required=True);p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    with sdpa_kernel(SDPBackend.MATH): run(a.weights_dir,a.lesson,a.out,a.device)

if __name__=='__main__': main()
