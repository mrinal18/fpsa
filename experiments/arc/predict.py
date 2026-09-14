"""Label-free inference from a new FPSA/TRM checkpoint on official prepared data.

Test labels are not loaded. Select K task augmentations BEFORE model evaluation,
so candidate budget reduces actual inference work. Always finish configured
refinement segments; learned halt confidence ranks candidates but is not proof.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml
from .upstream import checkout, install_shims
from .codec import decode, inverse, parse_identifier, grid_key


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',default='.external/trm')
    p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',required=True,help='Resolved all_config.yaml from the exact training run')
    p.add_argument('--challenges',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--augmentations',type=int,default=1,help='K task variants including canonical')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if min(a.augmentations,a.batch_size)<1:
        raise ValueError('Candidate and batch budgets must be positive')
    upstream=checkout(a.upstream)
    install_shims(upstream)
    sys.path.insert(0,str(upstream))
    from utils.functions import load_model_class
    config=yaml.safe_load(Path(a.config).read_text())
    root=Path(a.data)
    meta=json.loads((root/'test/dataset.json').read_text())
    identifiers=json.loads((root/'identifiers.json').read_text())
    challenges=json.loads(Path(a.challenges).read_text())
    lookup={task:{} for task in challenges}
    for task,item in challenges.items():
        for qi,pair in enumerate(item['test']):
            lookup[task].setdefault(grid_key(pair['input']),[]).append(qi)
    raw=dict(config['arch'])
    name=raw.pop('name');raw.pop('loss',None)
    raw.update(batch_size=a.batch_size,seq_len=meta['seq_len'],vocab_size=meta['vocab_size'],
               num_puzzle_identifiers=meta['num_puzzle_identifiers'],causal=False)
    model=load_model_class(name)(raw).to(a.device).eval()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    state=checkpoint.get('model',checkpoint)
    prefixes=('_orig_mod.model.','model.')
    for prefix in prefixes:
        if state and all(k.startswith(prefix) for k in state):
            state={k[len(prefix):]:v for k,v in state.items()}
            break
    model.load_state_dict(state,strict=True)
    counts,allowed={},set()
    # Builder orders canonical then augmented variants for each original task.
    for tid,name_id in enumerate(identifiers):
        if tid==0 or name_id=='<blank>':continue
        task,_,_=parse_identifier(name_id)
        if task not in challenges:continue
        if counts.get(task,0)<a.augmentations:
            allowed.add(tid);counts[task]=counts.get(task,0)+1
    destination=Path(a.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError('Use a new candidate output filename')
    cost=dict(candidate_events=0,segments=0,attention_map_evals=0,known_map_counts=True)
    start=time.perf_counter()
    with destination.open('w') as handle, torch.inference_mode():
        for subset in meta['sets']:
            inputs=np.load(root/f'test/{subset}__inputs.npy',mmap_mode='r')
            pids=np.load(root/f'test/{subset}__puzzle_identifiers.npy')
            bounds=np.load(root/f'test/{subset}__puzzle_indices.npy')
            rowids=np.repeat(pids,np.diff(bounds))
            selected=np.flatnonzero(np.isin(rowids,list(allowed)))
            for start_idx in range(0,len(selected),a.batch_size):
                index=selected[start_idx:start_idx+a.batch_size]
                x=torch.as_tensor(np.array(inputs[index]),device=a.device,dtype=torch.long)
                ids=torch.as_tensor(rowids[index],device=a.device,dtype=torch.long)
                # Labels are synthetic ignored entries: query targets never enter the model.
                batch=dict(inputs=x,puzzle_identifiers=ids,labels=torch.full_like(x,-100))
                carry=model.initial_carry(batch)
                for _ in range(int(raw['halt_max_steps'])):
                    carry,out=model(carry=carry,batch=batch)
                    cost['segments']+=len(index)
                    infos=out.get('infos')
                    if infos is not None:
                        cost['attention_map_evals']+=sum(v.nfe for v in infos)*len(index)
                    else:cost['known_map_counts']=False
                    if bool(carry.halted.all()):break
                pred=out['logits'].argmax(-1).cpu().numpy()
                q=out['q_halt_logits'].double().sigmoid().cpu().tolist()
                for j,row_index in enumerate(index):
                    name_id=identifiers[int(rowids[row_index])]
                    task,tid,colors=parse_identifier(name_id)
                    inp=decode(inputs[row_index]);grid=decode(pred[j])
                    if inp is None:raise ValueError('Invalid prepared input')
                    query_key=grid_key(inverse(inp,tid,colors))
                    if query_key not in lookup[task]:raise ValueError('Challenge/manifest input mismatch')
                    for qi in lookup[task][query_key]:
                        record=dict(task_id=task,query_index=qi,
                             grid=None if grid is None else inverse(grid,tid,colors).tolist(),
                             confidence=q[j],source=a.source,candidate_id=f'{subset}:{row_index}',
                             canonical='|||' not in name_id)
                        handle.write(json.dumps(record)+'\n');cost['candidate_events']+=1
                handle.flush()
    if str(a.device).startswith('cuda'):torch.cuda.synchronize()
    cost.update(elapsed_seconds=time.perf_counter()-start,augmentations_requested=a.augmentations,
                token_positions=meta['seq_len']+int(raw.get('puzzle_emb_len',0)),
                note='Map-evaluation count is token-batch exposure, not measured FLOPs; no compaction assumed.')
    if not cost['known_map_counts']:cost['attention_map_evals']=None
    destination.with_suffix('.budget.json').write_text(json.dumps(cost,indent=2))
    print(json.dumps(cost,indent=2))

if __name__=='__main__':main()
