"""Independently compare saved LoRA tensors on CPU, without loading a base model."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def audit(initial_path,adapter_path,expected):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    saved=load_file(str(adapter_path),device='cpu')
    normalized={key.replace('.lora_A.default.','.lora_A.').replace('.lora_B.default.','.lora_B.'):value for key,value in initial.items()}
    if len(normalized)!=len(initial) or set(normalized)!=set(saved):raise ValueError('Adapter tensor identity mismatch')
    changed=0;squared=0.;tensors=[]
    for key,start in normalized.items():
        finish=saved[key]
        if start.shape!=finish.shape:raise ValueError('Adapter tensor shape mismatch: '+key)
        delta=finish.float()-start.float()
        if not torch.isfinite(delta).all():raise ValueError('Nonfinite adapter delta: '+key)
        count=int(torch.count_nonzero(delta));norm=float((delta*delta).sum())
        changed+=count;squared+=norm
        tensors.append({'name':key,'shape':list(delta.shape),'changed_parameters':count,'delta_l2':norm**.5})
    norm=squared**.5
    if changed!=expected['changed_parameters'] or not math.isclose(norm,expected['delta_l2'],rel_tol=1e-8,abs_tol=1e-10):
        raise ValueError('Saved tensors disagree with recorded adapter delta')
    return {'verified':True,'device':'cpu','tensor_count':len(tensors),'changed_parameters':changed,'delta_l2':norm,
        'initial_sha256':digest(initial_path),'adapter_sha256':digest(adapter_path),'tensors':tensors}


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.initial,args.adapter,json.loads(args.expected.read_text()))
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='tensors'},indent=2))

if __name__=='__main__':main()
