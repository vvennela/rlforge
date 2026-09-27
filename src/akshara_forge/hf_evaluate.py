"""Matched BF16 checkpoint evaluation; no training or teacher feedback to learner."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from .baseline import summarize
from .environment import load_problems,parse_response,grade_answer
from .io import digest_file,read_jsonl,write_json
from .worker import SYSTEM


def evaluate(root: Path,paper: str,model_name: str,output: Path,adapter: Path|None=None):
    import torch,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    if not torch.cuda.is_available():raise RuntimeError('This evaluation protocol requires CUDA')
    problems=load_problems(root,[paper],'test')
    if len(problems)!=20:raise ValueError('Expected the full 20-task evaluation')
    settings=json.loads((root/'configs/pilot.json').read_text())
    config={'model':model_name,'adapter':str(adapter) if adapter else None,'papers':[paper],'split':'test',
            'limit':None,'attempts':1,'seed':settings['seed'],'max_output_tokens':settings['max_output_tokens'],
            'temperature':settings['temperature'],'sandbox':'no-tool-model-generation','base_url':'local-hf-cuda',
            'step_penalty':0.0,'problem_hash':hashlib.sha256(json.dumps(problems,sort_keys=True).encode()).hexdigest()}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())!=config:
        raise ValueError('Evaluation directory already bound to another protocol')
    write_json(output/'config.json',config)
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False).to('cuda')
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model.eval()
    receipt={'backend':'huggingface','precision':'bfloat16','torch':torch.__version__,
             'transformers':transformers.__version__,'base_model':model_name,
             'base_commit':getattr(model.config,'_commit_hash',None),
             'evaluator_sha256':digest_file(Path(__file__)),'system_prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),
             'adapter_sha256':digest_file(adapter/'adapter_model.safetensors') if adapter else None}
    write_json(output/'model-receipt.json',receipt)
    records=read_jsonl(output/'episodes.jsonl') if (output/'episodes.jsonl').exists() else []
    completed={r['problem_id'] for r in records}
    for p in problems:
        if p['problem_id'] in completed:continue
        # Same independent seed and decoding protocol at both checkpoints.
        set_seed(config['seed'])
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':p['prompt']}]
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=tokenizer(text,return_tensors='pt').to('cuda');start=time.monotonic()
        with torch.inference_mode():
            tokens=model.generate(**inputs,max_new_tokens=config['max_output_tokens'],do_sample=True,
                                  temperature=config['temperature'],top_p=1.0,top_k=0,pad_token_id=tokenizer.eos_token_id)
        new=tokens[0,inputs['input_ids'].shape[1]:]
        response=tokenizer.decode(new,skip_special_tokens=True);action=parse_response(response)
        record={'problem_id':p['problem_id'],'paper_id':paper,'kind':p['kind'],'split':'test','attempt':0,
                'timestamp':dt.datetime.now(dt.timezone.utc).isoformat(),'response':response,'action':action,
                'seconds':time.monotonic()-start,'eval_count':len(new),
                'done_reason':'length' if len(new)==config['max_output_tokens'] else 'stop',**grade_answer(p,action)}
        records.append(record)
        with (output/'episodes.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        write_json(output/'summary.json',summarize(records))
        print(p['problem_id'],record['status'],record['reward'],round(record['seconds'],2),flush=True)
    return summarize(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--paper',required=True)
    p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--adapter',type=Path)
    a=p.parse_args();print(json.dumps(evaluate(a.root,a.paper,a.model,a.output,a.adapter),indent=2))


if __name__=='__main__':main()
