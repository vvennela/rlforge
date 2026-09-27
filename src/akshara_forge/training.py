"""Fixed-dataset GRPO adapter. No adaptive problem generation or self-play.

This CUDA training path is separate from Ollama inference. A training launch is
not an uplift result: evaluate both initial and final weights on the same backend.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .environment import load_problems, grade_answer
from .completion import parse_completion
from .io import digest_file, write_json
from .worker import SYSTEM


def training_rows(root: Path, paper: str) -> tuple[list[dict], dict]:
    # No held-out file is loaded by the trainer.
    problems = load_problems(root, [paper], 'train')
    if len(problems) != 80 or any(p['split'] != 'train' for p in problems):
        raise ValueError('Expected exactly 80 frozen training problems')
    audit_path=root/'runs/dataset-audit.json'
    if audit_path.exists():
        ids={p['problem_id'] for p in problems}
        flagged=[r['problem_id'] for r in json.loads(audit_path.read_text())['records']
                 if r['problem_id'] in ids and r['status']!='valid']
        if flagged:raise ValueError('Known prompt defects block training; use reviewed pilot-v2: '+', '.join(flagged))
    rows = [{'prompt': [{'role':'system','content':SYSTEM}, {'role':'user','content':p['prompt']}],
             'problem_id':p['problem_id']} for p in problems]
    return rows, {p['problem_id']:p for p in problems}


def make_reward(problems: dict, output: Path, teacher=None, process_teacher=None, context=None):
    def reward(completions, problem_id, **kwargs):
        def evaluate(item):
            completion,identifier=item
            text = completion[0]['content'] if isinstance(completion, list) else completion
            action = parse_completion(text)
            problem = problems[identifier]
            key = hashlib.sha256(json.dumps({'problem':problem,'action':action},sort_keys=True).encode()).hexdigest()
            result = grade_answer(problem, action)
            if process_teacher is not None:
                result = process_teacher(problem, action, output/'process-judgments'/key)
            elif result['status'] == 'needs_judge':
                if teacher is None:
                    raise RuntimeError('Proof tasks require Astra; never silently drop or zero-score them')
                result = teacher(problem, action, output/'teacher'/key)
            if result.get('reward') is None:
                raise RuntimeError('Unresolved teacher judgment stops training for review')
            return {'problem_id':identifier,'submission':action,'grade':result,
                    'timestamp':dt.datetime.now(dt.timezone.utc).isoformat(),
                    'training_context':context() if context else None}
        items=list(zip(completions,problem_id,strict=True))
        with ThreadPoolExecutor(max_workers=min(4,len(items) or 1)) as pool:
            records=list(pool.map(evaluate,items))
        values=[float(r['grade']['reward']) for r in records]
        output.mkdir(parents=True,exist_ok=True)
        with (output/'reward-traces.jsonl').open('a') as f:
            for record in records:f.write(json.dumps(record,allow_nan=False)+'\n')
        return values
    return reward


def preflight(root: Path, paper: str) -> dict:
    rows, problems = training_rows(root,paper)
    packages = {name: importlib.util.find_spec(name) is not None for name in ['torch','transformers','trl','peft','datasets']}
    cuda = False
    if packages['torch']:
        import torch
        cuda = torch.cuda.is_available()
    remote_teacher=bool(os.environ.get('AKSHARA_REWARD_URL') and os.environ.get('AKSHARA_REWARD_TOKEN_FILE'))
    return {'training_problems':len(rows), 'proofs_requiring_teacher':sum(p['verification']['type']=='judge' for p in problems.values()),
            'packages':packages,'cuda_available':cuda,'astra_cli_available':shutil.which('codex') is not None,
            'remote_teacher_configured':remote_teacher,
            'ready':all(packages.values()) and cuda and (remote_teacher or shutil.which('codex') is not None),
            'weight_updates_performed':False, 'step_penalty':0,
            'evaluation_requirement':'Initial and final checkpoints must use matching precision, prompt, decoding and backend. Ollama Q4 baseline cannot be directly attributed against a BF16 trained checkpoint.'}


def ordered_training_rows(rows: list[dict], problems: dict, curriculum: Path, source_file: Path) -> list[dict]:
    schedule=json.loads(curriculum.read_text())
    if schedule['split']!='train' or schedule['source_sha256']!=digest_file(source_file):
        raise ValueError('Curriculum must match the frozen training split')
    ids=schedule['ordered_ids']
    if len(ids)!=len(problems) or len(set(ids))!=len(ids) or set(ids)!=set(problems):
        raise ValueError('Curriculum must cover each training ID exactly once')
    lookup={r['problem_id']:r for r in rows}
    return [lookup[key] for key in ids]


def train(root: Path, paper: str, output: Path, epochs: int = 1, reward_mode: str = 'process', max_steps: int = -1, curriculum: Path|None=None):
    status = preflight(root,paper)
    if not status['ready']:
        raise RuntimeError('Training preflight failed: '+json.dumps(status))
    if output.exists():
        raise ValueError('Choose a new checkpoint directory')
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
    from .teacher import judge
    from .progress import prepare_contract, judge_progress
    rows, problems = training_rows(root,paper)
    if curriculum:
        rows=ordered_training_rows(rows,problems,curriculum,root/'data/problems'/paper/'train.jsonl')
    config = json.loads((root/'configs/pilot.json').read_text())
    write_json(output/'experiment.json', {'paper':paper,'model':config['hf_training_model'],
               'train_sha256':digest_file(root/'data/problems'/paper/'train.jsonl'),
               'epochs':epochs,'max_steps':max_steps,'seed':config['seed'],'reward_mode':reward_mode,
               'curriculum_sha256':digest_file(curriculum) if curriculum else None,
               'reward':'0.7 outcome + 0.3 fixed mathematical milestone coverage' if reward_mode=='process' else 'final correctness',
               'efficiency_reward':False,'self_play':False,'status':'starting'})
    process_teacher=None
    if os.environ.get('AKSHARA_REWARD_URL'):
        if reward_mode!='process':raise ValueError('Remote reward service supports the process contract only')
        from .reward_service import remote_reward
        process_teacher=remote_reward
    elif reward_mode=='process':
        # Freeze every contract before drawing the first learner rollout.
        def prepare(item):
            identifier,problem=item
            return identifier,prepare_contract(problem,output/'contracts'/identifier)
        with ThreadPoolExecutor(max_workers=2) as pool:
            contracts=dict(pool.map(prepare,problems.items()))
        def process_teacher(problem,action,destination):
            return judge_progress(problem,action,destination,contracts[problem['problem_id']])
    args = GRPOConfig(output_dir=str(output), learning_rate=5e-6, num_train_epochs=epochs,
                      max_steps=max_steps,
                      shuffle_dataset=curriculum is None,
                      per_device_train_batch_size=1, gradient_accumulation_steps=4,
                      num_generations=4, max_completion_length=config['max_output_tokens'],
                      temperature=0.7, bf16=True, gradient_checkpointing=True,
                      model_init_kwargs={'dtype':'bfloat16'}, seed=config['seed'],
                      report_to='none', logging_steps=1, save_strategy='steps',save_steps=10,save_total_limit=2,
                      remove_unused_columns=False)
    class TraceCallback(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            with (output/'optimizer-events.jsonl').open('a') as f:
                f.write(json.dumps({'event':'log','global_step':state.global_step,'epoch':state.epoch,
                                   'timestamp':dt.datetime.now(dt.timezone.utc).isoformat(),'metrics':logs},default=str)+'\n')
        def on_save(self,args,state,control,**kwargs):
            with (output/'optimizer-events.jsonl').open('a') as f:
                f.write(json.dumps({'event':'checkpoint','global_step':state.global_step,
                                   'checkpoint':str(output/f'checkpoint-{state.global_step}')})+'\n')
    trainer = GRPOTrainer(model=config['hf_training_model'], args=args,
                           train_dataset=Dataset.from_list(rows),
                           reward_funcs=make_reward(problems,output,teacher=judge,process_teacher=process_teacher,
                               context=lambda:{'global_step':trainer.state.global_step,'epoch':trainer.state.epoch,
                                               'base_model':config['hf_training_model']}),
                           callbacks=[TraceCallback()],
                           peft_config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,
                                                target_modules='all-linear',task_type='CAUSAL_LM'))
    import torch
    initial={name:p.detach().cpu().clone() for name,p in trainer.model.named_parameters() if p.requires_grad}
    write_json(output/'trainable-parameters.json',{'tensors':len(initial),'parameters':sum(p.numel() for p in initial.values())})
    try:
        trainer.train()
    except Exception as exc:
        trainer.save_model(str(output/'interrupted-adapter'))
        write_json(output/'interruption.json',{'error':str(exc),'optimizer_steps':trainer.state.global_step,
                   'status':'requires_review','uplift_measured':False})
        raise
    trainer.save_model(str(output/'adapter'))
    changed=0;sum_squares=0.0;max_abs=0.0
    for name,p in trainer.model.named_parameters():
        if name not in initial:continue
        delta=p.detach().cpu().float()-initial[name].float()
        changed+=int(torch.count_nonzero(delta).item())
        sum_squares+=float((delta*delta).sum().item())
        max_abs=max(max_abs,float(delta.abs().max().item()))
    write_json(output/'completion.json',{'weight_updates_performed':changed>0,'optimizer_steps':trainer.state.global_step,
               'changed_parameter_elements':changed,'adapter_delta_l2':sum_squares**0.5,
               'adapter_delta_max_abs':max_abs,'uplift_measured':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--paper',required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--reward-mode',choices=['process','outcome'],default='process')
    parser.add_argument('--max-steps',type=int,default=-1,help='Optimizer-step limit for infrastructure smoke tests; -1 runs epochs')
    parser.add_argument('--curriculum',type=Path)
    args=parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.root,args.paper),indent=2));return
    if args.output is None:parser.error('--output is required for training')
    train(args.root,args.paper,args.output,args.epochs,args.reward_mode,args.max_steps,args.curriculum)


if __name__=='__main__':main()
