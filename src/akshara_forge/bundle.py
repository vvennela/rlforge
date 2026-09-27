from __future__ import annotations
import json
import shutil
from pathlib import Path
from .io import digest_file, write_json


ENTRY = '''import argparse,json,sys
from pathlib import Path
from .environment import MathEnvironment,load_problems
p=argparse.ArgumentParser(description="Standalone AksharaForge JSON-lines environment")
p.add_argument("--split",choices=["train","test"],default="train")
a=p.parse_args()
root=Path(__file__).resolve().parent.parent
papers=json.loads((root/"configs/pilot.json").read_text())["papers"]
env=MathEnvironment(load_problems(root,papers,a.split))
for line in sys.stdin:
 try:
  request=json.loads(line)
  if request["op"]=="reset":result=env.reset(request["problem_id"])
  elif request["op"]=="step":result=env.step(request["action"])
  else:raise ValueError("Operation must be reset or step")
 except Exception as e:result={"error":str(e)}
 print(json.dumps(result,allow_nan=False),flush=True)
'''


def export(root: Path, output: Path) -> dict:
    if output.exists():raise ValueError('Choose a new bundle output')
    (output/'akshara_forge').mkdir(parents=True)
    for name in ['__init__.py','environment.py','io.py']:
        shutil.copyfile(Path(__file__).with_name(name),output/'akshara_forge'/name)
    (output/'akshara_forge/__main__.py').write_text(ENTRY)
    (output/'configs').mkdir()
    shutil.copyfile(root/'configs/pilot.json',output/'configs/pilot.json')
    shutil.copytree(root/'data/problems',output/'data/problems')
    (output/'README.md').write_text('''# Standalone AksharaForge environment

Python 3.11+ standard library only. No PDFs, OCR service, Ollama, network or training libraries are required for reset/step and finite-answer grading.

Run `python -m akshara_forge --split train` from this directory. Send one JSON object per line:

`{"op":"reset","problem_id":"a problem_id from data/problems/.../train.jsonl"}`

`{"op":"step","action":{"steps":["written solution step"],"answer":42}}`

One submission per reset. Responses expose public prompts, grades and feedback. Proofs return `needs_judge` with null reward. The private reference dataset belongs to the evaluator: do not mount this directory in a learner sandbox or include it in model context. A strong judge can be supplied using the Python MathEnvironment callback interface. This package is an environment, not a trainer or inference server.
''')
    files={str(p.relative_to(output)):digest_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'manifest.json',{'files':files,'dependencies':'Python standard library only','proof_grading':'external judge required'})
    return {'output':str(output),'files':len(files)}
