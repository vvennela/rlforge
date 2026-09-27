from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='AksharaForge: frozen paper artifacts and standalone mathematics environment')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    for name in ('import-akshara', 'extract'):
        sub.add_parser(name)
    canonical = sub.add_parser('canonicalize')
    canonical.add_argument('pdf', type=Path)
    canonical.add_argument('--output', type=Path, required=True)
    baseline = sub.add_parser('baseline')
    baseline.add_argument('--split', choices=['train','test','all'], default='test')
    baseline.add_argument('--output', type=Path, required=True)
    baseline.add_argument('--limit', type=int)
    baseline.add_argument('--tokens', type=int, default=1024)
    baseline.add_argument('--model', default='qwen2.5:7b-instruct')
    baseline.add_argument('--sandbox', choices=['auto','macos','docker'], default='auto')
    grade = sub.add_parser('grade')
    grade.add_argument('run', type=Path)
    grade.add_argument('--limit', type=int)
    draft = sub.add_parser('propose')
    draft.add_argument('paper')
    draft.add_argument('--pages', type=int, nargs='+', required=True)
    draft.add_argument('--count', type=int, default=5)
    draft.add_argument('--output', type=Path, required=True)
    serve = sub.add_parser('serve', help='JSON-lines reset/step environment; controller access only')
    serve.add_argument('--split', choices=['train','test'], default='train')
    export = sub.add_parser('export', help='Export a standard-library-only environment bundle')
    export.add_argument('--output', type=Path, required=True)
    comparison = sub.add_parser('compare')
    comparison.add_argument('before', type=Path)
    comparison.add_argument('after', type=Path)
    relevance = sub.add_parser('relevance')
    relevance.add_argument('--semantic', action='store_true', help='Call Luna for mathematical artifact candidates')
    sub.add_parser('browser')
    compile_parser = sub.add_parser('build-environment')
    compile_parser.add_argument('paper')
    compile_parser.add_argument('--count',type=int,default=5)
    compile_parser.add_argument('--output',type=Path,required=True)
    from_pdf_parser=sub.add_parser('from-pdf',help='Canonicalize a new PDF, assess relevance, generate/review tasks, export a QA environment')
    from_pdf_parser.add_argument('pdf',type=Path)
    from_pdf_parser.add_argument('--count',type=int,default=5)
    from_pdf_parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='from-pdf':
        from .pipeline import from_pdf
        print(json.dumps(from_pdf(args.pdf,args.output,args.count),indent=2));return
    if args.command=='canonicalize':
        from .generic import canonicalize
        print(json.dumps(canonicalize(args.pdf,args.output),indent=2));return
    root = args.root.resolve()
    config = json.loads((root / 'configs/pilot.json').read_text())
    papers = config['papers']
    result = None
    if args.command == 'doctor':
        import urllib.request
        try:
            with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=5) as r:
                models = [m['name'] for m in json.load(r)['models']]
        except Exception as exc:
            models = {'error': str(exc)}
        result = {'python': sys.version, 'tools': {k: shutil.which(k) for k in ['ollama','codex','docker','tesseract']},
                  'ollama_models': models, 'papers': papers, 'training': 'Ollama is inference only; weight updates require separate trainer.'}
    elif args.command == 'import-akshara':
        from .akshara import import_frozen
        result = [import_frozen(root/'papers'/f'{p}.pdf', root/'inputs/akshara'/p, root/'canonical-v2'/p, paper_id=p) for p in papers]
    elif args.command == 'extract':
        from .artifacts import extract
        from .visuals import extract_visuals
        result = [{'artifacts':extract(root/'canonical-v2'/p, root/'data/artifacts'/p),
                   'visuals':extract_visuals(root/'canonical-v2'/p, root/'data/visual-artifacts'/p)} for p in papers]
    elif args.command == 'canonicalize':
        from .generic import canonicalize
        result = canonicalize(args.pdf, args.output)
    elif args.command == 'baseline':
        from .baseline import run_baseline
        result = run_baseline(root, papers=papers, split=args.split, model=args.model, output=args.output,
                              limit=args.limit, max_output_tokens=args.tokens, sandbox=args.sandbox,
                              seed=config['seed'], temperature=config['temperature'])
    elif args.command == 'grade':
        from .teacher import grade_run
        result = grade_run(root, args.run, args.limit)
    elif args.command == 'propose':
        from .teacher import propose
        result = propose(root, args.paper, args.pages, args.output, args.count)
    elif args.command == 'export':
        from .bundle import export
        result = export(root, args.output)
    elif args.command == 'compare':
        from .compare import compare
        result = compare(args.before, args.after)
    elif args.command == 'browser':
        from .report import build_browser
        result = {'path':str(build_browser(root))}
    elif args.command == 'relevance':
        from .relevance import classify_semantic
        if not args.semantic:
            parser.error('Use --semantic for actual Luna assessments; existing full-coverage triage is explicitly rule-based.')
        result = {'assessments':classify_semantic(root/'data/artifacts', root/'canonical-v2', root/'data/relevance')}
    elif args.command == 'build-environment':
        from .compiler import compile_environment
        result = compile_environment(root,args.paper,args.output,args.count)
    elif args.command == 'serve':
        from .environment import MathEnvironment, load_problems
        env = MathEnvironment(load_problems(root, papers, args.split))
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request['op'] == 'reset':
                    response = env.reset(request['problem_id'])
                elif request['op'] == 'step':
                    response = env.step(request['action'])
                else:
                    raise ValueError('Operation must be reset or step')
            except Exception as exc:
                response = {'error': str(exc)}
            print(json.dumps(response, allow_nan=False), flush=True)
        return
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
