"""One-command, paper-agnostic OCR-to-QA-environment pipeline."""
from __future__ import annotations
from pathlib import Path
from .io import digest_file,write_json


def from_pdf(pdf: Path,output: Path,count: int=5) -> dict:
    if not pdf.is_file():raise ValueError('PDF input is missing')
    if not 1<=count<=20:raise ValueError('This prototype compiles 1..20 exercises per request')
    if output.exists():raise ValueError('Choose a new output directory; failed evidence is preserved')
    from .generic import canonicalize
    from .artifacts import extract
    from .visuals import extract_visuals
    from .relevance import classify_all,classify_semantic,classify_relevant_prose_semantic,build_full_coverage
    from .report import build_browser
    from .compiler import compile_environment
    output=output.resolve();paper='paper-'+digest_file(pdf)[:12]
    write_json(output/'configs/pilot.json',{'papers':[paper],'seed':20260926,
               'model':'qwen2.5:7b-instruct','hf_training_model':'Qwen/Qwen2.5-7B-Instruct',
               'max_output_tokens':1024,'temperature':0.2,'purpose':'automatic QA environment compilation, not an uplift benchmark'})
    canonical=output/'canonical-v2'/paper
    print('Canonicalizing complete PDF',flush=True)
    canonicalize(pdf,canonical,paper_id=paper)
    extract(canonical,output/'data/artifacts'/paper)
    extract_visuals(canonical,output/'data/visual-artifacts'/paper)
    print('Assessing mathematical artifact relevance with Luna',flush=True)
    classify_all(output/'data/artifacts',output/'canonical-v2',output/'data/relevance')
    classify_semantic(output/'data/artifacts',output/'canonical-v2',output/'data/relevance')
    classify_relevant_prose_semantic(output/'data/artifacts',output/'canonical-v2',output/'data/relevance')
    build_full_coverage(output/'data/artifacts',output/'data/relevance')
    build_browser(output)
    print('Generating and reviewing exercises with Astra',flush=True)
    result=compile_environment(output,paper,output/'environment',count)
    result={**result,'pipeline_root':str(output),'input_pdf_sha256':digest_file(pdf),
            'ocr':'complete native/Tesseract local acquisition; source mathematics unverified',
            'scope':'A generic math question-answer environment instantiated from source evidence, not arbitrary simulator synthesis'}
    write_json(output/'PIPELINE_COMPLETE.json',result)
    return result
