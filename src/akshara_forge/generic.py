"""Local acquisition adapter for arbitrary PDFs without an existing OCR receipt.

Preserves full native text or raw Tesseract TSV, then reuses the same immutable
Akshara import contract. This local OCR fallback is not mathematically reliable.
"""
from __future__ import annotations
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pymupdf as fitz
from .canonical import _page
from .akshara import import_frozen
from .io import digest_file, write_json


def canonicalize(pdf: Path, out: Path, *, paper_id: str | None=None, workers: int=3, progress=None) -> dict:
    pdf=pdf.resolve();paper_id=paper_id or pdf.stem
    if out.exists():
        raise ValueError('Use a new immutable output directory for new PDF acquisition')
    acquisition=out.with_name(out.name+'-acquisition')
    if acquisition.exists():
        raise ValueError('Acquisition directory already exists; preserve it and choose a new output')
    for folder in ['raw','pages','strips','ocr','provenance']:
        (acquisition/folder).mkdir(parents=True,exist_ok=True)
    with fitz.open(pdf) as doc:count=len(doc)
    pages=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(_page,str(pdf),i,str(acquisition),200,1000,100,False) for i in range(count)]
        for job in as_completed(jobs):
            pages.append(job.result())
            if progress:progress(len(pages),count)
    pages.sort(key=lambda p:p['page'])
    engine=subprocess.run(['tesseract','--version'],capture_output=True,text=True).stdout.splitlines()[0] if shutil.which('tesseract') else None
    raw={'model':'local-native-text-with-tesseract-fallback','engine_version':engine,
         'pages':[{'index':p['page']-1,'markdown':p['raw_text'],'header':None,'footer':None,
                   'images':[],'tables':[],'method':p['method'],'lines':p['lines'],
                   'dimensions':{'dpi':200,'width':p['width_pixels'],'height':p['height_pixels']}} for p in pages]}
    write_json(acquisition/'ocr/raw.json',raw)
    write_json(acquisition/'provenance/source.json',{'id':paper_id,'pdf_sha256':digest_file(pdf),
                'ocr_sha256':digest_file(acquisition/'ocr/raw.json'),
                'ocr_model':raw['model'],'ocr_engine_version':engine})
    write_json(acquisition/'manifest.json',{'pages':count,'status':'local_acquisition_not_mathematically_verified',
               'raw_evidence':{str(p.relative_to(acquisition)):digest_file(p) for p in sorted((acquisition/'raw').glob('*'))}})
    return import_frozen(pdf,acquisition,out,paper_id=paper_id)
