"""Web upload adapter for the existing immutable Akshara OCR/artifact stack."""
import json,zipfile
from pathlib import Path
from collections import Counter
import pymupdf
from ..generic import canonicalize
from ..artifacts import extract
from ..visuals import extract_visuals
from ..akshara import verify
from ..io import digest_file


def prepare(pdf,folder,update):
    with pymupdf.open(pdf) as doc:
        if doc.needs_pass:raise ValueError('Upload an unlocked PDF.')
        if not 1<=len(doc)<=200:raise ValueError('Upload a PDF of 1–200 pages.')
        # Reject oversized raster requests without discarding or resizing source pages.
        if any(p.rect.width*p.rect.height*(200/72)**2>30_000_000 for p in doc):
            raise ValueError('A page exceeds the 200-DPI raster limit. Upload a smaller-format copy.')
        total=len(doc)
    evidence=folder/'evidence';evidence.mkdir()
    canonical=evidence/'canonical';paper='upload-'+digest_file(pdf)[:12]
    update(status='ocr',ocr_completed=0,ocr_total=total,message='Reading every page and preserving source evidence.')
    manifest=canonicalize(pdf,canonical,paper_id=paper,workers=2,
        progress=lambda done,total:update(status='ocr',ocr_completed=done,ocr_total=total,message=f'OCR pages {done} / {total}'))
    update(status='artifacts',message='Extracting statements, equations, figures, and page artifacts.')
    extracted=extract(canonical,evidence/'artifacts')
    visuals=extract_visuals(canonical,evidence/'visual-artifacts')
    integrity=verify(canonical)
    document=json.loads((canonical/'document.json').read_text())
    raw=json.loads((canonical/'ocr/raw.json').read_text())
    methods=dict(Counter(p['method'] for p in raw['pages']))
    summary={'pages':total,'strips':manifest['strips'],'artifacts':extracted['artifact_count'],
             'artifact_kinds':extracted['kinds'],'image_occurrences':visuals['image_occurrences'],
             'methods':methods,'ocr_engine':raw['model'],'source_sha256':manifest['source_sha256'],
             'ocr_sha256':manifest['ocr_sha256'],'integrity':integrity,'page_dpi':200,
             'download':f'/api/generation/{folder.name}/evidence',
             'preview':f'/api/generation/{folder.name}/page/1'}
    (evidence/'summary.json').write_text(json.dumps(summary,indent=2))
    with zipfile.ZipFile(folder/'source-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(evidence.rglob('*')):
            if p.is_file():
                info=zipfile.ZipInfo(str(p.relative_to(evidence)),date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                z.writestr(info,p.read_bytes())
    summary['bundle_sha256']=digest_file(folder/'source-evidence.zip')
    update(ocr=summary,pages=total,message='OCR complete. Source artifacts are ready.')
    records=[json.loads(l) for l in (evidence/'artifacts/index.jsonl').read_text().splitlines()]
    source_artifacts=[{k:r[k] for k in ('artifact_id','kind','page','content')} for r in records if r['kind'] in ('statement_candidate','equation','code_candidate','table','table_candidate')]
    return [{'page':p['page'],'text':p['text']} for p in document['pages']],source_artifacts
