"""Source-bound artifact extraction; semantic relevance and correctness are later stages."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from .akshara import inventory, text_hash, verify
from .io import write_json, write_jsonl


def extract(canonical: Path, output: Path) -> dict:
    verify(canonical)
    if output.exists():
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest['canonical_sha256'] != hashlib.sha256((canonical / 'document.json').read_bytes()).hexdigest():
            raise ValueError('Artifacts belong to other frozen canonical data')
        expected = json.loads((output / 'checksums.json').read_text())
        if expected != inventory(output):
            raise ValueError('Artifact inventory modified')
        return manifest
    doc = json.loads((canonical / 'document.json').read_text())
    raw = json.loads((canonical / 'ocr/raw.json').read_text())
    records = []
    def emit(page, kind, content, *, start=None, end=None, field='body_markdown', details=None):
        binding = {'paper_id': doc['paper_id'], 'page': page['page'], 'kind': kind,
                   'field': field, 'start': start, 'end': end, 'content_sha256': text_hash(content)}
        artifact_id = f"{doc['paper_id']}-p{page['page']:04d}-{kind}-" + text_hash(json.dumps(binding, sort_keys=True))[:12]
        record = {**binding, 'artifact_id': artifact_id, 'content': content,
                  'source_sha256': doc['source_sha256'], 'ocr_sha256': doc['ocr_sha256'],
                  'page_body_sha256': page['body_sha256'], 'source_image': str((canonical / page['image']).resolve()),
                  'source_strips': [str((canonical / s['image']).resolve()) for s in page['strips']],
                  'status': 'extracted_candidate_unverified',
                  'text_path': f'{kind}/{artifact_id}.txt', 'details': details or {}}
        path = output / record['text_path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        records.append(record)
    for page, provider in zip(doc['pages'], raw['pages']):
        body = page['body_markdown']
        emit(page, 'page', page['text'], field='full_page')
        for field in ('header', 'footer'):
            if page[field]:
                emit(page, field, page[field], field=field)
        # Exact character offsets into unchanged provider body; no math repair.
        for match in re.finditer(r'\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]', body):
            emit(page, 'equation', match.group(), start=match.start(), end=match.end())
        for match in re.finditer(r'(?m)^#{1,6}[ \t]+[^\n]+', body):
            emit(page, 'heading', match.group(), start=match.start(), end=match.end())
        for match in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)', body):
            text = match.group()
            kind = 'paragraph'
            if re.search(r'\b(theorem|lemma|proposition|corollary|definition|proof)\b', text, re.I):
                kind = 'statement_candidate'
            elif re.match(r'(?i)\s*(figure|fig\.|table)\s+\w', text):
                kind = 'caption_candidate'
            elif text.count('|') >= 4:
                kind = 'table_candidate'
            elif text.startswith('```'):
                kind = 'code_candidate'
            emit(page, kind, text, start=match.start(), end=match.end(),
                 details={'detection': 'syntax_only; not a verified semantic classification'})
        for i, figure in enumerate(provider.get('images', [])):
            description = {k:v for k,v in figure.items() if k != 'image_base64'}
            emit(page, 'figure', json.dumps(description, ensure_ascii=False), field=f'provider.images[{i}]',
                 details={'image_bytes_preserved_in_canonical_raw': True, 'association_verified': False})
        for i, table in enumerate(provider.get('tables', [])):
            emit(page, 'table', json.dumps(table, ensure_ascii=False), field=f'provider.tables[{i}]',
                 details={'association_verified': False})
    write_jsonl(output / 'index.jsonl', records)
    from collections import Counter
    manifest = {'schema_version': 'akshara-forge.artifacts.v1', 'paper_id': doc['paper_id'],
                'canonical_sha256': hashlib.sha256((canonical / 'document.json').read_bytes()).hexdigest(),
                'source_sha256': doc['source_sha256'], 'ocr_sha256': doc['ocr_sha256'],
                'source_pages': doc['page_count'], 'artifact_count': len(records),
                'kinds': dict(Counter(r['kind'] for r in records)),
                'coverage': 'Every page retained; syntax-detected artifacts may overlap and are not exhaustive semantic theorem extraction.',
                'relevance': 'not_yet_assessed', 'mathematical_accuracy': 'not_yet_verified'}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'checksums.json', inventory(output))
    return manifest
