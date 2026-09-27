"""Offline, immutable import of Akshara's complete frozen OCR evidence.

Raw provider output is authoritative evidence, not a verified mathematical text.
Reviewed replacements produce a separate reading layer with exact hash binding.
"""
from __future__ import annotations

import base64
import hashlib
import json
import platform
import shutil
from pathlib import Path

import pymupdf as fitz

from .canonical import strip_ranges
from .io import digest_file, read_jsonl, write_json, write_jsonl


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def inventory(out: Path) -> dict:
    return {str(p.relative_to(out)): digest_file(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name != 'checksums.json'}


def verify(out: Path) -> dict:
    expected = json.loads((out / 'checksums.json').read_text())
    actual = inventory(out)
    if expected != actual:
        raise ValueError('Artifact inventory/hash mismatch; frozen output has changed')
    doc = json.loads((out / 'document.json').read_text())
    raw = json.loads((out / 'ocr/raw.json').read_text())
    if [p['index'] for p in raw['pages']] != list(range(doc['page_count'])):
        raise ValueError('OCR pages are missing, duplicated or out of order')
    for page, source in zip(doc['pages'], raw['pages']):
        if page['body_markdown'] != source['markdown'] or page['header'] != source.get('header') or page['footer'] != source.get('footer'):
            raise ValueError('Complete OCR projection changed')
    return {'files_verified': len(actual), 'pages_verified': doc['page_count'],
            'mathematical_accuracy': 'not_established_by_integrity_checks'}


def import_frozen(pdf: Path, frozen: Path, out: Path, *, paper_id: str, dpi: int = 200,
                  strip_height: int = 1000, overlap: int = 100) -> dict:
    source = json.loads((frozen / 'provenance/source.json').read_text())
    raw_path = frozen / 'ocr/raw.json'
    if digest_file(pdf) != source['pdf_sha256'] or digest_file(raw_path) != source['ocr_sha256']:
        raise ValueError('Frozen PDF/OCR differs from Akshara source receipt')
    raw = json.loads(raw_path.read_text())
    toolchain = {'python': platform.python_version(), 'pymupdf': fitz.VersionBind,
                 'platform': platform.system(), 'machine': platform.machine(),
                 'importer_sha256': digest_file(Path(__file__)),
                 'strip_code_sha256': digest_file(Path(__file__).with_name('canonical.py'))}
    settings = {'dpi': dpi, 'strip_height': strip_height, 'overlap': overlap,
                'pdf_sha256': source['pdf_sha256'], 'ocr_sha256': source['ocr_sha256'],
                'source_receipt_sha256': digest_file(frozen / 'provenance/source.json'),
                'toolchain': toolchain}
    fingerprint = text_hash(json.dumps(settings, sort_keys=True))
    if out.exists():
        manifest = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
        if manifest.get('fingerprint') != fingerprint:
            raise ValueError('Immutable output exists with other inputs/settings/toolchain; use a new output directory')
        verify(out)
        return {**manifest, 'cached': True}
    with fitz.open(pdf) as document:
        count = len(document)
        if [p.get('index') for p in raw['pages']] != list(range(count)):
            raise ValueError('Require every source page and contiguous zero-based provider page indexes')
        stage = out.with_name(out.name + '.staging')
        if stage.exists():
            raise ValueError(f'Incomplete staging directory exists: {stage}; inspect before retrying')
        for directory in ('ocr', 'pages', 'strips', 'machine', 'images', 'provenance', 'native'):
            (stage / directory).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, stage / 'original.pdf')
        shutil.copyfile(raw_path, stage / 'ocr/raw.json')
        for p in (frozen / 'provenance').glob('*.json'):
            shutil.copyfile(p, stage / 'provenance' / ('upstream-' + p.name))
        shutil.copyfile(frozen / 'manifest.json', stage / 'provenance/upstream-manifest.json')
        pages, figures, tables = [], [], []
        markdown = [f'# {paper_id}', '', 'Complete frozen OCR; mathematical accuracy remains unverified.', '']
        for i, provider in enumerate(raw['pages']):
            page = document[i]
            name = f'page-{i+1:04d}'
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            pix.save(stage / 'pages' / (name + '.png'))
            (stage / 'native' / (name + '.txt')).write_text(page.get_text('text', sort=True))
            body = provider['markdown']
            full = '\n\n'.join(x for x in (provider.get('header'), body, provider.get('footer')) if x is not None)
            (stage / 'ocr' / (name + '.md')).write_text(full)
            strips = []
            for j, (top, bottom) in enumerate(strip_ranges(pix.height, strip_height, overlap), 1):
                image = f'strips/{name}-strip-{j:02d}.png'
                scale = dpi / 72
                clip = fitz.Rect(0, top / scale, page.rect.width, min(bottom / scale, page.rect.height))
                page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csRGB, alpha=False).save(stage / image)
                strips.append({'image': image, 'bbox_pixels': [0, top, pix.width, bottom],
                               'source_page': i+1, 'page_text': f'ocr/{name}.md',
                               'text_alignment': 'page_level_only; no provider word coordinates'})
            for image_number, record in enumerate(provider.get('images', [])):
                occurrence = {'page': i+1, 'provider_record': record, 'status': 'unreviewed_provider_occurrence'}
                encoded = record.get('image_base64')
                if encoded:
                    payload = encoded.split(',', 1)[-1]
                    image_bytes = base64.b64decode(payload, validate=True)
                    artifact = f'images/{name}-image-{image_number+1:03d}.bin'
                    (stage / artifact).write_bytes(image_bytes)
                    occurrence['artifact'] = artifact
                figures.append(occurrence)
            for record in provider.get('tables', []):
                tables.append({'page': i+1, 'provider_record': record, 'status': 'unreviewed_provider_table'})
            pages.append({'page': i+1, 'image': f'pages/{name}.png', 'width_pixels': pix.width,
                          'height_pixels': pix.height, 'method': raw.get('model'),
                          'body_markdown': body, 'header': provider.get('header'), 'footer': provider.get('footer'),
                          'text': full, 'raw_text': full, 'body_sha256': text_hash(body), 'strips': strips,
                          'flags': ['mathematical_notation_unverified'] + (['sparse_page_requires_visual_review'] if len(body.strip()) < 80 else [])})
            markdown += [f'## PDF page {i+1}', f'Source image: pages/{name}.png', '', full, '']
        doc = {'schema_version': 'akshara-forge.frozen.v2', 'paper_id': paper_id, 'page_count': count,
               'source_sha256': source['pdf_sha256'], 'ocr_sha256': source['ocr_sha256'],
               'settings': settings, 'pages': pages, 'status': 'complete_ocr_unverified_mathematics'}
        write_json(stage / 'document.json', doc)
        write_json(stage / 'machine/document.json', doc)
        write_jsonl(stage / 'machine/pages.jsonl', pages)
        write_jsonl(stage / 'machine/tables.jsonl', tables)
        write_jsonl(stage / 'images/figures.jsonl', figures)
        (stage / 'document.md').write_text('\n'.join(markdown))
        write_json(stage / 'provenance/toolchain.json', toolchain)
        write_json(stage / 'provenance/quality.json', {'complete_source_review': False,
                   'submission_ready': False, 'notes': ['Raw OCR is unchanged.', 'No scientific corrections applied.',
                   'All full pages retained; no headers, footers, references or repeated paragraphs removed.',
                   'Source-region semantic detection and word-level alignment are not inferred.']})
        manifest = {'fingerprint': fingerprint, 'paper_id': paper_id, 'page_count': count,
                    'strips': sum(len(p['strips']) for p in pages), 'ocr_model': raw.get('model'),
                    'source_sha256': source['pdf_sha256'], 'ocr_sha256': source['ocr_sha256'],
                    'document_sha256': digest_file(stage / 'document.json'), 'settings': settings}
        write_json(stage / 'manifest.json', manifest)
        write_json(stage / 'checksums.json', inventory(stage))
        verify(stage)
        stage.rename(out)
    return manifest


def apply_corrections(canonical: Path, ledger_path: Path, out: Path) -> dict:
    """Exact-span corrections only; never mutates canonical/raw evidence."""
    verify(canonical)
    if out.exists():
        raise ValueError('Reviewed reading output must be new')
    document = json.loads((canonical / 'document.json').read_text())
    ledger = json.loads(ledger_path.read_text())
    if ledger['source_sha256'] != document['source_sha256'] or ledger['ocr_sha256'] != document['ocr_sha256']:
        raise ValueError('Correction ledger bound to different PDF/OCR')
    edits = {}
    for correction in ledger['corrections']:
        index = correction['page'] - 1
        page = document['pages'][index]
        if correction['body_sha256'] != page['body_sha256']:
            raise ValueError('Stale page hash')
        original = page['body_markdown']
        old = correction['original']
        if not old or original.count(old) != 1:
            raise ValueError('Correction must match exactly one original span')
        if not correction.get('reviewer') or not correction.get('reason'):
            raise ValueError('Corrections require reviewer and reason')
        start = original.index(old)
        for prior in edits.get(index, []):
            if start < prior[1] and prior[0] < start + len(old):
                raise ValueError('Overlapping corrections')
        edits.setdefault(index, []).append((start, start+len(old), correction['replacement']))
    for index, changes in edits.items():
        page = document['pages'][index]
        text = page['body_markdown']
        for start, end, replacement in sorted(changes, reverse=True):
            text = text[:start] + replacement + text[end:]
        page['body_markdown'] = text
        page['text'] = '\n\n'.join(x for x in (page['header'], text, page['footer']) if x is not None)
    write_jsonl(out / 'pages.jsonl', document['pages'])
    write_json(out / 'corrections.json', ledger)
    (out / 'document.md').write_text('\n\n'.join(f"## PDF page {p['page']}\n\n{p['text']}" for p in document['pages']))
    write_json(out / 'checksums.json', inventory(out))
    return {'corrected_spans': len(ledger['corrections']), 'raw_inputs_unchanged': True}
