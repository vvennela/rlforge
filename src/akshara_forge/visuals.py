"""Preserve provider image bytes and validated 300-DPI source-region crops."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import pymupdf as fitz
from .akshara import verify, inventory
from .io import digest_file, write_json, write_jsonl


def extract_visuals(canonical: Path, output: Path) -> dict:
    verify(canonical)
    if output.exists():
        if json.loads((output/'checksums.json').read_text()) != inventory(output):
            raise ValueError('Visual artifacts changed')
        return json.loads((output/'manifest.json').read_text())
    raw=json.loads((canonical/'ocr/raw.json').read_text())
    source=json.loads((canonical/'manifest.json').read_text())
    output.mkdir(parents=True)
    records=[]
    with fitz.open(canonical/'original.pdf') as pdf:
        for provider in raw['pages']:
            page=pdf[provider['index']]
            dimensions=provider.get('dimensions') or {}
            for number,item in enumerate(provider.get('images',[]),1):
                prefix=f"page-{provider['index']+1:04d}-image-{number:03d}"
                record={'page':provider['index']+1,'provider_id':item.get('id'),
                        'source_sha256':source['source_sha256'],'ocr_sha256':source['ocr_sha256'],
                        'source_image':str((canonical/f"pages/page-{provider['index']+1:04d}.png").resolve()),
                        'semantic_association':'unverified','status':'requires_review'}
                encoded=item.get('image_base64')
                if encoded:
                    metadata,payload=encoded.split(',',1) if ',' in encoded else ('',encoded)
                    suffix='.jpeg' if 'image/jpeg' in metadata else '.png' if 'image/png' in metadata else '.bin'
                    target=output/(prefix+'-provider'+suffix)
                    target.write_bytes(base64.b64decode(payload,validate=True))
                    record.update(provider_image=target.name,provider_image_sha256=digest_file(target))
                try:
                    w,h=float(dimensions['width']),float(dimensions['height'])
                    x0,y0,x1,y1=[float(item[k]) for k in ['top_left_x','top_left_y','bottom_right_x','bottom_right_y']]
                    if not 0<=x0<x1<=w or not 0<=y0<y1<=h:
                        raise ValueError('Provider crop outside page bounds')
                    if abs((w/h)/(page.rect.width/page.rect.height)-1)>.03:
                        raise ValueError('Provider/source page aspect ratio mismatch')
                    rect=fitz.Rect(x0/w*page.rect.width,y0/h*page.rect.height,x1/w*page.rect.width,y1/h*page.rect.height)
                    target=output/(prefix+'-source-300dpi.png')
                    page.get_pixmap(dpi=300,clip=rect,colorspace=fitz.csRGB,alpha=False).save(target)
                    record.update(source_crop=target.name,source_crop_sha256=digest_file(target),
                                  provider_bbox=[x0,y0,x1,y1],source_bbox_points=list(rect),
                                  status='coordinate_validated_not_semantically_verified')
                except (ValueError,KeyError,TypeError,ZeroDivisionError) as exc:
                    record['crop_review_reason']=str(exc)
                records.append(record)
    write_jsonl(output/'index.jsonl',records)
    result={'image_occurrences':len(records),'source_crops':sum('source_crop' in r for r in records),
            'source_sha256':source['source_sha256'],'ocr_sha256':source['ocr_sha256'],
            'scope':'All provider-detected image occurrences; full source pages preserve undetected figures.'}
    write_json(output/'manifest.json',result)
    write_json(output/'checksums.json',inventory(output))
    return result
