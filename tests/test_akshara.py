import json
from pathlib import Path
import pymupdf as fitz
import pytest
from akshara_forge.akshara import import_frozen, verify, apply_corrections, text_hash
from akshara_forge.artifacts import extract
from akshara_forge.canonical import strip_ranges, normalize_text
from akshara_forge.io import digest_file, write_json, read_jsonl


@pytest.fixture
def frozen(tmp_path):
    pdf = tmp_path/'paper.pdf'
    with fitz.open() as d:
        for i in range(2):
            p=d.new_page();p.insert_text((40,40),'Header repeated. Theorem example.');
        d.save(pdf)
    source=tmp_path/'source'; raw=source/'ocr/raw.json'
    pages=[{'index':i,'header':'REPEATED HEADER','footer':str(i+1),'markdown':'Theorem. The sum is\n\n$$2+2=4$$\n\nProof. Add two.','images':[],'tables':[]} for i in range(2)]
    write_json(raw, {'model':'frozen-test','pages':pages})
    write_json(source/'provenance/source.json', {'pdf_sha256':digest_file(pdf),'ocr_sha256':digest_file(raw)})
    write_json(source/'manifest.json', {'test':True})
    out=tmp_path/'canonical'
    import_frozen(pdf,source,out,paper_id='example',dpi=72,strip_height=400,overlap=40)
    return pdf,source,out


def test_complete_projection_and_source_spans(frozen,tmp_path):
    _,source,out=frozen
    assert (out/'ocr/raw.json').read_bytes()==(source/'ocr/raw.json').read_bytes()
    doc=json.loads((out/'document.json').read_text())
    assert all('REPEATED HEADER' in p['text'] for p in doc['pages'])
    assert verify(out)['pages_verified']==2
    result=extract(out,tmp_path/'artifacts')
    artifacts=read_jsonl(tmp_path/'artifacts/index.jsonl')
    assert result['kinds']['equation']==2
    for a in artifacts:
        if a['start'] is not None:
            assert doc['pages'][a['page']-1]['body_markdown'][a['start']:a['end']]==a['content']


def test_immutable_output_and_hash_tampering(frozen):
    pdf,source,out=frozen
    with pytest.raises(ValueError,match='Immutable'):
        import_frozen(pdf,source,out,paper_id='example',dpi=73)
    (out/'ocr/page-0001.md').write_text('corruption')
    with pytest.raises(ValueError,match='mismatch'):verify(out)


def test_exact_correction_layer_and_stale_rejection(frozen,tmp_path):
    _,_,out=frozen
    d=json.loads((out/'document.json').read_text())
    correction={'page':1,'body_sha256':d['pages'][0]['body_sha256'],'original':'Add two.',
                'replacement':'Add two integers.','reviewer':'test','reason':'test-only example'}
    ledger={'source_sha256':d['source_sha256'],'ocr_sha256':d['ocr_sha256'],'corrections':[correction]}
    path=tmp_path/'ledger.json';write_json(path,ledger)
    original=(out/'document.json').read_bytes()
    apply_corrections(out,path,tmp_path/'reading')
    assert (out/'document.json').read_bytes()==original
    assert 'Add two integers.' in (tmp_path/'reading/document.md').read_text()
    correction['body_sha256']='bad';write_json(path,ledger)
    with pytest.raises(ValueError,match='Stale'):apply_corrections(out,path,tmp_path/'reading2')


def test_strips_cover_full_page_and_preserve_math_unicode():
    ranges=strip_ranges(2339,1000,100)
    assert ranges[0][0]==0 and ranges[-1][1]==2339
    assert all(a[1]-b[0]==100 for a,b in zip(ranges,ranges[1:]))
    assert normalize_text('x² + ﬁ')=='x² + fi'
