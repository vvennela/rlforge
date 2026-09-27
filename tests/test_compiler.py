import json
import pytest
from akshara_forge.compiler import select_pages,compile_environment


def test_page_selection_rejects_stale_relevance(tmp_path):
    a=tmp_path/'data/artifacts/p';a.mkdir(parents=True)
    (a/'index.jsonl').write_text(json.dumps({'artifact_id':'x','content_sha256':'new','kind':'equation'})+'\n')
    r=tmp_path/'data/relevance';r.mkdir()
    (r/'semantic-candidates.jsonl').write_text(json.dumps({'artifact_id':'x','paper_id':'p','relevance':'high','content_sha256':'old','page':1})+'\n')
    with pytest.raises(ValueError,match='Stale'):select_pages(tmp_path,'p')


def test_cached_environment_must_match_requested_source_and_count(tmp_path,monkeypatch):
    d=tmp_path/'canonical-v2/p';d.mkdir(parents=True)
    (d/'document.json').write_text(json.dumps({'source_sha256':'pdf','ocr_sha256':'ocr'}))
    o=tmp_path/'out';o.mkdir();(o/'COMPLETE.json').write_text('{}')
    (o/'compile-request.json').write_text(json.dumps({'paper_id':'p','count_requested':5,'source_sha256':'pdf','ocr_sha256':'ocr'}))
    monkeypatch.setattr('akshara_forge.compiler.verify',lambda _:None)
    with pytest.raises(ValueError,match='different inputs'):compile_environment(tmp_path,'p',o,count=6)
