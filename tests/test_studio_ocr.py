import json,shutil,zipfile
import pymupdf,pytest
from akshara_forge.studio.ocr import prepare
from akshara_forge.akshara import verify

@pytest.mark.skipif(not shutil.which('tesseract'),reason='Tesseract required for real scanned-page integration')
def test_mixed_native_and_scanned_pdf_preserves_all_evidence(tmp_path):
    text='Theorem 1. The sum of two even integers is even. Proof: write the integers as twice two integers and add them. This sentence is source evidence.'
    pdf=pymupdf.open();p=pdf.new_page(width=420,height=500);p.insert_textbox(pymupdf.Rect(30,40,390,440),text,fontsize=16)
    image=p.get_pixmap(dpi=200).tobytes('png')
    scan=pdf.new_page(width=420,height=500);scan.insert_image(scan.rect,stream=image)
    source=tmp_path/'source.pdf';pdf.save(source);pdf.close()
    job=tmp_path/'job';job.mkdir();updates=[]
    pages,artifacts=prepare(source,job,lambda **x:updates.append(x))
    assert len(pages)==2
    assert 'Theorem' in pages[0]['text'] and 'Theorem' in pages[1]['text']
    receipt=updates[-1]['ocr']
    assert receipt['methods']=={'native_pdf_text':1,'tesseract':1}
    assert receipt['pages']==2 and receipt['strips']>=4
    assert any(r['kind']=='statement_candidate' for r in artifacts)
    assert verify(job/'evidence/canonical')['pages_verified']==2
    assert (job/'evidence/canonical/original.pdf').read_bytes()==source.read_bytes()
    with zipfile.ZipFile(job/'source-evidence.zip') as z:
        assert 'canonical-acquisition/raw/page-0002.ocr.tsv' in z.namelist()
        assert 'artifacts/index.jsonl' in z.namelist()
        assert 'canonical/pages/page-0002.png' in z.namelist()
    assert [u['ocr_completed'] for u in updates if 'ocr_completed' in u]==[0,1,2]
    with pytest.raises(FileExistsError):prepare(source,job,lambda **x:None)
