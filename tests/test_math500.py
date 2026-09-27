import json
import pytest

from akshara_forge.math500 import messages, score, freeze, completed_prefix, compare


def test_reference_never_reaches_model():
    text = json.dumps(messages({'problem': 'Compute 1+1.', 'answer': 'SECRET', 'solution': 'PRIVATE'}))
    assert 'SECRET' not in text and 'PRIVATE' not in text
    assert 'Compute 1+1.' in text


def test_equivalent_answer_and_wrong_final():
    pytest.importorskip('math_verify')
    assert score(r'\frac{1}{2}', r'The result is \boxed{0.5}.')['correct']
    assert not score('2', r'I considered 2, but my final answer is \boxed{3}.')['correct']
    assert not score('2', 'I do not know.')['correct']


def test_protocol_and_resume_guard(tmp_path):
    p = tmp_path/'protocol.json'
    freeze(p, {'tokens': 4096})
    with pytest.raises(ValueError, match='differs'):
        freeze(p, {'tokens': 1024})
    rows = [{'unique_id': str(i)} for i in range(3)]
    assert completed_prefix(rows, rows[:2]) == 2
    with pytest.raises(ValueError, match='prefix'):
        completed_prefix(rows, [rows[1]])


def test_compare_requires_complete_matched_protocol(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    for p in (a, b):
        p.mkdir()
        (p/'protocol.json').write_text('{}')
        (p/'runtime.json').write_text('{}')
        (p/'episodes.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='500'):
        compare(a, b)
    (b/'protocol.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='mismatch'):
        compare(a, b)
