from akshara_forge.completion import parse_completion


def test_complete_answer_in_prose_preserves_work_and_values():
    text='Differentiate: f\'(x)=2x.\n\nThus x=0.\n\n```json\n{"x":0}\n```'
    r=parse_completion(text)
    assert r['answer']=={'x':0}
    assert r['steps']==["Differentiate: f'(x)=2x.",'Thus x=0.']
    assert text[r['answer_start_offset']:].startswith('{"x":0}')


def test_no_truncation_repair_or_example_extraction():
    for text in ['The derivative is 2x.\n\n{"answer":','Example {"x":0}. But my answer is different.']:
        r=parse_completion(text)
        assert r['parse_error'] and r['answer']==text and r['extraction']=='verbatim_text_blocks'


def test_nested_envelope_and_strict_json():
    r=parse_completion('Work above.\n\n{"steps":["derive x"],"answer":{"x":2}}')
    assert r['answer']=={'x':2} and r['steps']==['Work above.','derive x']
    assert parse_completion('{"steps":["a"],"answer":2}')['extraction']=='original_json'
