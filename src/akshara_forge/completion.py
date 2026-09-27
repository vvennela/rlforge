"""Conservative transport extraction: preserve work; never invent/fix answer values."""
from __future__ import annotations
import json
import re
from .environment import parse_response


def parse_completion(text: str) -> dict:
    original=parse_response(text)
    if not original['parse_error']:
        return {**original,'extraction':'original_json'}
    # Only accept a COMPLETE JSON value at the end of the response, optionally
    # followed by a closing code fence. Do not repair truncated/malformed JSON.
    candidates=[]
    decoder=json.JSONDecoder(parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    for match in re.finditer(r'[\[{]',text):
        start=match.start()
        try:value,length=decoder.raw_decode(text[start:])
        except (ValueError,json.JSONDecodeError):continue
        suffix=text[start+length:].strip()
        if suffix not in {'','```'}:continue
        candidates.append((start,value))
    if candidates:
        start,value=min(candidates,key=lambda x:x[0])
        prefix=re.sub(r'```(?:json)?\s*$','',text[:start]).strip()
        work=[s.strip() for s in re.split(r'\n\s*\n',prefix) if s.strip()]
        if isinstance(value,dict) and 'answer' in value:
            supplied=value.get('steps',[])
            if isinstance(supplied,list) and all(isinstance(s,str) for s in supplied):work+=supplied
            answer=value['answer']
        else:answer=value
        return {'steps':work,'answer':answer,'parse_error':False,
                'extraction':'complete_trailing_json','answer_start_offset':start}
    # A free-text or truncated submission can still establish an intermediate
    # milestone. The unresolved final answer remains verbatim, not reconstructed.
    work=[s.strip() for s in re.split(r'\n\s*\n',text) if s.strip()]
    return {**original,'steps':work,'extraction':'verbatim_text_blocks'}
