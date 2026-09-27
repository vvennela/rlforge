"""Bind model quotations to preserved source text without correcting source content."""
import hashlib
import re


def bind_quote(row, source):
    quote=row.get('source_quote')
    if not isinstance(quote,str) or not quote.strip():raise ValueError('Source quotation is missing.')
    if quote in source:return
    tokens=quote.split()
    pattern=r'\s+'.join(re.escape(token) for token in tokens)
    matches=list(re.finditer(pattern,source))
    if len(matches)!=1:raise ValueError('Source quotation does not match a unique span of the uploaded document.')
    match=matches[0]
    row['source_quote_submitted']=quote
    row['source_quote']=match.group()
    row['source_quote_alignment']={'method':'whitespace_only','start_char':match.start(),'end_char':match.end(),
                                   'source_text_sha256':hashlib.sha256(source.encode()).hexdigest()}


def passages(source, size=1800):
    """Deterministic contiguous source spans, split at line boundaries when possible."""
    result=[];start=0
    while start<len(source):
        end=min(start+size,len(source))
        if end<len(source):
            boundary=source.rfind('\n',start+size//2,end)
            if boundary>=0:end=boundary+1
        result.append({'id':f'source-{len(result)+1:03}','text':source[start:end],
                       'start_char':start,'end_char':end})
        start=end
    return result


def bind_reference(row,source):
    if 'source_id' not in row:
        bind_quote(row,source)
        return
    by_id={p['id']:p for p in passages(source)}
    identifier=row['source_id']
    if not isinstance(identifier,str) or identifier not in by_id:
        raise ValueError('Unknown source_id; select one of the supplied source passage IDs.')
    passage=by_id[identifier]
    if 'source_quote' in row:row['source_quote_submitted']=row['source_quote']
    row['source_quote']=passage['text']
    row['source_quote_alignment']={'method':'source_id_lookup','start_char':passage['start_char'],
        'end_char':passage['end_char'],'source_text_sha256':hashlib.sha256(source.encode()).hexdigest()}


def answer_schema(value):
    """Describe the response shape without disclosing answer values or array lengths."""
    if value is None:return {}  # A singleton null type would disclose the answer.
    if isinstance(value,bool):return {'type':'boolean'}
    if isinstance(value,int):return {'type':'integer'}
    if isinstance(value,float):return {'type':'number'}
    if isinstance(value,str):return {'type':'string'}
    if isinstance(value,dict):
        if not value:return {'type':'object'}
        return {'type':'object','properties':{k:answer_schema(v) for k,v in value.items()},
                'required':list(value),'additionalProperties':False}
    if isinstance(value,list):
        kinds=[]
        for item in value:
            shape=answer_schema(item)
            if shape not in kinds:kinds.append(shape)
        return {'type':'array','items':kinds[0] if len(kinds)==1 else {'anyOf':kinds} if kinds else {}}
    raise ValueError('Unsupported answer type.')
