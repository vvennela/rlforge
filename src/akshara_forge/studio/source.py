"""Bind model quotations to preserved source bytes without correcting source content."""
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
    row['source_quote_alignment']={'method':'whitespace_only','start':match.start(),'end':match.end(),
                                   'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
