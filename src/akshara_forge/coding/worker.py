"""Runs only inside an unprivileged, networkless container. No expected answers here."""
import contextlib, io, json, sys, copy
request=json.loads(sys.stdin.readline(100001));results=[]
try:
    namespace={}
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        exec(compile(request['code'],'candidate.py','exec'),namespace)
    fn=namespace['solve']
    for item in request['inputs']:
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                value=fn(copy.deepcopy(item))
            json.dumps(value,allow_nan=False)
            results.append({'value':value})
        except Exception as exc:results.append({'error':type(exc).__name__})
    encoded=json.dumps({'results':results},allow_nan=False)
    if len(encoded)>60000:raise ValueError('Output too large')
    print(encoded,flush=True)
except BaseException as exc:
    print(json.dumps({'error':type(exc).__name__}),flush=True)
