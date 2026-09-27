import importlib.util
from pathlib import Path
import urllib.error
import pytest

spec=importlib.util.spec_from_file_location('netbird_session',Path(__file__).parents[1]/'deploy/vultr/netbird-session.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.mark.parametrize('error',[TimeoutError(),ConnectionResetError(),urllib.error.URLError('temporary'),urllib.error.HTTPError('http://local',503,'unavailable',{},None)])
def test_health_observation_failure_does_not_end_session(monkeypatch,error):
    calls=[]
    def api(method,path,body=None):
        calls.append(method)
        if len(calls)==1:raise error
        return {'enabled':True}
    monkeypatch.setattr(module,'api',api)
    assert module.check_health('service') is False
    assert module.check_health('service') is True
    assert calls==['GET','GET']  # Never DELETE the service after a temporary read failure.

def test_explicit_disable_is_still_respected(monkeypatch):
    monkeypatch.setattr(module,'api',lambda *args:{'enabled':False})
    with pytest.raises(RuntimeError,match='disabled'):module.check_health('service')
