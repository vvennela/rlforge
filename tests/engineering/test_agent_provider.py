import json
import httpx
import pytest
from akshara_forge.engineering import agent

class Episode:
 def __init__(self):self.episode='test';self.actions=[]
 def observe(self):return {'done':bool(self.actions),'evaluation':{'success':bool(self.actions)}}
 def step(self,actions,actor):self.actions.append((actions,actor));return self.observe()

@pytest.mark.parametrize('returned_model,expected_steps',[('qwen3.8-27b',1),('different-model',0),(None,0)])
def test_vultr_identity_and_secret_handling(monkeypatch,tmp_path,returned_model,expected_steps):
 monkeypatch.setenv('AKSHARA_AGENT_PROVIDER','vultr');monkeypatch.setenv('VULTR_INFERENCE_API_KEY','test-secret-not-for-logs');monkeypatch.delenv('AKSHARA_AGENT_MODEL',raising=False)
 def handler(request):
  assert request.headers['Authorization']=='Bearer test-secret-not-for-logs'
  if request.url.path.endswith('/models'):return httpx.Response(200,json={'data':[{'id':'qwen3.8-27b','hugging_face_id':'Qwen/example'}]})
  payload=json.loads(request.content);assert payload['model']=='qwen3.8-27b'
  return httpx.Response(200,json={'model':returned_model,'choices':[{'message':{'content':'{"actions":[{"tool":"remove","id":"a"}]}'}}]})
 client=httpx.Client
 monkeypatch.setattr(agent.httpx,'Client',lambda **kw:client(transport=httpx.MockTransport(handler),**kw))
 env=Episode();agent.repair(env,tmp_path,rounds=1)
 assert len(env.actions)==expected_steps
 assert all('test-secret-not-for-logs' not in p.read_text() for p in tmp_path.glob('*.json'))
 if expected_steps:assert env.actions[0][1]=='model:qwen3.8-27b'

def test_missing_key_fails_before_rollout(monkeypatch,tmp_path):
 monkeypatch.setenv('AKSHARA_AGENT_PROVIDER','vultr');monkeypatch.delenv('VULTR_INFERENCE_API_KEY',raising=False)
 with pytest.raises(ValueError,match='not configured'):agent.repair(Episode(),tmp_path)


def test_agent_uses_shared_server_secret(monkeypatch,tmp_path):
 config=tmp_path/'inference.json'
 config.write_text(json.dumps({'provider':'vultr','model':'glm-5.3','key':'shared-private-key'}))
 monkeypatch.setenv('AKSHARA_INFERENCE_CONFIG',str(config))
 monkeypatch.setenv('AKSHARA_AGENT_PROVIDER','vultr')
 monkeypatch.setenv('AKSHARA_AGENT_MODEL','glm-5.3')
 monkeypatch.delenv('VULTR_INFERENCE_API_KEY',raising=False)
 assert agent.configuration()=={'provider':'vultr','model':'glm-5.3','configured':True}
 assert 'shared-private-key' not in json.dumps(agent.configuration())
