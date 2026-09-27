import json
import platform
import subprocess
import sys
from pathlib import Path
import pytest
from akshara_forge.sandbox import mac_profile, run_learner


@pytest.mark.skipif(platform.system()!='Darwin',reason='macOS seatbelt integration')
def test_real_os_sandbox_blocks_private_reads_writes_and_other_ports(tmp_path):
    sentinel=tmp_path/'private-reference.txt';sentinel.write_text('private answer')
    destination=tmp_path/'unauthorized-output'
    code='''import json,socket
results={}
for name,path,mode in [('read',READ,'r'),('write',WRITE,'w')]:
 try:
  with open(path,mode) as f:
   if mode=='r':f.read()
   else:f.write('bad')
  results[name]='allowed'
 except PermissionError:results[name]='blocked'
try:
 s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',11435));results['network']='allowed'
except PermissionError:results['network']='blocked'
except OSError as e:results['network']='not_permission_error:'+str(e)
print(json.dumps(results))
'''.replace('READ',repr(str(sentinel))).replace('WRITE',repr(str(destination)))
    result=subprocess.run(['/usr/bin/sandbox-exec','-p',mac_profile(),sys.executable,'-I','-B','-c',code],
                          capture_output=True,text=True,cwd='/tmp',timeout=10)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=={'read':'blocked','write':'blocked','network':'blocked'}
    assert not destination.exists()


def test_no_unsandboxed_fallback():
    with pytest.raises(ValueError,match='no unsandboxed'):run_learner({},backend='none')
