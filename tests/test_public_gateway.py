from akshara_forge.studio.public_gateway import permitted, trusted_request


def test_judge_can_upload_and_download_but_cannot_change_provider():
    assert permitted('POST', '/api/generate')
    assert permitted('GET', '/api/generation/'+'a'*32+'/evidence')
    assert permitted('GET', '/studio/vendor/katex/fonts/KaTeX_Main-Regular.woff2')
    for method, path in [('POST','/api/provider'), ('POST','/api/agent'), ('GET','/workbench'), ('GET','/api/models'), ('GET','/api/generation/../../secret')]:
        assert not permitted(method, path)


def test_only_the_exact_session_origin_can_write():
    origin = 'https://demo.netbird.example.org'
    assert trusted_request('demo.netbird.example.org', origin, origin, 'POST')
    assert trusted_request('demo.netbird.example.org', None, origin, 'GET')
    assert not trusted_request('demo.netbird.example.org', None, origin, 'POST')
    assert not trusted_request('demo.netbird.example.org', 'https://attacker.example.org', origin, 'POST')
    assert not trusted_request('other.netbird.example.org', None, origin, 'GET')
