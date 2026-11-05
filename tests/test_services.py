import io
import pytest
from agentic_rag_vlm import services


def test_explicit_and_environment_model_do_not_query_service(monkeypatch):
    monkeypatch.setenv('AGENTIC_VLM_MODEL','env-model')
    monkeypatch.setattr(services,'urlopen',lambda *a,**kw:pytest.fail('unexpected network'))
    assert services.resolve_model('unused','cli-model')=='cli-model'
    assert services.resolve_model('unused')=='env-model'


def test_discovery_uses_identity_of_actual_service(monkeypatch):
    monkeypatch.delenv('AGENTIC_VLM_MODEL',raising=False);seen=[]
    def open_url(url,timeout):
        seen.append(url);return io.BytesIO(b'{"requested_model":"/partner/models/Qwen3.5-4B"}')
    monkeypatch.setattr(services,'urlopen',open_url)
    assert services.resolve_model('http://local:123/v1/chat/completions')=='/partner/models/Qwen3.5-4B'
    assert seen==['http://local:123/health']


@pytest.mark.parametrize('payload',[b'{}',b'{"requested_model":null}',b'{"requested_model":" "}'])
def test_missing_identity_has_no_fake_checkpoint_fallback(monkeypatch,payload):
    monkeypatch.delenv('AGENTIC_VLM_MODEL',raising=False)
    monkeypatch.setattr(services,'urlopen',lambda *a,**kw:io.BytesIO(payload))
    with pytest.raises(ValueError,match='Supply --model'):services.resolve_model('http://local/v1/chat/completions')


def test_endpoint_environment_is_portable(monkeypatch):
    monkeypatch.setenv('AGENTIC_VLM_ENDPOINT','https://partner/v1/chat/completions')
    monkeypatch.setenv('AGENTIC_CLIP_ENDPOINT','http://partner:5678')
    assert services.vlm_endpoint()=='https://partner/v1/chat/completions'
    assert services.clip_endpoint()=='http://partner:5678'
