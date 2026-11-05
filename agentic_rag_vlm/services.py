"""Portable inference configuration; no workstation-specific checkpoint paths."""
import json
import os
from urllib.request import urlopen


def vlm_endpoint():return os.environ.get('AGENTIC_VLM_ENDPOINT','http://127.0.0.1:18074/v1/chat/completions')
def clip_endpoint():return os.environ.get('AGENTIC_CLIP_ENDPOINT','http://127.0.0.1:18072')


def health_url(endpoint):
    suffix='/v1/chat/completions'
    if not endpoint.rstrip('/').endswith(suffix):
        raise ValueError('VLM endpoint must include /v1/chat/completions')
    return endpoint.rstrip('/')[:-len(suffix)]+'/health'


def resolve_model(endpoint,model=None):
    """Explicit CLI > environment > identity reported by our inference service.

    No model aliases or remote hardware are queried when an explicit model is
    supplied. The frozen benchmark separately checks service/checkpoint identity.
    """
    explicit=model or os.environ.get('AGENTIC_VLM_MODEL')
    if explicit and explicit.strip():return explicit.strip()
    try:
        with urlopen(health_url(endpoint),timeout=10) as response:payload=json.load(response)
        value=payload.get('requested_model')
        if isinstance(value,str) and value.strip():return value.strip()
        raise ValueError('service did not report requested_model')
    except Exception as error:
        raise ValueError('Cannot resolve the served model. Supply --model or AGENTIC_VLM_MODEL, '
                         'or start serve_local_vlm.py with its /health endpoint. '+str(error)) from error
