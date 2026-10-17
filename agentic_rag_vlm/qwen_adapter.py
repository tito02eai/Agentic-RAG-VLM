"""Small OpenAI-compatible Qwen-VL client with auditable JSON parsing."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import json
import mimetypes
from pathlib import Path
import time
from typing import Any
from urllib import request


@dataclass(frozen=True)
class VLMResult:
    parsed: dict[str, Any] | None
    raw_text: str
    latency_ms: float
    requested_model: str
    response_model: str | None
    usage: dict[str, Any]
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object, tolerating Markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response did not contain a JSON object")


class OpenAICompatibleQwenVL:
    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout_s: float = 120.0,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _image_part(path: Path) -> dict[str, Any]:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    def infer(self, images: list[Path], prompt: str) -> VLMResult:
        content = [self._image_part(path) for path in images]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        encoded = json.dumps(payload).encode("utf-8")
        call = request.Request(self.endpoint, data=encoded, headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with request.urlopen(call, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - start) * 1000.0
        raw = str(body["choices"][0]["message"].get("content", ""))
        parsed = None
        parse_error = None
        try:
            parsed = extract_json_object(raw)
        except ValueError as error:
            parse_error = str(error)
        return VLMResult(
            parsed=parsed,
            raw_text=raw,
            latency_ms=latency_ms,
            requested_model=self.model,
            response_model=body.get("model"),
            usage=dict(body.get("usage") or {}),
            parse_error=parse_error,
        )
