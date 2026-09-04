"""Contract tests for the Inworld TTS provider."""

from __future__ import annotations

import base64
from pathlib import Path

from tools.audio.inworld_tts import InworldTTS
from tools.base_tool import ToolStatus


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def test_registry_discovers_inworld_tts(monkeypatch, isolated_tool_registry):
    monkeypatch.delenv("INWORLD_API_KEY", raising=False)
    isolated_tool_registry.discover("tools")
    tool = isolated_tool_registry.get("inworld_tts")
    assert tool is not None
    assert tool.capability == "tts"
    assert tool.provider == "inworld"


def test_status_and_cost(monkeypatch):
    monkeypatch.delenv("INWORLD_API_KEY", raising=False)
    tool = InworldTTS()
    assert tool.get_status() == ToolStatus.UNAVAILABLE
    monkeypatch.setenv("INWORLD_API_KEY", "test-key")
    assert tool.get_status() == ToolStatus.AVAILABLE
    assert tool.estimate_cost({"text": "a" * 1000}) == 0.025


def test_rejects_long_text_before_request(monkeypatch):
    monkeypatch.setenv("INWORLD_API_KEY", "test-key")
    monkeypatch.setattr(
        "tools.audio.inworld_tts.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call API")),
    )
    result = InworldTTS().execute({"text": "x" * 2001})
    assert not result.success
    assert "2,000" in result.error


def test_execute_writes_decoded_audio(monkeypatch, tmp_path):
    monkeypatch.setenv("INWORLD_API_KEY", "test-key")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResponse(
            {
                "audioContent": base64.b64encode(b"fake-mp3").decode("ascii"),
                "usage": {"processedCharactersCount": 5},
                "timestampInfo": {"wordAlignment": []},
            }
        )

    monkeypatch.setattr("tools.audio.inworld_tts.requests.post", fake_post)
    monkeypatch.setattr("tools.analysis.audio_probe.probe_duration", lambda path: 1.25)
    output = tmp_path / "speech.mp3"
    result = InworldTTS().execute(
        {"text": "Hello", "voice_id": "Alex", "output_path": str(output)}
    )

    assert result.success
    assert output.read_bytes() == b"fake-mp3"
    assert captured["url"].endswith("/tts/v1/voice")
    assert captured["headers"]["Authorization"] == "Basic test-key"
    assert captured["json"]["voiceId"] == "Alex"
    assert captured["json"]["modelId"] == "inworld-tts-2"
    assert result.data["audio_duration_seconds"] == 1.25
    assert result.data["processed_characters"] == 5


def test_list_voices_uses_current_voice_api(monkeypatch):
    monkeypatch.setenv("INWORLD_API_KEY", "test-key")
    captured = {}

    def fake_get(url, headers, params, timeout):
        captured.update(url=url, headers=headers, params=params, timeout=timeout)
        return FakeResponse({"voices": [{"voiceId": "Alex"}]})

    monkeypatch.setattr("tools.audio.inworld_tts.requests.get", fake_get)
    voices = InworldTTS().list_voices(language="en-US")
    assert voices == [{"voiceId": "Alex"}]
    assert captured["url"].endswith("/voices/v1/voices")
    assert captured["params"] == {"language": "en-US"}


def test_selector_can_choose_inworld(monkeypatch, isolated_tool_registry):
    from tools.base_tool import ToolResult

    monkeypatch.setenv("INWORLD_API_KEY", "test-key")
    isolated_tool_registry.discover("tools")
    tool = isolated_tool_registry.get("inworld_tts")
    monkeypatch.setattr(
        tool,
        "execute",
        lambda inputs: ToolResult(success=True, data={}, artifacts=["speech.mp3"]),
    )

    result = isolated_tool_registry.get("tts_selector").execute(
        {
            "text": "Inworld narration",
            "preferred_provider": "inworld",
            "allowed_providers": ["inworld"],
        }
    )
    assert result.success
    assert result.data["selected_provider"] == "inworld"


def test_selector_aliases_and_wav_encoding(monkeypatch, tmp_path):
    monkeypatch.setenv("INWORLD_API_KEY", "test-key")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(json=json)
        return FakeResponse(
            {"audioContent": base64.b64encode(b"fake-wav").decode("ascii")}
        )

    monkeypatch.setattr("tools.audio.inworld_tts.requests.post", fake_post)
    monkeypatch.setattr("tools.analysis.audio_probe.probe_duration", lambda path: 0.5)
    result = InworldTTS().execute(
        {
            "text": "Bonjour",
            "voice": "Dennis",
            "model": "inworld-tts-2",
            "format": "wav",
            "language_code": "fr-FR",
            "apply_text_normalization": "auto",
            "output_path": str(tmp_path / "speech.wav"),
        }
    )

    assert result.success
    assert captured["json"]["voiceId"] == "Dennis"
    assert captured["json"]["audioConfig"]["audioEncoding"] == "LINEAR16"
    assert captured["json"]["language"] == "fr-FR"
    assert (
        captured["json"]["applyTextNormalization"]
        == "APPLY_TEXT_NORMALIZATION_UNSPECIFIED"
    )


def test_normalization_schema_matches_selector_contract():
    schema = InworldTTS.input_schema["properties"]["apply_text_normalization"]
    assert schema["default"] == "auto"
    assert schema["enum"] == ["auto", "on", "off"]


def test_idempotency_key_covers_audio_affecting_inputs():
    tool = InworldTTS()
    baseline = {"text": "Hello"}
    variants = {
        "voice_id": "Alex",
        "model_id": "inworld-tts-2-max",
        "format": "wav",
        "sample_rate_hertz": 24000,
        "language": "fr-FR",
        "delivery_mode": "CREATIVE",
        "timestamp_type": "CHARACTER",
        "apply_text_normalization": "off",
    }

    for field, value in variants.items():
        assert tool.idempotency_key(baseline) != tool.idempotency_key(
            {**baseline, field: value}
        ), field


def test_idempotency_key_canonicalizes_selector_aliases_and_defaults():
    tool = InworldTTS()
    canonical = {
        "text": "Bonjour",
        "voice_id": "Alex",
        "model_id": "inworld-tts-2",
        "language": "fr-FR",
        "timestamp_type": "WORD",
        "apply_text_normalization": "auto",
    }
    aliases = {
        "text": "Bonjour",
        "voice": "Alex",
        "model": "inworld-tts-2",
        "language_code": "fr-FR",
        "timestamps": True,
        "apply_text_normalization": "AUTO",
    }

    assert tool.idempotency_key(canonical) == tool.idempotency_key(aliases)
    assert tool.idempotency_key({"text": "Hello"}) == tool.idempotency_key(
        {
            "text": "Hello",
            "voice_id": "Dennis",
            "model_id": "inworld-tts-2",
            "format": "mp3",
            "sample_rate_hertz": 48000,
            "delivery_mode": "BALANCED",
            "timestamp_type": "WORD",
            "apply_text_normalization": "auto",
        }
    )
    assert tool.idempotency_key({"text": "Hello", "timestamps": False}) != (
        tool.idempotency_key({"text": "Hello", "timestamps": True})
    )
