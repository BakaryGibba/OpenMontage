"""Inworld text-to-speech provider tool."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class InworldTTS(BaseTool):
    """Generate narration with Inworld TTS 2."""

    name = "inworld_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "inworld"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set INWORLD_API_KEY to the Base64 credential shown in the Inworld "
        "Portal's API Keys page."
    )
    fallback = "piper_tts"
    fallback_tools = ["piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "multilingual",
        "word_timestamps",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
        "word_timestamps": True,
        "expressive_delivery": True,
    }
    best_for = [
        "expressive narration",
        "multilingual narration",
        "word-timestamped speech for captions",
    ]
    not_good_for = ["offline production", "voice cloning"]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {
                "type": "string",
                "maxLength": 2000,
                "description": "Text to synthesize (maximum 2,000 characters).",
            },
            "voice_id": {
                "type": "string",
                "default": "Dennis",
                "description": "Inworld built-in or custom voice ID.",
            },
            "model_id": {
                "type": "string",
                "default": "inworld-tts-2",
            },
            "format": {
                "type": "string",
                "default": "mp3",
                "enum": ["mp3", "wav"],
            },
            "sample_rate_hertz": {
                "type": "integer",
                "default": 48000,
                "minimum": 8000,
                "maximum": 48000,
            },
            "language": {
                "type": "string",
                "description": "Optional BCP-47 language code, such as en-US.",
            },
            "delivery_mode": {
                "type": "string",
                "default": "BALANCED",
                "enum": ["STABLE", "BALANCED", "CREATIVE"],
            },
            "timestamp_type": {
                "type": "string",
                "default": "WORD",
                "enum": ["TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"],
            },
            "apply_text_normalization": {
                "type": "string",
                "default": "ON",
                "enum": ["ON", "OFF", "APPLY_TEXT_NORMALIZATION_UNSPECIFIED"],
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=2, retryable_errors=["rate_limit", "timeout", "server_error"]
    )
    idempotency_key_fields = [
        "text",
        "voice_id",
        "model_id",
        "format",
        "delivery_mode",
    ]
    side_effects = ["writes audio file to output_path", "calls Inworld API"]
    user_visible_verification = ["Listen to generated audio for intelligibility and tone"]

    _SYNTHESIS_URL = "https://api.inworld.ai/tts/v1/voice"
    _VOICES_URL = "https://api.inworld.ai/voices/v1/voices"
    _ENCODINGS = {"mp3": "MP3", "wav": "LINEAR16"}

    def get_status(self) -> ToolStatus:
        return (
            ToolStatus.AVAILABLE
            if os.environ.get("INWORLD_API_KEY")
            else ToolStatus.UNAVAILABLE
        )

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Inworld TTS-2 on-demand list price: $25 per million characters.
        return round(len(inputs.get("text", "")) * 0.000025, 4)

    def _headers(self) -> dict[str, str]:
        api_key = os.environ.get("INWORLD_API_KEY")
        if not api_key:
            raise RuntimeError("No Inworld API key. " + self.install_instructions)
        return {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _raise_api_error(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            try:
                detail = response.json().get("message") or response.json().get("error")
            except (ValueError, AttributeError):
                detail = None
            suffix = f": {str(detail)[:300]}" if detail else ""
            raise RuntimeError(f"Inworld API returned HTTP {response.status_code}{suffix}") from exc

    def list_voices(self, language: str | None = None) -> list[dict[str, Any]]:
        """List available voices using Inworld's current Voice API endpoint."""
        params = {"language": language} if language else None
        response = requests.get(
            self._VOICES_URL,
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        self._raise_api_error(response)
        payload = response.json()
        voices = payload.get("voices", [])
        if not isinstance(voices, list):
            raise RuntimeError("Inworld voices response did not contain a voices list.")
        return voices

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if not os.environ.get("INWORLD_API_KEY"):
            return ToolResult(success=False, error="No Inworld API key. " + self.install_instructions)

        text = str(inputs.get("text", ""))
        if not text.strip():
            return ToolResult(success=False, error="Inworld TTS text cannot be empty.")
        if len(text) > 2000:
            return ToolResult(
                success=False,
                error="Inworld TTS accepts at most 2,000 characters per request.",
            )

        started = time.time()
        try:
            result = self._generate(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"Inworld TTS failed: {exc}")

        result.duration_seconds = round(time.time() - started, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        from tools.analysis.audio_probe import probe_duration

        text = str(inputs["text"])
        voice_id = inputs.get("voice_id") or inputs.get("voice") or "Dennis"
        model_id = inputs.get("model_id") or inputs.get("model") or "inworld-tts-2"
        fmt = str(inputs.get("format", "mp3")).lower()
        if fmt not in self._ENCODINGS:
            raise ValueError("Inworld TTS format must be 'mp3' or 'wav'.")

        payload: dict[str, Any] = {
            "text": text,
            "voiceId": voice_id,
            "modelId": model_id,
            "deliveryMode": inputs.get("delivery_mode", "BALANCED"),
            "timestampType": inputs.get("timestamp_type", "WORD"),
            "applyTextNormalization": self._normalization_mode(inputs),
            "audioConfig": {
                "audioEncoding": self._ENCODINGS[fmt],
                "sampleRateHertz": int(inputs.get("sample_rate_hertz", 48000)),
            },
        }
        language = inputs.get("language") or inputs.get("language_code")
        if language:
            payload["language"] = language

        response = requests.post(
            self._SYNTHESIS_URL,
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        self._raise_api_error(response)
        response_payload = response.json()
        encoded_audio = response_payload.get("audioContent")
        if not encoded_audio:
            raise RuntimeError("Inworld response did not contain audioContent.")
        try:
            audio = base64.b64decode(encoded_audio, validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Inworld returned invalid Base64 audioContent.") from exc
        if not audio:
            raise RuntimeError("Inworld returned empty audioContent.")

        output_path = Path(inputs.get("output_path", f"inworld_tts.{fmt}"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio)
        duration = probe_duration(output_path)

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model_id,
                "voice": voice_id,
                "format": fmt,
                "text_length": len(text),
                "processed_characters": response_payload.get("usage", {}).get(
                    "processedCharactersCount"
                ),
                "audio_duration_seconds": round(duration, 2) if duration else None,
                "timestamp_info": response_payload.get("timestampInfo"),
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
            model=model_id,
        )

    @staticmethod
    def _normalization_mode(inputs: dict[str, Any]) -> str:
        value = str(inputs.get("apply_text_normalization", "ON")).upper()
        return {
            "AUTO": "APPLY_TEXT_NORMALIZATION_UNSPECIFIED",
            "APPLY_TEXT_NORMALIZATION_UNSPECIFIED": (
                "APPLY_TEXT_NORMALIZATION_UNSPECIFIED"
            ),
            "ON": "ON",
            "OFF": "OFF",
        }.get(value, "ON")
