"""
Audio API
=========

High-level client for the `/v1/audio/*` endpoints.

This module wraps the "Audio" section of the OpenAI API:

- POST /v1/audio/speech         → text → speech (TTS)
- POST /v1/audio/transcriptions → speech → text (STT)
- POST /v1/audio/translations   → speech → English text

It also defines simple models for transcription results and the
audio/transcript streaming events described in the docs.

Design notes
------------

- For *JSON* responses (transcriptions / translations), we assume
  `MerlinHTTPClient.post` returns `response.json()` as in other modules.
- For *binary audio* responses (TTS), we assume `MerlinHTTPClient.post`
  supports an `expect_json: bool = True` keyword and returns raw
  `bytes` when `expect_json=False`.

  A possible `MerlinHTTPClient.post` signature is:

      def post(self, endpoint: str, *, json=None, data=None,
               files=None, params=None, expect_json: bool = True) -> Any

  If your implementation differs, adapt the calls in `create_speech()`
  accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from merlin.http_client import MerlinHTTPClient


JSON = Dict[str, Any]


# ───────────────────────────────────────────────────────────────
# Transcription / Translation results
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AudioUsage:
    """
    Token or duration usage statistics for an audio request.

    This is a light wrapper around the `usage` objects in the docs.
    The exact shape varies (tokens vs. duration), so we keep the raw
    dict as-is and surface only a few convenience fields.
    """

    type: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    seconds: Optional[float]
    raw: JSON = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AudioUsage":
        if not isinstance(data, Mapping):
            return cls(
                type=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                seconds=None,
                raw={},
            )

        return cls(
            type=data.get("type"),
            input_tokens=_safe_int(data.get("input_tokens")),
            output_tokens=_safe_int(data.get("output_tokens")),
            total_tokens=_safe_int(data.get("total_tokens")),
            seconds=_safe_float(data.get("seconds")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class TranscriptionResult:
    """
    Generic transcription result.

    This unifies the three documented shapes:

    - "The transcription object (JSON)"         → text + usage
    - "The transcription object (Diarized JSON)"→ task, duration, text, segments, usage
    - "The transcription object (Verbose JSON)" → task, language, duration, text,
                                                  segments, words, usage
    """

    text: Optional[str]
    task: Optional[str]
    duration: Optional[float]
    language: Optional[str]
    segments: Optional[Sequence[JSON]]
    words: Optional[Sequence[JSON]]
    usage: Optional[AudioUsage]
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TranscriptionResult":
        usage_raw = data.get("usage")
        usage = AudioUsage.from_dict(usage_raw) if isinstance(usage_raw, Mapping) else None

        return cls(
            text=data.get("text"),
            task=data.get("task"),
            duration=_safe_float(data.get("duration")),
            language=data.get("language"),
            segments=data.get("segments"),
            words=data.get("words"),
            usage=usage,
            raw=dict(data),
        )


@dataclass(frozen=True)
class TranslationResult:
    """
    Audio translation result.

    Docs shape:

        {
          "text": "...",
        }

    Some future variants may include usage; we keep `raw` for that.
    """

    text: str
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TranslationResult":
        return cls(
            text=str(data.get("text", "")),
            raw=dict(data),
        )


# ───────────────────────────────────────────────────────────────
# Audio / transcript streaming events
# ───────────────────────────────────────────────────────────────


class AudioStreamEventTypes:
    """
    String constants for audio-related streaming events.

    These cover:

    - TTS (audio) streaming:
        - "speech.audio.delta"
        - "speech.audio.done"

    - Transcription streaming:
        - "transcript.text.delta"
        - "transcript.text.segment"
        - "transcript.text.done"
    """

    # TTS
    SPEECH_AUDIO_DELTA = "speech.audio.delta"
    SPEECH_AUDIO_DONE = "speech.audio.done"

    # STT
    TRANSCRIPT_TEXT_DELTA = "transcript.text.delta"
    TRANSCRIPT_TEXT_SEGMENT = "transcript.text.segment"
    TRANSCRIPT_TEXT_DONE = "transcript.text.done"


@dataclass(frozen=True)
class AudioStreamEvent:
    """
    Generic representation of an audio-related stream event.

    Fields are optional; each event type uses a subset of them.

    TTS events:
        type == "speech.audio.delta"
            audio: base64-encoded audio chunk
        type == "speech.audio.done"
            usage: AudioUsage for the request

    Transcription events:
        type == "transcript.text.delta"
            delta: incremental text
        type == "transcript.text.segment"
            id, start, end, speaker, text
        type == "transcript.text.done"
            text: complete transcript
            usage: AudioUsage
    """

    type: str

    # For TTS
    audio: Optional[str]

    # For STT deltas / segments / done
    delta: Optional[str]
    text: Optional[str]
    segment_id: Optional[str]
    start: Optional[float]
    end: Optional[float]
    speaker: Optional[str]

    usage: Optional[AudioUsage]

    # Raw event JSON
    raw: JSON

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AudioStreamEvent":
        event_type = str(data.get("type", ""))

        # Usage (speech.audio.done / transcript.text.done)
        usage_raw = data.get("usage")
        usage = AudioUsage.from_dict(usage_raw) if isinstance(usage_raw, Mapping) else None

        # Transcript segment-specific fields
        start = _safe_float(data.get("start"))
        end = _safe_float(data.get("end"))

        return cls(
            type=event_type,
            audio=data.get("audio"),
            delta=data.get("delta"),
            text=data.get("text"),
            segment_id=data.get("segment_id") or data.get("id"),
            start=start,
            end=end,
            speaker=data.get("speaker"),
            usage=usage,
            raw=dict(data),
        )


def parse_audio_stream_event(data: Mapping[str, Any]) -> AudioStreamEvent:
    """
    Parse a raw audio/transcript streaming event into an AudioStreamEvent.
    """
    return AudioStreamEvent.from_dict(data)


# ───────────────────────────────────────────────────────────────
# Client mixin
# ───────────────────────────────────────────────────────────────


class AudioMixin:
    """
    Mixin providing convenience methods for the Audio API.

    Assumptions:
        - The consuming client defines `self._http` as a MerlinHTTPClient.
    """

    _http: MerlinHTTPClient  # for type checkers

    # ---- Text-to-speech --------------------------------------------------

    def create_speech(
        self,
        *,
        model: str,
        input: str,
        voice: str,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        stream_format: Optional[str] = None,
        instructions: Optional[str] = None,
        **extra: Any,
    ) -> bytes:
        """
        Generate speech audio from text.

        POST /v1/audio/speech

        Args:
            model:
                One of: "tts-1", "tts-1-hd", "gpt-4o-mini-tts".
            input:
                Text to generate audio for (max 4096 characters).
            voice:
                Voice name, e.g. "alloy", "ash", "echo", etc.
            response_format:
                Audio format: "mp3", "opus", "aac", "flac", "wav", "pcm".
                Defaults to "mp3" (server-side) if not provided.
            speed:
                Playback speed, between 0.25 and 4.0. Default 1.0.
            stream_format:
                Streaming format: "sse" or "audio". Defaults to "audio".
            instructions:
                Extra instructions for the voice (not supported for tts-1 / tts-1-hd).
            extra:
                Any additional fields accepted by the API now or in the future.

        Returns:
            Raw audio bytes.

        Notes:
            This method assumes `MerlinHTTPClient.post` accepts an
            `expect_json=False` flag and returns `bytes` in that case.
        """
        payload: JSON = {
            "model": model,
            "input": input,
            "voice": voice,
        }

        if response_format is not None:
            payload["response_format"] = response_format
        if speed is not None:
            payload["speed"] = speed
        if stream_format is not None:
            payload["stream_format"] = stream_format
        if instructions is not None:
            payload["instructions"] = instructions

        payload.update(extra)

        # Expect binary audio (not JSON).
        audio_bytes = self._http.post(
            "/v1/audio/speech",
            json=payload,
            expect_json=False,  # requires MerlinHTTPClient support
        )
        return audio_bytes

    # ---- Transcriptions --------------------------------------------------

    def create_transcription(
        self,
        *,
        file: Any,
        model: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        timestamp_granularities: Optional[Sequence[str]] = None,
        include: Optional[Sequence[str]] = None,
        chunking_strategy: Optional[Any] = None,
        known_speaker_names: Optional[Sequence[str]] = None,
        known_speaker_references: Optional[Sequence[str]] = None,
        stream: Optional[bool] = None,
        **extra: Any,
    ) -> TranscriptionResult:
        """
        Transcribe audio into the input language.

        POST /v1/audio/transcriptions

        Args:
            file:
                Binary file-like object or tuple accepted by `requests` as a file.
                Formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm.
            model:
                One of: "gpt-4o-transcribe", "gpt-4o-mini-transcribe",
                "whisper-1", "gpt-4o-transcribe-diarize".
            language:
                Optional ISO-639-1 language code (e.g., "en") to improve accuracy.
            prompt:
                Text prompt to guide style / continue previous audio (not for diarize).
            response_format:
                One of: "json", "text", "srt", "verbose_json", "vtt", "diarized_json"
                (depending on model).
            temperature:
                Sampling temperature, 0–1. Default 0.
            timestamp_granularities:
                ["word", "segment"] when response_format=verbose_json.
            include:
                Extra info to include (e.g. ["logprobs"]).
            chunking_strategy:
                "auto" or an object configuring server VAD parameters.
            known_speaker_names / known_speaker_references:
                For diarization with known speakers.
            stream:
                If True, response is streamed with transcript events.
                (In that case you typically use a streaming client instead
                 of this helper.)

        Returns:
            TranscriptionResult (for non-streaming usage).
        """
        # Multipart form data: non-file fields go into "data", file(s) into "files".
        data: Dict[str, Any] = {
            "model": model,
        }

        if language is not None:
            data["language"] = language
        if prompt is not None:
            data["prompt"] = prompt
        if response_format is not None:
            data["response_format"] = response_format
        if temperature is not None:
            data["temperature"] = temperature
        if timestamp_granularities is not None:
            data["timestamp_granularities"] = list(timestamp_granularities)
        if include is not None:
            data["include"] = list(include)
        if chunking_strategy is not None:
            data["chunking_strategy"] = chunking_strategy
        if known_speaker_names is not None:
            data["known_speaker_names"] = list(known_speaker_names)
        if known_speaker_references is not None:
            data["known_speaker_references"] = list(known_speaker_references)
        if stream is not None:
            data["stream"] = stream

        data.update(extra)

        files = {
            "file": file,
        }

        # We expect JSON back (transcription object).
        resp = self._http.post(
            "/v1/audio/transcriptions",
            data=data,
            files=files,
            expect_json=True,  # default for most MerlinHTTPClient impls
        )
        return TranscriptionResult.from_dict(resp)

    # ---- Translations ----------------------------------------------------

    def create_translation(
        self,
        *,
        file: Any,
        model: str,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        **extra: Any,
    ) -> TranslationResult:
        """
        Translate audio into English.

        POST /v1/audio/translations

        Args:
            file:
                Binary file-like object or tuple accepted by `requests` as a file.
                Formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm.
            model:
                Currently "whisper-1" is supported.
            prompt:
                English text prompt to guide style or continue a segment.
            response_format:
                "json", "text", "srt", "verbose_json", or "vtt".
            temperature:
                Sampling temperature, 0–1. Default 0.

        Returns:
            TranslationResult
        """
        data: Dict[str, Any] = {
            "model": model,
        }

        if prompt is not None:
            data["prompt"] = prompt
        if response_format is not None:
            data["response_format"] = response_format
        if temperature is not None:
            data["temperature"] = temperature

        data.update(extra)

        files = {
            "file": file,
        }

        resp = self._http.post(
            "/v1/audio/translations",
            data=data,
            files=files,
            expect_json=True,
        )
        return TranslationResult.from_dict(resp)


# ───────────────────────────────────────────────────────────────
# Internal helpers
# ───────────────────────────────────────────────────────────────


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "AudioUsage",
    "TranscriptionResult",
    "TranslationResult",
    "AudioStreamEventTypes",
    "AudioStreamEvent",
    "parse_audio_stream_event",
    "AudioMixin",
]
