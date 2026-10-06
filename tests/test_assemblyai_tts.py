#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AssemblyAITTSService against a fake Streaming TTS server."""

import asyncio
import base64
import json
from urllib.parse import parse_qs, urlparse

import pytest
import websockets
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.assemblyai import tts as assemblyai_tts
from pipecat.services.assemblyai.tts import AssemblyAITTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

# 80 ms of 24 kHz 16-bit mono audio per Audio frame, two frames per request.
AUDIO_CHUNK = b"\x01\x02" * 1920
CHUNK_MS = 80
WORD_MS = 50


def _server(
    captured: dict,
    *,
    word_boundaries: bool = True,
    stall_first_flush: bool = False,
    error_after_begin: dict | None = None,
):
    """Build a fake server following the documented Streaming TTS protocol.

    Each Flush is answered with two Audio frames, a FlushDone and (unless
    ``word_boundaries`` is False) a WordBoundaries frame placing the request's
    words on the session timeline. With ``stall_first_flush`` the first request
    sends one Audio frame and then waits, so it is still in flight when the
    client cancels it.
    """

    async def handler(ws):
        captured["connections"] = captured.get("connections", 0) + 1
        captured["path"] = ws.request.path
        captured["auth"] = ws.request.headers.get("Authorization")
        await ws.send(
            json.dumps(
                {
                    "type": "Begin",
                    "id": "session-id",
                    "expires_at": 2000000000,
                    "configuration": {"voice": "jane", "word_boundaries": word_boundaries},
                }
            )
        )
        if error_after_begin:
            await ws.send(json.dumps({"type": "Error", **error_after_begin}))
            await ws.close(error_after_begin["error_code"], "See Error message for details")
            return

        flush_id = 0
        session_ms = 0
        buffer = ""
        stalled = stall_first_flush
        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                if msg["type"] == "Generate":
                    buffer += msg["text"]
                elif msg["type"] == "Flush":
                    audio = base64.b64encode(AUDIO_CHUNK).decode()
                    frame = {"type": "Audio", "audio": audio, "flush_id": flush_id}
                    if stalled:
                        stalled = False
                        await ws.send(json.dumps(frame))
                        buffer = ""
                        continue
                    await ws.send(json.dumps(frame))
                    await ws.send(json.dumps(frame))
                    duration = 2 * CHUNK_MS
                    await ws.send(
                        json.dumps(
                            {
                                "type": "FlushDone",
                                "flush_id": flush_id,
                                "audio_duration_ms": duration,
                            }
                        )
                    )
                    if word_boundaries:
                        words = [
                            {
                                "word": word,
                                "start": session_ms + i * WORD_MS,
                                "end": session_ms + (i + 1) * WORD_MS,
                                "confidence": 0.9,
                            }
                            for i, word in enumerate(buffer.split())
                        ]
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "WordBoundaries",
                                    "flush_id": flush_id,
                                    "audio_start_ms": session_ms,
                                    "words": words,
                                }
                            )
                        )
                    session_ms += duration
                    flush_id += 1
                    buffer = ""
                elif msg["type"] == "Cancel":
                    await ws.send(json.dumps({"type": "Cancelled", "audio_duration_seconds": 0.08}))
                    flush_id += 1
                    buffer = ""
        except websockets.ConnectionClosed:
            pass

    return handler


def _service(server, **kwargs) -> AssemblyAITTSService:
    port = next(iter(server.sockets)).getsockname()[1]
    return AssemblyAITTSService(
        api_key="test-key", url=f"ws://127.0.0.1:{port}/v1/ws", sample_rate=24000, **kwargs
    )


def _sent(captured: dict, msg_type: str) -> list[dict]:
    return [m for m in captured["messages"] if m["type"] == msg_type]


@pytest.mark.asyncio
async def test_speak_streams_audio_and_words():
    """A TTSSpeakFrame is sent as one flushed request and its audio and words come back."""
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        down_frames, up_frames = await run_test(
            _service(server),
            frames_to_send=[
                TTSSpeakFrame(text="Hello from AssemblyAI."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert not any(isinstance(f, ErrorFrame) for f in [*down_frames, *up_frames])
    assert sum(isinstance(f, TTSStartedFrame) for f in down_frames) == 1
    assert sum(isinstance(f, TTSStoppedFrame) for f in down_frames) == 1
    audio = b"".join(f.audio for f in down_frames if isinstance(f, TTSAudioRawFrame))
    assert audio == AUDIO_CHUNK * 2
    words = [f.text for f in down_frames if isinstance(f, TTSTextFrame)]
    assert words == ["Hello", "from", "AssemblyAI."]

    # The key goes in the header raw, and the session is configured in the URL.
    assert captured["auth"] == "test-key"
    query = parse_qs(urlparse(captured["path"]).query)
    assert query["voice"] == ["jane"]
    assert query["language"] == ["english"]
    assert query["sample_rate"] == ["24000"]
    assert query["encoding"] == ["pcm_s16le"]
    assert [m["type"] for m in captured["messages"]] == ["Generate", "Flush"]


@pytest.mark.asyncio
async def test_llm_turn_flushes_each_sentence_into_one_context():
    """Each sentence is its own request, and the turn's words stay in order."""
    captured: dict = {"messages": []}
    tokens = "Hi there! Thanks for calling. How can I help?".split(" ")
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        down_frames, _ = await run_test(
            _service(server),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                *[LLMTextFrame(text=token + " ") for token in tokens],
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    generates = [m["text"] for m in _sent(captured, "Generate")]
    assert generates == ["Hi there!", "Thanks for calling.", "How can I help?"]
    assert len(_sent(captured, "Flush")) == 3

    assert sum(isinstance(f, TTSStartedFrame) for f in down_frames) == 1
    assert sum(isinstance(f, TTSStoppedFrame) for f in down_frames) == 1
    word_frames = [f for f in down_frames if isinstance(f, TTSTextFrame)]
    assert [f.text for f in word_frames] == tokens
    # Later requests are offset by the audio of the earlier ones in the turn.
    pts = [f.pts for f in word_frames]
    assert pts == sorted(pts)
    assert pts[2] - pts[0] == CHUNK_MS * 2 * 1_000_000


@pytest.mark.asyncio
async def test_missing_word_boundaries_still_close_the_context(monkeypatch):
    """Without WordBoundaries the context closes after the grace period and keeps its text."""
    monkeypatch.setattr(assemblyai_tts, "WORD_BOUNDARIES_GRACE_S", 0.05)
    captured: dict = {"messages": []}
    async with serve(_server(captured, word_boundaries=False), "127.0.0.1", 0) as server:
        down_frames, _ = await run_test(
            _service(server),
            frames_to_send=[
                TTSSpeakFrame(text="Hello from AssemblyAI."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert sum(isinstance(f, TTSStoppedFrame) for f in down_frames) == 1
    text = " ".join(f.text for f in down_frames if isinstance(f, TTSTextFrame))
    assert text == "Hello from AssemblyAI."


@pytest.mark.asyncio
async def test_interruption_cancels_without_reconnecting():
    """An interruption sends Cancel, drops the cancelled audio and reuses the session."""
    captured: dict = {"messages": []}
    async with serve(_server(captured, stall_first_flush=True), "127.0.0.1", 0) as server:
        service = _service(server)
        down_frames, _ = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="This reply gets cut off."),
                SleepFrame(sleep=0.2),
                InterruptionFrame(),
                SleepFrame(sleep=0.1),
                TTSSpeakFrame(text="Go ahead."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert len(_sent(captured, "Cancel")) == 1
    assert captured["connections"] == 1
    words = [f.text for f in down_frames if isinstance(f, TTSTextFrame)]
    assert words[-2:] == ["Go", "ahead."]
    assert not service.get_audio_contexts()


@pytest.mark.asyncio
async def test_interruption_after_synthesis_finished_sends_no_cancel():
    """Once every request has been answered there is nothing to cancel on the server."""
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        await run_test(
            _service(server),
            frames_to_send=[
                TTSSpeakFrame(text="Hello from AssemblyAI."),
                SleepFrame(sleep=0.3),
                InterruptionFrame(),
                SleepFrame(sleep=0.1),
            ],
        )

    assert not _sent(captured, "Cancel")


@pytest.mark.asyncio
async def test_rejected_voice_makes_the_service_unusable():
    """A voice the session can't serve fails the same way on every retry."""
    captured: dict = {"messages": []}
    error = {"error_code": 3006, "error": "Unsupported preset voice 'nobody'."}
    async with serve(_server(captured, error_after_begin=error), "127.0.0.1", 0) as server:
        service = _service(server, settings=AssemblyAITTSService.Settings(voice="nobody"))
        _, up_frames = await run_test(service, frames_to_send=[SleepFrame(sleep=0.3)])

    assert not service.is_usable
    assert any(
        "Unsupported preset voice" in f.error for f in up_frames if isinstance(f, ErrorFrame)
    )
    assert captured["connections"] == 1


def test_language_is_derived_from_the_voice():
    service = AssemblyAITTSService(
        api_key="test-key", settings=AssemblyAITTSService.Settings(voice="lola")
    )
    service._sample_rate = 16000
    query = parse_qs(urlparse(service._build_websocket_url()).query)
    assert query["voice"] == ["lola"]
    assert query["language"] == ["spanish"]
    assert query["sample_rate"] == ["16000"]


def test_language_enum_maps_to_language_name():
    service = AssemblyAITTSService(
        api_key="test-key",
        settings=AssemblyAITTSService.Settings(voice="juergen", language=Language.DE_DE),
    )
    assert service._settings.language == "german"


def test_unsupported_sample_rate_is_rejected():
    with pytest.raises(ValueError):
        AssemblyAITTSService(api_key="test-key", sample_rate=11025)


def test_unset_voice_is_left_to_the_server():
    service = AssemblyAITTSService(
        api_key="test-key", settings=AssemblyAITTSService.Settings(voice=None)
    )
    service._sample_rate = 24000
    query = parse_qs(urlparse(service._build_websocket_url()).query)
    assert "voice" not in query
    assert "language" not in query
