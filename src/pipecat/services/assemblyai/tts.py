#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AssemblyAI text-to-speech service implementation.

This module provides integration with AssemblyAI's Streaming TTS WebSocket API,
which synthesizes text into streamed PCM audio with per-word timings.
"""

import asyncio
import base64
import json
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from websockets.protocol import State

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import is_given

# Sample rates the API can stream. 24 kHz is the model's native rate; any other
# is resampled on the server.
ASSEMBLYAI_TTS_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
ASSEMBLYAI_TTS_DEFAULT_SAMPLE_RATE = 24000

# Longest text a single Generate message accepts, in Unicode code points, when
# the session's Begin doesn't report its own limit.
MAX_GENERATE_CHARS = 2000

# How long before a session's expires_at to replace it between requests, at
# most half the session's length. The server stops reading text at expires_at,
# which would cut off a reply that straddles it.
SESSION_RENEWAL_MARGIN_S = 120.0

# How long a graceful stop waits for Termination after sending Terminate.
TERMINATE_TIMEOUT_S = 2.0

# Error types the server reports when it ends a session for a routine reason.
# The receive loop reconnects, so they aren't errors.
ROUTINE_ERROR_TYPES = frozenset({"inactivity_timeout", "session_expired", "service_restart"})
ROUTINE_ERROR_CODES = frozenset({1012, 3008})

# Error type -> category. INVALID_REQUEST and AUTHENTICATION are permanent and
# leave the service unusable, so they're kept for errors in the session's own
# configuration. A refused voice or language is one too; see _classify_error.
# A refused frame or text concerns that text only.
ERROR_TYPES: dict[str, ErrorCategory] = {
    "unauthorized": ErrorCategory.AUTHENTICATION,
    "insufficient_funds": ErrorCategory.QUOTA,
    "invalid_parameter": ErrorCategory.INVALID_REQUEST,
    "invalid_message": ErrorCategory.UNKNOWN,
    "invalid_request": ErrorCategory.UNKNOWN,
    "rate_limited": ErrorCategory.RATE_LIMIT,
    "input_rate_exceeded": ErrorCategory.RATE_LIMIT,
    "auth_unavailable": ErrorCategory.SERVER,
    "at_capacity": ErrorCategory.SERVER,
    "upstream_unavailable": ErrorCategory.SERVER,
    "voice_service_unavailable": ErrorCategory.SERVER,
    "internal_error": ErrorCategory.SERVER,
}

# Close code -> category, for an Error whose error_type is missing or unknown.
# A 3006 covers both bad parameters and idle timeouts, so it's left to the
# reconnect to tell them apart: a bad parameter fails again at once.
ERROR_CODES: dict[int, ErrorCategory] = {
    1008: ErrorCategory.AUTHENTICATION,
    1011: ErrorCategory.SERVER,
    3005: ErrorCategory.SERVER,
    3009: ErrorCategory.RATE_LIMIT,
    3010: ErrorCategory.RATE_LIMIT,
}

# How long to wait, after a turn's last FlushDone, for the WordBoundaries frame
# that trails it. The frame normally lands within tens of milliseconds but is
# best-effort, so the turn's audio context closes without it after this long.
WORD_BOUNDARIES_GRACE_S = 1.0


def language_to_assemblyai_tts_language(language: Language) -> str:
    """Convert a Language enum to an AssemblyAI Streaming TTS language.

    The TTS API takes the full English name of the language (``"spanish"``),
    unlike the STT APIs, which take short codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding AssemblyAI TTS language name.
    """
    LANGUAGE_MAP = {
        Language.DE: "german",
        Language.EN: "english",
        Language.ES: "spanish",
        Language.FR: "french",
        Language.IT: "italian",
        Language.PT: "portuguese",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class AssemblyAITTSSettings(TTSSettings):
    """Settings for AssemblyAITTSService.

    ``voice`` is an AssemblyAI voice name (case-insensitive), and ``language``
    is the language the session speaks, ``english`` when unset. The language
    picks the session's model, which must speak the voice in that language;
    when ``voice`` is unset the server uses that model's default voice.
    AssemblyAI has no model selection, so ``model`` is unused.
    """

    pass


@dataclass
class _Request:
    """Text sent to the server for one context, concluded by one Flush."""

    context_id: str
    flushed: bool = False


@dataclass
class _ContextState:
    """Per-audio-context bookkeeping for requests and word timings.

    Parameters:
        audio_ms: Audio delivered so far for the context, from FlushDone
            durations. Positions each request's words within the context.
        turn_done: Whether the turn has ended, so no more text will be sent.
        awaiting_words: Flush IDs whose WordBoundaries frame is still due.
        grace_task: Task that closes the context if WordBoundaries never come.
    """

    audio_ms: int = 0
    turn_done: bool = False
    awaiting_words: set[int] = field(default_factory=set)
    grace_task: asyncio.Task | None = None


class AssemblyAITTSService(WebsocketTTSService):
    """AssemblyAI Streaming TTS service.

    Streams text to AssemblyAI over a single WebSocket session and plays the
    audio as it arrives. Each sentence is sent as its own request (a
    ``Generate`` followed by a ``Flush``). With ``TextAggregationMode.TOKEN``
    the turn's text streams as it is generated and is flushed when the turn
    ends; the server starts speaking once the first sentence is complete.
    Interruptions send ``Cancel``, which keeps the session open, and per-word
    timings drive word-level ``TTSTextFrame`` output.

    The session's voice, language and sample rate are fixed at connect time,
    so changing the voice or language reconnects. A session lasts at most an
    hour; the service replaces it between requests before it expires, and
    keeps an idle session open when the account sets an inactivity timeout.

    Example::

        tts = AssemblyAITTSService(
            api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            settings=AssemblyAITTSService.Settings(voice="jane"),
        )
    """

    Settings = AssemblyAITTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://streaming-tts.assemblyai.com/v1/ws",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the AssemblyAI TTS service.

        Args:
            api_key: AssemblyAI API key, or a temporary token minted with
                ``product=tts``.
            url: Streaming TTS WebSocket URL. Use
                ``wss://streaming-tts.us.assemblyai.com/v1/ws`` or
                ``wss://streaming-tts.eu.assemblyai.com/v1/ws`` to keep text and
                audio in the US or EU.
            sample_rate: Output sample rate. One of
                ``ASSEMBLYAI_TTS_SAMPLE_RATES``; if None, uses the pipeline's
                output rate when the API supports it, and 24000 otherwise.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent WebsocketTTSService.

        Raises:
            ValueError: If ``sample_rate`` is not one the API supports.
        """
        if sample_rate is not None and sample_rate not in ASSEMBLYAI_TTS_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample_rate {sample_rate}. "
                f"Must be one of {ASSEMBLYAI_TTS_SAMPLE_RATES}."
            )

        default_settings = self.Settings(
            model=None,
            voice=None,
            language=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_stop_frames=True,
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url

        self._receive_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None

        # Session state reported by Begin.
        self._session_id: str | None = None
        # When to replace the session, ahead of its expires_at.
        self._renew_at: float | None = None
        self._max_generate_chars = MAX_GENERATE_CHARS
        # Whether the session will send WordBoundaries.
        self._word_boundaries = False

        # When the last client frame was sent, for the keepalive.
        self._last_send_time = 0.0
        # Set by Termination while a graceful stop waits for it.
        self._termination: asyncio.Event | None = None
        self._ending = False
        # The wait an Error asked for before the next connection attempt.
        self._reconnect_delay = 0.0

        # Requests the server has not finished, oldest first. The server works
        # through them in order and never interleaves their audio, so audio and
        # FlushDone frames belong to the oldest one.
        self._requests: deque[_Request] = deque()
        self._contexts: dict[str, _ContextState] = {}
        # Flush ID -> (context ID, offset of that request's audio within the
        # context), recorded on FlushDone for the WordBoundaries that follows.
        self._flush_offsets: dict[int, tuple[str, int]] = {}
        # Cancels sent but not yet answered. Frames for the cancelled requests
        # can still arrive until the server's Cancelled reply, and are dropped.
        self._cancels_pending = 0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as AssemblyAI TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to an AssemblyAI TTS language name.

        Args:
            language: The language to convert.

        Returns:
            The AssemblyAI TTS language name.
        """
        return language_to_assemblyai_tts_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Connecting here, rather than on the first sentence, keeps the
        connection setup off the first response's latency.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        if self._sample_rate not in ASSEMBLYAI_TTS_SAMPLE_RATES:
            logger.warning(
                f"{self}: AssemblyAI TTS cannot stream at {self._sample_rate} Hz; "
                f"using {ASSEMBLYAI_TTS_DEFAULT_SAMPLE_RATE} Hz"
            )
            self._sample_rate = ASSEMBLYAI_TTS_DEFAULT_SAMPLE_RATE
        await self._connect()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting if the voice or language changed.

        Args:
            delta: A :class:`TTSSettings` (or ``AssemblyAITTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if "voice" in changed or "language" in changed:
            await self._disconnect()
            await self._connect()
        else:
            self._warn_unhandled_updated_settings(changed)
        return changed

    async def stop(self, frame: EndFrame):
        """Stop the service, ending the session with ``Terminate``.

        Args:
            frame: The end frame.
        """
        self._ending = True
        await super().stop(frame)

    def _build_websocket_url(self) -> str:
        """Build the connect URL, which carries the whole session configuration."""
        voice = self._settings.voice
        language = self._settings.language
        params: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "encoding": "pcm_s16le",
            "word_boundaries": "true",
        }
        # The server falls back to its default voice and to English.
        if voice:
            params["voice"] = voice
        if is_given(language) and language:
            params["language"] = language
        return f"{self._url}?{urlencode(params)}"

    # ------------------------------------------------------------------
    # WebSocket connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to AssemblyAI and start the receive task."""
        await super()._connect()
        self._reconnect_delay = 0.0
        await self._open_session()

    async def _disconnect(self):
        """Disconnect from AssemblyAI and stop the receive task."""
        if self._ending:
            await self._terminate_session()
        await super()._disconnect()
        await self.stop_all_metrics()
        await self._close_session()

    async def _open_session(self):
        """Open a session and start reading it."""
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _close_session(self):
        """Stop reading the session and close it."""
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _renew_session_if_expiring(self):
        """Replace the session ahead of its expiry, while nothing is in flight.

        Audio contexts stay open, so a turn in progress carries on in the new
        session.
        """
        if self._renew_at is None or self._requests or self._flush_offsets:
            return
        if time.time() < self._renew_at:
            return
        logger.debug(f"{self}: session {self._session_id} expires soon; opening a new one")
        await self._close_session()
        await self._open_session()

    async def _terminate_session(self):
        """End the session with Terminate and wait briefly for its Termination."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        # The server closes the socket after Termination; that is not a drop
        # to reconnect from.
        self._disconnecting = True
        self._termination = asyncio.Event()
        try:
            await self._send({"type": "Terminate"})
            await asyncio.wait_for(self._termination.wait(), timeout=TERMINATE_TIMEOUT_S)
        except Exception as e:
            logger.debug(f"{self}: no Termination after Terminate: {e!r}")
        finally:
            self._termination = None

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect, first waiting as long as the last Error asked.

        The lost session's contexts are closed before the wait, so a turn cut
        off by the error ends rather than holding the pipeline.
        """
        await self._close_connection_contexts()
        if self._reconnect_delay:
            delay, self._reconnect_delay = self._reconnect_delay, 0.0
            logger.debug(f"{self}: waiting {delay} s before reconnecting, as the server asked")
            await asyncio.sleep(delay)
        return await super()._reconnect_websocket(attempt_number)

    async def _connect_websocket(self):
        """Open the WebSocket session."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            url = self._build_websocket_url()
            logger.debug(f"{self}: connecting to {url}")
            headers = {
                # Without a "Bearer " prefix, which not every deployment accepts.
                "Authorization": self._api_key,
                "User-Agent": f"AssemblyAI/1.0 (integration=Pipecat/{pipecat_version()})",
            }
            self._websocket = await self._websocket_connect(url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to AssemblyAI TTS: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket session and forget its state."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug(f"{self}: disconnecting")
                # Closing without Terminate discards synthesis still in flight,
                # which is what a disconnect wants.
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)
        finally:
            self._websocket = None
            await self._reset_session_state()
            await self._call_event_handler("on_disconnected")

    async def _reset_session_state(self):
        """Forget per-session state, which a new session doesn't carry over."""
        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None
        self._requests.clear()
        self._flush_offsets.clear()
        self._cancels_pending = 0
        self._session_id = None
        self._renew_at = None
        self._max_generate_chars = MAX_GENERATE_CHARS
        self._word_boundaries = False

    async def _close_connection_contexts(self):
        """Forget the bookkeeping of the contexts a replaced connection closes."""
        for context_id in list(self._contexts):
            await self._forget_context(context_id)
        await super()._close_connection_contexts()

    def _get_websocket(self):
        """Return the active WebSocket connection or raise if disconnected."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _send(self, message: dict):
        """Send a JSON message to the server.

        Args:
            message: The message to serialize and send.
        """
        await self._get_websocket().send(json.dumps(message))
        self._last_send_time = time.monotonic()

    async def _keepalive_task_handler(self, interval: float):
        """Send KeepAlive whenever the session has been idle for ``interval``."""
        while True:
            remaining = interval - (time.monotonic() - self._last_send_time)
            if remaining > 0:
                await asyncio.sleep(remaining)
                continue
            try:
                await self._send({"type": "KeepAlive"})
            except Exception as e:
                logger.debug(f"{self}: keepalive stopped: {e!r}")
                return

    # ------------------------------------------------------------------
    # Requests and audio contexts
    # ------------------------------------------------------------------

    def _context_state(self, context_id: str) -> _ContextState:
        if context_id not in self._contexts:
            self._contexts[context_id] = _ContextState()
        return self._contexts[context_id]

    def _open_request(self, context_id: str) -> _Request:
        """Return the context's unflushed request, starting one if there isn't one."""
        if self._requests and not self._requests[-1].flushed:
            request = self._requests[-1]
            if request.context_id == context_id:
                return request
        request = _Request(context_id=context_id)
        self._requests.append(request)
        return request

    async def _maybe_close_context(self, context_id: str):
        """Close a context once its turn has ended and the server owes it nothing.

        A context stays open until its turn ends, every request has been
        answered with FlushDone, and every WordBoundaries frame due has arrived
        (or the grace period for one has run out).
        """
        state = self._contexts.get(context_id)
        if not state or not state.turn_done:
            return
        if any(request.context_id == context_id for request in self._requests):
            return
        if state.awaiting_words:
            if not state.grace_task:
                state.grace_task = self.create_task(
                    self._word_boundaries_grace(context_id), f"{self}::word_boundaries_grace"
                )
            return
        await self._close_context(context_id)

    async def _word_boundaries_grace(self, context_id: str):
        await asyncio.sleep(WORD_BOUNDARIES_GRACE_S)
        state = self._contexts.get(context_id)
        if state:
            logger.debug(f"{self}: no WordBoundaries for flushes {state.awaiting_words}")
            state.grace_task = None
            state.awaiting_words.clear()
            await self._close_context(context_id)

    async def _forget_context(self, context_id: str):
        """Drop a context's bookkeeping, including WordBoundaries still due for it."""
        state = self._contexts.pop(context_id, None)
        if state and state.grace_task:
            await self.cancel_task(state.grace_task)
        self._flush_offsets = {
            flush_id: offset
            for flush_id, offset in self._flush_offsets.items()
            if offset[0] != context_id
        }

    async def _close_context(self, context_id: str):
        await self._forget_context(context_id)
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Conclude the turn's text so the context can close once it is spoken.

        In sentence aggregation each sentence is already flushed by
        :meth:`run_tts`. When streaming tokens, the turn's text is flushed here.

        Args:
            context_id: The context whose turn has ended.
        """
        context_id = context_id or self.get_active_audio_context_id()
        if not context_id:
            return

        request = self._requests[-1] if self._requests else None
        if request and request.context_id == context_id and not request.flushed:
            request.flushed = True
            try:
                await self._send({"type": "Flush"})
            except Exception as e:
                logger.warning(f"{self}: error sending Flush: {e}")

        self._context_state(context_id).turn_done = True
        await self._maybe_close_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the server's in-flight synthesis when the bot is interrupted.

        ``Cancel`` discards everything the server has not yet delivered and
        keeps the session open, so the next response needs no reconnect.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self.stop_all_metrics()
        if self._requests:
            self._requests.clear()
            if self._websocket and self._websocket.state is State.OPEN:
                self._cancels_pending += 1
                try:
                    await self._send({"type": "Cancel"})
                except Exception as e:
                    logger.warning(f"{self}: error sending Cancel: {e}")
        await self._forget_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    # ------------------------------------------------------------------
    # Server messages
    # ------------------------------------------------------------------

    async def _receive_messages(self):
        """Receive and dispatch server frames.

        Called by ``WebsocketService._receive_task_handler``, which reconnects
        when the session closes.
        """
        async for message in self._get_websocket():
            if not isinstance(message, str):
                continue
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: invalid JSON message: {message}")
                continue

            msg_type = msg.get("type")
            if msg_type == "Audio":
                await self._handle_audio(msg)
            elif msg_type == "FlushDone":
                await self._handle_flush_done(msg)
            elif msg_type == "WordBoundaries":
                await self._handle_word_boundaries(msg)
            elif msg_type == "Cancelled":
                logger.debug(f"{self}: synthesis cancelled: {msg}")
                self._cancels_pending = max(0, self._cancels_pending - 1)
            elif msg_type == "Begin":
                self._handle_begin(msg)
            elif msg_type == "Error":
                await self._handle_error(msg)
            elif msg_type == "Termination":
                logger.debug(f"{self}: session terminated: {msg}")
                if self._termination:
                    self._termination.set()
            else:
                # The protocol may add frame types, which are safe to ignore.
                logger.trace(f"{self}: unhandled message: {msg}")

    def _handle_begin(self, msg: dict):
        configuration = msg.get("configuration") or {}
        limits = configuration.get("limits") or {}
        self._session_id = msg.get("id")
        expires_at = msg.get("expires_at")
        if expires_at:
            margin = min(SESSION_RENEWAL_MARGIN_S, (expires_at - time.time()) / 2)
            self._renew_at = expires_at - margin
        self._max_generate_chars = limits.get("max_generate_text_length") or MAX_GENERATE_CHARS
        self._word_boundaries = configuration.get("word_boundaries") is True
        logger.debug(f"{self}: session {self._session_id} started: {configuration}")

        # An account can set an idle timeout the client didn't ask for.
        inactivity_timeout = configuration.get("inactivity_timeout")
        if inactivity_timeout and not self._keepalive_task:
            self._last_send_time = time.monotonic()
            self._keepalive_task = self.create_task(
                self._keepalive_task_handler(inactivity_timeout / 2),
                f"{self}::keepalive",
            )

    async def _handle_audio(self, msg: dict):
        if self._cancels_pending or not self._requests:
            return
        context_id = self._requests[0].context_id
        if not self.audio_context_available(context_id):
            return
        frame = TTSAudioRawFrame(
            audio=base64.b64decode(msg["audio"]),
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=context_id,
        )
        await self.append_to_audio_context(context_id, frame)

    async def _handle_flush_done(self, msg: dict):
        if self._cancels_pending or not self._requests:
            return
        request = self._requests.popleft()
        flush_id = msg.get("flush_id")
        audio_ms = int(msg.get("audio_duration_ms") or 0)
        logger.trace(f"{self}: flush {flush_id} done ({audio_ms} ms)")

        state = self._context_state(request.context_id)
        if isinstance(flush_id, int) and audio_ms > 0:
            self._flush_offsets[flush_id] = (request.context_id, state.audio_ms)
            if self._word_boundaries:
                state.awaiting_words.add(flush_id)
        state.audio_ms += audio_ms
        await self._maybe_close_context(request.context_id)

    async def _handle_word_boundaries(self, msg: dict):
        flush_id = msg.get("flush_id")
        logger.trace(f"{self}: word boundaries for flush {flush_id}")
        if self._cancels_pending:
            return
        if not isinstance(flush_id, int) or flush_id not in self._flush_offsets:
            return
        context_id, offset_ms = self._flush_offsets.pop(flush_id)

        # Word times are on the session's audio timeline. Subtracting the
        # request's own start places each word within the request, and the
        # offset places the request within the context.
        audio_start_ms = msg.get("audio_start_ms") or 0
        word_times = [
            (word["word"], (word["start"] - audio_start_ms + offset_ms) / 1000.0)
            for word in msg.get("words", [])
        ]
        if word_times:
            await self.add_word_timestamps(word_times, context_id)

        state = self._contexts.get(context_id)
        if state:
            state.awaiting_words.discard(flush_id)
            await self._maybe_close_context(context_id)

    async def _handle_error(self, msg: dict):
        """Report an Error frame. The server closes the session right after it.

        A rejected credential, voice or parameter fails identically on every
        retry, so its category leaves the service unusable. Transient failures are
        retried by the receive loop's reconnect, after any wait the server
        asks for.
        """
        code = msg.get("error_code")
        error_type = msg.get("error_type")
        label = f"{code} {error_type}" if error_type else f"{code}"
        text = f"AssemblyAI TTS error {label}: {msg.get('error', '')}"

        retry_after = msg.get("retry_after_seconds")
        if retry_after:
            self._reconnect_delay = float(retry_after)
        if error_type in ROUTINE_ERROR_TYPES or (not error_type and code in ROUTINE_ERROR_CODES):
            logger.debug(f"{self}: {text}")
            return

        category = self._classify_error(code, error_type, msg.get("param"))
        await self.push_error(error_msg=text, category=category)

    @staticmethod
    def _classify_error(
        code: int | None, error_type: str | None, param: str | None
    ) -> ErrorCategory:
        """Return the category of an Error frame."""
        if error_type == "invalid_request" and param in ("voice", "language"):
            return ErrorCategory.INVALID_REQUEST
        if error_type in ERROR_TYPES:
            return ERROR_TYPES[error_type]
        return ERROR_CODES.get(code or 0, ErrorCategory.UNKNOWN)

    # ------------------------------------------------------------------
    # TTS generation
    # ------------------------------------------------------------------

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to AssemblyAI for synthesis.

        Audio arrives on the receive task. In sentence aggregation each call is
        flushed as its own request, since only a flush guarantees the server
        speaks all of the text.

        Args:
            text: The text to synthesize.
            context_id: The audio context the text belongs to.

        Yields:
            Frame: None, as audio arrives via the WebSocket receive task.
        """
        if self._reconnect_in_progress:
            # The receive loop owns the connection until it reconnects.
            yield ErrorFrame(error="AssemblyAI TTS is reconnecting; text not spoken")
            yield TTSStoppedFrame(context_id=context_id)
            return

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()
            else:
                await self._renew_session_if_expiring()

            # Registered before sending, so audio that arrives while the
            # messages are still being sent is attributed to this context.
            request = self._open_request(context_id)
            flush = not self._is_streaming_tokens
            if flush:
                request.flushed = True

            # Consecutive Generate texts are joined exactly as sent.
            size = self._max_generate_chars
            for start in range(0, len(text), size):
                await self._send({"type": "Generate", "text": text[start : start + size]})
            if flush:
                await self._send({"type": "Flush"})

            await self.start_tts_usage_metrics(text)
        except Exception as e:
            yield ErrorFrame(error=f"AssemblyAI TTS error: {e}")
            yield TTSStoppedFrame(context_id=context_id)
            await self._disconnect()
            await self._connect()
            return

        yield None
