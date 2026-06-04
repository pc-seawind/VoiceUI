from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from voiceui.audio import (
    pcm16_rms,
    resolve_sounddevice_device,
    select_pcm16_channel,
    write_pcm16_wav,
)
from voiceui.audio_dump import configure_audio_dump
from voiceui.config import load_config
from voiceui.core import VoiceAssistant
from voiceui.env import load_dotenv
from voiceui.logs import configure_log_files, configure_logging, log_event
from voiceui.models import AssistantConfig, WakeAckConfig, WakeEvent
from voiceui.vad import SpeechStartTimeoutError
from voiceui.wake import (
    _best_prediction,
    _download_openwakeword_models,
    _resolve_openwakeword_models,
)
from voiceui.wake_ack import create_wake_ack_player

DEFAULT_DEVICE_A_INPUT = (
    "回音消除话筒 (reSpeaker XVF3800 4-Mic Array), Windows WASAPI (2 in, 0 out)"
)
DEFAULT_DEVICE_B_INPUT = (
    "回音消除话筒 (2- reSpeaker XVF3800 4-Mic Array), Windows WASAPI (2 in, 0 out)"
)
DEFAULT_DEVICE_A_OUTPUT = (
    "回音消除话筒 (reSpeaker XVF3800 4-Mic Array), Windows WASAPI (0 in, 2 out)"
)
DEFAULT_DEVICE_B_OUTPUT = (
    "回音消除话筒 (2- reSpeaker XVF3800 4-Mic Array), Windows WASAPI (0 in, 2 out)"
)
DEFAULT_PROD_LIVE_CONFIG = "config.demo.wake.aliyun.yaml"
DEFAULT_POSITIONS = "near_xvf1:xvf1,near_xvf2:xvf2,center:"
DEFAULT_E2E_POSITIONS = "near_xvf1:xvf1,near_xvf2:xvf2"
DEFAULT_PROXIMITY_CHANNEL = "0"
DEFAULT_WAKE_CHANNEL = "0"
DEFAULT_RAW_PROXIMITY_CHANNEL = "1"
DEFAULT_WAKE_WINDOW_PRE_MS = 1300
DEFAULT_WAKE_WINDOW_POST_MS = 300
DEFAULT_NON_TRIGGERED_OVERRIDE_RMS_RATIO = 1.5
DEFAULT_NON_TRIGGERED_OVERRIDE_MIN_SNR_MARGIN_DB = -3.0
MIN_NOISE_WINDOW_MS = 100


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    label: str
    device: str | int
    wake_channel: str
    proximity_channel: str


@dataclass(slots=True)
class ChannelMetrics:
    channel: int
    audio_ms: int
    chunks: int
    best_label: str
    best_confidence: float
    best_confidence_ms: int | None
    first_trigger_ms: int | None
    trigger_count: int
    noise_rms: float
    mean_rms: float
    peak_rms: float
    snr_db: float
    predict_avg_ms: float
    predict_max_ms: float
    wav_path: str = ""

    @property
    def triggered(self) -> bool:
        return self.first_trigger_ms is not None


@dataclass(slots=True)
class DeviceMetrics:
    label: str
    device: str | int
    resolved_device: int | None
    channel: int
    proximity_channel: int
    audio_ms: int
    chunks: int
    overflow_count: int
    best_label: str
    best_confidence: float
    first_trigger_ms: int | None
    trigger_count: int
    noise_rms: float
    mean_rms: float
    peak_rms: float
    snr_db: float
    predict_avg_ms: float
    predict_max_ms: float
    score: float = 0.0
    wav_path: str = ""
    proximity_wav_path: str = ""
    proximity_window_wav_path: str = ""
    candidate_channels: str = ""
    channel_metrics: dict[str, ChannelMetrics] = field(default_factory=dict)
    wake_window_start_ms: int = 0
    wake_window_end_ms: int = 0
    segment_duration_ms: int = 0
    band_rms: float = 0.0
    band_snr_db: float = 0.0
    speech_band_ratio: float = 0.0

    @property
    def triggered(self) -> bool:
        return self.first_trigger_ms is not None


@dataclass(slots=True)
class TrialResult:
    trial_id: int
    position: str
    expected_device: str
    selected_device: str
    correct: bool | None
    margin: float
    started_at: str
    listen_seconds: float
    baseline_seconds: float
    threshold: float
    model: str
    devices: dict[str, DeviceMetrics]
    trigger_source_device: str = ""
    global_wake_window_start_ms: int = 0
    global_wake_window_end_ms: int = 0
    ack_output_device: str = ""
    ack_latency_ms: int = 0
    ack_error: str = ""
    assistant_transcript: str = ""
    assistant_reply: str = ""
    assistant_error: str = ""


@dataclass(slots=True)
class LiveWakeResult:
    trial: TrialResult
    wake_pcm_by_label: dict[str, bytes]


@dataclass(slots=True)
class _TimedPcmChunk:
    start_ms: int
    end_ms: int
    pcm: bytes


@dataclass(slots=True)
class _LiveWakeCandidate:
    label: str
    channel: int
    wake_label: str
    confidence: float
    event_ms: int


@dataclass(frozen=True, slots=True)
class PositionSpec:
    name: str
    expected_device: str


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    confidence: float = 0.15
    rms: float = 0.25
    snr: float = 0.60
    late_penalty: float = 0.0


@dataclass(frozen=True, slots=True)
class SummaryBucket:
    trials: int
    scored_trials: int
    correct: int
    accuracy: float | None
    avg_margin: float


class _DeviceTrialWorker(threading.Thread):
    def __init__(
        self,
        *,
        spec: DeviceSpec,
        models: dict[int, Any],
        sd,
        sample_rate: int,
        channels: int,
        block_ms: int,
        listen_seconds: float,
        baseline_seconds: float,
        threshold: float,
        wake_window_pre_ms: int,
        wake_window_post_ms: int,
        start_event: threading.Event,
    ):
        super().__init__(name=f"wake-proximity-{spec.label}", daemon=True)
        self.spec = spec
        self.models = models
        self.sd = sd
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_ms = block_ms
        self.listen_seconds = listen_seconds
        self.baseline_seconds = baseline_seconds
        self.threshold = threshold
        self.wake_window_pre_ms = wake_window_pre_ms
        self.wake_window_post_ms = wake_window_post_ms
        self.start_event = start_event
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.metrics: DeviceMetrics | None = None
        self.pcm_by_channel: dict[int, bytes] = {}
        self.proximity_window_pcm = b""

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.error = exc
            self.ready.set()

    def _run(self) -> None:
        import numpy as np  # type: ignore[import-untyped]

        for model in self.models.values():
            reset = getattr(model, "reset", None)
            if callable(reset):
                reset()

        resolved_device = resolve_sounddevice_device(self.sd, self.spec.device, kind="input")
        channel_candidates = _parse_channel_candidates(self.spec.wake_channel, self.channels)
        proximity_channel = _parse_proximity_channel(self.spec.proximity_channel, self.channels)
        states = {
            channel: _ChannelCaptureState(model=self.models[channel])
            for channel in channel_candidates
        }
        proximity_chunks: list[bytes] = []
        frames = max(1, int(self.sample_rate * self.block_ms / 1000))
        with self.sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=frames,
            device=resolved_device,
        ) as stream:
            self.ready.set()
            self.start_event.wait()
            started = time.monotonic()
            deadline = started + self.listen_seconds
            overflow_count = 0

            while time.monotonic() < deadline:
                data, overflowed = stream.read(frames)
                if overflowed:
                    overflow_count += 1
                    continue
                chunk = bytes(data)
                elapsed = time.monotonic() - started
                proximity_chunk = (
                    select_pcm16_channel(
                        chunk,
                        channels=self.channels,
                        selected_channel=proximity_channel,
                    )
                    if self.channels > 1
                    else chunk
                )
                proximity_chunks.append(proximity_chunk)
                for channel, state in states.items():
                    channel_chunk = (
                        select_pcm16_channel(
                            chunk,
                            channels=self.channels,
                            selected_channel=channel,
                        )
                        if self.channels > 1
                        else chunk
                    )
                    state.chunks.append(channel_chunk)
                    rms = pcm16_rms(channel_chunk)
                    state.rms_values.append(rms)
                    if elapsed <= self.baseline_seconds:
                        state.baseline_rms_values.append(rms)

                    samples = np.frombuffer(channel_chunk, dtype=np.int16)
                    predict_started = time.monotonic()
                    predictions = state.model.predict(samples)
                    predict_ms = (time.monotonic() - predict_started) * 1000
                    state.predict_ms_values.append(predict_ms)
                    label, confidence = _best_prediction(predictions)
                    if confidence > state.best_confidence:
                        state.best_label = label
                        state.best_confidence = confidence
                        state.best_confidence_ms = int(elapsed * 1000)
                    if confidence >= self.threshold:
                        state.trigger_count += 1
                        if state.first_trigger_ms is None:
                            state.first_trigger_ms = int(elapsed * 1000)

        channel_metrics: dict[str, ChannelMetrics] = {}
        for channel, state in states.items():
            pcm = b"".join(state.chunks)
            self.pcm_by_channel[channel] = pcm
            noise_rms = _mean_or_zero(state.baseline_rms_values)
            peak_rms = max(state.rms_values, default=0.0)
            channel_metrics[str(channel)] = ChannelMetrics(
                channel=channel,
                audio_ms=int(len(pcm) / 2 / self.sample_rate * 1000),
                chunks=len(state.chunks),
                best_label=state.best_label,
                best_confidence=state.best_confidence,
                best_confidence_ms=state.best_confidence_ms,
                first_trigger_ms=state.first_trigger_ms,
                trigger_count=state.trigger_count,
                noise_rms=noise_rms,
                mean_rms=_mean_or_zero(state.rms_values),
                peak_rms=peak_rms,
                snr_db=_db_ratio(peak_rms, noise_rms),
                predict_avg_ms=_mean_or_zero(state.predict_ms_values),
                predict_max_ms=max(state.predict_ms_values, default=0.0),
            )

        best_channel = select_best_channel(channel_metrics, threshold=self.threshold)
        proximity_pcm = b"".join(proximity_chunks)
        self.pcm_by_channel[proximity_channel] = proximity_pcm
        event_ms = best_channel.first_trigger_ms or best_channel.best_confidence_ms or 0
        wake_window_start_ms, wake_window_end_ms = wake_window_bounds(
            event_ms,
            pre_ms=self.wake_window_pre_ms,
            post_ms=self.wake_window_post_ms,
            audio_ms=_pcm_duration_ms(proximity_pcm, self.sample_rate),
        )
        segment_features = proximity_segment_features(
            proximity_pcm,
            sample_rate=self.sample_rate,
            start_ms=wake_window_start_ms,
            end_ms=wake_window_end_ms,
            noise_end_ms=int(self.baseline_seconds * 1000),
        )
        self.proximity_window_pcm = segment_features["segment_pcm"]
        self.metrics = DeviceMetrics(
            label=self.spec.label,
            device=self.spec.device,
            resolved_device=resolved_device,
            channel=best_channel.channel,
            proximity_channel=proximity_channel,
            audio_ms=segment_features["audio_ms"],
            chunks=best_channel.chunks,
            overflow_count=overflow_count,
            best_label=best_channel.best_label,
            best_confidence=best_channel.best_confidence,
            first_trigger_ms=best_channel.first_trigger_ms,
            trigger_count=best_channel.trigger_count,
            noise_rms=segment_features["noise_rms"],
            mean_rms=segment_features["mean_rms"],
            peak_rms=segment_features["peak_rms"],
            snr_db=segment_features["snr_db"],
            predict_avg_ms=best_channel.predict_avg_ms,
            predict_max_ms=best_channel.predict_max_ms,
            candidate_channels=",".join(str(channel) for channel in channel_candidates),
            channel_metrics=channel_metrics,
            wake_window_start_ms=wake_window_start_ms,
            wake_window_end_ms=wake_window_end_ms,
            segment_duration_ms=segment_features["duration_ms"],
            band_rms=segment_features["band_rms"],
            band_snr_db=segment_features["band_snr_db"],
            speech_band_ratio=segment_features["speech_band_ratio"],
        )


@dataclass(slots=True)
class _ChannelCaptureState:
    model: Any
    chunks: list[bytes] = field(default_factory=list)
    rms_values: list[float] = field(default_factory=list)
    baseline_rms_values: list[float] = field(default_factory=list)
    predict_ms_values: list[float] = field(default_factory=list)
    best_label: str = ""
    best_confidence: float = 0.0
    best_confidence_ms: int | None = None
    first_trigger_ms: int | None = None
    trigger_count: int = 0


class _LiveDeviceMonitor(threading.Thread):
    def __init__(
        self,
        *,
        spec: DeviceSpec,
        models: dict[int, Any],
        sd,
        sample_rate: int,
        channels: int,
        block_ms: int,
        threshold: float,
        buffer_seconds: float,
        start_event: threading.Event,
        stop_event: threading.Event,
        event_queue: queue.Queue[_LiveWakeCandidate],
    ):
        super().__init__(name=f"wake-live-{spec.label}", daemon=True)
        self.spec = spec
        self.models = models
        self.sd = sd
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_ms = block_ms
        self.threshold = threshold
        self.max_buffer_ms = max(1000, int(buffer_seconds * 1000))
        self.start_event = start_event
        self.stop_event = stop_event
        self.event_queue = event_queue
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.resolved_device: int | None = None
        self.wake_channels: list[int] = []
        self.proximity_channel = 0
        self.audio_ms = 0
        self.overflow_count = 0
        self._lock = threading.Lock()
        self._proximity_chunks: deque[_TimedPcmChunk] = deque()
        self._wake_chunks: dict[int, deque[_TimedPcmChunk]] = defaultdict(deque)
        self._states: dict[int, _ChannelCaptureState] = {}
        self._trigger_latched = False
        self._reset_requested = False

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.error = exc
            self.ready.set()

    def request_reset(self) -> None:
        with self._lock:
            self._reset_requested = True

    def snapshot_channel_metrics(self) -> dict[str, ChannelMetrics]:
        with self._lock:
            audio_ms = self.audio_ms
            states = {
                channel: _copy_channel_state(state)
                for channel, state in self._states.items()
            }
        metrics: dict[str, ChannelMetrics] = {}
        for channel, state in states.items():
            noise_rms = _mean_or_zero(state.baseline_rms_values)
            peak_rms = max(state.rms_values, default=0.0)
            metrics[str(channel)] = ChannelMetrics(
                channel=channel,
                audio_ms=audio_ms,
                chunks=len(state.rms_values),
                best_label=state.best_label,
                best_confidence=state.best_confidence,
                best_confidence_ms=state.best_confidence_ms,
                first_trigger_ms=state.first_trigger_ms,
                trigger_count=state.trigger_count,
                noise_rms=noise_rms,
                mean_rms=_mean_or_zero(state.rms_values),
                peak_rms=peak_rms,
                snr_db=_db_ratio(peak_rms, noise_rms),
                predict_avg_ms=_mean_or_zero(state.predict_ms_values),
                predict_max_ms=max(state.predict_ms_values, default=0.0),
            )
        return metrics

    def proximity_pcm_between(self, start_ms: int, end_ms: int) -> bytes:
        with self._lock:
            chunks = list(self._proximity_chunks)
        return _slice_timed_chunks(chunks, start_ms=start_ms, end_ms=end_ms)

    def wake_pcm_between(self, channel: int, start_ms: int, end_ms: int) -> bytes:
        with self._lock:
            chunks = list(self._wake_chunks.get(channel, ()))
        return _slice_timed_chunks(chunks, start_ms=start_ms, end_ms=end_ms)

    def wait_until_audio_ms(self, target_ms: int, timeout_seconds: float = 1.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self.error is not None:
                raise RuntimeError(f"{self.spec.label} live capture failed: {self.error}")
            with self._lock:
                if self.audio_ms >= target_ms:
                    return
            time.sleep(0.01)

    def _run(self) -> None:
        import numpy as np  # type: ignore[import-untyped]

        self.resolved_device = resolve_sounddevice_device(self.sd, self.spec.device, kind="input")
        self.wake_channels = _parse_channel_candidates(self.spec.wake_channel, self.channels)
        self.proximity_channel = _parse_proximity_channel(
            self.spec.proximity_channel,
            self.channels,
        )
        self._reset_states(reset_models=True)
        frames = max(1, int(self.sample_rate * self.block_ms / 1000))
        with self.sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=frames,
            device=self.resolved_device,
        ) as stream:
            self.ready.set()
            self.start_event.wait()
            started = time.monotonic()
            while not self.stop_event.is_set():
                self._reset_if_requested()
                data, overflowed = stream.read(frames)
                if overflowed:
                    with self._lock:
                        self.overflow_count += 1
                    continue
                raw_chunk = bytes(data)
                end_ms = int((time.monotonic() - started) * 1000)
                duration_ms = _pcm_duration_ms(raw_chunk, self.sample_rate * self.channels)
                start_ms = max(0, end_ms - duration_ms)
                proximity_chunk = (
                    select_pcm16_channel(
                        raw_chunk,
                        channels=self.channels,
                        selected_channel=self.proximity_channel,
                    )
                    if self.channels > 1
                    else raw_chunk
                )
                with self._lock:
                    self.audio_ms = end_ms
                    self._append_timed_chunk(
                        self._proximity_chunks,
                        _TimedPcmChunk(start_ms, end_ms, proximity_chunk),
                    )

                for channel in self.wake_channels:
                    channel_chunk = (
                        select_pcm16_channel(
                            raw_chunk,
                            channels=self.channels,
                            selected_channel=channel,
                        )
                        if self.channels > 1
                        else raw_chunk
                    )
                    self._process_wake_chunk(
                        channel=channel,
                        chunk=channel_chunk,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        samples=np.frombuffer(channel_chunk, dtype=np.int16),
                    )

    def _process_wake_chunk(
        self,
        *,
        channel: int,
        chunk: bytes,
        start_ms: int,
        end_ms: int,
        samples,
    ) -> None:
        state = self._states[channel]
        rms = pcm16_rms(chunk)
        predict_started = time.monotonic()
        predictions = state.model.predict(samples)
        predict_ms = (time.monotonic() - predict_started) * 1000
        label, confidence = _best_prediction(predictions)
        should_emit = False
        with self._lock:
            self._append_timed_chunk(
                self._wake_chunks[channel],
                _TimedPcmChunk(start_ms, end_ms, chunk),
            )
            state.rms_values.append(rms)
            state.predict_ms_values.append(predict_ms)
            if confidence > state.best_confidence:
                state.best_label = label
                state.best_confidence = confidence
                state.best_confidence_ms = end_ms
            if confidence >= self.threshold:
                state.trigger_count += 1
                if state.first_trigger_ms is None:
                    state.first_trigger_ms = end_ms
                if not self._trigger_latched:
                    self._trigger_latched = True
                    should_emit = True
        if should_emit:
            self.event_queue.put(
                _LiveWakeCandidate(
                    label=self.spec.label,
                    channel=channel,
                    wake_label=label,
                    confidence=confidence,
                    event_ms=end_ms,
                )
            )

    def _append_timed_chunk(
        self,
        chunks: deque[_TimedPcmChunk],
        chunk: _TimedPcmChunk,
    ) -> None:
        chunks.append(chunk)
        trim_before_ms = chunk.end_ms - self.max_buffer_ms
        while chunks and chunks[0].end_ms < trim_before_ms:
            chunks.popleft()

    def _reset_if_requested(self) -> None:
        with self._lock:
            reset_requested = self._reset_requested
            self._reset_requested = False
        if reset_requested:
            self._reset_states(reset_models=True)

    def _reset_states(self, *, reset_models: bool) -> None:
        if reset_models:
            for model in self.models.values():
                reset = getattr(model, "reset", None)
                if callable(reset):
                    reset()
        with self._lock:
            self._states = {
                channel: _ChannelCaptureState(model=self.models[channel])
                for channel in self.wake_channels
            }
            self._trigger_latched = False


def _copy_channel_state(state: _ChannelCaptureState) -> _ChannelCaptureState:
    return _ChannelCaptureState(
        model=state.model,
        chunks=[],
        rms_values=list(state.rms_values),
        baseline_rms_values=list(state.baseline_rms_values),
        predict_ms_values=list(state.predict_ms_values),
        best_label=state.best_label,
        best_confidence=state.best_confidence,
        best_confidence_ms=state.best_confidence_ms,
        first_trigger_ms=state.first_trigger_ms,
        trigger_count=state.trigger_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect XVF3800 proximity wake data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_collect_parser(subparsers)
    _add_free_parser(subparsers)
    _add_e2e_parser(subparsers)
    _add_wake_live_parser(subparsers)
    _add_prod_live_parser(subparsers)
    _add_summarize_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "collect":
        return _run_collect(args)
    if args.command == "free":
        return _run_free(args)
    if args.command == "e2e":
        return _run_e2e(args)
    if args.command == "wake-live":
        return _run_wake_live(args)
    if args.command == "prod-live":
        return _run_prod_live(args)
    if args.command == "summarize":
        return _run_summarize(args)
    raise ValueError(f"Unsupported command: {args.command}")


def _add_collect_parser(subparsers) -> None:
    parser = subparsers.add_parser("collect", help="Run guided two-device wake collection")
    parser.add_argument("--device-a", default=DEFAULT_DEVICE_A_INPUT)
    parser.add_argument("--device-b", default=DEFAULT_DEVICE_B_INPUT)
    parser.add_argument("--label-a", default="xvf1")
    parser.add_argument("--label-b", default="xvf2")
    parser.add_argument("--channel-a", dest="wake_channel_a", default=argparse.SUPPRESS)
    parser.add_argument("--channel-b", dest="wake_channel_b", default=argparse.SUPPRESS)
    parser.add_argument("--wake-channel-a", default=DEFAULT_WAKE_CHANNEL)
    parser.add_argument("--wake-channel-b", default=DEFAULT_WAKE_CHANNEL)
    parser.add_argument("--proximity-channel-a", default=DEFAULT_RAW_PROXIMITY_CHANNEL)
    parser.add_argument("--proximity-channel-b", default=DEFAULT_RAW_PROXIMITY_CHANNEL)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--block-ms", type=int, default=80)
    parser.add_argument("--model", default="alexa")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--inference-framework", choices=["onnx", "tflite"], default="onnx")
    parser.add_argument("--listen-seconds", type=float, default=5.0)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--wake-window-pre-ms", type=int, default=DEFAULT_WAKE_WINDOW_PRE_MS)
    parser.add_argument("--wake-window-post-ms", type=int, default=DEFAULT_WAKE_WINDOW_POST_MS)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--positions",
        default=DEFAULT_POSITIONS,
        help="Comma list of position:expected_label. Empty expected label is unscored.",
    )
    parser.add_argument("--output-dir", default="debug_sessions/wake_proximity")
    parser.add_argument("--no-wav", action="store_true", help="Do not save per-device WAV files")
    parser.add_argument(
        "--allow-non-triggered-winner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow a below-threshold device to win after another device wakes, "
            "when raw-channel proximity evidence is stronger. On by default."
        ),
    )
    parser.add_argument(
        "--non-triggered-override-rms-ratio",
        type=float,
        default=DEFAULT_NON_TRIGGERED_OVERRIDE_RMS_RATIO,
        help=(
            "Minimum raw speech-band RMS ratio for a below-threshold device "
            "to beat a triggered one."
        ),
    )
    parser.add_argument(
        "--non-triggered-override-min-snr-margin-db",
        type=float,
        default=DEFAULT_NON_TRIGGERED_OVERRIDE_MIN_SNR_MARGIN_DB,
        help="Minimum speech-band SNR margin for a below-threshold device override.",
    )
    parser.add_argument("--confidence-weight", type=float, default=0.15)
    parser.add_argument("--rms-weight", type=float, default=0.25)
    parser.add_argument("--snr-weight", type=float, default=0.60)
    parser.add_argument("--late-penalty-weight", type=float, default=0.0)


def _add_free_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "free",
        help="Continuously run free-form two-device wake arbitration",
    )
    _add_common_live_parser_args(parser)
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Number of listen windows. 0 means keep running until Ctrl+C.",
    )
    parser.add_argument("--gap-seconds", type=float, default=0.5)
    parser.add_argument("--output-dir", default="debug_sessions/wake_proximity_free")
    parser.add_argument("--no-wav", action="store_true", help="Do not save per-device WAV files")


def _add_e2e_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "e2e",
        help="Run guided nearest-wake input-to-selected-output tests",
    )
    _add_common_live_parser_args(parser)
    parser.add_argument("--output-a", default=DEFAULT_DEVICE_A_OUTPUT)
    parser.add_argument("--output-b", default=DEFAULT_DEVICE_B_OUTPUT)
    parser.add_argument("--ack-wav", default="default")
    parser.add_argument("--no-ack", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--positions",
        default=DEFAULT_E2E_POSITIONS,
        help="Comma list of position:expected_label for scored E2E trials.",
    )
    parser.add_argument("--output-dir", default="debug_sessions/wake_proximity_e2e")
    parser.add_argument("--no-wav", action="store_true", help="Do not save per-device WAV files")


def _add_wake_live_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "wake-live",
        help="Run production-like nearest wake acknowledgement forever",
    )
    _add_common_live_parser_args(parser, include_listen_seconds=False)
    _add_live_ack_args(parser)
    parser.add_argument("--output-dir", default="debug_sessions/wake_proximity_live_wake")
    parser.add_argument("--no-wav", action="store_true", help="Do not save per-device WAV files")
    parser.add_argument("--cooldown-seconds", type=float, default=1.2)
    parser.add_argument("--buffer-seconds", type=float, default=8.0)


def _add_prod_live_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "prod-live",
        help="Run nearest wake followed by the configured production assistant forever",
    )
    _add_common_live_parser_args(parser, include_listen_seconds=False)
    _add_live_ack_args(parser)
    parser.add_argument("--config", default=DEFAULT_PROD_LIVE_CONFIG)
    parser.add_argument("--output-dir", default="debug_sessions/wake_proximity_prod_live")
    parser.add_argument("--no-wav", action="store_true", help="Do not save per-device WAV files")
    parser.add_argument(
        "--system-input-dump",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the assistant system input dump during production turns. Off by default.",
    )
    parser.add_argument("--cooldown-seconds", type=float, default=0.2)
    parser.add_argument("--buffer-seconds", type=float, default=8.0)


def _add_live_ack_args(parser) -> None:
    parser.add_argument("--output-a", default=DEFAULT_DEVICE_A_OUTPUT)
    parser.add_argument("--output-b", default=DEFAULT_DEVICE_B_OUTPUT)
    parser.add_argument("--ack-wav", default="default")
    parser.add_argument("--no-ack", action="store_true")


def _add_common_live_parser_args(parser, *, include_listen_seconds: bool = True) -> None:
    parser.add_argument("--device-a", default=DEFAULT_DEVICE_A_INPUT)
    parser.add_argument("--device-b", default=DEFAULT_DEVICE_B_INPUT)
    parser.add_argument("--label-a", default="xvf1")
    parser.add_argument("--label-b", default="xvf2")
    parser.add_argument("--channel-a", dest="wake_channel_a", default=argparse.SUPPRESS)
    parser.add_argument("--channel-b", dest="wake_channel_b", default=argparse.SUPPRESS)
    parser.add_argument("--wake-channel-a", default=DEFAULT_WAKE_CHANNEL)
    parser.add_argument("--wake-channel-b", default=DEFAULT_WAKE_CHANNEL)
    parser.add_argument("--proximity-channel-a", default=DEFAULT_RAW_PROXIMITY_CHANNEL)
    parser.add_argument("--proximity-channel-b", default=DEFAULT_RAW_PROXIMITY_CHANNEL)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--block-ms", type=int, default=80)
    parser.add_argument("--model", default="alexa")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--inference-framework", choices=["onnx", "tflite"], default="onnx")
    if include_listen_seconds:
        parser.add_argument("--listen-seconds", type=float, default=5.0)
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=1.0,
        help="Noise lookback seconds before the wake window.",
    )
    parser.add_argument("--wake-window-pre-ms", type=int, default=DEFAULT_WAKE_WINDOW_PRE_MS)
    parser.add_argument("--wake-window-post-ms", type=int, default=DEFAULT_WAKE_WINDOW_POST_MS)
    parser.add_argument(
        "--allow-non-triggered-winner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow a below-threshold device to win after another device wakes, "
            "when raw-channel proximity evidence is stronger. On by default."
        ),
    )
    parser.add_argument(
        "--non-triggered-override-rms-ratio",
        type=float,
        default=DEFAULT_NON_TRIGGERED_OVERRIDE_RMS_RATIO,
        help=(
            "Minimum raw speech-band RMS ratio for a below-threshold device "
            "to beat a triggered one."
        ),
    )
    parser.add_argument(
        "--non-triggered-override-min-snr-margin-db",
        type=float,
        default=DEFAULT_NON_TRIGGERED_OVERRIDE_MIN_SNR_MARGIN_DB,
        help="Minimum speech-band SNR margin for a below-threshold device override.",
    )
    parser.add_argument("--confidence-weight", type=float, default=0.15)
    parser.add_argument("--rms-weight", type=float, default=0.25)
    parser.add_argument("--snr-weight", type=float, default=0.60)
    parser.add_argument("--late-penalty-weight", type=float, default=0.0)


def _add_summarize_parser(subparsers) -> None:
    parser = subparsers.add_parser("summarize", help="Summarize a wake proximity run")
    parser.add_argument("path", help="Run directory or trials.jsonl path")


def _run_collect(args) -> int:
    if args.sample_rate != 16000:
        raise RuntimeError("openWakeWord requires 16 kHz audio.")
    if args.listen_seconds <= args.baseline_seconds:
        raise RuntimeError("--listen-seconds must be greater than --baseline-seconds.")

    specs = [
        DeviceSpec(
            label=args.label_a,
            device=args.device_a,
            wake_channel=str(args.wake_channel_a or DEFAULT_WAKE_CHANNEL),
            proximity_channel=str(args.proximity_channel_a),
        ),
        DeviceSpec(
            label=args.label_b,
            device=args.device_b,
            wake_channel=str(args.wake_channel_b or DEFAULT_WAKE_CHANNEL),
            proximity_channel=str(args.proximity_channel_b),
        ),
    ]
    positions = parse_positions(args.positions)
    weights = ScoreWeights(
        confidence=args.confidence_weight,
        rms=args.rms_weight,
        snr=args.snr_weight,
        late_penalty=args.late_penalty_weight,
    )
    run_dir = _create_run_dir(Path(args.output_dir))
    audio_dir = run_dir / "audio"
    if not args.no_wav:
        audio_dir.mkdir(parents=True, exist_ok=True)

    sd = _require_sounddevice()
    models = _load_models_for_specs(
        specs,
        model=args.model,
        inference_framework=args.inference_framework,
        channels=args.channels,
    )
    _write_run_config(run_dir, args, specs, positions, weights)
    print(f"输出目录: {run_dir}")
    print("采集规则: 每轮按 Enter 后先保持安静 1 秒，再喊一次唤醒词。")
    print("按 Ctrl+C 可以中断；已完成的轮次会保留在 trials.jsonl/csv。")

    results: list[TrialResult] = []
    trial_id = 0
    try:
        for position in positions:
            for repetition in range(1, args.repetitions + 1):
                trial_id += 1
                result = _run_one_trial(
                    trial_id=trial_id,
                    position=position,
                    repetition=repetition,
                    args=args,
                    specs=specs,
                    models=models,
                    sd=sd,
                    weights=weights,
                    audio_dir=audio_dir if not args.no_wav else None,
                )
                results.append(result)
                append_trial_jsonl(run_dir / "trials.jsonl", result)
                write_trials_csv(run_dir / "trials.csv", results, [spec.label for spec in specs])
                _print_trial_result(result)
    except KeyboardInterrupt:
        print("\n采集中断，开始汇总已完成轮次。")

    summary = summarize_trial_results(results)
    write_summary(run_dir / "summary.json", summary)
    print_summary(summary)
    return 0


def _run_free(args) -> int:
    _validate_live_args(args)
    specs = _device_specs_from_args(args)
    positions = [PositionSpec(name="free", expected_device="")]
    weights = _score_weights_from_args(args)
    run_dir = _create_run_dir(Path(args.output_dir))
    audio_dir = run_dir / "audio"
    if not args.no_wav:
        audio_dir.mkdir(parents=True, exist_ok=True)

    sd = _require_sounddevice()
    models = _load_models_for_specs(
        specs,
        model=args.model,
        inference_framework=args.inference_framework,
        channels=args.channels,
    )
    _write_run_config(run_dir, args, specs, positions, weights)
    print(f"输出目录: {run_dir}")
    print("自由唤醒模式：脚本会自动重复监听；每轮开始后先安静 1 秒，再自由喊唤醒词。")
    print("按 Ctrl+C 停止；已完成轮次会保留在 trials.jsonl/csv。")

    results: list[TrialResult] = []
    trial_id = 0
    try:
        while args.rounds <= 0 or trial_id < args.rounds:
            trial_id += 1
            if trial_id > 1 and args.gap_seconds > 0:
                time.sleep(args.gap_seconds)
            result = _run_one_trial(
                trial_id=trial_id,
                position=positions[0],
                repetition=trial_id,
                args=args,
                specs=specs,
                models=models,
                sd=sd,
                weights=weights,
                audio_dir=audio_dir if not args.no_wav else None,
                prompt=False,
            )
            results.append(result)
            append_trial_jsonl(run_dir / "trials.jsonl", result)
            write_trials_csv(run_dir / "trials.csv", results, [spec.label for spec in specs])
            _print_trial_result(result)
    except KeyboardInterrupt:
        print("\n自由唤醒测试停止，开始汇总已完成轮次。")

    summary = summarize_trial_results(results)
    write_summary(run_dir / "summary.json", summary)
    print_summary(summary)
    return 0


def _run_e2e(args) -> int:
    _validate_live_args(args)
    specs = _device_specs_from_args(args)
    positions = parse_positions(args.positions)
    weights = _score_weights_from_args(args)
    run_dir = _create_run_dir(Path(args.output_dir))
    audio_dir = run_dir / "audio"
    if not args.no_wav:
        audio_dir.mkdir(parents=True, exist_ok=True)

    sd = _require_sounddevice()
    models = _load_models_for_specs(
        specs,
        model=args.model,
        inference_framework=args.inference_framework,
        channels=args.channels,
    )
    _write_run_config(run_dir, args, specs, positions, weights)
    output_devices = {
        args.label_a: args.output_a,
        args.label_b: args.output_b,
    }
    print(f"输出目录: {run_dir}")
    print("端到端模式：按提示站到指定位置喊唤醒词，脚本会选择近端设备并在该设备播放 ack。")
    print("按 Ctrl+C 可以中断；已完成轮次会保留在 trials.jsonl/csv。")

    results: list[TrialResult] = []
    trial_id = 0
    try:
        for position in positions:
            for repetition in range(1, args.repetitions + 1):
                trial_id += 1
                result = _run_one_trial(
                    trial_id=trial_id,
                    position=position,
                    repetition=repetition,
                    args=args,
                    specs=specs,
                    models=models,
                    sd=sd,
                    weights=weights,
                    audio_dir=audio_dir if not args.no_wav else None,
                )
                _play_selected_wake_ack(result, args=args, output_devices=output_devices)
                results.append(result)
                append_trial_jsonl(run_dir / "trials.jsonl", result)
                write_trials_csv(run_dir / "trials.csv", results, [spec.label for spec in specs])
                _print_trial_result(result)
    except KeyboardInterrupt:
        print("\n端到端测试中断，开始汇总已完成轮次。")

    summary = summarize_trial_results(results)
    write_summary(run_dir / "summary.json", summary)
    print_summary(summary)
    return 0


def _run_wake_live(args) -> int:
    _validate_live_args(args)
    specs = _device_specs_from_args(args)
    positions = [PositionSpec(name="live", expected_device="")]
    weights = _score_weights_from_args(args)
    output_devices = _output_devices_from_args(args)
    run_dir = _create_run_dir(Path(args.output_dir))
    audio_dir = run_dir / "audio"
    if not args.no_wav:
        audio_dir.mkdir(parents=True, exist_ok=True)

    sd = _require_sounddevice()
    models = _load_models_for_specs(
        specs,
        model=args.model,
        inference_framework=args.inference_framework,
        channels=args.channels,
    )
    _write_run_config(run_dir, args, specs, positions, weights)
    print(f"输出目录: {run_dir}")
    print("纯唤醒 live：直接喊唤醒词，脚本会选择近端设备并在该设备播放 ack。Ctrl+C 停止。")

    results: list[TrialResult] = []
    monitors, event_queue = _start_live_monitors(
        args=args,
        specs=specs,
        models=models,
        sd=sd,
    )
    try:
        while True:
            live = _wait_for_live_wake(
                args=args,
                specs=specs,
                monitors=monitors,
                event_queue=event_queue,
                weights=weights,
                trial_id=len(results) + 1,
                position=positions[0],
            )
            _play_selected_wake_ack(live.trial, args=args, output_devices=output_devices)
            if audio_dir is not None:
                _write_live_wavs(audio_dir, live, args=args, monitors=monitors)
            results.append(live.trial)
            append_trial_jsonl(run_dir / "trials.jsonl", live.trial)
            write_trials_csv(run_dir / "trials.csv", results, [spec.label for spec in specs])
            _print_trial_result(live.trial)
            _reset_live_monitors(monitors, event_queue=event_queue)
            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)
    except KeyboardInterrupt:
        print("\n纯唤醒 live 已停止。")
    finally:
        _stop_live_monitors(monitors)

    summary = summarize_trial_results(results)
    write_summary(run_dir / "summary.json", summary)
    print_summary(summary)
    return 0


def _run_prod_live(args) -> int:
    _validate_live_args(args)
    load_dotenv()
    specs = _device_specs_from_args(args)
    positions = [PositionSpec(name="prod_live", expected_device="")]
    weights = _score_weights_from_args(args)
    output_devices = _output_devices_from_args(args)
    input_devices = {spec.label: spec.device for spec in specs}
    run_dir = _create_run_dir(Path(args.output_dir))
    audio_dir = run_dir / "audio"
    if not args.no_wav:
        audio_dir.mkdir(parents=True, exist_ok=True)

    sd = _require_sounddevice()
    models = _load_models_for_specs(
        specs,
        model=args.model,
        inference_framework=args.inference_framework,
        channels=args.channels,
    )
    _write_run_config(run_dir, args, specs, positions, weights)
    print(f"输出目录: {run_dir}")
    print("生产端到端 live：直接喊唤醒词，然后按正式链路说指令；Ctrl+C 停止。")
    print("正在预热正式链路：VAD/STT/LLM/TTS 会按两台设备各初始化一次...")
    assistants = _build_production_assistants(
        args,
        run_dir=run_dir,
        input_devices=input_devices,
        output_devices=output_devices,
    )
    print("正式链路预热完成，开始监听近端唤醒。")

    results: list[TrialResult] = []
    results_lock = threading.Lock()
    shutdown_event = threading.Event()
    active_labels: set[str] = set()
    active_lock = threading.Lock()
    monitor_lock = threading.Lock()
    turn_threads: list[threading.Thread] = []
    spec_by_label = {spec.label: spec for spec in specs}
    next_trial_id = 0
    monitors, event_queue = _start_live_monitors(
        args=args,
        specs=specs,
        models=models,
        sd=sd,
    )
    try:
        while True:
            print("近端唤醒监听中...")
            with monitor_lock:
                current_monitors = dict(monitors)
            _raise_live_errors(current_monitors.values())
            try:
                candidate = event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with active_lock:
                if candidate.label in active_labels:
                    continue
                with monitor_lock:
                    eligible_labels = {
                        label for label in monitors
                        if label not in active_labels and label in spec_by_label
                    }
            if candidate.label not in eligible_labels:
                continue

            eligible_specs = [spec_by_label[label] for label in eligible_labels]
            next_trial_id += 1
            with monitor_lock:
                current_monitors = dict(monitors)
            live = _build_live_wake_from_candidate(
                args=args,
                specs=eligible_specs,
                monitors=current_monitors,
                event_queue=event_queue,
                weights=weights,
                candidate=candidate,
                trial_id=next_trial_id,
                position=positions[0],
            )
            selected = live.trial.selected_device
            if audio_dir is not None:
                _write_live_wavs(audio_dir, live, args=args, monitors=current_monitors)
            if selected == "no_wake":
                _reset_live_monitors(monitors, event_queue=event_queue)
                _record_live_result(
                    live.trial,
                    results=results,
                    results_lock=results_lock,
                    run_dir=run_dir,
                    labels=[spec.label for spec in specs],
                )
                _print_trial_result(live.trial)
                continue
            with active_lock:
                if selected in active_labels:
                    continue
                active_labels.add(selected)
            with monitor_lock:
                selected_monitor = monitors.pop(selected, None)
            if selected_monitor is not None:
                _stop_live_monitor(selected_monitor)
            with monitor_lock:
                reset_monitors = {
                    label: monitor
                    for label, monitor in monitors.items()
                    if label in eligible_labels
                }
            _reset_live_monitors(
                reset_monitors,
                event_queue=event_queue,
                drain_labels=set(eligible_labels),
                max_event_ms=live.trial.global_wake_window_end_ms,
            )

            def run_turn(selected_label: str, live_result: LiveWakeResult) -> None:
                try:
                    _run_selected_production_turn(
                        live_result,
                        args=args,
                        assistants=assistants,
                        output_devices=output_devices,
                    )
                finally:
                    _record_live_result(
                        live_result.trial,
                        results=results,
                        results_lock=results_lock,
                        run_dir=run_dir,
                        labels=[spec.label for spec in specs],
                    )
                    _print_trial_result(live_result.trial)
                    with active_lock:
                        active_labels.discard(selected_label)
                    if not shutdown_event.is_set():
                        replacement = _start_live_monitor(
                            args=args,
                            spec=spec_by_label[selected_label],
                            models=models[selected_label],
                            sd=sd,
                            event_queue=event_queue,
                        )
                        with monitor_lock:
                            monitors[selected_label] = replacement
                    if args.cooldown_seconds > 0:
                        time.sleep(args.cooldown_seconds)

            thread = threading.Thread(
                target=run_turn,
                args=(selected, live),
                name=f"wake-proximity-prod-turn-{selected}-{live.trial.trial_id}",
                daemon=True,
            )
            turn_threads.append(thread)
            thread.start()
    except KeyboardInterrupt:
        print("\n生产端到端 live 已停止。")
    finally:
        shutdown_event.set()
        with monitor_lock:
            current_monitors = dict(monitors)
        _stop_live_monitors(current_monitors)
        for thread in turn_threads:
            thread.join(timeout=2.0)
        _close_production_assistants(assistants)

    summary = summarize_trial_results(results)
    write_summary(run_dir / "summary.json", summary)
    print_summary(summary)
    return 0


def _start_live_monitors(
    *,
    args,
    specs: list[DeviceSpec],
    models: dict[str, dict[int, Any]],
    sd,
) -> tuple[dict[str, _LiveDeviceMonitor], queue.Queue[_LiveWakeCandidate]]:
    event_queue: queue.Queue[_LiveWakeCandidate] = queue.Queue()
    monitors = _start_live_monitors_for_specs(
        args=args,
        specs=specs,
        models=models,
        sd=sd,
        event_queue=event_queue,
    )
    return monitors, event_queue


def _start_live_monitors_for_specs(
    *,
    args,
    specs: list[DeviceSpec],
    models: dict[str, dict[int, Any]],
    sd,
    event_queue: queue.Queue[_LiveWakeCandidate],
) -> dict[str, _LiveDeviceMonitor]:
    monitors = {
        spec.label: _start_live_monitor(
            args=args,
            spec=spec,
            models=models[spec.label],
            sd=sd,
            event_queue=event_queue,
        )
        for spec in specs
    }
    return monitors


def _start_live_monitor(
    *,
    args,
    spec: DeviceSpec,
    models: dict[int, Any],
    sd,
    event_queue: queue.Queue[_LiveWakeCandidate],
) -> _LiveDeviceMonitor:
    start_event = threading.Event()
    stop_event = threading.Event()
    monitor = _LiveDeviceMonitor(
        spec=spec,
        models=models,
        sd=sd,
        sample_rate=args.sample_rate,
        channels=args.channels,
        block_ms=args.block_ms,
        threshold=args.threshold,
        buffer_seconds=args.buffer_seconds,
        start_event=start_event,
        stop_event=stop_event,
        event_queue=event_queue,
    )
    monitor.start()
    _wait_for_live_ready([monitor])
    start_event.set()
    return monitor


def _wait_for_live_wake(
    *,
    args,
    specs: list[DeviceSpec],
    monitors: dict[str, _LiveDeviceMonitor],
    event_queue: queue.Queue[_LiveWakeCandidate],
    weights: ScoreWeights,
    trial_id: int,
    position: PositionSpec,
) -> LiveWakeResult:
    while True:
        _raise_live_errors(monitors.values())
        try:
            candidate = event_queue.get(timeout=0.2)
            break
        except queue.Empty:
            continue

    global_start_ms = max(0, candidate.event_ms - max(0, args.wake_window_pre_ms))
    global_end_ms = candidate.event_ms + max(0, args.wake_window_post_ms)
    for monitor in monitors.values():
        monitor.wait_until_audio_ms(global_end_ms, timeout_seconds=1.0)

    live = _build_live_wake_result(
        args=args,
        specs=specs,
        monitors=monitors,
        weights=weights,
        candidate=candidate,
        trial_id=trial_id,
        position=position,
        global_start_ms=global_start_ms,
        global_end_ms=global_end_ms,
    )
    return live


def _build_live_wake_from_candidate(
    *,
    args,
    specs: list[DeviceSpec],
    monitors: dict[str, _LiveDeviceMonitor],
    event_queue: queue.Queue[_LiveWakeCandidate],
    weights: ScoreWeights,
    candidate: _LiveWakeCandidate,
    trial_id: int,
    position: PositionSpec,
) -> LiveWakeResult:
    global_start_ms = max(0, candidate.event_ms - max(0, args.wake_window_pre_ms))
    global_end_ms = candidate.event_ms + max(0, args.wake_window_post_ms)
    for spec in specs:
        monitor = monitors.get(spec.label)
        if monitor is not None:
            monitor.wait_until_audio_ms(global_end_ms, timeout_seconds=1.0)
    _drain_live_events(
        event_queue,
        labels={spec.label for spec in specs},
        max_event_ms=global_end_ms,
    )
    return _build_live_wake_result(
        args=args,
        specs=specs,
        monitors=monitors,
        weights=weights,
        candidate=candidate,
        trial_id=trial_id,
        position=position,
        global_start_ms=global_start_ms,
        global_end_ms=global_end_ms,
    )


def _build_live_wake_result(
    *,
    args,
    specs: list[DeviceSpec],
    monitors: dict[str, _LiveDeviceMonitor],
    weights: ScoreWeights,
    candidate: _LiveWakeCandidate,
    trial_id: int,
    position: PositionSpec,
    global_start_ms: int,
    global_end_ms: int,
) -> LiveWakeResult:
    devices: dict[str, DeviceMetrics] = {}
    wake_pcm_by_label: dict[str, bytes] = {}
    noise_start_ms = max(0, global_start_ms - int(args.baseline_seconds * 1000))
    for spec in specs:
        monitor = monitors[spec.label]
        channel_metrics = monitor.snapshot_channel_metrics()
        best_channel = select_best_channel(channel_metrics, threshold=args.threshold)
        segment_pcm = monitor.proximity_pcm_between(global_start_ms, global_end_ms)
        noise_pcm = monitor.proximity_pcm_between(noise_start_ms, global_start_ms)
        segment_features = proximity_features_from_pcm(
            segment_pcm,
            noise_pcm,
            sample_rate=args.sample_rate,
            duration_ms=global_end_ms - global_start_ms,
        )
        with monitor._lock:
            overflow_count = monitor.overflow_count
        metrics = DeviceMetrics(
            label=spec.label,
            device=spec.device,
            resolved_device=monitor.resolved_device,
            channel=best_channel.channel,
            proximity_channel=monitor.proximity_channel,
            audio_ms=segment_features["audio_ms"],
            chunks=best_channel.chunks,
            overflow_count=overflow_count,
            best_label=best_channel.best_label,
            best_confidence=best_channel.best_confidence,
            first_trigger_ms=best_channel.first_trigger_ms,
            trigger_count=best_channel.trigger_count,
            noise_rms=segment_features["noise_rms"],
            mean_rms=segment_features["mean_rms"],
            peak_rms=segment_features["peak_rms"],
            snr_db=segment_features["snr_db"],
            predict_avg_ms=best_channel.predict_avg_ms,
            predict_max_ms=best_channel.predict_max_ms,
            candidate_channels=",".join(str(channel) for channel in monitor.wake_channels),
            channel_metrics=channel_metrics,
            wake_window_start_ms=global_start_ms,
            wake_window_end_ms=global_end_ms,
            segment_duration_ms=segment_features["duration_ms"],
            band_rms=segment_features["band_rms"],
            band_snr_db=segment_features["band_snr_db"],
            speech_band_ratio=segment_features["speech_band_ratio"],
        )
        devices[spec.label] = metrics
        wake_pcm_by_label[spec.label] = monitor.wake_pcm_between(
            best_channel.channel,
            global_start_ms,
            global_end_ms,
        )

    apply_scores(
        devices,
        weights,
        threshold=args.threshold,
        listen_seconds=max(1.0, global_end_ms / 1000),
    )
    selected_device, margin = select_winner(
        devices,
        require_trigger=not args.allow_non_triggered_winner,
        non_triggered_override_rms_ratio=args.non_triggered_override_rms_ratio,
        non_triggered_override_min_snr_margin_db=(
            args.non_triggered_override_min_snr_margin_db
        ),
    )
    correct = selected_device == position.expected_device if position.expected_device else None
    return LiveWakeResult(
        trial=TrialResult(
            trial_id=trial_id,
            position=position.name,
            expected_device=position.expected_device,
            selected_device=selected_device,
            correct=correct,
            margin=margin,
            started_at=datetime.now().isoformat(timespec="milliseconds"),
            listen_seconds=0.0,
            baseline_seconds=args.baseline_seconds,
            threshold=args.threshold,
            model=args.model,
            devices=devices,
            trigger_source_device=candidate.label,
            global_wake_window_start_ms=global_start_ms,
            global_wake_window_end_ms=global_end_ms,
        ),
        wake_pcm_by_label=wake_pcm_by_label,
    )


def _write_live_wavs(
    audio_dir: Path,
    live: LiveWakeResult,
    *,
    args,
    monitors: dict[str, _LiveDeviceMonitor],
) -> None:
    for label, metrics in live.trial.devices.items():
        proximity_pcm = monitors[label].proximity_pcm_between(
            live.trial.global_wake_window_start_ms,
            live.trial.global_wake_window_end_ms,
        )
        proximity_filename = (
            f"wake_{live.trial.trial_id:03d}_{label}_ch{metrics.proximity_channel}.wav"
        )
        proximity_path = audio_dir / proximity_filename
        write_pcm16_wav(proximity_path, proximity_pcm, args.sample_rate)
        metrics.proximity_window_wav_path = str(proximity_path)
        metrics.proximity_wav_path = str(proximity_path)
        wake_pcm = live.wake_pcm_by_label.get(label, b"")
        wake_filename = f"wake_{live.trial.trial_id:03d}_{label}_wake_ch{metrics.channel}.wav"
        wake_path = audio_dir / wake_filename
        write_pcm16_wav(wake_path, wake_pcm, args.sample_rate)
        metrics.wav_path = str(wake_path)
        channel_metrics = metrics.channel_metrics.get(str(metrics.channel))
        if channel_metrics is not None:
            channel_metrics.wav_path = str(wake_path)


def _reset_live_monitors(
    monitors: dict[str, _LiveDeviceMonitor],
    *,
    event_queue: queue.Queue[_LiveWakeCandidate],
    drain_labels: set[str] | None = None,
    max_event_ms: int | None = None,
) -> None:
    if drain_labels is None:
        drain_labels = set(monitors)
    _drain_live_events(event_queue, labels=drain_labels, max_event_ms=max_event_ms)
    for monitor in monitors.values():
        monitor.request_reset()


def _drain_live_events(
    event_queue: queue.Queue[_LiveWakeCandidate],
    *,
    labels: set[str],
    max_event_ms: int | None = None,
) -> None:
    preserved: list[_LiveWakeCandidate] = []
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break
        should_drain = event.label in labels
        if max_event_ms is not None:
            should_drain = should_drain and event.event_ms <= max_event_ms
        if not should_drain:
            preserved.append(event)
    for event in preserved:
        event_queue.put(event)


def _record_live_result(
    result: TrialResult,
    *,
    results: list[TrialResult],
    results_lock: threading.Lock,
    run_dir: Path,
    labels: list[str],
) -> None:
    with results_lock:
        results.append(result)
        append_trial_jsonl(run_dir / "trials.jsonl", result)
        write_trials_csv(run_dir / "trials.csv", list(results), labels)


def _stop_live_monitors(
    monitors: dict[str, _LiveDeviceMonitor],
) -> None:
    for monitor in monitors.values():
        monitor.stop_event.set()
    for monitor in monitors.values():
        monitor.join(timeout=2.0)


def _stop_live_monitor(monitor: _LiveDeviceMonitor) -> None:
    monitor.stop_event.set()
    monitor.join(timeout=2.0)


def _wait_for_live_ready(monitors: list[_LiveDeviceMonitor]) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        _raise_live_errors(monitors)
        if all(monitor.ready.is_set() for monitor in monitors):
            return
        time.sleep(0.02)
    not_ready = [monitor.spec.label for monitor in monitors if not monitor.ready.is_set()]
    raise RuntimeError(f"Timed out opening live audio streams: {', '.join(not_ready)}")


def _raise_live_errors(monitors) -> None:
    for monitor in monitors:
        if monitor.error is not None:
            raise RuntimeError(
                f"{monitor.spec.label} live capture failed: {monitor.error}"
            ) from monitor.error


def _run_selected_production_turn(
    live: LiveWakeResult,
    *,
    args,
    assistants: dict[str, VoiceAssistant],
    output_devices: dict[str, str | int | None],
) -> None:
    selected = live.trial.selected_device
    if selected == "no_wake":
        return
    output_device = output_devices.get(selected)
    selected_metrics = live.trial.devices[selected]
    live.trial.ack_output_device = str(output_device or "")
    assistant = assistants.get(selected)
    if assistant is None:
        live.trial.assistant_error = f"No production assistant configured for {selected}"
        return
    print(f"近端唤醒: selected={selected}，开始正式交互。")
    _activate_production_assistant(assistant)
    wake_pcm = live.wake_pcm_by_label.get(selected, b"")
    wake = WakeEvent(
        engine="nearest_wake",
        label=selected_metrics.best_label or live.trial.trigger_source_device,
        confidence=selected_metrics.best_confidence,
        pcm=wake_pcm,
        sample_rate=args.sample_rate,
        duration_ms=_pcm_duration_ms(wake_pcm, args.sample_rate),
    )
    try:
        assistant._start_system_input_dump()
        assistant._duck_music("conversation")
        try:
            turn_index = assistant.audio_dump.begin_turn()
            wake_ack_handle = assistant._start_wake_ack() if not args.no_ack else None
            assistant.session.reset()
            reply, transcript = assistant._run_audio_turn(
                wake,
                wake_ms=0,
                wake_ack_handle=wake_ack_handle,
                turn_index=turn_index,
            )
            if wake_ack_handle is not None:
                live.trial.ack_latency_ms = wake_ack_handle.result.get("latency_ms", 0)
            transcripts = [transcript] if transcript else []
            replies = [reply.text] if reply.text else []
            follow_up_seconds = assistant.config.conversation.follow_up_seconds
            if (
                follow_up_seconds > 0
                and transcript
            ) or assistant._pending_barge_utterance is not None:
                while True:
                    if assistant._pending_barge_utterance is None:
                        if follow_up_seconds <= 0:
                            break
                        log_event(
                            "session",
                            "listening_for_follow_up",
                            log_id="session.listening_for_follow_up",
                            seconds=follow_up_seconds,
                            selected_device=selected,
                        )
                        speech_start_timeout_seconds = follow_up_seconds
                    else:
                        log_event(
                            "session",
                            "processing_barge_in",
                            log_id="session.processing_barge_in",
                            selected_device=selected,
                        )
                        speech_start_timeout_seconds = 0.0
                    follow_up_wake = WakeEvent(
                        engine="follow_up",
                        confidence=1.0,
                        label="no_wake",
                    )
                    try:
                        reply, transcript = assistant._run_audio_turn(
                            follow_up_wake,
                            wake_ms=0,
                            speech_start_timeout_seconds=speech_start_timeout_seconds,
                        )
                    except SpeechStartTimeoutError:
                        log_event(
                            "session",
                            "follow_up_timeout",
                            log_id="session.follow_up_timeout",
                            next_state="returning_to_wake",
                            selected_device=selected,
                        )
                        break
                    if transcript:
                        transcripts.append(transcript)
                    if reply.text:
                        replies.append(reply.text)
                    if not transcript:
                        log_event(
                            "session",
                            "empty_follow_up",
                            log_id="session.empty_follow_up",
                            next_state="returning_to_wake",
                            selected_device=selected,
                        )
                        break
            live.trial.assistant_transcript = " | ".join(transcripts)
            live.trial.assistant_reply = " | ".join(replies)
        finally:
            assistant._unduck_music("conversation")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        live.trial.assistant_error = str(exc)
        log_event("error", "prod_live", log_id="wake_proximity.prod_live.error", error=exc)
    finally:
        assistant.audio_dump.stop_system_input_dump()


def _build_production_assistants(
    args,
    *,
    run_dir: Path,
    input_devices: dict[str, str | int | None],
    output_devices: dict[str, str | int | None],
) -> dict[str, VoiceAssistant]:
    assistants: dict[str, VoiceAssistant] = {}
    for label, input_device in input_devices.items():
        config = _production_config_for_selected_device(
            args,
            run_dir=run_dir,
            label=label,
            input_device=input_device,
            output_device=output_devices.get(label),
        )
        configure_logging(config.logging)
        assistants[label] = VoiceAssistant(config)
    return assistants


def _activate_production_assistant(assistant: VoiceAssistant) -> None:
    configure_logging(assistant.config.logging)
    configure_audio_dump(assistant.audio_dump)
    configure_log_files(
        debug_log_path=assistant.audio_dump.debug_log_path(),
        text_record_dir=assistant.audio_dump.text_record_dir(),
    )


def _close_production_assistants(assistants: dict[str, VoiceAssistant]) -> None:
    for assistant in assistants.values():
        assistant.close()


def _production_config_for_selected_device(
    args,
    *,
    run_dir: Path,
    label: str,
    input_device: str | int | None,
    output_device: str | int | None,
) -> AssistantConfig:
    config = load_config(args.config)
    config.input.mode = "audio"
    config.audio.device = input_device
    config.wake.engine = "disabled"
    config.debug.output_dir = str(run_dir / "assistant_debug" / label)
    config.debug.system_input_dump_enabled = bool(args.system_input_dump)
    if args.no_ack:
        config.wake_ack.enabled = False
    else:
        config.wake_ack.enabled = True
        config.wake_ack.wav_path = args.ack_wav
        config.wake_ack.playback_device = output_device
    config.tts.playback_device = output_device
    config.music.playback_device = output_device
    return config


def _validate_live_args(args) -> None:
    if args.sample_rate != 16000:
        raise RuntimeError("openWakeWord requires 16 kHz audio.")
    if hasattr(args, "listen_seconds") and args.listen_seconds <= args.baseline_seconds:
        raise RuntimeError("--listen-seconds must be greater than --baseline-seconds.")


def _device_specs_from_args(args) -> list[DeviceSpec]:
    return [
        DeviceSpec(
            label=args.label_a,
            device=args.device_a,
            wake_channel=str(args.wake_channel_a or DEFAULT_WAKE_CHANNEL),
            proximity_channel=str(args.proximity_channel_a),
        ),
        DeviceSpec(
            label=args.label_b,
            device=args.device_b,
            wake_channel=str(args.wake_channel_b or DEFAULT_WAKE_CHANNEL),
            proximity_channel=str(args.proximity_channel_b),
        ),
    ]


def _score_weights_from_args(args) -> ScoreWeights:
    return ScoreWeights(
        confidence=args.confidence_weight,
        rms=args.rms_weight,
        snr=args.snr_weight,
        late_penalty=args.late_penalty_weight,
    )


def _output_devices_from_args(args) -> dict[str, str | int | None]:
    return {
        args.label_a: args.output_a,
        args.label_b: args.output_b,
    }


def _play_selected_wake_ack(
    result: TrialResult,
    *,
    args,
    output_devices: dict[str, str | int | None],
) -> None:
    if args.no_ack or result.selected_device == "no_wake":
        return
    output_device = output_devices.get(result.selected_device)
    if output_device is None:
        result.ack_error = f"No output device configured for {result.selected_device}"
        print(f"ack skipped: {result.ack_error}")
        return

    result.ack_output_device = str(output_device)
    started = time.monotonic()
    try:
        create_wake_ack_player(
            WakeAckConfig(
                enabled=True,
                wav_path=args.ack_wav,
                playback_device=output_device,
            )
        ).play()
        result.ack_latency_ms = int((time.monotonic() - started) * 1000)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.ack_latency_ms = int((time.monotonic() - started) * 1000)
        result.ack_error = str(exc)
        print(f"ack error: {result.ack_error}")


def _run_one_trial(
    *,
    trial_id: int,
    position: PositionSpec,
    repetition: int,
    args,
    specs: list[DeviceSpec],
    models: dict[str, dict[int, Any]],
    sd,
    weights: ScoreWeights,
    audio_dir: Path | None,
    prompt: bool = True,
) -> TrialResult:
    start_event = threading.Event()
    workers = [
        _DeviceTrialWorker(
            spec=spec,
            models=models[spec.label],
            sd=sd,
            sample_rate=args.sample_rate,
            channels=args.channels,
            block_ms=args.block_ms,
            listen_seconds=args.listen_seconds,
            baseline_seconds=args.baseline_seconds,
            threshold=args.threshold,
            wake_window_pre_ms=args.wake_window_pre_ms,
            wake_window_post_ms=args.wake_window_post_ms,
            start_event=start_event,
        )
        for spec in specs
    ]
    for worker in workers:
        worker.start()
    _wait_for_ready(workers)

    expected = position.expected_device or "unscored"
    if prompt:
        input(
            f"\n第 {trial_id} 轮 | 位置={position.name} | 期望={expected} | "
            f"重复={repetition}/{args.repetitions}\n"
            "按 Enter 开始..."
        )
    else:
        print(
            f"\n第 {trial_id} 轮 | 自由唤醒 | 监听 {args.listen_seconds:g}s | "
            f"先安静 {args.baseline_seconds:g}s 后喊唤醒词"
        )
    started_at = datetime.now().isoformat(timespec="milliseconds")
    print(f"开始监听 {args.listen_seconds:g}s：先安静 {args.baseline_seconds:g}s，然后喊唤醒词。")
    start_event.set()
    deadline = time.monotonic() + args.listen_seconds + 3.0
    for worker in workers:
        timeout = max(0.1, deadline - time.monotonic())
        worker.join(timeout=timeout)
    _raise_worker_errors(workers)

    captures = {worker.spec.label: worker for worker in workers}
    devices = {label: worker.metrics for label, worker in captures.items()}
    if any(metrics is None for metrics in devices.values()):
        missing = [label for label, metrics in devices.items() if metrics is None]
        raise RuntimeError(f"Device capture did not complete: {', '.join(missing)}")
    typed_devices: dict[str, DeviceMetrics] = {
        label: metrics for label, metrics in devices.items() if metrics is not None
    }
    proximity_pcm_by_label = {
        label: captures[label].pcm_by_channel.get(metrics.proximity_channel, b"")
        for label, metrics in typed_devices.items()
    }
    (
        trigger_source_device,
        global_wake_window_start_ms,
        global_wake_window_end_ms,
        proximity_window_pcm_by_label,
    ) = apply_global_wake_window(
        typed_devices,
        proximity_pcm_by_label,
        sample_rate=args.sample_rate,
        baseline_seconds=args.baseline_seconds,
        wake_window_pre_ms=args.wake_window_pre_ms,
        wake_window_post_ms=args.wake_window_post_ms,
    )
    for label, proximity_window_pcm in proximity_window_pcm_by_label.items():
        captures[label].proximity_window_pcm = proximity_window_pcm

    apply_scores(
        typed_devices,
        weights,
        threshold=args.threshold,
        listen_seconds=args.listen_seconds,
    )
    selected_device, margin = select_winner(
        typed_devices,
        require_trigger=not args.allow_non_triggered_winner,
        non_triggered_override_rms_ratio=args.non_triggered_override_rms_ratio,
        non_triggered_override_min_snr_margin_db=(
            args.non_triggered_override_min_snr_margin_db
        ),
    )
    correct = (
        selected_device == position.expected_device
        if position.expected_device
        else None
    )
    if audio_dir is not None:
        for label, worker in captures.items():
            metrics = typed_devices[label]
            for channel, pcm in worker.pcm_by_channel.items():
                path = audio_dir / (
                    f"trial_{trial_id:03d}_{position.name}_{label}_ch{channel}.wav"
                )
                write_pcm16_wav(path, pcm, args.sample_rate)
                channel_metrics = metrics.channel_metrics.get(str(channel))
                if channel_metrics is not None:
                    channel_metrics.wav_path = str(path)
                if metrics.channel == channel:
                    metrics.wav_path = str(path)
                if metrics.proximity_channel == channel:
                    metrics.proximity_wav_path = str(path)
            window_path = audio_dir / (
                f"trial_{trial_id:03d}_{position.name}_{label}_prox_window.wav"
            )
            write_pcm16_wav(window_path, worker.proximity_window_pcm, args.sample_rate)
            metrics.proximity_window_wav_path = str(window_path)

    return TrialResult(
        trial_id=trial_id,
        position=position.name,
        expected_device=position.expected_device,
        selected_device=selected_device,
        correct=correct,
        margin=margin,
        started_at=started_at,
        listen_seconds=args.listen_seconds,
        baseline_seconds=args.baseline_seconds,
        threshold=args.threshold,
        model=args.model,
        devices=typed_devices,
        trigger_source_device=trigger_source_device,
        global_wake_window_start_ms=global_wake_window_start_ms,
        global_wake_window_end_ms=global_wake_window_end_ms,
    )


def parse_positions(raw: str) -> list[PositionSpec]:
    positions: list[PositionSpec] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, expected = item.split(":", 1)
        else:
            name, expected = item, ""
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid empty position in {raw!r}")
        positions.append(PositionSpec(name=name, expected_device=expected.strip()))
    if not positions:
        raise ValueError("--positions must contain at least one position")
    return positions


def _parse_channel_candidates(raw: str, channels: int) -> list[int]:
    value = str(raw).strip().lower()
    if value in {"auto", "*", "all"}:
        return list(range(max(1, channels)))
    candidates: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        channel = int(item)
        if channel < 0 or channel >= channels:
            raise ValueError(f"channel={channel} is outside available channels={channels}")
        candidates.append(channel)
    if not candidates:
        raise ValueError(f"Invalid empty channel list: {raw!r}")
    return candidates


def _parse_proximity_channel(raw: str, channels: int) -> int:
    value = str(raw).strip().lower()
    if value in {"auto", "*", "all"}:
        return 1 if channels > 1 else 0
    candidates = _parse_channel_candidates(value, channels)
    if len(candidates) != 1:
        raise ValueError("--proximity-channel-* must select exactly one channel")
    return candidates[0]


def wake_window_bounds(
    event_ms: int,
    *,
    pre_ms: int,
    post_ms: int,
    audio_ms: int,
) -> tuple[int, int]:
    start_ms = max(0, int(event_ms) - max(0, int(pre_ms)))
    end_ms = min(max(0, int(audio_ms)), int(event_ms) + max(0, int(post_ms)))
    if end_ms <= start_ms:
        end_ms = min(max(0, int(audio_ms)), start_ms + max(20, int(post_ms)))
    return start_ms, max(start_ms, end_ms)


def proximity_segment_features(
    pcm: bytes,
    *,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
    noise_end_ms: int,
) -> dict[str, Any]:
    segment_pcm = _slice_pcm16_mono(pcm, sample_rate, start_ms, end_ms)
    noise_pcm = _slice_pcm16_mono(pcm, sample_rate, 0, min(start_ms, noise_end_ms))
    return proximity_features_from_pcm(
        segment_pcm,
        noise_pcm,
        sample_rate=sample_rate,
        duration_ms=max(0, int(end_ms) - int(start_ms)),
    )


def proximity_features_from_pcm(
    segment_pcm: bytes,
    noise_pcm: bytes,
    *,
    sample_rate: int,
    duration_ms: int,
) -> dict[str, Any]:
    has_noise_window = _pcm_duration_ms(noise_pcm, sample_rate) >= MIN_NOISE_WINDOW_MS
    noise_rms = pcm16_rms(noise_pcm) if has_noise_window else 0.0
    mean_rms = pcm16_rms(segment_pcm)
    peak_rms = _pcm16_peak(segment_pcm)
    band_rms, band_power, full_power = _speech_band_stats(segment_pcm, sample_rate)
    if has_noise_window:
        noise_band_rms, noise_band_power, _noise_full_power = _speech_band_stats(
            noise_pcm,
            sample_rate,
        )
    else:
        noise_band_rms = 0.0
        noise_band_power = 0.0
    return {
        "segment_pcm": segment_pcm,
        "audio_ms": _pcm_duration_ms(segment_pcm, sample_rate),
        "duration_ms": max(0, int(duration_ms)),
        "noise_rms": noise_rms,
        "mean_rms": mean_rms,
        "peak_rms": peak_rms,
        "snr_db": _db_ratio(mean_rms, noise_rms) if has_noise_window else 0.0,
        "band_rms": band_rms,
        "band_snr_db": _db_ratio(band_rms, noise_band_rms) if has_noise_window else 0.0,
        "speech_band_ratio": band_power / max(full_power, 1.0),
        "noise_band_power": noise_band_power,
    }


def select_trigger_source(devices: dict[str, DeviceMetrics]) -> DeviceMetrics | None:
    triggered = [device for device in devices.values() if device.triggered]
    if not triggered:
        return None
    return min(
        triggered,
        key=lambda device: (
            device.first_trigger_ms
            if device.first_trigger_ms is not None
            else math.inf,
            -device.best_confidence,
        ),
    )


def apply_global_wake_window(
    devices: dict[str, DeviceMetrics],
    proximity_pcm_by_label: dict[str, bytes],
    *,
    sample_rate: int,
    baseline_seconds: float,
    wake_window_pre_ms: int,
    wake_window_post_ms: int,
) -> tuple[str, int, int, dict[str, bytes]]:
    trigger_source = select_trigger_source(devices)
    if trigger_source is None:
        return "", 0, 0, {}

    event_ms = trigger_source.first_trigger_ms
    if event_ms is None:
        event_ms = 0
    source_pcm = proximity_pcm_by_label.get(trigger_source.label, b"")
    start_ms, end_ms = wake_window_bounds(
        event_ms,
        pre_ms=wake_window_pre_ms,
        post_ms=wake_window_post_ms,
        audio_ms=_pcm_duration_ms(source_pcm, sample_rate),
    )
    proximity_window_pcm_by_label: dict[str, bytes] = {}
    for label, device in devices.items():
        pcm = proximity_pcm_by_label.get(label, b"")
        segment_features = proximity_segment_features(
            pcm,
            sample_rate=sample_rate,
            start_ms=start_ms,
            end_ms=end_ms,
            noise_end_ms=int(baseline_seconds * 1000),
        )
        proximity_window_pcm_by_label[label] = segment_features["segment_pcm"]
        _apply_proximity_segment_features(
            device,
            segment_features=segment_features,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    return trigger_source.label, start_ms, end_ms, proximity_window_pcm_by_label


def _apply_proximity_segment_features(
    device: DeviceMetrics,
    *,
    segment_features: dict[str, Any],
    start_ms: int,
    end_ms: int,
) -> None:
    device.audio_ms = segment_features["audio_ms"]
    device.noise_rms = segment_features["noise_rms"]
    device.mean_rms = segment_features["mean_rms"]
    device.peak_rms = segment_features["peak_rms"]
    device.snr_db = segment_features["snr_db"]
    device.wake_window_start_ms = start_ms
    device.wake_window_end_ms = end_ms
    device.segment_duration_ms = segment_features["duration_ms"]
    device.band_rms = segment_features["band_rms"]
    device.band_snr_db = segment_features["band_snr_db"]
    device.speech_band_ratio = segment_features["speech_band_ratio"]


def apply_scores(
    devices: dict[str, DeviceMetrics],
    weights: ScoreWeights,
    *,
    threshold: float,
    listen_seconds: float,
) -> None:
    max_confidence = max((device.best_confidence for device in devices.values()), default=0.0)
    max_band_rms = max((device.band_rms for device in devices.values()), default=0.0)
    max_band_snr = max((max(0.0, device.band_snr_db) for device in devices.values()), default=0.0)
    for device in devices.values():
        confidence_score = _safe_ratio(device.best_confidence, max_confidence)
        rms_score = _safe_ratio(device.band_rms, max_band_rms)
        snr_score = _safe_ratio(max(0.0, device.band_snr_db), max_band_snr)
        late_penalty = 0.0
        if device.first_trigger_ms is None:
            late_penalty = 1.0 if device.best_confidence < threshold else 0.5
        elif listen_seconds > 0:
            late_penalty = min(1.0, device.first_trigger_ms / (listen_seconds * 1000))
        device.score = (
            weights.confidence * confidence_score
            + weights.rms * rms_score
            + weights.snr * snr_score
            - weights.late_penalty * late_penalty
        )


def select_best_channel(
    channel_metrics: dict[str, ChannelMetrics],
    *,
    threshold: float,
) -> ChannelMetrics:
    if not channel_metrics:
        raise RuntimeError("No channel metrics available.")
    return max(
        channel_metrics.values(),
        key=lambda metrics: (
            metrics.best_confidence >= threshold,
            metrics.best_confidence,
            metrics.snr_db,
            metrics.peak_rms,
        ),
    )


def select_winner(
    devices: dict[str, DeviceMetrics],
    *,
    require_trigger: bool = True,
    non_triggered_override_rms_ratio: float = DEFAULT_NON_TRIGGERED_OVERRIDE_RMS_RATIO,
    non_triggered_override_min_snr_margin_db: float = (
        DEFAULT_NON_TRIGGERED_OVERRIDE_MIN_SNR_MARGIN_DB
    ),
) -> tuple[str, float]:
    triggered = [device for device in devices.values() if device.triggered]
    if not triggered:
        return "no_wake", 0.0

    candidates = list(triggered)
    if not require_trigger:
        best_triggered = max(triggered, key=lambda device: device.score)
        candidates.extend(
            device
            for device in devices.values()
            if not device.triggered
            and _allows_non_triggered_override(
                device,
                best_triggered,
                rms_ratio=non_triggered_override_rms_ratio,
                min_snr_margin_db=non_triggered_override_min_snr_margin_db,
            )
        )
    if not candidates:
        return "no_wake", 0.0
    ordered = sorted(candidates, key=lambda device: device.score, reverse=True)
    if not ordered:
        return "no_wake", 0.0
    if len(ordered) == 1:
        return ordered[0].label, ordered[0].score
    return ordered[0].label, ordered[0].score - ordered[1].score


def _allows_non_triggered_override(
    non_triggered: DeviceMetrics,
    triggered: DeviceMetrics,
    *,
    rms_ratio: float,
    min_snr_margin_db: float,
) -> bool:
    if non_triggered.noise_rms <= 0 or triggered.noise_rms <= 0:
        return False
    observed_rms_ratio = non_triggered.band_rms / max(triggered.band_rms, 1.0)
    observed_snr_margin = non_triggered.band_snr_db - triggered.band_snr_db
    return (
        observed_rms_ratio >= max(1.0, rms_ratio)
        and observed_snr_margin >= min_snr_margin_db
    )


def append_trial_jsonl(path: Path, result: TrialResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(trial_to_dict(result), ensure_ascii=False) + "\n")


def write_trials_csv(path: Path, results: list[TrialResult], labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(labels)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(trial_to_csv_row(result, labels))


def read_trial_jsonl(path: str | Path) -> list[TrialResult]:
    jsonl_path = _resolve_jsonl_path(Path(path))
    results: list[TrialResult] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                results.append(trial_from_dict(json.loads(line)))
    return results


def summarize_trial_results(results: list[TrialResult]) -> dict[str, Any]:
    return {
        "overall": _summary_bucket(results),
        "by_position": {
            position: _summary_bucket(position_results)
            for position, position_results in _group_by_position(results).items()
        },
        "confusion": _confusion(results),
        "device_averages": _device_averages(results),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    serializable = _summary_to_dict(summary)
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def trial_to_dict(result: TrialResult) -> dict[str, Any]:
    data = asdict(result)
    data["devices"] = {
        label: asdict(metrics)
        for label, metrics in result.devices.items()
    }
    return data


def trial_from_dict(data: dict[str, Any]) -> TrialResult:
    devices = {}
    for label, metrics in data.get("devices", {}).items():
        copied = dict(metrics)
        copied.setdefault("proximity_channel", 1)
        copied.setdefault("proximity_wav_path", "")
        copied.setdefault("proximity_window_wav_path", "")
        copied.setdefault("wake_window_start_ms", 0)
        copied.setdefault("wake_window_end_ms", 0)
        copied.setdefault("segment_duration_ms", copied.get("audio_ms", 0))
        copied.setdefault("band_rms", copied.get("peak_rms", 0.0))
        copied.setdefault("band_snr_db", copied.get("snr_db", 0.0))
        copied.setdefault("speech_band_ratio", 0.0)
        copied["channel_metrics"] = {
            str(channel): ChannelMetrics(
                **{
                    **channel_metrics,
                    "best_confidence_ms": channel_metrics.get("best_confidence_ms"),
                }
            )
            for channel, channel_metrics in copied.get("channel_metrics", {}).items()
        }
        devices[label] = DeviceMetrics(**copied)
    return TrialResult(
        trial_id=int(data["trial_id"]),
        position=str(data["position"]),
        expected_device=str(data.get("expected_device") or ""),
        selected_device=str(data.get("selected_device") or ""),
        correct=data.get("correct"),
        margin=float(data.get("margin") or 0.0),
        started_at=str(data.get("started_at") or ""),
        listen_seconds=float(data.get("listen_seconds") or 0.0),
        baseline_seconds=float(data.get("baseline_seconds") or 0.0),
        threshold=float(data.get("threshold") or 0.0),
        model=str(data.get("model") or ""),
        devices=devices,
        trigger_source_device=str(data.get("trigger_source_device") or ""),
        global_wake_window_start_ms=int(data.get("global_wake_window_start_ms") or 0),
        global_wake_window_end_ms=int(data.get("global_wake_window_end_ms") or 0),
        ack_output_device=str(data.get("ack_output_device") or ""),
        ack_latency_ms=int(data.get("ack_latency_ms") or 0),
        ack_error=str(data.get("ack_error") or ""),
        assistant_transcript=str(data.get("assistant_transcript") or ""),
        assistant_reply=str(data.get("assistant_reply") or ""),
        assistant_error=str(data.get("assistant_error") or ""),
    )


def trial_to_csv_row(result: TrialResult, labels: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial_id": result.trial_id,
        "position": result.position,
        "expected_device": result.expected_device,
        "selected_device": result.selected_device,
        "correct": "" if result.correct is None else int(result.correct),
        "margin": f"{result.margin:.4f}",
        "started_at": result.started_at,
        "listen_seconds": f"{result.listen_seconds:.3f}",
        "baseline_seconds": f"{result.baseline_seconds:.3f}",
        "threshold": f"{result.threshold:.3f}",
        "model": result.model,
        "trigger_source_device": result.trigger_source_device,
        "global_wake_window_start_ms": result.global_wake_window_start_ms,
        "global_wake_window_end_ms": result.global_wake_window_end_ms,
        "ack_output_device": result.ack_output_device,
        "ack_latency_ms": result.ack_latency_ms,
        "ack_error": result.ack_error,
        "assistant_transcript": result.assistant_transcript,
        "assistant_reply": result.assistant_reply,
        "assistant_error": result.assistant_error,
    }
    for label in labels:
        metrics = result.devices[label]
        row.update(
            {
                f"{label}_score": f"{metrics.score:.4f}",
                f"{label}_wake_channel": metrics.channel,
                f"{label}_proximity_channel": metrics.proximity_channel,
                f"{label}_candidate_channels": metrics.candidate_channels,
                f"{label}_best_label": metrics.best_label,
                f"{label}_best_confidence": f"{metrics.best_confidence:.4f}",
                f"{label}_triggered": int(metrics.triggered),
                f"{label}_first_trigger_ms": (
                    "" if metrics.first_trigger_ms is None else metrics.first_trigger_ms
                ),
                f"{label}_trigger_count": metrics.trigger_count,
                f"{label}_noise_rms": f"{metrics.noise_rms:.2f}",
                f"{label}_mean_rms": f"{metrics.mean_rms:.2f}",
                f"{label}_peak_rms": f"{metrics.peak_rms:.2f}",
                f"{label}_snr_db": f"{metrics.snr_db:.2f}",
                f"{label}_band_rms": f"{metrics.band_rms:.2f}",
                f"{label}_band_snr_db": f"{metrics.band_snr_db:.2f}",
                f"{label}_speech_band_ratio": f"{metrics.speech_band_ratio:.4f}",
                f"{label}_wake_window_start_ms": metrics.wake_window_start_ms,
                f"{label}_wake_window_end_ms": metrics.wake_window_end_ms,
                f"{label}_segment_duration_ms": metrics.segment_duration_ms,
                f"{label}_resolved_device": metrics.resolved_device,
                f"{label}_overflow_count": metrics.overflow_count,
                f"{label}_predict_avg_ms": f"{metrics.predict_avg_ms:.2f}",
                f"{label}_predict_max_ms": f"{metrics.predict_max_ms:.2f}",
                f"{label}_wav_path": metrics.wav_path,
                f"{label}_proximity_wav_path": metrics.proximity_wav_path,
                f"{label}_proximity_window_wav_path": metrics.proximity_window_wav_path,
            }
        )
    return row


def print_summary(summary: dict[str, Any]) -> None:
    print("\n汇总")
    overall = summary["overall"]
    print(_format_bucket("overall", overall))
    for position, bucket in summary["by_position"].items():
        print(_format_bucket(position, bucket))
    print("\n混淆矩阵 expected -> selected")
    confusion = summary["confusion"]
    if not confusion:
        print("  <无带期望设备的轮次>")
    for expected, selected_counts in confusion.items():
        items = ", ".join(
            f"{selected}:{count}"
            for selected, count in sorted(selected_counts.items())
        )
        print(f"  {expected} -> {items}")
    print("\n设备均值")
    for label, values in summary["device_averages"].items():
        print(
            f"  {label}: confidence={values['best_confidence']:.3f} "
            f"band_rms={values['band_rms']:.1f} band_snr_db={values['band_snr_db']:.1f} "
            f"score={values['score']:.3f}"
        )


def _run_summarize(args) -> int:
    results = read_trial_jsonl(args.path)
    summary = summarize_trial_results(results)
    print_summary(summary)
    return 0


def _load_openwakeword_model(model: str, inference_framework: str):
    try:
        import openwakeword  # type: ignore[import-untyped]
        from openwakeword.model import Model  # type: ignore[import-untyped]
        from openwakeword.utils import download_models  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Wake proximity collection requires openWakeWord and onnxruntime. "
            "Install with: pip install -e \".[wake,audio]\""
        ) from exc

    model_args = _resolve_openwakeword_models(
        model,
        available_models=getattr(openwakeword, "MODELS", {}).keys(),
    )
    _download_openwakeword_models(download_models, model_args)
    return Model(wakeword_models=model_args, inference_framework=inference_framework)


def _load_models_for_specs(
    specs: list[DeviceSpec],
    *,
    model: str,
    inference_framework: str,
    channels: int,
) -> dict[str, dict[int, Any]]:
    loaded: dict[str, dict[int, Any]] = {}
    for spec in specs:
        loaded[spec.label] = {
            channel: _load_openwakeword_model(model, inference_framework)
            for channel in _parse_channel_candidates(spec.wake_channel, channels)
        }
    return loaded


def _require_sounddevice():
    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Wake proximity collection requires sounddevice. "
            "Install with: pip install -e \".[audio,wake]\""
        ) from exc
    return sd


def _wait_for_ready(workers: list[_DeviceTrialWorker]) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        _raise_worker_errors(workers)
        if all(worker.ready.is_set() for worker in workers):
            return
        time.sleep(0.02)
    not_ready = [worker.spec.label for worker in workers if not worker.ready.is_set()]
    raise RuntimeError(f"Timed out opening audio streams: {', '.join(not_ready)}")


def _raise_worker_errors(workers: list[_DeviceTrialWorker]) -> None:
    for worker in workers:
        if worker.error is not None:
            raise RuntimeError(
                f"{worker.spec.label} capture failed: {worker.error}"
            ) from worker.error


def _create_run_dir(base_dir: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_run_config(
    run_dir: Path,
    args,
    specs: list[DeviceSpec],
    positions: list[PositionSpec],
    weights: ScoreWeights,
) -> None:
    data = {
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "args": vars(args),
        "devices": [asdict(spec) for spec in specs],
        "positions": [asdict(position) for position in positions],
        "weights": asdict(weights),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _print_trial_result(result: TrialResult) -> None:
    correctness = ""
    if result.correct is True:
        correctness = " ok"
    elif result.correct is False:
        correctness = " wrong"
    print(
        f"结果: selected={result.selected_device} margin={result.margin:.3f}{correctness} "
        f"trigger_source={result.trigger_source_device or 'none'} "
        f"global_window={result.global_wake_window_start_ms}-{result.global_wake_window_end_ms}ms"
    )
    if result.ack_output_device or result.ack_error:
        print(
            f"  ack: output={result.ack_output_device or 'none'} "
            f"latency={result.ack_latency_ms}ms "
            f"error={result.ack_error or 'none'}"
        )
    if result.assistant_transcript or result.assistant_reply or result.assistant_error:
        print(
            f"  assistant: transcript={result.assistant_transcript or '<empty>'} "
            f"reply={result.assistant_reply or '<empty>'} "
            f"error={result.assistant_error or 'none'}"
        )
    for label, metrics in result.devices.items():
        print(
            f"  {label}: wake_ch={metrics.channel} prox_ch={metrics.proximity_channel} "
            f"score={metrics.score:.3f} "
            f"conf={metrics.best_confidence:.3f} "
            f"band_rms={metrics.band_rms:.1f} band_snr={metrics.band_snr_db:.1f}dB "
            f"window={metrics.wake_window_start_ms}-{metrics.wake_window_end_ms}ms "
            f"trigger_ms={metrics.first_trigger_ms}"
        )


def _csv_fieldnames(labels: list[str]) -> list[str]:
    base = [
        "trial_id",
        "position",
        "expected_device",
        "selected_device",
        "correct",
        "margin",
        "started_at",
        "listen_seconds",
        "baseline_seconds",
        "threshold",
        "model",
        "trigger_source_device",
        "global_wake_window_start_ms",
        "global_wake_window_end_ms",
        "ack_output_device",
        "ack_latency_ms",
        "ack_error",
        "assistant_transcript",
        "assistant_reply",
        "assistant_error",
    ]
    per_device = [
        "score",
        "wake_channel",
        "proximity_channel",
        "candidate_channels",
        "best_label",
        "best_confidence",
        "triggered",
        "first_trigger_ms",
        "trigger_count",
        "noise_rms",
        "mean_rms",
        "peak_rms",
        "snr_db",
        "band_rms",
        "band_snr_db",
        "speech_band_ratio",
        "wake_window_start_ms",
        "wake_window_end_ms",
        "segment_duration_ms",
        "resolved_device",
        "overflow_count",
        "predict_avg_ms",
        "predict_max_ms",
        "wav_path",
        "proximity_wav_path",
        "proximity_window_wav_path",
    ]
    for label in labels:
        base.extend(f"{label}_{field}" for field in per_device)
    return base


def _resolve_jsonl_path(path: Path) -> Path:
    if path.is_dir():
        return path / "trials.jsonl"
    return path


def _group_by_position(results: list[TrialResult]) -> dict[str, list[TrialResult]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for result in results:
        grouped[result.position].append(result)
    return dict(grouped)


def _summary_bucket(results: list[TrialResult]) -> SummaryBucket:
    scored = [result for result in results if result.correct is not None]
    correct = sum(1 for result in scored if result.correct)
    accuracy = correct / len(scored) if scored else None
    return SummaryBucket(
        trials=len(results),
        scored_trials=len(scored),
        correct=correct,
        accuracy=accuracy,
        avg_margin=_mean_or_zero([result.margin for result in results]),
    )


def _confusion(results: list[TrialResult]) -> dict[str, dict[str, int]]:
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in results:
        if result.expected_device:
            confusion[result.expected_device][result.selected_device] += 1
    return {expected: dict(counts) for expected, counts in confusion.items()}


def _device_averages(results: list[TrialResult]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        for label, metrics in result.devices.items():
            values[label]["best_confidence"].append(metrics.best_confidence)
            values[label]["peak_rms"].append(metrics.peak_rms)
            values[label]["snr_db"].append(metrics.snr_db)
            values[label]["band_rms"].append(metrics.band_rms)
            values[label]["band_snr_db"].append(metrics.band_snr_db)
            values[label]["score"].append(metrics.score)
    return {
        label: {
            metric: _mean_or_zero(numbers)
            for metric, numbers in metric_values.items()
        }
        for label, metric_values in values.items()
    }


def _summary_to_dict(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": asdict(summary["overall"]),
        "by_position": {
            position: asdict(bucket)
            for position, bucket in summary["by_position"].items()
        },
        "confusion": summary["confusion"],
        "device_averages": summary["device_averages"],
    }


def _format_bucket(name: str, bucket: SummaryBucket) -> str:
    accuracy = "n/a" if bucket.accuracy is None else f"{bucket.accuracy * 100:.1f}%"
    return (
        f"  {name}: trials={bucket.trials} scored={bucket.scored_trials} "
        f"correct={bucket.correct} accuracy={accuracy} avg_margin={bucket.avg_margin:.3f}"
    )


def _mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _pcm_duration_ms(pcm: bytes, sample_rate: int) -> int:
    if sample_rate <= 0:
        return 0
    return int(len(pcm) / 2 / sample_rate * 1000)


def _slice_pcm16_mono(pcm: bytes, sample_rate: int, start_ms: int, end_ms: int) -> bytes:
    start_sample = max(0, int(sample_rate * max(0, start_ms) / 1000))
    end_sample = max(start_sample, int(sample_rate * max(0, end_ms) / 1000))
    start_byte = min(len(pcm), start_sample * 2)
    end_byte = min(len(pcm), end_sample * 2)
    return pcm[start_byte:end_byte]


def _slice_timed_chunks(
    chunks: list[_TimedPcmChunk],
    *,
    start_ms: int,
    end_ms: int,
) -> bytes:
    selected: list[bytes] = []
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms, int(end_ms))
    for chunk in chunks:
        if chunk.end_ms <= start_ms or chunk.start_ms >= end_ms:
            continue
        chunk_duration_ms = max(1, chunk.end_ms - chunk.start_ms)
        chunk_samples = len(chunk.pcm) // 2
        start_sample = 0
        end_sample = chunk_samples
        if start_ms > chunk.start_ms:
            start_sample = min(
                chunk_samples,
                int((start_ms - chunk.start_ms) * chunk_samples / chunk_duration_ms),
            )
        if end_ms < chunk.end_ms:
            end_sample = max(
                start_sample,
                int((end_ms - chunk.start_ms) * chunk_samples / chunk_duration_ms),
            )
        selected.append(chunk.pcm[start_sample * 2 : end_sample * 2])
    return b"".join(selected)


def _pcm16_peak(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    peak = 0
    for index in range(0, len(pcm) - 1, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        peak = max(peak, abs(sample))
    return float(peak)


def _speech_band_stats(
    pcm: bytes,
    sample_rate: int,
    low_hz: int = 300,
    high_hz: int = 3400,
) -> tuple[float, float, float]:
    if not pcm or sample_rate <= 0:
        return 0.0, 0.0, 0.0
    try:
        import numpy as np  # type: ignore[import-untyped]
    except ImportError:
        rms = pcm16_rms(pcm)
        return rms, rms * rms, rms * rms

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return 0.0, 0.0, 0.0
    samples = samples - float(samples.mean())
    window = np.hanning(samples.size).astype(np.float32)
    spectrum = np.fft.rfft(samples * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    full_power = float(power.mean()) if power.size else 0.0
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    band_power = float(power[mask].mean()) if mask.any() else 0.0
    return math.sqrt(max(0.0, band_power)), band_power, full_power


def _db_ratio(signal: float, noise: float) -> float:
    if signal <= 0:
        return 0.0
    return 20.0 * math.log10(signal / max(noise, 1.0))


def _safe_ratio(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return value / maximum


if __name__ == "__main__":
    raise SystemExit(main())
