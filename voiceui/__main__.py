from __future__ import annotations

import argparse
import json
import sys
import time

from voiceui.audio import (
    RecordingAudioInput,
    create_audio_input,
    list_audio_devices,
    read_pcm16_wav,
)
from voiceui.audio_dump import AudioDumpManager, configure_audio_dump
from voiceui.config import config_to_dict, load_config
from voiceui.core import VoiceAssistant
from voiceui.debug import DebugRecorder, TurnDebugData
from voiceui.diagnostics import calibrate_vad, record_wav
from voiceui.env import load_dotenv
from voiceui.logs import configure_log_files, configure_logging, log_event, log_switch_rows
from voiceui.models import AssistantConfig, Utterance, WakeEvent
from voiceui.stt import create_stt
from voiceui.tts import synthesize_to_wav
from voiceui.wake import create_wake_detector, list_openwakeword_models
from voiceui.wake_ack import create_wake_ack_player, resolve_wake_ack_path

_DEFAULT_WAKE_ACK_STYLE = (
    "自然、清晰、亲切、短促，适合智能音箱被唤醒后的中文回应。"
    "语速稍快，不拖尾，不要夸张。"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoiceUI assistant runtime")
    parser.add_argument(
        "--config",
        help=(
            "Path to YAML or JSON config file, or 'auto' to use "
            "config.demo.wake.aliyun.yaml on Windows and "
            "config.demo.linux.wake.aliyun.yaml on Linux"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    parser.add_argument("--record-wav", help="Record a mono WAV from the configured audio device")
    parser.add_argument(
        "--transcribe-wav",
        help="Transcribe a WAV file with the configured STT backend",
    )
    parser.add_argument(
        "--calibrate-vad",
        action="store_true",
        help="Record room noise and print RMS statistics for energy vad.threshold",
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds for recording commands")
    parser.add_argument(
        "--audio-purpose",
        choices=["command", "wake"],
        default="command",
        help="Use command_stream_channel or wake_stream_channel",
    )
    parser.add_argument("--audio-channel", type=int, help="Override configured audio channel")
    parser.add_argument("--wake-test", action="store_true", help="Wait for one wake word and exit")
    parser.add_argument(
        "--wake-monitor",
        action="store_true",
        help="Print wake scores for --seconds without triggering the assistant",
    )
    parser.add_argument(
        "--list-wake-models",
        action="store_true",
        help="List available openWakeWord built-in models and exit",
    )
    parser.add_argument(
        "--list-log-switches",
        action="store_true",
        help="Print known log switch IDs and effective enabled states",
    )
    parser.add_argument("--wake-model", help="Override wake.model, e.g. any, alexa, hey_mycroft")
    parser.add_argument("--wake-threshold", type=float, help="Override wake.threshold")
    parser.add_argument(
        "--wake-debug",
        action="store_true",
        help="Print periodic wake score and audio-level diagnostics",
    )
    parser.add_argument(
        "--generate-wake-ack",
        action="store_true",
        help="Synthesize the local wake acknowledgement WAV with the configured TTS backend",
    )
    parser.add_argument("--wake-ack-text", default="我在", help="Text for --generate-wake-ack")
    parser.add_argument("--wake-ack-output", help="Output WAV path for --generate-wake-ack")
    parser.add_argument(
        "--wake-ack-style-prompt",
        default=_DEFAULT_WAKE_ACK_STYLE,
        help="Temporary TTS style prompt used by --generate-wake-ack",
    )
    parser.add_argument("--text", help="Run one text-only turn")
    parser.add_argument("--once", action="store_true", help="Run a single turn")
    args = parser.parse_args(argv)

    try:
        load_dotenv()
        if args.list_audio_devices:
            print(list_audio_devices())
            return 0
        if args.list_wake_models:
            for model_name in list_openwakeword_models():
                print(model_name)
            return 0

        config = load_config(args.config)
        if args.wake_model:
            config.wake.model = args.wake_model
        if args.wake_threshold is not None:
            config.wake.threshold = args.wake_threshold
        if args.wake_debug:
            config.wake.debug = True
            config.logging.continuous["wake.score"] = True

        if args.text is not None:
            config.input.mode = "text"
            config.wake.engine = "disabled"

        if args.wake_monitor:
            config.logging.continuous["wake.score"] = True

        configure_logging(config.logging)

        if args.list_log_switches:
            print(json.dumps(log_switch_rows(config), indent=2, ensure_ascii=True))
            return 0

        if args.dry_run:
            print(json.dumps(config_to_dict(config), indent=2, ensure_ascii=True))
            return 0

        audio_dump: AudioDumpManager | None = None
        needs_cli_debug_session = bool(
            args.generate_wake_ack
            or args.record_wav
            or args.transcribe_wav
            or args.calibrate_vad
            or args.wake_test
            or args.wake_monitor
        )
        if needs_cli_debug_session:
            audio_dump = AudioDumpManager(config.debug)
            configure_audio_dump(audio_dump)
            configure_log_files(
                debug_log_path=audio_dump.debug_log_path(),
                text_record_dir=audio_dump.text_record_dir(),
            )

        if args.generate_wake_ack:
            output_path = (
                args.wake_ack_output
                if args.wake_ack_output
                else resolve_wake_ack_path(config.wake_ack.wav_path)
            )
            original_style_prompt = config.tts.style_prompt
            config.tts.style_prompt = args.wake_ack_style_prompt or original_style_prompt
            path = synthesize_to_wav(config.tts, args.wake_ack_text, output_path)
            log_event("wake_ack", "generated", log_id="wake_ack.generated", path=path)
            return 0

        if args.record_wav:
            path = record_wav(
                config,
                output_path=args.record_wav,
                seconds=args.seconds,
                purpose=args.audio_purpose,
                channel_override=args.audio_channel,
            )
            log_event("audio", "recorded", log_id="audio.recorded", path=path)
            return 0

        if args.transcribe_wav:
            pcm, sample_rate = read_pcm16_wav(
                args.transcribe_wav,
                selected_channel=args.audio_channel or 0,
            )
            transcript = create_stt(config.stt).transcribe(
                Utterance(
                    pcm=pcm,
                    sample_rate=sample_rate,
                    duration_ms=int(len(pcm) / 2 / sample_rate * 1000),
                )
            )
            log_event(
                "stt",
                "completed",
                log_id="stt.completed",
                mode="transcribe_wav",
                text=transcript,
            )
            return 0

        if args.calibrate_vad:
            summary = calibrate_vad(
                config,
                seconds=args.seconds,
                purpose=args.audio_purpose,
                channel_override=args.audio_channel,
            )
            print(summary.to_json())
            return 0

        if args.wake_test or args.wake_monitor:
            assert audio_dump is not None
            audio_dump.start_system_input_dump(config.audio)
            channel = (
                args.audio_channel
                if args.audio_channel is not None
                else config.audio.wake_stream_channel
            )
            audio = create_audio_input(config.audio, enabled=True, selected_channel=channel)
            recording_audio = RecordingAudioInput(
                audio,
                max_seconds=(
                    args.seconds
                    if args.wake_monitor
                    else max(0.0, config.wake.debug_audio_seconds)
                ),
            )
            started = time.monotonic()
            if args.wake_monitor:
                config.wake.debug = True
                config.wake.model = args.wake_model or "any"
                config.wake.threshold = (
                    args.wake_threshold if args.wake_threshold is not None else 1.1
                )
                config.wake.max_wait_seconds = max(0.1, args.seconds)
                config.wake.debug_top_predictions = max(config.wake.debug_top_predictions, 10)
                log_event(
                    "wake",
                    "monitor_started",
                    log_id="wake.monitor_started",
                    seconds=f"{config.wake.max_wait_seconds:g}",
                    model=config.wake.model,
                    threshold=f"{config.wake.threshold:.3f}",
                )
            try:
                wake = create_wake_detector(config.wake).wait(recording_audio)
            except TimeoutError:
                if args.wake_monitor:
                    latency_ms = int((time.monotonic() - started) * 1000)
                    _save_wake_debug(
                        config,
                        _wake_event_from_recording(
                            recording_audio,
                            engine=config.wake.engine,
                            label="timeout",
                            confidence=0.0,
                        ),
                        wake_ms=latency_ms,
                        audio_dump=audio_dump,
                    )
                    log_event("wake", "monitor_done", log_id="wake.monitor_done")
                    audio_dump.stop_system_input_dump()
                    return 0
                raise
            latency_ms = int((time.monotonic() - started) * 1000)
            if not wake.pcm:
                wake = _wake_event_from_recording(
                    recording_audio,
                    engine=wake.engine,
                    label=wake.label,
                    confidence=wake.confidence,
                )
            log_event(
                "wake",
                "detected",
                log_id="wake.detected",
                engine=wake.engine,
                label=wake.label,
                confidence=f"{wake.confidence:.3f}",
                latency_ms=latency_ms,
            )
            _save_wake_debug(config, wake, wake_ms=latency_ms, audio_dump=audio_dump)
            if args.wake_monitor:
                audio_dump.stop_system_input_dump()
                return 0
            ack_started = time.monotonic()
            try:
                create_wake_ack_player(
                    config.wake_ack,
                    fallback_device=config.tts.playback_device,
                ).play()
                ack_ms = int((time.monotonic() - ack_started) * 1000)
                if ack_ms:
                    log_event(
                        "wake_ack",
                        "played",
                        log_id="wake_ack.played",
                        latency_ms=ack_ms,
                    )
            except Exception as exc:
                log_event("wake_ack", "error", log_id="wake_ack.error", error=exc)
            audio_dump.stop_system_input_dump()
            return 0

        assistant = VoiceAssistant(config)
        if args.text is not None:
            assistant.run_text_turn(args.text)
            return 0

        if args.once:
            assistant.run_once()
            return 0

        assistant.run_forever()
        return 0
    except KeyboardInterrupt:
        if "audio_dump" in locals() and audio_dump is not None:
            audio_dump.stop_system_input_dump()
        return 130
    except Exception as exc:
        if "audio_dump" in locals() and audio_dump is not None:
            audio_dump.stop_system_input_dump()
        log_event("error", "runtime", log_id="error.runtime", error=exc)
        return 2


def _wake_event_from_recording(
    audio: RecordingAudioInput,
    *,
    engine: str,
    label: str,
    confidence: float,
) -> WakeEvent:
    pcm = audio.pcm()
    return WakeEvent(
        engine=engine,
        confidence=confidence,
        label=label,
        pcm=pcm,
        sample_rate=audio.sample_rate,
        duration_ms=_pcm_duration_ms(pcm, audio.sample_rate),
    )


def _save_wake_debug(
    config: AssistantConfig,
    wake: WakeEvent,
    wake_ms: int,
    audio_dump: AudioDumpManager | None = None,
) -> None:
    debug_dir = DebugRecorder(config.debug, audio_dump=audio_dump).save_turn(
        TurnDebugData(
            node_id=config.node.id,
            room=config.node.room,
            wake={
                "engine": wake.engine,
                "label": wake.label,
                "confidence": wake.confidence,
            },
            timings_ms={"wake": wake_ms},
        ),
        wake_audio=wake,
    )
    if debug_dir:
        log_event("debug", "saved", log_id="debug.saved", path=debug_dir)


def _pcm_duration_ms(pcm: bytes, sample_rate: int) -> int:
    if sample_rate <= 0:
        return 0
    return int(len(pcm) / 2 / sample_rate * 1000)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
