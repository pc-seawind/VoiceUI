from __future__ import annotations

import argparse
import json
import sys
import time

from voiceui.audio import create_audio_input, list_audio_devices, read_pcm16_wav
from voiceui.config import config_to_dict, load_config
from voiceui.core import VoiceAssistant
from voiceui.diagnostics import calibrate_vad, record_wav
from voiceui.models import Utterance
from voiceui.stt import create_stt
from voiceui.tts import synthesize_to_wav
from voiceui.wake import create_wake_detector
from voiceui.wake_ack import create_wake_ack_player, resolve_wake_ack_path

_DEFAULT_WAKE_ACK_STYLE = (
    "自然、清晰、亲切、短促，适合智能音箱被唤醒后的中文回应。"
    "语速稍快，不拖尾，不要夸张。"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoiceUI assistant runtime")
    parser.add_argument("--config", help="Path to YAML or JSON config file")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    parser.add_argument("--list-audio-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--record-wav", help="Record a mono WAV from the configured audio device")
    parser.add_argument("--transcribe-wav", help="Transcribe a WAV file with the configured STT backend")
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
        if args.list_audio_devices:
            print(list_audio_devices())
            return 0

        config = load_config(args.config)

        if args.text is not None:
            config.input.mode = "text"
            config.wake.engine = "disabled"

        if args.dry_run:
            print(json.dumps(config_to_dict(config), indent=2, ensure_ascii=True))
            return 0

        if args.generate_wake_ack:
            output_path = (
                args.wake_ack_output
                if args.wake_ack_output
                else resolve_wake_ack_path(config.wake_ack.wav_path)
            )
            original_style_prompt = config.tts.style_prompt
            config.tts.style_prompt = args.wake_ack_style_prompt or original_style_prompt
            path = synthesize_to_wav(config.tts, args.wake_ack_text, output_path)
            print(f"wake_ack> generated={path}")
            return 0

        if args.record_wav:
            path = record_wav(
                config,
                output_path=args.record_wav,
                seconds=args.seconds,
                purpose=args.audio_purpose,
                channel_override=args.audio_channel,
            )
            print(f"recorded> {path}")
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
            print(f"transcript> {transcript}")
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

        if args.wake_test:
            channel = (
                args.audio_channel
                if args.audio_channel is not None
                else config.audio.wake_stream_channel
            )
            audio = create_audio_input(config.audio, enabled=True, selected_channel=channel)
            started = time.monotonic()
            wake = create_wake_detector(config.wake).wait(audio)
            latency_ms = int((time.monotonic() - started) * 1000)
            print(
                f"wake> engine={wake.engine} label={wake.label} "
                f"confidence={wake.confidence:.3f} latency_ms={latency_ms}"
            )
            ack_started = time.monotonic()
            try:
                create_wake_ack_player(
                    config.wake_ack,
                    fallback_device=config.tts.playback_device,
                ).play()
                ack_ms = int((time.monotonic() - ack_started) * 1000)
                if ack_ms:
                    print(f"wake_ack> latency_ms={ack_ms}")
            except Exception as exc:
                print(f"wake_ack> error={exc}")
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
        return 130
    except Exception as exc:
        print(f"error> {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
