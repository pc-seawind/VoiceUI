from __future__ import annotations

import argparse
import json
import sys

from voiceui.audio import list_audio_devices
from voiceui.config import config_to_dict, load_config
from voiceui.core import VoiceAssistant
from voiceui.diagnostics import calibrate_vad, record_wav


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoiceUI assistant runtime")
    parser.add_argument("--config", help="Path to YAML or JSON config file")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    parser.add_argument("--list-audio-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--record-wav", help="Record a mono WAV from the configured audio device")
    parser.add_argument(
        "--calibrate-vad",
        action="store_true",
        help="Record room noise and print RMS statistics for vad.threshold",
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds for recording commands")
    parser.add_argument(
        "--audio-purpose",
        choices=["command", "wake"],
        default="command",
        help="Use command_stream_channel or wake_stream_channel",
    )
    parser.add_argument("--audio-channel", type=int, help="Override configured audio channel")
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

        if args.calibrate_vad:
            summary = calibrate_vad(
                config,
                seconds=args.seconds,
                purpose=args.audio_purpose,
                channel_override=args.audio_channel,
            )
            print(summary.to_json())
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
