from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voiceui.config import load_config
from voiceui.core import VoiceAssistant
from voiceui.env import load_dotenv
from voiceui.logs import configure_log_files, configure_logging, log_event
from voiceui.models import AssistantConfig

_DEFAULT_SERVICE_CONFIG = "config.demo.wake.aliyun.yaml"
_SERVICE_STDOUT_MODE = "errors_and_voice_context"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoiceUI long-running production service")
    parser.add_argument(
        "--config",
        default=_DEFAULT_SERVICE_CONFIG,
        help=f"Config file for the production service. Default: {_DEFAULT_SERVICE_CONFIG}",
    )
    parser.add_argument(
        "--output-dir",
        help="Override debug/log output directory for this service run",
    )
    parser.add_argument(
        "--audio-dump",
        action="store_true",
        help="Enable audio dumps. Disabled by default for production service runs.",
    )
    args = parser.parse_args(argv)

    assistant: VoiceAssistant | None = None
    try:
        load_dotenv()
        config = load_config(args.config)
        _prepare_service_config(
            config,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            audio_dump=args.audio_dump,
        )
        configure_logging(config.logging)
        configure_log_files(stdout_mode=_SERVICE_STDOUT_MODE)

        assistant = VoiceAssistant(config)
        log_event(
            "service",
            "started",
            log_id="service.started",
            config=args.config,
            output_dir=config.debug.output_dir,
            audio_dump=args.audio_dump,
        )
        assistant.run_forever()
        return 0
    except KeyboardInterrupt:
        if assistant is not None:
            _restore_service_log_files(assistant)
        log_event("service", "stopped", log_id="service.stopped", reason="keyboard_interrupt")
        return 130
    except Exception as exc:
        if assistant is not None:
            _restore_service_log_files(assistant)
        log_event("service", "error", log_id="service.error", error=exc)
        return 2


def _prepare_service_config(
    config: AssistantConfig,
    *,
    output_dir: Path | None,
    audio_dump: bool,
) -> None:
    config.input.mode = "audio"
    config.debug.enabled = True
    if output_dir is not None:
        config.debug.output_dir = str(output_dir)
    config.debug.save_audio = bool(audio_dump)
    config.debug.save_metadata = True
    config.debug.system_input_dump_enabled = bool(audio_dump)
    config.debug.voice_path_dump_enabled = bool(audio_dump)


def _restore_service_log_files(assistant: VoiceAssistant) -> None:
    configure_log_files(
        debug_log_path=assistant.audio_dump.debug_log_path(),
        text_record_dir=assistant.audio_dump.text_record_dir(),
        stdout_mode=_SERVICE_STDOUT_MODE,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
