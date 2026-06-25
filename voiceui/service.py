from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voiceui.config import AUTO_CONFIG, load_config
from voiceui.core import VoiceAssistant
from voiceui.env import load_dotenv
from voiceui.logs import configure_log_files, configure_logging, log_event
from voiceui.models import AssistantConfig
from voiceui.web import VoiceUiWebConsole, start_web_console

_DEFAULT_SERVICE_CONFIG = AUTO_CONFIG
_SERVICE_STDOUT_MODE = "errors_and_voice_context"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoiceUI long-running production service")
    parser.add_argument(
        "--config",
        default=_DEFAULT_SERVICE_CONFIG,
        help=(
            "Config file for the production service, or 'auto' to use "
            "config.demo.wake.aliyun.yaml on Windows and "
            "config.demo.linux.wake.aliyun.yaml on Linux. "
            f"Default: {_DEFAULT_SERVICE_CONFIG}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Override debug/log output directory for this service run",
    )
    parser.add_argument(
        "--audio-dump",
        action="store_true",
        help=(
            "Enable all audio dumps, including the long-running raw system input dump. "
            "Disabled by default for production service runs."
        ),
    )
    parser.add_argument(
        "--voice-path-dump",
        action="store_true",
        help=(
            "Enable per-turn voice-path dumps without opening an extra continuous "
            "input stream. Useful on devices that cannot open the microphone twice."
        ),
    )
    parser.add_argument(
        "--system-input-dump",
        action="store_true",
        help=(
            "Enable only the long-running raw system input dump. This may require "
            "the input device to support multiple simultaneous readers."
        ),
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Enable the VoiceUI web console for logs, debug sessions, and text input.",
    )
    parser.add_argument("--web-host", help="Override web console bind host.")
    parser.add_argument("--web-port", type=int, help="Override web console bind port.")
    args = parser.parse_args(argv)

    assistant: VoiceAssistant | None = None
    web_console: VoiceUiWebConsole | None = None
    try:
        load_dotenv()
        config = load_config(args.config)
        _prepare_service_config(
            config,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            audio_dump=args.audio_dump,
            voice_path_dump=args.voice_path_dump,
            system_input_dump=args.system_input_dump,
        )
        if args.web:
            config.web.enabled = True
        if args.web_host:
            config.web.host = args.web_host
        if args.web_port is not None:
            config.web.port = args.web_port
        configure_logging(config.logging)
        configure_log_files(stdout_mode=_SERVICE_STDOUT_MODE)

        assistant = VoiceAssistant(config)
        if config.web.enabled:
            web_console = start_web_console(
                assistant,
                host=config.web.host,
                port=config.web.port,
                title="VoiceUI Service",
            )
        log_event(
            "service",
            "started",
            log_id="service.started",
            config=args.config,
            output_dir=config.debug.output_dir,
            audio_dump=args.audio_dump,
            voice_path_dump=config.debug.voice_path_dump_enabled,
            system_input_dump=config.debug.system_input_dump_enabled,
            web=bool(web_console),
            web_url=web_console.url if web_console is not None else "",
        )
        try:
            assistant.run_forever()
        finally:
            if web_console is not None:
                web_console.stop()
        return 0
    except KeyboardInterrupt:
        if web_console is not None:
            web_console.stop()
        if assistant is not None:
            _restore_service_log_files(assistant)
        log_event("service", "stopped", log_id="service.stopped", reason="keyboard_interrupt")
        return 130
    except Exception as exc:
        if web_console is not None:
            web_console.stop()
        if assistant is not None:
            _restore_service_log_files(assistant)
        log_event("service", "error", log_id="service.error", error=exc)
        return 2


def _prepare_service_config(
    config: AssistantConfig,
    *,
    output_dir: Path | None,
    audio_dump: bool,
    voice_path_dump: bool = False,
    system_input_dump: bool = False,
) -> None:
    config.input.mode = "audio"
    config.debug.enabled = True
    if output_dir is not None:
        config.debug.output_dir = str(output_dir)
    enable_voice_path_dump = bool(audio_dump or voice_path_dump)
    enable_system_input_dump = bool(audio_dump or system_input_dump)
    config.debug.save_audio = bool(enable_voice_path_dump or enable_system_input_dump)
    config.debug.save_metadata = True
    # Long-running service mode uses one timestamped debug session per
    # service process. Startup, idle logs, turns, metadata, and audio dumps all
    # stay under that single run directory for easier remote inspection.
    config.debug.session_scope = "run"
    config.debug.system_input_dump_enabled = enable_system_input_dump
    config.debug.voice_path_dump_enabled = enable_voice_path_dump


def _restore_service_log_files(assistant: VoiceAssistant) -> None:
    configure_log_files(
        debug_log_path=assistant.audio_dump.debug_log_path(),
        text_record_dir=assistant.audio_dump.text_record_dir(),
        stdout_mode=_SERVICE_STDOUT_MODE,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
