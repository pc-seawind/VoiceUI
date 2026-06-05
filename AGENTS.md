# Project Agent Instructions

These instructions are project-level rules for future coding agents working in
this repository.

## Runtime Logging

- Runtime logs must use `voiceui.logs.log_event()` for event-style logs and
  `voiceui.logs.log_continuous()` for loop/repeating logs.
- Runtime logs must not use `print()`. `print()` is allowed only for explicit
  command output such as device lists, JSON config dumps, interactive prompts,
  or final summary tables.
- Every recurring or user-facing runtime log must have a `LogSpec` entry in
  `voiceui/logs.py`, so it appears in `--list-log-switches` and can be toggled
  by config.
- The log format is fixed:
  `timestamp | module=<module> | event=<event> | params=<key=value ...>`.
  Timestamps are emitted to milliseconds.
- ASR/STT and TTS text must be passed as `text=...` to `log_event()`, so
  `voiceui.logs` can render it on the highlighted next line and write text
  records.
- Continuous logs, such as wake scores and limiters, must default to off unless
  they are explicitly enabled by config or CLI flags.
- Production service mode uses `stdout_mode="errors_and_voice_context"`:
  non-error runtime logs go to `debug.log`, error logs still go to stdout, and
  stdout also prints concise voice context lines.

## Debug And Audio Dumps

- Each program run uses one timestamped debug session directory:
  `debug_sessions/<run>/`.
- Runtime debug logs go to `debug_sessions/<run>/debug.log`.
- Per-run metadata is appended into one `debug_sessions/<run>/metadata.json`.
  Do not create one metadata file per turn.
- Audio dump files go directly under `debug_sessions/<run>/audio_dumps/`.
  Do not add nested dump folders unless a new requirement explicitly asks for
  them.
- Long-running raw system input dumps do not include turn numbers:
  `system_input_hh.mm.ss.mmm_hh.mm.ss.mmm.wav`.
- Voice-path dumps include the short turn number:
  `wake_01_hh.mm.ss.mmm_hh.mm.ss.mmm.wav`,
  `utterance_01_hh.mm.ss.mmm_hh.mm.ss.mmm.wav`,
  `barge_in_monitor_01_hh.mm.ss.mmm_hh.mm.ss.mmm.wav`,
  `tts_output_01_hh.mm.ss.mmm_hh.mm.ss.mmm.wav`.
- Dump times are relative to the long-running system input origin and use
  `hh.mm.ss.mmm`, not raw millisecond-only names.

## Verification

- When changing logging, run focused tests around `tests/test_logs.py` and any
  touched module tests.
- When changing production service behavior, run focused tests around
  `tests/test_service.py`.
- When changing debug/audio dumps, run focused tests around
  `tests/test_audio_dump.py`, `tests/test_debug.py`, and relevant flow tests.
- Before finishing substantial changes, run `python -m pytest` and
  `python -m ruff check .` when feasible.

## Git Hygiene

- Commit promptly after finishing a coherent module or feature slice.
- Prefer small, focused commits over leaving a large dirty working tree.
- Do not avoid commits out of fear of mistakes; follow-up fixes and reverts are
  acceptable when needed.
- Avoid ending work with uncommitted changes unless the user explicitly asks to
  pause before committing or the remaining changes are known to belong to
  someone else.
