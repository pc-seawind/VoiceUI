# VoiceUI Web Console

VoiceUI now has a small built-in web console for remote operation:

- live `debug.log` viewing
- conversation history from `debug_sessions/text_records/*.jsonl`
- debug session / metadata / audio dump browsing
- text chat input as a second input source alongside voice

## Production service

The Linux wake demo config enables the console on `0.0.0.0:8765`:

```bash
voiceui-service --config auto --web
```

Or override the bind address explicitly:

```bash
voiceui-service --config auto --web --web-host 0.0.0.0 --web-port 8765
```

Open `http://<host>:8765/` from the remote machine.

Production service audio dumps are off unless explicitly requested. To collect
per-turn wake / utterance / TTS dumps without opening a second microphone input
stream, use:

```bash
voiceui-service --config auto --voice-path-dump
```

Use `--system-input-dump` only when the audio device supports a second
continuous reader; `--audio-dump` enables both dump modes.

On the home deployment, the existing `vhome.emox.space` Cloudflare tunnel
also exposes the console under:

```text
https://vhome.emox.space/voiceui/
```

The `/voiceui/` path is proxied by the local vhome static server to
`127.0.0.1:8765`. The browser API endpoints intentionally use the
`_rpc/...` relative path alias so they do not collide with vhome's existing
`/api/*` route for Visual Memory.

The home deployment also mounts the NAS `Homespace/VoiceUI` share at
`~/.voiceui-nas` and starts the service with:

```bash
voiceui-service --config auto --output-dir ~/.voiceui-nas/debug_sessions --voice-path-dump
```

Production service mode uses one run-scoped timestamped debug session per service
process. On service startup, VoiceUI creates `<output-dir>/<run>/`; startup/idle
logs, wake/text-turn logs, `metadata.json`, and `audio_dumps/*.wav` all stay
directly under that run directory for the lifetime of the process.
`text_records/*.jsonl` stays directly under `<output-dir>/text_records/`.

## Standalone viewer / text assistant

Start a text-mode assistant with the web console:

```bash
voiceui-web --config auto --host 0.0.0.0 --port 8765
```

View existing logs only, without attaching input to a running assistant:

```bash
voiceui-web --viewer-only --output-dir debug_sessions --host 0.0.0.0 --port 8765
```

## Homespace artifact

`homespace.yaml` exports `voiceui-web-console`, so Homespace can show/start this page from the VoiceUI project artifact list.
