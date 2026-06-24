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
