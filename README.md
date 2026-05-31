# VoiceUI

VoiceUI is a modular voice assistant core for an XVF3800 microphone array, a
speaker, local or cloud speech models, and future whole-home voice control.

The first target is a single smart-speaker loop:

```text
Wake word -> VAD endpointing -> STT -> LLM or home intent -> TTS -> speaker
```

The project is intentionally adapter-based. You can run it without hardware in
text mode, then enable audio, wake word, local STT, and TTS one component at a
time.

## Quick Start

Validate the default runtime without optional dependencies:

```powershell
python -m voiceui --dry-run
python -m voiceui --text "What can you do?"
```

Install optional packages for the local audio stack:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[config,audio,wake,stt,tts]"
```

For the manual first demo, the smaller dependency set is enough:

```powershell
pip install -e ".[demo]"
```

For wake-word testing, install the wake extra as well:

```powershell
pip install -e ".[demo,wake]"
```

The demo extra includes `webrtcvad-wheels`, which is used by the current
Mify/wake configs for endpointing.

List audio devices:

```powershell
python -m voiceui --list-audio-devices
```

Record from the configured XVF3800 command stream:

```powershell
python -m voiceui --config config.example.yaml --record-wav recordings\command.wav --seconds 5
```

Transcribe a saved WAV through the configured ASR backend:

```powershell
python -m voiceui --config config.demo.mify.yaml --transcribe-wav recordings\command.wav
```

Estimate an initial `vad.threshold` from room noise:

```powershell
python -m voiceui --config config.example.yaml --calibrate-vad --seconds 10
```

This calibration is only useful for `vad.engine: energy`. WebRTC VAD ignores
`vad.threshold`; tune `vad.webrtc_mode`, `vad.silence_ms`, and
`vad.min_speech_ms` instead.

Run once with a config file:

```powershell
python -m voiceui --config config.example.yaml --once
```

Run continuously:

```powershell
python -m voiceui --config config.example.yaml
```

In continuous audio mode, one wake word starts a conversation session. After
each answer, VoiceUI listens for a follow-up for `conversation.follow_up_seconds`
without requiring another wake word. If no speech starts before the timeout, it
returns to wake-word listening. `--once` is intentionally still a single-turn
smoke test.

## First Demo

Use the mock demo first to prove that the microphone, VAD, and speaker path
work. It does not call ASR or an LLM backend:

```powershell
python -m voiceui --config config.demo.mock.yaml
```

Then run the real Mify-backed demo:

```powershell
$env:MIFY_API_KEY="your-token-if-required"
python -m voiceui --config config.demo.mify.yaml --text "你好，介绍一下你自己"
python -m voiceui --config config.demo.mify.yaml --transcribe-wav recordings\command.wav
python -m voiceui --config config.demo.mify.yaml
```

For the audio demo, press Enter when prompted, speak one command, and wait for
the transcript plus assistant reply.

After the manual demo works, verify wake detection with openWakeWord:

```powershell
python -m voiceui --config config.demo.wake.yaml --wake-test
```

Say "hey jarvis". The first run downloads the openWakeWord feature model and
`hey_jarvis` ONNX model. A successful detection prints `wake>` with the label,
confidence, and latency, then plays the local wake acknowledgement WAV.

Then run the wake-word Mify demo:

```powershell
python -m voiceui --config config.demo.wake.yaml
```

Flow: say "hey jarvis", speak one command after the wake log appears, then wait
for VAD, ASR, LLM, and TTS. After the answer, speak the next turn within
`conversation.follow_up_seconds` to continue the same LLM conversation without
another wake word.

The wake acknowledgement is configured separately from TTS:

```yaml
wake_ack:
  enabled: true
  wav_path: default
  playback_device: default
```

`default` uses the bundled local `voiceui/resources/wake_ack_wo_zai.wav`
clip. Replace `wav_path` with another 16-bit PCM WAV if you want a different
phrase or voice.

Regenerate the bundled acknowledgement with the configured MiMo TTS backend:

```powershell
$env:MIFY_API_KEY="your-token-if-required"
python -m voiceui --config config.demo.wake.yaml --generate-wake-ack
```

Use `--wake-ack-text` or `--wake-ack-output` if you want a different phrase or
file.

Each audio turn writes debug artifacts under `debug_sessions/` when
`debug.enabled` is true. The folder contains `utterance.wav` and
`metadata.json` with wake/VAD/STT/LLM/TTS timings, transcript, and reply.

## Recommended MVP Setup

For the first XVF3800 prototype:

- Wake word: `openwakeword` with `hey_jarvis`.
- Endpointing: WebRTC VAD for the current hardware demo. The default
  `vad.silence_ms` is tuned for lower latency; raise it toward `900` if endings
  are clipped, or lower it toward `500` if the assistant still waits too long.
- STT: `faster_whisper` on GPU if available, otherwise CPU `int8` with a smaller
  model.
- LLM: Ollama or any OpenAI-compatible endpoint.
- TTS: console for bring-up, Piper HTTP or Piper CLI for local speech output.

## Mify / MiMo Backend

VoiceUI cannot use the Codex session itself as a production LLM API. For the
Mify/MiMo path, LLM and ASR use `xiaomi/mimo-v2.5`, while TTS uses
`xiaomi/mimo-v2.5-tts`.

Example LLM config:

```yaml
llm:
  provider: mify
  endpoint: https://api.xiaomimimo.com/v1
  api_key_env: MIFY_API_KEY
  model: xiaomi/mimo-v2.5
```

Example ASR config:

```yaml
stt:
  provider: mify
  endpoint: https://api.xiaomimimo.com/v1
  api_key_env: MIFY_API_KEY
  model: xiaomi/mimo-v2.5
  language: zh
```

For ASR, VoiceUI sends the captured WAV as `input_audio` with a
`data:audio/wav;base64,...` payload and asks MiMo to output only the transcript.
The MiMo-compatible path uses an `api-key` header populated from `api_key_env`.
Replace `endpoint` with your Mify MiMo-compatible base URL when you have it.

Example TTS config:

```yaml
tts:
  provider: mify
  endpoint: https://api.xiaomimimo.com/v1
  api_key_env: MIFY_API_KEY
  model: xiaomi/mimo-v2.5-tts
  voice: mimo_default
  audio_format: pcm
  sample_rate: 24000
  stream: false
```

For TTS, VoiceUI puts the text to synthesize in an `assistant` message and sends
`audio: {format, voice}`. The returned `message.audio.data` is base64-decoded
and played through the configured speaker.

MiMo-V2.5-TTS currently does not provide true low-latency streaming output. If
`tts.stream: true` is enabled, VoiceUI sends `stream: true` and can play PCM
chunks as they arrive, but the current MiMo API documents this as a
compatibility mode that returns once after inference completes. VoiceUI logs
`tts> synth_latency_ms`, `tts> playback_latency_ms`, and when streaming is
enabled, `tts> stream_first_audio_ms` so the bottleneck is visible.

Full templates are available in [config.demo.mify.yaml](config.demo.mify.yaml)
[config.demo.wake.yaml](config.demo.wake.yaml), and
[config.mify.example.yaml](config.mify.example.yaml). The runnable first demo
flow is documented in [docs/first-demo.md](docs/first-demo.md).

See [docs/implementation-plan.md](docs/implementation-plan.md) and
[docs/xvf3800.md](docs/xvf3800.md) for the detailed plan.
