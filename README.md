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

For the manual first demo, the demo extra includes Silero VAD and the audio
playback dependencies:

```powershell
pip install -e ".[demo]"
```

For wake-word testing, install the wake extra as well:

```powershell
pip install -e ".[demo,wake]"
```

Create local secrets from the template:

```powershell
Copy-Item .env.example .env
```

Then fill `BAILIAN_API_KEY` for Bailian LLM and `MIFY_API_KEY` if you use
Mify/MiMo ASR or TTS. The real `.env` is ignored by git and is loaded
automatically by the CLI.

List audio devices:

```powershell
python -m voiceui --list-audio-devices
```

The current demo configs explicitly select the reSpeaker XVF3800 WASAPI devices
instead of the system default. On this machine, the input device is index `24`
and the output device is index `22`, so configs use `audio.device: 24` and
`playback_device: 22`.
The WASAPI XVF3800 input endpoint reports two capture channels, so the demo
configs use `audio.channels: 2`, `audio.wake_stream_channel: 1`, and
`audio.command_stream_channel: 0`. If wake detection stops responding after an
audio-device change, confirm the next `wake_debug>` line prints
`channels=2 selected_channel=1`.

Record from the configured XVF3800 command stream:

```powershell
python -m voiceui --config config.example.yaml --record-wav recordings\command.wav --seconds 5
```

The demo configs keep `audio.input_gain_db: 0.0` by default. Raise it only for
input-level experiments; if `utterance.wav` sounds clipped or distorted, bring
it back down.

Transcribe a saved WAV through the configured ASR backend:

```powershell
python -m voiceui --config config.demo.mify.yaml --transcribe-wav recordings\command.wav
```

The current demo configs use Silero VAD. `vad.threshold` is a speech
probability in this mode; start with `0.6`, raise it toward `0.7` if background
noise starts turns, or lower it toward `0.5` if it misses speech. Demo configs
keep `vad.pre_roll_ms: 640` so the saved utterance includes audio before the
speech-start confirmation point.

If you switch a config back to `vad.engine: energy`, estimate an initial
RMS-based `vad.threshold` from room noise:

```powershell
python -m voiceui --config config.example.yaml --calibrate-vad --seconds 10
```

WebRTC VAD remains available as an optional engine, but it is no longer the
default for the hardware demo.

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
smoke test. Wake demo configs also enable `conversation.barge_in_enabled`, so
VoiceUI keeps VAD active while streaming TTS is playing. When speech starts, the
current playback is stopped and the captured utterance becomes the next turn.

## First Demo

Use the mock demo first to prove that the microphone, VAD, and speaker path
work. It does not call ASR or an LLM backend:

```powershell
python -m voiceui --config config.demo.mock.yaml
```

Then run the real Mify-backed demo:

```powershell
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

Say "alexa". The first run downloads the openWakeWord feature model and
`alexa` ONNX model. A successful detection prints `wake>` with the label,
confidence, and latency, then plays the local wake acknowledgement WAV.
Wake demo configs enable `wake.debug: true`, so `--wake-test` and the full
assistant loop print periodic `wake_debug>` lines with audio level, top model
scores, threshold, and inference latency. You can also force this on for any
config with `--wake-debug`.
For hardware bring-up, `config.demo.wake.yaml` uses `wake.threshold: 0.5`.
Lower it temporarily toward `0.35` if it misses real wake words, or raise it if
it false-wakes.
If input level is low, try `audio.input_gain_db: 6.0` or `12.0` first. Use
`20.0` only as an aggressive diagnostic value because it can clip.

Use the debug fields like this:

- `rms` / `peak` near zero: wrong input device, wrong channel, muted input, or
  capture level too low.
- `clipped_pct` above 0: input gain is too high and may hurt wake detection.
- `best_window` rises but stays below `threshold`: lower `wake.threshold` for
  bring-up or improve microphone placement.
- healthy audio but very low scores: the current wake word/model is not a good
  match for the utterance or the XVF3800 output channel.

If `alexa` is still not reliable enough, compare the built-in openWakeWord models:

```powershell
python -m voiceui --list-wake-models
python -m voiceui --config config.demo.wake.aliyun.yaml --wake-monitor --wake-model any --seconds 20
```

During `--wake-monitor`, try "alexa", "hey mycroft", "hey rhasspy", "timer",
and "weather". Use the model whose `top=` score is highest and most repeatable:

```powershell
python -m voiceui --config config.demo.wake.aliyun.yaml --wake-model alexa --wake-threshold 0.5
```

Then run the wake-word Mify demo:

```powershell
python -m voiceui --config config.demo.wake.yaml
```

Flow: say "alexa", speak one command after the wake log appears, then wait
for VAD, ASR, LLM, and TTS. After the answer, speak the next turn within
`conversation.follow_up_seconds` to continue the same LLM conversation without
another wake word.

The wake acknowledgement is configured separately from TTS:

```yaml
wake_ack:
  enabled: true
  wav_path: default
  playback_device: 22
```

`default` uses the bundled local `voiceui/resources/wake_ack_wo_zai.wav`
clip. Replace `wav_path` with another 16-bit PCM WAV if you want a different
phrase or voice.

Regenerate the bundled acknowledgement with the configured MiMo TTS backend:

```powershell
python -m voiceui --config config.demo.wake.yaml --generate-wake-ack
```

Use `--wake-ack-text` or `--wake-ack-output` if you want a different phrase or
file.

For lower TTS latency with a local Qwen3-TTS server, use:

```powershell
python -m voiceui --config config.demo.wake.local-tts.yaml
```

That config keeps Mify ASR/LLM but routes TTS to a local OpenAI-compatible
`/v1/audio/speech` server. Setup details are in
[docs/local-tts.md](docs/local-tts.md).

For the same wake-word, wake acknowledgement, Silero VAD, and multi-turn flow
with Aliyun ASR/TTS, use:

```powershell
python -m voiceui --config config.demo.wake.aliyun.yaml
```

Flow: say "alexa", wait for the local "我在" acknowledgement, then speak.
This config uses Aliyun NLS for ASR and TTS, keeps Mify for LLM, and keeps
`conversation.follow_up_seconds: 10` for follow-up turns without another wake
word. It also enables `conversation.barge_in_enabled: true`, so you can speak
over a TTS answer to interrupt it and start the next turn.
The local wake acknowledgement plays in the background; VAD starts immediately
after wake detection so command audio is not blocked by the "我在" WAV.

Each audio turn writes debug artifacts under `debug_sessions/` when
`debug.enabled` is true. The folder contains `utterance.wav` and
`metadata.json` with wake/VAD/STT/LLM/TTS timings, transcript, and reply.
For clipped-start issues, listen to `utterance.wav`: if the beginning is missing
there, tune VAD; if the WAV is complete but the transcript is missing the
beginning, tune ASR. The Aliyun demo also prints `stt_debug>` with the exact
audio length sent to NLS and adds `stt.leading_silence_ms: 200` before sending.
It also prints `audio_debug>` for command-stream startup latency; large
`stream_opened latency_ms` or `first_chunk read_ms` values mean the capture path
is not ready early enough.
If you speak the wake word and command as one continuous phrase, the command can
still start before the wake detector returns. That requires a future rolling
audio buffer around wake detection.

## Recommended MVP Setup

For the first XVF3800 prototype:

- Wake word: `openwakeword` with `alexa`.
- Endpointing: Silero VAD for the current hardware demo. Tune
  `vad.threshold` as a probability threshold, raise it if background noise
  triggers speech, and adjust `vad.silence_ms` if command endings are clipped
  or the assistant waits too long. If command starts are clipped, inspect
  `vad_debug>` and raise `vad.pre_roll_ms`.
- STT: `faster_whisper` on GPU if available, otherwise CPU `int8` with a smaller
  model.
- LLM: Ollama or any OpenAI-compatible endpoint.
- TTS: console for bring-up, Piper HTTP or Piper CLI for local speech output.

## Cloud Backends

VoiceUI cannot use the Codex session itself as a production LLM API. For the
current cloud path, the LLM provider is Bailian with `qwen3.6-flash`, while
Mify/MiMo ASR continues to use `xiaomi/mimo-v2.5`. For lower-latency TTS, the
demo uses `xiaomi/mimo-v2-tts` with streaming enabled.

Example LLM config:

```yaml
llm:
  provider: bailian
  endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key_env: BAILIAN_API_KEY
  model: qwen3.6-flash
  stream: true
  extra_body:
    enable_thinking: false
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
`api_key_env: MIFY_API_KEY` resolves from either the process environment or the
local `.env` file.
When `llm.stream: true` is enabled, VoiceUI requests streaming chat completions
and logs `llm> first_token_ms`, total `latency_ms`, and `stream_chunks`.
Streaming LLM output is sent directly into `tts.speak_text_stream()`. Aliyun NLS
TTS uses true stream-input synthesis, while other TTS providers speak short
sentence-sized segments as they become available.

Aliyun NLS can be used as a non-LLM ASR backend. Install its SDK dependencies
with `pip install -e ".[aliyun]"`, then put these values in `.env`:
`ALIYUN_AccessKeyId`, `ALIYUN_AccessKeySecret`, and `ALIYUN_NLS_APPKEY`.

```yaml
stt:
  provider: aliyun_nls
  endpoint: wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1
  access_key_id_env: ALIYUN_AccessKeyId
  access_key_secret_env: ALIYUN_AccessKeySecret
  app_key_env: ALIYUN_NLS_APPKEY
  timeout_seconds: 20
```

Run a quick file transcription test:

```powershell
python -m voiceui --config config.demo.aliyun-asr.yaml --transcribe-wav recordings\command.wav
```

Example TTS config:

```yaml
tts:
  provider: mify
  endpoint: https://api.xiaomimimo.com/v1
  api_key_env: MIFY_API_KEY
  model: xiaomi/mimo-v2-tts
  voice: mimo_default
  audio_format: pcm16
  sample_rate: 24000
  stream: true
```

For TTS, VoiceUI puts the text to synthesize in an `assistant` message and sends
`audio: {format, voice}`. The returned `message.audio.data` is base64-decoded
and played through the configured speaker.

When `tts.stream: true` is enabled, VoiceUI sends `stream: true` and requests
`audio.format: pcm16`, then plays base64 PCM16 chunks as they arrive. VoiceUI
logs `tts> stream_first_audio_ms`, `stream_chunks`, and
`playback_latency_ms` so the streaming bottleneck is visible. The MiMo-V2.5-TTS
series still documents low-latency streaming as not yet available, so keep the
V2 TTS model for this low-latency path unless your backend exposes a newer
streaming-capable model.

Aliyun NLS stream-input TTS is also available. It uses the same
`ALIYUN_AccessKeyId`, `ALIYUN_AccessKeySecret`, and `ALIYUN_NLS_APPKEY`
environment variables as Aliyun ASR. The stream-input TTS large-model voices are
documented by Aliyun against the Beijing gateway, so the demo uses that endpoint.

```yaml
tts:
  provider: aliyun_nls
  endpoint: wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1
  access_key_id_env: ALIYUN_AccessKeyId
  access_key_secret_env: ALIYUN_AccessKeySecret
  app_key_env: ALIYUN_NLS_APPKEY
  voice: longxiaochun
  audio_format: pcm
  sample_rate: 24000
  stream: true
```

Quick synthesis test:

```powershell
python -m voiceui --config config.demo.aliyun-tts.yaml --generate-wake-ack --wake-ack-text "你好，我在。" --wake-ack-output debug_sessions\aliyun_tts\ack.wav
```

In local tests, Chinese output was stable. Mixed English/Chinese text such as
`Second time时间。` was understandable but not accurate enough for direct
playback, so prefer sending pure Chinese assistant replies to TTS.

Full templates are available in [config.demo.mify.yaml](config.demo.mify.yaml),
[config.demo.wake.yaml](config.demo.wake.yaml),
[config.demo.aliyun-asr.yaml](config.demo.aliyun-asr.yaml),
[config.demo.aliyun-tts.yaml](config.demo.aliyun-tts.yaml),
[config.demo.wake.aliyun.yaml](config.demo.wake.aliyun.yaml), and
[config.mify.example.yaml](config.mify.example.yaml). The runnable first demo
flow is documented in [docs/first-demo.md](docs/first-demo.md).
Local streaming TTS setup is documented in [docs/local-tts.md](docs/local-tts.md).

See [docs/implementation-plan.md](docs/implementation-plan.md) and
[docs/xvf3800.md](docs/xvf3800.md) for the detailed plan.
