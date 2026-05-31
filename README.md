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

Run once with a config file:

```powershell
python -m voiceui --config config.example.yaml --once
```

Run continuously:

```powershell
python -m voiceui --config config.example.yaml
```

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

## Recommended MVP Setup

For the first XVF3800 prototype:

- Wake word: `openwakeword` with `hey_jarvis`.
- Endpointing: built-in energy VAD first, then switch to Silero VAD after audio
  levels are stable.
- STT: `faster_whisper` on GPU if available, otherwise CPU `int8` with a smaller
  model.
- LLM: Ollama or any OpenAI-compatible endpoint.
- TTS: console for bring-up, Piper HTTP or Piper CLI for local speech output.

## Mify / OpenAI-Compatible Backend

VoiceUI cannot use the Codex session itself as a production LLM API. For runtime
LLM and ASR, point the config at Mify, Dify, OpenAI, or another
OpenAI-compatible backend.

Example LLM config:

```yaml
llm:
  provider: mify
  endpoint: http://localhost:8000
  api_key_env: MIFY_API_KEY
  model: qwen2.5:7b-instruct
```

Example ASR config:

```yaml
stt:
  provider: mify
  endpoint: http://localhost:8000/v1/audio/transcriptions
  api_key_env: MIFY_API_KEY
  model: whisper-large-v3
  language: zh
```

Full templates are available in [config.demo.mify.yaml](config.demo.mify.yaml)
and [config.mify.example.yaml](config.mify.example.yaml). The runnable first
demo flow is documented in [docs/first-demo.md](docs/first-demo.md).

See [docs/implementation-plan.md](docs/implementation-plan.md) and
[docs/xvf3800.md](docs/xvf3800.md) for the detailed plan.
