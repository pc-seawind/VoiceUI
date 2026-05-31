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

List audio devices:

```powershell
python -m voiceui --list-audio-devices
```

Run once with a config file:

```powershell
python -m voiceui --config config.example.yaml --once
```

Run continuously:

```powershell
python -m voiceui --config config.example.yaml
```

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

See [docs/implementation-plan.md](docs/implementation-plan.md) and
[docs/xvf3800.md](docs/xvf3800.md) for the detailed plan.
