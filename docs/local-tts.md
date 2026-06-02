# Local Streaming TTS

VoiceUI can use a local OpenAI-compatible `/v1/audio/speech` server for lower
TTS latency. This keeps the TTS model hot in a separate process and lets
VoiceUI stream raw PCM chunks directly to the speaker.

## Faster Qwen3-TTS

`andimarafioti/faster-qwen3-tts` provides a local OpenAI-compatible speech
server. It requires Python 3.10+, PyTorch 2.5.1+, and an NVIDIA GPU with CUDA
for real-time performance.

Set it up outside this repo:

```powershell
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
setup_windows.bat
git apply F:\Workspace\Projects\VocieUI\docs\faster-qwen3-tts-openai-server-xvec.patch
```

Start the server with a small model first. For the Chinese VoiceUI demo, use a
voice config instead of `--ref-text` command-line quoting:

```powershell
python examples/openai_server.py `
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base `
  --voices F:\Workspace\Projects\VocieUI\docs\faster-qwen3-tts-voices.zh.json `
  --host 127.0.0.1 `
  --port 8000
```

The provided voice config uses the bundled `ref_audio.wav` and its matching
transcript from `faster-qwen3-tts`, sets `language: Chinese`, and enables
`xvec_only: true` for cleaner cross-language output. It also sets
`max_audio_seconds: 8` so an occasional bad generation cannot hold the model
lock indefinitely. Without those settings, short Chinese replies can turn into
long or unintelligible audio.

The stock `examples/openai_server.py` may need the local
`faster-qwen3-tts-openai-server-xvec.patch` patch that passes
`voice_cfg["xvec_only"]` into `generate_voice_clone_streaming()` and
`generate_voice_clone()`. The local checkout used for this demo has that patch.

Then run VoiceUI with the local-TTS config:

```powershell
python -m voiceui --config config.demo.wake.local-tts.yaml --text "你好，介绍一下你自己"
python -m voiceui --config config.demo.wake.local-tts.yaml
```

The relevant config block is:

```yaml
tts:
  provider: openai_speech
  endpoint: http://127.0.0.1:8000
  model: tts-1
  voice: default
  audio_format: pcm
  sample_rate: 16000
  stream: true
```

For `openai_speech`, `stream: true` requests `response_format: pcm`, then
VoiceUI plays the HTTP response body as PCM16 chunks. The log line
`tts> stream_first_audio_ms=... stream_chunks=... playback_latency_ms=...`
is directly comparable with the MiMo streaming path.

## Notes

- `pcm` is preferred for streaming because it avoids WAV header handling and
  starts playback from the first audio bytes.
- Keep the local TTS server running between VoiceUI turns; cold model startup is
  much slower than per-turn TTFA.
- The first request also captures CUDA graphs, so it can take several seconds.
  Hot requests on the RTX 5090 test machine returned first audio in about
  270-320 ms for short Chinese replies with `xvec_only: true`.
- Closed-loop check on the generated audio:
  `好的，我在。有什么可以帮你？` was transcribed by MiMo ASR as
  `好的，我在。有什么可以帮您。`
- The 0.6B model is the first latency target. Try 1.7B only after the chain is
  stable.
