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
```

Start the server with a small model first:

```powershell
python examples/openai_server.py `
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base `
  --ref-audio ref_audio.wav `
  --ref-text "reference audio transcript" `
  --language Auto `
  --port 8000
```

Use a real reference WAV and matching transcript for quality. The server
registers a `default` voice when `--ref-audio` is used.

Then run VoiceUI with the local-TTS config:

```powershell
python -m voiceui --config config.demo.wake.local-tts.yaml --text "你好，介绍一下你自己"
python -m voiceui --config config.demo.wake.local-tts.yaml
```

The relevant config block is:

```yaml
tts:
  provider: openai_speech
  endpoint: http://localhost:8000
  model: tts-1
  voice: default
  audio_format: pcm
  sample_rate: 24000
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
- The 0.6B model is the first latency target. Try 1.7B only after the chain is
  stable.
