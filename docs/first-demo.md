# First Demo Runbook

This demo proves the first usable assistant loop, then switches the same backend
to wake-word mode:

```text
Press Enter -> VAD records one utterance -> ASR -> LLM -> MiMo TTS -> speaker
Wake word -> VAD records one utterance -> ASR -> LLM -> MiMo TTS -> speaker
```

## 1. Install Minimal Demo Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[demo]"
```

For the wake-word demo:

```powershell
pip install -e ".[demo,wake]"
```

Create a local `.env` and fill in the cloud backend tokens:

```powershell
Copy-Item .env.example .env
```

Set `BAILIAN_API_KEY=...` for Bailian LLM. If you use Mify/MiMo ASR or TTS,
also set `MIFY_API_KEY=...`. This file is ignored by git and is loaded
automatically by the CLI.

## 2. Confirm Audio Devices

```powershell
python -m voiceui --list-audio-devices
```

On the first development machine, the XVF3800 appeared as:

```text
reSpeaker XVF3800 4-Mic Array
```

The demo configs pin the current machine's XVF3800 devices explicitly:
`audio.device: 24` for WASAPI capture and `playback_device: 22` for WASAPI wake
acknowledgement and TTS playback. If Windows changes the device indexes, update
those fields from `python -m voiceui --list-audio-devices`.
The XVF3800 WASAPI capture endpoint is two-channel on this machine. The demo
configs therefore use `audio.channels: 2`, `audio.wake_stream_channel: 1`, and
`audio.command_stream_channel: 0`. In wake debug logs, verify the openWakeWord
line prints `channels=2 selected_channel=1`.
The XVF3800 WASAPI output endpoint accepts `16000Hz` on this machine, so demo
TTS configs use `tts.sample_rate: 16000`. The bundled wake acknowledgement keeps
its original WAV sample rate and is resampled to the device rate at playback.

## 3. Record a Smoke Sample

```powershell
python -m voiceui --config config.demo.mock.yaml --record-wav recordings\smoke.wav --seconds 5
```

Play the file with any media player and confirm that speech is clear enough for
ASR.

The demo configs keep `audio.input_gain_db: 0.0` by default. If the recording is
too quiet, try `6.0` or `12.0` first. Use `20.0` only as an aggressive
diagnostic value because it can clip.

## 4. Calibrate VAD

The demo configs now use `engine: silero`. In this mode `vad.threshold` is a
speech probability; start with `0.6`, raise it toward `0.7` if background noise
starts turns, or lower it toward `0.5` if it misses speech.

If you switch a config back to `engine: energy`, run this while the room is
quiet:

```powershell
python -m voiceui --config config.demo.mock.yaml --calibrate-vad --seconds 10
```

Then set `vad.threshold` in the energy config to the returned
`recommended_vad_threshold`. Use at least 10 seconds for a stable value because
short smoke tests can be skewed by transient noise.

`config.demo.mock.yaml`, `config.demo.mify.yaml`, and
`config.demo.wake.yaml` all use `engine: silero`. If command endings are
clipped, raise `vad.silence_ms`; if the assistant waits too long after you stop
speaking, lower it gradually.

## 5. Verify ASR Separately

For Mify/MiMo ASR, VoiceUI uses the MiMo audio-understanding chat-completions
format. The captured WAV is sent as `input_audio` using
`data:audio/wav;base64,...`.

```powershell
python -m voiceui --config config.demo.mify.yaml --transcribe-wav recordings\smoke.wav
```

Expected output:

```text
transcript> ...
```

## 6. Verify LLM Separately

```powershell
python -m voiceui --config config.demo.mify.yaml --text "你好，介绍一下你自己"
```

Expected output:

```text
assistant> ...
```

With `config.demo.mify.yaml`, MiMo TTS also speaks the answer through the
configured output device.
The demo configs enable `llm.stream: true`, so this step should also print
`llm> first_token_ms=... latency_ms=... stream_chunks=...`. For a quick
LLM-only streaming smoke test without audio playback, temporarily set
`tts.provider` to `console` in a copied config.

## 7. Run the First Voice Demo

```powershell
python -m voiceui --config config.demo.mify.yaml
```

Flow:

1. Press Enter when prompted.
2. Speak one command.
3. Wait for `vad>`, `stt>`, and `assistant>` logs.
4. Listen for the TTS output.
5. Check `debug_sessions\<timestamp>-0001\metadata.json` and `utterance.wav`.

If ASR seems to miss the beginning, first listen to `utterance.wav`. If the WAV
itself starts late, it is a VAD boundary issue; inspect `vad_debug>` and raise
`vad.pre_roll_ms`. If the WAV is complete but the transcript starts late, it is
an ASR issue; the Aliyun demo prints `stt_debug>` and sends
`stt.leading_silence_ms: 200` before the utterance.
It also prints `audio_debug>` for stream open and first-chunk latency; if those
numbers are large, the capture stream is not ready early enough.
The local wake acknowledgement plays in the background, so VAD starts
immediately after wake detection instead of waiting for the "我在" WAV to finish.
If you say the wake word and command as one continuous phrase, the command can
still begin before wake detection returns; that needs a future rolling audio
buffer around wake detection.

With `config.demo.mify.yaml`, TTS also uses Mify/MiMo through streaming
`xiaomi/mimo-v2-tts`. If you only want local OS speech while testing LLM/ASR,
change `tts.provider` back to `system`.

For streaming TTS, VoiceUI requests `audio.format: pcm16` and plays chunks as
they arrive. If you switch to a MiMo-V2.5-TTS model, note that the current
MiMo-V2.5-TTS docs describe low-latency streaming as compatibility mode rather
than true chunked output.
LLM streaming is connected to TTS streaming: text chunks are fed into
`tts.speak_text_stream()` immediately. Aliyun NLS stream-input TTS consumes
those chunks directly; OpenAI-compatible speech and MiMo TTS fall back to
sentence-sized segment playback.

To re-test ASR with a saved turn:

```powershell
python -m voiceui --config config.demo.mify.yaml --transcribe-wav debug_sessions\<turn>\utterance.wav
python -m voiceui --config config.demo.wake.aliyun.yaml --transcribe-wav debug_sessions\<turn>\utterance.wav
```

For hardware-only smoke testing without Mify:

```powershell
python -m voiceui --config config.demo.mock.yaml
```

That confirms microphone, VAD, and speaker output but uses mock ASR/LLM text.

## 8. Verify Wake Word

The wake demo uses openWakeWord with the built-in `alexa` model. VoiceUI
loads it through ONNXRuntime on Windows.

```powershell
python -m voiceui --config config.demo.wake.yaml --wake-test
```

Say "alexa". On the first run, openWakeWord downloads its feature model and
the `alexa` model. A successful detection plays the local "我在" WAV and
prints:

```text
wake> engine=openwakeword label=alexa confidence=... latency_ms=...
```

The wake demo configs enable `wake.debug: true`, so you should also see
periodic lines like:

```text
wake_debug> elapsed_s=1.0 chunks=13 audio_ms=1040 rms=... peak=... dbfs=... near_zero_pct=... clipped_pct=... last=alexa:... best_window=alexa:... threshold=0.500 top=alexa:... predict_avg_ms=...
```

When `debug.enabled` and `debug.save_audio` are true, wake runs save
`debug_sessions\<turn>\wake.wav` with the exact wake channel passed into
openWakeWord. A full assistant turn saves both `wake.wav` and `utterance.wav`.
`--wake-monitor` saves `wake.wav` even when the wake word is not detected:

```powershell
python -m voiceui --config config.demo.wake.aliyun.yaml --wake-monitor --wake-model alexa --seconds 10
```

Interpretation:

1. `rms` and `peak` near zero means the selected input/channel is probably
   wrong, muted, or too quiet.
2. `clipped_pct` above 0 means the gain is too high.
3. `best_window` close to but below `threshold` means the wake model hears
   something similar; lower `wake.threshold` temporarily or improve placement.
4. Healthy audio with very low scores usually points to a bad wake-word/model
   match or the wrong XVF3800 output channel.

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

To regenerate the local "我在" WAV with MiMo TTS:

```powershell
python -m voiceui --config config.demo.wake.yaml --generate-wake-ack
```

If it does not trigger reliably, try these in order:

1. Confirm `audio.device`, `audio.channels`, and `audio.wake_stream_channel`.
2. Speak toward the XVF3800 at a normal smart-speaker distance.
3. If it misses real wake words, lower `wake.threshold` from `0.5` toward `0.35`.
   If it false-wakes, raise it from `0.5`.
4. Record `--audio-purpose wake` and inspect the saved WAV level.

## 9. Run the Wake-Word Full Chain

```powershell
python -m voiceui --config config.demo.wake.yaml
```

Flow:

1. Say "alexa".
2. Wait for the `wake>` log and the local "我在" acknowledgement.
3. Speak one command.
4. Wait for `vad>`, `stt>`, `llm>`, `tts>`, and the spoken answer.
5. You can interrupt a streaming TTS answer by speaking over it. VoiceUI prints
   `barge_in> speech_start`, stops playback, captures your utterance, and starts
   the next turn.
6. When `session> listening_for_follow_up seconds=10` appears, speak the next
   turn directly. The next turn keeps the same LLM message history.
7. If you stay silent for 10 seconds, VoiceUI prints
   `session> follow_up_timeout returning_to_wake` and waits for the wake word
   again.

To test local streaming TTS instead of cloud TTS, start a local
OpenAI-compatible `/v1/audio/speech` server and run:

```powershell
python -m voiceui --config config.demo.wake.local-tts.yaml
```

See [local-tts.md](local-tts.md) for the faster-qwen3-tts setup.
