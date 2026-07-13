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

The demo configs pin the current machine's XVF3800 devices by full WASAPI
display string, not by numeric index. The input endpoint should look like
`回音消除话筒 (reSpeaker XVF3800 4-Mic Array), Windows WASAPI (2 in, 0 out)`;
the matching output endpoint should look like
`回音消除话筒 (reSpeaker XVF3800 4-Mic Array), Windows WASAPI (0 in, 2 out)`.
If you use the second XVF3800, select the same strings with the `2-` prefix in
the endpoint name. VoiceUI resolves the configured names to the current
sounddevice indexes at runtime.
The XVF3800 WASAPI capture endpoint is two-channel on this machine. The demo
configs therefore use `audio.channels: 2`, `audio.wake_stream_channel: 0`, and
`audio.command_stream_channel: 0`. On the current XVF3800 firmware, channel 0
is the denoised/AEC-oriented stream and channel 1 is the raw/noisier stream, so
use channel 0 for wake detection and proximity scoring. In wake debug logs,
verify the openWakeWord line prints `channels=2 selected_channel=0`.
The XVF3800 WASAPI output endpoint accepts `16000Hz` on this machine, so demo
TTS configs keep `tts.sample_rate: 24000` as the source/model rate and use
`tts.playback_sample_rate: 16000` plus `tts.playback_channels: 2` for device
playback. The bundled wake acknowledgement keeps its original WAV sample rate
and is also resampled to the device rate at playback.

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
speaking, lower it gradually. `vad.trailing_silence_trim_ms` trims up to 500 ms
of confirmed trailing silence from the WAV sent to STT without changing the
silence needed to decide that speech ended.

## 5. Verify ASR Separately

For Mify/MiMo ASR, VoiceUI uses the MiMo audio-understanding chat-completions
format. The captured WAV is sent as `input_audio` using
`data:audio/wav;base64,...`.

```powershell
python -m voiceui --config config.demo.mify.yaml --transcribe-wav recordings\smoke.wav
```

Expected output:

```text
2026-06-03T19:30:01.234 | module=stt | event=completed | params=mode=transcribe_wav
    >>> ASR TEXT: ...
```

## 6. Verify LLM Separately

```powershell
python -m voiceui --config config.demo.mify.yaml --text "你好，介绍一下你自己"
```

Expected output:

```text
2026-06-03T19:30:02.234 | module=tts | event=completed | params=latency_ms=...
    >>> TTS TEXT: ...
```

With `config.demo.mify.yaml`, MiMo TTS also speaks the answer through the
configured output device.
The demo configs enable `llm.stream: true`, so this step should also print
`module=llm | event=first_token` and `module=llm | event=stream_completed`.
For a quick LLM-only streaming smoke test without audio playback, temporarily
set `tts.provider` to `console` in a copied config.

## 7. Run the First Voice Demo

```powershell
python -m voiceui --config config.demo.mify.yaml
```

Flow:

1. Press Enter when prompted.
2. Speak one command.
3. Wait for `module=vad | event=completed`, `module=stt | event=completed`,
   and `module=tts | event=completed`.
4. Listen for the TTS output.
5. Check `debug_sessions\<run>\debug.log`,
   `debug_sessions\<run>\metadata.json`, and
   `debug_sessions\<run>\audio_dumps\utterance_01_<start>_<end>.wav`.

If ASR seems to miss the beginning, first listen to
`utterance_01_<start>_<end>.wav`. If the WAV itself starts late, it is a VAD
boundary issue; inspect
`module=vad | event=debug_start` / `module=vad | event=debug_stop` and raise
`vad.pre_roll_ms`. If the WAV is complete but the transcript starts late, it is
an ASR issue. In live audio turns, the Aliyun demo logs
`module=stt | event=streaming_started` when VAD confirms speech start, sends
buffered pre-roll into NLS first, then streams command audio while VAD continues
endpointing. `--transcribe-wav` stays a full-WAV test path and sends
`stt.leading_silence_ms: 200` before the utterance. Barge-in capture uses the
same streaming ASR path and reuses that transcript when the interrupted
utterance becomes the next turn.
If logs show `module=barge_in | event=no_speech`, listen to
`debug_sessions\<run>\audio_dumps\barge_in_monitor_01_<start>_<end>.wav`;
that file is the actual command-channel audio monitored during TTS playback.
For raw multi-channel routing, use
`debug_sessions\<run>\audio_dumps\system_input_<start>_<end>.wav`.
Enable `logging.events.audio.stream_opened` and `logging.events.audio.first_chunk`
for stream open and first-chunk latency; if those numbers are large, the
capture stream is not ready early enough.
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
python -m voiceui --config config.demo.mify.yaml --transcribe-wav debug_sessions\<run>\audio_dumps\utterance_01_<start>_<end>.wav
python -m voiceui --config config.demo.wake.aliyun.yaml --transcribe-wav debug_sessions\<run>\audio_dumps\utterance_01_<start>_<end>.wav
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
python -m voiceui --config config.demo.wake.aliyun.yaml --wake-test --wake-debug
```

Say "Hi Leela" or "Hello Leela". A successful detection prints the wake label,
confidence, and latency, then immediately returns to the waiting state. The
command only prints wake logs and runtime errors, and continues until `Ctrl+C`:

```text
2026-06-03T19:30:01.234 | module=wake | event=detected | params=engine=openwakeword label=alexa confidence=... latency_ms=...
```

`module=wake | event=score` is a continuous log and stays off by default to
avoid console spam. Enable `logging.continuous.wake.score: true`, or run with
`--wake-debug` / `--wake-monitor`, when you need periodic lines like:

```text
2026-06-03T19:30:02.234 | module=wake | event=score | params=elapsed_s=1.0 chunks=13 audio_ms=1040 rms=... peak=... dbfs=... near_zero_pct=... clipped_pct=... last=alexa:... best_window=alexa:... threshold=0.500 top=alexa:... predict_avg_ms=...
```

When `debug.enabled` and `debug.save_audio` are true, wake runs save
`debug_sessions\<run>\audio_dumps\wake_01_<start>_<end>.wav` with the exact
wake channel passed into openWakeWord. A full assistant turn saves both
`wake_01_<start>_<end>.wav` and `utterance_01_<start>_<end>.wav` in the same
flat `audio_dumps\` directory.
`--wake-monitor` saves a wake dump even when the wake word is not detected:

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
2. Wait for `module=wake | event=detected` and the local "我在" acknowledgement.
3. Speak one command.
4. Wait for `module=vad | event=completed`, `module=stt | event=completed`,
   `module=llm`, `module=tts`, and the spoken answer.
5. You can interrupt a streaming TTS answer by speaking over it. VoiceUI prints
   `module=barge_in | event=speech_start`, stops playback, captures your
   utterance, and starts the next turn.
6. When `module=session | event=listening_for_follow_up` appears with
   `seconds=10`, speak the next turn directly. The next turn keeps the same LLM
   message history.
7. If you stay silent for 10 seconds, VoiceUI prints
   `module=session | event=follow_up_timeout` and waits for the wake word again.

To test local streaming TTS instead of cloud TTS, start a local
OpenAI-compatible `/v1/audio/speech` server and run:

```powershell
python -m voiceui --config config.demo.wake.local-tts.yaml
```

See [local-tts.md](local-tts.md) for the faster-qwen3-tts setup.
