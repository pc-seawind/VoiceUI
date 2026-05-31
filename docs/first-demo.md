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

## 2. Confirm Audio Devices

```powershell
python -m voiceui --list-audio-devices
```

On the first development machine, the XVF3800 appeared as:

```text
reSpeaker XVF3800 4-Mic Array
```

If the default input is not XVF3800, copy `config.demo.mock.yaml` and set
`audio.device` to the device index or exact device name from the list.

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
$env:MIFY_API_KEY="your-token-if-required"
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

With `config.demo.mify.yaml`, TTS also uses Mify/MiMo through
`xiaomi/mimo-v2.5-tts`. If you only want local OS speech while testing LLM/ASR, change
`tts.provider` back to `system`.

Current MiMo-V2.5-TTS low-latency streaming is not available yet. VoiceUI has
an optional `tts.stream: true` path for compatibility testing, but the MiMo API
currently returns the audio after inference completes.

To re-test ASR with a saved turn:

```powershell
python -m voiceui --config config.demo.mify.yaml --transcribe-wav debug_sessions\<turn>\utterance.wav
```

For hardware-only smoke testing without Mify:

```powershell
python -m voiceui --config config.demo.mock.yaml
```

That confirms microphone, VAD, and speaker output but uses mock ASR/LLM text.

## 8. Verify Wake Word

The wake demo uses openWakeWord with the built-in `hey_jarvis` model. VoiceUI
loads it through ONNXRuntime on Windows.

```powershell
python -m voiceui --config config.demo.wake.yaml --wake-test
```

Say "hey jarvis". On the first run, openWakeWord downloads its feature model and
the `hey_jarvis` model. A successful detection plays the local "我在" WAV and
prints:

```text
wake> engine=openwakeword label=hey_jarvis confidence=... latency_ms=...
```

To regenerate the local "我在" WAV with MiMo TTS:

```powershell
$env:MIFY_API_KEY="your-token-if-required"
python -m voiceui --config config.demo.wake.yaml --generate-wake-ack
```

If it does not trigger reliably, try these in order:

1. Confirm `audio.device`, `audio.channels`, and `audio.wake_stream_channel`.
2. Speak toward the XVF3800 at a normal smart-speaker distance.
3. If it is still hard to wake, lower `wake.threshold` from `0.35` to `0.25`
   for bring-up only. If it false-wakes, raise it toward `0.5`.
4. Record `--audio-purpose wake` and inspect the saved WAV level.

## 9. Run the Wake-Word Full Chain

```powershell
$env:MIFY_API_KEY="your-token-if-required"
python -m voiceui --config config.demo.wake.yaml
```

Flow:

1. Say "hey jarvis".
2. Wait for the `wake>` log and the local "我在" acknowledgement.
3. Speak one command.
4. Wait for `vad>`, `stt>`, `llm>`, `tts>`, and the spoken answer.
5. When `session> listening_for_follow_up seconds=10` appears, speak the next
   turn directly. The next turn keeps the same LLM message history.
6. If you stay silent for 10 seconds, VoiceUI prints
   `session> follow_up_timeout returning_to_wake` and waits for the wake word
   again.
