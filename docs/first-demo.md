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

Re-run that install after pulling updates; the demo now uses `webrtcvad-wheels`
for endpointing.

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

## 4. Calibrate VAD

Run this while the room is quiet:

```powershell
python -m voiceui --config config.demo.mock.yaml --calibrate-vad --seconds 10
```

Set `vad.threshold` in the demo config to the returned
`recommended_vad_threshold`. Use at least 10 seconds for a stable value because
short smoke tests can be skewed by transient noise.

For `config.demo.mify.yaml` and `config.demo.wake.yaml`, VAD is now
`engine: webrtc`. WebRTC ignores `vad.threshold`; if it still clips endings,
raise `vad.silence_ms` first. If it misses speech, try lowering
`vad.webrtc_mode` from `2` to `1`.

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

If it does not trigger reliably, try these in order:

1. Confirm `audio.device`, `audio.channels`, and `audio.wake_stream_channel`.
2. Speak toward the XVF3800 at a normal smart-speaker distance.
3. Lower `wake.threshold` from `0.5` to `0.35` for bring-up only.
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
