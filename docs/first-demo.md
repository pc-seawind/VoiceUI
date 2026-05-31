# First Demo Runbook

This demo proves the first usable assistant loop without depending on a wake-word
model:

```text
Press Enter -> VAD records one utterance -> ASR -> LLM -> system speaker output
```

## 1. Install Minimal Demo Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[demo]"
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

## 4. Calibrate VAD

Run this while the room is quiet:

```powershell
python -m voiceui --config config.demo.mock.yaml --calibrate-vad --seconds 10
```

Set `vad.threshold` in the demo config to the returned
`recommended_vad_threshold`. Use at least 10 seconds for a stable value because
short smoke tests can be skewed by transient noise.

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

The `system` TTS provider also speaks the answer through the OS default output.

## 7. Run the First Voice Demo

```powershell
python -m voiceui --config config.demo.mify.yaml
```

Flow:

1. Press Enter when prompted.
2. Speak one command.
3. Wait for `vad>`, `stt>`, and `assistant>` logs.
4. Listen for the system TTS output.

With `config.demo.mify.yaml`, TTS also uses Mify/MiMo through
`mimo-v2.5-tts`. If you only want local OS speech while testing LLM/ASR, change
`tts.provider` back to `system`.

For hardware-only smoke testing without Mify:

```powershell
python -m voiceui --config config.demo.mock.yaml
```

That confirms microphone, VAD, and speaker output but uses mock ASR/LLM text.
