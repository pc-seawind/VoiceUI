# VoiceUI Implementation Plan

## Goal

Build a voice assistant that first behaves like a single smart speaker and then
scales into a whole-home voice hub with multiple XVF3800 nodes.

## Architecture

```text
Satellite node
  XVF3800 capture, wake word, VAD, playback, local health
        |
        | WebSocket/gRPC/MQTT in a Wyoming-like event format
        v
Voice Core
  arbitration, sessions, STT, LLM, tools, TTS, permissions
        |
        +-- Home Assistant / MQTT / Matter / Zigbee2MQTT / IR
        +-- Ollama / vLLM / OpenAI-compatible LLM
        +-- faster-whisper / whisper.cpp / cloud STT
        +-- Piper / Kokoro / cloud TTS
```

## Phase 0: Hardware Bring-Up

Acceptance criteria:

- XVF3800 appears as an audio capture device.
- 16 kHz PCM recording works for at least 30 seconds.
- Speaker output is routed through the intended playback device.
- Playing TTS does not overload or clip the mic input.
- The ASR or beamformer output channel is identified and documented.

Deliverables:

- Audio device listing command.
- Repeatable recording command.
- Known-good config values for sample rate, channel count, and device name.

## Phase 1: Single-Device Voice Loop

Acceptance criteria:

- A manual-wake demo runs without a wake-word model.
- Wake word triggers reliably in a quiet room.
- VAD records a command and stops within 800-1200 ms after speech ends.
- STT transcript is passed to the LLM.
- TTS or console output returns an answer.
- A text-only mode works for debugging without hardware.

Implementation:

- `voiceui.audio`: audio capture and device listing.
- `voiceui.wake`: disabled/manual/openWakeWord adapters.
- `voiceui.vad`: energy VAD first, Silero adapter later.
- `voiceui.stt`: mock, faster-whisper, MiMo audio-understanding, and
  OpenAI-compatible multipart ASR adapters.
- `voiceui.llm`: Ollama, MiMo/Mify chat completions, OpenAI-compatible chat
  completions, and mock adapters.
- `voiceui.tts`: console, OS system TTS, MiMo/Mify TTS, Piper HTTP, and Piper
  CLI adapters.
- `voiceui.core`: conversation loop and state transitions.

## Phase 2: Smart-Speaker Behavior

Acceptance criteria:

- Follow-up conversation window works for 8-12 seconds.
- Timers, alarms, weather, and common home-control actions are routed before
  generic chat.
- Sensitive actions require confirmation.
- Logs include latency for wake, VAD, STT, LLM, TTS, and playback.

Implementation:

- Intent router before LLM fallback.
- Home Assistant REST service adapter.
- Session memory with bounded history.
- Latency spans around every pipeline stage.

## Phase 3: Low-Latency Interaction

Targets:

- Wake detection: less than 200 ms after keyword end.
- End-of-speech commit: 500-1000 ms after user stops.
- First assistant audio: less than 1.5 s for common commands.

Implementation:

- Streaming STT or chunked Whisper policy.
- Sentence-aware LLM streaming.
- Sentence-level TTS streaming and response cancellation.
- Warm model loading at process start.

## Phase 4: Barge-In

Acceptance criteria:

- User can interrupt TTS within 300 ms of speaking.
- The assistant cancels current LLM/TTS work.
- Echo from the speaker does not trigger interruption in normal playback.

Implementation:

- Keep VAD active during playback.
- Prefer XVF3800 AEC output or provide far-end reference where possible.
- Add a stop-word detector for commands like "stop".
- Mark the session as interrupted and keep the follow-up window open.

## Phase 5: Multiple XVF3800 Nodes

Acceptance criteria:

- Two nodes can hear the same wake word but only one owns the session.
- The winning node is usually the closest one.
- Response plays in the selected room.

Arbitration event:

```json
{
  "type": "wake.detected",
  "node_id": "living_room_xvf3800",
  "room": "living_room",
  "wake_confidence": 0.72,
  "vad_snr": 18.4,
  "rms": 1200,
  "doa": 35,
  "timestamp": 0
}
```

Ranking policy:

1. Reject detections below threshold.
2. Prefer higher wake confidence.
3. Prefer stronger SNR or RMS when confidence is similar.
4. Prefer earlier detection when scores are close.
5. Lock the session to one node for the follow-up window.

## Phase 6: Custom Wake Words

Short term:

- Use openWakeWord built-ins for English bring-up.
- Use sherpa-onnx KWS for Chinese or open-vocabulary wake phrase experiments.

Long term:

- Collect false accepts and false rejects in the real home.
- Train or tune wake models with room impulse response and TV/music negatives.
- Maintain separate thresholds per room because acoustics differ.

## Risks

- Speaker echo will dominate barge-in quality if playback is not in the AEC
  reference path.
- Bluetooth speakers add delay and make AEC harder.
- Wake words trained on clean synthetic data may fail in TV or kitchen noise.
- Whole-home control needs a permission model before locks, security, and high
  power devices are enabled.
