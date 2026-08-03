from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LeelaMhaOnnxRuntime:
    """Run the packaged two-label MFCC MHA model with ONNX Runtime."""

    backend = "onnx_int8"

    def __init__(self, model: str | Path) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "WeKWS MHA wake detection requires numpy and onnxruntime. "
                'Install with: pip install -e ".[wake]"'
            ) from exc

        self.model_path = Path(model).expanduser()
        self.deployment_path = self.model_path.with_name("deployment.json")
        for path in (self.model_path, self.deployment_path):
            if not path.is_file():
                raise FileNotFoundError(f"WeKWS MHA runtime file does not exist: {path}")
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(f"WeKWS MHA runtime requires an ONNX model: {self.model_path}")

        deployment = json.loads(self.deployment_path.read_text(encoding="utf-8"))
        if not isinstance(deployment, dict):
            raise TypeError(f"WeKWS MHA deployment must be a mapping: {self.deployment_path}")
        self._validate_deployment_model(deployment)

        self.sample_rate = int(deployment["sample_rate"])
        self.pcm_samples = int(deployment["pcm_samples"])
        self.input_shape = tuple(int(value) for value in deployment["input_shape"])
        self.output_shape = tuple(int(value) for value in deployment["output_shape"])
        self.labels = [str(label) for label in deployment["labels"]]
        self.thresholds = {
            str(label): float(threshold)
            for label, threshold in dict(deployment.get("thresholds", {})).items()
        }
        if self.input_shape != (1, 198, 80):
            raise ValueError(f"Unsupported WeKWS MHA input shape: {self.input_shape}")
        if self.output_shape != (1, 198, len(self.labels)):
            raise ValueError(f"Invalid WeKWS MHA output shape: {self.output_shape}")
        if float(deployment.get("runtime_dither", 0.0)) != 0.0:
            raise ValueError("WeKWS MHA NumPy frontend requires runtime_dither=0.0")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._np = np
        self._mfcc = _KaldiMfccFrontend(np, sample_rate=self.sample_rate)
        self._session.run(
            [self._output_name],
            {self._input_name: np.zeros(self.input_shape, dtype=np.float32)},
        )

    def score_pcm(self, pcm: bytes, sample_rate: int) -> dict[str, float]:
        if sample_rate != self.sample_rate:
            raise RuntimeError(
                f"Leela MHA requires {self.sample_rate} Hz audio, got sample_rate={sample_rate}"
            )
        samples = self._np.frombuffer(pcm, dtype="<i2")
        if samples.size != self.pcm_samples:
            raise ValueError(
                "Leela MHA requires "
                f"{self.pcm_samples} PCM samples, got {samples.size}"
            )
        features = self._mfcc(samples)
        if features.shape != self.input_shape[1:]:
            raise RuntimeError(
                f"Leela MHA MFCC shape mismatch: expected {self.input_shape[1:]}, "
                f"got {features.shape}"
            )
        scores = self._session.run(
            [self._output_name],
            {self._input_name: features[self._np.newaxis, ...]},
        )[0]
        if tuple(scores.shape) != self.output_shape:
            raise RuntimeError(
                f"Leela MHA output shape mismatch: expected {self.output_shape}, "
                f"got {tuple(scores.shape)}"
            )
        peak_scores = scores[0].max(axis=0)
        return {
            label: float(score)
            for label, score in zip(self.labels, peak_scores, strict=True)
        }

    def _validate_deployment_model(self, deployment: dict[str, object]) -> None:
        expected_name = str(deployment.get("model") or "")
        if expected_name and self.model_path.name != expected_name:
            raise ValueError(
                f"WeKWS deployment selects {expected_name}, got {self.model_path.name}"
            )
        expected_sha256 = str(deployment.get("model_sha256") or "").lower()
        if not expected_sha256:
            return
        actual_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"WeKWS model SHA256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )


class _KaldiMfccFrontend:
    """NumPy implementation of the fixed Kaldi MFCC frontend used for export."""

    def __init__(self, np, *, sample_rate: int) -> None:
        self._np = np
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * 0.025)
        self.window_shift = int(sample_rate * 0.010)
        self.fft_size = 1 << (self.window_size - 1).bit_length()
        self.num_mel_bins = 80
        self.num_ceps = 80
        self.window = self._create_povey_window()
        self.mel_banks = self._create_mel_banks()
        self.dct = self._create_dct()
        indices = np.arange(self.num_ceps, dtype=np.float32)
        self.lifter = (
            np.float32(1.0)
            + np.float32(11.0)
            * np.sin(np.float32(np.pi) * indices / np.float32(22.0))
        ).astype(np.float32)

    def __call__(self, samples):
        np = self._np
        waveform = np.asarray(samples, dtype=np.float32)
        if waveform.size < self.window_size:
            return np.empty((0, self.num_ceps), dtype=np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(
            waveform,
            self.window_size,
        )[:: self.window_shift].copy()
        frames -= frames.mean(axis=1, keepdims=True, dtype=np.float32)
        previous = np.empty_like(frames)
        previous[:, 0] = frames[:, 0]
        previous[:, 1:] = frames[:, :-1]
        frames -= np.float32(0.97) * previous
        frames *= self.window

        spectrum = np.abs(
            np.fft.rfft(frames, n=self.fft_size, axis=1)
        ).astype(np.float32)
        spectrum = np.square(spectrum, dtype=np.float32)
        mel_energies = spectrum @ self.mel_banks.T
        mel_energies = np.log(
            np.maximum(mel_energies, np.finfo(np.float32).eps)
        ).astype(np.float32)
        return ((mel_energies @ self.dct) * self.lifter).astype(np.float32)

    def _create_povey_window(self):
        np = self._np
        indices = np.arange(self.window_size, dtype=np.float32)
        phase = np.float32(2.0 * np.pi) * indices / np.float32(self.window_size - 1)
        hann = np.float32(0.5) - np.float32(0.5) * np.cos(phase)
        return np.power(hann, np.float32(0.85)).astype(np.float32)

    def _create_mel_banks(self):
        np = self._np
        num_fft_bins = self.fft_size // 2
        nyquist = self.sample_rate / 2.0
        mel_low = np.float32(1127.0 * np.log(1.0 + 20.0 / 700.0))
        mel_high = np.float32(1127.0 * np.log(1.0 + nyquist / 700.0))
        mel_delta = np.float32(
            (mel_high - mel_low) / np.float32(self.num_mel_bins + 1)
        )
        bins = np.arange(self.num_mel_bins, dtype=np.float32)[:, np.newaxis]
        left = mel_low + bins * mel_delta
        center = left + mel_delta
        right = center + mel_delta
        frequencies = (
            np.arange(num_fft_bins, dtype=np.float32)
            * np.float32(self.sample_rate / self.fft_size)
        )
        mel = np.float32(1127.0) * np.log(
            np.float32(1.0) + frequencies / np.float32(700.0)
        )
        up_slope = (mel - left) / (center - left)
        down_slope = (right - mel) / (right - center)
        banks = np.maximum(np.float32(0.0), np.minimum(up_slope, down_slope))
        return np.pad(banks.astype(np.float32), ((0, 0), (0, 1)))

    def _create_dct(self):
        np = self._np
        mel_indices = np.arange(self.num_mel_bins, dtype=np.float32)[:, np.newaxis]
        cep_indices = np.arange(self.num_ceps, dtype=np.float32)[np.newaxis, :]
        dct = np.cos(
            np.float32(np.pi / self.num_mel_bins)
            * (mel_indices + np.float32(0.5))
            * cep_indices
        ).astype(np.float32)
        dct *= np.float32((2.0 / self.num_mel_bins) ** 0.5)
        dct[:, 0] = np.float32((1.0 / self.num_mel_bins) ** 0.5)
        return dct
