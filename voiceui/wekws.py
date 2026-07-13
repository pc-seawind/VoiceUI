from __future__ import annotations

from pathlib import Path


class MHAKWSModel:
    """Lazy wrapper around the packaged Leela MHA model implementation."""

    def __new__(cls, *args, **kwargs):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as functional
        except ImportError as exc:
            raise RuntimeError(
                "WeKWS MHA wake detection requires torch and torchaudio. "
                'Install with: pip install -e ".[wake]"'
            ) from exc

        class NMA(nn.Module):
            def __init__(self, input_dim: int, num_units: int = 15) -> None:
                super().__init__()
                self.weight = nn.Linear(input_dim, num_units)

            def forward(self, value):
                attention = self.weight(value).transpose(1, 2)
                return torch.matmul(attention, value)

        class SoftTripleClassifier(nn.Module):
            def __init__(
                self,
                input_dim: int,
                output_dim: int,
                num_centers: int = 2,
                scale: float = 20.0,
                gamma: float = 1.0,
            ) -> None:
                super().__init__()
                self.scale = scale
                self.gamma = gamma
                self.centers = nn.Parameter(torch.empty(output_dim, num_centers, input_dim))
                nn.init.normal_(self.centers)

            def forward(self, value):
                value = functional.normalize(value, dim=-1)
                centers = functional.normalize(self.centers, dim=-1)
                logits = torch.einsum("bd,kcd->bkc", value, centers)
                weights = torch.softmax(logits / self.gamma, dim=-1)
                return (logits * weights).sum(dim=-1) * self.scale

        class Model(nn.Module):
            def __init__(
                self,
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                num_layers: int = 4,
                num_heads: int = 20,
                nma_units: int = 15,
                dropout: float = 0.0,
                output_type: str = "frame",
                classifier: str = "linear",
                num_centers: int = 2,
            ) -> None:
                super().__init__()
                if hidden_dim % num_heads != 0:
                    raise ValueError("hidden_dim must be divisible by num_heads")
                if output_type not in {"frame", "utterance"}:
                    raise ValueError("output_type must be frame or utterance")
                self.output_type = output_type
                self.bn0 = nn.BatchNorm1d(input_dim)
                self.gru = nn.GRU(
                    input_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.mha = nn.MultiheadAttention(
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.nma = NMA(hidden_dim, nma_units)
                self.frame_classifier = nn.Linear(hidden_dim, output_dim)
                self.utterance_classifier = (
                    SoftTripleClassifier(hidden_dim, output_dim, num_centers)
                    if classifier == "soft_triple"
                    else nn.Linear(hidden_dim, output_dim)
                )
                if classifier not in {"linear", "soft_triple"}:
                    raise ValueError("classifier must be linear or soft_triple")
                self.sigmoid = nn.Sigmoid()

            def forward(self, value):
                value = self.bn0(value.transpose(1, 2)).transpose(1, 2)
                value, _ = self.gru(value)
                value, _ = self.mha(value, value, value, need_weights=False)
                if self.output_type == "utterance":
                    value = self.nma(value).sum(dim=1)
                    return self.utterance_classifier(value)
                return self.sigmoid(self.frame_classifier(value))

        return Model(*args, **kwargs)


class LeelaMhaRuntime:
    """Run the packaged MFCC MHA wake model over PCM16 audio."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        try:
            import torch
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "WeKWS MHA wake detection requires torch, torchaudio, and PyYAML. "
                'Install with: pip install -e ".[wake]"'
            ) from exc

        self.checkpoint = Path(checkpoint).expanduser()
        self.config_path = self.checkpoint.with_name("config.yaml")
        self.words_path = self.checkpoint.with_name("words.txt")
        for path in (self.checkpoint, self.config_path, self.words_path):
            if not path.is_file():
                raise FileNotFoundError(f"WeKWS MHA runtime file does not exist: {path}")

        with self.config_path.open("r", encoding="utf-8") as stream:
            configs = yaml.safe_load(stream)
        if not isinstance(configs, dict):
            raise TypeError(f"WeKWS MHA config must be a mapping: {self.config_path}")

        self.feature_conf = configs["dataset_conf"]["feature_extraction_conf"]
        if self.feature_conf.get("feature_type") != "mfcc":
            raise ValueError("Leela MHA runtime only supports the packaged MFCC feature config")
        self.sample_rate = int(
            configs["dataset_conf"].get("resample_conf", {}).get("resample_rate", 16000)
        )
        model_conf = configs["model"]
        backbone_conf = model_conf.get("backbone", {})
        if backbone_conf.get("type") != "mha":
            raise ValueError("Leela MHA runtime requires an MHA model config")

        self.labels = self._load_labels(int(model_conf["output_dim"]))
        self.device = torch.device(device)
        self._torch = torch
        self.model = MHAKWSModel(
            input_dim=int(model_conf["input_dim"]),
            hidden_dim=int(model_conf["hidden_dim"]),
            output_dim=int(model_conf["output_dim"]),
            num_layers=int(backbone_conf.get("num_layers", 4)),
            num_heads=int(backbone_conf.get("num_heads", 20)),
            nma_units=int(backbone_conf.get("nma_units", 15)),
            dropout=float(backbone_conf.get("dropout", 0.0)),
            output_type=str(backbone_conf.get("output_type", "frame")),
            classifier=str(backbone_conf.get("classifier", "linear")),
            num_centers=int(backbone_conf.get("num_centers", 2)),
        )
        try:
            state_dict = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(self.checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _load_labels(self, output_dim: int) -> list[str]:
        labels = [str(index) for index in range(output_dim)]
        for line in self.words_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            label, raw_index = parts
            try:
                index = int(raw_index)
            except ValueError:
                continue
            if 0 <= index < output_dim:
                labels[index] = label
        return labels

    def score_pcm(self, pcm: bytes, sample_rate: int) -> dict[str, float]:
        if sample_rate != self.sample_rate:
            raise RuntimeError(
                f"Leela MHA requires {self.sample_rate} Hz audio, got sample_rate={sample_rate}"
            )
        if not pcm:
            return {}
        try:
            import numpy as np
            import torchaudio.compliance.kaldi as kaldi
        except ImportError as exc:
            raise RuntimeError(
                "WeKWS MHA wake detection requires numpy and torchaudio. "
                'Install with: pip install -e ".[wake]"'
            ) from exc

        samples = np.frombuffer(pcm, dtype=np.int16).copy()
        waveform = self._torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
        features = kaldi.mfcc(
            waveform,
            num_ceps=int(self.feature_conf.get("num_ceps", self.feature_conf["num_mel_bins"])),
            num_mel_bins=int(self.feature_conf.get("num_mel_bins", 80)),
            frame_length=float(self.feature_conf.get("frame_length", 25)),
            frame_shift=float(self.feature_conf.get("frame_shift", 10)),
            dither=float(self.feature_conf.get("dither", 0.0)),
            energy_floor=0.0,
            sample_frequency=float(self.sample_rate),
        )
        if features.numel() == 0:
            return {}
        with self._torch.no_grad():
            scores = self.model(features.unsqueeze(0).to(self.device)).squeeze(0)
        peak_scores = scores.amax(dim=0).detach().cpu().tolist()
        return {label: float(score) for label, score in zip(self.labels, peak_scores, strict=True)}
