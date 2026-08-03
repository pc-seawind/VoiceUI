from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from voiceui.wekws_onnx import LeelaMhaOnnxRuntime, _KaldiMfccFrontend

_MODEL_PATH = (
    Path(__file__).parents[1]
    / "models"
    / "wake"
    / "mfcc_mha_2label"
    / "leela_mha_198f_dynamic_int8.onnx"
)


class KaldiMfccFrontendTests(unittest.TestCase):
    def test_matches_torchaudio_kaldi_reference(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        sample_indices = np.arange(32000, dtype=np.float64)
        samples = (
            np.sin(sample_indices * 2.0 * np.pi * 440.0 / 16000.0) * 20000.0
        ).astype(np.int16)

        features = _KaldiMfccFrontend(np, sample_rate=16000)(samples)

        self.assertEqual(features.shape, (198, 80))
        np.testing.assert_allclose(
            features[0, :5],
            np.array(
                [73.945435, 91.56212, 67.20061, -46.523415, -89.29194],
                dtype=np.float32,
            ),
            atol=0.02,
            rtol=0.0,
        )


@unittest.skipUnless(
    importlib.util.find_spec("onnxruntime") is not None and _MODEL_PATH.is_file(),
    "ONNX Runtime or the packaged wake model is unavailable",
)
class LeelaMhaOnnxRuntimeTests(unittest.TestCase):
    def test_scores_two_second_pcm_without_torch(self) -> None:
        runtime = LeelaMhaOnnxRuntime(_MODEL_PATH)

        scores = runtime.score_pcm(bytes(64000), 16000)

        self.assertEqual(runtime.backend, "onnx_int8")
        self.assertEqual(set(scores), {"hey_leela", "hello_leela"})
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores.values()))

    def test_rejects_wrong_pcm_window_size(self) -> None:
        runtime = LeelaMhaOnnxRuntime(_MODEL_PATH)

        with self.assertRaisesRegex(ValueError, "32000 PCM samples"):
            runtime.score_pcm(bytes(32000), 16000)


if __name__ == "__main__":
    unittest.main()
