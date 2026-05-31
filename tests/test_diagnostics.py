from __future__ import annotations

import unittest

from voiceui.diagnostics import summarize_rms


class DiagnosticsTests(unittest.TestCase):
    def test_summarize_rms_recommends_threshold_from_noise_floor(self) -> None:
        summary = summarize_rms([100, 110, 120, 130, 140], duration_seconds=1.0)

        self.assertEqual(summary.chunks, 5)
        self.assertEqual(summary.duration_seconds, 1.0)
        self.assertGreaterEqual(summary.recommended_vad_threshold, 300)
        self.assertEqual(summary.max, 140)

    def test_summarize_empty_rms_has_safe_default(self) -> None:
        summary = summarize_rms([], duration_seconds=0.0)

        self.assertEqual(summary.chunks, 0)
        self.assertEqual(summary.recommended_vad_threshold, 450)


if __name__ == "__main__":
    unittest.main()
