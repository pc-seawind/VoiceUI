from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from voiceui.logs import log_switch_rows, reset_logging
from voiceui.wake_proximity import (
    DEFAULT_DEVICE_A_OUTPUT,
    DEFAULT_DEVICE_B_OUTPUT,
    DEFAULT_PROD_LIVE_CONFIG,
    DEFAULT_PROXIMITY_CHANNEL,
    DEFAULT_RAW_PROXIMITY_CHANNEL,
    DEFAULT_WAKE_CHANNEL,
    DeviceMetrics,
    ScoreWeights,
    TrialResult,
    _parse_channel_candidates,
    _print_trial_result,
    _production_config_for_selected_device,
    append_trial_jsonl,
    apply_global_wake_window,
    apply_scores,
    parse_positions,
    proximity_segment_features,
    read_trial_jsonl,
    select_winner,
    summarize_trial_results,
    wake_window_bounds,
    write_trials_csv,
)


class WakeProximityTests(unittest.TestCase):
    def test_parse_positions_supports_unscored_center(self) -> None:
        positions = parse_positions("near_xvf1:xvf1, near_xvf2:xvf2, center:")

        self.assertEqual(
            [(position.name, position.expected_device) for position in positions],
            [("near_xvf1", "xvf1"), ("near_xvf2", "xvf2"), ("center", "")],
        )

    def test_parse_channel_candidates_defaults_and_auto(self) -> None:
        self.assertEqual(_parse_channel_candidates("0", 2), [0])
        self.assertEqual(_parse_channel_candidates("auto", 2), [0, 1])
        self.assertEqual(_parse_channel_candidates("0,1", 2), [0, 1])

    def test_collect_defaults_to_channel_zero(self) -> None:
        self.assertEqual(DEFAULT_PROXIMITY_CHANNEL, "0")
        self.assertEqual(DEFAULT_WAKE_CHANNEL, "0")
        self.assertEqual(DEFAULT_RAW_PROXIMITY_CHANNEL, "1")

    def test_e2e_defaults_use_named_wasapi_outputs(self) -> None:
        self.assertIn("Windows WASAPI (0 in, 2 out)", DEFAULT_DEVICE_A_OUTPUT)
        self.assertIn("Windows WASAPI (0 in, 2 out)", DEFAULT_DEVICE_B_OUTPUT)
        self.assertIn("2- reSpeaker XVF3800", DEFAULT_DEVICE_B_OUTPUT)
        self.assertEqual(DEFAULT_PROD_LIVE_CONFIG, "config.demo.wake.aliyun.yaml")

    def test_prod_live_config_uses_selected_devices_and_keeps_follow_up(self) -> None:
        args = SimpleNamespace(
            config=DEFAULT_PROD_LIVE_CONFIG,
            no_ack=False,
            ack_wav="default",
            sample_rate=16000,
            channels=2,
            block_ms=80,
            system_input_dump=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = _production_config_for_selected_device(
                args,
                run_dir=Path(temp_dir),
                label="xvf1",
                input_device="input device",
                output_device="output device",
            )

        self.assertEqual(config.input.mode, "audio")
        self.assertEqual(config.audio.device, "input device")
        self.assertEqual(config.audio.command_stream_channel, 0)
        self.assertEqual(config.wake.engine, "disabled")
        self.assertEqual(config.stt.provider, "aliyun_nls")
        self.assertEqual(config.tts.provider, "aliyun_nls")
        self.assertEqual(config.llm.provider, "bailian")
        self.assertTrue(config.llm.stream)
        self.assertTrue(config.tts.stream)
        self.assertEqual(config.conversation.follow_up_seconds, 10)
        self.assertTrue(config.conversation.barge_in_enabled)
        self.assertFalse(config.debug.system_input_dump_enabled)
        self.assertEqual(config.wake_ack.playback_device, "output device")
        self.assertEqual(config.tts.playback_device, "output device")

    def test_default_score_weights_use_segmented_raw_audio_features(self) -> None:
        weights = ScoreWeights()

        self.assertEqual(weights.confidence, 0.15)
        self.assertEqual(weights.rms, 0.25)
        self.assertEqual(weights.snr, 0.60)
        self.assertEqual(weights.late_penalty, 0.0)

    def test_wake_window_bounds_clamps_to_audio(self) -> None:
        self.assertEqual(
            wake_window_bounds(1200, pre_ms=1300, post_ms=300, audio_ms=5000),
            (0, 1500),
        )
        self.assertEqual(
            wake_window_bounds(4900, pre_ms=1300, post_ms=300, audio_ms=5000),
            (3600, 5000),
        )

    def test_proximity_segment_features_use_only_requested_window(self) -> None:
        quiet = [10] * 16000
        speech = [2000] * 16000
        pcm = b"".join(
            sample.to_bytes(2, "little", signed=True)
            for sample in quiet + speech + quiet
        )

        features = proximity_segment_features(
            pcm,
            sample_rate=16000,
            start_ms=1000,
            end_ms=2000,
            noise_end_ms=1000,
        )

        self.assertEqual(features["duration_ms"], 1000)
        self.assertGreater(features["mean_rms"], 1000)
        self.assertGreater(features["snr_db"], 20)

    def test_proximity_segment_features_zero_snr_without_noise_window(self) -> None:
        speech = [2000] * 16000
        features = proximity_segment_features(
            _pcm(speech),
            sample_rate=16000,
            start_ms=0,
            end_ms=500,
            noise_end_ms=1000,
        )

        self.assertGreater(features["mean_rms"], 1000)
        self.assertEqual(features["snr_db"], 0)
        self.assertEqual(features["band_snr_db"], 0)

    def test_apply_scores_selects_louder_confident_device(self) -> None:
        devices = {
            "xvf1": _metrics("xvf1", confidence=0.7, peak_rms=900, snr_db=14),
            "xvf2": _metrics("xvf2", confidence=0.4, peak_rms=250, snr_db=6),
        }

        apply_scores(
            devices,
            ScoreWeights(confidence=0.35, rms=0.45, snr=0.20, late_penalty=0.10),
            threshold=0.5,
            listen_seconds=5,
        )
        winner, margin = select_winner(devices)

        self.assertEqual(winner, "xvf1")
        self.assertGreater(margin, 0)

    def test_non_triggered_device_can_win_with_stronger_raw_proximity(self) -> None:
        devices = {
            "xvf1": _metrics("xvf1", confidence=0.2, peak_rms=1800, snr_db=16),
            "xvf2": _metrics("xvf2", confidence=0.8, peak_rms=600, snr_db=12),
        }

        apply_scores(devices, ScoreWeights(), threshold=0.5, listen_seconds=5)
        winner, margin = select_winner(
            devices,
            require_trigger=False,
            non_triggered_override_rms_ratio=1.5,
            non_triggered_override_min_snr_margin_db=-3.0,
        )

        self.assertEqual(winner, "xvf1")
        self.assertGreater(margin, 0)

    def test_non_triggered_device_needs_raw_rms_margin(self) -> None:
        devices = {
            "xvf1": _metrics("xvf1", confidence=0.2, peak_rms=200, snr_db=70),
            "xvf2": _metrics("xvf2", confidence=0.8, peak_rms=1000, snr_db=12),
        }

        apply_scores(devices, ScoreWeights(), threshold=0.5, listen_seconds=5)
        winner, margin = select_winner(
            devices,
            require_trigger=False,
            non_triggered_override_rms_ratio=1.5,
            non_triggered_override_min_snr_margin_db=-3.0,
        )

        self.assertEqual(winner, "xvf2")
        self.assertGreater(margin, 0)

    def test_no_wake_when_no_device_triggers(self) -> None:
        devices = {
            "xvf1": _metrics("xvf1", confidence=0.2, peak_rms=1800, snr_db=16),
            "xvf2": _metrics("xvf2", confidence=0.3, peak_rms=600, snr_db=12),
        }

        apply_scores(devices, ScoreWeights(), threshold=0.5, listen_seconds=5)
        winner, margin = select_winner(devices, require_trigger=False)

        self.assertEqual(winner, "no_wake")
        self.assertEqual(margin, 0)

    def test_global_wake_window_recomputes_all_raw_channel_features(self) -> None:
        devices = {
            "xvf1": _metrics("xvf1", confidence=0.8, peak_rms=100, snr_db=6),
            "xvf2": _metrics("xvf2", confidence=0.2, peak_rms=100, snr_db=6),
        }
        xvf1_pcm = _pcm([20] * 16000 + [500] * 16000 + [20] * 16000)
        xvf2_pcm = _pcm([20] * 16000 + [2000] * 16000 + [20] * 16000)

        trigger_source, start_ms, end_ms, windows = apply_global_wake_window(
            devices,
            {"xvf1": xvf1_pcm, "xvf2": xvf2_pcm},
            sample_rate=16000,
            baseline_seconds=1,
            wake_window_pre_ms=1000,
            wake_window_post_ms=300,
        )

        self.assertEqual(trigger_source, "xvf1")
        self.assertEqual((start_ms, end_ms), (200, 1500))
        self.assertEqual(devices["xvf1"].wake_window_start_ms, 200)
        self.assertEqual(devices["xvf2"].wake_window_start_ms, 200)
        self.assertGreater(devices["xvf2"].mean_rms, devices["xvf1"].mean_rms)
        self.assertIn("xvf1", windows)
        self.assertIn("xvf2", windows)

    def test_summarize_trial_results_groups_accuracy(self) -> None:
        results = [
            _trial(1, position="near_xvf1", expected="xvf1", selected="xvf1", correct=True),
            _trial(2, position="near_xvf1", expected="xvf1", selected="xvf2", correct=False),
            _trial(3, position="center", expected="", selected="xvf1", correct=None),
        ]

        summary = summarize_trial_results(results)

        self.assertEqual(summary["overall"].trials, 3)
        self.assertEqual(summary["overall"].scored_trials, 2)
        self.assertEqual(summary["overall"].correct, 1)
        self.assertAlmostEqual(summary["overall"].accuracy, 0.5)
        self.assertEqual(summary["by_position"]["center"].scored_trials, 0)
        self.assertEqual(summary["confusion"], {"xvf1": {"xvf1": 1, "xvf2": 1}})

    def test_jsonl_and_csv_round_trip(self) -> None:
        result = _trial(
            1,
            position="near_xvf1",
            expected="xvf1",
            selected="xvf1",
            correct=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_path = temp_path / "trials.jsonl"
            csv_path = temp_path / "trials.csv"

            append_trial_jsonl(jsonl_path, result)
            loaded = read_trial_jsonl(jsonl_path)
            write_trials_csv(csv_path, loaded, ["xvf1", "xvf2"])

            csv_text = csv_path.read_text(encoding="utf-8-sig")

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].selected_device, "xvf1")
        self.assertEqual(loaded[0].ack_output_device, "xvf1 output")
        self.assertEqual(loaded[0].ack_latency_ms, 42)
        self.assertEqual(loaded[0].assistant_transcript, "hello")
        self.assertEqual(loaded[0].assistant_reply, "world")
        self.assertIn("xvf1_best_confidence", csv_text)
        self.assertIn("trigger_source_device", csv_text)
        self.assertIn("ack_output_device", csv_text)
        self.assertIn("assistant_transcript", csv_text)
        self.assertIn("near_xvf1", csv_text)

    def test_trial_result_uses_structured_log_output(self) -> None:
        reset_logging()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _print_trial_result(
                _trial(
                    1,
                    position="near_xvf1",
                    expected="xvf1",
                    selected="xvf1",
                    correct=True,
                )
            )
        reset_logging()

        text = output.getvalue()
        self.assertNotIn("结果:", text)
        self.assertIn("module=wake_proximity | event=trial_completed", text)
        self.assertIn("module=wake_proximity | event=device_score", text)

    def test_wake_proximity_logs_are_listed_as_switches(self) -> None:
        ids = {row["id"] for row in log_switch_rows()}

        self.assertIn("wake_proximity.run_started", ids)
        self.assertIn("wake_proximity.trial_completed", ids)
        self.assertIn("wake_proximity.device_score", ids)


def _metrics(
    label: str,
    *,
    confidence: float = 0.5,
    peak_rms: float = 500.0,
    snr_db: float = 10.0,
) -> DeviceMetrics:
    return DeviceMetrics(
        label=label,
        device=label,
        resolved_device=1,
        channel=1,
        proximity_channel=1,
        audio_ms=5000,
        chunks=63,
        overflow_count=0,
        best_label="alexa",
        best_confidence=confidence,
        first_trigger_ms=1200 if confidence >= 0.5 else None,
        trigger_count=1 if confidence >= 0.5 else 0,
        noise_rms=50,
        mean_rms=peak_rms / 2,
        peak_rms=peak_rms,
        snr_db=snr_db,
        predict_avg_ms=3,
        predict_max_ms=6,
        band_rms=peak_rms,
        band_snr_db=snr_db,
    )


def _trial(
    trial_id: int,
    *,
    position: str,
    expected: str,
    selected: str,
    correct: bool | None,
) -> TrialResult:
    devices = {
        "xvf1": _metrics("xvf1", confidence=0.7, peak_rms=900, snr_db=14),
        "xvf2": _metrics("xvf2", confidence=0.4, peak_rms=250, snr_db=6),
    }
    apply_scores(devices, ScoreWeights(), threshold=0.5, listen_seconds=5)
    return TrialResult(
        trial_id=trial_id,
        position=position,
        expected_device=expected,
        selected_device=selected,
        correct=correct,
        margin=0.2,
        started_at="2026-06-04T12:00:00.000",
        listen_seconds=5,
        baseline_seconds=1,
        threshold=0.5,
        model="alexa",
        devices=devices,
        trigger_source_device="xvf1",
        global_wake_window_start_ms=200,
        global_wake_window_end_ms=1500,
        ack_output_device="xvf1 output",
        ack_latency_ms=42,
        assistant_transcript="hello",
        assistant_reply="world",
    )


def _pcm(samples: list[int]) -> bytes:
    return b"".join(
        sample.to_bytes(2, "little", signed=True)
        for sample in samples
    )


if __name__ == "__main__":
    unittest.main()
