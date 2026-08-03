# MFCC-MHA two-label deployment models

These artifacts were imported from `../wekws` commit
`ec050dd80a7f0166a02e2d7b7976109fdd06cd83` on 2026-07-21.

## Deployment candidate

- `leela_mha_198f_dynamic_int8.onnx`
- SHA256: `2c0ec214757b1bdb5dd8de1e8b38d855e7b2c6ceff33a13f2946dc5e87ef35c1`
- input: float32 MFCC `[1, 198, 80]`
- output: sigmoid frame scores `[1, 198, 2]`
- output order: `hey_leela`, `hello_leela`
- ONNX opset: 17; tested with ONNX Runtime 1.27, CPU, one thread

The model uses dynamic weight quantization; “dynamic” in its filename refers
to INT8 quantization, not batch or time dimensions. Both dimensions are fixed.

The historical dev calibration at `FAR <= 0.2/h` produced these per-keyword
thresholds:

| Keyword | Threshold | Recall | FAR/h |
|---|---:|---:|---:|
| `hey_leela` | 0.9958701 | 81.575% | 0.1939 |
| `hello_leela` | 0.9970402 | 86.281% | 0.1970 |

These are deployment starting points, not universal constants. The source dev
and test manifests were identical, so recalibrate on an independent target
dataset before release. Full provenance is in `fixed_int8_far.json`.

## Regression reference

- `leela_mha_198f_fp32.onnx`
- SHA256: `dea5b36390671371e63948ae1d07cc15de0a4adbf107ff13c106a87243c4b710`
- source checkpoint SHA256:
  `bc8b119d293936202dd0e746634d5098bf39b7c12c832ff9b3d7d5e056e104af`

Use the FP32 model only for numerical comparison and diagnosis. The INT8 model
is the recommended deployment artifact. The adjacent validation and regression
JSON files preserve the original measurements.
