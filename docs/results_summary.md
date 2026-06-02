# Q-LOT-DC / Q-LOT-DC+ — Results summary (TinyLlama-1.1B)

All numbers are measured in this repository with the `torch_reference` backend
(fake-quantized, correctness-only). **No speedup is claimed.** No tuning on the
final test set: variant selection uses a small validation split; the full
WikiText-2 test is evaluated once.

## Q-LOT-DC+ selection (small validation split, seq_len 1024, 32 chunks)

| variant | val PPL | fp_ratio |
|---|---|---|
| fp16 (reference) | 8.8232 | — |
| **qlot_dc_biascorr_adaptive_fp (winner)** | **8.8236** | 0.04 |
| qlot_dc_output_aware | 8.8245 | 0.06 |
| qlot_dc_adaptive_fp | 8.8252 | 0.20 |
| qlot_dc_biascorr | 8.8267 | 0.06 |
| qlot_dc_median | 8.8270 | 0.06 |
| sadnd | 8.8294 | 0.06 |
| int8_ptq | 8.8313 | 0.00 |
| qlot_dc_lowrank_r2 / r4 | 8.8359 | 0.06 |

On the validation split the winner cleared the margin
(8.8236 < min(int8, sadnd) − 0.001 = 8.8284). Low-rank correction *hurt* here
(8.8359) and was not selected.

## Q-LOT-DC+ full test (WikiText-2 test, seq_len 2048, 165 chunks)

| variant | PPL |
|---|---|
| fp16 | 8.0267 |
| int8_ptq | 8.0310 |
| sadnd | 8.0300 |
| **selected Q-LOT-DC+** | **8.0294** |

### Strict margin check
- `min(INT8, SADND) = 8.0300`
- `threshold = 8.0300 − 0.001 = 8.0290`
- `selected Q-LOT-DC+ = 8.0294`
- **8.0294 > 8.0290, so the strict margin is not cleared** → `clear_improvement = False`.

### Interpretation
- Q-LOT-DC+ is the **best non-FP16 variant** on the full test.
- It improves over **INT8 PTQ by 0.0016 PPL** (8.0310 → 8.0294).
- It improves over **SADND by approximately 0.0006 PPL** (8.0300 → 8.0294).
- However, it **does not clear the predefined 0.001 margin over the stronger
  baseline SADND**.
- Therefore, the result is a **modest improvement, not a decisive win**.
- **TinyLlama is a near-lossless INT8 regime** (FP16 8.0267 vs INT8 8.0310 is
  only +0.0043 PPL), so the available PPL gap is very small.
- **Larger models or more aggressive quantization are needed** to test whether
  Q-LOT-DC+ gives a decisive quality gain.

## Selected configuration (records the actual winning candidate)

`configs`/`results/qlot_dc_plus_select_tinyllama/selected_config.json`:

| field | value |
|---|---|
| method | qlot_dc_plus |
| routing_score | sadnd |
| fp_ratio | 0.04 |
| fp_budget_mode | fixed (for full test) |
| use_static_diag_comp | true |
| diag_comp_mode | median_scale |
| use_projection_bias_correction | true |
| use_lowrank_correction | false |

## Caveats

- **No speedup claim** — `torch_reference` is fake-quantized; `custom_packed`
  remains experimental.
- **No strong quality-win claim** — the full-test improvement is modest and does
  not clear the strict margin over SADND.
- Result files: `results/qlot_dc_plus_select_tinyllama/` (selection) and
  `results/qlot_dc_plus_full_tinyllama/` (full test).
