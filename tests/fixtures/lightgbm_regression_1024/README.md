# Saved LightGBM regression oracle

This fixture uses the model trained by `scripts/measure_lightgbm.py` for the
100,000-row regression measurement described in `BENCHMARKS.md`. The training
run used LightGBM 4.7.0 and synthetic NumPy-generated data. `model.json` and
`model.txt` are byte-identical to that run. `rows.f64` and
`native_scores.f64` contain the first 1,024 rows and native raw scores,
respectively, as little-endian float64 values. The full source files match
the hashes recorded in `reports/gbm_workflow.json`.

`manifest.json` pins the hashes of these four files. The routine gate checks
the hashes before comparing baiez CLI and Python predictions with the saved
LightGBM scores. Regenerate the fixture from an official LightGBM run and
review the score differences before changing the manifest.
