# Model artifacts

Running `python scripts/run_pipeline.py` writes these local artifacts:

- `supervised_candidates.joblib` — fitted logistic-regression and HistGradientBoosting pipelines.
- `final_supervised_model.joblib` — validation-selected supervised pipeline.
- `isolation_forest.joblib` — legitimate-reference Isolation Forest pipeline.
- `model_metadata.json` — selected model, columns, and validation-selected thresholds.

The joblib files are git-ignored by default to keep the public repository focused on source and reproducible results. Recreate them from the dataset and fixed-seed pipeline. **Only load joblib/pickle files from trusted sources**, because loading may execute code. Thresholds are specific to this dataset/split and illustrative costs; recalibrate them against an institution's validated operating data before use.
