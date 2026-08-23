# tabicl-classifier-dataset-validator

DIMER dataset validator for the TabICLv2 classifier pipeline. It checks that an uploaded
tabular-classification dataset zip meets the CSV contract before fine-tuning runs.

- Runs as a CPU Kubernetes Job.
- DIMER builds the root `Dockerfile` into an image and runs `validate.py` (a thin entrypoint that
  delegates to the tested `validator.py`).
- Pairs with `tabicl-classifier-finetuner`.

The complete pipeline documentation, dataset specification, and the fine-tuner are in the
[tabicl-classifier-pipeline](https://github.com/kurtvalcorza/tabicl-classifier-pipeline) project.

## Contract summary

The dataset zip must contain a `train.csv` with a categorical `target` column whose distinct
values are the class labels — at least 2, with a validator guard at 1,000 classes (an operational
ceiling to catch high-cardinality mistakes, not a TabICL limit). Every other non-dropped column is
a feature; features may be numeric or categorical. Optional `val.csv` / `test.csv` must share the
schema, and each `test`/`val` label must appear in `train`. When `val.csv` is absent the fine-tuner
draws a **stratified** holdout, so the validator also checks split feasibility — every class must
fit in both the stratified cap and the smaller holdout side.

The validator reports pass/fail per check in `result.json` and rejects duplicate split candidates,
nested zips, path-traversal members, and oversized / zip-bomb archives (≤1 GiB uncompressed,
≤512 MiB per CSV, compression-ratio guard). All limits are overridable by platform environment
variables. See the project's dataset specification for the full rules.
