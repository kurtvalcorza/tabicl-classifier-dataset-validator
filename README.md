# tabicl-classifier-dataset-validator

DIMER dataset validator for TabICLv2 classification.

It validates `train.csv` plus optional `val.csv` / `test.csv`, rejects duplicate split candidates, counts usable non-null-target rows, checks class stratification viability, enforces the initial 2,000-feature operational ceiling, and guards against oversized/nested archives.

Runtime contract:

- input: `DIMER_DATASET_DIR`
- output: `DIMER_RESULT_PATH`
- completion callback: `DIMER_DONE_CALLBACK`
- CPU-only container

Pairs with `tabicl-classifier-finetuner`. Full dataset and deployment documentation is in `tabicl-classifier-pipeline`.
