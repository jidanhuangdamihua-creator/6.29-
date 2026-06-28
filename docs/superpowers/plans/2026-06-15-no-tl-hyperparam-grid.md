# No-TL Hyperparameter Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Dataset1 and Dataset3 No-TL hyperparameter grid over learning_rate, epochs, and clipnorm while preserving the existing split and CNN backbone.

**Architecture:** Add a dedicated audit script that reuses existing preprocessing helpers from `scripts/audits/cnn_lr_clipnorm_ablation.py` and the unchanged `build_base_cnn` model. The script writes detail rows, per-dataset best summaries, logs, and a markdown summary under `outputs/no_tl_hyperparam_grid/`.

**Tech Stack:** Python, TensorFlow/Keras, pandas, unittest/pytest.

---

### Task 1: Contract Tests

**Files:**
- Create: `tests/test_no_tl_hyperparam_grid.py`
- Create: `scripts/audits/no_tl_hyperparam_grid.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `scripts.audits.no_tl_hyperparam_grid` and assert:
- datasets are `["Dataset1", "Dataset3"]`
- learning rates are `[1e-3, 1e-4, 1e-5, 3e-4]`
- epochs are `[2, 5, 10, 20]`
- clipnorms are `[None, 1.0, 0.5]`
- expected row count is 96 for one seed
- detail CSV path is `outputs/no_tl_hyperparam_grid/no_tl_lr_epoch_clipnorm_results.csv`
- detail columns include train/validation loss, test MAE/RMSE, normalized RMSE, original-scale RMSE, MAPE, training time, and anomaly flags.

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_no_tl_hyperparam_grid.py -q`
Expected: FAIL because `scripts.audits.no_tl_hyperparam_grid` does not exist yet.

### Task 2: Grid Script

**Files:**
- Create: `scripts/audits/no_tl_hyperparam_grid.py`

- [ ] **Step 1: Implement grid script**

Implement:
- argparse options for datasets, learning rates, epochs, clipnorms, seeds, output dir, and optional limit.
- unchanged data preparation via `_load_config`, `_metric_protocol`, and `_prepare_sequences`.
- unchanged CNN backbone via `build_base_cnn`, with only Adam optimizer learning_rate and clipnorm varied.
- row-level training, prediction, metrics, loss anomaly checks, overfitting checks, and per-run log files.
- CSV outputs and markdown summary requested by the user.

- [ ] **Step 2: Verify contract tests pass**

Run: `pytest tests/test_no_tl_hyperparam_grid.py -q`
Expected: PASS.

### Task 3: Execute Experiments

**Files:**
- Create output files under `outputs/no_tl_hyperparam_grid/`

- [ ] **Step 1: Run full grid**

Run: `python scripts/audits/no_tl_hyperparam_grid.py`
Expected: 96 attempted rows for 2 datasets x 4 learning rates x 4 epoch counts x 3 clipnorm values x 1 seed.

- [ ] **Step 2: Verify result files**

Run a small Python validation checking:
- detail CSV exists and has 96 rows.
- per-dataset best CSV files exist.
- markdown summary exists.
- log count is at least 96.
- no missing required detail columns.
