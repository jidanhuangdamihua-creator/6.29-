# D1/D2 source pool regeneration and isolated experiment runs

## Purpose

Make the checked-in D1 and D2 source inputs sufficient for with-sharing
protocol preflight, and prevent a newly started D1-D6 experiment from
silently replacing files from an earlier run. This change does not run model
training, create a confirmed matrix, add result provenance columns, or alter
the result schema.

## Source-pool regeneration

`scripts/regenerate_d1_d2_parquets.py` will retain its existing long-table
support. Its D2 input path will additionally recognize the repository's wide
raw file form:

- one `DATE` column;
- demand columns named `QTY_B<brand>_<item>`;
- optional promotion columns named `PROMO_B<brand>_<item>`.

The reader will select exactly Brand 1--3 and Item 1--10, unpivot the demand
columns to the canonical long columns (`date`, `brand_id`, `item_id`,
`sales`), and attach the matching promotion value when present. It will reject
missing required demand columns, invalid dates, duplicate keys, or a source
and target set that differs from the strict protocol contract. Existing long
D2 inputs keep the current alias-based path.

The command will default D2 input to the checked-in
`数据集/原始数据/Dataset 2/hierarchical_sales_data.csv`, so a normal D1/D2
regeneration uses repository raw inputs. The output remains the explicitly
selected solidified-data directory; this is an intentional replacement of the
currently incomplete D1/D2 protocol inputs, not an experiment-result output.

## Run-directory isolation

All D1--D6 executable runners will use a shared run-directory contract:

- If `--output-dir` is omitted, create a new `outputs/runs/<timestamp>` (or
  timestamp-plus-label) directory atomically. A same-second collision retries
  with the next timestamp instead of overwriting.
- If `--output-dir` names an existing directory, fail before writing any
  result, configuration, report, or CSV.
- A caller that intentionally coordinates several subprocesses may create a
  fresh run directory once, then pass it to its child tasks. This capability is
  private to the unified orchestrator so standalone calls cannot reuse an old
  run accidentally.

`run_unified_d1_d6.py` is the complete cross-D1--D6 orchestration entrypoint.
It will allocate its shared run directory before constructing tasks, pass it
to the D1--D3 and D4--D6 runners, and preserve all task outputs under that one
new directory. `run_full_paper_experiments.py` and the D4, D5, and D6 runners
will use the new-run validation when run directly.

The obsolete `tests/test_run_all_d1_d6_aliases.py` imports a missing script
and describes a different alias interface. It will be removed rather than
reviving `scripts/run_all_d1_d6.py`; the unified runner's existing tests are
the authoritative entrypoint coverage.

## Tests and verification

Tests will be written first and demonstrate:

1. a minimal D2 wide input is converted into the exact 27 source keys and the
   Brand1/Item10 target, including a matching promotion value;
2. missing required wide demand columns fail explicitly;
3. standalone explicit existing output directories are rejected before any
   file is written;
4. default directory allocation remains unique under a simulated collision;
5. the unified runner makes one fresh shared directory for all generated
   tasks.

Focused Python tests and static compilation will use the repository's
180-second timeout wrapper. No D1--D6 training, dataset pipeline, or formal
matrix run is part of this implementation verification.
