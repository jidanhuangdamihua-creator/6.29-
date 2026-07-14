# Mode-Level Bounded Parallel Supervisor Design

## Status and scope

- Design: approved on 2026-07-14.
- Implementation base: `7d8221f06baa22da959eaccf71207b304d8e3c0e` on
  `codex/preseal-blocker-fixes`.
- Scope: restore bounded dataset-by-mode process supervision on top of the
  existing formal acceptance, manifest, run-plan, `RunLayout`, and resume
  contracts.
- Out of scope: models, datasets, protocol definitions, metrics, formal target
  keys, result schemas, and the semantics implemented by
  `src/utils/result_acceptance.py`.

The implementation must not restore the historical CSV-existence acceptance,
copy result files, concatenate mode CSVs with `awk`, or collect results without
manifests.

## Decision

`scripts/parallel_mode_runner.sh` becomes the only supported top-level formal
supervisor. It owns the run root and process lifecycle, but it does not own
result semantics. `scripts/run_unified_d1_d6.py` remains the Python execution
and publication authority and gains explicit internal lifecycle operations:

1. prepare a complete immutable 300-cell formal run;
2. execute and accept one 25-cell dataset-mode worker;
3. validate all 12 accepted modes and publish one global aggregate.

The shell supervisor invokes these operations in that order. The shell may use
exit status and process metadata, but it must not parse CSV content or
reimplement acceptance, manifest, hashing, or aggregation rules.

## Alternatives considered

### Separate Python control entrypoint

A new Python script could prepare and aggregate while the unified runner only
executes modes. This has a clean implementation boundary, but creates a second
formal Python control surface and makes the audit path less direct. It is not
selected.

### Shell-owned artifact validation

The shell could read JSON sidecars, calculate hashes, and assemble accepted
paths. This duplicates the formal Python authority and risks recreating the
legacy hand-built collection path. It is explicitly rejected.

## Formal lifecycle

### Prepare

Before any mode process is started, the supervisor asks the unified runner to
prepare the run root. Preparation must:

- require a clean worktree;
- lock the exact git commit and existing `CodeIdentity`;
- discover and lock the existing formal input identity;
- derive all paths with `RunLayout`;
- build the complete six-dataset, two-mode, five-horizon, five-seed plan;
- require exactly 300 cells and 300 unique expected result paths;
- lock the existing method, schema-registry, horizon, and seed identities;
- compute an immutable `run_identity` from the canonical plan payload; and
- atomically create or exactly validate `run_plan.json`.

Preparation on a new run refuses an existing run root. Resume requires an
existing plan whose canonical payload, code identity, input identity, expected
paths, and run identity exactly match the current invocation. A different git
commit cannot resume the run.

### Mode worker

The supervisor launches exactly these 12 logical workers:

```text
d1_without d1_with d2_without d2_with d3_without d3_with
d4_without d4_with d5_without d5_with d6_without d6_with
```

Every worker invokes `scripts/run_unified_d1_d6.py` with one `--only dN`, one
`--info-sharing without|with`, and `--output-dir` set to that mode's unique
directory beneath the shared run root. An explicit internal execution scope
distinguishes a supervised mode worker from a standalone full unified run.

The worker reads the global plan from the parent run root and selects exactly
its 25 planned cells. It may execute or resume only those cells. It never
publishes a global aggregate.

A mode worker returns success only after verifying:

- exactly 25 planned cell paths exist and no extra plan cell belongs to the
  mode;
- every expected cell CSV exists;
- every cell `.acceptance.json` exists, is readable, and has `passed=true`;
- every cell `.manifest.json` exists and identifies a `formal_cell`;
- every cell manifest embeds passing acceptance;
- every cell manifest `code_identity.git_commit` equals the locked commit and
  the full code identity equals the plan identity;
- every cell manifest hash equals the actual CSV SHA-256;
- re-running the existing cell acceptance against the CSV succeeds;
- the mode CSV, acceptance report, and manifest exist after publication;
- the mode acceptance report has `passed=true`;
- the mode manifest identifies a `formal_mode_matrix`, has the locked code
  identity, embeds passing acceptance, and hashes the actual mode CSV; and
- re-running the existing mode-matrix acceptance succeeds with the exact
  formal run-plan cell set.

These checks are orchestration guards around the existing acceptance
functions. They must not change acceptance rules or result promotion semantics.

On resume, a fully accepted mode may be reused only after all checks above are
performed again. If a mode publication is missing or invalid but its cells are
valid, the worker may republish the mode through the existing
`publish_mode_matrix` path. Invalid cells are not reused.

### Global aggregation

Only the parent supervisor may request global aggregation, and only after all
12 workers have returned success. Before publication, the unified runner must
revalidate the 12 mode CSVs, acceptance reports, manifests, hashes, code
identity, and exact run-plan coverage. It then calls the existing
`publish_global_aggregate` authority once with the full formal contract.

If any worker fails, is interrupted, cannot be started, or fails post-run
validation, global aggregation is forbidden. A four-mode probe is also never
eligible for global publication.

The final global artifact must cover exactly:

```text
6 datasets × 2 modes × 5 horizons × 5 seeds × 6 methods
```

with target multiplicity governed by the existing formal target-key contract.

## Filesystem layout

One shared formal run root contains the immutable plan and 12 isolated mode
directories:

```text
<run-root>/run_plan.json
<run-root>/d1_without/
<run-root>/d1_with/
...
<run-root>/d6_with/
<run-root>/results/d1_d6_results.csv
```

Cell and mode paths remain those produced by `RunLayout`. Logs and process
metadata live in a supervisor log directory associated with the same run
identity. The supervisor never copies or concatenates result CSVs.

## Scheduler and process supervision

The default global cap is `MAX_JOBS=6`. `MAX_JOBS` may be overridden only with
an integer from 1 through 12; all other values fail before preparation or
launch. `D5_MAX_JOBS=1` is a fixed safety cap, so `d5_without` and `d5_with`
must never overlap even when global capacity is available.

The bounded scheduler scans queued work so a blocked second D5 task does not
prevent non-D5 tasks from filling available global slots. A successful test
run must demonstrate at least two non-D5 modes overlapping in time.

Every mode has an independent output directory and log. Linux production
launches use `setsid --wait`; the supervisor records the launcher PID, resolved
experiment PID, PGID, timestamps, state transitions, exit code, log path, and
output path in `pids.tsv`. It propagates the worker's actual exit code.

If a launch fails before a safe PID/PGID is established, the supervisor cleans
up any process it created and stops scheduling. On the first worker failure it
stops submitting new work, terminates active worker process groups with TERM,
allows a bounded grace interval, then uses KILL only for surviving groups. It
waits/reaps all launchers before returning nonzero. Already accepted mode
artifacts remain available for same-identity resume.

INT and TERM use the same process-group cleanup. Cleanup is idempotent and must
not signal the supervisor's own process group.

## Operator interface

The supervisor accepts configuration through validated environment variables:

- `MAX_JOBS` (default `6`, range `1..12`);
- `RUN_ROOT` (optional explicit new or resume root);
- `RESUME` (`0` or `1`);
- `DRY_RUN` (`0` or `1`);
- a probe-only mode selection containing exactly the approved four modes; and
- a probe switch that disables global publication.

The normal invocation selects all 12 modes and requires final global
publication. The probe invocation selects `d1_without`, `d1_with`,
`d2_without`, and `d2_with`, uses `MAX_JOBS=4`, and cannot request global
publication.

Dry-run performs no Python or worker launch. It prints the 12 unique mode
commands, caps, output paths, and the fixed formal summary `cells=300
unique=300`. Static dry-run output is backed by tests comparing the task list
and path shape with the Python plan builder.

## Testing strategy

Tests use fake short-lived mode workers and isolated temporary directories for
shell supervision. They must not train models or write under formal output
roots. Required coverage includes:

- 12 unique mode tasks and output directories;
- global concurrency never exceeding `MAX_JOBS`;
- D5 concurrency never exceeding one;
- at least two modes overlapping;
- worker failure preventing global aggregation;
- missing or failing acceptance, missing manifest, and hash mismatch causing
  worker failure;
- a different git commit or code identity preventing resume;
- INT cleanup reaching every active worker process group;
- dry-run launching no child and reporting 300 cells/300 unique paths; and
- accepted global coverage for all formal dimensions.

Focused tests are followed by the complete pytest suite, `compileall`,
`bash -n`, `git diff --check`, and the global dry-run. Commands subject to the
repository experiment policy run through
`python tools/protection/codex_timeout.py` and stop all work immediately if the
wrapper returns 124.

## Four-mode server probe

The implementation is not validated by launching the complete experiment.
Before a formal 12-mode run, the server operator runs only D1-without, D1-with,
D2-without, and D2-with with `MAX_JOBS=4` and global publication disabled. The
probe is accepted only when logs and process metadata show real overlap, CPU
utilization materially above a serial run, all 100 cell acceptances and four
mode acceptances passing, unique paths, and no `ProtocolViolation`,
`Traceback`, or `ResultAcceptanceError`.

Codex may execute the probe only through the repository timeout wrapper. If it
exits 124, Codex stops immediately and reports the exact manual terminal
command without retrying, splitting, simplifying, or resuming the probe.

