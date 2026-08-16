# D1-D6 h1 seed42 formal-run evidence

This directory contains the small, Git-tracked audit evidence for the formal run below. The full numerical outputs are intentionally kept outside Git in a compressed archive.

## Run identity

- Run root: `20260816_183000_d1d6_630ae4ef_h1s42_d5first_p8`
- Producing code commit: `630ae4ef33a271187ed7225437d0609d8a13e941`
- Producing worktree: clean
- Upstream run identity: `4e88557cb38e6daf9efde93d993390189e828c5a772454f031661eeb2fa7a4ce`
- Scope: D1-D6, `without` and `with`, horizon 1, seed 42
- Scheduler concurrency: `MAX_JOBS=8`
- Exit code: `0`

## Acceptance result

- Dataset-mode groups: 12
- Mode files: 12
- Aggregate rows: 216
- Global aggregate acceptance: passed
- Aggregate CSV SHA-256: `4c0ea68e72354cc7363c1b5bb82c2816573856faad16e97981e279b625098175`

The aggregate CSV itself is not tracked here. Its manifest records the hashes of all 12 accepted mode result files.

## Full archive

- Local ignored path: `outputs/archives/20260816_183000_d1d6_630ae4ef_h1s42_d5first_p8.tar.gz`
- Compressed bytes: `15787949`
- SHA-256: `c9db1c1e9ba9eeb7c4baed513b86b3f5ad097ab9932599a579f40c841b380d7e`
- Original transferred run bytes: `119152079`
- Original run files: 162

The archive contains the complete downloaded run root plus its launch log and exit-code file. It is excluded from Git by the repository's existing `outputs/` ignore rule.

## Tracked evidence contents

- `run/run_plan.json`: exact 12-cell execution plan
- `run/**/run_config.json`: available D4-D6 cell configurations
- `run/**/*.acceptance.json`: cell, mode, and aggregate acceptance records
- `run/**/*.manifest.json`: artifact identities and result hashes
- `run/supervisor/**`: worker logs, scheduler log, and PID lifecycle table
- `launcher/launch.log`: top-level launch and completion log
- `launcher/exit_code.txt`: final process exit code
- `SHA256SUMS`: hashes of every tracked evidence file except the checksum file itself

Large CSV result matrices, plots, and the compressed archive are intentionally excluded from this Git evidence package.
