
## Execution policy for experiments

- Codex must not run long experiments directly.
- Codex may run lightweight checks only if they are expected to finish within 3 minutes.
- Any command that runs Python scripts, model training, dataset pipelines, D1-D6 experiments, or validation jobs must be executed through:

  python tools/protection/codex_timeout.py <command...>

- If the timeout wrapper exits with code 124, Codex must stop immediately.
- After timeout, Codex must not simplify, retry, split, resume, or continue the experiment.
- After timeout, Codex must output the exact command for the user to run manually in Terminal.
- Codex may still run static checks, syntax checks, grep/rg searches, and small import checks without the wrapper.
- Codex must never run these directly without the 180-second wrapper:
  - python scripts/run_all_d1_d6.py
  - python scripts/run_d4_experiment.py
  - python scripts/run_d5_experiment.py
  - python scripts/run_d6_experiment.py
  - any command that trains models across multiple datasets
  - any command that writes large outputs under outputs/runs/
