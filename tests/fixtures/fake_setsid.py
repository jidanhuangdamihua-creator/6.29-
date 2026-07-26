#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    if "--help" in sys.argv[1:]:
        print("usage: fake_setsid [--wait] command")
        return 0
    args = [value for value in sys.argv[1:] if value != "--wait"]
    if not args:
        return 2
    transient = None
    if os.environ.get("FAKE_SETSID_PREEXEC_SHIM") == "1":
        transient = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os, time; "
                    "time.sleep(float(os.environ.get('FAKE_SETSID_SHIM_DELAY', '0.2')))"
                ),
            ]
        )
    try:
        completed = subprocess.run(args, check=False, start_new_session=True)
    finally:
        if transient is not None:
            transient.terminate()
            transient.wait()
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
